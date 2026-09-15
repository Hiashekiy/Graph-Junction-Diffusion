"""DiDi 成都真实数据准备入口（实施方案第 14、17-D 节）。

这是**数据转换的唯一正式入口**，三个阶段一次到位：

    阶段 0  python scripts/prepare_didi.py --config ... --scan-only
            确认 edge_features 列名 / 道路长度统计 / road→junction 转换率
            -> 据此填写 data.length_column

    阶段 1  python scripts/prepare_didi.py --config ... --scan-corridor
            在 train split 上比较 rho 候选，选出满足 train GT containment
            >= target 的**最小** rho，写回配置并冻结

    阶段 2  python scripts/prepare_didi.py --config ... --build
            生成 data/didi_chengdu_gjd/{train,val,test,test_1000,
            shuffled_od_1000}.pkl + split_manifest.csv + metadata.json + stats.json

设计约束（方案第 4.3 / 6.1 / 17-A 节）：

* 任何过滤都不许静默发生 —— 每一步的保留率都进 ``stats.json`` 的 funnel。
* corridor **只**由 ``full graph + edge length + start + goal`` 决定，
  函数签名里没有 GT path（测试会守住这一点）。
* GT 是真实车辆历史路径，**任何地方都不调用** ``nx.shortest_path`` 去生成它；
  Dijkstra 只用来算 ``C*`` 这个标尺。
* 候选轨迹会被缓存（``_didi_candidates.pkl``），``--scan-corridor`` 与
  ``--build`` 复用同一份，避免重复解析 600MB CSV。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import didi_dataset as didi  # noqa: E402
from src.data.dataset import GraphQueryDataset, validate_sample  # noqa: E402
from src.data.dataset_builder import (  # noqa: E402
    build_sample,
    build_sample_from_observed_path,
)
from src.utils.config import Config, flatten_overrides, load_config  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="prepare the DiDi Chengdu real-road dataset for GJD"
    )
    parser.add_argument(
        "--config", default="configs/graph_flow_didi_weighted.yaml"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scan-only", action="store_true", help="阶段 0：只扫描不建数据")
    mode.add_argument(
        "--scan-corridor", action="store_true", help="阶段 1：比较 rho 候选"
    )
    mode.add_argument("--build", action="store_true", help="阶段 2：生成正式数据集")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "最终数据集规模（去重之后、划分之前抽多少条候选，覆盖 "
            "data.max_dataset_samples）。它和 data.max_candidates 不是一回事："
            "后者只是解析期的内存/时间闸门，前者决定落在磁盘上的数据集大小。"
        ),
    )
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None, help="覆盖 paths.data_dir")
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="覆盖 data.corridor.rho（--build 时用扫描选出的值）",
    )
    parser.add_argument("--refresh-cache", action="store_true", help="强制重新解析 CSV")
    parser.add_argument("--no-cache", action="store_true", help="不读也不写候选缓存")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置项，可重复：--set data.corridor.rho=1.8",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 配置 / 路径
# ---------------------------------------------------------------------------
def resolve_paths(config: Config, config_path: str, out_dir: Optional[str]) -> Dict[str, Path]:
    root = Path(str(config.get("data.root")))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    data_dir = Path(out_dir or str(config.get("paths.data_dir", "data/didi_chengdu_gjd")))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return {
        "root": root,
        "dicts": root / str(config.get("data.dicts_file", "dicts.pkl")),
        "edge_features": root / str(config.get("data.edge_features_file", "edge_features.csv")),
        "line_graph": root / str(config.get("data.line_graph_file", "line_graph_edge_idx.npy")),
        "data_dir": data_dir,
        "config": Path(config_path),
    }


def load_real_graph(config: Config, paths: Dict[str, Path]):
    """读 dicts + edge_features，建全局无向有权 junction graph。"""
    idx2edge = didi.load_dicts(paths["dicts"])
    columns, lengths = didi.load_edge_lengths(
        paths["edge_features"], config.get("data.length_column", None)
    )
    graph, stats = didi.build_global_weighted_graph(idx2edge, lengths)
    return graph, stats, idx2edge, columns, lengths


def build_filter(config: Config) -> didi.TrajectoryFilter:
    return didi.TrajectoryFilter(
        min_road_segments=int(config.get("data.min_road_segments", 10)),
        max_road_segments=int(config.get("data.max_road_segments", 100)),
        require_simple_gt=bool(config.get("data.require_simple_gt", True)),
    )


# ---------------------------------------------------------------------------
# 候选轨迹收集
# ---------------------------------------------------------------------------
def collect_candidates(
    graph: nx.Graph,
    idx2edge: Dict[int, Tuple[int, int, int]],
    lengths: Dict[int, float],
    files: Sequence[Path],
    filter_cfg: didi.TrajectoryFilter,
    max_rows_per_file: Optional[int],
    max_candidates: Optional[int],
    funnel: didi.FunnelStats,
    progress_every: int = 20000,
    verbose: bool = True,
) -> List[didi.TrajectoryCandidate]:
    """流式解析所有轨迹 CSV，产出清洗后的候选（方案第 4 节）。"""
    distance = didi.ShortestDistanceCache(graph)
    candidates: List[didi.TrajectoryCandidate] = []
    start_time = time.time()
    seen_paths: Counter = Counter()

    for file_path in files:
        date = file_path.stem
        for _index, row in didi.iter_trajectory_csv(file_path, max_rows=max_rows_per_file):
            funnel.add("raw_rows")
            try:
                road_ids = didi.parse_road_path(row["path"])
            except (ValueError, SyntaxError, TypeError):
                funnel.add("parse_failed")
                continue
            if len(road_ids) < 2:
                funnel.add("parse_failed")
                continue
            funnel.add("parse_valid")

            # 删除连续重复 road id（方案第 4.2 节）
            cleaned: List[int] = []
            for road_id in road_ids:
                if cleaned and cleaned[-1] == road_id:
                    continue
                cleaned.append(road_id)

            if any(road_id not in idx2edge for road_id in cleaned):
                funnel.add("road_id_invalid")
                continue
            funnel.add("road_id_valid")

            conversion = didi.road_path_to_junction_path(cleaned, idx2edge, graph)
            if not conversion.ok:
                funnel.add(f"reject_{conversion.reason}")
                continue
            funnel.add("continuous_valid")
            if conversion.u_turns:
                # 平行路段掉头（a -> b -> a）会让 junction 重复，几乎必然被
                # require_simple_gt 丢掉；在**过滤前**记一笔，才能从报告里看出
                # 原始数据里到底有多少掉头，而不是看到一个恒为 0 的统计。
                funnel.add("rows_with_u_turn")

            junction_path = conversion.path
            assert junction_path is not None

            if filter_cfg.require_simple_gt and not didi.is_simple_path(junction_path):
                funnel.add("reject_non_simple_gt")
                continue

            num_road_segments = len(cleaned) - conversion.dropped_self_loop_roads
            if not filter_cfg.road_length_ok(num_road_segments):
                funnel.add("reject_length_invalid")
                continue
            funnel.add("length_valid")
            funnel.add("simple_valid")

            raw_cost = sum(float(lengths.get(road_id, 0.0)) for road_id in cleaned)
            gt_cost = didi.path_cost(graph, junction_path)
            dijkstra_cost = distance.distance(junction_path[0], junction_path[-1])
            if not math.isfinite(gt_cost) or not math.isfinite(dijkstra_cost):
                funnel.add("reject_unreachable")
                continue

            seen_paths[junction_path[0], junction_path[-1], tuple(junction_path)] += 1
            candidates.append(
                didi.TrajectoryCandidate(
                    order_id=str(row.get("order_id", "")),
                    date=date,
                    junction_path=junction_path,
                    raw_road_len=int(num_road_segments),
                    junction_len=len(junction_path),
                    raw_road_cost=float(raw_cost),
                    gt_cost=float(gt_cost),
                    dijkstra_cost=float(dijkstra_cost),
                    conversion=conversion.method,
                    u_turns=int(conversion.u_turns),
                )
            )
            if max_candidates is not None and len(candidates) >= int(max_candidates):
                break
            if verbose and len(candidates) % progress_every == 0:
                rate = len(candidates) / max(time.time() - start_time, 1e-6)
                print(
                    f"  collected {len(candidates)} candidates "
                    f"({rate:.0f}/s, {time.time() - start_time:.1f}s)",
                    flush=True,
                )
        if max_candidates is not None and len(candidates) >= int(max_candidates):
            if verbose:
                print(f"  reached max_candidates={max_candidates}, stop reading", flush=True)
            break

    funnel.add("candidates", len(candidates))
    funnel.add("unique_path_keys", len(seen_paths))
    return candidates


# ---------------------------------------------------------------------------
# 候选缓存
# ---------------------------------------------------------------------------
def _file_signature(path: Path) -> List[Any]:
    stat = path.stat()
    return [str(path), int(stat.st_size), int(stat.st_mtime)]


def collection_signature(
    paths: Dict[str, Path],
    files: Sequence[Path],
    config: Config,
    filter_cfg: didi.TrajectoryFilter,
    max_rows_per_file: Optional[int],
    max_candidates: Optional[int],
) -> Dict[str, Any]:
    return {
        "version": 1,
        "dicts": _file_signature(paths["dicts"]),
        "edge_features": _file_signature(paths["edge_features"]),
        "length_column": str(config.get("data.length_column")),
        "files": [_file_signature(path) for path in files],
        "filter": filter_cfg.to_dict(),
        "max_rows_per_file": max_rows_per_file,
        "max_candidates": max_candidates,
    }


def load_or_collect(
    config: Config,
    paths: Dict[str, Path],
    graph: nx.Graph,
    idx2edge: Dict[int, Tuple[int, int, int]],
    lengths: Dict[int, float],
    files: Sequence[Path],
    filter_cfg: didi.TrajectoryFilter,
    max_rows_per_file: Optional[int],
    max_candidates: Optional[int],
    refresh: bool,
    use_cache: bool,
    verbose: bool = True,
) -> Tuple[List[didi.TrajectoryCandidate], didi.FunnelStats, bool]:
    """读缓存或重新收集；返回 (candidates, funnel, from_cache)。"""
    cache_path = paths["data_dir"] / "_didi_candidates.pkl"
    signature = collection_signature(
        paths, files, config, filter_cfg, max_rows_per_file, max_candidates
    )
    if use_cache and not refresh and cache_path.exists():
        try:
            with open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
        except (pickle.UnpicklingError, EOFError, OSError):
            payload = None
        if isinstance(payload, dict) and payload.get("signature") == signature:
            funnel = didi.FunnelStats(Counter(payload.get("funnel", {})))
            if verbose:
                print(f"loaded {len(payload['candidates'])} cached candidates from {cache_path}")
            return list(payload["candidates"]), funnel, True
        if verbose:
            print("candidate cache is stale (inputs/config changed), re-collecting")

    funnel = didi.FunnelStats()
    candidates = collect_candidates(
        graph,
        idx2edge,
        lengths,
        files,
        filter_cfg,
        max_rows_per_file,
        max_candidates,
        funnel,
        verbose=verbose,
    )
    if use_cache:
        paths["data_dir"].mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(
                {
                    "signature": signature,
                    "funnel": dict(funnel.counts),
                    "candidates": candidates,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if verbose:
            print(f"wrote candidate cache -> {cache_path}")
    return candidates, funnel, False


# ---------------------------------------------------------------------------
# 阶段 0：扫描
# ---------------------------------------------------------------------------
def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p5": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def run_scan_only(
    config: Config, paths: Dict[str, Path], args: argparse.Namespace
) -> int:
    graph, stats, idx2edge, columns, lengths = load_real_graph(config, paths)
    filter_cfg = build_filter(config)
    files = didi.discover_trajectory_files(
        paths["root"], str(config.get("data.trajectory_glob", "201610*.csv"))
    )
    scan_rows = int(config.get("data.scan_rows_per_file", 3000))

    report: Dict[str, Any] = {
        "edge_features_columns": columns,
        "length_column_used": str(config.get("data.length_column")),
        "length_like_columns": didi.detect_length_columns(columns),
        "num_roads_in_dicts": len(idx2edge),
        "num_roads_with_length": len(lengths),
        "road_length_stats": _percentiles(list(lengths.values())),
        "graph": stats.to_dict(),
        "trajectory_files": [str(path.name) for path in files],
        "scan_rows_per_file": scan_rows,
    }

    funnel = didi.FunnelStats()
    candidates = collect_candidates(
        graph,
        idx2edge,
        lengths,
        files,
        filter_cfg,
        max_rows_per_file=scan_rows,
        max_candidates=None,
        funnel=funnel,
        verbose=False,
    )

    road_lengths = [candidate.raw_road_len for candidate in candidates]
    junction_lengths = [candidate.junction_len for candidate in candidates]
    ratios = [candidate.gt_cost_ratio for candidate in candidates]
    u_turns = [candidate.u_turns for candidate in candidates]
    conversions = Counter(candidate.conversion for candidate in candidates)
    dedup_keys = {candidate.dedup_key() for candidate in candidates}

    continuous = funnel.counts.get("continuous_valid", 0)
    non_simple = funnel.counts.get("reject_non_simple_gt", 0)
    report.update(
        {
            "funnel": funnel.to_dict(),
            "continuity_failure_rate": (
                1.0 - continuous / max(funnel.counts.get("road_id_valid", 0), 1)
            ),
            "loop_gt_rate": non_simple / max(continuous, 1),
            "raw_road_length_stats": _percentiles(road_lengths),
            "junction_length_stats": _percentiles(junction_lengths),
            "u_turns_stats": _percentiles(u_turns),
            "rows_with_u_turn_fraction": (
                funnel.counts.get("rows_with_u_turn", 0) / max(continuous, 1)
            ),
            "conversion_methods": dict(conversions),
            "unique_path_keys": len(dedup_keys),
            "duplicate_path_rate": 1.0 - len(dedup_keys) / max(len(candidates), 1),
            "gt_cost_ratio_stats": _percentiles(ratios),
            "gt_cost_ratio_below_1": sum(1 for value in ratios if value < 1.0 - 1e-9),
            "first_candidates": [
                {
                    "date": candidate.date,
                    "junction_len": candidate.junction_len,
                    "raw_road_len": candidate.raw_road_len,
                    "gt_cost_ratio": candidate.gt_cost_ratio,
                }
                for candidate in candidates[:3]
            ],
        }
    )

    if report["length_like_columns"] and str(config.get("data.length_column")) == "None":
        report["hint"] = (
            "data.length_column is not set; set it to one of "
            f"{report['length_like_columns']} before running --build"
        )

    print(json.dumps(report, indent=1, ensure_ascii=False))
    out = paths["data_dir"] / "scan_only.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    print(f"\nsaved -> {out}")
    print(
        "\n下一步：确认 data.length_column 正确，然后跑 "
        "`prepare_didi.py --scan-corridor` 选定 rho。"
    )
    return 0


# ---------------------------------------------------------------------------
# 阶段 1：corridor rho 扫描
# ---------------------------------------------------------------------------
def run_scan_corridor(
    config: Config, paths: Dict[str, Path], args: argparse.Namespace
) -> int:
    graph, stats, idx2edge, _columns, lengths = load_real_graph(config, paths)
    filter_cfg = build_filter(config)
    files = didi.discover_trajectory_files(
        paths["root"], str(config.get("data.trajectory_glob", "201610*.csv"))
    )
    max_rows = _max_rows_per_file(config, args)
    max_candidates = _max_candidates(config, args)

    candidates, funnel, from_cache = load_or_collect(
        config, paths, graph, idx2edge, lengths, files, filter_cfg,
        max_rows, max_candidates, args.refresh_cache, not args.no_cache,
        verbose=not args.quiet,
    )
    candidates, dedup_dropped = didi.deduplicate_candidates(candidates)
    print(f"candidates={len(candidates)} (dedup dropped {dedup_dropped})")
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    candidates, sampling_stats = _maybe_subsample(candidates, config, args, seed)
    print(f"candidates   : {len(candidates)} used for the rho scan")

    fractions = _split_fractions(config)
    splits = didi.split_real_paths(candidates, fractions, seed=seed)
    train = splits["train"]
    print(f"train/val/test = {len(train)}/{len(splits['val'])}/{len(splits['test'])}")

    rhos = [float(value) for value in config.get("data.corridor.rho_candidates", [])]
    if not rhos:
        raise SystemExit("data.corridor.rho_candidates is empty; nothing to scan")
    target = float(config.get("data.corridor.target_gt_containment", 0.98))

    distance = didi.ShortestDistanceCache(graph)
    accumulators: Dict[float, Dict[str, Any]] = {
        rho: {"n": 0, "contained": 0, "nodes": [], "decisions": []} for rho in rhos
    }
    start_time = time.time()
    for index, candidate in enumerate(train):
        start, goal = candidate.junction_path[0], candidate.junction_path[-1]
        masks = distance.corridor_mask(start, goal, rhos)
        for rho in rhos:
            keep = masks[rho]
            contained = all(node in keep for node in candidate.junction_path)
            acc = accumulators[rho]
            acc["n"] += 1
            acc["contained"] += int(contained)
            acc["nodes"].append(len(keep))
            acc["decisions"].append(
                didi.corridor_decision_count(graph, keep, start, goal)
            )
        if (index + 1) % 20000 == 0:
            print(f"  scanned {index + 1}/{len(train)} ({time.time() - start_time:.1f}s)",
                  flush=True)

    rows = []
    for rho in rhos:
        acc = accumulators[rho]
        n = max(acc["n"], 1)
        rows.append(
            {
                "rho": float(rho),
                "gt_containment": acc["contained"] / n,
                "mean_corridor_nodes": float(np.mean(acc["nodes"])) if acc["nodes"] else 0.0,
                "p50_corridor_nodes": float(np.percentile(acc["nodes"], 50)) if acc["nodes"] else 0.0,
                "p95_corridor_nodes": float(np.percentile(acc["nodes"], 95)) if acc["nodes"] else 0.0,
                "mean_decisions": float(np.mean(acc["decisions"])) if acc["decisions"] else 0.0,
                "p50_decisions": float(np.percentile(acc["decisions"], 50)) if acc["decisions"] else 0.0,
                "p95_decisions": float(np.percentile(acc["decisions"], 95)) if acc["decisions"] else 0.0,
            }
        )

    feasible = [row for row in rows if row["gt_containment"] >= target]
    chosen: Optional[Dict[str, Any]] = None
    if feasible:
        chosen = min(feasible, key=lambda row: row["rho"])
    report = {
        "num_train_candidates": len(train),
        "target_gt_containment": target,
        "rho_candidates": rhos,
        "rows": rows,
        "chosen_rho": chosen["rho"] if chosen else None,
        "chosen_row": chosen,
        "from_cache": from_cache,
        "funnel": funnel.to_dict(),
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))

    out = paths["data_dir"] / "scan_corridor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, ensure_ascii=False)
    print(f"\nsaved -> {out}")
    if chosen is None:
        print(
            f"\n没有 rho 达到 {target:.0%} 的 train GT containment；"
            "把 data.corridor.rho_candidates 往大里加（例如 3.0 / 4.0）再扫一次。"
        )
        return 2
    print(
        f"\n选定 rho = {chosen['rho']}（满足 train GT containment >= {target:.0%} 的最小值）。"
        f"\n请把它写进 configs 的 data.corridor.rho 并**冻结**，val/test 不得再按 GT 调整。"
    )
    return 0


# ---------------------------------------------------------------------------
# 阶段 2：正式生成
# ---------------------------------------------------------------------------
def _split_fractions(config: Config) -> Dict[str, float]:
    split_cfg = config.get("split", {}) or {}
    fractions = {
        "train": float(split_cfg.get("train", 0.72)),
        "val": float(split_cfg.get("val", 0.08)),
        "test": float(split_cfg.get("test", 0.20)),
    }
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"split fractions must sum to 1.0, got {total} ({fractions})")
    return fractions


def _max_rows_per_file(config: Config, args: argparse.Namespace) -> Optional[int]:
    value = args.max_rows_per_file
    if value is None:
        value = config.get("data.max_rows_per_file", None)
    return None if value in (None, 0) else int(value)


def _max_candidates(config: Config, args: argparse.Namespace) -> Optional[int]:
    value = config.get("data.max_candidates", None)
    return None if value in (None, 0) else int(value)


def _max_dataset_samples(config: Config, args: argparse.Namespace) -> Optional[int]:
    """最终数据集规模（去重后抽样的条数）。

    为什么要和 ``max_candidates`` 分开：候选必须**从所有日期文件里读**才有
    OD / 时段多样性，但真实 corridor 样本很大（中位数约 90KB），全量落盘会得到
    好几个 GB。所以先尽量多收集，再按固定 seed 均匀抽到目标规模。
    """
    value = args.max_samples
    if value is None:
        value = config.get("data.max_dataset_samples", None)
    return None if value in (None, 0) else int(value)


def _maybe_subsample(
    candidates: List[didi.TrajectoryCandidate],
    config: Config,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[List[didi.TrajectoryCandidate], Dict[str, Any]]:
    """按 **(日期 × GT 长度分位)** 分层抽到 ``data.max_dataset_samples``。

    确定性、可复现，并且保证 10 个日期都有数据、长度分布不被截断破坏
    （见 :func:`src.data.didi_dataset.stratified_subsample`）。
    """
    target = _max_dataset_samples(config, args)
    if target is None or target >= len(candidates):
        return candidates, {
            "pool_size": len(candidates),
            "requested": None,
            "selected": len(candidates),
            "note": "no subsampling requested",
        }
    subset, stats = didi.stratified_subsample(candidates, target, seed=seed)
    print(
        f"stratified subsample {len(candidates)} -> {len(subset)} "
        f"(dates {stats['num_dates_covered']}/{stats['num_dates_total']}, "
        f"strata {stats['num_nonempty_strata']}/{stats['num_strata']}, "
        f"len edges={[round(value, 1) for value in stats['length_bucket_edges']]})"
    )
    return subset, stats


def resolve_rho(config: Config, args: argparse.Namespace) -> float:
    rho = args.rho
    if rho is None:
        rho = config.get("data.corridor.rho", None)
    if rho in (None, "None", ""):
        raise SystemExit(
            "data.corridor.rho is not set. Run `prepare_didi.py --scan-corridor` "
            "first and freeze the selected value into the config (implementation "
            "plan section 6.3: rho must be chosen on the train split and then "
            "frozen for val/test)."
        )
    rho = float(rho)
    if rho < 1.0:
        raise SystemExit(f"rho must be >= 1, got {rho}")
    return rho


#: 用于"被丢弃 vs 被保留"分布对比的候选特征
ANALYSIS_FEATURES = (
    "raw_road_len",
    "junction_len",
    "dijkstra_cost",
    "gt_cost",
    "gt_cost_ratio",
)


def _candidate_features(candidate: didi.TrajectoryCandidate) -> Dict[str, float]:
    return {
        "raw_road_len": float(candidate.raw_road_len),
        "junction_len": float(candidate.junction_len),
        "dijkstra_cost": float(candidate.dijkstra_cost),
        "gt_cost": float(candidate.gt_cost),
        "gt_cost_ratio": float(candidate.gt_cost_ratio),
    }


def _record_outcome(
    analysis: Optional[Dict[str, List[Dict[str, float]]]],
    outcome: str,
    candidate: didi.TrajectoryCandidate,
) -> None:
    if analysis is None:
        return
    analysis.setdefault(outcome, []).append(_candidate_features(candidate))


def _compare_outcome_groups(
    analysis: Mapping[str, Sequence[Mapping[str, float]]]
) -> Dict[str, Any]:
    """比较各组（retained / corridor_miss / ...）的特征分布。

    为什么必须做这件事：rho=1.5 会丢掉约 9% 的 GT。如果"绕路明显、OD 距离长、
    路径复杂"的轨迹**更容易**被 corridor miss 掉，那留下训练的就变成了"更像最短路"
    的子集 —— 这是一种隐性偏差，只看 retention 百分比是发现不了的。
    """
    report: Dict[str, Any] = {}
    for outcome, rows in analysis.items():
        if not rows:
            continue
        entry: Dict[str, Any] = {"n": len(rows)}
        for key in ANALYSIS_FEATURES:
            values = [float(row[key]) for row in rows if key in row]
            if values:
                entry[key] = _percentiles(values)
        report[outcome] = entry

    retained = report.get("retained")
    if retained:
        # 直接给出"被丢弃组 / 保留组"的中位数比值，> 1 说明被丢的那批确实更长/更绕
        for outcome, entry in report.items():
            if outcome == "retained":
                continue
            comparison: Dict[str, float] = {}
            for key in ANALYSIS_FEATURES:
                base = (retained.get(key) or {}).get("p50")
                other = (entry.get(key) or {}).get("p50")
                if base and other is not None and base > 0:
                    comparison[f"{key}_p50_ratio_vs_retained"] = float(other / base)
            if comparison:
                entry["shift_vs_retained"] = comparison
    return report


#: GT/Dijkstra cost ratio 分桶（用来检查"绕路的 GT 是不是更容易被 corridor 丢掉"）
DETOUR_BUCKETS: Tuple[Tuple[str, float, float], ...] = (
    ("1.00-1.05", 1.0, 1.05),
    ("1.05-1.15", 1.05, 1.15),
    ("1.15-1.30", 1.15, 1.30),
    ("1.30-1.50", 1.30, 1.50),
    ("1.50-2.00", 1.50, 2.00),
    (">=2.00", 2.00, float("inf")),
)


def _detour_bucket(ratio: float) -> str:
    for name, low, high in DETOUR_BUCKETS:
        if low <= ratio < high:
            return name
    return DETOUR_BUCKETS[-1][0]


def _retention_by_detour_bucket(
    analysis: Mapping[str, Sequence[Mapping[str, float]]]
) -> Dict[str, Any]:
    """按 GT/Dijkstra cost ratio 分桶看**保留率**。

    这是 rho 取舍最关键的偏差检查：如果"司机绕得越远"越容易被 corridor miss 掉，
    那最终数据集就系统性地偏向"接近最短路"的驾驶行为 —— 此时模型在 test 上看起来
    很好，但它学到的是被裁剪过的行为分布，而不是成都司机的真实行为。
    """
    pool: Counter = Counter()
    kept: Counter = Counter()
    for outcome, rows in analysis.items():
        for row in rows:
            name = _detour_bucket(float(row.get("gt_cost_ratio", 1.0)))
            pool[name] += 1
            if outcome == "retained":
                kept[name] += 1
    total_pool = sum(pool.values())
    total_kept = sum(kept.values())
    if not total_pool:
        return {}
    report: Dict[str, Any] = {}
    for name, _low, _high in DETOUR_BUCKETS:
        pool_count = pool.get(name, 0)
        kept_count = kept.get(name, 0)
        report[name] = {
            "pool": pool_count,
            "kept": kept_count,
            "retention": (kept_count / pool_count) if pool_count else 0.0,
            "pool_share": pool_count / total_pool,
            "data_share": (kept_count / total_kept) if total_kept else 0.0,
        }
    report["_overall"] = {
        "pool": total_pool,
        "kept": total_kept,
        "retention": total_kept / total_pool if total_pool else 0.0,
    }
    return report


def build_split(
    candidates: Sequence[didi.TrajectoryCandidate],
    graph: nx.Graph,
    distance: didi.ShortestDistanceCache,
    rho: float,
    max_corridor_nodes: Optional[int],
    split_name: str,
    funnel: didi.FunnelStats,
    progress_every: int = 20000,
    analysis: Optional[Dict[str, List[Dict[str, float]]]] = None,
) -> List[Tuple[didi.TrajectoryCandidate, Any]]:
    """把候选轨迹变成 ``GraphSample``；返回 (candidate, sample) 对。

    ``analysis`` 非空时，把每个候选按结局（retained / corridor_miss /
    corridor_too_large / build_failed）连同它的特征记下来，供
    :func:`build_stats_report` 比较"被丢掉的样本"和"留下的样本"分布是否不同。
    """
    built: List[Tuple[didi.TrajectoryCandidate, Any]] = []
    start_time = time.time()
    for index, candidate in enumerate(candidates):
        funnel.add(f"{split_name}_input")
        start, goal = candidate.junction_path[0], candidate.junction_path[-1]
        corridor = didi.build_od_corridor(
            graph, start, goal, rho, max_nodes=max_corridor_nodes
        )
        if corridor is None:
            funnel.add(f"{split_name}_corridor_too_large")
            _record_outcome(analysis, "corridor_too_large", candidate)
            continue
        if not didi.path_contained(corridor.graph, candidate.junction_path):
            # 方案第 6.3 节：val/test 的 corridor miss 不进主指标，单独报告
            candidate.corridor_miss = True
            funnel.add(f"{split_name}_corridor_miss")
            _record_outcome(analysis, "corridor_miss", candidate)
            continue
        candidate.corridor_nodes = corridor.num_nodes
        candidate.corridor_edges = corridor.num_edges
        candidate.num_decisions = corridor.num_decisions

        meta = {
            "source": "didi_chengdu",
            "date": candidate.date,
            "order_id": candidate.order_id,
            "split": split_name,
            "rho": float(rho),
            "raw_road_len": int(candidate.raw_road_len),
            "junction_len": int(candidate.junction_len),
            "raw_road_cost": float(candidate.raw_road_cost),
            "gt_cost": float(candidate.gt_cost),
            "dijkstra_cost": float(candidate.dijkstra_cost),
            "gt_cost_ratio": float(candidate.gt_cost_ratio),
            "num_corridor_nodes": int(corridor.num_nodes),
            "num_corridor_edges": int(corridor.num_edges),
            "conversion": candidate.conversion,
            "u_turns": int(candidate.u_turns),
            # relabel 之前的真实 OSM 节点编号，shuffled OD 集要用
            "start_node": int(start),
            "goal_node": int(goal),
        }
        try:
            sample = build_sample_from_observed_path(
                corridor.graph, candidate.junction_path, meta=meta
            )
            validate_sample(sample)
        except (ValueError, AssertionError, nx.NetworkXError) as error:
            funnel.add(f"{split_name}_build_failed")
            _record_outcome(analysis, "build_failed", candidate)
            if funnel.counts[f"{split_name}_build_failed"] <= 3:
                print(f"  [warn] {split_name} build failed: {error}")
            continue
        funnel.add(f"{split_name}_built")
        _record_outcome(analysis, "retained", candidate)
        built.append((candidate, sample))
        if (index + 1) % progress_every == 0:
            print(
                f"  built {split_name} {len(built)}/{index + 1} "
                f"({time.time() - start_time:.1f}s)",
                flush=True,
            )
    return built


def build_shuffled_od(
    test_pairs: Sequence[Tuple[didi.TrajectoryCandidate, Any]],
    graph: nx.Graph,
    distance: didi.ShortestDistanceCache,
    rho: float,
    max_corridor_nodes: Optional[int],
    size: int,
    seed: int,
    funnel: didi.FunnelStats,
    max_rounds: int = 40,
) -> Tuple[List[Any], Dict[str, Any]]:
    """方案第 7.5 节的 shuffled OD 集：这个集合**没有真实 GT path**。

    做法：从 test_1000 取 starts / goals，固定 seed 打乱 goals 后重新配对，排除
    ``s == g``、保证可达、corridor 能建出来。为了复用同一套 ``GraphSample`` 结构，
    这里的 ``gt_path`` 是 Dijkstra 占位（``meta['no_real_gt']=True``），**不参与任何
    路径相似度指标**。

    **必须凑满 ``size`` 条。** 只跑一轮 shuffle 会得到 ``test_1000 - 拒绝数`` 条
    （实测只有 741/1000），等于 shuffled test 自己又被过滤了一次 —— GDP 报的
    shuffled OD Hit Ratio 是在**完整 1000 条**上算的，凑不满就没法直接对照，也容易
    产生 selection bias。所以这里**多轮重新 shuffle**：每一轮换一个 seed 重新配对，
    并用 ``used_pairs`` 去掉已经用过的 OD，直到凑够或达到 ``max_rounds``。

    Returns:
        ``(samples, stats)``；stats 记录 attempts 与每一类拒绝原因（写进 metadata /
        stats.json，任何过滤都不许静默发生）。
    """
    candidate_list = [candidate for candidate, _sample in test_pairs]
    starts = [candidate.junction_path[0] for candidate in candidate_list]
    goals = [candidate.junction_path[-1] for candidate in candidate_list]
    if not starts:
        return [], {"attempts": 0, "built": 0, "rounds": 0}

    stats: Counter = Counter()
    samples: List[Any] = []
    used_pairs: set = set()
    total = len(starts)
    # 原始 test OD 集合：shuffled 集必须是"训练/测试分布以外的新 OD"，
    # 撞上任何一条原始 OD 的配对都要换掉（否则 shuffled 指标里混进了原题）。
    original_pairs = set(zip(starts, goals))

    for round_index in range(int(max_rounds)):
        if len(samples) >= int(size):
            break
        # 关键：置换要作用在 **goals** 上，配成 (starts[position], goals[perm[position]])。
        # 如果只置换下标、然后取 starts[i] 与 goals[i]，配出来的还是**原来那条 OD**，
        # 整个 shuffled OD 集就退化成 test_1000 的副本（实现时真的踩过这个坑：
        # 40 轮里 39009 次 reject_duplicate，因为每一轮配的都是同一批 OD）。
        order = np.random.default_rng(int(seed) + round_index).permutation(total)
        stats["rounds"] += 1
        for position in range(total):
            if len(samples) >= int(size):
                break
            source = int(order[position])
            start, goal = starts[position], goals[source]
            stats["attempts"] += 1
            if start == goal:
                stats["reject_same_od"] += 1
                continue
            if (start, goal) in original_pairs:
                # 撞上原始 test OD（包括置换的不动点），换掉
                stats["reject_original_od"] += 1
                continue
            if (start, goal) in used_pairs:
                stats["reject_duplicate"] += 1
                continue
            if not math.isfinite(distance.distance(start, goal)):
                stats["reject_disconnected"] += 1
                continue
            corridor = didi.build_od_corridor(
                graph, start, goal, rho, max_nodes=max_corridor_nodes
            )
            if corridor is None:
                stats["reject_corridor"] += 1
                continue
            meta = {
                "source": "didi_chengdu",
                "split": "shuffled_od",
                "rho": float(rho),
                "no_real_gt": True,
                "gt_source": "dijkstra_placeholder",
                "start_node": int(start),
                "goal_node": int(goal),
                "num_corridor_nodes": int(corridor.num_nodes),
                "num_corridor_edges": int(corridor.num_edges),
            }
            try:
                sample = build_sample(corridor.graph, start, goal, meta=meta)
                validate_sample(sample)
            except (ValueError, AssertionError, nx.NetworkXError) as error:
                stats["reject_structure"] += 1
                if stats["reject_structure"] <= 3:
                    print(f"  [warn] shuffled OD build failed: {error}")
                continue
            # 占位 GT 明确标注：任何真实路径指标都必须跳过这个集合
            assert sample.meta["gt_source"] == "dijkstra_placeholder"
            used_pairs.add((start, goal))
            samples.append(sample)

    stats["built"] = len(samples)
    for key, value in stats.items():
        funnel.add(f"shuffled_od_{key}", value)
    if len(samples) < int(size):
        # 只报告，不静默：调用方会把这段原样写进 metadata.json
        print(
            f"  [warn] shuffled OD only reached {len(samples)}/{size} after "
            f"{stats['rounds']} rounds; reasons={dict(stats)}"
        )
    return samples, dict(stats)


def write_manifest(
    rows: Sequence[Dict[str, Any]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id", "order_id", "date", "split", "raw_road_len", "junction_len",
        "num_corridor_nodes", "num_corridor_edges", "num_decisions", "gt_cost",
        "dijkstra_cost", "gt_cost_ratio", "corridor_miss", "conversion", "u_turns",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        import csv as _csv

        writer = _csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def run_build(config: Config, paths: Dict[str, Path], args: argparse.Namespace) -> int:
    rho = resolve_rho(config, args)
    graph, stats, idx2edge, columns, lengths = load_real_graph(config, paths)
    filter_cfg = build_filter(config)
    files = didi.discover_trajectory_files(
        paths["root"], str(config.get("data.trajectory_glob", "201610*.csv"))
    )
    max_rows = _max_rows_per_file(config, args)
    max_candidates = _max_candidates(config, args)
    max_corridor_nodes = config.get("data.corridor.max_corridor_nodes", None)
    max_corridor_nodes = (
        None if max_corridor_nodes in (None, 0) else int(max_corridor_nodes)
    )
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))

    print(f"rho          : {rho}")
    print(f"corridor cap : {max_corridor_nodes}")
    print(f"split seed   : {seed}")

    candidates, funnel, from_cache = load_or_collect(
        config, paths, graph, idx2edge, lengths, files, filter_cfg,
        max_rows, max_candidates, args.refresh_cache or bool(config.get("data.refresh_cache", False)),
        bool(config.get("data.use_cache", True)) and not args.no_cache,
        verbose=not args.quiet,
    )
    funnel.add("candidates_collected", len(candidates))
    deduped, dedup_dropped = didi.deduplicate_candidates(candidates)
    funnel.add("dedup_dropped", dedup_dropped)
    funnel.add("after_dedup", len(deduped))
    print(f"candidates   : {len(candidates)} -> after dedup {len(deduped)} "
          f"(dropped {dedup_dropped})")
    if not deduped:
        raise SystemExit("no candidates survived cleaning; check length_column / filters")
    deduped, sampling_stats = _maybe_subsample(deduped, config, args, seed)
    funnel.add("used_for_dataset", len(deduped))

    splits = didi.split_real_paths(deduped, _split_fractions(config), seed=seed)
    distance = didi.ShortestDistanceCache(graph)

    built: Dict[str, List[Tuple[didi.TrajectoryCandidate, Any]]] = {}
    # 被丢弃 vs 被保留的分布对比（rho 取舍的偏差检查）
    analysis: Dict[str, List[Dict[str, float]]] = {}
    for name in ("train", "val", "test"):
        built[name] = build_split(
            splits[name], graph, distance, rho, max_corridor_nodes, name, funnel,
            analysis=analysis,
        )
        print(f"{name:<5s}        : {len(built[name])}/{len(splits[name])} built")

    data_dir = paths["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    sample_id = 0
    for name in ("train", "val", "test"):
        for candidate, _sample in built[name]:
            manifest_rows.append(candidate.to_manifest_row(sample_id, name))
            sample_id += 1

    # 写数据集
    artifacts: Dict[str, str] = {}
    for name in ("train", "val", "test"):
        dataset = GraphQueryDataset(
            [sample for _candidate, sample in built[name]], name=f"didi_chengdu_{name}"
        )
        out = data_dir / f"{name}.pkl"
        dataset.save(out)
        artifacts[name] = str(out)

    # GDP 风格固定 test_1000（方案第 7.4 节）
    subset_size = int(config.get("split.test_subset_size", 1000))
    subset_pairs = _sample_pairs(built["test"], subset_size, seed=seed + 1)
    test_1000 = GraphQueryDataset(
        [sample for _candidate, sample in subset_pairs], name="didi_chengdu_test_1000"
    )
    out = data_dir / "test_1000.pkl"
    test_1000.save(out)
    artifacts["test_1000"] = str(out)

    # shuffled OD（方案第 7.5 节）
    shuffled_size = int(config.get("split.shuffled_od_size", 1000))
    shuffled, shuffled_stats = build_shuffled_od(
        subset_pairs, graph, distance, rho, max_corridor_nodes,
        shuffled_size, seed=seed + 2, funnel=funnel,
    )
    shuffled_dataset = GraphQueryDataset(shuffled, name="didi_chengdu_shuffled_od_1000")
    out = data_dir / "shuffled_od_1000.pkl"
    shuffled_dataset.save(out)
    artifacts["shuffled_od_1000"] = str(out)
    print(
        f"shuffled OD  : {len(shuffled_dataset)}/{shuffled_size} built "
        f"({shuffled_stats.get('rounds')} shuffle round(s), "
        f"attempts={shuffled_stats.get('attempts')}, "
        f"rejects={ {k: v for k, v in shuffled_stats.items() if k.startswith('reject')} })"
    )

    # 全局图
    graph_out = data_dir / "graph_global.pkl"
    with open(graph_out, "wb") as handle:
        pickle.dump(
            {
                "graph": graph,
                "graph_stats": stats.to_dict(),
                "length_column": str(config.get("data.length_column")),
                "rho": float(rho),
                "num_roads_in_dicts": len(idx2edge),
                "edge_features_columns": columns,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    artifacts["graph_global"] = str(graph_out)

    # split manifest
    manifest_path = data_dir / "split_manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    artifacts["split_manifest"] = str(manifest_path)

    # metadata / stats
    metadata = {
        "source": "didi_chengdu",
        "data_root": str(paths["root"]),
        "config": str(paths["config"]),
        "seed": seed,
        "rho": float(rho),
        "length_column": str(config.get("data.length_column")),
        "edge_features_columns": columns,
        "graph": stats.to_dict(),
        "trajectory_filter": filter_cfg.to_dict(),
        "split": _split_fractions(config),
        "split_by_graph": bool(config.get("split.split_by_graph", False)),
        "max_corridor_nodes": max_corridor_nodes,
        "max_rows_per_file": max_rows,
        "max_candidates": max_candidates,
        "max_dataset_samples": _max_dataset_samples(config, args),
        "trajectory_files": [path.name for path in files],
        "candidates_from_cache": from_cache,
        "num_train": len(built["train"]),
        "num_val": len(built["val"]),
        "num_test": len(built["test"]),
        "num_test_1000": len(test_1000),
        "num_shuffled_od": len(shuffled_dataset),
        # 抽样与 shuffled OD 的完整账目（任何过滤都不许静默发生）
        "sampling": sampling_stats,
        "shuffled_od_stats": shuffled_stats,
        "num_shuffled_od_requested": shuffled_size,
        "shuffled_od_reached_target": len(shuffled_dataset) >= int(shuffled_size),
        "note": (
            "gt_path is the OBSERVED historical vehicle route, not a Dijkstra "
            "shortest path; Dijkstra is only used for the C* yardstick."
        ),
    }
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=1, ensure_ascii=False)
    artifacts["metadata"] = str(metadata_path)

    stats_report = build_stats_report(
        config, built, test_1000, shuffled_dataset, funnel, rho,
        shuffled_stats=shuffled_stats, sampling_stats=sampling_stats,
        analysis=analysis,
    )
    stats_path = data_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats_report, handle, indent=1, ensure_ascii=False)
    artifacts["stats"] = str(stats_path)

    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=1))
    print("\nfunnel:")
    print(json.dumps(stats_report["funnel"], indent=1, ensure_ascii=False))
    print("\ncorridor retention:")
    print(json.dumps(stats_report["corridor_retention"], indent=1, ensure_ascii=False))
    print("\ndataset summary (train):")
    print(json.dumps(stats_report["splits"]["train"], indent=1, ensure_ascii=False))
    return 0


def _sample_pairs(
    pairs: Sequence[Tuple[didi.TrajectoryCandidate, Any]], size: int, seed: int
) -> List[Tuple[didi.TrajectoryCandidate, Any]]:
    if size >= len(pairs):
        return list(pairs)
    order = np.random.default_rng(int(seed)).permutation(len(pairs))[:size]
    return [pairs[int(index)] for index in sorted(order.tolist())]


def build_stats_report(
    config: Config,
    built: Dict[str, List[Tuple[didi.TrajectoryCandidate, Any]]],
    test_1000: GraphQueryDataset,
    shuffled: GraphQueryDataset,
    funnel: didi.FunnelStats,
    rho: float,
    shuffled_stats: Optional[Dict[str, Any]] = None,
    sampling_stats: Optional[Dict[str, Any]] = None,
    analysis: Optional[Mapping[str, Sequence[Mapping[str, float]]]] = None,
) -> Dict[str, Any]:
    splits_report: Dict[str, Any] = {}
    for name, pairs in built.items():
        if not pairs:
            splits_report[name] = {"num_samples": 0}
            continue
        dataset = GraphQueryDataset([sample for _c, sample in pairs], name=name)
        summary = dataset.summary()
        ratios = [candidate.gt_cost_ratio for candidate, _s in pairs]
        summary["gt_cost_ratio_stats"] = _percentiles(ratios)
        summary["num_decisions_stats"] = _percentiles(
            [sample.num_decisions for sample in dataset]
        )
        summary["corridor_nodes_stats"] = _percentiles(
            [sample.num_nodes for sample in dataset]
        )
        summary["target_null_fraction"] = float(
            np.mean([np.mean(sample.field.candidates.candidate_is_null) for sample in dataset])
        )
        summary["gt_source"] = sorted(
            {str(sample.meta.get("gt_source", "unknown")) for sample in dataset}
        )
        splits_report[name] = summary

    retention: Dict[str, float] = {}
    for name in ("train", "val", "test"):
        requested = funnel.counts.get(f"{name}_input", 0)
        built_count = funnel.counts.get(f"{name}_built", 0)
        retention[name] = built_count / requested if requested else 0.0
        retention[f"{name}_input"] = requested
        retention[f"{name}_built"] = built_count
        retention[f"{name}_corridor_miss"] = funnel.counts.get(
            f"{name}_corridor_miss", 0
        )
    # split 之间不得有交集（方案第 7.1 节 / 测试第 9 条）
    keys = {
        name: {candidate.dedup_key() for candidate, _s in built[name]}
        for name in ("train", "val", "test")
    }
    overlaps = {
        "train_val": len(keys["train"] & keys["val"]),
        "train_test": len(keys["train"] & keys["test"]),
        "val_test": len(keys["val"] & keys["test"]),
    }
    retention["split_overlaps"] = overlaps

    return {
        "rho": float(rho),
        "funnel": funnel.to_dict(),
        "corridor_retention": retention,
        "splits": splits_report,
        "test_1000": test_1000.summary() if len(test_1000) else {"num_samples": 0},
        "shuffled_od": (
            shuffled.summary() if len(shuffled) else {"num_samples": 0}
        ),
        "shuffled_od_stats": dict(shuffled_stats or {}),
        "shuffled_od_has_real_gt": False,
        # ① corridor miss vs 保留样本的分布对比（防止"绕路/长距离更容易被丢"
        #    造成隐性偏差）；② max_dataset_samples 的分层抽样结果
        "corridor_analysis": {
            "by_outcome": _compare_outcome_groups(analysis or {}),
            "retention_by_gt_cost_ratio": _retention_by_detour_bucket(analysis or {}),
            "note": (
                "retention_by_gt_cost_ratio 是 rho 取舍的偏差检查：如果高绕路比的 GT "
                "保留率明显更低，说明数据集被系统性地推向'接近最短路'的行为分布。"
            ),
        },
        "sampling": dict(sampling_stats or {}),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    overrides = flatten_overrides(args.overrides)
    if args.seed is not None:
        overrides.append(f"seed={int(args.seed)}")
    config = load_config(args.config, overrides)
    paths = resolve_paths(config, args.config, args.out_dir)

    print(f"config       : {args.config}")
    print(f"data root    : {paths['root']}")
    print(f"data dir     : {paths['data_dir']}")
    print(f"length column: {config.get('data.length_column')}")

    if args.scan_only:
        return run_scan_only(config, paths, args)
    if args.scan_corridor:
        return run_scan_corridor(config, paths, args)
    return run_build(config, paths, args)


if __name__ == "__main__":
    raise SystemExit(main())
