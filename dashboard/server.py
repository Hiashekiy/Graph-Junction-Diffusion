"""Zero-dependency web server for the interactive experiment dashboard.

The browser UI is static HTML/CSS/JavaScript.  This module exposes a small JSON
API that discovers local runs/datasets, reads existing evaluation artifacts and
runs one-query inference through the project's real decoding pipeline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import mimetypes
import sys
import threading
import traceback
import webbrowser
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class RunInfo:
    id: str
    label: str
    path: Path
    checkpoint: Path
    # Weighted 扩展之后，模型分三类，面板需要把这件事显式标出来：
    #   weighted=True  + use_edge_cost=True   -> 带权模型（看得到 edge cost）
    #   weighted=True  + use_edge_cost=False  -> cost 消融对照
    #   weighted=False                        -> 无权模型
    weighted: bool = False
    use_edge_cost: bool = False
    flow_steps: int = 0
    data_dir: str = ""
    train_dataset: str = ""

    @property
    def kind(self) -> str:
        if not self.weighted:
            return "unweighted"
        return "weighted" if self.use_edge_cost else "ablated"

    @property
    def kind_label(self) -> str:
        return {"weighted": "带权", "ablated": "带权·无cost", "unweighted": "无权"}[self.kind]


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    label: str
    path: Path
    size_mb: float
    group: str = ""          # 所属子目录（controlled / long / oldv1 / mixed / smoke …）
    relative: str = ""       # 相对仓库根的可读路径，给面板当提示用


def _relative_id(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def discover_runs(root: Path = PROJECT_ROOT) -> Dict[str, RunInfo]:
    runs_root = root / "outputs" / "runs"
    found: Dict[str, RunInfo] = {}
    if not runs_root.exists():
        return found
    for config_path in sorted(runs_root.rglob("run_config.json")):
        run_dir = config_path.parent
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            continue
        run_id = _relative_id(run_dir, runs_root)
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        if not isinstance(config, Mapping):
            config = {}
        data_cfg = config.get("data") if isinstance(config.get("data"), Mapping) else {}
        model_cfg = config.get("model") if isinstance(config.get("model"), Mapping) else {}
        found[run_id] = RunInfo(
            run_id,
            run_id.replace("/", " / "),
            run_dir,
            checkpoint,
            weighted=bool(data_cfg.get("weighted", False)),
            use_edge_cost=bool(model_cfg.get("use_edge_cost", False)),
            flow_steps=int(model_cfg.get("flow_steps", 0) or 0),
            data_dir=str(config.get("paths", {}).get("data_dir", "")) if isinstance(config.get("paths"), Mapping) else "",
            train_dataset=str(data_cfg.get("train_dataset", "")),
        )
    return found


def discover_datasets(root: Path = PROJECT_ROOT) -> Dict[str, DatasetInfo]:
    """Discover every dataset below ``data/``.

    Datasets live in per-family sub-directories (``controlled`` / ``long`` /
    ``oldv1`` / ``mixed`` / ``smoke``), so the scan is recursive and the listing
    comes back grouped.  The id stays the bare file name: existing
    ``eval*.json`` / ``mp_*.json`` artefacts record their dataset by file name,
    and the id is what the browser sends back on every request.
    """
    data_root = root / "data"
    found: Dict[str, DatasetInfo] = {}
    if not data_root.exists():
        return found
    for path in sorted(data_root.rglob("*.pkl")):
        dataset_id = path.name
        if dataset_id in found:  # never let one group shadow another
            dataset_id = _relative_id(path, data_root)
        label = path.stem.replace("_", " ")
        group = path.parent.name if path.parent != data_root else ""
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:  # 数据集在仓库外（测试用的 tmp_path）
            relative = path.as_posix()
        found[dataset_id] = DatasetInfo(
            dataset_id,
            label,
            path,
            round(path.stat().st_size / (1024 * 1024), 2),
            group,
            relative,
        )
    return found


def _dataset_id_from_path(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name


def _dataset_from_eval_name(name: str) -> Optional[str]:
    stem = Path(name).stem.lower()
    if "oldv1" in stem:
        return "oldv1_test.pkl"
    if "long" in stem:
        return "controlled_long.pkl"
    if "test" in stem:
        return "controlled_test.pkl"
    return None


def _dataset_from_payload(
    payload: Mapping[str, Any], path: Path, run: Optional["RunInfo"] = None
) -> Optional[str]:
    """评测产物的数据集归属。

    优先级：产物自己记录的 ``data``（新格式）> 文件名推断 > 加权 run 的兜底。

    加权数据集是后来才有的，早期产物里没有 ``data`` 字段，而文件名统一叫
    ``eval_test.json`` —— 旧的名字推断会把它错认成无权的 ``controlled_test.pkl``。
    所以对 ``data.weighted=true`` 的 run，"test" 这个名字改判到加权测试集。
    """
    recorded = payload.get("data")
    if recorded:
        candidate = _dataset_id_from_path(str(recorded))
        if candidate.endswith(".pkl"):
            return candidate
    guessed = _dataset_from_eval_name(path.name)
    if run is not None and run.weighted and guessed == "controlled_test.pkl":
        return "weighted_controlled_test.pkl"
    return guessed


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _metric_row(
    run_id: str,
    dataset_id: str,
    decoding: str,
    source: Path,
    metrics: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    keys = (
        "num_queries",
        "goal_hit_rate",
        "optimal_path_rate",
        "success_cost_ratio",
        "loop_rate",
        "broken_rate",
        "mean_elapsed",
        "soft_goal_reachability",
        "coverage_rate",
        "optimal_coverage_rate",
        # Weighted 扩展：加权图上按真实 cost 判定的最优覆盖率
        "weighted_optimal_coverage_rate",
        # 多分支增强：路径表规模与"必死 branch 预筛选"的统计
        "mean_goal_paths",
        "mean_finished_paths",
        "mean_filtered_dead_branches",
        "mean_pruned",
    )
    row: Dict[str, Any] = {
        "model": run_id,
        "dataset": dataset_id,
        "decoding": decoding,
        "source": source.name,
    }
    row.update({key: metrics.get(key) for key in keys})
    if extra:
        row.update(extra)
    return _finite(row)


def discover_metrics(
    runs: Mapping[str, RunInfo], root: Path = PROJECT_ROOT
) -> List[Dict[str, Any]]:
    """Read compact metric summaries without returning per-query records."""
    rows: List[Dict[str, Any]] = []
    for run_id, run in runs.items():
        for path in sorted(run.path.glob("eval*.json")):
            # Flow-step files are ablations, not distinct trained models.  Keeping
            # them out avoids silently mixing inference configurations.
            if "_flow" in path.stem:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            metrics = payload.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            dataset_id = _dataset_from_payload(payload, path, run)
            if dataset_id is None:
                continue
            decoding = "multi" if "multi" in path.stem else "single"
            extra = {
                key: payload[key]
                for key in (
                    "coverage_rate",
                    "optimal_coverage_rate",
                    "weighted_optimal_coverage_rate",
                )
                if key in payload
            }
            extra["weighted"] = bool(run.weighted) or dataset_id.startswith("weighted")
            # 新格式（``--decode multi``）：payload["multi"] 里一次带三条口径，各占一行。
            multi = payload.get("multi")
            if isinstance(multi, Mapping):
                info = multi.get("info") if isinstance(multi.get("info"), Mapping) else {}
                shared = dict(extra)
                for key in (
                    "coverage_rate",
                    "optimal_coverage_rate",
                    "mean_goal_paths",
                    "mean_finished_paths",
                    "mean_filtered_dead_branches",
                ):
                    if key in info:
                        shared[key] = info[key]
                for mode_key, suffix in (
                    ("multi_best", ""),
                    ("multi_best_goal", " · best_goal"),
                    ("multi_best_goal_cost", " · best_goal_cost"),
                ):
                    mode_metrics = multi.get(mode_key)
                    if isinstance(mode_metrics, Mapping):
                        rows.append(
                            _metric_row(run_id, dataset_id, decoding + suffix, path, mode_metrics, shared)
                        )
                continue
            rows.append(_metric_row(run_id, dataset_id, decoding, path, metrics, extra))

        for path in sorted(run.path.glob("mp_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            results = payload.get("results") or []
            if not results or not isinstance(results[0], Mapping):
                continue
            dataset_id = _dataset_id_from_path(payload.get("data", ""))
            result = results[0]
            if not dataset_id or not isinstance(result.get("multi_best"), Mapping):
                continue
            base = f"multi k={payload.get('top_k', '?')} / {payload.get('null_policy', 'stop')}"
            extra = {
                "coverage_rate": result.get("coverage_rate"),
                "optimal_coverage_rate": result.get("optimal_coverage_rate"),
                "weighted_optimal_coverage_rate": result.get("weighted_optimal_coverage_rate"),
                "mean_goal_paths": result.get("mean_goal_paths"),
                "mean_finished_paths": result.get("mean_finished_paths"),
                "mean_filtered_dead_branches": result.get("mean_filtered_dead_branches"),
                "mean_pruned": result.get("mean_pruned"),
                "weighted": bool(result.get("dataset_is_weighted", False)),
            }
            # 一次搜索给出三条口径（方案第 8 节）：每条各占一行，面板上可以直接对比。
            for mode_key, suffix in (
                ("multi_best", ""),
                ("multi_best_goal", " · best_goal"),
                ("multi_best_goal_cost", " · best_goal_cost"),
            ):
                metrics = result.get(mode_key)
                if not isinstance(metrics, Mapping):
                    continue
                rows.append(
                    _metric_row(run_id, dataset_id, base + suffix, path, metrics, extra)
                )

    # 同一个 (run, dataset, decoding) 有多份产物时：best.pt 的评测优先于 `*_last.json`，
    # 其余情况保留排序靠后（更新）的那份。
    unique: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["dataset"], row["decoding"])
        previous = unique.get(key)
        if previous is not None:
            previous_is_last = "_last" in str(previous.get("source", ""))
            current_is_last = "_last" in str(row.get("source", ""))
            if current_is_last and not previous_is_last:
                continue
        unique[key] = row
    return sorted(unique.values(), key=lambda row: (row["dataset"], row["model"], row["decoding"]))


#: 面板要展示的报告 / 汇总产物。(id, 标题, 仓库内相对路径)
REPORT_ARTIFACTS: Tuple[Tuple[str, str, str], ...] = (
    ("report", "多分支解码 / 加权模型 评测报告", "docs/REPORT_multipath_and_weighted.md"),
    ("all_models", "全模型统一评测汇总（8 配置 × 2 测试集）", "outputs/all_models_multipath_summary.json"),
    ("weighted_experiment", "加权模型 vs cost 消融（单路径口径）", "outputs/weighted_experiment.json"),
    ("multipath_weighted", "加权模型 beam=64：过滤 on/off 对照", "outputs/multipath_weighted_summary.json"),
    ("multipath_filteron", "只开过滤的 beam=64 结果", "outputs/multipath_weighted_filteron_summary.json"),
    ("beam_compare", "beam=64 vs beam=3 对照（CPU）", "outputs/multipath_weighted_beam_compare.json"),
    ("regression", "零破坏回归：旧 checkpoint 逐位复现", "outputs/runs/v2_rev2_mixed/regression_after_weighted_extension.json"),
)


def discover_reports(root: Path = PROJECT_ROOT) -> List[Dict[str, Any]]:
    """报告 / 汇总产物的清单（只读，路径写死，不做目录遍历）。"""
    reports: List[Dict[str, Any]] = []
    for report_id, label, relative in REPORT_ARTIFACTS:
        path = root / relative
        exists = path.is_file()
        reports.append(
            {
                "id": report_id,
                "label": label,
                "path": relative,
                "kind": "markdown" if path.suffix.lower() == ".md" else "json",
                "exists": exists,
                "size_kb": round(path.stat().st_size / 1024, 1) if exists else None,
            }
        )
    return reports


def _graph_payload(sample: Any) -> Dict[str, Any]:
    import networkx as nx

    graph = sample.graph
    if graph.number_of_nodes() == 1:
        positions = {next(iter(graph.nodes)): (0.5, 0.5)}
    else:
        raw = nx.spring_layout(graph, seed=17, iterations=120)
        xs = [float(point[0]) for point in raw.values()]
        ys = [float(point[1]) for point in raw.values()]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
        positions = {
            node: ((float(point[0]) - x0) / dx, (float(point[1]) - y0) / dy)
            for node, point in raw.items()
        }
    decisions = {int(node) for node in sample.segments.decision_nodes}
    nodes = []
    for node in sorted(graph.nodes):
        kind = "ordinary"
        if int(node) == int(sample.start):
            kind = "start"
        elif int(node) == int(sample.goal):
            kind = "goal"
        elif int(node) in decisions:
            kind = "decision"
        x, y = positions[node]
        nodes.append({"id": int(node), "x": x, "y": y, "kind": kind})
    edges = [{"source": int(u), "target": int(v)} for u, v in graph.edges]
    return {"nodes": nodes, "edges": edges}


def _has_edge_weights(graph: Any) -> bool:
    """图上的边是否带 weight（无权图的边没有这个属性）。"""
    try:
        return any("weight" in data for _, _, data in graph.edges(data=True))
    except (AttributeError, TypeError, ValueError):
        return False


def _weights_cost(graph: Any, edges: Iterable[Any], fallback: float) -> float:
    """按边权求和；图对象不支持按边取属性时退回 fallback（= 跳数）。

    测试与工具里会用轻量 stub 当图（只有 ``has_edge``），所以这里必须容错，
    不能让面板因为一个 stub 就 500。
    """
    try:
        return float(
            sum(float(graph.edges[u, v].get("weight", 1.0)) for u, v in edges)
        )
    except (AttributeError, TypeError, KeyError, ValueError):
        return float(fallback)


def _route_payload(path: Any, rank: int, graph: Any) -> Dict[str, Any]:
    if hasattr(path, "nodes"):
        nodes = path.nodes
        status = path.status
        reason = path.reason
        log_prob = path.log_prob
        cost = path.cost
    else:
        nodes = path.path
        status = path.status
        reason = path.reason
        log_prob = None
        cost = max(len(nodes) - 1, 0)
    node_ids = [int(node) for node in nodes]
    edges = [[source, target] for source, target in zip(node_ids[:-1], node_ids[1:])]
    invalid_edges = [edge for edge in edges if not graph.has_edge(*edge)]
    if invalid_edges:
        raise ValueError(f"解码路径包含不属于原图的边：{invalid_edges[:3]}")
    # 真实 cost = sum of edge weights；无权图（没有 weight 属性）自然退化成跳数。
    # 多分支解码器已经在 PathCandidate.path_cost 里算好了，单路径在这里补算。
    path_cost = getattr(path, "path_cost", None)
    if path_cost is None:
        path_cost = _weights_cost(graph, edges, fallback=float(cost))
    return _finite(
        {
            "rank": rank,
            "nodes": node_ids,
            "edges": edges,
            "status": status,
            "reason": reason,
            "cost": int(cost),
            "path_cost": float(path_cost),
            "weighted": _has_edge_weights(graph),
            "log_prob": log_prob,
        }
    )


def _decode_nodes(sample: Any, state: Any) -> Dict[str, Any]:
    """把「链最终携带的状态」解出来，作为诊断口径（不是 single 的答案）。"""
    from src.evaluation.path_decoder import decode_flat

    decoded = decode_flat(sample, state, decision_offset=0, candidate_offset=0)
    return {
        "status": decoded.status,
        "reason": decoded.reason,
        **_path_stats(sample, decoded.path),
    }


def _path_stats(sample: Any, nodes: Iterable[Any]) -> Dict[str, Any]:
    """路径的跳数与真实 cost（加权图按边权）。"""
    node_ids = [int(node) for node in nodes]
    edges = [[u, v] for u, v in zip(node_ids[:-1], node_ids[1:])]
    hops = max(len(node_ids) - 1, 0)
    return {"hops": hops, "path_cost": _weights_cost(sample.graph, edges, fallback=float(hops))}


def _single_decode_contrast(
    sample: Any,
    frames: List[Dict[str, Any]],
    decoded: Any,
    chain_state: Any,
    readout: str,
    stochastic: bool,
) -> Optional[Dict[str, Any]]:
    """把「single 解码实际用的状态」与「扩散链自己携带的状态」摆在一起。

    现在两者**默认就不是同一个东西**（方案要求）：

    * single 解码用 readout：最终 candidate_prob 的**组内 argmax**（``final_prob_argmax`），
      或者勾选确定性开关后的全程 argmax rollout（``deterministic_rollout`）；
    * 扩散链自己携带的状态（``chain_state`）默认是**采样**出来的，只用于可视化/诊断。

    实测（加权测试集 #92，seed 0）：readout 18 跳到达（node 8 -> 9），而采样链在最后一个
    路口抽到 NULL → 16 跳 broken。面板把两者都摊开，避免再被误读成解码 bug。
    """
    if not frames:
        return None
    last = frames[-1]
    clean_path = [int(node) for node in last.get("clean_path", [])]
    chain_decoded = _decode_nodes(sample, chain_state)
    return {
        "readout_mode": readout,
        "chain_stochastic": bool(stochastic),
        "chain_state": chain_decoded,
        "readout": {
            "status": decoded.status,
            "reason": decoded.reason,
            **_path_stats(sample, decoded.path),
        },
        "clean_prediction": {
            "status": last.get("clean_status"),
            "reason": last.get("clean_reason"),
            **_path_stats(sample, clean_path),
        },
        "clean_path": clean_path,
    }


def _diffusion_payload(sample: Any, trace: Any, candidate_owner: Any) -> Dict[str, Any]:
    """Serialize the real reverse-chain trace in a compact browser-friendly form.

    Each categorical state contains one choice for every decision variable,
    including decisions that are unreachable from the sample start.  Decode
    both states here so the UI can distinguish the actual reachable path from
    that full decision field.
    """
    import torch

    from src.evaluation.path_decoder import decode_flat
    from src.training.losses import grouped_argmax

    candidates = sample.field.candidates
    candidate_visuals: List[Dict[str, Any]] = []
    for index, owner in enumerate(candidates.candidate_owner):
        branch = candidates.candidate_branch[index]
        edges: List[List[int]] = []
        end: Optional[int] = None
        if branch is not None:
            edges = [
                [int(source), int(target)]
                for source, target in zip(branch.nodes[:-1], branch.nodes[1:])
            ]
            end = int(branch.end)
        candidate_visuals.append(
            {
                "index": index,
                "owner": int(sample.segments.decision_nodes[int(owner)]),
                "end": end,
                "is_null": bool(candidates.candidate_is_null[index]),
                "edges": edges,
            }
        )

    frames: List[Dict[str, Any]] = []
    previous_clean = None
    expected_owners = [int(node) for node in sample.segments.decision_nodes]

    def selected_indices(state: Any) -> List[int]:
        indices = [int(value) for value in state.detach().cpu().tolist()]
        owners = [candidate_visuals[index]["owner"] for index in indices]
        if owners != expected_owners:
            raise RuntimeError(
                "扩散状态必须为每个 decision node 且仅选择一个 candidate；"
                f"期望 owners={expected_owners}，实际 owners={owners}"
            )
        return indices

    for offset, log_prob in enumerate(trace.log_prob):
        clean = grouped_argmax(log_prob, candidate_owner, sample.num_decisions)
        noisy = trace.z_path[offset + 1]
        input_noisy = trace.z_path[offset]
        clean_decoded = decode_flat(sample, clean)
        noisy_decoded = decode_flat(sample, noisy)
        confidence = torch.exp(log_prob[clean]).mean().item() if clean.numel() else 0.0
        frames.append(
            {
                "t": len(trace.log_prob) - offset,
                "clean": selected_indices(clean),
                "noisy": selected_indices(noisy),
                "clean_path": [int(node) for node in clean_decoded.path],
                "clean_status": clean_decoded.status,
                "clean_reason": clean_decoded.reason,
                "noisy_path": [int(node) for node in noisy_decoded.path],
                "noisy_status": noisy_decoded.status,
                "noisy_reason": noisy_decoded.reason,
                "mean_confidence": float(confidence),
                "noisy_changed": int((noisy != input_noisy).sum().item()),
                "clean_changed": (
                    None
                    if previous_clean is None
                    else int((clean != previous_clean).sum().item())
                ),
            }
        )
        previous_clean = clean

    forced_nodes = [int(node) for node in sample.segments.source_forced_nodes]
    return {
        "T": len(frames),
        "candidates": candidate_visuals,
        "forced_edges": [
            [source, target]
            for source, target in zip(forced_nodes[:-1], forced_nodes[1:])
        ],
        "frames": frames,
    }


class DashboardService:
    def __init__(self, root: Path = PROJECT_ROOT, device: str = "auto") -> None:
        self.root = root
        self.device_request = device
        self.runs = discover_runs(root)
        self.datasets = discover_datasets(root)
        self.reports = discover_reports(root)
        self._dataset_cache: Dict[str, Any] = {}
        self._model_key: Optional[str] = None
        self._model_bundle: Optional[Tuple[Any, Any, Any, Any]] = None
        self._lock = threading.RLock()

    def catalog(self) -> Dict[str, Any]:
        return {
            "models": [
                {
                    "id": item.id,
                    "label": item.label,
                    "kind": item.kind,
                    "kind_label": item.kind_label,
                    "weighted": item.weighted,
                    "use_edge_cost": item.use_edge_cost,
                    "flow_steps": item.flow_steps,
                    "data_dir": item.data_dir,
                }
                for item in self.runs.values()
            ],
            "datasets": [
                {
                    "id": item.id,
                    "label": item.label,
                    "size_mb": item.size_mb,
                    "group": item.group,
                    "relative": item.relative,
                }
                for item in self.datasets.values()
            ],
            "metrics": discover_metrics(self.runs, self.root),
            "reports": self.reports,
        }

    def report(self, report_id: str) -> Dict[str, Any]:
        """取一份报告内容：Markdown 返回原文，JSON 返回解析后的对象。"""
        entry = next((item for item in self.reports if item["id"] == report_id), None)
        if entry is None:
            raise ValueError(f"未知报告：{report_id}")
        path = self.root / entry["path"]
        if not path.is_file():
            raise ValueError(f"报告文件不存在：{entry['path']}")
        text = path.read_text(encoding="utf-8")
        payload: Dict[str, Any] = dict(entry)
        if entry["kind"] == "json":
            payload["json"] = json.loads(text)
        else:
            payload["markdown"] = text
        return payload

    def _dataset(self, dataset_id: str) -> Any:
        info = self.datasets.get(dataset_id)
        if info is None:
            raise ValueError(f"未知数据集：{dataset_id}")
        if dataset_id not in self._dataset_cache:
            from src.data.dataset import GraphQueryDataset

            # Retain at most two datasets to avoid keeping every large pkl in RAM.
            if len(self._dataset_cache) >= 2:
                self._dataset_cache.pop(next(iter(self._dataset_cache)))
            self._dataset_cache[dataset_id] = GraphQueryDataset.load(info.path)
        return self._dataset_cache[dataset_id]

    def dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        with self._lock:
            dataset = self._dataset(dataset_id)
            return {"id": dataset_id, "num_queries": len(dataset), "name": dataset.name}

    def _bundle(self, run_id: str) -> Tuple[Any, Any, Any, Any]:
        if run_id not in self.runs:
            raise ValueError(f"未知模型：{run_id}")
        if self._model_key == run_id and self._model_bundle is not None:
            return self._model_bundle

        import torch
        from src.training.checkpoint import load_checkpoint
        from src.training.setup import build_diffusion, build_model, get_device
        from src.utils.config import load_config

        if self._model_bundle is not None:
            self._model_bundle = None
            self._model_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        info = self.runs[run_id]
        config = load_config(info.path / "run_config.json")
        requested = self.device_request
        if requested == "auto":
            requested = str(config.get("training.device", "auto"))
        device = get_device(requested)
        model = build_model(config, device)
        checkpoint = load_checkpoint(info.checkpoint, model=model, map_location=device)
        model = model.to(device)
        model.eval()
        diffusion = build_diffusion(config)
        self._model_key = run_id
        self._model_bundle = (model, diffusion, device, checkpoint)
        return self._model_bundle

    def infer(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        import torch
        from src.data.collate import collate_samples
        from src.diffusion.sampler import sample_reverse_chain
        from src.evaluation.multi_path_decoder import decode_multi_path
        from src.evaluation.path_decoder import decode_flat
        from src.utils.seed import make_generator, set_seed

        run_id = str(request.get("model", ""))
        dataset_id = str(request.get("dataset", ""))
        index = int(request.get("index", 0))
        seed = int(request.get("seed", 0))
        decode = str(request.get("decode", "single"))
        if decode not in {"single", "multi"}:
            raise ValueError("decode 必须是 single 或 multi")

        with self._lock, torch.no_grad():
            dataset = self._dataset(dataset_id)
            if index < 0 or index >= len(dataset):
                raise ValueError(f"样本下标越界：{index}（数据集共 {len(dataset)} 条）")
            sample = dataset[index]
            model, diffusion, device, checkpoint = self._bundle(run_id)
            set_seed(seed)
            batch = collate_samples([sample], device=device)
            include_diffusion = bool(request.get("include_diffusion", False))
            chain = sample_reverse_chain(
                diffusion,
                model,
                batch,
                generator=make_generator(seed, device="cpu"),
                stochastic=not bool(request.get("deterministic", False)),
                record=include_diffusion,
            )
            candidate_prob = chain["candidate_prob"][: sample.num_candidates]
            deterministic = bool(request.get("deterministic", False))
            if decode == "single":
                # 新默认 single readout：最终 candidate_prob 的组内 argmax（链本身仍然采样）。
                # 勾了确定性开关时，链本身就是"每步 posterior argmax"的 rollout，解码它自己的
                # 最终状态即可 —— 那是另一条路径（deterministic_rollout），不要和默认混为一谈。
                if deterministic:
                    state = chain["z0"]
                    readout = "deterministic_rollout"
                else:
                    from src.evaluation.readout import single_path_state

                    state = single_path_state(chain, batch, "single")
                    readout = "final_prob_argmax"
                decoded = decode_flat(sample, state, decision_offset=0, candidate_offset=0)
                routes = [_route_payload(decoded, 1, sample.graph)]
                summary = {
                    "coverage": decoded.status == "goal",
                    "num_finished": 1,
                    "num_goal_paths": int(decoded.status == "goal"),
                    "pruned": 0,
                    "num_filtered_dead_branches": 0,
                }
            else:
                top_k = max(1, min(int(request.get("top_k", 2)), 6))
                beam_width = max(1, min(int(request.get("beam_width", 64)), 256))
                display_paths = max(1, min(int(request.get("display_paths", 8)), 24))
                null_policy = str(request.get("null_policy", "stop"))
                filter_dead_branches = bool(request.get("filter_dead_branches", False))
                decoded = decode_multi_path(
                    sample,
                    candidate_prob,
                    top_k=top_k,
                    beam_width=beam_width,
                    null_policy=null_policy,
                    filter_dead_branches=filter_dead_branches,
                )
                routes = [
                    _route_payload(path, rank + 1, sample.graph)
                    for rank, path in enumerate(decoded.finished[:display_paths])
                ]
                summary = decoded.summary()

            response = {
                    "model": run_id,
                    "dataset": dataset_id,
                    "index": index,
                    "num_queries": len(dataset),
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    "sample": {
                        "start": int(sample.start),
                        "goal": int(sample.goal),
                        "num_nodes": int(sample.num_nodes),
                        "num_decisions": int(sample.num_decisions),
                        "gt_length": int(sample.gt_length),
                        "difficulty": sample.meta.get("difficulty", "n/a"),
                        "mode": sample.meta.get("mode", "n/a"),
                        "gt_path": [int(node) for node in sample.gt_path],
                    },
                    "graph": _graph_payload(sample),
                    "routes": routes,
                    "summary": summary,
                }
            if include_diffusion:
                response["diffusion"] = _diffusion_payload(
                    sample, chain["trace"], batch.candidate_owner
                )
                if decode == "single":
                    # single readout vs 链实际携带的（采样）状态：面板直接对照
                    response["contrast"] = _single_decode_contrast(
                        sample,
                        response["diffusion"]["frames"],
                        decoded,
                        chain["z0"],
                        readout,
                        stochastic=not deterministic,
                    )
            return _finite(response)


class DashboardHandler(BaseHTTPRequestHandler):
    service: DashboardService

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[dashboard] " + fmt % args + "\n")

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_finite(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/catalog":
            self._json(self.service.catalog())
            return
        if parsed.path.startswith("/api/reports/"):
            try:
                report_id = unquote(parsed.path.removeprefix("/api/reports/"))
                self._json(self.service.report(report_id))
            except Exception as exc:  # user-facing validation boundary
                self._error(str(exc))
            return
        if parsed.path.startswith("/api/datasets/"):
            try:
                dataset_id = unquote(parsed.path.removeprefix("/api/datasets/"))
                self._json(self.service.dataset_info(dataset_id))
            except Exception as exc:  # user-facing validation boundary
                self._error(str(exc))
            return
        self._static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/path":
            self._error("Not found", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                raise ValueError("请求体为空或过大")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("请求必须是 JSON object")
            self._json(self.service.infer(payload))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._error(str(exc))
        except Exception as exc:  # keep the server alive and return a useful message
            traceback.print_exc()
            self._error(f"推理失败：{exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else unquote(request_path.lstrip("/"))
        candidate = (STATIC_ROOT / relative).resolve()
        try:
            candidate.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._error("Not found", HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self._error("Not found", HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(body)))
        # Always re-read the assets: this is a local workbench and a browser
        # that keeps serving yesterday's app.js makes the panel look broken.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-Junction-Diffusion interactive dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = DashboardService(PROJECT_ROOT, device=args.device)
    handler = type("BoundDashboardHandler", (DashboardHandler,), {"service": service})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Graph-Junction-Diffusion dashboard: {url}")
    print(f"发现 {len(service.runs)} 个模型、{len(service.datasets)} 个数据集")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
