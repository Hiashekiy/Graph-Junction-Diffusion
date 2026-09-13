"""Dataset construction (实施指南第 2、5 节 + 修改清单 P0-2 / P1-1 / P1-2).

生成流程：

    G, s, g  ->  (统一 relabel 到 0..N-1)  ->  Branch Segments  ->  z_0

每个 (G, s, g) 都经过语义校验（分支边属于真实图边、source 非 NULL、goal 不是
decision node、off-path junction 为 NULL）。不满足语义的 OD 对直接丢弃。

修改清单落地：
- **P0-2**：同一张底层图的所有 OD query 共享一个 ``graph_id``，``split_dataset``
  按 graph_id 划分，保证 train/val/test 的图集合两两不相交（无 topology leakage）。
- **P1-1**：``weighted=True`` 直接抛 NotImplementedError —— 模型当前只能看到
  selected/unselected，看不到 edge cost，带权任务在信息上不可辨识。
- **P1-2**：``build_sample`` 一进来就把 graph / start / goal 统一 relabel 到
  0..N-1，之后 graph / gt_path / segments 全部共用同一套编号。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from src.data import branch_segments as bs
from src.data.controlled_graph import (
    ACCEPT_BRANCH_FACTOR,
    ACCEPT_DECISIONS,
    ACCEPT_HOPS,
    generate_controlled_junction_graph,
)
from src.data.dataset import GraphQueryDataset, GraphSample, validate_sample
from src.data.decision_field import build_decision_field, validate_decision_field
from src.data.graph_generators import generate_connected_graph

CONTROLLED_JUNCTION = "controlled_junction"


# ---------------------------------------------------------------------------
# node relabel（P1-2）
# ---------------------------------------------------------------------------
def relabel_to_contiguous(
    graph: nx.Graph, start: Any, goal: Any
) -> Tuple[nx.Graph, Any, Any]:
    """把节点统一重编号成 0..N-1，返回 (新图, 新 start, 新 goal)。

    编号顺序是 ``sorted(labels)``，对同一张图是确定性的；已经是 0..N-1 的图直接
    原样返回（不复制）。这样 ``graph`` / ``gt_path`` / ``segments`` 只会存在**一套**
    节点编号空间。
    """
    nodes = list(graph.nodes())
    if nodes == list(range(len(nodes))):
        return graph, start, goal

    mapping = {node: index for index, node in enumerate(sorted(nodes))}
    relabelled = nx.relabel_nodes(graph, mapping, copy=True)
    return relabelled, mapping[start], mapping[goal]


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_od_pair(
    graph: nx.Graph,
    rng: np.random.Generator,
    min_distance: int = 0,
    max_attempts: int = 200,
) -> Tuple[int, int]:
    """抽一个满足 d_G(s, g) >= min_distance 的有序 OD 对。

    实现说明：对每个候选 source 只算一次单源最短路，随机挑一个足够远的 goal。
    """
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("graph needs at least two nodes")

    for _ in range(max_attempts):
        start = int(rng.choice(nodes))
        distances = nx.single_source_shortest_path_length(graph, start)
        reachable = [
            node
            for node, distance in distances.items()
            if node != start and distance >= min_distance
        ]
        if reachable:
            goal = int(reachable[int(rng.integers(0, len(reachable)))])
            return start, goal
    raise RuntimeError(
        f"could not sample an OD pair with distance >= {min_distance}"
    )


# ---------------------------------------------------------------------------
# single sample
# ---------------------------------------------------------------------------
def build_sample(
    graph: nx.Graph,
    start: Any,
    goal: Any,
    meta: Optional[Dict[str, Any]] = None,
    graph_id: Optional[int] = None,
) -> GraphSample:
    """从 (G, s, g) 构造完整样本，并校验数据语义。

    P1-2：先把 graph/start/goal 统一 relabel 成 0..N-1，再交给 ``extract_segments``
    （``relabel=False``），保证不存在第二套节点编号空间。
    """
    graph, start, goal = relabel_to_contiguous(graph, start, goal)
    graph = bs.set_od(graph, start, goal)
    segments = bs.extract_segments(graph, start, goal, relabel=False)

    gt_path = [int(v) for v in nx.shortest_path(graph, int(start), int(goal))]
    field = build_decision_field(segments, gt_path)
    validate_decision_field(segments, field, gt_path)

    sample_meta = dict(meta or {})
    # P0-2：同一张底层图的所有 query 共享 graph_id，供 split_dataset 按图划分
    sample_meta.setdefault("graph_id", -1 if graph_id is None else int(graph_id))

    return GraphSample(
        graph=graph,
        start=int(start),
        goal=int(goal),
        gt_path=gt_path,
        segments=segments,
        field=field,
        meta=sample_meta,
    )


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def build_dataset(
    num_samples: int,
    graph_type: str = "er",
    num_nodes: int | Sequence[int] = (20, 40),
    min_od_distance: int = 5,
    seed: int = 0,
    queries_per_graph: int = 1,
    generator_cfg: Optional[Dict[str, Any]] = None,
    weighted: bool = False,
    component_fallback: bool = True,
    max_rejects: int = 5000,
    progress_every: int = 0,
) -> GraphQueryDataset:
    """生成 ``num_samples`` 个 (G, s, g) 样本。

    ``graph_type`` 支持：
      * ``"controlled_junction"`` —— 走 :func:`build_controlled_dataset`（推荐，
        指南第 3-12 节的 Controlled Junction Graph）；
      * ``"er" / "ba" / "ws" / "geometric" / "grid"`` —— 旧的随机图生成器。

    ``queries_per_graph`` > 1 时同一张图上抽多个 OD 对；这些样本共享同一个
    ``graph_id``，所以按 graph 划分时它们会一起进同一个 split。
    """
    if weighted:
        # P1-1：模型边输入只有 selected/unselected，看不到 edge cost。
        # 拓扑/OD/edge-state 相同但 cost 不同的两个样本，模型输入完全一样而最优路径
        # 可能不同 —— 信息上不可辨识，所以第一版直接禁用。
        raise NotImplementedError(
            "V2 baseline does not encode edge cost yet: "
            "weighted=True would make the task unidentifiable from the model's "
            "inputs. Keep weighted=false until an Edge Cost Encoder is implemented."
        )

    if graph_type == CONTROLLED_JUNCTION:
        return build_controlled_dataset(
            num_samples=num_samples,
            seed=seed,
            difficulty_mix=(generator_cfg or {}).get("difficulty_mix"),
            structure_mix=(generator_cfg or {}).get("structure_mix"),
            forced_source_probability=float(
                (generator_cfg or {}).get("source_forced_probability", 0.70)
            ),
            branch_cfg=(generator_cfg or {}).get("branch"),
            progress_every=progress_every,
        )

    rng = np.random.default_rng(seed)
    if isinstance(num_nodes, (int, np.integer)):
        node_lo = node_hi = int(num_nodes)
    else:
        node_lo, node_hi = int(num_nodes[0]), int(num_nodes[1])

    samples: List[GraphSample] = []
    rejects = 0
    graph_id = 0
    while len(samples) < num_samples:
        num = int(rng.integers(node_lo, node_hi + 1))
        graph, params = generate_connected_graph(
            graph_type,
            num,
            rng,
            generator_cfg=generator_cfg,
            min_od_distance=min_od_distance,
            component_fallback=component_fallback,
            weighted=False,
        )
        accepted = 0
        for _ in range(queries_per_graph * 4):
            if accepted >= queries_per_graph or len(samples) >= num_samples:
                break
            try:
                start, goal = sample_od_pair(graph, rng, min_od_distance)
                sample = build_sample(
                    graph.copy(),
                    start,
                    goal,
                    meta={
                        "graph_type": graph_type,
                        "params": params,
                        "num_nodes": graph.number_of_nodes(),
                    },
                    graph_id=graph_id,
                )
                validate_sample(sample)
            except (ValueError, RuntimeError, AssertionError, nx.NetworkXError):
                rejects += 1
                if rejects > max_rejects:
                    raise RuntimeError(
                        f"rejected {rejects} OD pairs while generating {num_samples} "
                        "samples; the data semantics never validated"
                    )
                continue
            samples.append(sample)
            accepted += 1
        graph_id += 1

    return GraphQueryDataset(samples, name=f"{graph_type}_{num_samples}")


# ---------------------------------------------------------------------------
# Controlled Junction Graph 数据集（指南第 3-14 节）
# ---------------------------------------------------------------------------
def build_controlled_dataset(
    num_samples: int,
    seed: int = 0,
    difficulty_mix: Optional[Dict[str, float]] = None,
    structure_mix: Optional[Dict[str, float]] = None,
    forced_source_probability: float = 0.70,
    branch_cfg: Optional[Dict[str, Any]] = None,
    progress_every: int = 0,
    max_attempts_per_sample: int = 300,
) -> GraphQueryDataset:
    """用 Controlled Junction Graph 生成样本。

    每张合格图产生一个 OD query（start/goal 由生成器决定），并把难度 / 结构模式 /
    干扰分支统计写进 ``sample.meta``，供 :func:`dataset_statistics` 汇总。
    """
    rng = np.random.default_rng(seed)
    samples: List[GraphSample] = []
    attempts = 0
    t0 = time.time()
    branch_kwargs: Dict[str, Any] = {}
    if branch_cfg:
        if branch_cfg.get("ordinary_nodes_per_segment"):
            branch_kwargs["ordinary_nodes_per_segment"] = tuple(
                branch_cfg["ordinary_nodes_per_segment"]
            )
        if branch_cfg.get("candidates_per_decision"):
            branch_kwargs["candidates_per_decision"] = tuple(
                branch_cfg["candidates_per_decision"]
            )

    while len(samples) < num_samples:
        candidate = generate_controlled_junction_graph(
            rng,
            difficulty_mix=difficulty_mix,
            structure_mix=structure_mix,
            forced_source_probability=forced_source_probability,
            **branch_kwargs,
        )
        attempts += 1
        if not candidate.accepted:
            if attempts > max_attempts_per_sample * num_samples:
                raise RuntimeError(
                    f"gave up after {attempts} attempts for {num_samples} samples; "
                    f"last reject: {candidate.reject_reason}"
                )
            continue

        graph_id = len(samples)
        try:
            sample = build_sample(
                candidate.graph,
                candidate.start,
                candidate.goal,
                meta={
                    "graph_type": CONTROLLED_JUNCTION,
                    "difficulty": candidate.difficulty,
                    "mode": candidate.mode,
                    "target_decisions": candidate.target_decisions,
                    "target_hops": int(candidate.metrics.get("target_hops", -1)),
                    "distractors": _distractor_counts(candidate.distractors),
                    "num_nodes": candidate.graph.number_of_nodes(),
                },
                graph_id=graph_id,
            )
            validate_sample(sample)
        except (ValueError, RuntimeError, AssertionError, nx.NetworkXError):
            continue
        samples.append(sample)

        if progress_every and len(samples) % progress_every == 0:
            rate = len(samples) / max(time.time() - t0, 1e-6)
            print(
                f"  accepted {len(samples)}/{num_samples} "
                f"(attempts={attempts}, {rate:.1f} samples/s)",
                flush=True,
            )

    dataset = GraphQueryDataset(samples, name=f"controlled_{num_samples}")
    dataset.attempts = attempts
    del rng, attempts, t0, branch_kwargs
    return dataset


def _distractor_counts(distractors: Iterable[Any]) -> Dict[str, int]:
    counts = {"dead_end": 0, "detour": 0, "loop": 0}
    for item in distractors:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    return counts


def _stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def dataset_statistics(dataset: GraphQueryDataset) -> Dict[str, Any]:
    """输出指南第 16 节要求的数据集指标。"""
    if len(dataset) == 0:
        return {"num_graphs": 0, "num_queries": 0}

    rows = []
    for sample in dataset:
        segments = sample.segments
        branch_counts = [len(group) for group in segments.branches]
        candidates = sample.field.candidates
        distractors = sample.meta.get("distractors", {})
        rows.append(
            {
                "nodes": sample.num_nodes,
                "hops": sample.gt_length,
                "decisions": sample.num_decisions,
                "branch_factor": (
                    float(np.mean(branch_counts)) if branch_counts else 0.0
                ),
                "null_fraction": float(np.mean(candidates.candidate_is_null)),
                "source_is_decision": float(segments.start in set(segments.decision_nodes)),
                "source_forced": float(bool(segments.source_forced_edge_ids)),
                "dead_end": float(distractors.get("dead_end", 0)),
                "detour": float(distractors.get("detour", 0)),
                "loop": float(distractors.get("loop", 0)),
            }
        )

    graph_ids = {sample.graph_id for sample in dataset}
    total_distractors = sum(row["dead_end"] + row["detour"] + row["loop"] for row in rows)
    stats: Dict[str, Any] = {
        "num_graphs": len(graph_ids),
        "num_queries": len(dataset),
        "queries_per_graph": len(dataset) / max(len(graph_ids), 1),
        "attempts": getattr(dataset, "attempts", None),
    }
    for key in ("nodes", "hops", "decisions", "branch_factor", "null_fraction"):
        stats[key] = _stats([row[key] for row in rows])

    stats["source_as_decision_fraction"] = float(
        np.mean([row["source_is_decision"] for row in rows])
    )
    stats["source_forced_fraction"] = float(
        np.mean([row["source_forced"] for row in rows])
    )
    for kind in ("dead_end", "detour", "loop"):
        amount = sum(row[kind] for row in rows)
        stats[f"{kind}_branch_fraction"] = (
            amount / total_distractors if total_distractors else 0.0
        )
        stats[f"{kind}_per_graph"] = amount / max(len(rows), 1)

    if any("difficulty" in sample.meta for sample in dataset):
        for level in ("easy", "medium", "hard"):
            count = sum(1 for s in dataset if s.meta.get("difficulty") == level)
            stats[f"difficulty_{level}_fraction"] = count / len(dataset)
        for mode in ("branch_heavy", "long_chain", "loop_detour"):
            count = sum(1 for s in dataset if s.meta.get("mode") == mode)
            stats[f"mode_{mode}_fraction"] = count / len(dataset)

    stats["acceptance_contract"] = {
        "hops": list(ACCEPT_HOPS),
        "decisions": list(ACCEPT_DECISIONS),
        "branch_factor": list(ACCEPT_BRANCH_FACTOR),
    }
    return stats


def _allocate_group_shares(
    total: int, names: Sequence[str], fractions: Dict[str, float]
) -> Tuple[Dict[str, int], List[str]]:
    """按比例把 ``total`` 个 graph 分给各 split，并保证尽量每个 split 非空。

    纯按比例取整会让小数据集出事：4 张图、0.8/0.1/0.1 时 val/test 都会取整成 0，
    于是 val 集为空、validate() 什么都不返回、early-stop / best-checkpoint 直接
    失效（这是真实踩过的坑）。这里在 "有足够的图" 时强制给每个 split 至少一个，
    不够时按参数顺序优先前面的 split，并返回被挤掉的名字供调用方打印警告。
    """
    shares = {name: int(round(float(fractions[name]) * total)) for name in names}

    # 把多余/缺失的名额调整到总和 == total
    while sum(shares.values()) > total:
        largest = max(names, key=lambda name: shares[name])
        shares[largest] -= 1
    while sum(shares.values()) < total:
        shares[names[0]] += 1

    squeezed: List[str] = []
    if total >= len(names):
        changed = True
        while changed:
            changed = False
            empty = [name for name in names if shares[name] == 0]
            if not empty:
                break
            for name in empty:
                donors = [n for n in names if shares[n] > 1]
                if not donors:
                    squeezed.extend(empty)
                    return shares, squeezed
                donor = max(donors, key=lambda n: shares[n])
                shares[donor] -= 1
                shares[name] += 1
                changed = True
    else:
        # 图比 split 还少：只保证前 total 个 split 有图
        squeezed = list(names[total:])
        shares = {name: (1 if index < total else 0) for index, name in enumerate(names)}
    return shares, squeezed


def split_dataset(
    dataset: GraphQueryDataset,
    fractions: Dict[str, float],
    seed: int = 0,
) -> Dict[str, GraphQueryDataset]:
    """按 **graph_id** 划分 train / val / test（修改清单 P0-2）。

    先按 graph_id 归组，再打乱 graph 顺序并切分，最后把每组的所有 query 收集到
    对应的 split。这样同一张图不会同时出现在两个 split 里，避免 topology leakage。

    另外保证：只要图的数量够，每个 split 至少分到一张图（否则 val 为空会让
    ``validate()`` 静默失效）。

    没有 graph_id 的样本（``meta['graph_id'] == -1``）按"一图一样本"处理，退化成
    按样本划分 —— 安全，但会牺牲一点统计效率。
    """
    groups: Dict[Any, List[GraphSample]] = {}
    for index, sample in enumerate(dataset):
        key = sample.graph_id
        if key is None or key == -1:
            key = f"sample-{index}"
        groups.setdefault(key, []).append(sample)

    keys = list(groups)
    order = np.random.default_rng(seed).permutation(len(keys)).tolist()
    shuffled = [keys[i] for i in order]

    names = list(fractions)
    shares, squeezed = _allocate_group_shares(len(shuffled), names, fractions)
    if squeezed:
        print(
            f"[split_dataset] warning: only {len(shuffled)} graph(s) for "
            f"{len(names)} splits; {squeezed} will be empty",
            flush=True,
        )

    splits: Dict[str, GraphQueryDataset] = {}
    cursor = 0
    for name in names:
        count = shares[name]
        collected: List[GraphSample] = []
        for key in shuffled[cursor : cursor + count]:
            collected.extend(groups[key])
        cursor += count
        splits[name] = GraphQueryDataset(collected, name=f"{dataset.name}_{name}")
    return splits


def graph_ids_of(dataset: GraphQueryDataset) -> set:
    """该 split 覆盖的 graph_id 集合（用于验收 topology leakage）。"""
    return {
        sample.graph_id
        for index, sample in enumerate(dataset)
        if sample.graph_id not in (None, -1)
    }


def tiny_overfit_dataset(
    num_samples: int = 16,
    num_nodes: int = 24,
    seed: int = 0,
    graph_type: str = "er",
    min_od_distance: int = 4,
) -> GraphQueryDataset:
    """实施指南第 27 节：固定 seed 的小数据集，用来先验证整条链能否过拟合。"""
    return build_dataset(
        num_samples=num_samples,
        graph_type=graph_type,
        num_nodes=num_nodes,
        min_od_distance=min_od_distance,
        seed=seed,
        queries_per_graph=2,
    )
