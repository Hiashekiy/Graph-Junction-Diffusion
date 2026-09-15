"""Dataset construction (实施指南第 2、5 节 + 修改清单 P0-2 / P1-1 / P1-2).

生成流程：

    G, s, g  ->  (统一 relabel 到 0..N-1)  ->  Branch Segments  ->  z_0

每个 (G, s, g) 都经过语义校验（分支边属于真实图边、source 非 NULL、goal 不是
decision node、off-path junction 为 NULL）。不满足语义的 OD 对直接丢弃。

修改清单落地：
- **P0-2**：同一张底层图的所有 OD query 共享一个 ``graph_id``，``split_dataset``
  按 graph_id 划分，保证 train/val/test 的图集合两两不相交（无 topology leakage）。
- **P1-1（Weighted 扩展后解除）**：``weighted=True`` 现在**不再**报错。模型侧的
  EdgeCostEncoder 让 edge cost 真正进入网络，所以带权任务在信息上变得可辨识；
  但这条链是**可选扩展**：``weighted=false`` 时模型结构、参数集合、checkpoint
  语义与改动前完全一致。
- **P1-2**：``build_sample`` 一进来就把 graph / start / goal 统一 relabel 到
  0..N-1，之后 graph / gt_path / segments 全部共用同一套编号。

真实数据接入（实施方案第 5 节）新增两个函数，**旧函数的语义一个字都没改**：

    relabel_graph_and_path_to_contiguous()    graph 与整条 GT 一起重编号
    build_sample_from_observed_path()         GT = 真实观测路径（不是 Dijkstra）

两条 pipeline 的分工是硬约束，不要互相调用：

    synthetic  ->  build_sample()                    （GT = 最短路）
    real DiDi  ->  build_sample_from_observed_path() （GT = 历史车辆路径）
"""

from __future__ import annotations

import math
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
from src.data.graph_generators import (
    attach_edge_weights,
    generate_connected_graph,
    resolve_edge_weight_spec,
)

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


def relabel_graph_and_path_to_contiguous(
    graph: nx.Graph,
    gt_path: Sequence[Any],
) -> Tuple[nx.Graph, List[int], Dict[Any, int]]:
    """把 graph **和整条 gt_path** 一起重编号到 0..N-1（方案第 5.3 节）。

    只 relabel ``graph/start/goal`` 是不够的：``segments``、``field`` 与
    ``gt_path`` 必须全部落在**同一套**编号空间里，否则 ``build_decision_field``
    会拿旧编号的 GT 去匹配新编号的 branch，静默产出错误的 z_0。

    Returns:
        ``(relabelled_graph, relabelled_gt_path, old_to_new)``。已经是 0..N-1 的
        图原样返回（不复制），此时 ``old_to_new`` 是恒等映射。
    """
    path = list(gt_path)
    nodes = list(graph.nodes())
    if nodes == list(range(len(nodes))):
        return graph, [int(node) for node in path], {node: node for node in nodes}

    mapping = {node: index for index, node in enumerate(sorted(nodes))}
    for node in path:
        if node not in mapping:
            raise ValueError(
                f"gt_path node {node!r} is not a node of the graph; the observed "
                "path and the graph must come from the same construction"
            )
    relabelled = nx.relabel_nodes(graph, mapping, copy=True)
    return relabelled, [int(mapping[node]) for node in path], mapping


# ---------------------------------------------------------------------------
# observed GT sample（方案第 5 节）
# ---------------------------------------------------------------------------
def build_sample_from_observed_path(
    graph: nx.Graph,
    gt_path: Sequence[Any],
    meta: Optional[Dict[str, Any]] = None,
    graph_id: Optional[int] = None,
) -> GraphSample:
    """用**真实车辆历史路径**当 GT 构造样本（方案第 5.2 节）。

    与 :func:`build_sample` 的唯一、也是本质的差别：

        build_sample()                    weighted graph -> Dijkstra -> GT 是最短路
        build_sample_from_observed_path() GT 就是传进来的观测路径

    真实司机路线**未必**是最短路（实测成都数据 GT/Dijkstra cost ratio 中位数
    1.11、p99 2.77），所以真实数据这条链上绝不能出现 ``nx.shortest_path``。
    旧函数语义一个字都没改，synthetic pipeline 完全可复现。

    处理顺序（方案第 5.2 节的硬要求）：

        gt_path 长度 >= 2
          -> start = gt_path[0], goal = gt_path[-1]
          -> graph 与整条 gt_path 一起 relabel 到 0..N-1
          -> bs.set_od()
          -> bs.extract_segments(relabel=False)
          -> build_decision_field(segments, gt_path)
          -> validate_decision_field(segments, field, gt_path)
    """
    path = list(gt_path)
    if len(path) < 2:
        raise ValueError(f"observed gt_path needs at least 2 nodes, got {len(path)}")

    graph, path, mapping = relabel_graph_and_path_to_contiguous(graph, path)
    start, goal = int(path[0]), int(path[-1])
    if start == goal:
        raise ValueError("observed gt_path starts and ends at the same node")

    graph = bs.set_od(graph, start, goal)
    segments = bs.extract_segments(graph, start, goal, relabel=False)
    field = build_decision_field(segments, path)
    validate_decision_field(segments, field, path)

    sample_meta = dict(meta or {})
    sample_meta.setdefault("graph_id", -1 if graph_id is None else int(graph_id))
    # 自证：这份样本的 GT 不是 Dijkstra 算出来的，事后能从 meta 查出来
    sample_meta.setdefault("gt_source", "observed")
    # local -> global 节点编号反查表（**必须存**）：
    # 每个样本的 corridor 都被独立 relabel 成 0..N-1，所以"样本内的编号"在样本之间
    # 没有可比性。凡是**跨样本聚合**的统计（KLEV / JSEV 的 edge visit 分布、以及
    # km-based DTW 的坐标查表）都必须先映射回全局 OSM id，否则会把"A 样本的 0-1 边"
    # 和"B 样本的 0-1 边"当成同一条路。单样本指标（nLCS / Edge F1 / CostRatio）
    # 在局部编号空间里算就够了。
    if not sample_meta.get("local_to_global"):
        inverse: List[Optional[Any]] = [None] * len(mapping)
        for old, new in mapping.items():
            inverse[int(new)] = old
        sample_meta["local_to_global"] = [
            int(node) if isinstance(node, (int, np.integer)) else node
            for node in inverse
        ]

    return GraphSample(
        graph=graph,
        start=start,
        goal=goal,
        gt_path=path,
        segments=segments,
        field=field,
        meta=sample_meta,
    )


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

    # GT：无权 = BFS 最少跳数路径；加权 = Dijkstra 最小 cost 路径。
    # weighted 标记由生成器写进 graph.graph（见 attach_edge_weights），所以这里
    # 用的是同一份事实来源，不存在"图带权但 GT 仍按跳数算"的错配。
    weight_key = "weight" if graph.graph.get("weighted", False) else None
    gt_path = [
        int(v)
        for v in nx.shortest_path(graph, int(start), int(goal), weight=weight_key)
    ]
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
    min_decisions: int = 0,
    edge_weight: Optional[Dict[str, Any]] = None,
) -> GraphQueryDataset:
    """生成 ``num_samples`` 个 (G, s, g) 样本。

    ``graph_type`` 支持：
      * ``"controlled_junction"`` —— 走 :func:`build_controlled_dataset`（推荐，
        指南第 3-12 节的 Controlled Junction Graph）；
      * ``"er" / "ba" / "ws" / "geometric" / "grid"`` —— 旧的随机图生成器。

    ``queries_per_graph`` > 1 时同一张图上抽多个 OD 对；这些样本共享同一个
    ``graph_id``，所以按 graph 划分时它们会一起进同一个 split。

    ``min_decisions`` > 0 时只保留 GT 路径决策数 >= 该值的样本（用于造"长决策链"
    专项评测集；对随机图生成器无意义，会被忽略）。

    ``weighted=True`` 启用带权扩展（方案第 1、5、9、10 节）：

    * 每条边拿到正 cost（``edge_weight`` 决定分布与范围，默认 U(1, 10)）；
    * GT 从 BFS 最少跳数路径换成 **Dijkstra 最小 cost 路径**；
    * 数据集**仍然**按拓扑难度契约（hops / decisions / branch factor）验收 —— 这个
      契约是拓扑层面的，赋权不会改变它。
    """
    weight_spec = resolve_edge_weight_spec(edge_weight)

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
            min_decisions=int(min_decisions),
            weighted=weighted,
            edge_weight=edge_weight,
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
            weighted=bool(weighted),
            weight_range=weight_spec["weight_range"],
            weight_distribution=weight_spec["distribution"],
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
    min_decisions: int = 0,
    weighted: bool = False,
    edge_weight: Optional[Dict[str, Any]] = None,
) -> GraphQueryDataset:
    """用 Controlled Junction Graph 生成样本。

    每张合格图产生一个 OD query（start/goal 由生成器决定），并把难度 / 结构模式 /
    干扰分支统计写进 ``sample.meta``，供 :func:`dataset_statistics` 汇总。

    ``min_decisions`` > 0 时只保留 GT 决策数 >= 该值的样本。这一条是给**长决策链
    专项评测集**用的：正常采样下 decisions >= 9 的样本只占 ~3.7%（hard 档接受率
    只有 4.1%），所以长链桶在 300 条测试集里只有 11 条，统计上什么都说明不了。

    ``weighted=True`` 时每张**已经通过拓扑验收**的图会被赋上正边权，然后 GT 用
    Dijkstra 重算（方案第 10 节）。
    """
    rng = np.random.default_rng(seed)
    weight_spec = resolve_edge_weight_spec(edge_weight)
    samples: List[GraphSample] = []
    attempts = 0
    rejected_by_min_decisions = 0
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

        if weighted:
            # 顺序是硬要求（方案第 10 节）：拓扑验收 -> attach_edge_weights ->
            # Dijkstra 重算 GT。先在无权图上算好 GT、之后才给边加 weight，会得到
            # "GT 是跳数最短路、但图是带权图"的自相矛盾样本。
            # 难度契约（hops / decisions / branch factor）在拓扑层面验收，赋权不改它。
            attach_edge_weights(candidate.graph, rng, **weight_spec)

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
                    "weighted": bool(weighted),
                    "edge_weight": dict(weight_spec),
                },
                graph_id=graph_id,
            )
            validate_sample(sample)
        except (ValueError, RuntimeError, AssertionError, nx.NetworkXError):
            continue
        if min_decisions and int(sample.num_decisions) < int(min_decisions):
            # 长链专项：难度过滤通过、但决策数不够的样本直接丢掉（记数便于报接受率）
            rejected_by_min_decisions += 1
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
    # 记下过滤条件，便于 generate_dataset.py 写进 summary（长链专项评测集要能自证）
    dataset.filter_info = {
        "min_decisions": int(min_decisions),
        "rejected_by_min_decisions": int(rejected_by_min_decisions),
        "weighted": bool(weighted),
        "edge_weight": dict(weight_spec),
    }
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


# ---------------------------------------------------------------------------
# Weighted sanity check（方案第 11 节）
# ---------------------------------------------------------------------------
# 这两个阈值是"报警线"，不是验收标准：如果 weighted 数据集的 conflict rate 只有
# 3%，那说明模型即使完全忽略 edge weight 也能蒙对，这份数据证明不了任何事情。
WEIGHTED_CONFLICT_WARN = 0.20
WEIGHTED_BFS_COST_RATIO_WARN = 1.05


def _path_cost(graph: nx.Graph, path: Sequence[int]) -> float:
    """路径 cost = sum of edge weights（不是跳数）。"""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            return float("inf")
        total += float(graph.edges[u, v].get("weight", 1.0))
    return total


def weighted_statistics(dataset: GraphQueryDataset) -> Dict[str, Any]:
    """weighted 数据集的有效性自检："忽略 edge weight 会怎样"。

        ConflictRate = P(P*_weighted != P*_hop)
        BFSCostRatio = C(P*_hop) / C(P*_weighted)

    另外顺带验证 GT 本身确实就是 Dijkstra 解（gt_path == 加权最短路），这是
    "GT 与权重必须同时生成"这条要求在数据侧的落地检查。
    """
    weights: List[float] = []
    hop_ratios: List[float] = []
    conflicts = 0
    gt_matches_optimal = 0
    counted = 0
    for sample in dataset:
        graph = sample.graph
        if not graph.graph.get("weighted", False):
            continue
        counted += 1
        weights.extend(
            float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True)
        )
        start, goal = int(sample.start), int(sample.goal)
        hop_path = [int(v) for v in nx.shortest_path(graph, start, goal, weight=None)]
        cost_path = [
            int(v) for v in nx.shortest_path(graph, start, goal, weight="weight")
        ]
        conflicts += int(hop_path != cost_path)
        gt_matches_optimal += int(list(sample.gt_path) == cost_path)
        optimal = _path_cost(graph, cost_path)
        hop = _path_cost(graph, hop_path)
        if optimal > 0 and math.isfinite(hop) and math.isfinite(optimal):
            hop_ratios.append(hop / optimal)
    if not counted:
        return {}

    weight_stats = _stats(weights)
    stats: Dict[str, Any] = {
        "num_weighted_samples": counted,
        # 权重本身的长相（方案第 11 节要求逐项报出来）
        "weight_mean": weight_stats["mean"],
        "weight_std": weight_stats["std"],
        "weight_min": weight_stats["min"],
        "weight_max": weight_stats["max"],
        "weighted_conflict_rate": conflicts / counted,
        "bfs_cost_ratio": (
            sum(hop_ratios) / len(hop_ratios) if hop_ratios else float("nan")
        ),
        "bfs_cost_ratio_p90": (
            float(np.percentile(np.asarray(hop_ratios), 90))
            if hop_ratios
            else float("nan")
        ),
        # Dijkstra 对自己恒等于 1：这一项是 oracle 口径的自证（方案第 13 节）
        "dijkstra_cost_ratio": 1.0,
        "gt_path_is_weighted_optimal_fraction": gt_matches_optimal / counted,
    }
    warnings: List[str] = []
    if stats["weighted_conflict_rate"] < WEIGHTED_CONFLICT_WARN:
        warnings.append(
            f"weighted_conflict_rate={stats['weighted_conflict_rate']:.3f} < "
            f"{WEIGHTED_CONFLICT_WARN}: 忽略 edge weight 也能蒙对绝大多数 decision，"
            "这份数据集证明不了 weighted 能力"
        )
    if stats["bfs_cost_ratio"] < WEIGHTED_BFS_COST_RATIO_WARN:
        warnings.append(
            f"bfs_cost_ratio={stats['bfs_cost_ratio']:.3f} < "
            f"{WEIGHTED_BFS_COST_RATIO_WARN}: 加权最优与跳数最优的 cost 差距很小，"
            "success_cost_ratio 会很难区分模型（可把 data.edge_weight.distribution "
            "换成 loguniform 拉大跨度）"
        )
    if gt_matches_optimal != counted:
        warnings.append(
            f"{counted - gt_matches_optimal}/{counted} 个样本的 gt_path 不是加权最短路"
            "（GT 与权重不是同一次生成的）"
        )
    stats["warnings"] = warnings
    return stats

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

    # Weighted 扩展：把 sanity check 一并写进 summary（方案第 11 节）。
    # 无权数据集这里返回 {}，summary 的结构与改动前完全一致。
    weight_stats = weighted_statistics(dataset)
    if weight_stats:
        stats["weighted"] = weight_stats
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
