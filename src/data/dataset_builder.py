"""Dataset construction (实施指南第 2、5 节).

生成流程：

    G, s, g  ->  Branch Segments  ->  candidate table  ->  z_0

每个 (G, s, g) 都经过语义校验（分支边属于真实图边、source 非 NULL、goal 不是
decision node、off-path junction 为 NULL）。不满足语义的 OD 对直接丢弃，不会
被塞进数据集。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from src.data import branch_segments as bs
from src.data.dataset import GraphQueryDataset, GraphSample, validate_sample
from src.data.decision_field import build_decision_field, validate_decision_field
from src.data.graph_generators import generate_connected_graph


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
    start: int,
    goal: int,
    meta: Optional[Dict[str, Any]] = None,
) -> GraphSample:
    """从 (G, s, g) 构造完整样本，并校验数据语义。

    GT path 与评测口径必须一致：带权图用 Dijkstra（``weight``），无权图用跳数，
    否则训练目标 z_0 编码的是跳数最短路，而 Optimal Path Rate 按 cost 最短路统计。
    """
    graph = bs.set_od(graph, start, goal)
    segments = bs.extract_segments(graph, start, goal)

    weight = "weight" if graph.graph.get("weighted") else None
    gt_path = [
        int(v) for v in nx.shortest_path(graph, int(start), int(goal), weight=weight)
    ]
    field = build_decision_field(segments, gt_path)
    validate_decision_field(segments, field, gt_path)

    return GraphSample(
        graph=graph,
        start=int(start),
        goal=int(goal),
        gt_path=gt_path,
        segments=segments,
        field=field,
        meta=dict(meta or {}),
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
) -> GraphQueryDataset:
    """生成 ``num_samples`` 个 (G, s, g) 样本。

    ``queries_per_graph`` > 1 时同一张图上抽多个 OD 对，用来提高生成效率。
    """
    rng = np.random.default_rng(seed)
    if isinstance(num_nodes, (int, np.integer)):
        node_lo = node_hi = int(num_nodes)
    else:
        node_lo, node_hi = int(num_nodes[0]), int(num_nodes[1])

    samples: List[GraphSample] = []
    rejects = 0
    while len(samples) < num_samples:
        num = int(rng.integers(node_lo, node_hi + 1))
        graph, params = generate_connected_graph(
            graph_type,
            num,
            rng,
            generator_cfg=generator_cfg,
            min_od_distance=min_od_distance,
            component_fallback=component_fallback,
            weighted=weighted,
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

    return GraphQueryDataset(samples, name=f"{graph_type}_{num_samples}")


def split_dataset(
    dataset: GraphQueryDataset,
    fractions: Dict[str, float],
    seed: int = 0,
) -> Dict[str, GraphQueryDataset]:
    """按比例切分 train / val / test（先打乱再切，保证可复现）。"""
    total = len(dataset)
    order = np.random.default_rng(seed).permutation(total)
    names = list(fractions)

    cuts: Dict[str, Tuple[int, int]] = {}
    start = 0
    for index, name in enumerate(names):
        if index == len(names) - 1:
            end = total
        else:
            end = min(start + int(round(float(fractions[name]) * total)), total)
        cuts[name] = (start, end)
        start = end

    return {
        name: GraphQueryDataset(
            [dataset[int(i)] for i in order[lo:hi]], name=f"{dataset.name}_{name}"
        )
        for name, (lo, hi) in cuts.items()
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
