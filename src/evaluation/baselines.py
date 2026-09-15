"""Reference baselines (实施指南第 24 节的对照).

    shortest_path   用 Dijkstra 直接算最优解，是 cost ratio 的标尺
    greedy_bfs      每一步走到"离 goal 更近"的邻居（按跳数）
    random_walk     随机游走时只在能靠近 goal 的邻居里选

Greedy 在带权图上会明显变差，这正是从 BFS 任务切换到 Dijkstra 任务的意义。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx


def shortest_path(graph: nx.Graph, start: int, goal: int) -> List[int]:
    """最优路径的节点序列。

    weighted 图上必须是 **Dijkstra**（``weight="weight"``），否则它就不是 cost
    ratio 的 oracle：方案第 13 节要求 Dijkstra baseline 的 CostRatio 恒等于 1.0。
    无权图保持原来的 BFS（``weight=None``），行为逐字不变。
    """
    weight = "weight" if graph.graph.get("weighted", False) else None
    return [
        int(v) for v in nx.shortest_path(graph, int(start), int(goal), weight=weight)
    ]


def shortest_path_cost(graph: nx.Graph, start: int, goal: int) -> float:
    if graph.graph.get("weighted"):
        return float(nx.shortest_path_length(graph, int(start), int(goal), weight="weight"))
    return float(nx.shortest_path_length(graph, int(start), int(goal)))


def greedy_bfs_path(graph: nx.Graph, start: int, goal: int) -> Optional[List[int]]:
    """每次选择"到 goal 的剩余跳数最小"的邻居，禁止回退。"""
    distances = nx.single_source_shortest_path_length(graph, int(goal))
    if int(start) not in distances:
        return None

    path = [int(start)]
    visited = {int(start)}
    current = int(start)
    while current != int(goal):
        neighbours = [
            int(v)
            for v in graph.neighbors(current)
            if v not in visited and v in distances
        ]
        if not neighbours:
            return None
        nxt = min(neighbours, key=lambda v: distances[v])
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path


def random_greedy_path(
    graph: nx.Graph,
    start: int,
    goal: int,
    rng,
    max_steps: int = 4096,
) -> Optional[List[int]]:
    """随机游走，但只在"比当前更接近 goal"的邻居中均匀选。"""
    distances = nx.single_source_shortest_path_length(graph, int(goal))
    if int(start) not in distances:
        return None

    path = [int(start)]
    visited = {int(start)}
    current = int(start)
    for _ in range(max_steps):
        if current == int(goal):
            return path
        neighbours = [
            int(v)
            for v in graph.neighbors(current)
            if v not in visited and v in distances and distances[v] < distances[current]
        ]
        if not neighbours:
            return None
        nxt = int(neighbours[int(rng.integers(0, len(neighbours)))])
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return None


def baseline_summary(dataset, rng=None) -> Dict[str, Dict[str, float]]:
    """在一批 samples 上跑三个 baseline，给出 cost ratio。"""
    import numpy as np

    rng = rng if rng is not None else np.random.default_rng(0)
    stats: Dict[str, Dict[str, float]] = {}

    for name in ("shortest_path", "greedy_bfs", "random_greedy"):
        ratios: List[float] = []
        hits = 0
        for sample in dataset:
            graph = sample.graph
            optimal = shortest_path_cost(graph, sample.start, sample.goal)
            if name == "shortest_path":
                path = shortest_path(graph, sample.start, sample.goal)
            elif name == "greedy_bfs":
                path = greedy_bfs_path(graph, sample.start, sample.goal)
            else:
                path = random_greedy_path(graph, sample.start, sample.goal, rng)
            if path is None:
                ratios.append(float("inf"))
                continue
            hits += 1
            cost = sum(
                float(graph.edges[u, v].get("weight", 1.0))
                for u, v in zip(path[:-1], path[1:])
            )
            ratios.append(cost / optimal if optimal > 0 else float("inf"))

        finite = [r for r in ratios if r != float("inf")]
        stats[name] = {
            "goal_hit_rate": hits / max(len(dataset), 1),
            "mean_cost_ratio": (sum(finite) / len(finite)) if finite else float("nan"),
        }
    return stats


def real_baseline_summary(dataset, rng=None) -> Dict[str, Dict[str, float]]:
    """真实数据上的 baseline 口径（方案第 13 节）。

    Dijkstra 的价值不是"比模型更短"，而是回答：

        真实司机路线 / 模型生成的路线，相对几何最短路到底偏离了多少？

    所以这里对齐真实 GT 报 **nLCS / paired Edge F1 / CostRatio**，而不只是
    GoalHit。没有 ``gt_source == 'observed'`` 的样本（shuffled OD 的 Dijkstra
    占位 GT）会被跳过 —— 拿占位 GT 算相似度等于自己跟自己比。
    """
    import numpy as np

    from src.evaluation import real_path_metrics as rpm

    rng = rng if rng is not None else np.random.default_rng(0)
    stats: Dict[str, Dict[str, float]] = {}

    for name in ("shortest_path", "greedy_bfs", "random_greedy"):
        rows: List[Dict[str, float]] = []
        for sample in dataset:
            if str(sample.meta.get("gt_source", "shortest")) != "observed":
                continue
            graph = sample.graph
            optimal = shortest_path_cost(graph, sample.start, sample.goal)
            if name == "shortest_path":
                path = shortest_path(graph, sample.start, sample.goal)
            elif name == "greedy_bfs":
                path = greedy_bfs_path(graph, sample.start, sample.goal)
            else:
                path = random_greedy_path(graph, sample.start, sample.goal, rng)
            if path is None:
                continue
            cost = sum(
                float(graph.edges[u, v].get("weight", 1.0))
                for u, v in zip(path[:-1], path[1:])
            )
            record = rpm.pair_record(
                path,
                sample.gt_path,
                goal_hit=True,
                gt_cost_ratio=float(sample.meta.get("gt_cost_ratio", float("nan"))),
                pred_cost_ratio=(cost / optimal) if optimal > 0 else float("inf"),
            )
            rows.append(record.to_dict())
        if rows:
            stats[name] = rpm.aggregate_pair_dicts(rows)
    return stats
