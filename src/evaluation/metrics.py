"""Evaluation metrics (实施指南第 24 节).

第一版主指标只有：

    Goal Hit Rate          #到达 g / #queries
    Optimal Path Rate      成功路径的 cost == GT 最短路径 cost
    Success Cost Ratio     L_pred / L_optimal（只在成功样本上统计）
    Loop Rate              重复访问 structural node
    Broken Rate            选 NULL / 走到 dead-end / 越界
    Inference Time         每条 query 的平均采样耗时

``decision accuracy`` 只作为 debug metric，不作为模型选择主指标。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import networkx as nx

from src.evaluation.path_decoder import DecodeResult


def path_cost(graph: nx.Graph, path: Sequence[int]) -> float:
    """路径 cost（无权重图就是跳数）。"""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if graph.has_edge(u, v):
            total += float(graph.edges[u, v].get("weight", 1.0))
        else:
            # 走到了一条不存在的边：按无穷大处理，调用方会把它当 broken
            return float("inf")
    return total


def is_valid_path(graph: nx.Graph, path: Sequence[int], start: int, goal: int) -> bool:
    if not path or path[0] != start or path[-1] != goal:
        return False
    return all(graph.has_edge(u, v) for u, v in zip(path[:-1], path[1:]))


@dataclass
class SampleRecord:
    """一个 query 的评测记录。"""

    status: str
    goal_hit: bool
    optimal: bool
    pred_cost: float
    optimal_cost: float
    cost_ratio: float
    num_nodes_in_path: int
    num_branches: int
    gt_length: int
    reason: str = ""
    elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_sample(
    sample,
    result: DecodeResult,
    elapsed: float = 0.0,
) -> SampleRecord:
    """把一个解码结果变成可累加的记录。"""
    graph = sample.graph
    optimal_cost = float(
        nx.shortest_path_length(graph, sample.start, sample.goal, weight="weight")
        if graph.graph.get("weighted")
        else nx.shortest_path_length(graph, sample.start, sample.goal)
    )

    if result.status != "goal":
        return SampleRecord(
            status=result.status,
            goal_hit=False,
            optimal=False,
            pred_cost=float("inf"),
            optimal_cost=optimal_cost,
            cost_ratio=float("inf"),
            num_nodes_in_path=len(result.path),
            num_branches=result.num_branches,
            gt_length=sample.gt_length,
            reason=result.reason,
            elapsed=elapsed,
        )

    pred_cost = path_cost(graph, result.path)
    valid = is_valid_path(graph, result.path, sample.start, sample.goal)
    if not valid:
        return SampleRecord(
            status="broken",
            goal_hit=False,
            optimal=False,
            pred_cost=pred_cost,
            optimal_cost=optimal_cost,
            cost_ratio=float("inf"),
            num_nodes_in_path=len(result.path),
            num_branches=result.num_branches,
            gt_length=sample.gt_length,
            reason="decoded path contains a non-edge",
            elapsed=elapsed,
        )

    ratio = pred_cost / optimal_cost if optimal_cost > 0 else float("inf")
    return SampleRecord(
        status="goal",
        goal_hit=True,
        optimal=abs(pred_cost - optimal_cost) < 1e-6,
        pred_cost=pred_cost,
        optimal_cost=optimal_cost,
        cost_ratio=ratio,
        num_nodes_in_path=len(result.path),
        num_branches=result.num_branches,
        gt_length=sample.gt_length,
        reason="",
        elapsed=elapsed,
    )


def aggregate(records: Sequence[SampleRecord]) -> Dict[str, float]:
    """把逐样本记录汇总成主指标。"""
    total = len(records)
    if total == 0:
        return {"num_queries": 0}

    hits = [r for r in records if r.goal_hit]
    ratios = [r.cost_ratio for r in hits]
    return {
        "num_queries": float(total),
        "goal_hit_rate": len(hits) / total,
        "optimal_path_rate": sum(1 for r in records if r.optimal) / total,
        "success_cost_ratio": (sum(ratios) / len(ratios)) if ratios else float("nan"),
        "loop_rate": sum(1 for r in records if r.status == "loop") / total,
        "broken_rate": sum(1 for r in records if r.status == "broken") / total,
        "mean_pred_cost": (sum(r.pred_cost for r in hits) / len(hits)) if hits else float("nan"),
        "mean_optimal_cost": sum(r.optimal_cost for r in records) / total,
        "mean_path_nodes": sum(r.num_nodes_in_path for r in records) / total,
        "mean_elapsed": sum(r.elapsed for r in records) / total,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """一行式摘要（给人看的）。"""
    keys = [
        ("num_queries", "queries"),
        ("goal_hit_rate", "goal_hit"),
        ("optimal_path_rate", "optimal"),
        ("success_cost_ratio", "cost_ratio"),
        ("loop_rate", "loop"),
        ("broken_rate", "broken"),
        # 可微的软可达性代理指标，和 Hard Goal Hit 并排报告（第二轮修订）
        ("soft_goal_reachability", "soft_goal"),
        ("mean_elapsed", "sec/query"),
    ]
    parts = []
    for key, label in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if key == "num_queries":
            parts.append(f"{label}={int(value)}")
        elif key == "mean_elapsed":
            parts.append(f"{label}={value:.4f}")
        else:
            parts.append(f"{label}={value:.4f}")
    return " | ".join(parts)
