"""真实道路数据上的路径相似度 / 分布指标（实施方案第 12、17-F 节）。

真实司机路线**未必**是 Dijkstra 最短路（成都数据实测 GT/Dijkstra cost ratio
中位数 1.11、p99 2.77），所以旧的主指标 ``Optimal Path Rate`` 在这里不再是
"模型对不对"的判据。本模块提供真实数据真正需要的那一组：

条件路径规划主指标
    Goal Hit Rate               到终点比例（沿用旧定义）
    Normalized LCS              ``LCS(P_pred, P_gt) / |P_gt|``
    PathSimilarityScore         ``mean( 1(goal_hit) * nLCS )`` —— 选 best.pt 用
    nLCS(success)               只在成功到达的样本上算 nLCS
    Paired Edge Precision/Recall/F1   无向边集合重合度

合法性指标（旧口径，来自 :mod:`src.evaluation.metrics`）
    Loop Rate / Broken Rate

辅助指标（**不是**主指标）
    Dijkstra Cost / Pred Cost Ratio / GT Cost Ratio

DiffPath 风格全局分布指标
    KLEV / JSEV                 真实 vs 生成路径的 edge visit 分布

本模块**不 import torch**：它只做纯 Python / numpy 的路径比较，可以被数据准备
脚本、离线分析和单测直接使用。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

#: KLEV 平滑项（方案第 12.4 节）
DEFAULT_EPS = 1e-12

#: decision 数分桶边界（方案第 7.6 节的"项目特有诊断"）
DECISION_BUCKETS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7-9", 7, 9),
    (">=10", 10, None),
)


# ---------------------------------------------------------------------------
# 路径 -> 无向边集合
# ---------------------------------------------------------------------------
def canonical_edge(u: Any, v: Any) -> Tuple[Any, Any]:
    """无向边的规范化键 ``(min(u,v), max(u,v))``（方案第 12.1-C 节）。"""
    return (u, v) if u <= v else (v, u)


def canonical_edges(path: Sequence[Any]) -> List[Tuple[Any, Any]]:
    """路径经过的无向边序列（保留重复，供 visit frequency 使用）。"""
    return [canonical_edge(u, v) for u, v in zip(path[:-1], path[1:])]


def edge_set(path: Sequence[Any]) -> set:
    """路径的**去重**无向边集合（供 paired precision / recall 使用）。"""
    return set(canonical_edges(path))


# ---------------------------------------------------------------------------
# LCS
# ---------------------------------------------------------------------------
def lcs_length(a: Sequence[Any], b: Sequence[Any]) -> int:
    """最长公共子序列长度（标准 DP，两行滚动）。

    路径节点数是 O(10..100)，DP 完全够用，不需要位并行优化。
    """
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0] * (len(b) + 1)
        for index, y in enumerate(b, start=1):
            if x == y:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return int(previous[-1])


def normalized_lcs(pred: Sequence[Any], gt: Sequence[Any]) -> float:
    """``LCS(P_pred, P_gt) / |P_gt|``（方案第 12.1-B 节）。"""
    if not gt:
        return 0.0
    return lcs_length(pred, gt) / float(len(gt))


# ---------------------------------------------------------------------------
# paired edge P/R/F1
# ---------------------------------------------------------------------------
def paired_edge_prf(
    pred: Sequence[Any], gt: Sequence[Any]
) -> Tuple[float, float, float]:
    """无向边集合的 (precision, recall, f1)（方案第 12.1-C 节）。"""
    pred_edges = edge_set(pred)
    gt_edges = edge_set(gt)
    if not pred_edges and not gt_edges:
        return 0.0, 0.0, 0.0
    common = len(pred_edges & gt_edges)
    precision = common / len(pred_edges) if pred_edges else 0.0
    recall = common / len(gt_edges) if gt_edges else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


# ---------------------------------------------------------------------------
# 逐样本 paired 记录
# ---------------------------------------------------------------------------
@dataclass
class PathPairRecord:
    """一个 query 的真实数据指标（可累加）。"""

    goal_hit: bool
    normalized_lcs: float
    normalized_lcs_selection: float
    lcs: int
    precision: float
    recall: float
    f1: float
    gt_cost_ratio: float = float("nan")
    pred_cost_ratio: float = float("nan")
    #: ``C(P_pred) / C(P_GT)`` —— 模型路径相对**真实司机路径**的代价比。
    #: 这是唯一一个直接回答"模型比人绕多少"的量；``pred_cost_ratio`` 只回答了
    #: "模型比几何最短路绕多少"。真实数据上后者高不一定是坏事（人本来就不走最短路），
    #: 前者接近 1 才是真的学到了人的行为。
    pred_over_gt_cost_ratio: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_hit": bool(self.goal_hit),
            "normalized_lcs": float(self.normalized_lcs),
            "normalized_lcs_selection": float(self.normalized_lcs_selection),
            "lcs": int(self.lcs),
            "edge_precision": float(self.precision),
            "edge_recall": float(self.recall),
            "edge_f1": float(self.f1),
            "gt_cost_ratio": float(self.gt_cost_ratio),
            "pred_cost_ratio": float(self.pred_cost_ratio),
            "pred_over_gt_cost_ratio": float(self.pred_over_gt_cost_ratio),
        }


def pair_record(
    pred_path: Sequence[Any],
    gt_path: Sequence[Any],
    goal_hit: bool,
    gt_cost_ratio: float = float("nan"),
    pred_cost_ratio: float = float("nan"),
) -> PathPairRecord:
    """比较一条预测路径与真实 GT 路径。

    ``normalized_lcs_selection`` 在**没到终点时记 0** —— 这正是
    PathSimilarityScore 的定义（方案第 12.1-B 节），避免"没到终点但前半段很像"
    拿到虚高分。

    ``pred_over_gt_cost_ratio`` 由两个比值相除得到
    （``C(pred)/C*`` ÷ ``C(gt)/C*`` = ``C(pred)/C(gt)``），任一不可用则为 NaN。
    """
    value = normalized_lcs(pred_path, gt_path)
    precision, recall, f1 = paired_edge_prf(pred_path, gt_path)
    if (
        math.isfinite(pred_cost_ratio)
        and math.isfinite(gt_cost_ratio)
        and gt_cost_ratio > 0
    ):
        pred_over_gt = float(pred_cost_ratio) / float(gt_cost_ratio)
    else:
        pred_over_gt = float("nan")
    return PathPairRecord(
        goal_hit=bool(goal_hit),
        normalized_lcs=value,
        normalized_lcs_selection=value if goal_hit else 0.0,
        lcs=lcs_length(pred_path, gt_path),
        precision=precision,
        recall=recall,
        f1=f1,
        gt_cost_ratio=float(gt_cost_ratio),
        pred_cost_ratio=float(pred_cost_ratio),
        pred_over_gt_cost_ratio=pred_over_gt,
    )


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def aggregate_pair_records(records: Sequence[PathPairRecord]) -> Dict[str, float]:
    """把逐样本记录汇总成真实数据指标。"""
    total = len(records)
    if total == 0:
        return {"real_num_queries": 0}
    hits = [record for record in records if record.goal_hit]
    return {
        "real_num_queries": float(total),
        # 主指标
        "goal_hit_rate": len(hits) / total,
        "path_similarity_score": sum(r.normalized_lcs_selection for r in records) / total,
        "normalized_lcs": _mean([r.normalized_lcs for r in records]),
        "normalized_lcs_success": _mean([r.normalized_lcs for r in hits]),
        "lcs_success": _mean([float(r.lcs) for r in hits]),
        "edge_precision": _mean([r.precision for r in records]),
        "edge_recall": _mean([r.recall for r in records]),
        "edge_f1": _mean([r.f1 for r in records]),
        # 辅助指标（真实数据上 Optimal Path Rate 只作 secondary，方案第 12.3 节）
        "gt_cost_ratio": _mean([r.gt_cost_ratio for r in records]),
        "pred_cost_ratio": _mean([r.pred_cost_ratio for r in records]),
        "pred_over_gt_cost_ratio": _mean(
            [r.pred_over_gt_cost_ratio for r in records]
        ),
    }


def path_similarity_score(records: Sequence[PathPairRecord]) -> float:
    """单独暴露选模型用的那个数（方案第 12.1-B 节）。"""
    if not records:
        return float("nan")
    return sum(record.normalized_lcs_selection for record in records) / len(records)


def aggregate_pair_dicts(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """``PathPairRecord.to_dict()`` 列表的汇总（评测 JSON / 分桶用同一套口径）。"""
    total = len(records)
    if total == 0:
        return {"real_num_queries": 0}
    hits = [row for row in records if row.get("goal_hit")]
    selection = [float(row.get("normalized_lcs_selection", 0.0)) for row in records]
    return {
        "real_num_queries": float(total),
        "goal_hit_rate": len(hits) / total,
        "path_similarity_score": sum(selection) / total,
        "normalized_lcs": _mean([float(row.get("normalized_lcs", 0.0)) for row in records]),
        "normalized_lcs_success": _mean(
            [float(row.get("normalized_lcs", 0.0)) for row in hits]
        ),
        "lcs_success": _mean([float(row.get("lcs", 0.0)) for row in hits]),
        "edge_precision": _mean([float(row.get("edge_precision", 0.0)) for row in records]),
        "edge_recall": _mean([float(row.get("edge_recall", 0.0)) for row in records]),
        "edge_f1": _mean([float(row.get("edge_f1", 0.0)) for row in records]),
        "gt_cost_ratio": _mean([float(row.get("gt_cost_ratio", float("nan"))) for row in records]),
        "pred_cost_ratio": _mean(
            [float(row.get("pred_cost_ratio", float("nan"))) for row in records]
        ),
        "pred_over_gt_cost_ratio": _mean(
            [float(row.get("pred_over_gt_cost_ratio", float("nan"))) for row in records]
        ),
    }


# ---------------------------------------------------------------------------
# 全局 edge visit distribution / KLEV / JSEV
# ---------------------------------------------------------------------------
def edge_visit_distribution(paths: Iterable[Sequence[Any]]) -> Counter:
    """统计所有路径的**无向边 visit 次数**（重复经过会重复计数）。"""
    counter: Counter = Counter()
    for path in paths:
        counter.update(canonical_edges(path))
    return counter


def _smoothed_counts(
    counter: Mapping[Any, float], support: Sequence[Any], eps: float
) -> List[float]:
    """``q_e = (count_e + eps) / sum_j(count_j + eps)``（方案第 12.4 节）。"""
    total = sum(float(counter.get(key, 0.0)) for key in support) + eps * len(support)
    if total <= 0:
        return [1.0 / len(support)] * len(support) if support else []
    return [(float(counter.get(key, 0.0)) + eps) / total for key in support]


def _kl(p: Sequence[float], q: Sequence[float]) -> float:
    total = 0.0
    for p_value, q_value in zip(p, q):
        if p_value <= 0.0:
            continue
        if q_value <= 0.0:
            return float("inf")
        total += p_value * math.log(p_value / q_value)
    return float(total)


def klev(
    gt_counter: Mapping[Any, float],
    pred_counter: Mapping[Any, float],
    eps: float = DEFAULT_EPS,
) -> float:
    """``KLEV = D_KL(p_gt || q_pred)``，在 union support 上平滑（方案第 12.4 节）。"""
    support = sorted(set(gt_counter) | set(pred_counter), key=repr)
    if not support:
        return float("nan")
    gt_total = sum(float(v) for v in gt_counter.values())
    if gt_total <= 0:
        return float("nan")
    p = [float(gt_counter.get(key, 0.0)) / gt_total for key in support]
    q = _smoothed_counts(pred_counter, support, eps)
    return _kl(p, q)


def jsev(
    gt_counter: Mapping[Any, float],
    pred_counter: Mapping[Any, float],
    eps: float = DEFAULT_EPS,
) -> float:
    """``JSEV = JS(p_gt || q_pred)``，比 KLEV 稳定（方案第 12.4 节）。"""
    support = sorted(set(gt_counter) | set(pred_counter), key=repr)
    if not support:
        return float("nan")
    gt_total = sum(float(v) for v in gt_counter.values())
    if gt_total <= 0:
        return float("nan")
    p = [float(gt_counter.get(key, 0.0)) / gt_total for key in support]
    q = _smoothed_counts(pred_counter, support, eps)
    m = [(p_value + q_value) / 2.0 for p_value, q_value in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def distribution_metrics(
    gt_paths: Iterable[Sequence[Any]],
    pred_paths: Iterable[Sequence[Any]],
    eps: float = DEFAULT_EPS,
) -> Dict[str, float]:
    """dataset-level 的 KLEV / JSEV（**不是**单样本指标）。"""
    gt_counter = edge_visit_distribution(gt_paths)
    pred_counter = edge_visit_distribution(pred_paths)
    return {
        "klev": klev(gt_counter, pred_counter, eps=eps),
        "jsev": jsev(gt_counter, pred_counter, eps=eps),
        "gt_edge_support": float(len(gt_counter)),
        "pred_edge_support": float(len(pred_counter)),
        "shared_edge_support": float(len(set(gt_counter) & set(pred_counter))),
    }


# ---------------------------------------------------------------------------
# cost ratio
# ---------------------------------------------------------------------------
def graph_path_cost(graph: nx.Graph, path: Sequence[Any]) -> float:
    """路径 cost；出现非边返回 ``inf``。"""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            return float("inf")
        total += float(graph.edges[u, v].get("weight", 1.0))
    return total


def dijkstra_cost(graph: nx.Graph, start: Any, goal: Any) -> float:
    weight = "weight" if graph.graph.get("weighted", False) else None
    return float(nx.shortest_path_length(graph, start, goal, weight=weight))


def gt_cost_ratio(graph: nx.Graph, gt_path: Sequence[Any]) -> float:
    """``C(P_GT) / C*``（方案第 12.3-C 节）。

    这是真实数据上最重要的参考量：GT CostRatio = 1.15 说明真实司机本来就不走
    最短路，此时 Pred CostRatio = 1.17 反而是"模型贴近真实行为"的证据。
    """
    if len(gt_path) < 2:
        return float("nan")
    optimal = dijkstra_cost(graph, gt_path[0], gt_path[-1])
    if optimal <= 0:
        return float("nan")
    return graph_path_cost(graph, gt_path) / optimal


# ---------------------------------------------------------------------------
# 分桶（方案第 7.6 节）
# ---------------------------------------------------------------------------
def length_buckets(
    dataset: Sequence[Any],
    num_buckets: int = 3,
    key: str = "gt_length",
) -> Dict[str, List[int]]:
    """GDP 风格：按 ``gt_path`` 长度把样本**等量**切成 short / medium / long。

    ``key`` 可以是 ``"gt_length"``（跳数）或 ``"junction_len"``（meta 里的 junction
    数）；等量切分保证每个桶的样本数相同，桶间指标可比。
    """
    names = ["short", "medium", "long"]
    if num_buckets != 3:
        names = [f"bucket{index}" for index in range(num_buckets)]

    def value_of(sample) -> float:
        if key == "gt_length":
            return float(sample.gt_length)
        if key in getattr(sample, "meta", {}):
            return float(sample.meta[key])
        return float(sample.gt_length)

    order = sorted(range(len(dataset)), key=lambda index: (value_of(dataset[index]), index))
    groups: Dict[str, List[int]] = {name: [] for name in names}
    total = len(order)
    for bucket_index, name in enumerate(names):
        start = (bucket_index * total) // num_buckets
        end = ((bucket_index + 1) * total) // num_buckets
        groups[name] = order[start:end]
    return groups


def decision_buckets(dataset: Sequence[Any]) -> Dict[str, List[int]]:
    """项目特有诊断：按 ``num_decisions`` 分桶（长 decision chain 更容易积累错误）。"""
    groups: Dict[str, List[int]] = {name: [] for name, _, _ in DECISION_BUCKETS}
    for index, sample in enumerate(dataset):
        count = int(sample.num_decisions)
        for name, low, high in DECISION_BUCKETS:
            if count >= low and (high is None or count <= high):
                groups[name].append(index)
                break
        else:  # pragma: no cover - DECISION_BUCKETS 覆盖 >= 1 的全部取值
            groups[">=10"].append(index)
    return groups


def bucket_report(
    dataset: Sequence[Any],
    records: Sequence[Optional[Mapping[str, Any]]],
    groups: Mapping[str, Sequence[int]],
    summarizer=None,
) -> Dict[str, Dict[str, float]]:
    """对一组 bucket 逐个调 ``summarizer(records_subset)``。

    ``records`` 必须与 ``dataset`` **等长且同序**；没有真实 GT 的样本（例如
    shuffled OD 集）用 ``None`` 占位，分桶时自动跳过 —— 否则桶里会混进
    "占位 GT" 的相似度，指标就假了。
    """
    summarizer = summarizer or aggregate_pair_dicts
    report: Dict[str, Dict[str, float]] = {}
    for name, indices in groups.items():
        subset = [
            records[index]
            for index in indices
            if index < len(records) and records[index] is not None
        ]
        if not subset:
            continue
        report[name] = summarizer(subset)
    return report


__all__ = [
    "DEFAULT_EPS",
    "DECISION_BUCKETS",
    "PathPairRecord",
    "aggregate_pair_dicts",
    "aggregate_pair_records",
    "bucket_report",
    "canonical_edge",
    "canonical_edges",
    "decision_buckets",
    "dijkstra_cost",
    "distribution_metrics",
    "edge_set",
    "edge_visit_distribution",
    "graph_path_cost",
    "gt_cost_ratio",
    "jsev",
    "klev",
    "lcs_length",
    "length_buckets",
    "normalized_lcs",
    "pair_record",
    "paired_edge_prf",
    "path_similarity_score",
]
