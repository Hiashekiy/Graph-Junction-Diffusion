"""按难度 / 结构模式 / source / 决策数给测试集分桶（评测报告与对比工具共用）。

只有一个数据源（`GraphQueryDataset` 的 ``sample.meta``）和一套口径，避免
``tools/breakdown_eval.py`` 与 ``tools/compare_runs.py`` 各写一份、慢慢漂移。

分桶维度（bucket 名 -> 该桶的样本下标）：

    ALL                             全部
    difficulty=<easy|medium|hard>
    mode=<branch_heavy|long_chain|loop_detour>
    source=<forced|decision>        单出口 source 的被迫段 / source 也是 decision
    gt_decisions=<3k>-<3k+2>        按 GT 决策条数每 3 条一档（3-5、6-8、9-11）

``summarize_records`` 与 ``src/evaluation/metrics.py`` 的整体指标口径一致：
cost ratio 只在成功（goal_hit）的 query 上求平均，失败 query 的 ``inf`` 不参与。
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Sequence

METRICS = (
    "goal_hit_rate",
    "optimal_path_rate",
    "success_cost_ratio",
    "loop_rate",
    "broken_rate",
)


def bucket_indices(dataset) -> Dict[str, List[int]]:
    """返回 bucket 名 -> 记录下标列表。"""
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, sample in enumerate(dataset):
        groups["ALL"].append(index)
        groups[f"difficulty={sample.meta.get('difficulty', 'n/a')}"].append(index)
        groups[f"mode={sample.meta.get('mode', 'n/a')}"].append(index)
        groups[
            "source="
            + ("forced" if sample.segments.source_forced_edge_ids else "decision")
        ].append(index)
        low = sample.num_decisions // 3 * 3
        groups[f"gt_decisions={low}-{low + 2}"].append(index)
    return groups


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """把 ``SampleRecord.to_dict()`` 列表汇总成指标（与整体评测同口径）。"""
    total = len(records)
    if total == 0:
        return {"num_queries": 0, **{key: float("nan") for key in METRICS}}
    hits = [r for r in records if r["goal_hit"]]
    ratios = [
        r["cost_ratio"] for r in hits if r["cost_ratio"] not in (None, float("inf"))
    ]
    return {
        "num_queries": total,
        "goal_hit_rate": len(hits) / total,
        "optimal_path_rate": sum(1 for r in records if r["optimal"]) / total,
        "success_cost_ratio": statistics.fmean(ratios) if ratios else float("nan"),
        "loop_rate": sum(1 for r in records if r["status"] == "loop") / total,
        "broken_rate": sum(1 for r in records if r["status"] == "broken") / total,
    }


def breakdown(records: Sequence[Dict[str, Any]], dataset) -> Dict[str, Dict[str, float]]:
    """一次算出所有 bucket 的指标；records 顺序必须与 dataset 一致。"""
    if len(records) != len(dataset):
        raise ValueError(
            f"record count {len(records)} != dataset size {len(dataset)}; "
            "the eval json and the dataset must come from the same split"
        )
    result: Dict[str, Dict[str, float]] = {}
    for name, index in bucket_indices(dataset).items():
        result[name] = summarize_records([records[i] for i in index])
    return result


__all__ = ["METRICS", "bucket_indices", "breakdown", "summarize_records"]
