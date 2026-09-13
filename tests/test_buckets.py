"""``src/evaluation/buckets.py`` 的分桶与汇总口径测试。

这是评测报告（breakdown_eval / compare_runs）共用的口径，错在这里会让所有
"哪个难度更难"的结论一起错，所以单独测：

* bucket 划分覆盖所有样本、且各维度互不重叠地拼回全量；
* cost ratio 只在成功 query 上平均（失败 query 的 inf 不能污染均值）；
* loop / broken / optimal 的定义与 ``SampleRecord`` 字段一致。
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.buckets import METRICS, bucket_indices, breakdown, summarize_records


def _record(status="goal", goal_hit=True, optimal=False, cost_ratio=1.0):
    return {
        "status": status,
        "goal_hit": goal_hit,
        "optimal": optimal,
        "cost_ratio": cost_ratio,
        "pred_cost": cost_ratio,
        "optimal_cost": 1.0,
        "num_nodes_in_path": 10,
        "num_branches": 3,
        "gt_length": 9,
        "reason": "",
        "elapsed": 0.01,
    }


def test_summarize_counts_every_metric():
    records = [
        _record(optimal=True, cost_ratio=1.0),
        _record(optimal=False, cost_ratio=2.0),
        _record(status="loop", goal_hit=False, cost_ratio=float("inf")),
        _record(status="broken", goal_hit=False, cost_ratio=float("inf")),
    ]
    stats = summarize_records(records)
    assert stats["num_queries"] == 4
    assert stats["goal_hit_rate"] == pytest.approx(0.5)
    assert stats["optimal_path_rate"] == pytest.approx(0.25)
    # 失败 query 的 inf 不参与均值：只有 1.0 和 2.0 两条
    assert stats["success_cost_ratio"] == pytest.approx(1.5)
    assert stats["loop_rate"] == pytest.approx(0.25)
    assert stats["broken_rate"] == pytest.approx(0.25)


def test_summarize_is_nan_without_successes():
    stats = summarize_records([_record(status="broken", goal_hit=False, cost_ratio=float("inf"))])
    assert math.isnan(stats["success_cost_ratio"])
    assert stats["goal_hit_rate"] == 0.0


def test_summarize_empty_is_nan_not_zero():
    stats = summarize_records([])
    assert stats["num_queries"] == 0
    for key in METRICS:
        assert math.isnan(stats[key]), key


def test_bucket_indices_partition_the_dataset(manual_batch):
    dataset = _tiny_dataset()
    groups = bucket_indices(dataset)

    assert set(groups["ALL"]) == set(range(len(dataset)))
    for dimension in ("difficulty", "mode", "source", "gt_decisions"):
        keys = [key for key in groups if key.startswith(dimension + "=")]
        assert keys, f"no bucket for {dimension}"
        union = sorted(index for key in keys for index in groups[key])
        assert union == sorted(groups["ALL"]), dimension


def test_breakdown_matches_per_bucket_manual_computation():
    dataset = _tiny_dataset()
    records = [_record(cost_ratio=1.0 + index) for index in range(len(dataset))]
    result = breakdown(records, dataset)
    for name, stats in result.items():
        assert stats["num_queries"] > 0
        assert stats["goal_hit_rate"] == pytest.approx(1.0)
    assert result["ALL"]["num_queries"] == len(dataset)


def test_breakdown_rejects_length_mismatch():
    dataset = _tiny_dataset()
    with pytest.raises(ValueError):
        breakdown([_record()], dataset)


# ---------------------------------------------------------------------------
def _tiny_dataset():
    """用 conftest 的手工图造 3 条 query，meta 里塞不同难度 / 模式。

    ``source=`` 这一维由 ``sample.segments`` 自己决定，所以这里不强行指定 ——
    测试只断言"每个维度的桶能恰好拼回全量"。
    """
    from conftest import make_manual_graph

    from src.data.dataset import GraphQueryDataset
    from src.data.dataset_builder import build_sample

    graph, start, goal = make_manual_graph()
    samples = [
        build_sample(
            graph,
            start,
            goal,
            meta={"difficulty": difficulty, "mode": mode},
            graph_id=index,
        )
        for index, (difficulty, mode) in enumerate(
            [
                ("easy", "branch_heavy"),
                ("medium", "long_chain"),
                ("hard", "loop_detour"),
            ]
        )
    ]
    return GraphQueryDataset(samples, name="tiny_buckets")
