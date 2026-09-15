"""真实数据路径指标单测（实施方案第 12、17-K 节第 8 条）。

这里的每个指标都配一个**能手算**的小样例 —— LCS / Edge F1 / KLEV / JSEV 这类
东西一旦实现错了不会报错，只会安静地给出好看的数字，必须有独立口径对照。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import networkx as nx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import real_path_metrics as rpm  # noqa: E402


# ---------------------------------------------------------------------------
# LCS
# ---------------------------------------------------------------------------
def test_lcs_hand_example():
    # [2, 4, 5] 是公共子序列，长度 3
    assert rpm.lcs_length([1, 2, 3, 4, 5], [2, 4, 5, 6]) == 3
    assert rpm.lcs_length([1, 2, 3], [1, 2, 3]) == 3
    assert rpm.lcs_length([1, 2, 3], [4, 5, 6]) == 0
    assert rpm.lcs_length([], [1, 2]) == 0


def test_normalized_lcs_is_divided_by_gt_length():
    pred = [1, 2, 3, 4, 5]
    gt = [2, 4, 5, 6]
    assert rpm.normalized_lcs(pred, gt) == pytest.approx(3 / 4)
    # 分母是 |P_gt|，不是 max(|pred|, |gt|)，也不是 LCS 的某种对称归一
    assert rpm.normalized_lcs(gt, pred) == pytest.approx(3 / 5)


def test_normalized_lcs_of_identical_path_is_one():
    path = [0, 1, 2, 3, 4]
    assert rpm.normalized_lcs(path, path) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# paired edge P/R/F1
# ---------------------------------------------------------------------------
def test_paired_edge_prf_hand_example():
    pred = [0, 1, 2, 3]        # edges {01, 12, 23}
    gt = [0, 1, 4, 3]          # edges {01, 14, 34}
    precision, recall, f1 = rpm.paired_edge_prf(pred, gt)
    assert precision == pytest.approx(1 / 3)
    assert recall == pytest.approx(1 / 3)
    assert f1 == pytest.approx(1 / 3)


def test_paired_edge_prf_is_direction_agnostic():
    """同一条无向边正着走和反着走必须算同一条边（方案第 12.1-C 节）。"""
    assert rpm.edge_set([0, 1, 2]) == rpm.edge_set([2, 1, 0])
    forward = rpm.paired_edge_prf([0, 1, 2, 3], [0, 1, 2, 3])
    backward = rpm.paired_edge_prf([3, 2, 1, 0], [0, 1, 2, 3])
    assert forward == pytest.approx(backward)
    assert forward == pytest.approx((1.0, 1.0, 1.0))


def test_paired_edge_prf_perfect_and_disjoint():
    assert rpm.paired_edge_prf([0, 1, 2], [0, 1, 2]) == pytest.approx((1.0, 1.0, 1.0))
    assert rpm.paired_edge_prf([0, 1, 2], [5, 6, 7]) == pytest.approx((0.0, 0.0, 0.0))


def test_canonical_edge_orders_endpoints():
    assert rpm.canonical_edge(5, 2) == (2, 5)
    assert rpm.canonical_edge(2, 5) == (2, 5)


# ---------------------------------------------------------------------------
# PathSimilarityScore
# ---------------------------------------------------------------------------
def test_path_similarity_score_zeroes_out_unreached_goals():
    """没到 goal 的 query 贡献 0 —— 这是该指标的定义（方案第 12.1-B 节）。"""
    reached = rpm.pair_record([0, 1, 2, 3], [0, 1, 2, 3], goal_hit=True)
    missed = rpm.pair_record([0, 1, 2, 3], [0, 1, 2, 3], goal_hit=False)

    assert reached.normalized_lcs == pytest.approx(1.0)
    assert reached.normalized_lcs_selection == pytest.approx(1.0)
    # 路径完全一样，但没到终点 -> selection 记 0
    assert missed.normalized_lcs == pytest.approx(1.0)
    assert missed.normalized_lcs_selection == pytest.approx(0.0)

    aggregate = rpm.aggregate_pair_records([reached, missed])
    assert aggregate["path_similarity_score"] == pytest.approx(0.5)
    # nLCS(success) 只在成功样本上算，所以仍然是 1.0
    assert aggregate["normalized_lcs_success"] == pytest.approx(1.0)
    assert aggregate["goal_hit_rate"] == pytest.approx(0.5)


def test_aggregate_pair_records_ignores_non_finite_cost_ratios():
    record = rpm.pair_record([0, 1], [0, 1], goal_hit=True, pred_cost_ratio=float("inf"))
    aggregate = rpm.aggregate_pair_records([record])
    assert math.isnan(aggregate["pred_cost_ratio"])


def test_path_similarity_score_prefers_the_model_that_actually_arrives():
    """高分但常常到不了终点的模型，**不应该**赢过稳定到达的模型。

    这是把 ``path_similarity_score`` 用作 best.pt 选择指标的全部理由：它天然包含
    GoalHit（未到达记 0），所以在 success-only 口径下"50% 到达但每次都完美"会拿到
    1.0 的假高分，而这里只有 0.5。
    """
    # 模型 A：50% 到达终点，到达时路径完全正确
    model_a = [
        rpm.pair_record([0, 1, 2, 3], [0, 1, 2, 3], goal_hit=True),
        rpm.pair_record([0, 1, 2, 3], [0, 1, 2, 3], goal_hit=False),
    ]
    # 模型 B：100% 到达终点，但路径只有一半像 GT
    model_b = [
        rpm.pair_record([0, 1, 9, 3], [0, 1, 2, 3], goal_hit=True),
        rpm.pair_record([0, 1, 9, 3], [0, 1, 2, 3], goal_hit=True),
    ]

    aggregate_a = rpm.aggregate_pair_records(model_a)
    aggregate_b = rpm.aggregate_pair_records(model_b)

    # success-only 口径下 A 反而"更完美" —— 这正是不能用它选模型的原因
    assert aggregate_a["normalized_lcs_success"] == pytest.approx(1.0)
    assert aggregate_b["normalized_lcs_success"] == pytest.approx(0.75)
    # 完整口径下 A 被 GoalHit 拉下来，B 才是更好的模型
    assert aggregate_a["path_similarity_score"] == pytest.approx(0.5)
    assert aggregate_b["path_similarity_score"] == pytest.approx(0.75)
    assert aggregate_b["path_similarity_score"] > aggregate_a["path_similarity_score"]


def test_pred_over_gt_cost_ratio_is_relative_to_the_driver_not_the_shortest_path():
    """``C(P_pred)/C(P_GT)``：模型比真实司机绕多少（辅助指标，方案第 7 节）。"""
    record = rpm.pair_record(
        [0, 1, 2], [0, 1, 2], goal_hit=True,
        gt_cost_ratio=1.2, pred_cost_ratio=1.5,
    )
    assert record.pred_over_gt_cost_ratio == pytest.approx(1.25)

    # 人本来就走 1.2 倍最短路：模型 1.5 倍其实只比人多绕 25%，而不是 50%
    assert record.pred_over_gt_cost_ratio < record.pred_cost_ratio

    missing = rpm.pair_record([0, 1, 2], [0, 1, 2], goal_hit=True)
    assert math.isnan(missing.pred_over_gt_cost_ratio)

    aggregate = rpm.aggregate_pair_records([record])
    assert aggregate["pred_over_gt_cost_ratio"] == pytest.approx(1.25)


def test_aggregate_pair_records_on_empty_input():
    assert rpm.aggregate_pair_records([]) == {"real_num_queries": 0}


# ---------------------------------------------------------------------------
# edge visit distribution / KLEV / JSEV
# ---------------------------------------------------------------------------
def test_edge_visit_distribution_counts_every_traversal():
    counter = rpm.edge_visit_distribution([[0, 1, 2], [2, 1, 0], [0, 1]])
    assert counter[rpm.canonical_edge(0, 1)] == 3
    assert counter[rpm.canonical_edge(1, 2)] == 2


def test_klev_hand_example():
    """p = (0.75, 0.25)，q = (0.5, 0.5) 的 KL(p || q)，手算 0.130812。"""
    gt = {("a", "b"): 3.0, ("b", "c"): 1.0}
    pred = {("a", "b"): 1.0, ("b", "c"): 1.0}
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert rpm.klev(gt, pred) == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(0.1308120, abs=1e-6)


def test_jsev_hand_example():
    """同一组分布，JS 散度手算 0.0338220。"""
    gt = {("a", "b"): 3.0, ("b", "c"): 1.0}
    pred = {("a", "b"): 1.0, ("b", "c"): 1.0}
    expected = 0.0338220
    assert rpm.jsev(gt, pred) == pytest.approx(expected, abs=1e-6)


def test_klev_and_jsev_are_zero_for_identical_distributions():
    gt = {("a", "b"): 3.0, ("b", "c"): 1.0}
    assert rpm.klev(gt, dict(gt)) == pytest.approx(0.0, abs=1e-9)
    assert rpm.jsev(gt, dict(gt)) == pytest.approx(0.0, abs=1e-9)


def test_jsev_is_symmetric_and_bounded():
    gt = {("a", "b"): 9.0, ("b", "c"): 1.0}
    pred = {("a", "b"): 1.0, ("b", "c"): 9.0}
    forward = rpm.jsev(gt, pred)
    backward = rpm.jsev(pred, gt)
    assert forward == pytest.approx(backward, abs=1e-9)
    assert 0.0 <= forward <= math.log(2.0) + 1e-9


def test_klev_penalises_missing_predicted_edges_via_smoothing():
    """pred 里一条边都没有时，q 在 union support 上退化成均匀分布。

    这里 support 恰好是 gt 的两条边，所以 q = (0.5, 0.5)，KLEV 与
    "pred 均匀" 的解析值完全相同（关键是它**有限**，不会 NaN / inf）。
    """
    gt = {("a", "b"): 1.0, ("b", "c"): 1.0}
    value = rpm.klev(gt, {})
    assert math.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-9)
    # gt 不均匀时才能看出平滑的效果
    skewed = {("a", "b"): 3.0, ("b", "c"): 1.0}
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert rpm.klev(skewed, {}) == pytest.approx(expected, abs=1e-9)


def test_distribution_metrics_payload():
    gt_paths = [[0, 1, 2], [0, 1, 2]]
    pred_paths = [[0, 1, 2], [0, 1, 2]]
    metrics = rpm.distribution_metrics(gt_paths, pred_paths)
    assert metrics["klev"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["jsev"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["shared_edge_support"] == metrics["gt_edge_support"] == 2


# ---------------------------------------------------------------------------
# cost ratio
# ---------------------------------------------------------------------------
def test_gt_cost_ratio_is_one_for_the_shortest_path():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    graph.add_edge(0, 2, weight=5.0)
    graph.graph["weighted"] = True
    # C* = 2（走 0-1-2）
    assert rpm.gt_cost_ratio(graph, [0, 1, 2]) == pytest.approx(1.0)
    # 直接走 0-2 cost 5 -> 5 / 2 = 2.5
    assert rpm.gt_cost_ratio(graph, [0, 2]) == pytest.approx(2.5)


def test_graph_path_cost_returns_inf_for_non_edges():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.0)
    assert rpm.graph_path_cost(graph, [0, 1]) == pytest.approx(2.0)
    assert rpm.graph_path_cost(graph, [1, 0]) == pytest.approx(2.0)
    assert rpm.graph_path_cost(graph, [0, 5]) == float("inf")


# ---------------------------------------------------------------------------
# 分桶
# ---------------------------------------------------------------------------
class _StubSample:
    def __init__(self, gt_length, num_decisions):
        self.gt_length = gt_length
        self.num_decisions = num_decisions
        self.meta = {"junction_len": gt_length + 1}


def test_length_buckets_are_equal_sized_and_ordered():
    dataset = [_StubSample(gt_length=index + 1, num_decisions=1) for index in range(6)]
    groups = rpm.length_buckets(dataset, 3)
    assert list(groups) == ["short", "medium", "long"]
    assert groups["short"] == [0, 1]
    assert groups["medium"] == [2, 3]
    assert groups["long"] == [4, 5]
    lengths = [
        dataset[index].gt_length for group in groups.values() for index in group
    ]
    assert lengths == sorted(lengths)


def test_decision_buckets_cover_long_chains():
    decisions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    dataset = [_StubSample(gt_length=20, num_decisions=value) for value in decisions]
    groups = rpm.decision_buckets(dataset)
    assert groups["1-3"] == [0, 1, 2]
    assert groups["4-6"] == [3, 4, 5]
    assert groups["7-9"] == [6, 7, 8]
    assert groups[">=10"] == [9, 10]


def test_bucket_report_skips_samples_without_real_gt():
    """shuffled OD 的占位 GT 用 None 占位，分桶时必须跳过，不能当成 0 分。"""
    dataset = [_StubSample(gt_length=index + 1, num_decisions=2) for index in range(4)]
    records = [
        rpm.pair_record([0, 1, 2], [0, 1, 2], goal_hit=True).to_dict(),
        None,
        rpm.pair_record([0, 1, 9], [0, 1, 2], goal_hit=False).to_dict(),
        None,
    ]
    groups = {"all": [0, 1, 2, 3]}
    report = rpm.bucket_report(dataset, records, groups)
    assert report["all"]["real_num_queries"] == 2
    assert report["all"]["path_similarity_score"] == pytest.approx(0.5)


def test_bucket_report_supports_full_length_three_way_split():
    dataset = [_StubSample(gt_length=index + 1, num_decisions=2) for index in range(6)]
    records = [
        rpm.pair_record([0, 1, 2], [0, 1, 2], goal_hit=True).to_dict()
        for _ in range(6)
    ]
    report = rpm.bucket_report(dataset, records, rpm.length_buckets(dataset, 3))
    assert set(report) == {"short", "medium", "long"}
    for row in report.values():
        assert row["real_num_queries"] == 2
        assert row["path_similarity_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# aggregate_pair_dicts 与 aggregate_pair_records 同口径
# ---------------------------------------------------------------------------
def test_dict_and_record_aggregation_agree():
    records = [
        rpm.pair_record([0, 1, 2], [0, 1, 2], goal_hit=True, pred_cost_ratio=1.5),
        rpm.pair_record([0, 2, 1], [0, 1, 2], goal_hit=False, pred_cost_ratio=2.0),
    ]
    from_records = rpm.aggregate_pair_records(records)
    from_dicts = rpm.aggregate_pair_dicts([record.to_dict() for record in records])
    assert set(from_records) == set(from_dicts)
    for key in from_records:
        expected, actual = from_records[key], from_dicts[key]
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(actual), key
        else:
            assert actual == pytest.approx(expected), key
