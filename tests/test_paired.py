"""``src/evaluation/paired.py`` 的测试（配对比较 + McNemar 精确检验）。

这些数字会直接进结论（"多轮是不是真的更好"），所以统计口径必须自己先测对：

* 无不一致对 -> p = 1（无信息），不是 0；
* 不一致对全在一边 -> p 很小；
* p 值对称（交换 A/B 不变）；
* 率、delta、four-fold 计数与手算一致。
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.paired import exact_mcnemar_p, flag, paired_counts


def _record(goal_hit=True, optimal=False, status="goal"):
    return {
        "status": status,
        "goal_hit": goal_hit,
        "optimal": optimal,
        "cost_ratio": 1.0 if goal_hit else float("inf"),
    }


def test_mcnemar_no_discordant_pairs_is_uninformative():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(5, 5) == 1.0


def test_mcnemar_is_symmetric():
    for only_a, only_b in ((9, 1), (3, 8), (0, 7), (12, 4)):
        assert exact_mcnemar_p(only_a, only_b) == exact_mcnemar_p(only_b, only_a)


def test_mcnemar_all_discordant_on_one_side():
    # 10 个不一致对全在 B：p = 2 * P(X <= 0) = 2 / 2^10
    assert exact_mcnemar_p(0, 10) == pytest.approx(2 / 1024)
    assert exact_mcnemar_p(0, 20) < 1e-5


def test_mcnemar_matches_hand_computed_binomial_tail():
    # n = 7, min = 2 -> 2 * (C(7,0)+C(7,1)+C(7,2)) / 2^7
    expected = 2 * (1 + 7 + 21) / 128
    assert exact_mcnemar_p(5, 2) == pytest.approx(expected)


def test_p_value_never_exceeds_one():
    for only_a in range(0, 6):
        for only_b in range(0, 6):
            assert 0.0 < exact_mcnemar_p(only_a, only_b) <= 1.0


def test_flag_metrics():
    record = _record(goal_hit=False, optimal=False, status="broken")
    assert flag(record, "goal_hit") is False
    assert flag(record, "broken") is True
    assert flag(record, "loop") is False
    assert flag(_record(optimal=True), "optimal") is True
    with pytest.raises(ValueError):
        flag(record, "nope")


def test_paired_counts_four_fold_table():
    # 4 条 query：both / only_a / only_b / neither 各一条
    records_a = [
        _record(goal_hit=True),
        _record(goal_hit=True),
        _record(goal_hit=False, status="broken"),
        _record(goal_hit=False, status="broken"),
    ]
    records_b = [
        _record(goal_hit=True),
        _record(goal_hit=False, status="broken"),
        _record(goal_hit=True),
        _record(goal_hit=False, status="broken"),
    ]
    stats = paired_counts(records_a, records_b, "goal_hit", [0, 1, 2, 3])
    assert (stats["both"], stats["only_a"], stats["only_b"], stats["neither"]) == (1, 1, 1, 1)
    assert stats["rate_a"] == pytest.approx(0.5)
    assert stats["rate_b"] == pytest.approx(0.5)
    assert stats["delta"] == pytest.approx(0.0)
    assert stats["p_value"] == 1.0


def test_paired_counts_subset_indices():
    records_a = [_record(goal_hit=True), _record(goal_hit=True)]
    records_b = [_record(goal_hit=False, status="broken"), _record(goal_hit=True)]
    only_first = paired_counts(records_a, records_b, "goal_hit", [0])
    assert only_first["n"] == 1 and only_first["only_a"] == 1
    both = paired_counts(records_a, records_b, "goal_hit", [1])
    assert both["n"] == 1 and both["both"] == 1


def test_paired_counts_empty_indices_is_nan_not_crash():
    stats = paired_counts([], [], "goal_hit", [])
    assert stats["n"] == 0
    assert math.isnan(stats["rate_a"]) and math.isnan(stats["delta"])
