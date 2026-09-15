"""Path NLL + Sampled NULL 目标函数的单元测试。

背景：真实 DiDi corridor 里 93.6% 的 decision 标签是 NULL。旧 CE 对**每个 decision
等权平均**，于是"全押 NULL"就是 CE 的平凡最优解 —— 实测 5 个 epoch 就卡在
``x0_acc = 0.933 ≈ NULL 占比`` 且 loss 不再下降。

新目标把 active 当主监督、NULL 只做下采样辅助：

    L = L_path + lambda_null * L_null
    K_b = min(N_N^{(b)}, ceil(ratio * N_A^{(b)}), max_per_sample)

这里钉住四件事，任何一件错了都会让实验结论不可信：

1. 采样条数严格是 ``K = min(N_N, ceil(2 N_A), 64)``；
2. **样本内先求 mean**（长轨迹不能因为 active 多就拿到更高权重）；
3. 采样是随机的、且每个 batch 重新采（不是固定一批 NULL）；
4. ``loss_type="ce"`` 时行为与改动前逐位一致（向后兼容）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.losses import (  # noqa: E402
    PATH_NLL_SAMPLED_NULL,
    LossWeights,
    path_nll_sampled_null_loss,
    sample_null_decisions,
)


def _toy_topology(active_counts, null_counts):
    """构造一个最小 batch：每张图先排 active 再排 NULL。"""
    graph_ids, is_null = [], []
    for graph_index, (num_active, num_null) in enumerate(zip(active_counts, null_counts)):
        graph_ids.extend([graph_index] * (num_active + num_null))
        is_null.extend([False] * num_active + [True] * num_null)
    return (
        torch.tensor(graph_ids, dtype=torch.long),
        torch.tensor(is_null, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# 1. 采样条数 K = min(N_N, ceil(2 N_A), 64)
# ---------------------------------------------------------------------------
def test_null_sampling_quota_is_adaptive_not_fixed():
    """方案里的三个例子：8/120 -> 16，20/280 -> 40，35/500 -> 64（被上限截断）。"""
    graph_ids, is_null = _toy_topology([8, 20, 35], [120, 280, 500])
    out = sample_null_decisions(
        is_null, graph_ids, num_graphs=3, ratio=2.0, max_per_sample=64,
        generator=torch.Generator().manual_seed(0),
    )
    assert [int(v) for v in out["num_active"]] == [8, 20, 35]
    assert [int(v) for v in out["num_null"]] == [120, 280, 500]
    assert [int(v) for v in out["num_selected"]] == [16, 40, 64]


def test_null_sampling_is_clipped_by_available_nulls():
    """N_N 不够时只能采到 N_N 条。"""
    graph_ids, is_null = _toy_topology([30], [5])
    out = sample_null_decisions(
        is_null, graph_ids, num_graphs=1, ratio=2.0, max_per_sample=64
    )
    assert int(out["num_selected"][0]) == 5
    assert int(out["selected"].sum()) == 5


def test_null_sampling_only_selects_null_decisions():
    graph_ids, is_null = _toy_topology([4], [100])
    out = sample_null_decisions(
        is_null, graph_ids, num_graphs=1, ratio=2.0, max_per_sample=64,
        generator=torch.Generator().manual_seed(1),
    )
    selected = out["selected"]
    assert int(selected.sum()) == 8
    assert bool((selected & ~is_null).sum() == 0), "只能选中 NULL decision"


def test_disabled_sampling_falls_back_to_full_null():
    """enabled=False 是"全量监督 NULL"的消融对照（就是会 collapse 的口径）。"""
    graph_ids, is_null = _toy_topology([8], [120])
    out = sample_null_decisions(
        is_null, graph_ids, num_graphs=1, ratio=2.0, max_per_sample=64, enabled=False
    )
    assert int(out["num_selected"][0]) == 120
    assert int(out["selected"].sum()) == 120


def test_resampling_changes_the_selected_set():
    """每个 batch 重新采：同一份拓扑、不同随机种子应给出不同的 NULL 集合。"""
    graph_ids, is_null = _toy_topology([10], [200])
    first = sample_null_decisions(
        is_null, graph_ids, 1, generator=torch.Generator().manual_seed(0)
    )["selected"]
    second = sample_null_decisions(
        is_null, graph_ids, 1, generator=torch.Generator().manual_seed(1)
    )["selected"]
    assert int(first.sum()) == int(second.sum()) == 20
    assert not torch.equal(first, second), "两次采样不应该完全相同"


def test_sampling_is_deterministic_for_the_same_seed():
    graph_ids, is_null = _toy_topology([10], [200])
    a = sample_null_decisions(
        is_null, graph_ids, 1, generator=torch.Generator().manual_seed(7)
    )["selected"]
    b = sample_null_decisions(
        is_null, graph_ids, 1, generator=torch.Generator().manual_seed(7)
    )["selected"]
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 2/3. 损失：样本内先 mean；K 不影响 L_null 的权重
# ---------------------------------------------------------------------------
def _toy_batch(graph_ids, is_null):
    """构造 candidate / target：每个 decision 一组 [NULL, branch]（source 除外）。"""
    num_decisions = int(graph_ids.numel())
    candidate_owner, candidate_is_null, target = [], [], []
    for decision_index in range(num_decisions):
        if bool(is_null[decision_index]):
            owner = len(candidate_owner)
            candidate_owner.append(decision_index)
            candidate_is_null.append(True)
            candidate_owner.append(decision_index)
            candidate_is_null.append(False)
            target.append(owner)          # 选 NULL
        else:
            owner = len(candidate_owner)
            candidate_owner.append(decision_index)
            candidate_is_null.append(False)
            candidate_owner.append(decision_index)
            candidate_is_null.append(False)
            target.append(owner + 1)      # 选 branch
    return (
        torch.tensor(candidate_owner, dtype=torch.long),
        torch.tensor(candidate_is_null, dtype=torch.bool),
        torch.tensor(target, dtype=torch.long),
    )


def test_sample_level_mean_gives_every_trajectory_equal_weight():
    """一条 8-active / 一条 30-active：两条轨迹必须等权。

    把所有 decision 混在一起求 mean 的话，长轨迹的权重会是短轨迹的 3.75 倍。
    """
    graph_ids, is_null = _toy_topology([8, 30], [0, 0])
    owner, cand_null, target = _toy_batch(graph_ids, is_null)
    num_candidates = int(owner.numel())
    # 让短轨迹的 branch 概率很低（loss 大）、长轨迹的概率很高（loss 小）
    log_prob = torch.full((num_candidates,), -5.0)
    log_prob[target[:8]] = -2.0        # 图0：nll = 2.0
    log_prob[target[8:]] = -0.1        # 图1：nll = 0.1
    weights = LossWeights(loss_type=PATH_NLL_SAMPLED_NULL, null_loss_weight=0.0)
    sampling = {
        "selected": torch.zeros(int(graph_ids.numel()), dtype=torch.bool),
        "num_active": torch.tensor([8.0, 30.0]),
        "num_null": torch.zeros(2),
        "num_selected": torch.zeros(2),
    }
    total, path, null, metrics = path_nll_sampled_null_loss(
        log_prob, log_prob.exp(), target, owner, cand_null,
        graph_ids, int(graph_ids.numel()), 2, weights, sampling,
    )
    # 每条轨迹先 mean，再对 2 条取 mean -> (2.0 + 0.1) / 2
    assert float(path) == pytest.approx((2.0 + 0.1) / 2, rel=1e-5)
    # 反例：混在一起 mean 会得到 (8*2 + 30*0.1)/38 = 0.5
    mixed = (8 * 2.0 + 30 * 0.1) / 38
    assert abs(float(path) - mixed) > 0.1
    assert float(metrics["mean_num_active"]) == pytest.approx(19.0)


def test_null_loss_weight_scales_only_the_null_term():
    graph_ids, is_null = _toy_topology([4, 4], [8, 8])
    owner, cand_null, target = _toy_batch(graph_ids, is_null)
    num_candidates = int(owner.numel())
    log_prob = torch.full((num_candidates,), -1.0)
    sampling = sample_null_decisions(
        is_null, graph_ids, 2, ratio=2.0, max_per_sample=64,
        generator=torch.Generator().manual_seed(3),
    )
    base = LossWeights(loss_type=PATH_NLL_SAMPLED_NULL, null_loss_weight=0.0)
    total0, path, null, _ = path_nll_sampled_null_loss(
        log_prob, log_prob.exp(), target, owner, cand_null,
        graph_ids, int(graph_ids.numel()), 2, base, sampling,
    )
    scaled = LossWeights(loss_type=PATH_NLL_SAMPLED_NULL, null_loss_weight=0.2)
    total1, path1, null1, _ = path_nll_sampled_null_loss(
        log_prob, log_prob.exp(), target, owner, cand_null,
        graph_ids, int(graph_ids.numel()), 2, scaled, sampling,
    )
    assert float(path) == pytest.approx(float(path1))
    assert float(total1) == pytest.approx(float(path1) + 0.2 * float(null1))
    assert float(total0) == pytest.approx(float(path))


def test_metrics_expose_active_and_null_behaviour():
    graph_ids, is_null = _toy_topology([4], [40])
    owner, cand_null, target = _toy_batch(graph_ids, is_null)
    num_candidates = int(owner.numel())
    # 全部猜对：active 选 branch、NULL 选 NULL
    log_prob = torch.full((num_candidates,), -4.0)
    log_prob[target] = 0.0
    sampling = sample_null_decisions(
        is_null, graph_ids, 1, generator=torch.Generator().manual_seed(0)
    )
    weights = LossWeights(loss_type=PATH_NLL_SAMPLED_NULL, null_loss_weight=0.2)
    _, _, _, metrics = path_nll_sampled_null_loss(
        log_prob, log_prob.exp(), target, owner, cand_null,
        graph_ids, int(graph_ids.numel()), 1, weights, sampling,
    )
    assert float(metrics["active_branch_acc"]) == pytest.approx(1.0)
    assert float(metrics["sampled_null_acc"]) == pytest.approx(1.0)
    assert float(metrics["mean_gt_branch_prob"]) == pytest.approx(1.0)
    assert float(metrics["pred_active_rate"]) == pytest.approx(4 / 44, abs=1e-5)
    assert float(metrics["mean_num_sampled_null"]) == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# 4. 向后兼容
# ---------------------------------------------------------------------------
def test_default_loss_type_is_still_plain_ce():
    weights = LossWeights()
    assert weights.loss_type == "ce"
    assert weights.is_sampled_null is False


def test_invalid_loss_type_is_rejected():
    with pytest.raises(ValueError):
        LossWeights(loss_type="nope").validate()
    with pytest.raises(ValueError):
        LossWeights(loss_type=PATH_NLL_SAMPLED_NULL, null_sampling_ratio=0).validate()
    with pytest.raises(ValueError):
        LossWeights(loss_type=PATH_NLL_SAMPLED_NULL, null_sampling_max=0).validate()
