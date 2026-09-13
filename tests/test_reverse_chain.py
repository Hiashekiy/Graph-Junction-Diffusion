"""Reverse chain / sampler / decoder / training loop 集成测试
（实施指南第 17-21、25 节 + Milestone E/F 验收）。

验收点：

    (H_t, z_t) -> (H_{t-1}, z_{t-1}) 完整跑 T 步不报错
    forward trajectory 用 alpha_t（单步）而不是 alpha_bar_t
    posterior 在 candidate 组内是一个合法分布
    decoder 能正确沿 branch segment 前进
    teacher-forced recurrent 训练可以 backward（梯度穿过整条链）
"""

from __future__ import annotations

import torch
import pytest

from src.diffusion.categorical import CategoricalDiffusion
from src.diffusion.sampler import reverse_step, sample_prior, sample_reverse_chain
from src.diffusion.schedule import NoiseSchedule
from src.evaluation.metrics import aggregate, evaluate_sample, path_cost
from src.evaluation.path_decoder import decode_flat
from src.models.denoiser import GraphFlowDenoiser
from src.training.losses import LossWeights, recurrent_reverse_loss


def make_model(d_model: int = 16) -> GraphFlowDenoiser:
    return GraphFlowDenoiser(d_model=d_model, ffn_hidden=32)


def make_diffusion(T: int = 5) -> CategoricalDiffusion:
    return CategoricalDiffusion(
        NoiseSchedule(T=T, schedule="linear", beta_start=0.05, beta_end=0.5)
    )


# ---------------------------------------------------------------------------
# forward trajectory
# ---------------------------------------------------------------------------
def test_forward_trajectory_has_T_plus_one_states(tiny_batches):
    _, batch = tiny_batches
    diffusion = make_diffusion(T=5)
    trajectory = diffusion.sample_forward_trajectory(
        batch.target_candidate, batch.candidate_owner, batch.num_decisions
    )
    assert trajectory.shape == (6, batch.num_decisions)
    assert torch.equal(trajectory[0], batch.target_candidate)


def test_forward_trajectory_uses_single_step_alpha(tiny_batches):
    """单步 keep 频率必须匹配 alpha_t + (1-alpha_t)/C_i（用的是单步 alpha_t）。

    用 **alpha_bar_t** 会得到明显更低的频率，所以这个测试能区分两者。
    注意：batch 里每个 decision 独立保留，所以只能统计**逐 decision** 的 keep 频率，
    不能用 ``(z_next == z_prev).all()``（32 个 decision 全等的概率接近 0）。
    """
    _, batch = tiny_batches
    diffusion = make_diffusion(T=5)
    torch.manual_seed(0)

    t = 3
    alpha_t = float(diffusion.schedule.alpha_at(t))
    alpha_bar = float(diffusion.schedule.alpha_bar_at(t))

    z_prev = batch.target_candidate.clone()
    keep = torch.zeros(batch.num_decisions)
    trials = 400
    for _ in range(trials):
        z_next = diffusion.sample_forward_step(
            z_prev, batch.candidate_owner, batch.num_decisions, t
        )
        keep += (z_next == z_prev).float()
    rate = float(keep.mean() / trials)

    # 经验频率应当落在单步 alpha_t 附近，而明显高于 alpha_bar_t
    assert rate > alpha_t - 0.05, (rate, alpha_t, alpha_bar)
    assert abs(rate - alpha_bar) > 0.1, (rate, alpha_t, alpha_bar)


def test_forward_step_stays_inside_the_candidate_group(tiny_batches):
    _, batch = tiny_batches
    diffusion = make_diffusion(T=5)
    z = diffusion.sample_forward_trajectory(
        batch.target_candidate, batch.candidate_owner, batch.num_decisions
    )
    for row in z:
        for decision_index in range(batch.num_decisions):
            assert int(batch.candidate_owner[row[decision_index]]) == decision_index


# ---------------------------------------------------------------------------
# reverse step
# ---------------------------------------------------------------------------
def test_reverse_step_shapes_and_valid_posterior(tiny_batches):
    _, batch = tiny_batches
    diffusion = make_diffusion(T=5)
    model = make_model()
    H_t = model.init_nodes(batch)
    z_t = sample_prior(diffusion, batch.candidate_owner, batch.num_decisions)

    step = reverse_step(diffusion, model, batch, H_t, z_t, 3, stochastic=False)
    assert step["z_prev"].shape == z_t.shape
    assert step["candidate_prob"].shape == (batch.num_candidates,)
    assert torch.isfinite(step["candidate_prob"]).all()
    assert (step["candidate_prob"] >= 0).all()
    assert not torch.allclose(step["H_next"], H_t)

    posterior = step["reverse_prob"]
    mask = step["mask"]
    assert torch.isfinite(posterior).all()
    sums = (posterior * mask).sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_full_chain_runs_for_T_steps(tiny_batches):
    _, batch = tiny_batches
    diffusion = make_diffusion(T=5)
    model = make_model()
    chain = sample_reverse_chain(
        diffusion, model, batch, stochastic=True, max_steps=5, record=True
    )
    assert chain["z0"].shape == (batch.num_decisions,)
    assert len(chain["trace"].H_path) == 6


def test_chain_matches_schedule_length(tiny_batches):
    _, batch = tiny_batches
    diffusion = make_diffusion(T=5)
    model = make_model()
    with pytest.raises(ValueError):
        sample_reverse_chain(diffusion, model, batch, max_steps=99)


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------
def test_decoder_follows_gt_branches_to_the_goal(manual_sample, manual_batch):
    result = decode_flat(manual_sample, manual_batch.target_candidate)
    assert result.status == "goal"
    assert result.path == manual_sample.gt_path[: len(result.path)]
    assert result.path[0] == manual_sample.start
    assert result.path[-1] == manual_sample.goal


def test_decoder_reports_broken_on_null(manual_sample, manual_batch):
    candidates = manual_sample.field.candidates
    j1 = manual_sample.segments.decision_nodes.index(2)
    null_index = next(
        index
        for index, (owner, is_null) in enumerate(
            zip(candidates.candidate_owner, candidates.candidate_is_null)
        )
        if owner == j1 and is_null
    )
    z = manual_batch.target_candidate.clone()
    z[j1] = null_index
    result = decode_flat(manual_sample, z)
    assert result.status == "broken"
    assert "NULL" in result.reason


def test_decoder_reports_loop(manual_sample, manual_batch):
    """J1 选择通往 start 的那条 branch，就会走回已经访问过的节点。"""
    candidates = manual_sample.field.candidates
    j1 = manual_sample.segments.decision_nodes.index(2)
    back_index = next(
        index
        for index, branch in enumerate(candidates.candidate_branch)
        if branch is not None and branch.owner == 2 and branch.end == 0
    )
    z = manual_batch.target_candidate.clone()
    z[j1] = back_index
    result = decode_flat(manual_sample, z)
    assert result.status == "loop"


def test_decoder_handles_batch_level_flat_space(tiny_batches):
    """批量解码必须同时用 decision_offset 与 candidate_offset。"""
    from src.evaluation.path_decoder import (
        candidate_offsets,
        decode_batch,
        decode_flat,
        decision_offsets,
    )

    samples, batch = tiny_batches
    decision_starts = decision_offsets(samples)
    candidate_starts = candidate_offsets(samples)
    z0 = batch.target_candidate  # flat decision 空间里的**全局** candidate 索引

    batched = decode_batch(samples, z0, decision_starts, candidate_starts)
    for index, sample in enumerate(samples):
        local_targets = torch.tensor(
            sample.field.candidates.target_candidate, dtype=torch.long
        )
        standalone = decode_flat(sample, local_targets)
        assert batched[index].status == standalone.status
        assert batched[index].path == standalone.path

    # 同一份 z0 在样本内空间里也能解（前提是索引其实是局部的）
    assert decision_starts[1] > 0 and candidate_starts[1] > 0


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_metrics_on_a_perfect_decode(manual_sample, manual_batch):
    result = decode_flat(manual_sample, manual_batch.target_candidate)
    record = evaluate_sample(manual_sample, result)
    assert record.goal_hit and record.optimal
    assert abs(record.cost_ratio - 1.0) < 1e-6

    metrics = aggregate([record])
    assert metrics["goal_hit_rate"] == 1.0
    assert metrics["optimal_path_rate"] == 1.0
    assert metrics["loop_rate"] == 0.0
    assert metrics["broken_rate"] == 0.0


def test_metrics_on_a_broken_decode(manual_sample, manual_batch):
    candidates = manual_sample.field.candidates
    j1 = manual_sample.segments.decision_nodes.index(2)
    null_index = next(
        index
        for index, (owner, is_null) in enumerate(
            zip(candidates.candidate_owner, candidates.candidate_is_null)
        )
        if owner == j1 and is_null
    )
    z = manual_batch.target_candidate.clone()
    z[j1] = null_index
    record = evaluate_sample(manual_sample, decode_flat(manual_sample, z))
    metrics = aggregate([record])
    assert metrics["goal_hit_rate"] == 0.0
    assert metrics["broken_rate"] == 1.0


def test_path_cost_on_manual_graph(manual_sample):
    assert path_cost(manual_sample.graph, manual_sample.gt_path) == manual_sample.gt_length


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def test_recurrent_loss_backward_reaches_every_timestep(manual_batch):
    model = make_model()
    diffusion = make_diffusion(T=4)
    out = recurrent_reverse_loss(
        model,
        diffusion,
        manual_batch,
        weights=LossWeights(),
        max_steps=4,
        record=True,
    )
    assert torch.isfinite(out.loss)
    assert len(out.per_step_loss) == 4
    out.loss.backward()
    assert model.graph_flow.q_proj.weight.grad is not None
    assert model.branch_scorer.branch_mlp[0].weight.grad is not None
    assert model.edge_state_encoder.embedding.weight.grad is not None


def test_accuracy_is_computed_per_decision_group(manual_batch):
    """accuracy 必须在每个 decision 自己的候选组内取 argmax。

    （对一维 flat 候选表用 ``argmax(dim=-1)`` 会得到 0-dim 张量，是一个曾经真实
    发生过的 bug：准确率会恒定错成"目标恰好是 0 号候选"的比例。）
    """
    from src.training.losses import accuracy, grouped_argmax

    logits = torch.zeros(manual_batch.num_candidates)
    logits[manual_batch.target_candidate] = 5.0
    assert float(accuracy(
        logits,
        manual_batch.target_candidate,
        manual_batch.candidate_owner,
        manual_batch.num_decisions,
    )) == 1.0

    # 组内 argmax 必须落在该 decision 自己的组里
    picks = grouped_argmax(
        logits, manual_batch.candidate_owner, manual_batch.num_decisions
    )
    assert len(picks) == manual_batch.num_decisions
    for decision_index, pick in enumerate(picks.tolist()):
        assert int(manual_batch.candidate_owner[pick]) == decision_index

    # 把某个 decision 的正确候选压掉，它就必须错，其它不受影响
    decision = 0
    wrong = logits.clone()
    wrong[manual_batch.target_candidate[decision]] = -5.0
    wrong[manual_batch.target_candidate[1]] = 5.0
    matches = grouped_argmax(
        wrong, manual_batch.candidate_owner, manual_batch.num_decisions
    ) == manual_batch.target_candidate
    assert not bool(matches[decision])
    assert bool(matches[1])


def test_loss_is_smaller_when_the_target_is_a_peaked_distribution(manual_batch):
    """loss 的定义：目标 candidate 的 -log p 加权平均，peaked 时更小。"""
    from src.training.losses import clean_state_loss
    from src.utils.segment_ops import grouped_log_softmax

    logits = torch.zeros(manual_batch.num_candidates)
    log_prob = grouped_log_softmax(
        logits, manual_batch.candidate_owner, manual_batch.num_decisions
    )
    uniform = clean_state_loss(
        log_prob,
        manual_batch.target_candidate,
        manual_batch.candidate_owner,
        manual_batch.candidate_is_null,
        manual_batch.num_decisions,
    )

    peaked = logits.clone()
    peaked[manual_batch.target_candidate] = 10.0
    peaked_log_prob = grouped_log_softmax(
        peaked, manual_batch.candidate_owner, manual_batch.num_decisions
    )
    sharp = clean_state_loss(
        peaked_log_prob,
        manual_batch.target_candidate,
        manual_batch.candidate_owner,
        manual_batch.candidate_is_null,
        manual_batch.num_decisions,
    )
    assert sharp < uniform


def test_tiny_overfit_step_reduces_loss(manual_batch):
    """几步优化之后，单步 loss 必须下降（Milestone F 的最小验收）。"""
    torch.manual_seed(0)
    model = make_model()
    diffusion = make_diffusion(T=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def run_loss():
        return recurrent_reverse_loss(
            model, diffusion, manual_batch, max_steps=3
        ).loss

    first = float(run_loss().detach())
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        loss = run_loss()
        loss.backward()
        optimizer.step()
    last = float(run_loss().detach())
    assert last < first, f"loss did not decrease: {first} -> {last}"
