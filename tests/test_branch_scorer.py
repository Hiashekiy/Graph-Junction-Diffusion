"""Branch Scorer / NULL Scorer / Grouped Softmax 测试
（实施指南第 13-14、25.8 节 + Milestone D 验收）。
"""

from __future__ import annotations

import torch
import pytest

from src.models.branch_scorer import BranchScorer, branch_mean_pool
from src.models.denoiser import GraphFlowDenoiser
from src.utils.segment_ops import grouped_log_softmax


# ---------------------------------------------------------------------------
# branch mean pool
# ---------------------------------------------------------------------------
def test_membership_excludes_the_owner(manual_sample, manual_batch):
    """branch_node_ids 里不允许出现 owner 节点本身。"""
    batch = manual_batch
    candidates = manual_sample.field.candidates
    for candidate_index, branch in enumerate(candidates.candidate_branch):
        if branch is None:
            continue
        owner_global = batch.decision_node[
            batch.candidate_owner[candidate_index]
        ].item()
        members = batch.branch_node_ids[candidate_index]
        length = int(batch.branch_node_lengths[candidate_index])
        assert owner_global not in members[:length].tolist()
        assert length == len(branch.nodes) - 1


def test_branch_mean_pool_is_the_mean_of_member_states(manual_sample, manual_batch):
    batch = manual_batch
    d_model = 8
    H = torch.arange(batch.num_nodes * d_model, dtype=torch.float32).reshape(
        batch.num_nodes, d_model
    )
    pooled = branch_mean_pool(
        H, batch.branch_node_ids, batch.branch_node_lengths, batch.num_candidates
    )
    candidates = manual_sample.field.candidates
    for candidate_index, branch in enumerate(candidates.candidate_branch):
        if branch is None:
            continue
        members = [node for node in branch.nodes[1:]]
        expected = torch.stack([H[node] for node in members]).mean(dim=0)
        assert torch.allclose(pooled[candidate_index], expected, atol=1e-5), (
            f"candidate {candidate_index} {branch.nodes}"
        )


def test_null_candidate_pool_is_zero(manual_sample, manual_batch):
    batch = manual_batch
    H = torch.ones(batch.num_nodes, 4)
    pooled = branch_mean_pool(
        H, batch.branch_node_ids, batch.branch_node_lengths, batch.num_candidates
    )
    null_ids = batch.candidate_is_null.nonzero().flatten()
    assert null_ids.numel() > 0
    assert torch.allclose(pooled[null_ids], torch.zeros(null_ids.numel(), 4))


def test_pool_ignores_padding_slots(manual_sample, manual_batch):
    """padding 位置即使被填上垃圾，也不能影响 mean。"""
    batch = manual_batch
    H = torch.ones(batch.num_nodes, 4)
    batch.branch_node_ids[batch.branch_node_lengths == 0] = 0
    if batch.branch_node_ids.shape[1] > 1:
        lengths = batch.branch_node_lengths
        for candidate_index in range(batch.num_candidates):
            length = int(lengths[candidate_index])
            if length and length < batch.branch_node_ids.shape[1]:
                batch.branch_node_ids[candidate_index, length:] = 0
    pooled = branch_mean_pool(
        H, batch.branch_node_ids, batch.branch_node_lengths, batch.num_candidates
    )
    expected = torch.ones(batch.num_candidates, 4)
    expected[batch.branch_node_lengths == 0] = 0
    assert torch.allclose(pooled, expected)


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------
def test_probabilities_sum_to_one_per_decision(manual_batch):
    scorer = BranchScorer(d_model=16)
    H = torch.randn(manual_batch.num_nodes, 16)
    tau = torch.randn(manual_batch.num_decisions, 16)
    out = scorer(H, manual_batch, tau)

    prob = out["candidate_prob"]
    sums = torch.zeros(manual_batch.num_decisions).index_add_(
        0, manual_batch.candidate_owner, prob
    )
    assert torch.allclose(sums, torch.ones(manual_batch.num_decisions), atol=1e-5)
    assert (prob > 0).all()
    log_prob = out["candidate_log_prob"]
    assert torch.allclose(log_prob.exp(), prob, atol=1e-6)


def test_logits_shape_matches_candidate_table(manual_batch):
    scorer = BranchScorer(d_model=16)
    H = torch.randn(manual_batch.num_nodes, 16)
    tau = torch.randn(manual_batch.num_decisions, 16)
    out = scorer(H, manual_batch, tau)
    assert out["candidate_logits"].shape == (manual_batch.num_candidates,)


def test_matches_grouped_log_softmax_of_raw_logits(manual_batch):
    scorer = BranchScorer(d_model=16)
    H = torch.randn(manual_batch.num_nodes, 16)
    tau = torch.randn(manual_batch.num_decisions, 16)
    out = scorer(H, manual_batch, tau)
    manual = grouped_log_softmax(
        out["candidate_logits"], manual_batch.candidate_owner, manual_batch.num_decisions
    )
    assert torch.allclose(out["candidate_log_prob"], manual, atol=1e-6)


def test_gradients_reach_both_heads(manual_batch):
    scorer = BranchScorer(d_model=16)
    H = torch.randn(manual_batch.num_nodes, 16, requires_grad=True)
    tau = torch.randn(manual_batch.num_decisions, 16, requires_grad=True)
    out = scorer(H, manual_batch, tau)
    loss = -out["candidate_log_prob"].mean()
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    assert tau.grad is not None and torch.isfinite(tau.grad).all()
    assert scorer.branch_mlp[0].weight.grad is not None
    assert scorer.null_mlp[0].weight.grad is not None


# ---------------------------------------------------------------------------
# denoiser-level integration
# ---------------------------------------------------------------------------
def test_denoiser_step_returns_valid_distribution(manual_batch):
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    H_t = model.init_nodes(manual_batch)
    out = model.step(manual_batch, H_t, manual_batch.target_candidate, 7)

    assert out.H_next.shape == H_t.shape
    sums = torch.zeros(manual_batch.num_decisions).index_add_(
        0, manual_batch.candidate_owner, out.candidate_prob
    )
    assert torch.allclose(sums, torch.ones(manual_batch.num_decisions), atol=1e-5)


def test_source_group_has_no_null_and_junction_group_has_null(manual_sample, manual_batch):
    candidates = manual_sample.field.candidates
    for decision_index, node in enumerate(manual_sample.segments.decision_nodes):
        null_count = sum(
            1
            for owner, is_null in zip(
                candidates.candidate_owner, candidates.candidate_is_null
            )
            if owner == decision_index and is_null
        )
        if node == manual_sample.segments.start:
            assert null_count == 0
        else:
            assert null_count == 1


def test_null_logit_only_depends_on_junction_state_and_tau(manual_batch):
    """NULL scorer 输入是 [h_i, tau_t]，不含 branch pool。"""
    scorer = BranchScorer(d_model=16)
    H = torch.randn(manual_batch.num_nodes, 16)
    tau = torch.randn(manual_batch.num_decisions, 16)
    logits_a = scorer.null_logits(H, manual_batch, tau)

    # 改掉所有 branch 成员节点（非 owner）的状态：NULL logits 必须完全不变
    members = manual_batch.branch_node_ids.reshape(-1)
    lengths = manual_batch.branch_node_lengths
    valid = torch.arange(manual_batch.branch_node_ids.shape[1])[None, :] < lengths[:, None]
    member_nodes = members[valid.reshape(-1)].unique()
    owner_nodes = manual_batch.decision_node[manual_batch.candidate_owner].unique()
    pure_members = torch.tensor(
        [int(n) for n in member_nodes.tolist() if int(n) not in set(owner_nodes.tolist())]
    )
    if pure_members.numel() == 0:
        pytest.skip("no branch-only nodes in this batch")

    H2 = H.clone()
    H2[pure_members] = torch.randn(pure_members.numel(), 16)

    logits_b = scorer.null_logits(H2, manual_batch, tau)
    assert torch.allclose(logits_a, logits_b)
    assert logits_a.dim() == 1
