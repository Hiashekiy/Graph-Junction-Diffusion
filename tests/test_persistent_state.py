"""Persistent state 测试（实施指南第 12、25.6、25.7 节 + Milestone C/E 验收）。

核心命题：

    H_t 是显式状态，每一步的输出直接是下一步的输入，**绝不重新初始化**；
    同一个 Graph Flow Cell 在整条 T-step chain 上共享参数。
"""

from __future__ import annotations

import torch

from src.diffusion.categorical import CategoricalDiffusion
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample_reverse_chain
from src.models.denoiser import GraphFlowDenoiser


def make_diffusion(T: int = 5) -> CategoricalDiffusion:
    return CategoricalDiffusion(
        NoiseSchedule(T=T, schedule="linear", beta_start=0.05, beta_end=0.5)
    )


def test_step_output_feeds_the_next_step(manual_batch):
    """第二步的输入必须就是第一步的输出（不是重新初始化）。"""
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = make_diffusion()
    H_T = model.init_nodes(manual_batch)
    z = manual_batch.target_candidate

    first = model.step(manual_batch, H_T, z, 5)
    H_after_first = first.H_next

    # 重新初始化出来的 H 一定和传播之后的 H 不同（否则说明 Graph Flow 没起作用）
    assert not torch.allclose(H_after_first, H_T)

    second = model.step(manual_batch, H_after_first, z, 4)
    # 用重新初始化的 H 跑同样的第二步，结果必须不同
    reinit = model.step(manual_batch, model.init_nodes(manual_batch), z, 4)
    assert not torch.allclose(second.H_next, reinit.H_next)


def test_state_accumulates_over_two_steps(manual_batch):
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    H_T = model.init_nodes(manual_batch)
    z = manual_batch.target_candidate
    H_1 = model.step(manual_batch, H_T, z, 2).H_next
    H_0 = model.step(manual_batch, H_1, z, 1).H_next
    assert not torch.allclose(H_0, H_1)
    assert not torch.allclose(H_0, H_T)


def test_start_goal_keep_their_initial_embedding_over_many_steps(manual_batch):
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    H = model.init_nodes(manual_batch)
    H_T = H.clone()
    z = manual_batch.target_candidate
    for t in (5, 4, 3, 2, 1):
        H = model.step(manual_batch, H, z, t).H_next
    assert torch.allclose(H[manual_batch.starts], H_T[manual_batch.starts])
    assert torch.allclose(H[manual_batch.goals], H_T[manual_batch.goals])


def test_no_mutable_hidden_state_inside_the_module(manual_batch):
    """H 必须是 API 的一部分，不能藏在 module 里。"""
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    assert not hasattr(model, "H")
    assert not hasattr(model, "hidden")
    buffers = {name for name, _ in model.named_buffers()}
    assert not any("H" == name or name.endswith(".H") for name in buffers)


def test_parameter_count_does_not_grow_with_T(manual_batch):
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    before = model.num_parameters()
    H = model.init_nodes(manual_batch)
    z = manual_batch.target_candidate
    for t in (50, 49, 48):
        model.step(manual_batch, H, z, t)
    assert model.num_parameters() == before


def test_only_one_graph_flow_cell_exists():
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    flows = [name for name, _ in model.named_modules() if name.split(".")[-1] == "graph_flow"]
    assert flows == ["graph_flow"]
    assert isinstance(model.graph_flow, torch.nn.Module)


def test_gradients_travel_along_the_recurrent_chain(manual_batch):
    """反向传播必须能穿过 H_T -> H_{T-1} -> H_{T-2}。"""
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    H_T = model.init_nodes(manual_batch)
    z = manual_batch.target_candidate

    H_1 = model.step(manual_batch, H_T, z, 3).H_next
    out = model.step(manual_batch, H_1, z, 2)
    loss = -out.candidate_log_prob.mean()
    loss.backward()

    # 第一步的参数也必须拿到梯度 —— 说明梯度穿过了第二步（recurrent）
    assert model.graph_flow.q_proj.weight.grad is not None
    assert model.graph_flow.q_proj.weight.grad.abs().sum() > 0


def test_reverse_chain_state_sequence(tiny_batches):
    """sampler 的 (H_t, z_t) 链：H 每一步都换、z 每一步合法。"""
    samples, batch = tiny_batches
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = make_diffusion(T=5)

    chain = sample_reverse_chain(diffusion, model, batch, stochastic=True, record=True)
    trace = chain["trace"]
    assert trace is not None
    assert len(trace.H_path) == 6      # H_5 ... H_0
    assert len(trace.z_path) == 6

    for index in range(1, len(trace.H_path)):
        assert not torch.allclose(trace.H_path[index], trace.H_path[index - 1])

    for z in trace.z_path:
        assert z.shape == (batch.num_decisions,)
        for decision_index in range(batch.num_decisions):
            owner = int(batch.candidate_owner[z[decision_index]])
            assert owner == decision_index


def test_reverse_chain_returns_final_state(tiny_batches):
    samples, batch = tiny_batches
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = make_diffusion(T=4)
    chain = sample_reverse_chain(diffusion, model, batch, stochastic=True)
    assert chain["z0"].shape == (batch.num_decisions,)
    assert chain["H0"].shape == (batch.num_nodes, 16)


def test_deterministic_sampling_is_reproducible(tiny_batches):
    """stochastic=False 时整条链完全可复现（用的是启发式初值，见 P2-1）。"""
    samples, batch = tiny_batches
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = make_diffusion(T=4)
    first = sample_reverse_chain(diffusion, model, batch, stochastic=False)
    second = sample_reverse_chain(diffusion, model, batch, stochastic=False)
    assert torch.equal(first["z0"], second["z0"])
    assert torch.equal(first["H0"], second["H0"])


def test_stochastic_prior_with_fixed_seed_is_reproducible(tiny_batches):
    """P2-1 推荐的复现方式：仍然 z_T ~ pi，但用固定 seed 的 generator。"""
    from src.utils.seed import make_generator

    samples, batch = tiny_batches
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = make_diffusion(T=4)

    first = sample_reverse_chain(
        diffusion, model, batch, stochastic=True, generator=make_generator(7)
    )
    second = sample_reverse_chain(
        diffusion, model, batch, stochastic=True, generator=make_generator(7)
    )
    third = sample_reverse_chain(
        diffusion, model, batch, stochastic=True, generator=make_generator(8)
    )
    assert torch.equal(first["z0"], second["z0"])
    assert not torch.equal(first["z0"], third["z0"])


def test_heuristic_prior_is_not_the_uniform_prior(tiny_batches):
    """P2-1：启发式初值不是 z_T ~ pi 的确定性等价物（两者取值明显不同）。"""
    from src.diffusion.sampler import (
        heuristic_prior_deterministic,
        sample_prior,
    )

    samples, batch = tiny_batches
    diffusion = make_diffusion(T=4)
    heuristic = heuristic_prior_deterministic(
        batch.candidate_owner, batch.candidate_is_null, batch.num_decisions
    )

    # 启发式规则：普通 Junction 取 NULL；source 是 decision 时取第一条 branch
    source_mask = batch.source_decision_mask
    assert bool(source_mask.any()), "这个 batch 里应该有 source decision"
    for decision_index in range(batch.num_decisions):
        pick = int(heuristic[decision_index])
        if bool(source_mask[decision_index]):
            assert not bool(batch.candidate_is_null[pick]), "source 应取第一条 branch"
        else:
            assert bool(batch.candidate_is_null[pick]), "普通 junction 应取 NULL"

    # 均匀先验 z_T ~ pi 会以正概率取到非 NULL 的 branch；启发式初值几乎全是 NULL
    uniform_null_fraction = []
    for _ in range(5):
        z = sample_prior(diffusion, batch.candidate_owner, batch.num_decisions)
        uniform_null_fraction.append(float(batch.candidate_is_null[z].float().mean()))
    heuristic_null_fraction = float(batch.candidate_is_null[heuristic].float().mean())
    assert heuristic_null_fraction > max(uniform_null_fraction)


def test_deterministic_prior_uses_null_for_junctions():
    """确定性初始状态：普通 Junction 取 NULL，source 取第一条 branch。"""
    from src.diffusion.sampler import heuristic_prior_deterministic

    torch.manual_seed(0)
    candidates = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    is_null = torch.tensor(
        [True, False, False, False, False, True, False, False, False]
    )
    z = heuristic_prior_deterministic(candidates, is_null, 2)
    assert z.tolist() == [0, 5]
