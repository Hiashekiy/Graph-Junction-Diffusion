"""Graph Flow Block 测试（实施指南第 10-11、25.5 节 + Milestone C 验收）。

验收点：

    Q/K/V shape 正确
    destination grouped softmax 正确（每个接收节点权重和 = 1）
    Start / Goal 只出不进 + 状态被 clamp
    Residual 保留 H（不是 H_next = m）
    AdaLN 真的吃到了 tau_t
"""

from __future__ import annotations

import torch

from src.models.edge_state import EdgeStateEncoder
from src.models.graph_flow import GraphFlowBlock
from src.models.time_encoder import TimeEncoder


def _inputs(batch, d_model=16):
    encoder = EdgeStateEncoder(d_model)
    edge_feat = encoder(batch, batch.target_candidate)
    H_t = torch.randn(batch.num_nodes, d_model)
    tau = torch.randn(batch.num_graphs, d_model)
    return H_t, edge_feat, tau


def test_forward_shapes(manual_batch):
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    H_t, edge_feat, tau = _inputs(manual_batch)
    out = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    assert out["H_next"].shape == H_t.shape
    assert out["attn"].shape == (manual_batch.edge_index.shape[1],)
    assert torch.isfinite(out["H_next"]).all()


def test_attention_is_grouped_by_destination(manual_batch):
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    H_t, edge_feat, tau = _inputs(manual_batch)
    out = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    attn = out["attn"]
    # 手动重算一遍：只保留 dst 不是 start/goal 的边，并按 dst 归一化
    fixed = manual_batch.start_goal_mask
    keep = ~fixed[manual_batch.edge_index[1]]
    dst = manual_batch.edge_index[1][keep]
    for node in dst.unique():
        rows = attn[keep][dst == node]
        assert torch.allclose(rows.sum(), torch.tensor(1.0), atol=1e-5)


def test_start_and_goal_do_not_receive_messages(manual_batch):
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    H_t, edge_feat, tau = _inputs(manual_batch)
    out = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    receives_at_fixed = manual_batch.start_goal_mask[manual_batch.edge_index[1]]
    assert receives_at_fixed.any(), "this batch must contain edges into start/goal"
    assert torch.allclose(out["attn"][receives_at_fixed], torch.zeros(1))


def test_start_and_goal_state_is_clamped(manual_batch):
    """无论邻居输入什么，h_s^{t-1} = h_s^t、h_g^{t-1} = h_g^t。"""
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    H_t, edge_feat, tau = _inputs(manual_batch)
    out = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    H_next = out["H_next"]
    assert torch.allclose(H_next[manual_batch.starts], H_t[manual_batch.starts])
    assert torch.allclose(H_next[manual_batch.goals], H_t[manual_batch.goals])


def test_non_fixed_nodes_do_change(manual_batch):
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    H_t, edge_feat, tau = _inputs(manual_batch)
    out = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    free = ~manual_batch.start_goal_mask
    assert not torch.allclose(out["H_next"][free], H_t[free])


def test_residual_keeps_old_state_and_gradients_flow(manual_batch):
    """H_next 必须依赖 H_t（residual），而且梯度能回到输入与参数。"""
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    H_t = torch.randn(manual_batch.num_nodes, 16, requires_grad=True)
    edge_feat = torch.randn(manual_batch.edge_index.shape[1], 16, requires_grad=True)
    tau = torch.zeros(manual_batch.num_graphs, 16)
    out = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    out["H_next"].sum().backward()
    assert H_t.grad is not None and torch.isfinite(H_t.grad).all()
    assert edge_feat.grad is not None and torch.isfinite(edge_feat.grad).all()
    assert block.q_proj.weight.grad is not None


def test_attention_uses_edge_state_as_key(manual_batch):
    """K 来自 edge state：把 selected/unselected 的 embedding 换掉，attention 必变。"""
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    H_t, _, tau = _inputs(manual_batch)
    edge_feat = torch.zeros(manual_batch.edge_index.shape[1], 16)

    with torch.no_grad():
        block.q_proj.weight.zero_()
        block.q_proj.bias.zero_()
        block.k_proj.weight.zero_()
        block.k_proj.bias.zero_()
        block.q_proj.weight[:, 0] = 1.0
        block.k_proj.weight[:, 0] = 1.0

    unselected = edge_feat.clone()
    selected = edge_feat.clone()
    selected[0, 0] = 5.0

    out_a = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=unselected,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    out_b = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=selected,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    assert not torch.allclose(out_a["attn"], out_b["attn"])


def test_time_conditioning_actually_depends_on_t(manual_batch):
    """tau_t 进入 AdaLN：把 tau 换掉，输出必须变化。"""
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    H_t, edge_feat, _ = _inputs(manual_batch)
    out_a = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=torch.zeros(manual_batch.num_graphs, 16),
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    out_b = block(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=torch.ones(manual_batch.num_graphs, 16),
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    assert not torch.allclose(out_a["H_next"], out_b["H_next"])


def test_time_encoder_produces_different_embeddings_for_different_t():
    encoder = TimeEncoder(d_model=16)
    tau_1 = encoder(1)
    tau_T = encoder(50)
    assert tau_1.shape == (16,)
    assert not torch.allclose(tau_1, tau_T)


def test_broadcast_time_matches_graph_ownership(manual_batch):
    from src.models.time_encoder import broadcast_time

    tau = torch.arange(manual_batch.num_graphs, dtype=torch.float32)[:, None].expand(
        manual_batch.num_graphs, 4
    )
    node_tau = broadcast_time(tau, manual_batch.graph_node_ptr)
    assert node_tau.shape[0] == manual_batch.num_nodes
    for graph_index in range(manual_batch.num_graphs):
        lo = manual_batch.graph_node_ptr[graph_index]
        hi = manual_batch.graph_node_ptr[graph_index + 1]
        assert torch.allclose(
            node_tau[lo:hi], tau[graph_index].expand(hi - lo, -1)
        )


def test_shares_one_cell_across_timesteps(manual_batch):
    """同一组参数可以被任意 timestep 复用（参数数量与 T 无关）。"""
    from src.models.denoiser import GraphFlowDenoiser

    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    H_t = model.init_nodes(manual_batch)
    z = manual_batch.target_candidate
    for t in (1, 17, 50):
        out = model.step(manual_batch, H_t, z, t)
        assert out.H_next.shape == H_t.shape


# ---------------------------------------------------------------------------
# 一个 reverse step 内部的多轮图信息交流（forward_multi）
# ---------------------------------------------------------------------------
def _multi_kwargs(batch):
    H_t, edge_feat, tau = _inputs(batch)
    return dict(
        H_t=H_t,
        edge_index=batch.edge_index,
        edge_feat=edge_feat,
        tau_t=tau,
        fixed_mask=batch.start_goal_mask,
        graph_node_ptr=batch.graph_node_ptr,
    )


def test_forward_multi_single_step_matches_forward(manual_batch):
    """flow_steps=1 必须与旧的单轮 forward 完全一致（向后兼容）。"""
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    kwargs = _multi_kwargs(manual_batch)
    single = block(**kwargs)
    multi = block.forward_multi(**kwargs, flow_steps=1)
    assert torch.allclose(single["H_next"], multi["H_next"])
    assert multi["flow_steps"] == 1
    assert len(multi["attn_per_slot"]) == 1


def test_forward_multi_reports_each_round(manual_batch):
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    kwargs = _multi_kwargs(manual_batch)
    out = block.forward_multi(**kwargs, flow_steps=3)
    assert out["flow_steps"] == 3
    assert len(out["attn_per_slot"]) == 3
    assert out["H_next"].shape == kwargs["H_t"].shape
    assert torch.isfinite(out["H_next"]).all()


def test_forward_multi_is_not_a_fixed_point_iteration(manual_batch):
    """多轮不等于"把同一个映射反复作用于同一输入"：每轮条件不同，输出也不同。"""
    import torch.nn as nn

    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    kwargs = _multi_kwargs(manual_batch)
    slot = nn.Embedding(4, 16)
    nn.init.normal_(slot.weight, std=0.5)

    plain = block.forward_multi(**kwargs, flow_steps=3)
    slotted = block.forward_multi(**kwargs, flow_steps=3, slot_embedding=slot)
    assert not torch.allclose(plain["H_next"], slotted["H_next"])
    # 逐轮之间也真的在变（不是第一轮之后就不动了）
    assert not torch.allclose(
        slotted["attn_per_slot"][0], slotted["attn_per_slot"][1]
    )


def test_forward_multi_clamps_start_and_goal_every_round(manual_batch):
    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    kwargs = _multi_kwargs(manual_batch)
    out = block.forward_multi(**kwargs, flow_steps=4)
    assert torch.allclose(out["H_next"][manual_batch.starts], kwargs["H_t"][manual_batch.starts])
    assert torch.allclose(out["H_next"][manual_batch.goals], kwargs["H_t"][manual_batch.goals])


def test_forward_multi_backward_reaches_every_round(manual_batch):
    import torch.nn as nn

    block = GraphFlowBlock(d_model=16, ffn_hidden=32)
    H_t, edge_feat, tau = _inputs(manual_batch)
    H_t = H_t.clone().requires_grad_(True)
    slot = nn.Embedding(3, 16)
    out = block.forward_multi(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
        flow_steps=3,
        slot_embedding=slot,
    )
    out["H_next"].sum().backward()
    assert H_t.grad is not None and torch.isfinite(H_t.grad).all()
    assert slot.weight.grad is not None and torch.isfinite(slot.weight.grad).all()


def test_denoiser_flow_steps_is_configurable(manual_batch):
    """denoiser 的 flow_steps 真的生效：轮数写进输出，且额外带了 slot embedding 参数。"""
    from src.models.denoiser import GraphFlowDenoiser

    one = GraphFlowDenoiser(d_model=16, ffn_hidden=32, flow_steps=1)
    three = GraphFlowDenoiser(d_model=16, ffn_hidden=32, flow_steps=3)
    assert one.flow_slot_embedding is None
    assert three.flow_slot_embedding is not None
    assert three.num_parameters() > one.num_parameters()
    expected = 3 * 16
    assert three.num_parameters() - one.num_parameters() == expected

    H_t = three.init_nodes(manual_batch)
    out = three.step(manual_batch, H_t, manual_batch.target_candidate, 10)
    assert out.flow_steps == 3
    assert len(out.attn_per_slot) == 3


def test_denoiser_flow_steps_changes_the_update(manual_batch):
    from src.models.denoiser import GraphFlowDenoiser

    one = GraphFlowDenoiser(d_model=16, ffn_hidden=32, flow_steps=1)
    three = GraphFlowDenoiser(d_model=16, ffn_hidden=32, flow_steps=3)
    three.load_state_dict(one.state_dict(), strict=False)
    z = manual_batch.target_candidate
    H_t = one.init_nodes(manual_batch)
    out_one = one.step(manual_batch, H_t, z, 10)
    out_three = three.step(manual_batch, H_t, z, 10)
    assert not torch.allclose(out_one.H_next, out_three.H_next)
