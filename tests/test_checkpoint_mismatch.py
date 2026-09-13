"""checkpoint 加载失败时给出的诊断（第三轮修订）。

`model.flow_steps` 会改变参数集合（每轮一个 flow_slot_embedding），所以拿
flow_steps=3 的 config 去加载 flow_steps=1 训出来的 checkpoint 必然失败。裸的
``load_state_dict`` 报错只会列出 key 名，看不出根因；这里断言我们给出的信息里
包含 flow_steps 提示。

写盘位置：沙箱里 pytest 的 ``tmp_path`` 不可用（权限问题），所以用一个
workspace 内的临时目录，测完即删。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from src.models.denoiser import GraphFlowDenoiser
from src.training.checkpoint import load_checkpoint, save_checkpoint

TMP_DIR = Path("outputs/_pytest_checkpoint")


def test_mismatched_flow_steps_raises_readable_error():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / "single.pt"
    try:
        single = GraphFlowDenoiser(d_model=16, ffn_hidden=32, flow_steps=1)
        save_checkpoint(
            path, single, model_config={"d_model": 16, "flow_steps": 1}
        )

        triple = GraphFlowDenoiser(d_model=16, ffn_hidden=32, flow_steps=3)
        with pytest.raises(RuntimeError) as info:
            load_checkpoint(path, model=triple)
        message = str(info.value)
        assert "flow_steps" in message
        assert "run_config.json" in message
        assert "flow_slot_embedding" in message
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_same_flow_steps_loads_fine():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / "same.pt"
    try:
        model = GraphFlowDenoiser(d_model=16, ffn_hidden=32, flow_steps=3)
        save_checkpoint(path, model, model_config={"d_model": 16, "flow_steps": 3})
        clone = GraphFlowDenoiser(d_model=16, ffn_hidden=32, flow_steps=3)
        payload = load_checkpoint(path, model=clone)
        assert payload["model_config"]["flow_steps"] == 3
        for (name_a, param_a), (name_b, param_b) in zip(
            model.state_dict().items(), clone.state_dict().items()
        ):
            assert name_a == name_b
            assert param_a.shape == param_b.shape
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# 第二轮修订：K 从 "只吃 edge" 变成 "node + edge"，旧 checkpoint 需要等价映射
# ---------------------------------------------------------------------------
def test_legacy_attention_state_dict_is_detected():
    from src.training.checkpoint import needs_attention_remap

    new_model = GraphFlowDenoiser(d_model=16, ffn_hidden=32, flow_steps=1)
    assert not needs_attention_remap(new_model.state_dict())

    legacy = {
        key: value
        for key, value in new_model.state_dict().items()
        if "k_node_proj" not in key and "k_edge_proj" not in key
    }
    legacy["k_proj.weight"] = torch.zeros(16, 16)
    legacy["k_proj.bias"] = torch.zeros(16)
    assert needs_attention_remap(legacy)


def test_legacy_checkpoint_loads_and_reproduces_the_old_attention(manual_batch):
    """旧 checkpoint（K = W_K e）映射后必须与旧模型**数值等价**。

    做法：造一个"退化"新版模型（k_node = 0），把它写成旧格式的 k_proj
    （k_proj = k_edge * k_pair_scale），再走 load_checkpoint 的自动映射；映射后的
    模型应当与退化模型逐位一致（attention 也一致）。
    """
    from src.models.graph_flow import GraphFlowBlock
    from src.training.checkpoint import (
        load_checkpoint,
        remap_legacy_attention_state_dict,
    )

    torch.manual_seed(0)
    degenerate = GraphFlowBlock(d_model=16, ffn_hidden=32)
    with torch.no_grad():
        degenerate.k_node_proj.weight.zero_()
        degenerate.k_node_proj.bias.zero_()

    state = degenerate.state_dict()
    legacy = {
        key: value
        for key, value in state.items()
        if "k_node_proj" not in key and "k_edge_proj" not in key
    }
    legacy["k_proj.weight"] = state["k_edge_proj.weight"] * degenerate.k_pair_scale
    legacy["k_proj.bias"] = state["k_edge_proj.bias"] * degenerate.k_pair_scale

    remapped = remap_legacy_attention_state_dict(legacy)
    clone = GraphFlowBlock(d_model=16, ffn_hidden=32)
    clone.load_state_dict(remapped)
    assert torch.equal(
        clone.k_node_proj.weight, torch.zeros_like(clone.k_node_proj.weight)
    )

    edge_feat = torch.randn(manual_batch.edge_index.shape[1], 16)
    H_t = torch.randn(manual_batch.num_nodes, 16)
    tau = torch.zeros(manual_batch.num_graphs, 16)
    kwargs = dict(
        H_t=H_t,
        edge_index=manual_batch.edge_index,
        edge_feat=edge_feat,
        tau_t=tau,
        fixed_mask=manual_batch.start_goal_mask,
        graph_node_ptr=manual_batch.graph_node_ptr,
    )
    before = degenerate(**kwargs)
    after = clone(**kwargs)
    assert torch.allclose(before["attn"], after["attn"], atol=1e-6)
    assert torch.allclose(before["H_next"], after["H_next"], atol=1e-6)

    # 走完整的 save -> load_checkpoint 路径（自动检测并映射）
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / "legacy.pt"
    try:
        torch.save({"model": legacy, "model_config": {"d_model": 16}}, path)
        loaded = GraphFlowBlock(d_model=16, ffn_hidden=32)
        load_checkpoint(path, model=loaded)
        assert torch.allclose(loaded(**kwargs)["H_next"], before["H_next"], atol=1e-6)
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
