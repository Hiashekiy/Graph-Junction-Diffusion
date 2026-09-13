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
