"""配置项必须真的生效（修改清单 P2-2）。

当前配置里有三类"看起来可切换、实际上没接线"的键：

    time.encoding / time.d_time / time.conditioning
    training.amp

这里逐个验证它们现在真的控制行为：取值未实现就报错，实现了就产生可观测差异。
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from src.models.denoiser import GraphFlowDenoiser
from src.models.time_encoder import TimeEncoder
from src.training.setup import model_kwargs
from src.utils.config import Config, load_config

CONFIG_PATH = "configs/controlled_unweighted.yaml"


# ---------------------------------------------------------------------------
# time encoder
# ---------------------------------------------------------------------------
def test_d_time_controls_the_sinusoidal_dimension():
    encoder = TimeEncoder(d_model=16, d_time=8)
    assert encoder.d_time == 8
    assert encoder.d_model == 16
    out = encoder(3)
    assert out.shape == (16,)
    # d_time != d_model 时必须有输入投影
    assert isinstance(encoder.input_proj, torch.nn.Linear)


def test_d_time_equal_to_d_model_uses_identity_projection():
    encoder = TimeEncoder(d_model=16, d_time=16)
    assert isinstance(encoder.input_proj, torch.nn.Identity)


def test_odd_d_time_is_rejected():
    with pytest.raises(ValueError):
        TimeEncoder(d_model=16, d_time=7)


def test_different_d_time_gives_different_parameters():
    small = TimeEncoder(d_model=16, d_time=8)
    large = TimeEncoder(d_model=16, d_time=32)
    assert sum(p.numel() for p in small.parameters()) != sum(
        p.numel() for p in large.parameters()
    )


# ---------------------------------------------------------------------------
# denoiser wiring
# ---------------------------------------------------------------------------
def test_model_respects_d_time_from_config(manual_batch):
    config = Config(
        {
            "model": {"d_model": 16, "ffn_hidden": 32},
            "time": {"encoding": "sinusoidal", "d_time": 8, "conditioning": "adaln"},
        }
    )
    kwargs = model_kwargs(config)
    assert kwargs["d_time"] == 8
    assert kwargs["time_encoding"] == "sinusoidal"
    assert kwargs["time_conditioning"] == "adaln"

    model = GraphFlowDenoiser(**kwargs)
    assert model.time_encoder.d_time == 8
    assert model.time_encoder.d_model == 16
    out = model.step(manual_batch, model.init_nodes(manual_batch), manual_batch.target_candidate, 5)
    assert out.H_next.shape == (manual_batch.num_nodes, 16)


def test_unimplemented_time_encoding_raises():
    with pytest.raises(NotImplementedError):
        GraphFlowDenoiser(d_model=16, time_encoding="learned")


def test_unimplemented_time_conditioning_raises():
    with pytest.raises(NotImplementedError):
        GraphFlowDenoiser(d_model=16, time_conditioning="concat")


def test_real_config_time_section_is_wired():
    """正式配置里的 time 段必须被真正读进模型（而不是只读不用）。"""
    config = load_config(CONFIG_PATH)
    kwargs = model_kwargs(config)
    assert kwargs["d_time"] == int(config.get("time.d_time", 128))
    assert kwargs["time_encoding"] == str(config.get("time.encoding", "sinusoidal"))
    assert kwargs["time_conditioning"] == str(
        config.get("time.conditioning", "adaln")
    )


def test_flow_steps_is_wired_from_config(manual_batch):
    """model.flow_steps 必须真的改变每个 reverse step 内部的交流轮数。"""
    config = Config(
        {
            "model": {
                "d_model": 16,
                "ffn_hidden": 32,
                "flow_steps": 4,
                "flow_slot_embedding": True,
            }
        }
    )
    kwargs = model_kwargs(config)
    assert kwargs["flow_steps"] == 4
    assert kwargs["slot_embedding"] is True

    model = GraphFlowDenoiser(**kwargs)
    assert model.flow_steps == 4
    out = model.step(
        manual_batch, model.init_nodes(manual_batch), manual_batch.target_candidate, 5
    )
    assert out.flow_steps == 4
    assert len(out.attn_per_slot) == 4


def test_real_config_flow_steps_is_wired():
    config = load_config(CONFIG_PATH)
    kwargs = model_kwargs(config)
    assert kwargs["flow_steps"] == int(config.get("model.flow_steps", 1))
    assert kwargs["flow_steps"] >= 1
    assert kwargs["slot_embedding"] == bool(
        config.get("model.flow_slot_embedding", True)
    )


# ---------------------------------------------------------------------------
# AMP（P2-2）
# ---------------------------------------------------------------------------
def _trainer(amp: bool, device: torch.device, run_dir=None):
    """构造一个最小 Trainer。

    默认 ``run_dir=None``：某些受限环境里临时目录的 ACL 会拒绝 pytest / 写入，
    这里只关心 AMP 接线，不需要落盘（Trainer 对 run_dir=None 是支持的）。
    """
    from src.data.dataset_builder import tiny_overfit_dataset
    from src.diffusion.categorical import CategoricalDiffusion
    from src.diffusion.schedule import NoiseSchedule
    from src.training.trainer import Trainer

    config = Config(
        {
            "seed": 0,
            "training": {
                "batch_size": 2,
                "epochs": 1,
                "amp": amp,
                "grad_clip": 1.0,
            },
            "loss": {"x0_ce": 1.0},
            "evaluation": {"stochastic_sampling": True, "batch_size": 2, "max_steps": 0},
            "model": {"d_model": 16, "ffn_hidden": 32},
            "time": {"d_time": 16, "encoding": "sinusoidal", "conditioning": "adaln"},
        }
    )
    dataset = tiny_overfit_dataset(num_samples=2, num_nodes=20, seed=0)
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32).to(device)
    diffusion = CategoricalDiffusion(NoiseSchedule(T=3, beta_start=0.05, beta_end=0.5))
    return Trainer(
        model=model,
        diffusion=diffusion,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        train_dataset=dataset,
        val_dataset=None,
        config=config,
        device=device,
        run_dir=run_dir,
    )


@pytest.fixture
def run_dir():
    """临时目录（只用于验证 Trainer 能建目录；失败就退化成 None）。"""
    path = Path(tempfile.mkdtemp(prefix="gjd-test-"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_amp_disabled_means_no_scaler_and_null_context():
    trainer = _trainer(amp=False, device=torch.device("cpu"))
    assert trainer.scaler is None
    assert isinstance(trainer._autocast(), contextlib.nullcontext)


def test_amp_on_cpu_is_ignored_not_crashing():
    """AMP 只在 cuda 上启用；CPU 上开启也不能崩（明确忽略）。"""
    trainer = _trainer(amp=True, device=torch.device("cpu"))
    assert trainer.amp_enabled is False
    assert trainer.scaler is None
    history = trainer.fit(epochs=1)
    assert history and "train_loss" in history[0]


def test_trainer_rejects_model_on_a_different_device():
    """模型和 batch 设备不一致时必须早失败，而不是在 forward 里报奇怪的错。"""
    from src.data.dataset_builder import tiny_overfit_dataset
    from src.diffusion.categorical import CategoricalDiffusion
    from src.diffusion.schedule import NoiseSchedule
    from src.training.trainer import Trainer

    config = Config({"training": {"batch_size": 2, "amp": False}})
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = CategoricalDiffusion(NoiseSchedule(T=3, beta_start=0.05, beta_end=0.5))
    with pytest.raises(ValueError):
        Trainer(
            model=model,
            diffusion=diffusion,
            optimizer=torch.optim.Adam(model.parameters()),
            train_dataset=tiny_overfit_dataset(num_samples=2, num_nodes=20, seed=0),
            config=config,
            device=torch.device("meta"),   # 故意和模型设备不同
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_amp_on_cuda_builds_a_scaler():
    trainer = _trainer(amp=True, device=torch.device("cuda"))
    assert trainer.amp_enabled is True
    assert trainer.scaler is not None
    history = trainer.fit(epochs=1)
    assert history and "train_loss" in history[0]
