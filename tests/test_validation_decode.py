"""验证 / 选 best.pt 的解码口径必须与最终评测**同一把尺子**。

背景（真实踩过的坑）：``Trainer.validate()`` 以前直接走 ``evaluate_dataset()`` 的
默认形参 —— ``decode="single"``、``strict_decode=False``、``beam_width=64`` —— 而
最终评测用的是 strict 多分支。于是 ``best.pt`` 是按 single 的 ``path_similarity_score``
挑的：碰到"single 后期变差、strict 后期反而继续提升"时，会选中更早也更差的那一轮。

这套测试守三件事：

1. **默认值没有破坏旧行为**：没写这些键的 config（controlled_unweighted /
   controlled_weighted）必须与改动前逐位一致（single / 非 strict / beam 64）。
2. **config 里的键真的被传进 ``evaluate_dataset``**：只断言 ``Trainer`` 存了字段
   是不够的 —— 字段存下来却没接线，正是这个 bug 的原始形态。
3. **配错要立刻报错**，不能静默降级成 single。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import tiny_overfit_dataset  # noqa: E402
from src.diffusion.categorical import CategoricalDiffusion  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.models.denoiser import GraphFlowDenoiser  # noqa: E402
from src.training import trainer as trainer_mod  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import Config, load_config  # noqa: E402


def _dataset():
    try:
        return tiny_overfit_dataset(num_samples=2, num_nodes=20, seed=0)
    except RuntimeError as error:  # pragma: no cover - 依赖随机性
        pytest.skip(f"dataset generation failed: {error}")


def build_trainer(config: Config, run_dir=None) -> Trainer:
    device = torch.device("cpu")
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32).to(device)
    diffusion = CategoricalDiffusion(NoiseSchedule(T=3, beta_start=0.05, beta_end=0.5))
    dataset = _dataset()
    return Trainer(
        model=model,
        diffusion=diffusion,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        train_dataset=dataset,
        val_dataset=dataset,
        config=config,
        device=device,
        run_dir=run_dir,
    )


def base_config(**evaluation) -> Config:
    """最小可用 config；``evaluation`` 里的键会覆盖进去。"""
    return Config(
        {
            "seed": 0,
            "training": {"batch_size": 2, "epochs": 1, "amp": False, "grad_clip": 1.0},
            "loss": {"x0_ce": 1.0},
            "evaluation": {
                "stochastic_sampling": True,
                "batch_size": 2,
                "max_steps": 0,
                **evaluation,
            },
            "model": {"d_model": 16, "ffn_hidden": 32},
            "time": {"d_time": 16, "encoding": "sinusoidal", "conditioning": "adaln"},
        }
    )


def capture_validate_kwargs(monkeypatch, trainer: Trainer) -> dict:
    """跑一次 validate()，把传给 ``evaluate_dataset`` 的关键字抓出来。

    ``evaluate_dataset`` 被 import 进了 ``src.training.trainer`` 的名字空间，所以要
    打在**那个**模块上。用 stub 返回一个最小 report —— 这里要验的是"传了什么口径"，
    而不是评测本身（评测另有测试覆盖）。
    """
    captured: dict = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            metrics={"path_similarity_score": 0.5},
            debug={"accuracy": 0.1, "loss": 0.2, "soft_goal": 0.3},
            records=[],
        )

    monkeypatch.setattr(trainer_mod, "evaluate_dataset", fake)
    trainer.validate(epoch=1)
    return captured


# ---------------------------------------------------------------------------
# 默认 = 旧行为（逐位不变）
# ---------------------------------------------------------------------------
def test_defaults_match_evaluate_dataset_signature():
    """没写 evaluation.decode 时，Trainer 的默认值必须等于 evaluate_dataset 的形参。

    这是"旧 config 行为一个字节都没变"的机械保证：只要这里对不上，
    controlled_unweighted / controlled_weighted 的历史验证指标就会漂。
    """
    import inspect

    signature = inspect.signature(trainer_mod.evaluate_dataset)
    trainer = build_trainer(base_config())
    assert trainer.eval_decode == signature.parameters["decode"].default == "single"
    assert (
        trainer.eval_strict_decode
        == signature.parameters["strict_decode"].default
        == False  # noqa: E712 - 显式表意
    )
    assert trainer.eval_top_k == signature.parameters["top_k"].default == 2
    assert trainer.eval_beam_width == signature.parameters["beam_width"].default == 64
    assert (
        trainer.eval_null_policy == signature.parameters["null_policy"].default == "stop"
    )
    assert (
        trainer.eval_filter_dead_branches
        == signature.parameters["filter_dead_branches"].default
        == False  # noqa: E712
    )


def test_default_trainer_forwards_the_historical_ruler(monkeypatch):
    """默认 config 必须转发 single / 非 strict / beam 64 给 evaluate_dataset。"""
    trainer = build_trainer(base_config())
    kwargs = capture_validate_kwargs(monkeypatch, trainer)
    assert kwargs["decode"] == "single"
    assert kwargs["strict_decode"] is False
    assert kwargs["beam_width"] == 64
    assert kwargs["top_k"] == 2
    # 旧的 stochastic_sampling 旋钮继续生效
    assert kwargs["stochastic"] is True


# ---------------------------------------------------------------------------
# 配置真的被接线
# ---------------------------------------------------------------------------
def test_didi_config_declares_the_strict_2_3_ruler():
    """真实数据 config 声明的就是最终推理那套：strict multi 2/3 + 确定性。"""
    config = load_config(PROJECT_ROOT / "configs" / "didi_chengdu.yaml")
    assert config.get("evaluation.decode") == "multi"
    assert config.get("evaluation.strict_decode") is True
    assert config.get("evaluation.top_k") == 2
    assert config.get("evaluation.beam_width") == 3
    # --deterministic 的等价写法；否则 val 与最终测试不是同一个数
    assert config.get("evaluation.stochastic_sampling") is False
    # 选模型用的正是 strict 口径算出来的 path_similarity_score
    assert config.get("training.selection_metric") == "path_similarity_score"


def test_didi_config_is_forwarded_to_evaluate_dataset(monkeypatch):
    """**核心断言**：config 里写的 strict 2/3 真的传到了 evaluate_dataset。

    只检查 Trainer 存了字段是不够的 —— 当初这个 bug 就是"字段/参数存在但没接上"。
    """
    config = load_config(PROJECT_ROOT / "configs" / "didi_chengdu.yaml")
    trainer = build_trainer(config)
    kwargs = capture_validate_kwargs(monkeypatch, trainer)

    assert kwargs["decode"] == "multi"
    assert kwargs["strict_decode"] is True
    assert kwargs["top_k"] == 2
    assert kwargs["beam_width"] == 3
    assert kwargs["null_policy"] == "stop"
    assert kwargs["stochastic"] is False, "验证必须与 evaluate.py --deterministic 同口径"


def test_strict_multi_is_not_the_train_miner(monkeypatch):
    """训练 miner 与评测 decoder 是**两套**参数，不要混。

    训练要看得见失败（historical 2/8、strict=false），评测只问"能不能到终点"
    （strict 2/3）。这条守住两者不会在某次重构里被合并成一套。
    """
    config = load_config(PROJECT_ROOT / "configs" / "didi_chengdu.yaml")
    miner = config.get("loss.trajectory")
    assert miner["strict"] is False
    assert (miner["top_k"], miner["beam_width"]) == (2, 8)
    assert config.get("evaluation.strict_decode") is True
    assert (config.get("evaluation.top_k"), config.get("evaluation.beam_width")) == (2, 3)


# ---------------------------------------------------------------------------
# 配错要报错，不能静默降级
# ---------------------------------------------------------------------------
def test_strict_decode_without_multi_is_rejected():
    """strict 只存在于多分支解码器里；配错会静默按 single 跑，必须直接报错。"""
    with pytest.raises(ValueError, match="strict_decode"):
        build_trainer(base_config(strict_decode=True))


def test_unknown_decode_is_rejected():
    with pytest.raises(ValueError, match="decode"):
        build_trainer(base_config(decode="banana"))


def test_decode_is_case_insensitive():
    trainer = build_trainer(base_config(decode="MULTI", strict_decode=True))
    assert trainer.eval_decode == "multi"


# ---------------------------------------------------------------------------
# scripts/evaluate.py 必须读同一组键，否则"最终测试"还是另一把尺子
# ---------------------------------------------------------------------------
def _evaluate_args(**overrides):
    """构造一个"命令行什么都没显式给"的 args（全 None = 交给 config 决定）。"""
    import argparse

    values = {
        "decode": None,
        "top_k": None,
        "beam_width": None,
        "null_policy": None,
        "filter_dead_branches": None,
        "strict_decode": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_evaluate_script_reads_the_config_ruler():
    """evaluate.py 不显式给参数时，必须落到 config 声明的 strict 2/3。

    否则 `Trainer.validate()` 用 strict 2/3 挑出来的 best.pt，最终报表却按 single
    出数字 —— 这正是本次要修的口径分叉。
    """
    from scripts.evaluate import resolve_decode_ruler

    config = load_config(PROJECT_ROOT / "configs" / "didi_chengdu.yaml")
    args = _evaluate_args()
    resolve_decode_ruler(args, config)

    assert args.decode == "multi"
    assert bool(args.strict_decode) is True
    assert args.top_k == 2
    assert args.beam_width == 3
    assert args.null_policy == "stop"


def test_evaluate_script_defaults_stay_historical():
    """旧 config 没有这些键 -> evaluate.py 落到与改动前完全相同的默认值。"""
    from scripts.evaluate import resolve_decode_ruler

    config = load_config(PROJECT_ROOT / "configs" / "controlled_unweighted.yaml")
    args = _evaluate_args()
    resolve_decode_ruler(args, config)

    assert args.decode == "single"
    assert bool(args.strict_decode) is False
    assert args.top_k == 2
    assert args.beam_width == 64
    assert args.null_policy == "stop"
    assert bool(args.filter_dead_branches) is False


def test_cli_still_overrides_the_config_ruler():
    """诊断时必须能临时切回 single：CLI 显式给了就赢过 config。"""
    from scripts.evaluate import resolve_decode_ruler

    config = load_config(PROJECT_ROOT / "configs" / "didi_chengdu.yaml")
    args = _evaluate_args(decode="single", beam_width=64)
    resolve_decode_ruler(args, config)

    assert args.decode == "single"
    assert args.beam_width == 64
    # decode 被 CLI 改成 single 后，config 里的 strict 就成了非法组合
    args = _evaluate_args(decode="single", strict_decode=False)
    resolve_decode_ruler(args, config)
    assert args.decode == "single"


def test_explicit_cli_single_turns_strict_off_instead_of_erroring():
    """`--decode single` 是诊断用法，要能跑通：single 下 strict 无意义，直接关掉。

    危险的方向是反过来的（以为在用 strict、其实跑了 single），那个由 config 校验挡住。
    """
    from scripts.evaluate import resolve_decode_ruler

    config = load_config(PROJECT_ROOT / "configs" / "didi_chengdu.yaml")
    args = _evaluate_args(decode="single")
    resolve_decode_ruler(args, config)
    assert args.decode == "single"
    assert bool(args.strict_decode) is False


def test_self_contradictory_config_is_rejected():
    """config 里写 strict=true + decode=single 是**配置错误**，必须起不来。

    静默按 single 跑完一整套评测、报表上却没有任何提示 —— 这是最贵的一类错误。
    """
    from scripts.evaluate import resolve_decode_ruler

    broken = Config({"evaluation": {"decode": "single", "strict_decode": True}})
    with pytest.raises(SystemExit, match="strict"):
        resolve_decode_ruler(_evaluate_args(), broken)


def test_config_decode_value_is_validated():
    from scripts.evaluate import resolve_decode_ruler

    broken = Config({"evaluation": {"decode": "banana"}})
    with pytest.raises(SystemExit, match="decode"):
        resolve_decode_ruler(_evaluate_args(), broken)
