"""``tools/summarize_run.py`` 的 epoch 校准测试。

背景：老版本 trainer 不往 history 记录里写 ``epoch``，而 ``history.json`` 可能只保留
了后一段（`v2_controlled_100ep` 只剩 epoch 21-100 的 80 条）。此时 ``index + 1``
会凭空少算 20 个 epoch，报出来的 "best validation: epoch 60" 其实是 epoch 80。

校准锚点是 ``val_records_epoch{N}.json`` 的文件名 + 内容里的 goal_hit。这里用假数据
目录测：

* 能对齐时返回正确偏移量；
* 值对不上时返回 None（不猜）；
* 记录里本来就有 ``epoch`` 字段时不受影响。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / "outputs" / "_pytest_summarize"


def _load_module():
    path = PROJECT_ROOT / "tools" / "summarize_run.py"
    spec = importlib.util.spec_from_file_location("_summarize_run_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def module():
    return _load_module()


def _write_val(run_dir: Path, epoch: int, goal_hits: int, total: int = 4) -> float:
    records = [
        {"goal_hit": index < goal_hits, "status": "goal" if index < goal_hits else "broken"}
        for index in range(total)
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"val_records_epoch{epoch}.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    return goal_hits / total


@pytest.fixture()
def run_dir():
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    yield TMP_DIR
    shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_reconstructs_offset_from_val_record_files(module, run_dir):
    """history 只剩 epoch 21-100：验证点每 5 个 epoch 一次。"""
    history = []
    for epoch in range(21, 101):
        record = {"train_loss": 0.5}
        if epoch % 5 == 0:
            value = _write_val(run_dir, epoch, goal_hits=1 + (epoch // 5) % 3)
            record["goal_hit_rate"] = value
        history.append(record)

    offset = module.reconstruct_epoch_offset(run_dir, history)
    assert offset == 20
    assert history[0].get("epoch") is None

    # 曲线上的 epoch 号必须是真实 epoch（第一个验证点是 epoch 25）
    values = module.curve(history, "goal_hit_rate", offset)
    assert values[0]["epoch"] == 25
    # 训练曲线（每条记录都有）也必须整体平移
    losses = module.curve(history, "train_loss", offset)
    assert losses[0]["epoch"] == 21
    assert losses[-1]["epoch"] == 100


def test_returns_none_when_values_do_not_match(module, run_dir):
    history = [{"train_loss": 0.5, "goal_hit_rate": 0.9}]
    _write_val(run_dir, 25, goal_hits=1)  # 0.25 != 0.9
    assert module.reconstruct_epoch_offset(run_dir, history) is None


def test_returns_none_without_val_files(module, run_dir):
    history = [{"train_loss": 0.5, "goal_hit_rate": 0.5}]
    run_dir.mkdir(parents=True, exist_ok=True)
    assert module.reconstruct_epoch_offset(run_dir, history) is None


def test_modern_history_with_epoch_field_is_untouched(module, run_dir):
    """新 run 每条记录都有 epoch 字段：偏移量算出来是 0，epoch 号原样使用。"""
    history = []
    for epoch in range(1, 6):
        record = {"train_loss": 0.5, "epoch": epoch}
        if epoch % 5 == 0:
            record["goal_hit_rate"] = _write_val(run_dir, epoch, goal_hits=2)
        history.append(record)

    offset = module.reconstruct_epoch_offset(run_dir, history)
    assert offset == 0
    assert [item["epoch"] for item in module.curve(history, "train_loss", 0)] == [
        1, 2, 3, 4, 5
    ]
    assert module.curve(history, "goal_hit_rate", 0)[0]["epoch"] == 5
