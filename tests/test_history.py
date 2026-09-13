"""``src/evaluation/history.py`` 的测试（epoch 校准与曲线对齐）。

背景：老版本 trainer 不往 history 记录里写 ``epoch``，而 ``history.json`` 可能只保留
了后一段（`v2_controlled_100ep` 只剩 epoch 21-100 的 80 条）。此时 ``index + 1``
会凭空少算 20 个 epoch，报出来的 "best validation: epoch 60" 其实是 epoch 80。

校准锚点是 ``val_records_epoch{N}.json`` 的文件名 + 内容里的 goal_hit。这里用假数据
目录测：

* 能对齐时返回正确偏移量，曲线 epoch 号整体平移；
* 值对不上时返回 None（不猜）；
* 记录里本来就有 ``epoch`` 字段时不受影响；
* 两个 run 的曲线只在共同 epoch 上比较。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.evaluation.history import (
    aligned_curves,
    curve,
    matched_values,
    reconstruct_epoch_offset,
    validation_records,
)

TMP_DIR = Path("outputs/_pytest_history")
OTHER_DIR = Path("outputs/_pytest_history_b")


@pytest.fixture(autouse=True)
def clean_dirs():
    for path in (TMP_DIR, OTHER_DIR):
        shutil.rmtree(path, ignore_errors=True)
    yield
    for path in (TMP_DIR, OTHER_DIR):
        shutil.rmtree(path, ignore_errors=True)


def _write_val(run_dir: Path, epoch: int, goal_hits: int, total: int = 4) -> float:
    records = [
        {
            "goal_hit": index < goal_hits,
            "status": "goal" if index < goal_hits else "broken",
        }
        for index in range(total)
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"val_records_epoch{epoch}.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    return goal_hits / total


def _legacy_history(run_dir: Path, first_epoch: int, last_epoch: int, step: int = 5):
    """造一份"没有 epoch 字段、只剩后一段"的 history。"""
    history = []
    for epoch in range(first_epoch, last_epoch + 1):
        record = {"train_loss": 0.5 - epoch * 0.001}
        if epoch % step == 0:
            record["goal_hit_rate"] = _write_val(
                run_dir, epoch, goal_hits=1 + (epoch // step) % 3
            )
        history.append(record)
    return history


def test_reconstructs_offset_from_val_record_files():
    history = _legacy_history(TMP_DIR, 21, 100)
    offset = reconstruct_epoch_offset(TMP_DIR, history)
    assert offset == 20

    losses = curve(history, "train_loss", offset)
    assert losses[0]["epoch"] == 21
    assert losses[-1]["epoch"] == 100
    assert curve(history, "goal_hit_rate", offset)[0]["epoch"] == 25


def test_returns_none_when_values_do_not_match():
    history = [{"train_loss": 0.5, "goal_hit_rate": 0.9}]
    _write_val(TMP_DIR, 25, goal_hits=1)  # 0.25 != 0.9
    assert reconstruct_epoch_offset(TMP_DIR, history) is None


def test_returns_none_without_val_files():
    history = [{"train_loss": 0.5, "goal_hit_rate": 0.5}]
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    assert reconstruct_epoch_offset(TMP_DIR, history) is None


def test_returns_none_when_there_are_fewer_files_than_checks():
    history = _legacy_history(TMP_DIR, 21, 100)
    # 删掉一半 val 文件：不能靠"猜"补齐，必须返回 None
    for path in sorted(TMP_DIR.glob("val_records_epoch*.json"))[:10]:
        path.unlink()
    assert reconstruct_epoch_offset(TMP_DIR, history) is None


def test_modern_history_with_epoch_field_is_untouched():
    history = []
    for epoch in range(1, 6):
        record = {"train_loss": 0.5, "epoch": epoch}
        if epoch % 5 == 0:
            record["goal_hit_rate"] = _write_val(TMP_DIR, epoch, goal_hits=2)
        history.append(record)

    assert reconstruct_epoch_offset(TMP_DIR, history) == 0
    assert [point["epoch"] for point in curve(history, "train_loss", 0)] == [
        1, 2, 3, 4, 5
    ]


def test_validation_records_use_enumerate_not_index():
    """两条内容完全相同的验证记录都要能正确落位（list.index 会指到前一条）。"""
    duplicate = {"train_loss": 0.4, "goal_hit_rate": 0.5}
    history = [dict(duplicate), {"train_loss": 0.3, "epoch": 7}, dict(duplicate)]
    checks = validation_records(history, epoch_offset=0)
    assert [epoch for epoch, _ in checks] == [1, 3]


def test_aligned_curves_and_matched_values_on_shifted_runs():
    """A 只剩后半段（偏移 20）、B 完整：只在共同 epoch 上比较。"""
    history_a = _legacy_history(TMP_DIR, 21, 30)
    history_b = []
    for epoch in range(1, 26):
        record = {"train_loss": 0.9 - epoch * 0.01, "epoch": epoch}
        history_b.append(record)
    OTHER_DIR.mkdir(parents=True, exist_ok=True)
    (OTHER_DIR / "history.json").write_text(json.dumps(history_b), encoding="utf-8")

    curves_a, offset_a = aligned_curves(TMP_DIR, keys=("train_loss",), history=history_a)
    curves_b, offset_b = aligned_curves(OTHER_DIR, keys=("train_loss",), history=history_b)
    assert offset_a == 20 and offset_b == 0

    pairs = matched_values(curves_a, curves_b, "train_loss")
    assert [epoch for epoch, _, _ in pairs] == list(range(21, 26))
    for epoch, value_a, value_b in pairs:
        assert value_a == pytest.approx(0.5 - epoch * 0.001)
        assert value_b == pytest.approx(0.9 - epoch * 0.01)
