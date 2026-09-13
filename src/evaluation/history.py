"""训练历史的读取、epoch 校准与曲线对齐。

``history.json`` 有两个会让报告出错的坑，集中在这里处理：

1. **记录里可能没有 ``epoch`` 字段**（老版本 trainer 不写）。这时用 ``index + 1``
   当 epoch 号只在"history 完整"时才对。基线 ``v2_controlled_100ep`` 的 history
   只剩 epoch 21-100 的 80 条，于是 best validation 被报成 epoch 60（真实是 80）。
   真实 epoch 只能从 ``val_records_epoch{N}.json`` 的文件名锚定，并用内容里的
   goal_hit 逐个核对 —— :func:`reconstruct_epoch_offset` 只在核对通过时才给偏移，
   对不上就返回 ``None``，不猜。

2. **两次 run 的对比必须按真实 epoch 对齐**。:func:`aligned_curves` 把偏移应用好，
   :func:`matched_values` 再按共同 epoch 取交集，避免"拿 A 的 epoch 5 比 B 的
   epoch 25"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CURVE_KEYS = (
    "train_loss",
    "train_x0_acc",
    "goal_hit_rate",
    "optimal_path_rate",
    "success_cost_ratio",
    "loop_rate",
    "broken_rate",
    "val_x0_acc",
)


def load_history(run_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(run_dir) / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"no history.json in {run_dir}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _val_file_goal_hit(path: Path) -> float:
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not records:
        raise ValueError(f"empty val records: {path}")
    return sum(1 for record in records if record["goal_hit"]) / len(records)


def _val_files(run_dir: Path) -> List[Path]:
    return sorted(
        Path(run_dir).glob("val_records_epoch*.json"),
        key=lambda path: int(path.stem.replace("val_records_epoch", "")),
    )


def reconstruct_epoch_offset(
    run_dir: str | Path, history: Sequence[Dict[str, Any]]
) -> Optional[int]:
    """从 ``val_records_epoch{N}.json`` 找回老 history 缺失的 epoch 偏移。

    做法是从**末尾**对齐：history 里第 j 条带验证的记录，对应最后
    (history 中验证记录条数) 个 val 文件里的第 j 个，并逐个核对 goal_hit。
    返回 ``epoch = index + 1 + offset`` 里的 offset；无法核对时返回 ``None``。
    """
    positions = [i for i, record in enumerate(history) if "goal_hit_rate" in record]
    if not positions:
        # 完全没有验证记录（例如只跑了几个 epoch 就中断）。如果每条记录自带
        # epoch 字段，history 是自描述的，偏移量按定义为 0。
        if history and all("epoch" in record for record in history):
            return 0
        return None
    files = _val_files(run_dir)
    if len(files) < len(positions):
        return None
    files = files[-len(positions):]
    for position, path in zip(positions, files):
        expected = float(history[position]["goal_hit_rate"])
        try:
            actual = _val_file_goal_hit(path)
        except (ValueError, KeyError, json.JSONDecodeError):
            return None
        if abs(expected - actual) > 1e-9:
            return None
    first_epoch = int(files[0].stem.replace("val_records_epoch", ""))
    return first_epoch - (positions[0] + 1)


def curve(
    history: Sequence[Dict[str, Any]], key: str, epoch_offset: int = 0
) -> List[Dict[str, float]]:
    """抽一条 ``[{epoch, value}]`` 曲线（过滤 NaN）。"""
    out: List[Dict[str, float]] = []
    for index, record in enumerate(history, start=1):
        value = record.get(key)
        if value is None or value != value:  # None 或 NaN
            continue
        out.append(
            {
                "epoch": int(record.get("epoch", index + epoch_offset)),
                "value": float(value),
            }
        )
    return out


def aligned_curves(
    run_dir: str | Path,
    keys: Iterable[str] = CURVE_KEYS,
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, List[Dict[str, float]]], Optional[int]]:
    """读一个 run 的所有曲线，并返回 ``(curves, epoch_offset)``。"""
    run_dir = Path(run_dir)
    history = list(history) if history is not None else load_history(run_dir)
    offset = reconstruct_epoch_offset(run_dir, history)
    curves = {key: curve(history, key, offset or 0) for key in keys}
    return curves, offset


def validation_records(
    history: Sequence[Dict[str, Any]], epoch_offset: int = 0
) -> List[Tuple[int, Dict[str, Any]]]:
    """``[(epoch, record)]``，只含带验证指标的记录（用 enumerate 定位，避免
    ``list.index`` 在两条记录内容完全相同时指错）。"""
    return [
        (int(record.get("epoch", index + 1 + epoch_offset)), record)
        for index, record in enumerate(history)
        if "goal_hit_rate" in record
    ]


def matched_values(
    curves_a: Dict[str, List[Dict[str, float]]],
    curves_b: Dict[str, List[Dict[str, float]]],
    key: str,
) -> List[Tuple[int, float, float]]:
    """按共同 epoch 对齐两条曲线，返回 ``[(epoch, a, b)]``。"""
    a = {int(point["epoch"]): float(point["value"]) for point in curves_a.get(key, [])}
    b = {int(point["epoch"]): float(point["value"]) for point in curves_b.get(key, [])}
    return [(epoch, a[epoch], b[epoch]) for epoch in sorted(set(a) & set(b))]


__all__ = [
    "CURVE_KEYS",
    "aligned_curves",
    "curve",
    "load_history",
    "matched_values",
    "reconstruct_epoch_offset",
    "validation_records",
]
