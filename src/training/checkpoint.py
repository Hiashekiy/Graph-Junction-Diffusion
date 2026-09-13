"""Checkpoint save / load (实施指南第 1 节).

保存内容刻意保持最小：

    model / optimizer / scheduler 的 state_dict
    epoch、全局 step、best metric
    模型与 diffusion 的构建配置
    RNG 状态（可复现续训）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from src.utils.seed import get_rng_state, set_rng_state


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: Optional[float] = None,
    model_config: Optional[Dict[str, Any]] = None,
    diffusion_config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "model_config": dict(model_config or {}),
        "diffusion_config": dict(diffusion_config or {}),
        "rng_state": get_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng and "rng_state" in payload:
        set_rng_state(payload["rng_state"])
    return payload


def latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Optional[Path]:
    directory = Path(directory)
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None
