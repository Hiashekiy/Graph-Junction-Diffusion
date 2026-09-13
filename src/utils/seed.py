"""Random seed utilities.

Every stochastic process in this project must be seedable (guide section 3).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (cpu + cuda)."""
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_rng_state() -> dict:
    """Snapshot every RNG the trainer touches (for checkpoint['rng_state'])."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_byte_tensor(state, device: str = "cpu") -> torch.Tensor:
    """torch.set_rng_state 要求 CPU 上的 ByteTensor。

    ``torch.load(..., map_location='cuda')`` 会把 checkpoint 里的 RNG state 也搬到
    CUDA 上，直接喂给 ``torch.set_rng_state`` 会报
    ``TypeError: RNG state must be a torch.ByteTensor``。
    """
    tensor = state if isinstance(state, torch.Tensor) else torch.tensor(state)
    return tensor.to(device=device, dtype=torch.uint8)


def set_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        # CUDA 的 RNG state 本身是 CPU 上的 ByteTensor；torch.load(map_location='cuda')
        # 会把它搬到 GPU，所以这里显式搬回 CPU。
        torch.cuda.set_rng_state_all([_as_byte_tensor(item) for item in state["cuda"]])


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g
