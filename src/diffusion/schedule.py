"""Noise schedule for the categorical forward diffusion (guide section 15).

    alpha_t     = 1 - beta_t
    alpha_bar_t = prod_{tau<=t} alpha_tau,      alpha_bar_0 = 1

The schedule is deliberately *not* baked into the diffusion class: it is a
configurable object, so beta schedules can be swapped without touching the
diffusion maths.
"""

from __future__ import annotations

import math
from typing import Union

import torch


class NoiseSchedule:
    """Discrete beta schedule for T steps (index 1..T; index 0 is the clean state)."""

    def __init__(
        self,
        T: int = 50,
        schedule: str = "linear",
        beta_start: float = 0.02,
        beta_end: float = 0.20,
        cosine_s: float = 0.008,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if T <= 0:
            raise ValueError("T must be positive")
        self.T = int(T)
        self.schedule_name = schedule
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.cosine_s = float(cosine_s)

        beta = self._build_beta(dtype, device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        # index 0 == clean state
        self.beta = beta
        self.alpha = alpha
        self.alpha_bar = torch.cat(
            [torch.ones(1, dtype=dtype, device=device), alpha_bar], dim=0
        )

    # ------------------------------------------------------------------
    def _build_beta(self, dtype: torch.dtype, device) -> torch.Tensor:
        T = self.T
        if self.schedule_name == "linear":
            return torch.linspace(
                self.beta_start, self.beta_end, T, dtype=dtype, device=device
            )
        if self.schedule_name == "cosine":
            steps = torch.arange(T + 1, dtype=dtype, device=device) / T
            f = torch.cos((steps + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            beta = torch.clamp(1.0 - alpha_bar[1:] / alpha_bar[:-1], max=0.999)
            return beta
        if self.schedule_name == "sqrt":
            return torch.linspace(
                self.beta_start**0.5, self.beta_end**0.5, T, dtype=dtype, device=device
            ) ** 2
        raise ValueError(f"unknown schedule {self.schedule_name!r}")

    # ------------------------------------------------------------------
    def to(self, device: Union[str, torch.device]) -> "NoiseSchedule":
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    @property
    def betas(self) -> torch.Tensor:
        return self.beta

    @property
    def alphas(self) -> torch.Tensor:
        return self.alpha

    @property
    def alpha_bars(self) -> torch.Tensor:
        return self.alpha_bar

    # ------------------------------------------------------------------
    def beta_at(self, t) -> torch.Tensor:
        """beta_t for t in 1..T (accepts int, 0-dim or tensor input)."""
        t = torch.as_tensor(t, device=self.beta.device)
        return self.beta[(t.long() - 1)]

    def alpha_at(self, t) -> torch.Tensor:
        t = torch.as_tensor(t, device=self.alpha.device)
        return self.alpha[(t.long() - 1)]

    def alpha_bar_at(self, t) -> torch.Tensor:
        """alpha_bar_t for t in 0..T; alpha_bar_at(0) == 1."""
        t = torch.as_tensor(t, device=self.alpha_bar.device)
        return self.alpha_bar[t.long()]

    # ------------------------------------------------------------------
    def terminal_alpha_bar(self) -> float:
        return float(self.alpha_bar[self.T].item())

    def validate(self, atol: float = 0.02) -> None:
        """alpha_bar_T must be close to zero so that z_T ~ pi."""
        value = self.terminal_alpha_bar()
        if value > atol:
            raise ValueError(
                f"alpha_bar_T = {value:.5f} is not close to 0; increase T or beta_end"
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"NoiseSchedule(T={self.T}, schedule={self.schedule_name!r}, "
            f"beta=[{self.beta_start}, {self.beta_end}], "
            f"alpha_bar_T={self.terminal_alpha_bar():.3e})"
        )
