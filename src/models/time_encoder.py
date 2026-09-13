"""Sinusoidal timestep encoding + MLP (guide section 26 / doc 02 section 13)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.utils.nn import build_mlp


def sinusoidal_encoding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Standard transformer-style encoding of integer timesteps, shape [..., dim]."""
    if dim % 2 != 0:
        raise ValueError("sinusoidal encoding dimension must be even")
    half = dim // 2
    device = t.device
    exponent = -math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32)
    exponent = exponent / half
    freqs = torch.exp(exponent)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        activation: str = "silu",
        normalization: str = "layernorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.mlp = build_mlp(
            d_model,
            d_model,
            d_model,
            num_layers=num_layers,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
        )

    def forward(self, t) -> Tensor:
        """t: int / 0-dim tensor / [B] tensor -> [B, d_model] (or [d_model])."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        scalar = t.dim() == 0
        if scalar:
            t = t.unsqueeze(0)
        encoding = sinusoidal_encoding(t.long(), self.d_model)
        out = self.mlp(encoding.to(next(self.parameters()).dtype))
        return out.squeeze(0) if scalar else out


# ---------------------------------------------------------------------------
# V2: timestep conditioning helpers
# ---------------------------------------------------------------------------
def broadcast_time(tau_graph: Tensor, graph_node_ptr: Tensor) -> Tensor:
    """每张图一个 tau -> 每个节点一个 tau。

    Args:
        tau_graph:      [B, d]（每个图当前 timestep 的 embedding）
        graph_node_ptr: [B+1] 节点分图指针
    Returns:
        [N, d]，节点 v 取到它所属图的 tau
    """
    sizes = graph_node_ptr[1:] - graph_node_ptr[:-1]
    return torch.repeat_interleave(tau_graph, sizes, dim=0)


def broadcast_time_to_decisions(tau_graph: Tensor, decision_graph_id: Tensor) -> Tensor:
    """每张图一个 tau -> 每个 decision 一个 tau（[M, d]）。"""
    return tau_graph[decision_graph_id]


class TimeConditioner(nn.Module):
    """tau_t -> (gamma_t, beta_t)（AdaLN / FiLM conditioning，指南第 10.2 节）。

        (gamma, beta) = MLP_cond(tau_t)
        h_hat = (1 + gamma) * LN(h) + beta
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        hidden = int(hidden_dim or d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * self.d_model),
        )

    def forward(self, tau) -> tuple[Tensor, Tensor]:
        """tau: [..., d] -> gamma, beta: [..., d]."""
        out = self.mlp(tau)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta
