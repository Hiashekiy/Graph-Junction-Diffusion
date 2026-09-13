"""Small neural-network building blocks shared by the denoiser."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn


def get_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"unknown activation {name!r}")


def get_normalization(name: str, dim: int) -> nn.Module:
    name = (name or "layernorm").lower()
    if name == "layernorm":
        return nn.LayerNorm(dim)
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"unknown normalization {name!r}")


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    activation: str = "silu",
    normalization: str = "layernorm",
    dropout: float = 0.0,
    normalize_output: bool = False,
) -> nn.Sequential:
    """Stack of `num_layers` linear layers (so `num_layers=1` == a single Linear)."""
    num_layers = max(1, int(num_layers))
    layers: List[nn.Module] = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == num_layers - 1
        if not is_last:
            if normalization not in ("none", "identity", None):
                layers.append(get_normalization(normalization, dims[i + 1]))
            layers.append(get_activation(activation))
            # NOTE: the Dropout module is always inserted, even for p = 0, so that
            # the positional indices inside the Sequential (and therefore the
            # state_dict keys) do not depend on the dropout rate.  This keeps
            # checkpoints interchangeable between dropout = 0 and dropout > 0.
            layers.append(nn.Dropout(dropout))
        elif normalize_output and normalization not in ("none", "identity", None):
            layers.append(get_normalization(normalization, dims[i + 1]))
    return nn.Sequential(*layers)
