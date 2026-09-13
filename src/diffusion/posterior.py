"""Reverse posterior of the categorical diffusion (guide section 18 / doc 03 sec. 35-37).

Exact one-step posterior given the clean state:

    q(z_{t-1}^i = a | z_t^i = b, z_0^i = c)
        = Qbar_{t-1}^i[c, a] Q_t^i[a, b] / Qbar_t^i[c, b]

Model posterior, marginalising the denoiser's clean prediction:

    p_theta(z_{t-1}^i = a | z_t)
        = sum_c q(a | z_t^i, z_0^i = c) p_theta(z_0^i = c | G, s, g, z_t, t)

With uniform base noise pi_i(c) = 1 / C_i and

    Qbar_t[c, b] = alpha_bar_t [c == b] + (1 - alpha_bar_t) pi_i(b)
    Q_t[a, b]    = alpha_t     [a == b] + (1 - alpha_t)     pi_i(b)

the sum over c collapses into two cheap factors

    h[a] = alpha_bar_{t-1} p0[a] / Qbar_t[a, b]
           + (1 - alpha_bar_{t-1}) pi_i(a) * sum_c p0[c] / Qbar_t[c, b]
    g[a] = Q_t[a, b]
    p(a) proportional to g[a] h[a]

which is what `reverse_posterior` evaluates.

Both `a` (the previous state) and `c` (the clean state) range over the candidate
group of decision node i, encoded densely as [M, C_max] with a boolean mask.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

_EPS = 1e-12


def uniform_noise_dense(mask: Tensor, sizes: Tensor, dtype: torch.dtype) -> Tensor:
    """pi_i(c) = 1 / C_i, zero outside the group."""
    return mask.to(dtype) / sizes.to(dtype)[:, None]


def q_t_matrix(num_categories: int, alpha_t: Tensor, dtype=torch.float64) -> Tensor:
    """Q_t = alpha_t I + (1 - alpha_t) 1 pi^T."""
    C = int(num_categories)
    pi = torch.full((C,), 1.0 / C, dtype=dtype)
    eye = torch.eye(C, dtype=dtype)
    return alpha_t.to(dtype) * eye + (1.0 - alpha_t.to(dtype)) * torch.ones(
        C, 1, dtype=dtype
    ) @ pi[None, :]


def qbar_t_matrix(num_categories: int, alpha_bar_t: Tensor, dtype=torch.float64) -> Tensor:
    """Qbar_t = alpha_bar_t I + (1 - alpha_bar_t) 1 pi^T."""
    C = int(num_categories)
    pi = torch.full((C,), 1.0 / C, dtype=dtype)
    eye = torch.eye(C, dtype=dtype)
    return alpha_bar_t.to(dtype) * eye + (1.0 - alpha_bar_t.to(dtype)) * torch.ones(
        C, 1, dtype=dtype
    ) @ pi[None, :]


def reverse_posterior(
    p0_dense: Tensor,
    mask: Tensor,
    sizes: Tensor,
    zt_local: Tensor,
    alpha_t: Tensor,
    alpha_bar_t: Tensor,
    alpha_bar_prev: Tensor,
) -> Tensor:
    """p(z_{t-1} = a | z_t) for every decision node, shape [M, C_max].

    Args:
        p0_dense:      [M, C_max] distribution over the clean state
        mask:          [M, C_max] boolean group mask
        sizes:         [M] number of categories per decision node
        zt_local:      [M] local index of the observed z_t category
        alpha_t:       [M] (or broadcastable)
        alpha_bar_t:   [M]
        alpha_bar_prev:[M]  alpha_bar_{t-1} (== 1 at t == 1)
    """
    M, Cmax = p0_dense.shape
    dtype = p0_dense.dtype
    pi = uniform_noise_dense(mask, sizes, dtype)

    onehot_b = (
        torch.arange(Cmax, device=p0_dense.device)[None, :] == zt_local[:, None]
    ).to(dtype)

    pi_b = (pi * onehot_b).sum(dim=1, keepdim=True)  # pi_i(z_t^i)
    denom = (alpha_bar_t[:, None] * onehot_b + (1.0 - alpha_bar_t)[:, None] * pi_b).clamp_min(
        _EPS
    )

    ratio = p0_dense / denom
    h = alpha_bar_prev[:, None] * ratio + (1.0 - alpha_bar_prev)[:, None] * pi * ratio.sum(
        dim=1, keepdim=True
    )
    g = alpha_t[:, None] * onehot_b + (1.0 - alpha_t)[:, None] * pi_b

    posterior = g * h * mask.to(dtype)
    normaliser = posterior.sum(dim=1, keepdim=True).clamp_min(_EPS)
    return posterior / normaliser


def one_hot_dense(
    index: Tensor, num_rows: int, num_cols: int, device, dtype=torch.float32
) -> Tensor:
    out = torch.zeros(num_rows, num_cols, dtype=dtype, device=device)
    out[torch.arange(num_rows, device=device), index] = 1.0
    return out


def true_reverse_posterior(
    z0_local: Tensor,
    zt_local: Tensor,
    mask: Tensor,
    sizes: Tensor,
    alpha_t: Tensor,
    alpha_bar_t: Tensor,
    alpha_bar_prev: Tensor,
) -> Tensor:
    """Exact posterior q(z_{t-1} | z_t, z_0) with a one-hot clean state."""
    num_decisions, max_group = mask.shape
    p0_dense = one_hot_dense(
        z0_local, num_decisions, max_group, mask.device, dtype=torch.float32
    )
    return reverse_posterior(
        p0_dense, mask, sizes, zt_local, alpha_t, alpha_bar_t, alpha_bar_prev
    )
