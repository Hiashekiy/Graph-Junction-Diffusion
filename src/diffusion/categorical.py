"""Non-absorbing categorical forward diffusion (guide sections 14-19 / doc 03).

One categorical variable per decision node.  With uniform base noise

    pi_i(c) = 1 / C_i

the single-step transition is

    Q_t^i = (1 - beta_t) I + beta_t 1 pi_i^T
          = alpha_t I + (1 - alpha_t) 1 pi_i^T

and the cumulative transition has the closed form

    Qbar_t^i = alpha_bar_t I + (1 - alpha_bar_t) 1 pi_i^T

so the training-time corruption is a single categorical draw

    z_t^i ~ Cat(e_{z_0^i} Qbar_t^i)

which is realised (without materialising any C_i x C_i matrix) as

    with probability alpha_bar_t keep z_0^i,
    otherwise resample uniformly from the candidate group of decision node i.

Note that the resampled class may coincide with the original one, so the
resulting keep-probability is  alpha_bar_t + (1 - alpha_bar_t)/C_i  and not
simply alpha_bar_t (guide section 16).

All state tensors use the *flattened candidate table* index space.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from src.diffusion.posterior import (
    one_hot_dense,
    q_t_matrix,
    qbar_t_matrix,
    reverse_posterior,
)
from src.diffusion.schedule import NoiseSchedule

_EPS = 1e-12


class CategoricalDiffusion:
    """Forward categorical corruption + exact reverse posterior machinery."""

    def __init__(self, schedule: NoiseSchedule, base_noise: str = "uniform"):
        if base_noise != "uniform":
            raise NotImplementedError(
                f"base_noise={base_noise!r} is not supported in V1 (only 'uniform')"
            )
        self.schedule = schedule
        self.base_noise = base_noise
        self.T = schedule.T

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def group_sizes(num_decisions: int, candidate_owner: Tensor) -> Tensor:
        return torch.bincount(
            candidate_owner, minlength=num_decisions
        ).to(candidate_owner.device)

    @staticmethod
    def group_starts(sizes: Tensor) -> Tensor:
        return torch.cumsum(sizes, dim=0) - sizes

    @staticmethod
    def group_positions(candidate_owner: Tensor, sizes: Tensor, starts: Tensor) -> Tensor:
        """Flat candidate index -> position inside its own group, shape [C]."""
        return torch.arange(candidate_owner.numel(), device=candidate_owner.device) - starts[
            candidate_owner
        ]

    @staticmethod
    def _rand(
        size: int, device: torch.device, generator: Optional[torch.Generator]
    ) -> Tensor:
        """torch.rand that tolerates a generator living on another device."""
        if generator is not None and generator.device.type != device.type:
            return torch.rand(size, device=generator.device, generator=generator).to(device)
        return torch.rand(size, device=device, generator=generator)

    def _uniform_group_draw(
        self,
        sizes: Tensor,
        starts: Tensor,
        num_decisions: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """One uniform category per decision node, returned as flat indices."""
        rand = self._rand(num_decisions, sizes.device, generator)
        local = torch.floor(rand * sizes.to(rand.dtype)).long()
        local = torch.clamp(local, max=(sizes - 1).clamp_min(0))
        return starts + local

    def to_dense(
        self,
        flat: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        max_group: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Scatter a flat [C] vector into a dense [M, C_max] matrix plus its mask."""
        sizes = self.group_sizes(num_decisions, candidate_owner)
        starts = self.group_starts(sizes)
        if max_group is None:
            max_group = int(sizes.max().item())
        positions = self.group_positions(candidate_owner, sizes, starts)
        mask = (
            torch.arange(max_group, device=flat.device)[None, :]
            < sizes[:, None]
        )
        dense = torch.zeros(
            num_decisions, max_group, dtype=flat.dtype, device=flat.device
        )
        dense[candidate_owner, positions] = flat
        return dense, mask

    def dense_from_flat_prob(
        self,
        prob_flat: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        max_group: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.to_dense(prob_flat, candidate_owner, num_decisions, max_group)

    def flat_from_dense(self, dense: Tensor, mask: Tensor) -> Tensor:
        return dense[mask]

    # ------------------------------------------------------------------
    # forward process
    # ------------------------------------------------------------------
    def sample_xt(
        self,
        target_candidate: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        alpha_bar_t: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample z_t ~ q(z_t | z_0) for every decision node.

        Args:
            target_candidate: flat candidate index of z_0, shape [M]
            candidate_owner:  flat candidate -> decision owner, shape [C]
            num_decisions:    M
            alpha_bar_t:      per decision alpha_bar_t, shape [M]
        Returns:
            z_t as flat candidate indices, shape [M]
        """
        sizes = self.group_sizes(num_decisions, candidate_owner)
        starts = self.group_starts(sizes)

        # schedule 默认建在 CPU 上，而 target_candidate 可能在 CUDA 上
        alpha_bar_t = torch.as_tensor(alpha_bar_t).to(target_candidate.device)
        if alpha_bar_t.dim() == 0:
            alpha_bar_t = alpha_bar_t.expand(num_decisions)

        keep = self._rand(num_decisions, target_candidate.device, generator) < alpha_bar_t
        resampled = self._uniform_group_draw(sizes, starts, num_decisions, generator)
        return torch.where(keep, target_candidate, resampled)

    def sample_xt_at_time(
        self,
        target_candidate: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        t: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Convenience wrapper taking an integer timestep instead of alpha_bar."""
        alpha_bar_t = self.schedule.alpha_bar_at(t)
        return self.sample_xt(
            target_candidate, candidate_owner, num_decisions, alpha_bar_t, generator
        )

    # ------------------------------------------------------------------
    # V2: coherent forward Markov trajectory (实施指南第 19 节)
    # ------------------------------------------------------------------
    def sample_forward_step(
        self,
        z_prev: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        t,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """单步前向转移 z_{t-1} -> z_t。

            z_t = z_{t-1}            prob alpha_t
            z_t ~ Cat(pi_i)          prob 1 - alpha_t

        注意用的是 **alpha_t**（单步），不是 alpha_bar_t。这样生成的
        ``z_0 -> z_1 -> ... -> z_T`` 是一条真正自洽的 forward Markov trajectory。
        """
        alpha_t = self.schedule.alpha_at(t)
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.expand(num_decisions)
        alpha_t = alpha_t.to(candidate_owner.device)

        sizes = self.group_sizes(num_decisions, candidate_owner)
        starts = self.group_starts(sizes)

        keep = self._rand(num_decisions, alpha_t.device, generator) < alpha_t
        resampled = self._uniform_group_draw(sizes, starts, num_decisions, generator)
        return torch.where(keep, z_prev, resampled)

    def sample_forward_trajectory(
        self,
        target_candidate: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        generator: Optional[torch.Generator] = None,
        max_steps: Optional[int] = None,
    ) -> Tensor:
        """完整前向加噪轨迹 ``[T+1, M]``，第 0 行是 z_0（= target_candidate）。"""
        steps = int(self.T if max_steps is None else max_steps)
        if steps > self.T:
            raise ValueError(f"max_steps={steps} exceeds the schedule T={self.T}")
        trajectory = [target_candidate]
        current = target_candidate
        for t in range(1, steps + 1):
            current = self.sample_forward_step(
                current, candidate_owner, num_decisions, t, generator
            )
            trajectory.append(current)
        return torch.stack(trajectory, dim=0)

    # ------------------------------------------------------------------
    # reverse posterior
    # ------------------------------------------------------------------
    def _alpha_tensors(self, t, num_decisions: int, device=None):
        """alpha_t / alpha_bar_t / alpha_bar_{t-1} broadcast to [M].

        ``device`` 非空时把 schedule 上的常量搬到与输入张量相同的设备
        （schedule 默认在 CPU 上构建，而 batch 可能在 CUDA 上）。
        """
        t = torch.as_tensor(t)
        alpha_t = self.schedule.alpha_at(t)
        alpha_bar_t = self.schedule.alpha_bar_at(t)
        alpha_bar_prev = self.schedule.alpha_bar_at(t - 1)
        if device is not None:
            device = torch.device(device)
            alpha_t = alpha_t.to(device)
            alpha_bar_t = alpha_bar_t.to(device)
            alpha_bar_prev = alpha_bar_prev.to(device)
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.expand(num_decisions)
            alpha_bar_t = alpha_bar_t.expand(num_decisions)
            alpha_bar_prev = alpha_bar_prev.expand(num_decisions)
        return alpha_t, alpha_bar_t, alpha_bar_prev

    def _reverse_core(
        self,
        p0_dense: Tensor,
        mask: Tensor,
        sizes: Tensor,
        zt_local: Tensor,
        alpha_t: Tensor,
        alpha_bar_t: Tensor,
        alpha_bar_prev: Tensor,
    ) -> Tensor:
        """p_theta(z_{t-1}=a | z_t) for every decision node, shape [M, C_max]."""
        return reverse_posterior(
            p0_dense, mask, sizes, zt_local, alpha_t, alpha_bar_t, alpha_bar_prev
        )

    def true_posterior(
        self,
        z0_candidate: Tensor,
        zt_candidate: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        t: Tensor,
        max_group: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Exact q(z_{t-1} | z_t, z_0) as a dense [M, C_max] matrix + mask."""
        sizes = self.group_sizes(num_decisions, candidate_owner)
        starts = self.group_starts(sizes)
        positions = self.group_positions(candidate_owner, sizes, starts)
        if max_group is None:
            max_group = int(sizes.max().item())

        # build the one-hot clean distribution directly in dense form
        mask = (
            torch.arange(max_group, device=z0_candidate.device)[None, :] < sizes[:, None]
        )
        p0_dense = one_hot_dense(
            positions[z0_candidate],
            num_decisions,
            max_group,
            z0_candidate.device,
            dtype=torch.float32,
        )

        zt_local = positions[zt_candidate]
        alpha_t, alpha_bar_t, alpha_bar_prev = self._alpha_tensors(
            t, num_decisions, device=p0_dense.device
        )
        return (
            self._reverse_core(
                p0_dense,
                mask,
                sizes,
                zt_local,
                alpha_t,
                alpha_bar_t,
                alpha_bar_prev,
            ),
            mask,
        )

    def model_reverse_posterior(
        self,
        clean_prob_dense: Tensor,
        mask: Tensor,
        zt_candidate: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        t: Tensor,
    ) -> Tensor:
        """p_theta(z_{t-1} | z_t) obtained by marginalising the predicted z_0."""
        sizes = self.group_sizes(num_decisions, candidate_owner)
        starts = self.group_starts(sizes)
        positions = self.group_positions(candidate_owner, sizes, starts)
        zt_local = positions[zt_candidate]
        alpha_t, alpha_bar_t, alpha_bar_prev = self._alpha_tensors(
            t, num_decisions, device=clean_prob_dense.device
        )
        return self._reverse_core(
            clean_prob_dense,
            mask,
            sizes,
            zt_local,
            alpha_t,
            alpha_bar_t,
            alpha_bar_prev,
        )

    # ------------------------------------------------------------------
    def sample_prev(
        self,
        reverse_prob_dense: Tensor,
        mask: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Categorical sample z_{t-1} from a dense [M, C_max] posterior."""
        sizes = self.group_sizes(num_decisions, candidate_owner)
        starts = self.group_starts(sizes)
        probs = reverse_prob_dense.masked_fill(~mask, 0.0)
        if generator is not None and generator.device.type != probs.device.type:
            local = (
                torch.multinomial(
                    probs.to(generator.device), num_samples=1, generator=generator
                )
                .squeeze(1)
                .to(probs.device)
            )
        else:
            local = torch.multinomial(
                probs, num_samples=1, generator=generator
            ).squeeze(1)
        return starts + local

    def sample_prev_deterministic(
        self,
        reverse_prob_dense: Tensor,
        mask: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
    ) -> Tensor:
        """z_{t-1} = argmax posterior（评测时可复现的确定性采样）。"""
        sizes = self.group_sizes(num_decisions, candidate_owner)
        starts = self.group_starts(sizes)
        probs = reverse_prob_dense.masked_fill(~mask, -1.0)
        local = probs.argmax(dim=1)
        return starts + local

    def model_reverse_posterior_dense(
        self,
        clean_prob_flat: Tensor,
        zt_candidate: Tensor,
        candidate_owner: Tensor,
        num_decisions: int,
        t,
    ) -> Tensor:
        """便利封装：flat 概率 -> dense -> posterior，返回 [M, C_max]。"""
        dense, mask = self.dense_from_flat_prob(
            clean_prob_flat, candidate_owner, num_decisions
        )
        return self.model_reverse_posterior(
            dense, mask, zt_candidate, candidate_owner, num_decisions, t
        )

    # ------------------------------------------------------------------
    # reference full matrices (used by the unit tests and for debugging)
    # ------------------------------------------------------------------
    def transition_matrix(self, num_categories: int, t: int, dtype=torch.float64) -> Tensor:
        """Q_t = alpha_t I + (1 - alpha_t) 1 pi^T for a node with C categories."""
        C = int(num_categories)
        beta_t = self.schedule.beta[t - 1].to(dtype)
        pi = torch.full((C,), 1.0 / C, dtype=dtype)
        return (1.0 - beta_t) * torch.eye(C, dtype=dtype) + beta_t * torch.ones(
            C, 1, dtype=dtype
        ) @ pi[None, :]

    def cumulative_transition_matrix(
        self, num_categories: int, t: int, dtype=torch.float64
    ) -> Tensor:
        """Closed form Qbar_t = alpha_bar_t I + (1 - alpha_bar_t) 1 pi^T."""
        C = int(num_categories)
        ab = self.schedule.alpha_bar[t].to(dtype)
        pi = torch.full((C,), 1.0 / C, dtype=dtype)
        return ab * torch.eye(C, dtype=dtype) + (1.0 - ab) * torch.ones(
            C, 1, dtype=dtype
        ) @ pi[None, :]

    def cumulative_transition_matrix_product(
        self, num_categories: int, t: int, dtype=torch.float64
    ) -> Tensor:
        """Brute force Q_1 Q_2 ... Q_t (reference implementation for tests)."""
        out = torch.eye(int(num_categories), dtype=dtype)
        for step in range(1, t + 1):
            out = out @ self.transition_matrix(num_categories, step, dtype)
        return out
