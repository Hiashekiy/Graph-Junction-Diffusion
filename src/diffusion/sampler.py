"""Reverse Sampler (实施指南第 17 节 / 设计报告 V2.1 第 16 节).

V2 与旧 sampler 最大的差异是状态本身：

    旧 : z_t
    新 : (H_t, z_t)

完整 reverse loop：

    H_t = model.init_nodes(batch)          # H_T = TypeEmbedding(V)，只做一次
    z_t = diffusion.sample_prior(batch)    # z_T ~ pi

    for t in range(T, 0, -1):
        out    = model.step(batch, H_t, z_t, t)
        p_prev = model_reverse_posterior(out.candidate_prob, z_t, t)
        z_prev = sample_prev(p_prev)
        H_t    = out.H_next                # ← persistent state 直接进下一步
        z_t    = z_prev

最关键的一行是 ``H_t = out.H_next``：**绝不允许**下一轮重新
``H_t = model.init_nodes(batch)``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from src.diffusion.categorical import CategoricalDiffusion
from src.models.denoiser import GraphFlowDenoiser


@dataclass
class ReverseTrace:
    """一次 reverse chain 的记录（主要用于测试与调试）。"""

    H_path: List[Tensor] = field(default_factory=list)   # H_T ... H_0
    z_path: List[Tensor] = field(default_factory=list)   # z_T ... z_0
    log_prob: List[Tensor] = field(default_factory=list)

    def append(self, H: Tensor, z: Tensor, log_prob: Optional[Tensor] = None) -> None:
        self.H_path.append(H)
        self.z_path.append(z)
        if log_prob is not None:
            self.log_prob.append(log_prob)


# ---------------------------------------------------------------------------
# prior
# ---------------------------------------------------------------------------
def sample_prior(
    diffusion: CategoricalDiffusion,
    candidate_owner: Tensor,
    num_decisions: int,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """z_T ~ pi：每个 decision 在自己的 candidate 组里均匀取一个（flat index）。"""
    sizes = diffusion.group_sizes(num_decisions, candidate_owner)
    starts = diffusion.group_starts(sizes)
    return diffusion._uniform_group_draw(
        sizes, starts, num_decisions, generator
    )


def heuristic_prior_deterministic(
    candidate_owner: Tensor,
    candidate_is_null: Tensor,
    num_decisions: int,
) -> Tensor:
    """**启发式**初始状态，不是 ``z_T ~ pi`` 的确定性等价物。

    规则：普通 Junction 取 NULL，Source 取第一条 branch。

    修改清单 P2-1 明确要求把语义写清楚：``z_T ~ pi`` 的定义是"每个 decision
    在自己的 candidate 组里均匀随机取一个"，而这里是一个明显偏向 NULL 的人造
    初值，因此：

    * 想复现实验（要的是"同样输入给同样输出"）→ 仍然用 ``z_T ~ pi``，但传入
      固定 seed 的 ``torch.Generator``，数学定义不变；
    * ``stochastic=False`` 时的这个函数只用于"绕开随机性的确定性对照"，
      不要把它当成 diffusion 的 deterministic 版本。
    """
    first = torch.full(
        (num_decisions,), candidate_owner.numel(), dtype=torch.long,
        device=candidate_owner.device,
    )
    first = first.scatter_reduce(
        0, candidate_owner, torch.arange(
            candidate_owner.numel(), device=candidate_owner.device, dtype=torch.long
        ), reduce="amin", include_self=True,
    )
    # 注意初值必须是"比任何真实候选编号都大"的哨兵：用 -1 的话 amin 会永远取 -1，
    # first 会全部变成 -1（这是个真实踩过的 bug，source decision 会拿到非法候选）。
    if bool((first >= candidate_owner.numel()).any()):
        raise RuntimeError("some decision node has no candidate; the table is corrupt")
    null_index = torch.full_like(first, -1)
    is_null = candidate_is_null
    if bool(is_null.any()):
        null_owner = candidate_owner[is_null]
        null_flat = torch.nonzero(is_null, as_tuple=False).squeeze(1)
        null_index = null_index.scatter(0, null_owner, null_flat)
    return torch.where(null_index >= 0, null_index, first)


# 旧名字保留为别名，避免外部脚本立刻失效；新代码请用 heuristic_prior_deterministic
prior_deterministic = heuristic_prior_deterministic


# ---------------------------------------------------------------------------
# one reverse step
# ---------------------------------------------------------------------------
@torch.no_grad()
def reverse_step(
    diffusion: CategoricalDiffusion,
    model: GraphFlowDenoiser,
    batch,
    H_t: Tensor,
    z_t: Tensor,
    t: int,
    generator: Optional[torch.Generator] = None,
    stochastic: bool = True,
) -> Dict[str, Any]:
    """(H_t, z_t) -> (H_{t-1}, z_{t-1})，一步。"""
    out = model.step(batch, H_t, z_t, t)

    clean_dense, mask = diffusion.dense_from_flat_prob(
        out.candidate_prob,
        batch.candidate_owner,
        batch.num_decisions,
    )
    reverse_prob = diffusion.model_reverse_posterior(
        clean_dense,
        mask,
        z_t,
        batch.candidate_owner,
        batch.num_decisions,
        t,
    )
    if stochastic:
        z_prev = diffusion.sample_prev(
            reverse_prob, mask, batch.candidate_owner, batch.num_decisions, generator
        )
    else:
        z_prev = diffusion.sample_prev_deterministic(
            reverse_prob, mask, batch.candidate_owner, batch.num_decisions
        )

    return {
        "H_next": out.H_next,
        "z_prev": z_prev,
        "candidate_prob": out.candidate_prob,
        "candidate_log_prob": out.candidate_log_prob,
        "reverse_prob": reverse_prob,
        "mask": mask,
        "logits": out.candidate_logits,
    }


# Backwards-friendly alias used by the guide text
reverse_categorical_step = reverse_step


# ---------------------------------------------------------------------------
# full chain
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_reverse_chain(
    diffusion: CategoricalDiffusion,
    model: GraphFlowDenoiser,
    batch,
    generator: Optional[torch.Generator] = None,
    stochastic: bool = True,
    max_steps: Optional[int] = None,
    record: bool = False,
) -> Dict[str, Any]:
    """跑完整条 reverse chain，返回最终的 z_0 与（可选）整条轨迹。

    ``max_steps`` 只能等于 ``diffusion.T``（或 None）—— 修改清单 P2-3：

        z_T ~ pi 只在 t = T 成立，一般 q(z_k) != pi。

    所以"从 t=k 开始跑 k 步"并不等价于一条更短的扩散链。第一版正式 sampler 只
    支持完整链；将来要做 accelerated sampling 必须单独设计 timestep skipping。
    """
    if max_steps is not None and int(max_steps) != int(diffusion.T):
        raise ValueError(
            f"max_steps={max_steps} is not allowed: the reverse chain must start from "
            f"z_T ~ pi, i.e. from t = T = {diffusion.T}. Truncating the loop would "
            "treat z_T as z_k, but q(z_k) != pi in general. Implement explicit "
            "timestep skipping if you need accelerated sampling."
        )
    steps = int(diffusion.T)

    H_t = model.init_nodes(batch)
    if stochastic:
        z_t = sample_prior(
            diffusion, batch.candidate_owner, batch.num_decisions, generator
        )
    else:
        z_t = heuristic_prior_deterministic(
            batch.candidate_owner, batch.candidate_is_null, batch.num_decisions
        )
    trace = ReverseTrace()
    if record:
        trace.append(H_t, z_t)

    last_log_prob = None
    for t in range(steps, 0, -1):
        step = reverse_step(
            diffusion, model, batch, H_t, z_t, t, generator, stochastic=stochastic
        )
        H_t = step["H_next"]
        z_t = step["z_prev"]
        last_log_prob = step["candidate_log_prob"]
        if record:
            trace.append(H_t, z_t, last_log_prob)

    return {
        "z0": z_t,
        "H0": H_t,
        "candidate_log_prob": last_log_prob,
        "trace": trace if record else None,
    }
