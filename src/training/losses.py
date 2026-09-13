"""Losses + Recurrent reverse-chain training (实施指南第 18-22 节).

这里包含两部分：

1. **loss 定义**（第 21 节）

       L_t = -(1/M) * sum_i w_i * log p_theta(z_0^i | H_{t-1}, z_t, t)
       L   = (1/T) * sum_{t=1}^{T} L_t

   即 clean-state categorical CE，`w_i` 由 NULL / active 权重决定（默认都是 1）。

2. **整条 reverse chain 的 teacher-forced 展开**（第 18-20 节）

   1. 从 GT z_0 出发，按单步 transition 采样一条完整的
      z_0 -> z_1 -> ... -> z_T 前向轨迹；
   2. 只初始化一次 H_T = TypeEmbedding(V)；
   3. 对 t = T, ..., 1 连续调用同一个 F_theta，每一步都用 teacher state z_t；
   4. 每一步都预测 z_0 并算 clean-state CE；
   5. 最后 L = (1/T) * sum_t L_t，一次 backward。

**不做** ``H_t = H_t.detach()``（除非显式设置 ``truncate_every``），否则会直接
切断跨 timestep 的学习。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor

from src.diffusion.categorical import CategoricalDiffusion
from src.models.denoiser import GraphFlowDenoiser

_EPS = 1e-12


# ---------------------------------------------------------------------------
# loss definition
# ---------------------------------------------------------------------------
@dataclass
class LossWeights:
    """clean-state CE 的权重（实施指南第 21 节）。"""

    x0_ce: float = 1.0
    null_weight: float = 1.0
    active_weight: float = 1.0


def per_decision_ce(
    candidate_log_prob: Tensor,      # [C]
    target_candidate: Tensor,        # [M] flat candidate index
    candidate_owner: Tensor,         # [C]
    candidate_is_null: Tensor,       # [C]
    num_decisions: int,
    null_weight: float = 1.0,
    active_weight: float = 1.0,
) -> Tensor:
    """每个 decision 的加权负对数似然，形状 [M]。

        L_t = (1/M) * sum_i w_i * [ -log p_theta(z_0^i) ]

    权重只乘在**该 decision 自己选中的那个候选**上，不做组内归一化：
    归一化会让大候选组的 loss 被稀释（均匀分布下 L 变成 log(C_i)/C_i 而不是
    log(C_i)），梯度尺度随候选数漂移。

    如果希望"每个 decision 贡献相同权重，与候选数无关"，把 ``null_weight`` /
    ``active_weight`` 当作纯粹的任务权重使用即可（默认都是 1）。
    """
    log_prob = candidate_log_prob[target_candidate]
    is_null = candidate_is_null[target_candidate]
    weight = torch.where(
        is_null,
        torch.full_like(log_prob, float(null_weight)),
        torch.full_like(log_prob, float(active_weight)),
    )
    return -weight * log_prob


def clean_state_loss(
    candidate_log_prob: Tensor,
    target_candidate: Tensor,
    candidate_owner: Tensor,
    candidate_is_null: Tensor,
    num_decisions: int,
    weights: Optional[LossWeights] = None,
) -> Tensor:
    """一步的 clean-state CE（按 decision 平均）。"""
    weights = weights or LossWeights()
    per_decision = per_decision_ce(
        candidate_log_prob,
        target_candidate,
        candidate_owner,
        candidate_is_null,
        num_decisions,
        null_weight=weights.null_weight,
        active_weight=weights.active_weight,
    )
    return weights.x0_ce * per_decision.mean()


def grouped_argmax(
    logits: Tensor, candidate_owner: Tensor, num_decisions: int
) -> Tensor:
    """每个 decision 在**自己的候选组内**取 argmax，返回 flat candidate index [M]。

    注意不能写成 ``logits.argmax(dim=-1)``：
    ``logits`` 是一维 flat 候选表，对一维张量取 argmax 得到的是 0-dim 张量
    （torch 2.x 里 ``argmax(dim=0)`` 也是 0-dim），既不是 [C] 的逐元素索引，
    也不是 per-decision 的索引。这里用"补齐到等宽候选组 + topk(1)"实现真正的
    组内 argmax。
    """
    device = logits.device
    sizes = torch.bincount(candidate_owner, minlength=num_decisions).to(device)
    max_group = int(sizes.max().item()) if sizes.numel() else 0
    if max_group == 0:
        return torch.zeros(num_decisions, dtype=torch.long, device=device)

    positions = _group_positions(candidate_owner, sizes)
    padded = torch.full(
        (num_decisions, max_group), float("-inf"), dtype=logits.dtype, device=device
    )
    padded[candidate_owner, positions] = logits
    local = padded.topk(1, dim=1).indices.squeeze(1)
    # topk 给的是组内位置，换回 flat candidate index
    starts = torch.cumsum(sizes, dim=0) - sizes
    return starts + local


def _group_positions(candidate_owner: Tensor, sizes: Tensor) -> Tensor:
    starts = torch.cumsum(sizes, dim=0) - sizes
    return torch.arange(
        candidate_owner.numel(), device=candidate_owner.device
    ) - starts[candidate_owner]


def accuracy(
    candidate_log_prob: Tensor,
    target_candidate: Tensor,
    candidate_owner: Optional[Tensor] = None,
    num_decisions: Optional[int] = None,
) -> Tensor:
    """teacher-forced clean-state 命中率（debug metric，不用于模型选择）。

    ``candidate_log_prob`` 是**一维** flat candidate 表，所以必须给出
    ``candidate_owner`` / ``num_decisions`` 才能在每个 decision 的候选组内取 argmax。
    不给出时退化为"全局 argmax 是否命中"（只在单 decision 时有意义）。
    """
    if candidate_owner is None:
        index = candidate_log_prob.argmax()
    else:
        index = grouped_argmax(
            candidate_log_prob, candidate_owner, int(num_decisions or 0)
        )
    return (index == target_candidate).to(torch.float32).mean()


# ---------------------------------------------------------------------------
# recurrent reverse-chain unroll
# ---------------------------------------------------------------------------
@dataclass
class RecurrentLossOutput:
    loss: Tensor
    per_step_loss: List[float] = field(default_factory=list)
    per_step_accuracy: List[float] = field(default_factory=list)
    final_log_prob: Optional[Tensor] = None
    final_accuracy: float = 0.0


def recurrent_reverse_loss(
    model: GraphFlowDenoiser,
    diffusion: CategoricalDiffusion,
    batch,
    weights: Optional[LossWeights] = None,
    generator: Optional[torch.Generator] = None,
    max_steps: Optional[int] = None,
    truncate_every: int = 0,
    record: bool = False,
) -> RecurrentLossOutput:
    """teacher-forced full-chain loss（默认整条链反传）。"""
    weights = weights or LossWeights()
    steps = int(diffusion.T if max_steps is None else max_steps)

    # 1) forward noising trajectory：z_path[t] 就是 teacher state
    z_path = diffusion.sample_forward_trajectory(
        batch.target_candidate,
        batch.candidate_owner,
        batch.num_decisions,
        generator=generator,
        max_steps=steps,
    )

    # 2) 只初始化一次
    H_t = model.init_nodes(batch)

    losses: List[Tensor] = []
    step_losses: List[float] = []
    step_accuracies: List[float] = []
    final_log_prob: Optional[Tensor] = None
    steps_done = 0

    for t in range(steps, 0, -1):
        z_t = z_path[t]

        out = model.step(batch, H_t, z_t, t)
        step_loss = clean_state_loss(
            out.candidate_log_prob,
            batch.target_candidate,
            batch.candidate_owner,
            batch.candidate_is_null,
            batch.num_decisions,
            weights,
        )
        losses.append(step_loss)
        steps_done += 1
        if record:
            step_losses.append(float(step_loss.detach()))
            step_accuracies.append(
                float(
                    accuracy(
                        out.candidate_log_prob,
                        batch.target_candidate,
                        batch.candidate_owner,
                        batch.num_decisions,
                    ).detach()
                )
            )
        if t == 1:
            final_log_prob = out.candidate_log_prob

        # 3) persistent state 直接进下一步
        H_t = out.H_next
        if truncate_every and steps_done % truncate_every == 0 and t > 1:
            H_t = H_t.detach()

    loss = torch.stack(losses).mean()
    final_accuracy = (
        float(
            accuracy(
                final_log_prob,
                batch.target_candidate,
                batch.candidate_owner,
                batch.num_decisions,
            ).detach()
        )
        if final_log_prob is not None
        else 0.0
    )
    return RecurrentLossOutput(
        loss=loss,
        per_step_loss=step_losses,
        per_step_accuracy=step_accuracies,
        final_log_prob=final_log_prob,
        final_accuracy=final_accuracy,
    )


@torch.no_grad()
def one_step_clean_state_metrics(
    model: GraphFlowDenoiser,
    diffusion: CategoricalDiffusion,
    batch,
    weights: Optional[LossWeights] = None,
    generator: Optional[torch.Generator] = None,
    t: Optional[int] = None,
) -> Dict[str, float]:
    """单步 debug 指标：**重新初始化 H**，只跑一个 reverse step。

    这个数会明显低于整条链的数字，因为 `H_t` 在真实推理里是从 `H_T` 一路传播
    过来的，而这里只有一个 step 的信息。它的用途是当"训练有没有把去噪器带起来"
    的早期健康检查（均匀基线约等于 `mean(log C_i)`，见 test_reverse_chain 的
    loss 单测），**不是**模型选择指标，也不要拿它和 Goal Hit 直接比较。
    """
    weights = weights or LossWeights()
    step = int(t if t is not None else max(1, diffusion.T // 2))
    z_t = diffusion.sample_xt_at_time(
        batch.target_candidate,
        batch.candidate_owner,
        batch.num_decisions,
        torch.tensor(step, device=batch.target_candidate.device),
        generator,
    )
    H_t = model.init_nodes(batch)
    out = model.step(batch, H_t, z_t, step)
    loss = clean_state_loss(
        out.candidate_log_prob,
        batch.target_candidate,
        batch.candidate_owner,
        batch.candidate_is_null,
        batch.num_decisions,
        weights,
    )
    acc = accuracy(
        out.candidate_log_prob,
        batch.target_candidate,
        batch.candidate_owner,
        batch.num_decisions,
    )
    return {"loss": float(loss), "accuracy": float(acc), "t": float(step)}
