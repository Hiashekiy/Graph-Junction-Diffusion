"""Losses + Recurrent reverse-chain training (实施指南第 18-22 节).

这里包含三部分：

1. **局部 CE**（第 21 节）

       L_t = -(1/M) * sum_i w_i * log p_theta(z_0^i | H_{t-1}, z_t, t)
       L_CE = (1/T) * sum_{t=1}^{T} L_t

   即 clean-state categorical CE，`w_i` 由 NULL / active 权重决定（默认都是 1）。

2. **Soft Goal Reachability**（第二轮修订 B 项，见 :mod:`src.training.soft_goal`）

       L_goal = sum_t omega_t * L_goal^{(t)} / sum_t omega_t
       L_goal^{(t)} = -(1/B) * sum_b log(P_goal,b^{(t)} + eps)
       omega_t = alpha_bar_t

   CE 负责"局部 branch 选得对不对"，Soft Goal 负责"这些 branch 概率组合起来
   能不能到 Goal"。

       最终： L = lambda_ce * L_CE + lambda_goal * L_goal

   ``lambda_goal = 0`` 时严格退化成旧的纯 CE baseline。

3. **整条 reverse chain 的 teacher-forced 展开**（第 18-20 节）

   1. 从 GT z_0 出发，按单步 transition 采样一条完整的
      z_0 -> z_1 -> ... -> z_T 前向轨迹；
   2. 只初始化一次 H_T = TypeEmbedding(V)；
   3. 对 t = T, ..., 1 连续调用同一个 F_theta，每一步都用 teacher state z_t；
   4. **每一步**都预测 z_0、算 clean-state CE，并同时算一次 Soft Goal；
   5. 一次 backward。

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
from src.training.soft_goal import soft_goal_loss, soft_goal_reachability

_EPS = 1e-12

# goal loss 的 timestep 加权方式
GOAL_TIMESTEP_WEIGHTINGS = ("alpha_bar", "uniform")


# ---------------------------------------------------------------------------
# loss definition
# ---------------------------------------------------------------------------
@dataclass
class LossWeights:
    """clean-state CE + Soft Goal Reachability 的权重（第二轮修订 B 项）。

    ``goal_reach_weight = 0`` 必须严格退化成旧的纯 CE baseline（数值完全一致）。
    """

    x0_ce: float = 1.0
    null_weight: float = 1.0
    active_weight: float = 1.0

    goal_reach_weight: float = 0.1
    goal_reach_eps: float = 1e-8
    # "alpha_bar"：t 越大噪声越大，Goal 级约束给得越弱（推荐）；
    # "uniform"  ：每个 timestep 等权（做 ablation 用）。
    goal_timestep_weighting: str = "alpha_bar"
    # soft goal 的 value iteration 轮数上限（None = 每张图迭代到自己的 decision 数）。
    # 只在"路口数 > 上限"的图上改变数值；对 controlled 数据（≤10）完全无影响。
    goal_horizon_cap: Optional[int] = None

    def validate(self) -> None:
        if self.goal_timestep_weighting not in GOAL_TIMESTEP_WEIGHTINGS:
            raise ValueError(
                f"loss.goal_timestep_weighting={self.goal_timestep_weighting!r} is not "
                f"supported (choose one of {GOAL_TIMESTEP_WEIGHTINGS})"
            )
        if self.goal_horizon_cap is not None and int(self.goal_horizon_cap) < 1:
            raise ValueError(
                f"loss.goal_horizon_cap must be >= 1 or None, got {self.goal_horizon_cap}"
            )


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
    """一次 reverse chain 的损失与拆开的日志字段（第二轮修订 B 项）。

    ``loss``          = ``x0_ce * ce_loss + goal_reach_weight * goal_loss``
    ``ce_loss``       = 逐 timestep 的 clean-state CE 平均
    ``goal_loss``     = 逐 timestep 的 Soft Goal Loss 按 ``omega_t`` 加权平均
    ``soft_goal_mean``= 逐 timestep ``P_goal`` 的均值（监控用，不参与反传）
    ``per_step_loss`` = 每个 timestep 的 CE（名字保持不变，向后兼容）
    """

    loss: Tensor
    ce_loss: Optional[Tensor] = None
    goal_loss: Optional[Tensor] = None
    soft_goal_mean: float = float("nan")
    per_step_loss: List[float] = field(default_factory=list)
    per_step_goal_loss: List[float] = field(default_factory=list)
    per_step_soft_goal: List[float] = field(default_factory=list)
    per_step_accuracy: List[float] = field(default_factory=list)
    final_log_prob: Optional[Tensor] = None
    final_accuracy: float = 0.0


def goal_timestep_weight(diffusion: CategoricalDiffusion, t: int, mode: str) -> float:
    """``omega_t``：t 越大噪声越大，Goal 级约束越弱。"""
    if mode == "uniform":
        return 1.0
    if mode == "alpha_bar":
        return float(diffusion.schedule.alpha_bar_at(t))
    raise ValueError(
        f"unknown goal timestep weighting {mode!r} "
        f"(choose one of {GOAL_TIMESTEP_WEIGHTINGS})"
    )


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
    """teacher-forced full-chain loss（默认整条链反传）。

        L = lambda_ce * L_CE + lambda_goal * L_goal

    每个 reverse timestep **同时**算局部 CE 与 Soft Goal Reachability：

        t=T   -> CE_T   + SoftGoal_T
        t=T-1 -> CE_T-1 + SoftGoal_T-1
        ...
        t=1   -> CE_1   + SoftGoal_1

    Soft Goal 用的是 Branch Scorer 的概率分布本身（不做 argmax、不跑 Path
    Decoder），所以梯度可以完整反传到 Branch logits。
    """
    weights = weights or LossWeights()
    weights.validate()
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

    ce_losses: List[Tensor] = []
    goal_losses: List[Tensor] = []
    goal_weights: List[float] = []
    soft_goal_means: List[Tensor] = []
    step_losses: List[float] = []
    step_goal_losses: List[float] = []
    step_soft_goals: List[float] = []
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
        ce_losses.append(step_loss)

        # Soft Goal Reachability（可微代理指标，直接吃 grouped softmax 概率）
        p_goal = soft_goal_reachability(
            out.candidate_prob, batch, horizon_cap=weights.goal_horizon_cap
        )
        step_goal_loss = soft_goal_loss(p_goal, weights.goal_reach_eps)
        goal_losses.append(step_goal_loss)
        goal_weights.append(
            goal_timestep_weight(diffusion, t, weights.goal_timestep_weighting)
        )
        soft_goal_means.append(p_goal.mean().detach())

        steps_done += 1
        if record:
            step_losses.append(float(step_loss.detach()))
            step_goal_losses.append(float(step_goal_loss.detach()))
            step_soft_goals.append(float(p_goal.mean().detach()))
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

    ce_loss = torch.stack(ce_losses).mean()
    # 注意：必须除以 sum_t omega_t，否则改 T 会改变 loss 的整体尺度
    omega = torch.tensor(goal_weights, device=ce_loss.device, dtype=ce_loss.dtype)
    goal_loss = (torch.stack(goal_losses) * omega).sum() / omega.sum().clamp_min(_EPS)
    loss = ce_loss + float(weights.goal_reach_weight) * goal_loss
    soft_goal_mean = float(torch.stack(soft_goal_means).mean())

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
        ce_loss=ce_loss,
        goal_loss=goal_loss,
        soft_goal_mean=soft_goal_mean,
        per_step_loss=step_losses,
        per_step_goal_loss=step_goal_losses,
        per_step_soft_goal=step_soft_goals,
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
    p_goal = soft_goal_reachability(
        out.candidate_prob, batch, horizon_cap=weights.goal_horizon_cap
    )
    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "soft_goal": float(p_goal.mean()),
        "t": float(step),
    }
