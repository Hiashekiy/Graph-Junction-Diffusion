"""Soft Goal Reachability（第二轮修订 B 项）.

局部 CE 只知道"这个 Junction 的 branch 选得对不对"，不知道"这些 branch 概率
组合起来还能不能走到 Goal"。这里把第 t 个 reverse timestep 的 Branch Scorer
概率分布 ``p_t(c)`` 当成**可微的转移概率**，做有限 horizon 的 value iteration：

    V_i^{(0)} = 0
    V_i^{(r)} = sum_{c in C_i} p_t(c) * R_c
        R_c = 1                        branch 直接到 Goal
        R_c = V_j^{(r-1)}              branch 到 decision j
        R_c = 0                        NULL / dead-end

最后从 source 的起点读出

    P_goal^{(t)} = V_{reach_start_decision}^{(r)}

它满足：

* **完全可微**：只有 softmax 概率 + gather + 乘法 + segment_sum，梯度能一路回到
  Branch Scorer 的 logits；
* **不需要 NetworkX**：所有拓扑（``candidate_next_decision`` /
  ``candidate_hits_goal`` / ``reach_start_*``）都由数据层预先翻译成整数编号，
  见 :mod:`src.data.collate`；
* **有限步截断**：horizon 取该图自己的 decision 数。存在环时它不像 hard decoder
  那样"重复节点立刻失败"，而是一个可微代理指标，因此**不能用它替代**验证阶段的
  Hard Goal Hit / Loop Rate / Broken Rate / Optimal Path Rate。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.utils.segment_ops import segment_sum

if TYPE_CHECKING:  # pragma: no cover - 只为类型提示
    from src.data.collate import Batch


def soft_goal_reachability(
    candidate_prob: Tensor, batch: "Batch", horizon_cap: int | None = None
) -> Tensor:
    """每个图从 source 出发在有限步内到达 Goal 的软概率，形状 ``[B]``。

    Args:
        candidate_prob: ``[C]`` 每个 candidate 的概率（decision 组内和 = 1，
            直接来自 grouped softmax，**不要**做 argmax）。
        batch:          :class:`src.data.collate.Batch`，需要第二轮修订新增的
            四个拓扑字段。
        horizon_cap:    可选的 horizon 上限（轮数）。默认 ``None`` = 每张图迭代到
            自己的 decision 数。**为什么要它**：value iteration 是一个 Python 循环
            且每轮都要扫全 batch 的 candidate，代价 ∝ max_rounds × C；混入
            "路口很多但路径很短" 的图（例如旧 V1 数据，单图 79 个 decision）时，
            一个这样的样本会把整个 batch 的轮数抬到 79，训练直接变成 CPU-bound
            （实测 GPU 利用率掉到 ~20%）。给一个上限（例如 24）后，对
            decision 数 ≤ 上限的图**行为完全不变**（controlled 数据最多 10），
            对超出的图只是把"软可达性"的传播截断在有限步内 —— 它本来就只是个
            可微代理指标，不是硬指标。

    Returns:
        ``[B]`` 的 ``P_goal``，值域 ``[0, 1]``。
    """
    device = candidate_prob.device
    dtype = candidate_prob.dtype
    num_decisions = int(batch.num_decisions)
    num_graphs = int(batch.num_graphs)
    if num_graphs == 0:
        return candidate_prob.new_zeros(0)
    if num_decisions == 0:
        # 整批图都没有 decision：source 的被迫段要么直接到 Goal，要么根本走不到
        p_goal = torch.zeros(num_graphs, device=device, dtype=dtype)
        return p_goal.masked_fill(batch.reach_start_is_goal, 1.0)

    _check_topology(candidate_prob, batch)

    next_decision = batch.candidate_next_decision
    hits_goal = batch.candidate_hits_goal

    # 每张图只需要迭代"自己有多少个 decision"轮：这一点既是有限步截断，
    # 也保证每张图的传播不会串到别的图上。
    decision_counts = torch.bincount(
        batch.decision_graph_id, minlength=num_graphs
    ).to(device)
    max_rounds = int(decision_counts.max().item())
    if horizon_cap is not None:
        max_rounds = min(max_rounds, int(horizon_cap))

    value = candidate_prob.new_zeros(num_decisions)          # V_i^{(0)}
    for round_index in range(1, max_rounds + 1):
        # 每个 candidate 在当前 horizon 下能到达 Goal 的概率
        candidate_value = candidate_prob.new_zeros(batch.num_candidates)
        if bool(hits_goal.any()):
            candidate_value = torch.where(
                hits_goal, torch.ones_like(candidate_value), candidate_value
            )
        has_next = next_decision >= 0
        if bool(has_next.any()):
            # 用 torch.where 而不是 masked_fill：masked_fill 只接受 0-dim value，
            # 而这里填的是逐元素的 V_j
            candidate_value = torch.where(
                has_next, value[next_decision.clamp_min(0)], candidate_value
            )

        new_value = segment_sum(
            candidate_prob * candidate_value, batch.candidate_owner, num_decisions
        )

        # 每张图只迭代自己需要的 horizon（小图先收敛就不再更新）
        active_graph = round_index <= decision_counts
        active_decision = active_graph[batch.decision_graph_id]
        value = torch.where(active_decision, new_value, value)

    p_goal = candidate_prob.new_zeros(num_graphs)
    reach_goal = batch.reach_start_is_goal
    if bool(reach_goal.any()):
        p_goal = torch.where(reach_goal, torch.ones_like(p_goal), p_goal)

    has_start = batch.reach_start_decision >= 0
    if bool(has_start.any()):
        p_goal = torch.where(
            has_start, value[batch.reach_start_decision.clamp_min(0)], p_goal
        )
    return p_goal.clamp(0.0, 1.0)


def soft_goal_loss(p_goal: Tensor, eps: float = 1e-8) -> Tensor:
    """``-log(P_goal + eps)`` 的 batch 平均（越小越接近"必到 Goal"）。"""
    return -(p_goal.clamp_min(0.0) + float(eps)).log().mean()


# ---------------------------------------------------------------------------
def _check_topology(candidate_prob: Tensor, batch: "Batch") -> None:
    """拓扑字段必须与 candidate / graph 数量对齐，否则给出可读的报错。"""
    _ = candidate_prob
    if (
        batch.candidate_next_decision.numel() != batch.num_candidates
        or batch.candidate_hits_goal.numel() != batch.num_candidates
    ):
        raise ValueError(
            "batch is missing the soft-goal topology fields "
            "(candidate_next_decision / candidate_hits_goal); rebuild it with "
            "collate_samples() or regenerate the dataset pickle"
        )
    if (
        batch.reach_start_decision.numel() != batch.num_graphs
        or batch.reach_start_is_goal.numel() != batch.num_graphs
    ):
        raise ValueError(
            "batch is missing the soft-goal source fields "
            "(reach_start_decision / reach_start_is_goal); rebuild it with "
            "collate_samples() or regenerate the dataset pickle"
        )
