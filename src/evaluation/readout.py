"""Single-path 的最终 readout：把"要不要采样"这件事从解码器里摘出来。

历史上 single 解码直接吃 reverse chain **采样**出来的 z0：

    stochastic reverse chain  ->  sampled z0  ->  decode_flat

问题是 sampled z0 只是"从模型分布里抽到的一条"，不等于模型最可能的那条。现在默认改成
**只把最后一步确定化**：

    stochastic reverse chain  ->  最后一个 reverse step 的 candidate probability
                              ->  每个 decision 组内 argmax  ->  z0_argmax
                              ->  decode_flat                （确定性走图）

三条路径必须严格区分（都在本模块里显式命名，避免以后又被混为一谈）：

    A. "single"（默认）
       reverse chain 仍然逐步按 posterior **采样**（z_T -> ... -> z_1 一点没改），
       只在**最终 readout** 处取 candidate probability 的 grouped argmax。

    B. "single_sampled"（诊断用，旧默认行为）
       同一条 stochastic chain，直接 decode 采样出来的 z0。

    C. deterministic rollout（``stochastic=False``，对照实验）
       **每一步** reverse posterior 都取 argmax，整条链从 z_T 起就是贪心的。

A 与 C 的差别是"确定化发生在哪一层"（最终 readout vs 整条链）；A 与 B 的差别只在最后一步。
single 解码**不再**读取 sampled z0；sampled z0 仍然保留，用于扩散可视化与诊断。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from src.training.losses import grouped_argmax

#: single-path 的两种 readout（论文/评测里的 "single" 指第一种）
SINGLE_READOUT_ARGMAX = "single"
SINGLE_READOUT_SAMPLED = "single_sampled"
SINGLE_READOUTS = (SINGLE_READOUT_ARGMAX, SINGLE_READOUT_SAMPLED)


def single_readout_state(
    candidate_prob: Tensor, candidate_owner: Tensor, num_decisions: int
) -> Tensor:
    """最终 candidate probability 的 grouped argmax -> z0_argmax [M]。

    必须是**每个 decision 在自己的候选组内**取 argmax（``grouped_argmax`` 保证），
    不能对整个 flat candidate tensor 做全局 argmax：不同 decision 的候选是不同变量，
    NULL / source decision / ordinary decision 的候选定义完全沿用 DecisionField。
    """
    decisions = int(num_decisions)
    if decisions == 0 or candidate_prob.numel() == 0:
        return torch.zeros(decisions, dtype=torch.long, device=candidate_prob.device)
    return grouped_argmax(candidate_prob, candidate_owner, decisions)


def single_path_state(
    chain: Mapping[str, Any], batch: Any, readout: str = SINGLE_READOUT_ARGMAX
) -> Tensor:
    """按 readout 名字，从一条已经跑完的 reverse chain 里取 single 解码要用的状态。

    * ``"single"``         -> 最终 candidate probability 的 grouped argmax（默认）
    * ``"single_sampled"`` -> 采样出来的 z0（诊断用，旧行为）

    注意本函数**不**改变链本身的随机性：链怎么采样是 ``sample_reverse_chain`` 的事，
    这里只决定"最后拿哪个状态去解码"。
    """
    if readout == SINGLE_READOUT_SAMPLED:
        return chain["z0"]
    if readout != SINGLE_READOUT_ARGMAX:
        raise ValueError(
            f"unknown single readout {readout!r} (choose one of {SINGLE_READOUTS})"
        )
    probability = chain.get("candidate_prob")
    if probability is None:
        raise ValueError(
            "reverse chain did not return candidate_prob; the final-argmax single "
            "readout needs the last reverse step's clean-state distribution"
        )
    return single_readout_state(
        probability, batch.candidate_owner, batch.num_decisions
    )


__all__ = [
    "SINGLE_READOUT_ARGMAX",
    "SINGLE_READOUT_SAMPLED",
    "SINGLE_READOUTS",
    "single_readout_state",
    "single_path_state",
]
