"""多轨迹集合损失（Multi-Trajectory Set Loss）。

方案：《Graph-Junction-Diffusion 多轨迹集合训练损失设计（重构版）》。

一句话：**让模型自己产生一组完整轨迹，再在这组轨迹之间重新分配概率质量。**

    for each sample:
        candidate_prob.detach()  ──beam miner──>  成功 / NULL / loop / dead-end / broken
                                                  + 强制加入 GT，按 nodes 去重
                                                          │
        candidate_log_prob（**未 detach**）───────────────┘
                                                          ↓
                                        S(P) = (1/L) Σ log p(c)      ← 长度归一
                                                          ↓
                                        π = softmax(S / τ)           ← 集合内归一
                                                          ↓
                            L_succ = -log Σ_{P∈success} π
                            L_sim  = -Σ_{P∈success} q log(π̂)   q = softmax(β·nLCS)
                            L_fail =  Σ_{P∈failure} π · c(P)   c = 1.5/1/1/1

--------------------------------------------------------------------------
三个容易做错的地方

**1. 训练 miner 必须用 historical decoder（``strict=False``）。**
strict 会在 top-k 之前把 NULL / loop / dead-end 全部 mask 掉，所以它**看不见失败**，
适合最终推理，不适合当训练采集器。这里显式固定：

    null_policy="stop"        NULL 真正形成 termination failure
    filter_dead_branches=False dead-end 不被提前删掉，保留成训练失败样本
    strict=False              loop / NULL / dead-end / broken 全部进 finished

**2. 搜索用 detach 的概率，打分用未 detach 的 log-prob。**
beam 只是**选**轨迹，不参与梯度；真正算分时按 ``PathCandidate.candidate_indices``
回查未 detach 的 ``candidate_log_prob`` 重算 ``S(P)``。这是本模块能反传的唯一原因。

**3. ``S(P)`` 必须是平均 log 概率，不能是累计。**
累计 ``Σ log p`` 天然偏向短路径 —— 一条 3 步就 NULL 掉的残骸会压过一条 18 步走到
终点的路线，这正是之前 historical decoder 的 ``best`` 踩过的坑。平均化之后
``S`` 衡量的是"模型沿整条轨迹平均每一步有多相信它"。

--------------------------------------------------------------------------
失败类型

从 :class:`~src.evaluation.multi_path_decoder.PathCandidate` 的 ``status`` / ``reason``
反推，四类各自最多进 1 条，不足 4 条再用剩余高分 failure 补满 —— 保证**失败多样性**，
而不是每个样本都拿 4 条 NULL。

    status == "loop"                          -> loop
    reason 以 "NULL selected" 开头             -> null
    reason 含 "no decision variable" / "dead end" -> dead_end
    其余 status == "broken"                    -> broken
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.evaluation.path_decoder import candidate_offsets, decision_offsets
from src.evaluation.real_path_metrics import normalized_lcs

__all__ = [
    "TrajectoryLossConfig",
    "MinedTrajectory",
    "classify_failure",
    "mine_success_and_failure",
    "trajectory_set_loss",
]

_EPS = 1e-8

#: 本模块暴露给训练日志的指标键。**必须与**
#: :class:`src.training.losses.RecurrentLossOutput` 的 ``traj_<key>`` 字段一一对应 ——
#: 多一个键就会让 ``RecurrentLossOutput(**metrics)`` 抛 TypeError。
METRIC_KEYS = (
    "num_candidates", "num_success", "num_failure",
    "success_mass", "failure_mass",
    "success_loss", "similarity_loss", "failure_loss",
    "mean_success_nlcs",
    "fail_null_mass", "fail_loop_mass", "fail_dead_mass", "fail_broken_mass",
    "raw_finished", "raw_success", "raw_null", "raw_loop", "raw_dead_end",
    "raw_broken", "raw_no_decision",
)

#: 失败类型 -> 默认代价（方案第 12 / 14 节）
DEFAULT_FAILURE_COST = {
    "null": 1.50,
    "loop": 1.00,
    "dead_end": 1.00,
    "broken": 1.00,
}


@dataclass
class TrajectoryLossConfig:
    """``config.loss.trajectory`` 的解析结果。"""

    enabled: bool = False
    # -- 训练轨迹采集（和推理 planner 的 strict 2/3 是两套参数，不要混）---------
    top_k: int = 2
    beam_width: int = 8
    null_policy: str = "stop"
    filter_dead_branches: bool = False
    strict: bool = False
    # -- 进入 loss 的候选数 -------------------------------------------------
    max_success: int = 4
    max_failure: int = 4
    # -- 集合 softmax ------------------------------------------------------
    temperature: float = 1.0
    # -- GT 相似度 ---------------------------------------------------------
    similarity: str = "nlcs"
    similarity_beta: float = 3.0
    # -- 三个子 loss 的权重 -------------------------------------------------
    weight: float = 0.50
    success_weight: float = 1.0
    similarity_weight: float = 1.0
    failure_weight: float = 0.50
    failure_cost: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FAILURE_COST)
    )
    #: 在哪个 reverse timestep 算 L_traj（方案第 16 节：只在最后一步算）
    timestep: int = 1

    @classmethod
    def from_config(cls, loss_cfg) -> "TrajectoryLossConfig":
        if loss_cfg is None:
            return cls()
        section = loss_cfg.get("trajectory", None)
        if section is None:
            return cls()
        get = section.get if hasattr(section, "get") else (lambda k, d=None: d)
        cost = get("failure_cost", None)
        failure_cost = dict(DEFAULT_FAILURE_COST)
        if cost is not None:
            for key in failure_cost:
                value = cost.get(key, None) if hasattr(cost, "get") else None
                if value is not None:
                    failure_cost[key] = float(value)
        return cls(
            enabled=bool(get("enabled", False)),
            top_k=int(get("top_k", 2)),
            beam_width=int(get("beam_width", 8)),
            null_policy=str(get("null_policy", "stop")),
            filter_dead_branches=bool(get("filter_dead_branches", False)),
            strict=bool(get("strict", False)),
            max_success=int(get("max_success", 4)),
            max_failure=int(get("max_failure", 4)),
            temperature=float(get("temperature", 1.0)),
            similarity=str(get("similarity", "nlcs")),
            similarity_beta=float(get("similarity_beta", 3.0)),
            weight=float(get("weight", 0.50)),
            success_weight=float(get("success_weight", 1.0)),
            similarity_weight=float(get("similarity_weight", 1.0)),
            failure_weight=float(get("failure_weight", 0.50)),
            failure_cost=failure_cost,
            timestep=int(get("timestep", 1)),
        )

    def describe(self) -> str:
        if not self.enabled:
            return "trajectory=off"
        return (
            f"trajectory[top_k={self.top_k}, beam={self.beam_width}, "
            f"null={self.null_policy}, strict={self.strict}, "
            f"max_succ={self.max_success}, max_fail={self.max_failure}, "
            f"tau={self.temperature}, beta={self.similarity_beta}, "
            f"w={self.weight}, t={self.timestep}]"
        )


@dataclass
class MinedTrajectory:
    """一条进入候选集合的轨迹。"""

    nodes: List[int]
    #: 本样本**局部**的 flat candidate index 序列（GT 是 active decision 的 target）
    candidate_indices: List[int]
    status: str                      # goal | null | loop | dead_end | broken | gt
    is_gt: bool = False
    nlcs: float = 0.0                # 与 GT 的 nLCS（GT 自己 = 1.0）

    @property
    def is_success(self) -> bool:
        """GT 与任何到达 goal 的轨迹都算 success。

        ``status == "gt"`` 也算 —— 它本来就是"GT 这条成功轨迹"的标记。早先只认
        ``is_gt`` 或 ``"goal"``，于是一条手搓的 ``status="gt"``（忘了同时给
        ``is_gt=True``）会被当成失败轨迹、按默认代价 1.0 罚，静默地把 loss 带偏。
        """
        return self.is_gt or self.status in ("goal", "gt")

    def failure_cost(self, costs: Optional[Dict[str, float]] = None) -> float:
        """本轨迹的失败代价；success 恒为 0。

        代价表**只能从 cfg 传入**（``TrajectoryLossConfig.failure_cost``）。这里
        刻意不做 ``DEFAULT_FAILURE_COST`` 兜底以外的默认：早先版本在本类上挂了一个
        用模块默认值的 ``failure_cost`` 属性，结果 config 里的 ``failure_cost``
        永远被覆盖、成了死配置。
        """
        if self.is_success:
            return 0.0
        table = DEFAULT_FAILURE_COST if costs is None else costs
        return float(table.get(self.status, 1.0))


def classify_failure(candidate) -> Optional[str]:
    """把一条 ``PathCandidate`` 归类；success 返回 ``None``。"""
    if candidate.status == "goal":
        return None
    if candidate.status == "loop":
        return "loop"
    reason = str(candidate.reason or "")
    if reason.startswith("NULL selected"):
        return "null"
    if "no decision variable" in reason or "dead end" in reason:
        return "dead_end"
    return "broken"


def mine_success_and_failure(
    sample,
    probs: Sequence[float],
    cfg: TrajectoryLossConfig,
) -> Tuple[List[MinedTrajectory], List[MinedTrajectory], Dict[str, int]]:
    """跑一次 beam miner，返回 ``(success, failure, raw_stats)``（**不含 GT**）。

    ``probs`` 必须是**已 detach** 的本样本局部 candidate 概率 —— 搜索只负责挑轨迹。
    """
    from src.evaluation.multi_path_decoder import decode_multi_path

    result = decode_multi_path(
        sample,
        list(probs),
        top_k=cfg.top_k,
        beam_width=cfg.beam_width,
        null_policy=cfg.null_policy,
        filter_dead_branches=cfg.filter_dead_branches,
        strict=cfg.strict,
    )
    gt_nodes = tuple(int(v) for v in sample.gt_path)
    gt_path = [int(v) for v in sample.gt_path]

    success: List[MinedTrajectory] = []
    failures: Dict[str, List[MinedTrajectory]] = {
        "null": [], "loop": [], "dead_end": [], "broken": []
    }
    unscoreable = 0
    for candidate in result.finished:
        nodes = [int(v) for v in candidate.nodes]
        if tuple(nodes) == gt_nodes:
            continue                      # GT 会单独加入，这里避免重复计分
        indices = [int(v) for v in candidate.candidate_indices]
        if not indices:
            # **模型一个 decision 都没做过**的轨迹，不能进候选池。
            #
            # decoder 里只有两种情况会产出空 trace：
            #   * forced walk 直接从 start 走到 goal（``status="goal"``）；
            #   * forced walk 在到达任何 decision node 之前就断了
            #     （``reason="dead end before any decision"`` / ``"ambiguous forced step"``）。
            # 两者都是"还没轮到模型选择"的路径，模型对它没有任何控制权。
            #
            # 而 :func:`_trajectory_scores` 对空 trace 记 ``S(P)=0``（空和），
            # 那基本是所有候选里的**最大值**（真实轨迹的 mean log p ≤ 0；只有在
            # p 饱和到 1 - 1e-8 以上时才会比 0 高出约 1e-8，可忽略）：
            # softmax 会把质量送给这条根本没法优化的轨迹，既稀释了真正该学的候选、
            # 又让 L_fail 去惩罚一个模型控制不了的结果。所以这里直接剔除并计数，
            # 而不是让它以 0 分混进去。
            #
            # GT 不走这条路径（它在 :func:`build_gt_trajectory` 里单独构造并强制入池）。
            unscoreable += 1
            continue
        kind = classify_failure(candidate)
        item = MinedTrajectory(
            nodes=nodes,
            candidate_indices=indices,
            status="goal" if kind is None else kind,
            nlcs=float(normalized_lcs(nodes, gt_path)),
        )
        if kind is None:
            success.append(item)
        else:
            failures[kind].append(item)

    # 先记"截断前"的产量，再砍 —— 否则 raw_* 等于 max_success/max_failure，
    # 完全看不出 miner 的 beam 预算还够不够（mining budget 消融就白做了）。
    stats = {
        "raw_finished": len(result.finished),
        "raw_success": len(success),
        "raw_null": len(failures["null"]),
        "raw_loop": len(failures["loop"]),
        "raw_dead_end": len(failures["dead_end"]),
        "raw_broken": len(failures["broken"]),
        # 被剔除的空 trace 数。它一直偏高说明 corridor 的 OD 太浅（起点附近就到终点
        # 或结构死角），那时候多轨迹项本身就没有信息可学，不是超参问题。
        "raw_no_decision": unscoreable,
    }

    # 成功：按**平均** log 概率降序（= 模型当前最相信的），取前 max_success
    success.sort(key=lambda item: -_mean_log_prob(probs, item.candidate_indices))
    success = success[: max(0, cfg.max_success)]

    # 失败：**先按类型各取 1**（保证多样性），不足再用剩余高分 failure 补满
    picked: List[MinedTrajectory] = []
    for kind in ("null", "loop", "dead_end", "broken"):
        if failures[kind]:
            picked.append(failures[kind][0])
    if len(picked) < cfg.max_failure:
        used = {id(item) for item in picked}
        rest = [
            item
            for kind in ("null", "loop", "dead_end", "broken")
            for item in failures[kind][1:]
            if id(item) not in used
        ]
        rest.sort(key=lambda item: -_mean_log_prob(probs, item.candidate_indices))
        picked.extend(rest[: max(0, cfg.max_failure) - len(picked)])
    picked = picked[: max(0, cfg.max_failure)]

    return success, picked, stats


def _mean_log_prob(probs: Sequence[float], indices: Sequence[int]) -> float:
    """``(1/L) Σ_ℓ log p(c_ℓ)`` —— 与最终 :func:`_trajectory_scores` 同一个量纲。

    候选**筛选**必须和最终**打分**用同一个分数。早先筛选用的是**累计** log p，
    于是出现"按累计口径挑进来、按平均口径打分"的精神分裂：累计口径偏好短轨迹
    （单步概率 0.75 × 3 步 = -0.863 会赢过 0.85 × 8 步 = -1.300，但模型其实更相信
    后者），被选进池子的恰好是平均分更低的那批。这正是 ``S(P)`` 之所以要用 mean
    想避免的偏置，不能在筛选这一步又把它放回来。

    ``indices`` 为空返回 0（调用方已经在 :func:`mine_success_and_failure` 里把空
    trace 全部剔除了，这里只是防御）。
    """
    if not indices:
        return 0.0
    total = 0.0
    for index in indices:
        total += math.log(max(float(probs[index]), 0.0) + _EPS)
    return total / len(indices)


def build_gt_trajectory(
    sample, target_candidate: Sequence[int], is_null_target: Sequence[bool]
) -> MinedTrajectory:
    """GT 轨迹：**只含真实经过的 active decisions**（NULL decision 不进 GT score）。

    方案第 7 节：GT 不参与 beam 搜索，直接从训练标签构造。

    ``target_candidate`` 必须是**本样本局部**的 candidate 下标（``0 <= idx <
    sample.num_candidates``），和 decoder 给出的 ``PathCandidate.candidate_indices``
    同一坐标系 —— 因为它要和它们一起进 :func:`_trajectory_scores` 索引同一个
    ``log_row``。传全局扁平下标（``batch.target_candidate`` 的原始内容）会越界。
    """
    indices = [
        int(candidate)
        for candidate, is_null in zip(target_candidate, is_null_target)
        if not bool(is_null)
    ]
    return MinedTrajectory(
        nodes=[int(v) for v in sample.gt_path],
        candidate_indices=indices,
        status="gt",
        is_gt=True,
        nlcs=1.0,
    )


def _trajectory_scores(
    log_prob_row: Tensor, trajectories: Sequence[MinedTrajectory]
) -> Tensor:
    """``S(P_k) = (1/L_k) Σ_ℓ log p(c_ℓ)`` —— **平均**，不是累计（方案第 8 节）。

    ``log_prob_row`` 是**未 detach** 的本样本 candidate log-prob 切片，梯度从这里回去。
    """
    scores: List[Tensor] = []
    for item in trajectories:
        if not item.candidate_indices:
            scores.append(log_prob_row.new_zeros(()))
            continue
        index = torch.as_tensor(
            item.candidate_indices, dtype=torch.long, device=log_prob_row.device
        )
        scores.append(log_prob_row[index].mean())
    return torch.stack(scores)


def trajectory_set_loss(
    candidate_log_prob: Tensor,
    batch,
    cfg: TrajectoryLossConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    """逐样本构造候选集合，返回 ``(L_traj, metrics)``。

    ``L_traj = λ_s·L_succ + λ_g·L_sim + λ_f·L_fail``，再按 **batch 等权平均**
    （方案第 14 节；不是把所有样本的候选混在一起平均，否则候选多的样本权重更大）。
    """
    samples = list(getattr(batch, "graph_samples", ()) or ())
    if not cfg.enabled or not samples:
        zero = candidate_log_prob.new_zeros(())
        return zero, {}

    cand_off = candidate_offsets(samples)
    dec_off = decision_offsets(samples)
    target = batch.target_candidate.detach()
    is_null_candidate = batch.candidate_is_null

    per_sample: List[Tensor] = []
    accum: Dict[str, List[float]] = {}

    for position, sample in enumerate(samples):
        c0 = cand_off[position]
        c1 = c0 + int(sample.num_candidates)
        d0 = dec_off[position]
        d1 = d0 + int(sample.num_decisions)

        log_row = candidate_log_prob[c0:c1]                 # 未 detach -> 可微
        with torch.no_grad():
            probs = log_row.detach().exp().tolist()
            # 注意：``batch.target_candidate`` 存的是**全局**扁平下标（整个 batch 的
            # 候选表拼在一起），而 ``log_row`` 是**本样本局部**的切片。GT 的下标必须
            # 减掉本样本的起点 ``c0``，否则第二个样本起就会索引越界。
            # ``candidate_is_null`` 则是全表，用全局下标查才对。
            gt_target = target[d0:d1]
            gt = build_gt_trajectory(
                sample,
                (gt_target - c0).tolist(),
                is_null_candidate[gt_target].tolist(),
            )
            success, failure, stats = mine_success_and_failure(sample, probs, cfg)

        # ---- 候选集合：GT + success + failure，按 tuple(nodes) 去重（方案第 7 节）
        pool: List[MinedTrajectory] = [gt]
        seen = {tuple(gt.nodes)}
        for item in list(success) + list(failure):
            key = tuple(item.nodes)
            if key in seen:
                continue
            seen.add(key)
            pool.append(item)
        # GT 恒在池子里，所以 ``pool`` 至少有一个元素 —— 不需要"池子为空"的分支。
        # 池子只有 GT（miner 什么都没挖到）时走下面的通式也自然得到 L_traj = 0：
        # succ_mass = 1，l_succ = -log(1 + _EPS) ~ 0，l_fail = 0，l_sim = 0。
        # 刻意**不为这种情况单独短路** —— 早先版本在这里提前 continue 并手写
        # num_success = 0，结果"池子里明明有 1 条 success (GT)"却报 0。
        scores = _trajectory_scores(log_row, pool)
        pi = torch.softmax(scores / max(cfg.temperature, _EPS), dim=0)

        is_success = torch.tensor(
            [item.is_success for item in pool], dtype=torch.bool, device=pi.device
        )
        succ_mass = pi[is_success].sum()
        fail_mass = pi[~is_success].sum()

        # ---- L_succ：把概率质量从失败推向"任何到达 goal 的轨迹"
        l_succ = -torch.log(succ_mass + _EPS)

        # ---- L_sim：只在 success 内部，按 nLCS 构造 soft target
        q = torch.softmax(
            torch.tensor(
                [item.nlcs for item in pool], dtype=pi.dtype, device=pi.device
            )[is_success]
            * float(cfg.similarity_beta),
            dim=0,
        )
        pi_hat = pi[is_success] / succ_mass.clamp_min(_EPS)
        l_sim = -(q * torch.log(pi_hat + _EPS)).sum()

        # ---- L_fail：只有失败轨迹贡献，NULL termination 代价更高
        cost = torch.tensor(
            [item.failure_cost(cfg.failure_cost) for item in pool],
            dtype=pi.dtype,
            device=pi.device,
        )
        l_fail = (pi * cost).sum()

        per_sample.append(
            float(cfg.success_weight) * l_succ
            + float(cfg.similarity_weight) * l_sim
            + float(cfg.failure_weight) * l_fail
        )

        with torch.no_grad():
            accum.setdefault("num_candidates", []).append(float(len(pool)))
            accum.setdefault("num_success", []).append(float(int(is_success.sum())))
            accum.setdefault("num_failure", []).append(
                float(int((~is_success).sum()))
            )
            accum.setdefault("success_mass", []).append(float(succ_mass.detach()))
            accum.setdefault("failure_mass", []).append(float(fail_mass.detach()))
            accum.setdefault("success_loss", []).append(float(l_succ.detach()))
            accum.setdefault("similarity_loss", []).append(float(l_sim.detach()))
            accum.setdefault("failure_loss", []).append(float(l_fail.detach()))
            accum.setdefault("mean_success_nlcs", []).append(
                float(
                    torch.tensor(
                        [item.nlcs for item in pool], dtype=torch.float32
                    )[is_success.cpu()].mean()
                )
            )
            for kind, label in (
                ("null", "null"), ("loop", "loop"),
                ("dead_end", "dead"), ("broken", "broken"),
            ):
                mass = sum(
                    float(pi[index].detach())
                    for index, item in enumerate(pool)
                    if item.status == kind
                )
                accum.setdefault(f"fail_{label}_mass", []).append(mass)
            for key, value in stats.items():
                accum.setdefault(key, []).append(float(value))

    loss = (
        torch.stack(per_sample).mean()
        if per_sample
        else candidate_log_prob.new_zeros(())
    )
    metrics = {
        key: float(sum(accum[key]) / len(accum[key]))
        for key in METRIC_KEYS
        if accum.get(key)
    }
    return float(cfg.weight) * loss, metrics
