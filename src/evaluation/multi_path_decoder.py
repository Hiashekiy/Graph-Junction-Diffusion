"""多分支（存活路径表 / beam over branches）解码。

单路径解码每个 decision 只取一个候选（采样或 argmax），一旦那一步选错（例如在正确
路口选了 NULL、或者挑了一条绕回去的 branch），整条 query 就判失败。但模型给出的往往
是"两条 branch 概率差不多"，此时只留一条会把另一条本来能到终点的路丢掉。

这个模块按"存活路径表"的方式解码：

    frontier = [起点]
    while frontier 非空:
        每个存活路径在当前 decision 取概率最高的 top_k 条 branch，各自分叉
        终止的路径（到 goal / 选 NULL / 撞 dead-end / 重复节点）移出 frontier，
        记入 finished
        其余留在 frontier 继续前进
        按累计 log 概率排序，超过 beam_width 的尾部丢掉（记入 pruned）

    -> finished 里就是"所有活到终止条件的路径"，按概率排序

两种 NULL 策略（``null_policy``）：

* ``stop``（默认，忠实于数据语义）：NULL 也参与 top-k 排名，被选中则该路径终止
  （原单路径解码也是这么判 broken 的）；
* ``skip``：NULL 不参与排名，只在非 NULL 候选里取 top-k —— 即"NULL 不停，继续走"。

输出的路径可以逐条转成单路径的 :class:`~src.evaluation.path_decoder.DecodeResult`，
这样 goal_hit / optimal / cost_ratio 的口径与现有评测**完全一致**，可以直接对比。

--------------------------------------------------------------------------
增强（《Multi-Path Decoder 增强修改指南》），三项都是**新增**，旧语义一律不变：

1. **必死 branch 预筛选**（``filter_dead_branches``，默认 ``False``）
   在 top-k 之前剔除"终点既不是 Goal、也不是 decision node"的非 NULL branch：
   这类 branch 走完必然 broken（下游没有 decision variable），却会白占一个 top-k
   名额，把真正能到终点的候选挤掉。默认关闭，历史结果逐位可复现。

       Branch -> Goal             保留
       Branch -> 下一个 decision   保留
       Branch -> 普通终止节点       过滤（只在这个开关打开时）

   NULL **不参与**这个筛选（``stop`` 时继续参与排名、``skip`` 时继续被跳过），
   loop 也**不在**预筛选里处理（``end in seen`` 仍然由展开后的旧逻辑判定）。

2. **真实 path cost**（``PathCandidate.path_cost``）
   ``sum_{(u,v) in P} w_uv``；无权图没有 ``weight`` 属性时自然退化成跳数。
   它**只用于记录 / 排序 / 评测**，绝不参与 beam 剪枝 —— beam 仍然只看累计
   log probability（``cost-aware beam`` 是明确禁止的）。历史字段
   ``PathCandidate.cost``（跳数）保持不变，可视化/看板仍在用。

3. **多条 Goal 路径输出**
   ``best_goal``（Goal 里概率最高）之外，新增 ``best_goal_cost_path``（Goal 里真实
   cost 最低，同 cost 取概率更高者）与 ``goal_paths_by_prob`` / ``goal_paths_by_cost``。

历史口径 ``multi.best`` 仍然是"finished 中累计 log probability 最高的路径"，
不管它是 goal / broken / loop —— 没有改成"最高概率的 Goal 路径"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import Tensor

from src.evaluation.path_decoder import DecodeResult

_EPS = 1e-12

#: NULL 处理策略
NULL_POLICIES = ("stop", "skip")


def edge_weight(graph, u: int, v: int) -> float:
    """边的真实 cost；没有 ``weight`` 属性时退化成 1.0（= 一跳）。"""
    return float(graph.edges[u, v].get("weight", 1.0))


@dataclass
class PathCandidate:
    """一条存活/终止路径。"""

    nodes: List[int]
    num_branches: int
    log_prob: float
    status: str                       # goal | broken | loop | pruned
    reason: str = ""
    stop_node: int = -1
    #: 真实路径 cost = sum of edge weights（无权图 = 跳数）。只用于记录/排序/评测。
    path_cost: float = 0.0

    @property
    def goal_hit(self) -> bool:
        return self.status == "goal"

    @property
    def cost(self) -> int:
        """跳数（历史字段，可视化与旧评测仍在用）。

        weighted 图请用 :attr:`path_cost`：它才是 `sum w_e`，两者在带权图上不等价。
        """
        return max(len(self.nodes) - 1, 0)

    def to_decode_result(self, reason: str = "") -> DecodeResult:
        """转成单路径解码结果，复用现有指标口径。"""
        return DecodeResult(
            status="goal" if self.goal_hit else ("loop" if self.status == "loop" else "broken"),
            path=list(self.nodes),
            num_branches=int(self.num_branches),
            reason=reason or self.reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        """稳定可序列化的导出（指南第 5.3 节）。"""
        return {
            "nodes": [int(v) for v in self.nodes],
            "log_prob": float(self.log_prob),
            "path_cost": float(self.path_cost),
            "hops": int(self.cost),
            "num_branches": int(self.num_branches),
            "status": self.status,
            "reason": self.reason,
        }


@dataclass
class MultiDecodeResult:
    """一次多分支解码的汇总。"""

    finished: List[PathCandidate] = field(default_factory=list)   # 按 log_prob 降序
    pruned: int = 0                    # 因 beam 上限被丢掉的路径数
    num_expanded: int = 0              # 一共展开过多少条路径（含中间态）
    max_depth: int = 0                 # 跑到的最大 branch 深度
    #: 被"必死 branch 预筛选"剔除的候选数（开关打开时才可能非 0）
    num_filtered_dead_branches: int = 0

    # ---- 以下字段只在 ``strict=True``（:mod:`strict_beam_decoder`）时有意义 ----
    #: ``strict`` 模式下 ``finished`` 的语义是"最终候选集 = success 池"（只有完整走到
    #: Goal 的路径）；失败路径一律进 :attr:`discarded`、**永不参与最终排名**。
    strict: bool = False
    #: 已经走到 Goal 的**完整**路径。strict 模式下它就是 ``finished``（同一个列表）。
    success: List[PathCandidate] = field(default_factory=list)
    #: NULL / loop / dead-end / 步数超限 —— **仅用于统计**，不是候选路径。
    discarded: List[PathCandidate] = field(default_factory=list)
    #: 合法性 mask 计数（在 top-k **之前**被剔除的候选 branch 条数）
    num_masked_null: int = 0
    num_masked_loop: int = 0
    num_masked_dead_end: int = 0
    num_masked_missing_branch: int = 0
    #: 搜索结束时仍活着的路径数。strict 模式下按定义必然是 0（``while alive``）。
    alive_left: int = 0

    @property
    def num_masked(self) -> int:
        """被合法性 mask 剔除的候选 branch 总数。"""
        return (
            self.num_masked_null
            + self.num_masked_loop
            + self.num_masked_dead_end
            + self.num_masked_missing_branch
        )

    @property
    def goal_paths(self) -> List[PathCandidate]:
        return [path for path in self.finished if path.goal_hit]

    @property
    def coverage(self) -> bool:
        """是否至少有一条路径到达终点（集合语义）。"""
        return bool(self.goal_paths)

    @property
    def best(self) -> Optional[PathCandidate]:
        """累计概率最高的那条（不管是否到终点）。"""
        return self.finished[0] if self.finished else None

    @property
    def best_goal(self) -> Optional[PathCandidate]:
        """累计概率最高的"到终点"路径（没有则 None）。"""
        goals = self.goal_paths
        return goals[0] if goals else None

    @property
    def best_goal_cost_path(self) -> Optional[PathCandidate]:
        """Goal 路径里**真实 cost 最低**的那条；同 cost 取概率更高者。"""
        goals = self.goal_paths
        if not goals:
            return None
        return min(goals, key=lambda path: (path.path_cost, -path.log_prob))

    @property
    def best_goal_path_cost(self) -> Optional[float]:
        """:attr:`best_goal_cost_path` 的真实 cost（没有 Goal 路径时 None）。"""
        path = self.best_goal_cost_path
        return None if path is None else float(path.path_cost)

    @property
    def best_goal_cost(self) -> Optional[int]:
        """历史字段：Goal 路径里最少的**跳数**（weighted 请用 best_goal_path_cost）。"""
        goals = self.goal_paths
        return min(path.cost for path in goals) if goals else None

    def goal_paths_by_prob(self, top_n: Optional[int] = None) -> List[PathCandidate]:
        """所有 Goal 路径按累计 log probability 从高到低。"""
        ranked = sorted(self.goal_paths, key=lambda path: path.log_prob, reverse=True)
        return ranked if top_n is None else ranked[: max(int(top_n), 0)]

    def goal_paths_by_cost(self, top_n: Optional[int] = None) -> List[PathCandidate]:
        """所有 Goal 路径按真实 cost 从低到高（同 cost 取概率更高者在前）。"""
        ranked = sorted(
            self.goal_paths, key=lambda path: (path.path_cost, -path.log_prob)
        )
        return ranked if top_n is None else ranked[: max(int(top_n), 0)]

    def goal_paths_by_prob_dicts(self, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        return [path.to_dict() for path in self.goal_paths_by_prob(top_n)]

    def goal_paths_by_cost_dicts(self, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        return [path.to_dict() for path in self.goal_paths_by_cost(top_n)]

    def summary(self) -> Dict[str, Any]:
        payload = {
            "num_finished": len(self.finished),
            "num_goal_paths": len(self.goal_paths),
            "coverage": self.coverage,
            "best_status": self.best.status if self.best else None,
            "best_log_prob": self.best.log_prob if self.best else None,
            "best_goal_cost": self.best_goal_cost,
            "best_goal_path_cost": self.best_goal_path_cost,
            "pruned": self.pruned,
            "num_expanded": self.num_expanded,
            "max_depth": self.max_depth,
            "num_filtered_dead_branches": self.num_filtered_dead_branches,
        }
        if self.strict:
            payload.update(
                {
                    "strict": True,
                    "num_success": len(self.success),
                    "num_discarded": len(self.discarded),
                    "num_masked": self.num_masked,
                    "num_masked_null": self.num_masked_null,
                    "num_masked_loop": self.num_masked_loop,
                    "num_masked_dead_end": self.num_masked_dead_end,
                    "num_masked_missing_branch": self.num_masked_missing_branch,
                    "alive_left": self.alive_left,
                }
            )
        return payload


# ---------------------------------------------------------------------------
def decode_multi_path(
    sample,
    candidate_prob: Tensor | Sequence[float],
    top_k: int = 2,
    beam_width: int = 64,
    null_policy: str = "stop",
    max_branches: int = 512,
    filter_dead_branches: bool = False,
    strict: bool = False,
) -> MultiDecodeResult:
    """对一个样本做"存活路径表"式解码。

    Args:
        sample:          :class:`src.data.dataset.GraphSample`
        candidate_prob:  **本样本局部**的 flat candidate 概率（decision 组内和 = 1），
                         形状 ``[C_local]``。批量评测时用 candidate offset 切片。
        top_k:           每个 decision 保留概率最高的几条 branch（>= 1）。
        beam_width:      存活路径表的最大长度（按累计 log 概率保留最好的）。
        null_policy:     ``stop`` / ``skip``，见模块 docstring。**``strict=True`` 时
                         无效**（NULL 在 strict 下永远被 mask）。
        max_branches:    单条路径的步数上限（防御用；loop 检查已经在起作用）。
        filter_dead_branches:
                         指南第 3 节：top-k 之前剔除"终点既不是 Goal 也不是
                         decision node"的非 NULL branch。默认 ``False``（历史行为）。
                         **``strict=True`` 时恒为真**。
        strict:          ``True`` 时改走 :mod:`src.evaluation.strict_beam_decoder`
                         的**三池**语义：失败路径（NULL / loop / dead-end）一律淘汰、
                         只有完整到 Goal 的路径进最终候选集，
                         ``P* = argmax_{P in success} log P_theta(P)``。
                         默认 ``False`` = 历史行为，逐位可复现。

    执行顺序（指南第 3.5 节）::

        当前 decision 的全部 candidate
            -> 按 null_policy 处理 NULL
            -> filter_dead_branches=True 时删除必死的非 NULL branch
            -> 按 candidate_prob 降序取 top_k
    """
    if strict:
        # 延迟 import：strict_beam_decoder 反向 import 本模块的 PathCandidate /
        # MultiDecodeResult / edge_weight，放模块顶部会成环。
        from src.evaluation.strict_beam_decoder import decode_strict_beam

        return decode_strict_beam(
            sample,
            candidate_prob,
            top_k=top_k,
            beam_width=beam_width,
            max_branches=max_branches,
        )
    if int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if int(beam_width) < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    if null_policy not in NULL_POLICIES:
        raise ValueError(
            f"null_policy={null_policy!r} is not supported (choose one of {NULL_POLICIES})"
        )
    top_k = int(top_k)
    beam_width = int(beam_width)
    filter_dead_branches = bool(filter_dead_branches)

    if hasattr(candidate_prob, "tolist"):
        probs: List[float] = candidate_prob.tolist()
    else:  # pragma: no cover - 纯 list 输入
        probs = [float(value) for value in candidate_prob]

    segments = sample.segments
    candidates = sample.field.candidates
    graph = sample.graph
    start, goal = segments.start, segments.goal
    decision_of = {int(node): index for index, node in enumerate(segments.decision_nodes)}

    # decision -> 它的候选下标（保持表中顺序）
    groups: List[List[int]] = [[] for _ in range(segments.num_decisions)]
    for index, owner in enumerate(candidates.candidate_owner):
        groups[int(owner)].append(index)

    result = MultiDecodeResult()

    # ---- 1) forced walk：从 s 走到第一个 structural node -----------------
    path: List[int] = [start]
    visited = {start}
    current = start
    path_cost = 0.0
    while current != goal and current not in decision_of:
        forward = [v for v in sorted(graph.neighbors(current)) if v not in visited]
        if len(forward) != 1:
            reason = "dead end before any decision" if not forward else "ambiguous forced step"
            result.finished.append(
                PathCandidate(
                    list(path), 0, 0.0, "broken", reason, current, path_cost=path_cost
                )
            )
            return result
        nxt = int(forward[0])
        path_cost += edge_weight(graph, current, nxt)
        current = nxt
        path.append(current)
        visited.add(current)

    if current == goal:
        result.finished.append(
            PathCandidate(list(path), 0, 0.0, "goal", "", goal, path_cost=path_cost)
        )
        return result

    # ---- 2) beam search over branches ------------------------------------
    # frontier 元素：(nodes, seen, node, log_prob, path_cost, depth)
    frontier: List[Tuple[List[int], set, int, float, float, int]] = [
        (list(path), set(visited), int(current), 0.0, path_cost, 0)
    ]

    while frontier:
        nxt: List[Tuple[List[int], set, int, float, float, int]] = []
        for nodes, seen, node, log_prob, cost_so_far, depth in frontier:
            if depth >= max_branches:
                result.finished.append(
                    PathCandidate(
                        list(nodes), depth, log_prob, "broken",
                        "step limit exceeded", node, path_cost=cost_so_far,
                    )
                )
                continue

            group = groups[decision_of[node]]
            ranked = sorted(group, key=lambda index: probs[index], reverse=True)
            if null_policy == "skip":
                ranked = [index for index in ranked if not candidates.candidate_is_null[index]]

            # 必死 branch 预筛选（默认关；NULL 不参与这个筛选）
            filtered_here = 0
            if filter_dead_branches:
                viable: List[int] = []
                for index in ranked:
                    if candidates.candidate_is_null[index]:
                        viable.append(index)
                        continue
                    branch = candidates.candidate_branch[index]
                    end = int(branch.end) if branch is not None else None
                    if end is not None and (end == goal or end in decision_of):
                        viable.append(index)
                    else:
                        filtered_here += 1
                ranked = viable
                result.num_filtered_dead_branches += filtered_here

            picked = ranked[:top_k]
            if not picked:
                reason = (
                    f"no viable candidate at {node}"
                    if filtered_here
                    else f"no non-NULL candidate at {node}"
                )
                result.finished.append(
                    PathCandidate(
                        list(nodes), depth, log_prob, "broken", reason, node,
                        path_cost=cost_so_far,
                    )
                )
                continue

            for index in picked:
                step_log_prob = log_prob + math.log(max(probs[index], 0.0) + _EPS)
                result.num_expanded += 1
                if candidates.candidate_is_null[index]:
                    # null_policy == "stop" 时才会走到这里：该路径在这里终止
                    result.finished.append(
                        PathCandidate(
                            list(nodes), depth, step_log_prob, "broken",
                            f"NULL selected at {node}", node, path_cost=cost_so_far,
                        )
                    )
                    continue

                branch = candidates.candidate_branch[index]
                if branch is None:  # pragma: no cover - 数据层保证非 NULL 都有 branch
                    result.finished.append(
                        PathCandidate(
                            list(nodes), depth, step_log_prob, "broken",
                            f"missing branch at {node}", node, path_cost=cost_so_far,
                        )
                    )
                    continue

                branch_cost = sum(
                    edge_weight(graph, u, v)
                    for u, v in zip(branch.nodes[:-1], branch.nodes[1:])
                )
                new_cost = cost_so_far + branch_cost
                new_nodes = list(nodes) + [int(v) for v in branch.nodes[1:]]
                end = int(branch.end)
                new_depth = depth + 1
                result.max_depth = max(result.max_depth, new_depth)

                if end == goal:
                    result.finished.append(
                        PathCandidate(
                            new_nodes, new_depth, step_log_prob, "goal", "", end,
                            path_cost=new_cost,
                        )
                    )
                elif end in seen:
                    result.finished.append(
                        PathCandidate(
                            new_nodes, new_depth, step_log_prob, "loop",
                            f"revisited node {end}", end, path_cost=new_cost,
                        )
                    )
                elif end not in decision_of:
                    result.finished.append(
                        PathCandidate(
                            new_nodes, new_depth, step_log_prob, "broken",
                            f"node {end} has no decision variable", end,
                            path_cost=new_cost,
                        )
                    )
                else:
                    new_seen = set(seen)
                    new_seen.add(end)
                    nxt.append((new_nodes, new_seen, end, step_log_prob, new_cost, new_depth))

        # 存活路径表：按累计 log 概率保留最好的 beam_width 条
        # （path_cost **不参与**剪枝 —— 指南第 4.2 节明确禁止 cost-aware beam）
        nxt.sort(key=lambda item: item[3], reverse=True)
        if len(nxt) > beam_width:
            result.pruned += len(nxt) - beam_width
            nxt = nxt[:beam_width]
        frontier = nxt

    result.finished.sort(key=lambda path: path.log_prob, reverse=True)
    return result
