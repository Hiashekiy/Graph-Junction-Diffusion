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


@dataclass
class PathCandidate:
    """一条存活/终止路径。"""

    nodes: List[int]
    num_branches: int
    log_prob: float
    status: str                       # goal | broken | loop | pruned
    reason: str = ""
    stop_node: int = -1

    @property
    def goal_hit(self) -> bool:
        return self.status == "goal"

    @property
    def cost(self) -> int:
        """跳数（与未加权图上的 path_cost 等价）。"""
        return max(len(self.nodes) - 1, 0)

    def to_decode_result(self, reason: str = "") -> DecodeResult:
        """转成单路径解码结果，复用现有指标口径。"""
        return DecodeResult(
            status="goal" if self.goal_hit else ("loop" if self.status == "loop" else "broken"),
            path=list(self.nodes),
            num_branches=int(self.num_branches),
            reason=reason or self.reason,
        )


@dataclass
class MultiDecodeResult:
    """一次多分支解码的汇总。"""

    finished: List[PathCandidate] = field(default_factory=list)   # 按 log_prob 降序
    pruned: int = 0                    # 因 beam 上限被丢掉的路径数
    num_expanded: int = 0              # 一共展开过多少条路径（含中间态）
    max_depth: int = 0                 # 跑到的最大 branch 深度

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
    def best_goal_cost(self) -> Optional[int]:
        goals = self.goal_paths
        return min(path.cost for path in goals) if goals else None

    def summary(self) -> Dict[str, Any]:
        return {
            "num_finished": len(self.finished),
            "num_goal_paths": len(self.goal_paths),
            "coverage": self.coverage,
            "best_status": self.best.status if self.best else None,
            "best_log_prob": self.best.log_prob if self.best else None,
            "best_goal_cost": self.best_goal_cost,
            "pruned": self.pruned,
            "num_expanded": self.num_expanded,
            "max_depth": self.max_depth,
        }


# ---------------------------------------------------------------------------
def decode_multi_path(
    sample,
    candidate_prob: Tensor | Sequence[float],
    top_k: int = 2,
    beam_width: int = 64,
    null_policy: str = "stop",
    max_branches: int = 512,
) -> MultiDecodeResult:
    """对一个样本做"存活路径表"式解码。

    Args:
        sample:          :class:`src.data.dataset.GraphSample`
        candidate_prob:  **本样本局部**的 flat candidate 概率（decision 组内和 = 1），
                         形状 ``[C_local]``。批量评测时用 candidate offset 切片。
        top_k:           每个 decision 保留概率最高的几条 branch（>= 1）。
        beam_width:      存活路径表的最大长度（按累计 log 概率保留最好的）。
        null_policy:     ``stop`` / ``skip``，见模块 docstring。
        max_branches:    单条路径的步数上限（防御用；loop 检查已经在起作用）。
    """
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
    while current != goal and current not in decision_of:
        forward = [v for v in sorted(graph.neighbors(current)) if v not in visited]
        if len(forward) != 1:
            reason = "dead end before any decision" if not forward else "ambiguous forced step"
            result.finished.append(
                PathCandidate(list(path), 0, 0.0, "broken", reason, current)
            )
            return result
        current = int(forward[0])
        path.append(current)
        visited.add(current)

    if current == goal:
        result.finished.append(PathCandidate(list(path), 0, 0.0, "goal", "", goal))
        return result

    # ---- 2) beam search over branches ------------------------------------
    frontier: List[Tuple[List[int], set, int, float, int]] = [
        (list(path), set(visited), int(current), 0.0, 0)
    ]

    while frontier:
        nxt: List[Tuple[List[int], set, int, float, int]] = []
        for nodes, seen, node, log_prob, depth in frontier:
            if depth >= max_branches:
                result.finished.append(
                    PathCandidate(list(nodes), depth, log_prob, "broken", "step limit exceeded", node)
                )
                continue

            group = groups[decision_of[node]]
            ranked = sorted(group, key=lambda index: probs[index], reverse=True)
            if null_policy == "skip":
                ranked = [index for index in ranked if not candidates.candidate_is_null[index]]
            picked = ranked[:top_k]
            if not picked:
                result.finished.append(
                    PathCandidate(list(nodes), depth, log_prob, "broken",
                                  f"no non-NULL candidate at {node}", node)
                )
                continue

            for index in picked:
                step_log_prob = log_prob + math.log(max(probs[index], 0.0) + _EPS)
                result.num_expanded += 1
                if candidates.candidate_is_null[index]:
                    # null_policy == "stop" 时才会走到这里：该路径在这里终止
                    result.finished.append(
                        PathCandidate(list(nodes), depth, step_log_prob, "broken",
                                      f"NULL selected at {node}", node)
                    )
                    continue

                branch = candidates.candidate_branch[index]
                if branch is None:  # pragma: no cover - 数据层保证非 NULL 都有 branch
                    result.finished.append(
                        PathCandidate(list(nodes), depth, step_log_prob, "broken",
                                      f"missing branch at {node}", node)
                    )
                    continue

                new_nodes = list(nodes) + [int(v) for v in branch.nodes[1:]]
                end = int(branch.end)
                new_depth = depth + 1
                result.max_depth = max(result.max_depth, new_depth)

                if end == goal:
                    result.finished.append(
                        PathCandidate(new_nodes, new_depth, step_log_prob, "goal", "", end)
                    )
                elif end in seen:
                    result.finished.append(
                        PathCandidate(new_nodes, new_depth, step_log_prob, "loop",
                                      f"revisited node {end}", end)
                    )
                elif end not in decision_of:
                    result.finished.append(
                        PathCandidate(new_nodes, new_depth, step_log_prob, "broken",
                                      f"node {end} has no decision variable", end)
                    )
                else:
                    new_seen = set(seen)
                    new_seen.add(end)
                    nxt.append((new_nodes, new_seen, end, step_log_prob, new_depth))

        # 存活路径表：按累计 log 概率保留最好的 beam_width 条
        nxt.sort(key=lambda item: item[3], reverse=True)
        if len(nxt) > beam_width:
            result.pruned += len(nxt) - beam_width
            nxt = nxt[:beam_width]
        frontier = nxt

    result.finished.sort(key=lambda path: path.log_prob, reverse=True)
    return result
