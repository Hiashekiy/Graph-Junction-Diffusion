"""严格存活路径竞争解码（strict beam）—— 三池语义。

与 :mod:`src.evaluation.multi_path_decoder` 的默认（历史）实现的**唯一但本质**差别：

    历史实现        一个 finished 池，装 goal / NULL / loop / dead-end 全部终止路径，
                    最后一起按累计 log 概率排序 —— 于是"最早被 NULL 打断的残骸"
                    因为负数加得少、log 概率最大，反而排到第 1 名，成为最终输出。

    本模块          三个池子，**失败路径一律淘汰、永不参与最终排名**：

                        alive      还能继续扩展的路径（全局最多 beam_width 条）
                        success    已经走到 Goal 的**完整**路径  ← 最终候选只有它
                        discarded  NULL / loop / dead-end / 步数超限，仅用于统计

    最终输出：

        P* = argmax_{P in success} Score_theta(P)          （Score = 累计 log 概率）
        success 为空  ->  decode failed

为什么必须这样：任务定义是"从 s 规划到 g"，一条停在 NULL 的路径**不是一条路线**，
是一个被放弃的半成品。拿它去和完整路线比 log 概率，比的是"谁先放弃"，不是"谁规划得好"。
旧的 ``multi.best`` 在 DiDi test_1000 上实测 ``mean_nodes = 7.30``，而真正走到终点的
路径 ``mean_nodes = 18.49`` —— 输出的根本不是路线，是路线的前缀。

--------------------------------------------------------------------------
合法性 mask（在 top-k **之前**做，这是"模型负责排序、decoder 负责淘汰非法路径"）

对当前 decision 的全部 candidate，按 ``candidate_prob`` 降序扫描，逐个判定：

    NULL                        -> mask（不是合法 branch）
    branch is None              -> mask（数据层不该出现，防御）
    branch 的任一非首节点已访问   -> mask（会成环 / 重复经过）
    branch.end 既不是 Goal
      也不是 decision node       -> mask（dead-end，走完必然无路可走）

只有通过判定的是**合法 branch**，再从合法 branch 里取 top_k 分叉。若某个 decision
**一条合法 branch 都没有**，这条路径才真正死亡，进 ``discarded``。

关键点：mask 掉一条候选**不会**让整条路径死掉 —— 会在剩下的合法 branch 里重新选
（例如 A 会回到已访问节点、B 不会，那就 mask A 取 B）。这正是"Beam=3 表示任何时刻
全局最多 3 条真正还活着的候选路径"这句话的实现方式。

--------------------------------------------------------------------------
与 ``null_policy`` / ``filter_dead_branches`` 的关系

strict 模式**内置** NULL mask 与 dead-end mask，所以：

* ``null_policy`` 在 strict 下不起作用（NULL 永远不合法）；
* ``filter_dead_branches`` 在 strict 下恒为真。

旧的两个开关只在 ``strict=False`` 的历史路径上有意义，两套语义互不干扰。

--------------------------------------------------------------------------
beam 剪枝的口径（**没有改**）

``alive`` 超限时仍然只看**累计 log 概率**，真实 path cost **绝不参与剪枝**
（cost-aware beam 是明确禁止的）。``path_cost`` 只用于记录与评测。

--------------------------------------------------------------------------
与历史实现的兼容

``strict=True`` 时 :class:`~src.evaluation.multi_path_decoder.MultiDecodeResult`
的 ``finished`` 字段语义变为"最终候选集 = success 池"，于是 ``best`` / ``goal_paths``
/ ``coverage`` 三个现有属性自动拿到正确语义，无需改调用方。

``strict=False``（默认）走历史实现，逐位可复现，旧 JSON 一个字段都不变。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from src.evaluation.multi_path_decoder import (
    _EPS,
    MultiDecodeResult,
    PathCandidate,
    edge_weight,
)

__all__ = ["decode_strict_beam"]

#: frontier 元素：(nodes, visited, node, log_prob, path_cost, depth)
_Alive = Tuple[List[int], set, int, float, float, int]


def decode_strict_beam(
    sample,
    candidate_prob,
    top_k: int = 2,
    beam_width: int = 3,
    max_branches: int = 512,
) -> MultiDecodeResult:
    """严格存活路径竞争解码。参数含义与 :func:`decode_multi_path` 一致。"""
    if int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if int(beam_width) < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
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

    groups: List[List[int]] = [[] for _ in range(segments.num_decisions)]
    for index, owner in enumerate(candidates.candidate_owner):
        groups[int(owner)].append(index)

    result = MultiDecodeResult(strict=True)

    # ---- 1) forced walk：从 s 走到第一个 structural node（与历史实现同一套逻辑）----
    path: List[int] = [start]
    visited = {int(start)}
    current = int(start)
    path_cost = 0.0
    while current != goal and current not in decision_of:
        forward = [v for v in sorted(graph.neighbors(current)) if v not in visited]
        if len(forward) != 1:
            reason = "dead end before any decision" if not forward else "ambiguous forced step"
            result.discarded.append(
                PathCandidate(list(path), 0, 0.0, "broken", reason, current, path_cost=path_cost)
            )
            result.alive_left = 0
            return result
        step = int(forward[0])
        path_cost += edge_weight(graph, current, step)
        current = step
        path.append(current)
        visited.add(current)

    if current == goal:
        result.finished = [
            PathCandidate(list(path), 0, 0.0, "goal", "", goal, path_cost=path_cost)
        ]
        result.success = list(result.finished)
        result.alive_left = 0
        return result

    # ---- 2) 存活路径竞争 --------------------------------------------------
    alive: List[_Alive] = [(list(path), set(visited), current, 0.0, path_cost, 0)]
    while alive:
        nxt_alive: List[_Alive] = []
        for nodes, seen, node, log_prob, cost_so_far, depth in alive:
            if depth >= max_branches:
                result.discarded.append(
                    PathCandidate(
                        list(nodes), depth, log_prob, "broken",
                        "step limit exceeded", node, path_cost=cost_so_far,
                    )
                )
                continue

            legal = _legal_candidates(
                candidates, groups[decision_of[node]], probs, seen, goal, decision_of, result
            )
            picked = legal[:top_k]
            if not picked:
                result.discarded.append(
                    PathCandidate(
                        list(nodes), depth, log_prob, "broken",
                        f"no legal branch at {node}", node, path_cost=cost_so_far,
                    )
                )
                continue

            for index in picked:
                step_log_prob = log_prob + math.log(max(probs[index], 0.0) + _EPS)
                result.num_expanded += 1
                branch = candidates.candidate_branch[index]
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
                    # 完整路线 -> success 池（**不进** alive，也不再和残骸比概率）
                    result.success.append(
                        PathCandidate(
                            new_nodes, new_depth, step_log_prob, "goal", "", end,
                            path_cost=new_cost,
                        )
                    )
                    continue

                new_seen = set(seen)
                new_seen.update(int(v) for v in branch.nodes[1:])
                nxt_alive.append((new_nodes, new_seen, end, step_log_prob, new_cost, new_depth))

        # 全局只保留 beam_width 条仍然可继续的路径（只看累计 log 概率）
        nxt_alive.sort(key=lambda item: item[3], reverse=True)
        if len(nxt_alive) > beam_width:
            result.pruned += len(nxt_alive) - beam_width
            nxt_alive = nxt_alive[:beam_width]
        alive = nxt_alive
        result.alive_left = len(alive)

    # ---- 3) 最终候选集 = success 池 ---------------------------------------
    result.success.sort(key=lambda candidate: candidate.log_prob, reverse=True)
    result.finished = result.success
    return result


def _legal_candidates(
    candidates, group: Sequence[int], probs: Sequence[float], seen: set,
    goal: int, decision_of: Dict[int, int], result: MultiDecodeResult,
) -> List[int]:
    """按概率降序筛出**合法 branch**，非法的只计数、不进入候选排名。

    合法性四条（见模块 docstring）：非 NULL、branch 存在、不重复经过已访问节点、
    ``end`` 是 Goal 或下一个 decision node。
    """
    ranked = sorted(group, key=lambda index: probs[index], reverse=True)
    legal: List[int] = []
    for index in ranked:
        if candidates.candidate_is_null[index]:
            result.num_masked_null += 1
            continue
        branch = candidates.candidate_branch[index]
        if branch is None:  # pragma: no cover - 数据层保证非 NULL 都有 branch
            result.num_masked_missing_branch += 1
            continue
        if any(int(v) in seen for v in branch.nodes[1:]):
            result.num_masked_loop += 1
            continue
        end = int(branch.end)
        if end != goal and end not in decision_of:
            result.num_masked_dead_end += 1
            continue
        legal.append(index)
    return legal
