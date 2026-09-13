"""Path Decoder (实施指南第 23 节).

V2 的 ``z_0`` 类别不再代表"下一个邻居"，而是代表**一整条 Branch Segment**。
解码就是从 s 出发、逐个 decision node 读取 branch、跳到 branch 终点：

    current = s
    path = [s]
    while current != g:
        decision = decision index of current
        target   = z0[decision]
        if target is NULL:            -> broken
        branch   = branch of target
        path    += branch.nodes[1:]
        current  = branch.end
        if current already visited:   -> loop

判定：

    goal_hit : 最终到达 g
    loop     : 重复访问同一个 structural node
    broken   : 选了 NULL / 终点是 dead-end / 超过步数上限
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from torch import Tensor



@dataclass
class DecodeResult:
    status: str                  # goal | loop | broken
    path: List[int] = field(default_factory=list)
    num_branches: int = 0
    reason: str = ""

    @property
    def goal_hit(self) -> bool:
        return self.status == "goal"

    @property
    def loop(self) -> bool:
        return self.status == "loop"

    @property
    def broken(self) -> bool:
        return self.status == "broken"


def _decision_index(decision_nodes: Sequence[int], node: int) -> Optional[int]:
    for index, value in enumerate(decision_nodes):
        if value == node:
            return index
    return None


def decode_flat(
    sample,
    z0: Tensor,
    decision_offset: int = 0,
    candidate_offset: int = 0,
    max_branches: int = 4096,
) -> DecodeResult:
    """用单个样本的 GT 结构信息解码 ``z0``。

    Args:
        z0:               **flat decision 空间**的候选索引（batch 级张量）。
                          单样本调用时长度等于本样本的 decision 数。
        decision_offset:  本样本在 flat decision 空间的起点；批量解码时必须给出，
                          否则会用别的样本的 decision 去索引。
        candidate_offset: 本样本在 flat **candidate** 空间的起点；如果 ``z0`` 里
                          装的是 batch 级 candidate 索引（例如 ``batch.target_candidate``）
                          就必须给出，否则会在错误样本的候选表里查找。
        max_branches:     防御性的步数上限。

    起点可能不是 decision node（source 只有一个出口时，它的第一步是被迫的），
    所以解码分两段：

        1. forced walk：从 s 沿唯一通路一直走到第一个 structural node
           （decision node / goal / dead-end）；
        2. branch walk：每个 decision node 读 ``z0``，跳到 branch 终点。
    """
    segments = sample.segments
    candidates = sample.field.candidates
    start, goal = segments.start, segments.goal

    path: List[int] = [start]
    visited = {start}
    num_branches = 0

    def target_of(local_decision: int) -> int:
        index = decision_offset + local_decision
        if index >= z0.numel():
            raise IndexError(
                f"decision index {index} is out of range for z0 of length {z0.numel()}; "
                "pass decision_offset when decoding inside a batch"
            )
        global_candidate = int(z0[index].item())
        local_candidate = global_candidate - candidate_offset
        if not 0 <= local_candidate < candidates.num_candidates:
            raise IndexError(
                f"candidate index {global_candidate} is outside sample "
                f"{candidate_offset}..{candidate_offset + candidates.num_candidates}; "
                "pass candidate_offset when decoding inside a batch"
            )
        return local_candidate

    # ---- 1. forced walk -------------------------------------------------
    current = start
    while current != goal and _decision_index(segments.decision_nodes, current) is None:
        neighbours = sorted(sample.graph.neighbors(current))
        forward = [v for v in neighbours if v not in visited]
        if len(forward) != 1:
            reason = (
                "dead end before any decision"
                if not forward
                else "ambiguous forced step"
            )
            return DecodeResult("broken", path, num_branches, reason)
        current = int(forward[0])
        path.append(current)
        if current in visited:  # pragma: no cover - 由上面过滤保证
            return DecodeResult("loop", path, num_branches, f"revisited node {current}")
        visited.add(current)

    if current == goal:
        return DecodeResult("goal", path, num_branches, "")

    # ---- 2. branch walk -------------------------------------------------
    while current != goal:
        if num_branches >= max_branches:
            return DecodeResult("broken", path, num_branches, "step limit exceeded")

        decision = _decision_index(segments.decision_nodes, current)
        if decision is None:
            return DecodeResult(
                "broken", path, num_branches, f"node {current} has no decision variable"
            )

        target = target_of(decision)
        if candidates.candidate_is_null[target]:
            return DecodeResult("broken", path, num_branches, f"NULL selected at {current}")

        branch = candidates.candidate_branch[target]
        if branch is None:
            return DecodeResult("broken", path, num_branches, f"missing branch at {current}")

        path.extend(int(v) for v in branch.nodes[1:])
        num_branches += 1
        current = int(branch.end)
        if current in visited:
            return DecodeResult("loop", path, num_branches, f"revisited node {current}")
        visited.add(current)

    return DecodeResult("goal", path, num_branches, "")


def decode_batch(
    samples: Sequence[Any],
    z0: Tensor,
    decision_offsets: Sequence[int],
    candidate_offsets: Optional[Sequence[int]] = None,
    max_branches: int = 4096,
) -> List[DecodeResult]:
    """批量解码。

    ``decision_offsets[i]`` / ``candidate_offsets[i]`` 分别是第 i 个样本在 flat
    decision / candidate 空间的起点。后者为空时假定 ``z0`` 装的是**样本内**的
    候选索引。
    """
    results: List[DecodeResult] = []
    for index, sample in enumerate(samples):
        results.append(
            decode_flat(
                sample,
                z0,
                decision_offset=int(decision_offsets[index]),
                candidate_offset=int(candidate_offsets[index]) if candidate_offsets else 0,
                max_branches=max_branches,
            )
        )
    return results


def decision_offsets(samples: Sequence[Any]) -> List[int]:
    offsets: List[int] = []
    running = 0
    for sample in samples:
        offsets.append(running)
        running += sample.num_decisions
    return offsets


def candidate_offsets(samples: Sequence[Any]) -> List[int]:
    offsets: List[int] = []
    running = 0
    for sample in samples:
        offsets.append(running)
        running += sample.num_candidates
    return offsets


# ---------------------------------------------------------------------------
# GT-conditioned branch identification (用于 debug metric 与 branch mapping 测试)
# ---------------------------------------------------------------------------
def gt_branch_prefixes(sample) -> List[int]:
    """通过 GT 采样得到的每个 branch candidate 与 GT path 的公共前缀长度。

    两条 branch 可能共享前若干跳（例如同一个 neighbor 的 dead-end 分支），
    但只要它们的 ``end`` 不同，prefix length 一般就不同。
    """
    segments = sample.segments
    prefixes: List[int] = []
    for group in segments.branches:
        for branch in group:
            origin = branch.owner
            path = sample.gt_path
            if origin in path:
                tail = path[path.index(origin) :]
            else:
                tail = []
            common = 0
            for a, b in zip(branch.nodes, tail):
                if a != b:
                    break
                common += 1
            prefixes.append(common)
    return prefixes
