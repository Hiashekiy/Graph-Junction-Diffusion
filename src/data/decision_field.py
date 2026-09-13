"""Clean decision field z_0 与扁平 candidate 表 (实施指南第 5 节).

每个 decision node 的候选集合：

    普通 Junction :  C_i = {NULL, B_i1, ..., B_iK}
    Source        :  C_s = {B_s1, ..., B_sK}       (没有 NULL)

候选被压扁成一张全局表 ``candidate_owner / candidate_is_null``，``z_t`` 直接使用
flat candidate index（这样可以原封不动复用 categorical diffusion 数学）。

z_0 的构造规则（实施指南 5.1）：

* 在 GT path 上的 decision node：选中与 GT path 后续节点完全一致的那条 branch；
* 不在 GT path 上的普通 Junction：NULL；
* Source 必须 active（数据层保证 Source 组里没有 NULL）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.data.branch_segments import (
    Branch,
    FlatCandidates,
    GraphSegments,
    branch_covers_path,
)


@dataclass
class DecisionField:
    """z_0 及其扁平候选表。"""

    candidates: FlatCandidates
    # 便于调试/评测：每个 decision 选中的 branch（NULL 为 None）
    decision_branch: List[Optional[Branch]]

    @property
    def num_decisions(self) -> int:
        return len(self.candidates.target_candidate)


def build_candidate_table(
    segments: GraphSegments, active_branch: Sequence[Optional[Branch]]
) -> FlatCandidates:
    """把每个 decision 的候选组压扁成全局表。

    Args:
        segments:      图结构
        active_branch: 每个 decision 的 GT branch（None 表示 NULL）
    """
    if len(active_branch) != segments.num_decisions:
        raise ValueError(
            f"active_branch has {len(active_branch)} entries but there are "
            f"{segments.num_decisions} decision nodes"
        )

    candidate_owner: List[int] = []
    candidate_is_null: List[bool] = []
    candidate_branch: List[Optional[Branch]] = []
    target_candidate: List[int] = []

    for decision_index, branches in enumerate(segments.branches):
        is_source = segments.decision_nodes[decision_index] == segments.start
        if not is_source:
            # 普通 Junction 的第一个候选永远是 NULL
            candidate_owner.append(decision_index)
            candidate_is_null.append(True)
            candidate_branch.append(None)
        for branch in branches:
            candidate_owner.append(decision_index)
            candidate_is_null.append(False)
            candidate_branch.append(branch)

        chosen = active_branch[decision_index]
        if chosen is None:
            if is_source:
                raise ValueError(
                    f"source decision node {segments.decision_nodes[decision_index]} "
                    "must always be active (source has no NULL candidate)"
                )
            # NULL 永远是组内第一个候选
            target_candidate.append(_group_start(candidate_owner, decision_index))
        else:
            index = _find_candidate(candidate_branch, candidate_owner, decision_index, chosen)
            target_candidate.append(index)

    return FlatCandidates(
        candidate_owner=candidate_owner,
        candidate_is_null=candidate_is_null,
        candidate_branch=candidate_branch,
        target_candidate=target_candidate,
        num_candidates=len(candidate_owner),
    )


def _group_start(candidate_owner: Sequence[int], decision_index: int) -> int:
    for index, owner in enumerate(candidate_owner):
        if owner == decision_index:
            return index
    raise ValueError(f"decision {decision_index} has no candidates")


def _find_candidate(
    candidate_branch: Sequence[Optional[Branch]],
    candidate_owner: Sequence[int],
    decision_index: int,
    branch: Branch,
) -> int:
    for index, (owner, other) in enumerate(zip(candidate_owner, candidate_branch)):
        if owner != decision_index or other is None:
            continue
        if other.end == branch.end and list(other.nodes) == list(branch.nodes):
            return index
    raise ValueError(
        f"branch starting at {branch.owner} ending at {branch.end} is not among the "
        "candidates of its own decision node"
    )


def active_branches_from_path(segments: GraphSegments, gt_path: Sequence[int]) -> List[Optional[Branch]]:
    """从 GT path 推出每个 decision node 的 z_0。

    规则：沿着 GT path 从 s 走到 g，途中经过的 decision node 使用与随后的 path
    完全一致的那条 branch；GT path 上没有经过的 decision node 一律 NULL。
    """
    path = [int(v) for v in gt_path]
    lookup: Dict[tuple, Branch] = {}
    for group in segments.branches:
        for branch in group:
            lookup[(branch.owner, branch.end)] = branch

    active: List[Optional[Branch]] = [None] * segments.num_decisions
    position = {node: index for index, node in enumerate(path)}

    for decision_index, node in enumerate(segments.decision_nodes):
        if node not in position:
            continue
        tail = path[position[node] :]
        if len(tail) < 2:
            continue
        # 优先匹配最长的一致 branch（branch 至少覆盖到下一个 endpoint）
        candidates = [
            branch
            for branch in segments.branches[decision_index]
            if branch_covers_path(branch, tail)
        ]
        if not candidates:
            continue
        active[decision_index] = max(candidates, key=lambda branch: len(branch.nodes))

    return active


def build_decision_field(segments: GraphSegments, gt_path: Sequence[int]) -> DecisionField:
    """z_0 的完整构造入口。"""
    active = active_branches_from_path(segments, gt_path)
    candidates = build_candidate_table(segments, active)
    return DecisionField(
        candidates=candidates,
        decision_branch=[candidates.candidate_branch[i] for i in candidates.target_candidate],
    )


# ---------------------------------------------------------------------------
# validators（实施指南 5.1 的六条强制检查）
# ---------------------------------------------------------------------------
def validate_decision_field(
    segments: GraphSegments,
    field: DecisionField,
    gt_path: Optional[Sequence[int]] = None,
) -> None:
    """任何一条不满足就抛 AssertionError，避免把错的语义带进 tensor。"""
    candidates = field.candidates

    # 1. every active branch really exists
    for index, is_null in enumerate(candidates.candidate_is_null):
        if is_null:
            continue
        branch = candidates.candidate_branch[index]
        assert branch is not None, f"candidate {index} is not NULL but has no branch"
        owner = segments.decision_nodes[candidates.candidate_owner[index]]
        assert branch.owner == owner, (
            f"candidate {index} owner mismatch: branch.owner={branch.owner} decision={owner}"
        )
        assert branch.nodes[0] == owner and branch.end == branch.nodes[-1]

    # 2. active branch node sequence matches GT path
    if gt_path is not None:
        path = [int(v) for v in gt_path]
        for decision_index, branch in enumerate(field.decision_branch):
            if branch is None:
                continue
            node = segments.decision_nodes[decision_index]
            assert node in path, f"active decision {node} is not on the GT path"
            position = path.index(node)
            tail = path[position :]
            assert branch_covers_path(branch, tail), (
                f"active branch at decision {node} does not match the GT path: "
                f"{branch.nodes} vs {tail[: len(branch.nodes)]}"
            )

    # 3. source never NULL
    for decision_index, node in enumerate(segments.decision_nodes):
        if node == segments.start:
            target = candidates.target_candidate[decision_index]
            assert not candidates.candidate_is_null[target], "source candidate must not be NULL"

    # 4. goal never decision
    assert segments.goal not in segments.decision_nodes, "goal must never be a decision node"

    # 5. ordinary off-path decision -> NULL
    if gt_path is not None:
        path = [int(v) for v in gt_path]
        for decision_index, node in enumerate(segments.decision_nodes):
            if node not in path:
                target = candidates.target_candidate[decision_index]
                assert candidates.candidate_is_null[target], (
                    f"decision {node} is off the GT path but its target is not NULL"
                )

    # 6. every branch consists only of real graph edges
    for branch in segments.branches_flat():
        assert len(branch.nodes) == len(branch.physical_edges) + 1, (
            f"branch at {branch.owner}: {len(branch.nodes)} nodes vs "
            f"{len(branch.physical_edges)} physical edges"
        )
        assert len(set(branch.nodes)) == len(branch.nodes), (
            f"branch at {branch.owner} visits a node twice: {branch.nodes}"
        )


def target_candidate_tensor(field: DecisionField):
    """便利函数：返回 torch.LongTensor [M]（延迟 import，方便纯 CPU 校验）。"""
    import torch

    return torch.tensor(field.candidates.target_candidate, dtype=torch.long)
