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


# ---------------------------------------------------------------------------
# closed-cycle branch（绕回 owner 的环）
# ---------------------------------------------------------------------------
def is_closed_cycle_branch(branch: Branch) -> bool:
    """``branch`` 是否是 ``O -> X -> Y -> O`` 这种**闭合环**（方案第 2 节的保护条件）。

    真实数据接入时 corridor 子图裁剪会**降低节点度**，把原本 deg >= 3 的节点降成
    deg == 2，于是 ``trace_branch`` 会从 owner 出发绕一圈又回到 owner，产生
    ``branch.end == branch.owner`` 的 branch（实测成都 corridor 上约 13% 的样本含
    这种 branch，占全部 branch 的 0.02%）。

    判定条件（三条必须同时满足）：

    1. ``branch.nodes[0] == branch.nodes[-1]``（首尾相同）；
    2. 长度 >= 3（``O -> X -> O`` 在简单图里不可能，最短是 4 个节点）；
    3. 内部节点全部唯一（``len(set) == len - 1``）。

    第 3 条把"只在首尾重复"和"内部也重复"分开：后者仍然是结构错误，必须继续报错
    （见 ``validate_decision_field`` 第 6 条与
    ``tests/test_didi_dataset.py::test_internal_repeat_branch_still_rejected``）。
    """
    nodes = list(branch.nodes)
    if len(nodes) < 3 or nodes[0] != nodes[-1]:
        return False
    return len(set(nodes)) == len(nodes) - 1


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
        nodes = list(branch.nodes)
        is_unique = len(set(nodes)) == len(nodes)
        # 唯一允许的例外：**绕回 owner 的 closed-cycle branch**（``O -> X -> Y -> O``，
        # 见 :func:`is_closed_cycle_branch`）。corridor 裁剪会把 deg >= 3 的节点降成
        # deg == 2，从而造出这种环；实测成都数据上约 13% 的样本含它，不放行就只能
        # 整条丢弃。它是**严格更宽松**的放宽：合成管线的 branch 本来就无重复节点，
        # 旧行为逐位不变；而且内部节点仍然不许重复。
        assert is_unique or is_closed_cycle_branch(branch), (
            f"branch at {branch.owner} visits a node twice: {nodes}"
        )

    # 7. closed-cycle branch 永远不能是 GT target（保护条件第 3 条）
    #
    # 这是放宽第 6 条时必须补上的那道保险：closed-cycle branch 在结构上合法、可以
    # 当干扰项，但 z_0(i) 一旦指向它，等于**用真实 GT 去训练模型制造 loop**。
    # 理论上不可能发生（``branch_covers_path`` 要求 branch 是 GT 尾部的前缀，而
    # require_simple_gt 保证 GT 里没有重复节点，所以 ``O -> ... -> O`` 永远匹配不上），
    # 但这条不变式是数据正确性的最后一道闸门，必须显式断言而不是"靠推理成立"。
    #
    # 等价说法：GT branch 必须是**离开 owner 的开放 branch**，即 end != owner。
    for decision_index, candidate_index in enumerate(candidates.target_candidate):
        if candidates.candidate_is_null[candidate_index]:
            continue
        branch = candidates.candidate_branch[candidate_index]
        assert branch is not None
        assert not is_closed_cycle_branch(branch), (
            f"decision {segments.decision_nodes[decision_index]} has a closed-cycle "
            f"branch ({branch.nodes}) as its GT target. GT must never be a loop; "
            "this would train the model to reproduce loops instead of routes."
        )
        assert branch.end != branch.owner, (
            f"decision {segments.decision_nodes[decision_index]} has a GT branch "
            f"that returns to its own owner: {branch.nodes}"
        )


def target_candidate_tensor(field: DecisionField):
    """便利函数：返回 torch.LongTensor [M]（延迟 import，方便纯 CPU 校验）。"""
    import torch

    return torch.tensor(field.candidates.target_candidate, dtype=torch.long)
