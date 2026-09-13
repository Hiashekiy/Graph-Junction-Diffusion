"""Clean decision field z_0 与 candidate 表测试（实施指南第 5、25 节）。"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from src.data.branch_segments import extract_segments, set_od
from src.data.dataset_builder import build_sample
from src.data.decision_field import (
    DecisionField,
    build_candidate_table,
    target_candidate_tensor,
    validate_decision_field,
)
from tests.conftest import (
    A,
    B,
    G,
    GT_PATH,
    H,
    I,
    J1,
    J1_BRANCH_NODES,
    J2,
    S,
    X,
    make_manual_sample,
)


def test_candidate_groups_have_null_except_source():
    sample = make_manual_sample()
    candidates = sample.field.candidates
    segments = sample.segments

    for decision_index in range(segments.num_decisions):
        node = segments.decision_nodes[decision_index]
        group = [
            bool(is_null)
            for owner, is_null in zip(
                candidates.candidate_owner, candidates.candidate_is_null
            )
            if owner == decision_index
        ]
        if node == segments.start:
            assert not any(group), "source must not have a NULL candidate"
        else:
            assert any(group), f"junction {node} must have a NULL candidate"
            assert group[0], "NULL must be the first candidate of a junction group"


def test_candidate_counts_match_branches():
    sample = make_manual_sample()
    segments = sample.segments
    candidates = sample.field.candidates
    expected = sum(len(group) for group in segments.branches) + sum(
        1 for node in segments.decision_nodes if node != segments.start
    )
    assert candidates.num_candidates == expected
    assert len(candidates.candidate_owner) == expected
    assert len(candidates.candidate_is_null) == expected
    # 手工图：J1 与 J2 各 1 个 NULL + 4 条 branch
    assert segments.num_decisions == 2
    assert candidates.num_candidates == 9


def test_z0_active_branch_matches_gt_path():
    sample = make_manual_sample()
    candidates = sample.field.candidates
    for decision_index, branch in enumerate(sample.field.decision_branch):
        target = candidates.target_candidate[decision_index]
        if candidates.candidate_is_null[target]:
            assert branch is None
            continue
        assert branch is not None
        node = sample.segments.decision_nodes[decision_index]
        position = sample.gt_path.index(node)
        tail = sample.gt_path[position:]
        assert branch.nodes == tail[: len(branch.nodes)]
        assert branch.owner == node


def test_gt_path_selects_the_a_b_x_branch_at_j1():
    sample = make_manual_sample()
    assert sample.gt_path == GT_PATH
    candidates = sample.field.candidates
    j1 = sample.segments.decision_nodes.index(J1)
    chosen = candidates.candidate_branch[candidates.target_candidate[j1]]
    assert chosen is not None
    assert chosen.nodes == J1_BRANCH_NODES


def test_gt_path_selects_the_goal_branch_at_j2():
    sample = make_manual_sample()
    candidates = sample.field.candidates
    j2 = sample.segments.decision_nodes.index(J2)
    chosen = candidates.candidate_branch[candidates.target_candidate[j2]]
    assert chosen is not None
    assert chosen.nodes == [J2, G]


def test_target_index_lies_inside_its_group():
    sample = make_manual_sample()
    candidates = sample.field.candidates
    for decision_index, target in enumerate(candidates.target_candidate):
        assert candidates.candidate_owner[target] == decision_index


def test_validator_accepts_the_manual_sample():
    sample = make_manual_sample()
    validate_decision_field(sample.segments, sample.field, sample.gt_path)


def test_off_path_junction_is_null():
    """GT 不经过的 junction 必须拿到 NULL（实施指南 5.1 第 5 条）。"""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4)])  # 0-1-2，另有 1-3-4 支路
    sample = build_sample(graph, 0, 2)

    # 节点 1 在 GT path 上（度为 3 所以是 decision node），它选中的是通向 2 的 branch
    assert sample.gt_path == [0, 1, 2]
    on_path = sample.segments.decision_nodes.index(1)
    target = sample.field.candidates.target_candidate[on_path]
    assert not sample.field.candidates.candidate_is_null[target]
    assert sample.field.candidates.candidate_branch[target].nodes == [1, 2]

    # 同一个 junction 的 dead-end 分支 [1, 3, 4] 不是被选中的那个
    chosen = sample.field.candidates.candidate_branch[target]
    assert chosen.nodes != [1, 3, 4]
    assert 3 not in sample.gt_path


def test_junction_outside_the_gt_path_is_null():
    """GT 完全绕开某个 junction 时它必须是 NULL。"""
    graph = nx.Graph()
    graph.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (6, 7), (7, 3), (3, 8)]
    )
    # 0-1-2-3-8 只有 4 跳，绕行 1-4-5-6-7-3 有 6 跳
    sample = build_sample(graph, 0, 8)
    assert sample.gt_path == [0, 1, 2, 3, 8]
    for decision_index, node in enumerate(sample.segments.decision_nodes):
        target = sample.field.candidates.target_candidate[decision_index]
        if node not in sample.gt_path:
            assert sample.field.candidates.candidate_is_null[target]
            assert sample.field.decision_branch[decision_index] is None


def test_source_decision_is_always_active():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
    sample = build_sample(graph, 0, 5)
    source_index = sample.segments.decision_nodes.index(0)
    target = sample.field.candidates.target_candidate[source_index]
    assert not sample.field.candidates.candidate_is_null[target]


def test_build_candidate_table_rejects_null_source():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
    segments = extract_segments(set_od(graph, 0, 5), 0, 5)
    with pytest.raises(ValueError):
        build_candidate_table(segments, [None] * segments.num_decisions)


def test_validate_rejects_active_branch_that_is_not_on_the_gt_path():
    """把 J1 的 z_0 换成 dead-end 分支，validator 必须报错。"""
    sample = make_manual_sample()
    segments = sample.segments
    j1 = segments.decision_nodes.index(J1)
    wrong = list(sample.field.decision_branch)
    wrong[j1] = [b for b in segments.branches[j1] if b.nodes == [J1, H, I]][0]

    candidates = build_candidate_table(segments, wrong)
    field = DecisionField(
        candidates=candidates,
        decision_branch=[
            candidates.candidate_branch[t] for t in candidates.target_candidate
        ],
    )
    with pytest.raises(AssertionError):
        validate_decision_field(segments, field, sample.gt_path)


def test_validate_rejects_goal_as_decision():
    sample = make_manual_sample()
    segments = sample.segments
    segments.decision_nodes = list(segments.decision_nodes) + [segments.goal]
    segments.branches = list(segments.branches) + [[]]
    with pytest.raises(AssertionError):
        validate_decision_field(segments, sample.field, sample.gt_path)


def test_validator_rejects_branch_with_wrong_edge_count():
    sample = make_manual_sample()
    segments = sample.segments
    branch = segments.branches[0][0]
    branch.physical_edges = list(branch.physical_edges) + [0]
    with pytest.raises(AssertionError):
        validate_decision_field(segments, sample.field, sample.gt_path)


def test_target_candidate_tensor_is_long():
    tensor = target_candidate_tensor(make_manual_sample().field)
    assert tensor.dtype == torch.long
    assert tensor.shape[0] == make_manual_sample().num_decisions
