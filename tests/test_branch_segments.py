"""Branch Segment 测试（实施指南第 3、25.1 节）。

最关键的一条：`J1 - a - b - x - J2` 必须被识别成**一整条** branch

    nodes = [J1, a, b, x, J2]

而不是旧版的 `[J1, a]`。
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.data.branch_segments import (
    GOAL,
    JUNCTION,
    ORDINARY,
    START,
    build_decision_nodes,
    build_endpoints,
    build_edge_tables,
    extract_segments,
    physical_edge_lookup,
    set_od,
    trace_branch,
)
from tests.conftest import (
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    GT_PATH,
    H,
    I,
    J1,
    J1_BRANCH_NODES,
    J2,
    S,
    X,
    make_manual_graph,
)


def _segments(graph=None, start=S, goal=G):
    graph = graph if graph is not None else make_manual_graph()[0]
    return extract_segments(set_od(graph, start, goal), start, goal)


def _j1_branches(segments):
    return segments.branches[segments.decision_nodes.index(J1)]


# ---------------------------------------------------------------------------
# 25.1 branch segment
# ---------------------------------------------------------------------------
def test_branch_segment_spans_the_whole_segment():
    segments = _segments()
    via_a = [b for b in _j1_branches(segments) if b.nodes == J1_BRANCH_NODES]
    assert len(via_a) == 1
    assert via_a[0].end == J2
    assert len(via_a[0].physical_edges) == len(J1_BRANCH_NODES) - 1


def test_branch_is_not_one_hop():
    """旧版一跳 candidate 只有 2 个节点；V2 必须更长。"""
    segments = _segments()
    lengths = [len(b.nodes) for b in segments.branches_flat()]
    assert max(lengths) == 5
    assert sum(1 for length in lengths if length == 2) > 0


def test_j1_has_four_branches_covering_all_neighbours():
    segments = _segments()
    j1_branches = _j1_branches(segments)
    assert len(j1_branches) == 4
    # J1 的邻居是 c(1)、h(8)、a(3)、d(10)，四条 branch 的首跳必须正好是这四个
    assert sorted(b.nodes[1] for b in j1_branches) == sorted([C, H, A, D])


def test_two_different_branches_can_end_at_the_same_junction():
    """J1 有两条 branch 都终止在 J2（经 a-b-x 与经 d-e-f），必须都在候选里。"""
    segments = _segments()
    to_j2 = [b for b in _j1_branches(segments) if b.end == J2]
    assert len(to_j2) == 2
    assert sorted(len(b.nodes) for b in to_j2) == [5, 5]
    assert {tuple(b.nodes) for b in to_j2} == {
        (J1, A, B, X, J2),
        (J1, D, E, F, J2),
    }


def test_trace_branch_matches_the_pseudocode():
    graph = make_manual_graph()[0]
    endpoints = set(build_endpoints(graph, S, G))
    branch = trace_branch(graph, J1, A, endpoints, physical_edge_lookup(graph))
    assert branch.nodes == J1_BRANCH_NODES
    assert branch.end == J2


def test_gt_path_of_the_manual_graph():
    segments = _segments()
    assert segments.decision_nodes == [J1, J2]
    assert GT_PATH == [S, C, J1, A, B, X, J2, G]
    assert len(GT_PATH) - 1 == 7
    # 手工图共 13 条无向边 / 13 个节点
    assert segments.num_physical_edges == 13
    assert len(segments.edge_index) == 26


# ---------------------------------------------------------------------------
# node types / decision set / endpoints
# ---------------------------------------------------------------------------
def test_node_types():
    segments = _segments()
    expected = {
        S: START,
        C: ORDINARY,
        J1: JUNCTION,
        A: ORDINARY,
        B: ORDINARY,
        X: ORDINARY,
        J2: JUNCTION,
        G: GOAL,
        H: ORDINARY,
        I: ORDINARY,
        D: ORDINARY,
        E: ORDINARY,
        F: ORDINARY,
    }
    for node, value in expected.items():
        assert segments.node_type[node] == value, f"node {node}"


def test_start_goal_priority_over_degree():
    """度为 4 的 start 仍然是 START，度为 4 的 goal 仍然是 GOAL。"""
    graph = nx.Graph()
    graph.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4), (1, 5), (2, 5), (3, 6), (4, 6)]
    )
    segments = extract_segments(set_od(graph, 0, 5), 0, 5)
    assert segments.node_type[0] == START
    assert segments.node_type[5] == GOAL
    assert graph.degree(0) >= 3


def test_decision_nodes_follow_the_definition():
    """D = (J ∪ {s | deg(s) > 1}) \\ {g}。"""
    graph, start, goal = make_manual_graph()
    assert build_decision_nodes(graph, start, goal) == [J1, J2]

    # source 有多个出口时也必须是 decision node
    graph2 = nx.Graph()
    graph2.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    decisions = build_decision_nodes(graph2, 0, 5)
    assert 0 in decisions
    assert 5 not in decisions


def test_source_with_single_exit_is_not_a_decision_node():
    graph, start, goal = make_manual_graph()
    assert graph.degree(start) == 1
    assert start not in build_decision_nodes(graph, start, goal)


def test_goal_is_never_a_decision_node():
    graph, start, goal = make_manual_graph()
    assert goal not in build_decision_nodes(graph, start, goal)
    assert goal not in _segments().decision_nodes


def test_endpoints_include_s_g_and_every_non_degree_two_node():
    graph = make_manual_graph()[0]
    endpoints = set(build_endpoints(graph, S, G))
    assert {S, J1, J2, G, I} <= endpoints
    for ordinary in (C, A, B, X, H, D, E):
        assert ordinary not in endpoints


# ---------------------------------------------------------------------------
# physical edge / message edge separation
# ---------------------------------------------------------------------------
def test_edge_tables_are_two_directions_per_physical_edge():
    graph = make_manual_graph()[0]
    edge_index, msg_to_phys, num_phys = build_edge_tables(graph)
    assert num_phys == graph.number_of_edges() == 13
    assert len(edge_index) == 2 * num_phys
    assert len(msg_to_phys) == 2 * num_phys

    undirected = {tuple(sorted(edge)) for edge in edge_index}
    assert len(undirected) == num_phys

    lookup = {(u, v): phys for (u, v), phys in zip(edge_index, msg_to_phys)}
    for (u, v), phys in zip(edge_index, msg_to_phys):
        assert lookup[(v, u)] == phys


def test_branch_edges_are_physical_ids_and_match_the_nodes():
    """branch 的 physical_edges 必须就是它节点序列上的那几条边。"""
    graph = make_manual_graph()[0]
    segments = _segments()
    lookup = physical_edge_lookup(graph)
    for branch in segments.branches_flat():
        expected = [
            lookup[(u, v)] for u, v in zip(branch.nodes[:-1], branch.nodes[1:])
        ]
        assert branch.physical_edges == expected
        assert all(0 <= e < segments.num_physical_edges for e in branch.physical_edges)


def test_different_owners_can_share_physical_edges():
    """J1 -> ... -> J2 与 J2 -> ... -> J1 是两个 candidate，但覆盖同一批物理边。"""
    segments = _segments()
    j1 = [b for b in _j1_branches(segments) if b.nodes == J1_BRANCH_NODES]
    j2 = [
        b
        for b in segments.branches[segments.decision_nodes.index(J2)]
        if b.nodes == list(reversed(J1_BRANCH_NODES))
    ]
    assert len(j1) == 1 and len(j2) == 1
    assert sorted(j1[0].physical_edges) == sorted(j2[0].physical_edges)


# ---------------------------------------------------------------------------
# dead-end handling
# ---------------------------------------------------------------------------
def test_dead_end_branch_stops_at_degree_one_node():
    segments = _segments()
    via_h = [b for b in _j1_branches(segments) if b.end == I]
    assert len(via_h) == 1
    assert via_h[0].nodes == [J1, H, I]


# ---------------------------------------------------------------------------
# relabelling / argument handling
# ---------------------------------------------------------------------------
def test_non_contiguous_labels_are_relabelled():
    """任意可排序标签都能用：内部重编号成 0..N-1，结构必须不变。"""
    graph = make_manual_graph()[0]
    shift = 100
    relabelled = nx.relabel_nodes(
        graph, {node: node + shift for node in graph.nodes()}
    )
    segments = extract_segments(
        set_od(relabelled, S + shift, G + shift), S + shift, G + shift
    )

    assert segments.num_nodes == 13
    assert sorted(segments.decision_nodes) == [J1, J2]
    assert segments.start == S and segments.goal == G
    assert segments.node_type[segments.start] == START
    assert segments.node_type[segments.goal] == GOAL
    assert segments.node_type[J1] == JUNCTION and segments.node_type[C] == ORDINARY
    lengths = sorted(len(b.nodes) for b in segments.branches_flat())
    assert lengths == [2, 3, 3, 5, 5, 5, 5]


def test_relabel_false_keeps_the_original_labels():
    """relabel=False 时保留原始编号。"""
    graph = make_manual_graph()[0]
    shift = 100
    relabelled = nx.relabel_nodes(
        graph, {node: node + shift for node in graph.nodes()}
    )
    segments = extract_segments(relabelled, S + shift, G + shift, relabel=False)
    assert segments.start == S + shift
    assert segments.goal == G + shift
    assert segments.num_nodes == 13
    assert segments.decision_nodes == [J1 + shift, J2 + shift]
    assert segments.node_type[S + shift] == START
    assert segments.node_type[G + shift] == GOAL
    assert segments.node_type[J1 + shift] == JUNCTION


def test_unknown_od_pair_raises():
    graph = make_manual_graph()[0]
    with pytest.raises(ValueError):
        extract_segments(set_od(graph, S, 999), S, 999)
    with pytest.raises(ValueError):
        extract_segments(set_od(graph, A, A), A, A)


def test_explicit_od_arguments_are_honoured_without_set_od():
    """extract_segments 必须使用传入的 start/goal，而不是 graph 上的旧属性。"""
    graph = make_manual_graph()[0]
    segments = extract_segments(graph, J2, G)
    assert segments.start == J2
    assert segments.goal == G
    # J2 与 J1 都是度为 >= 3 的 junction，所以 decision set 仍是两者
    assert segments.decision_nodes == [J1, J2]

    # 图属性里放另一对 OD，显式参数必须优先
    set_od(graph, S, G)
    other = extract_segments(graph, J2, G)
    assert other.start == J2
    assert other.goal == G
    assert extract_segments(graph, S, G).start == S
    # 端点集合必须跟着显式参数走：以 J2 为 source 时它仍然是 endpoint
    assert set(other.endpoints) == set(build_endpoints(graph, J2, G))
