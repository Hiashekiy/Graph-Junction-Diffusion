"""单出口 Source forced segment 永久 selected（修改清单 P0-1）。

    deg(s) == 1  =>  B_forced 永久 selected
    E_t = E_source-forced ∪ Psi(z_t)

Source 没有 decision variable，所以它的必经段不会被任何 z_t 写入；如果不额外
固定为 selected，这段信息从头到尾都传不出去。
"""

from __future__ import annotations

import networkx as nx
import torch

from src.data.branch_segments import (
    SELECTED,
    UNSELECTED,
    build_decision_nodes,
    build_source_forced_segment,
    extract_segments,
    physical_edge_lookup,
    set_od,
)
from src.data.collate import collate_samples
from src.data.dataset_builder import build_sample
from src.models.edge_state import expand_to_edge_state


def _physical_state(batch, z):
    state = expand_to_edge_state(
        z_t=z,
        candidate_is_null=batch.candidate_is_null,
        branch_edge_ids=batch.branch_edge_ids,
        branch_edge_owner=batch.branch_edge_owner,
        branch_edge_lengths=batch.branch_edge_lengths,
        msg_to_phys_edge=batch.msg_to_phys_edge,
        num_physical_edges=batch.num_physical_edges,
        source_forced_edge_ids=batch.source_forced_edge_ids,
    )
    physical = torch.full((batch.num_physical_edges,), -1, dtype=torch.long)
    physical.scatter_(0, batch.msg_to_phys_edge, state)
    return physical


def _single_exit_graph():
    """s(0) - a(1) - b(2) - J1(3)，J1 有两条出口。"""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)])
    return graph


# ---------------------------------------------------------------------------
# 1. deg(s) == 1 时 Source 不属于 decision set
# ---------------------------------------------------------------------------
def test_single_exit_source_is_not_a_decision_node():
    graph = _single_exit_graph()
    assert graph.degree(0) == 1
    decisions = build_decision_nodes(graph, 0, 6)
    assert 0 not in decisions
    assert 3 in decisions


# ---------------------------------------------------------------------------
# 2. Source -> first endpoint 整段 physical edges 永久 selected
# ---------------------------------------------------------------------------
def test_source_forced_segment_covers_the_whole_path_to_first_endpoint():
    graph = _single_exit_graph()
    segments = extract_segments(set_od(graph, 0, 6), 0, 6)

    assert segments.source_forced_nodes == [0, 1, 2, 3]
    lookup = physical_edge_lookup(graph)
    assert segments.source_forced_edge_ids == [
        lookup[(0, 1)],
        lookup[(1, 2)],
        lookup[(2, 3)],
    ]


def test_forced_edges_are_selected_for_every_z():
    sample = build_sample(_single_exit_graph(), 0, 6)
    batch = collate_samples([sample])
    lookup = physical_edge_lookup(sample.graph)

    # 无论 z_t 怎么变，被迫段都必须是 selected
    z_variants = [
        batch.target_candidate,
        torch.zeros(batch.num_decisions, dtype=torch.long),   # 全 NULL（普通 junction）
    ]
    for decision_index in range(batch.num_decisions):
        null_index = next(
            i
            for i, (owner, is_null) in enumerate(
                zip(batch.candidate_owner, batch.candidate_is_null)
            )
            if int(owner) == decision_index and bool(is_null)
        )
        variant = batch.target_candidate.clone()
        variant[decision_index] = null_index
        z_variants.append(variant)

    forced_edges = [(0, 1), (1, 2), (2, 3)]
    for z in z_variants:
        physical = _physical_state(batch, z)
        for edge in forced_edges:
            assert int(physical[lookup[edge]]) == SELECTED, (edge, z.tolist())


def test_forced_edges_are_marked_before_z_is_applied():
    """把 junction 全设成 NULL，只有被迫段应该还是 selected。"""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)])
    sample = build_sample(graph, 0, 5)
    batch = collate_samples([sample])

    z = batch.target_candidate.clone()
    for decision_index in range(batch.num_decisions):
        z[decision_index] = next(
            i
            for i, (owner, is_null) in enumerate(
                zip(batch.candidate_owner, batch.candidate_is_null)
            )
            if int(owner) == decision_index and bool(is_null)
        )

    physical = _physical_state(batch, z)
    lookup = physical_edge_lookup(sample.graph)
    for edge in [(0, 1), (1, 2), (2, 3)]:
        assert int(physical[lookup[edge]]) == SELECTED, edge
    assert int(physical[lookup[(3, 4)]]) == UNSELECTED
    assert int(physical[lookup[(3, 5)]]) == UNSELECTED


# ---------------------------------------------------------------------------
# 3. 改其它 Junction 的 z_t 不影响 forced segment
# ---------------------------------------------------------------------------
def test_changing_junction_z_does_not_touch_the_forced_segment():
    sample = build_sample(_single_exit_graph(), 0, 6)
    batch = collate_samples([sample])
    lookup = physical_edge_lookup(sample.graph)
    forced = [(0, 1), (1, 2), (2, 3)]

    for decision_index in range(batch.num_decisions):
        for candidate in range(batch.num_candidates):
            if int(batch.candidate_owner[candidate]) != decision_index:
                continue
            z = batch.target_candidate.clone()
            z[decision_index] = candidate
            physical = _physical_state(batch, z)
            for edge in forced:
                assert int(physical[lookup[edge]]) == SELECTED, (
                    decision_index,
                    candidate,
                )


# ---------------------------------------------------------------------------
# 4. deg(s) > 1 时不存在 source_forced_edge_ids
# ---------------------------------------------------------------------------
def test_multi_exit_source_has_no_forced_segment():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)])
    assert graph.degree(0) > 1

    segments = extract_segments(set_od(graph, 0, 5), 0, 5)
    assert segments.source_forced_edge_ids == []
    assert segments.source_forced_nodes == []
    assert 0 in segments.decision_nodes

    sample = build_sample(graph, 0, 5)
    batch = collate_samples([sample])
    assert batch.num_source_forced_edges == 0
    assert batch.source_forced_edge_ids.numel() == 0


def test_build_source_forced_segment_returns_none_for_multi_exit_source():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    endpoints = set(graph.nodes())
    assert build_source_forced_segment(graph, 0, endpoints) is None


# ---------------------------------------------------------------------------
# 5. 多图 batch 下 forced physical edge IDs 偏移正确
# ---------------------------------------------------------------------------
def test_forced_edge_offsets_across_graphs_in_a_batch():
    samples = [
        build_sample(_single_exit_graph(), 0, 6),
        build_sample(_single_exit_graph(), 0, 6),
    ]
    batch = collate_samples(samples)

    per_graph = samples[0].segments.num_physical_edges
    assert batch.num_source_forced_edges == 3 * len(samples)
    expected = [0, 1, 2, per_graph, per_graph + 1, per_graph + 2]
    assert batch.source_forced_edge_ids.tolist() == expected
    assert batch.num_physical_edges == per_graph * len(samples)


def test_forced_edges_of_one_graph_do_not_leak_into_another():
    samples = [
        build_sample(_single_exit_graph(), 0, 6),
        build_sample(_single_exit_graph(), 0, 6),
    ]
    batch = collate_samples(samples)

    z = batch.target_candidate.clone()
    for decision_index in range(batch.num_decisions):
        z[decision_index] = next(
            i
            for i, (owner, is_null) in enumerate(
                zip(batch.candidate_owner, batch.candidate_is_null)
            )
            if int(owner) == decision_index and bool(is_null)
        )

    physical = _physical_state(batch, z)
    per_graph = samples[0].segments.num_physical_edges
    # 两张图各自的被迫段都 selected；非被迫边全 unselected
    for graph_index in range(2):
        offset = graph_index * per_graph
        for local in (0, 1, 2):
            assert int(physical[offset + local]) == SELECTED
        for local in (3, 4, 5, 6):
            assert int(physical[offset + local]) == UNSELECTED


# ---------------------------------------------------------------------------
# 6. 两个 message directions 仍共享相同的 selected state
# ---------------------------------------------------------------------------
def test_forced_edges_share_state_across_both_message_directions():
    sample = build_sample(_single_exit_graph(), 0, 6)
    batch = collate_samples([sample])
    state = expand_to_edge_state(
        z_t=batch.target_candidate,
        candidate_is_null=batch.candidate_is_null,
        branch_edge_ids=batch.branch_edge_ids,
        branch_edge_owner=batch.branch_edge_owner,
        branch_edge_lengths=batch.branch_edge_lengths,
        msg_to_phys_edge=batch.msg_to_phys_edge,
        num_physical_edges=batch.num_physical_edges,
        source_forced_edge_ids=batch.source_forced_edge_ids,
    )
    for index in range(state.numel()):
        for other in range(index + 1, state.numel()):
            if int(batch.msg_to_phys_edge[index]) == int(batch.msg_to_phys_edge[other]):
                assert int(state[index]) == int(state[other])
