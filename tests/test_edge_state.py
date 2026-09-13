"""Edge-State 展开测试（实施指南第 9、25.3、25.4 节）。

    z_t -> edge_state_id_t -> E_edge[edge_state_id_t]

四条硬性质：

1. 选中 `J1-a-b-x-J2` 时，这条 branch 覆盖的**所有**物理边都是 selected；
2. 其它边保持 unselected；
3. 两个 message 方向共享同一个 state；
4. `z_t[J1] = NULL` 时 J1 的任何 branch 都不得写入 selected。
"""

from __future__ import annotations

import networkx as nx
import torch

from src.data.branch_segments import SELECTED, UNSELECTED, physical_edge_lookup
from src.data.collate import collate_samples
from src.data.dataset_builder import build_sample
from src.models.edge_state import EdgeStateEncoder, expand_to_edge_state
from tests.conftest import A, B, C, D, E, G, H, I, J1, J2, S, X


def _state_of_physical_edges(batch, z_t):
    state = expand_to_edge_state(
        z_t=z_t,
        candidate_is_null=batch.candidate_is_null,
        branch_edge_ids=batch.branch_edge_ids,
        branch_edge_owner=batch.branch_edge_owner,
        branch_edge_lengths=batch.branch_edge_lengths,
        msg_to_phys_edge=batch.msg_to_phys_edge,
        num_physical_edges=batch.num_physical_edges,
    )
    physical = torch.full((batch.num_physical_edges,), -1, dtype=torch.long)
    physical.scatter_(0, batch.msg_to_phys_edge, state)
    return state, physical


def _selected_physical_edges(sample, z):
    candidates = sample.field.candidates
    edges = set()
    for decision_index, target in enumerate(z):
        branch = candidates.candidate_branch[int(target)]
        if branch is not None:
            edges.update(branch.physical_edges)
    return edges


def _null_index_for(sample, decision_index):
    candidates = sample.field.candidates
    return next(
        index
        for index, (owner, is_null) in enumerate(
            zip(candidates.candidate_owner, candidates.candidate_is_null)
        )
        if owner == decision_index and is_null
    )


def test_selected_branch_marks_all_its_physical_edges(manual_sample, manual_batch):
    batch = manual_batch
    z0 = batch.target_candidate.clone()
    state, physical = _state_of_physical_edges(batch, z0)

    selected_edges = _selected_physical_edges(manual_sample, z0)
    for phys in range(batch.num_physical_edges):
        expected = SELECTED if phys in selected_edges else UNSELECTED
        assert int(physical[phys]) == expected, f"physical edge {phys}"

    lookup = physical_edge_lookup(manual_sample.graph)
    # GT 的 J1 -> a -> b -> x -> J2 段必须整段 selected
    for edge in [(J1, A), (A, B), (B, X), (X, J2), (J2, G)]:
        assert int(physical[lookup[edge]]) == SELECTED, f"edge {edge}"
    # 没被选中的绕行支路与 dead-end 支路必须 unselected
    for edge in [(J1, D), (D, E), (J1, H), (H, I), (S, C), (C, J1)]:
        assert int(physical[lookup[edge]]) == UNSELECTED, f"edge {edge}"


def test_both_message_directions_share_the_state(manual_sample, manual_batch):
    batch = manual_batch
    state, _ = _state_of_physical_edges(batch, batch.target_candidate.clone())
    for index in range(state.numel()):
        for other in range(index + 1, state.numel()):
            if int(batch.msg_to_phys_edge[index]) == int(batch.msg_to_phys_edge[other]):
                assert int(state[index]) == int(state[other])


def test_null_writes_no_selected_edge(manual_sample, manual_batch):
    """把 J1 的 z 换成 NULL，J1 的四条分支覆盖的边都不该 selected。"""
    batch = manual_batch
    j1 = manual_sample.segments.decision_nodes.index(J1)
    z = batch.target_candidate.clone()
    z[j1] = _null_index_for(manual_sample, j1)

    _, physical = _state_of_physical_edges(batch, z)
    lookup = physical_edge_lookup(manual_sample.graph)
    for edge in [(J1, A), (A, B), (J1, H), (H, I), (J1, D), (D, E), (C, J1)]:
        assert int(physical[lookup[edge]]) == UNSELECTED, f"edge {edge}"
    # J2 仍然是 active（选中 [J2, g]）
    assert int(physical[lookup[(J2, G)]]) == SELECTED


def test_all_junctions_null_leaves_only_the_source_active():
    """source 没有 NULL，所以 junction 全 NULL 时它的 branch 仍然 selected。"""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)])
    sample = build_sample(graph, 0, 6)
    batch = collate_samples([sample])

    z = batch.target_candidate.clone()
    for decision_index, node in enumerate(sample.segments.decision_nodes):
        if node == sample.segments.start:
            continue
        z[decision_index] = _null_index_for(sample, decision_index)

    _, physical = _state_of_physical_edges(batch, z)
    assert (physical == SELECTED).any(), "source branch must stay selected"


def test_encoder_output_shape_and_dtype(manual_batch):
    encoder = EdgeStateEncoder(d_model=16)
    features = encoder(manual_batch, manual_batch.target_candidate)
    assert features.shape == (manual_batch.edge_index.shape[1], 16)
    assert torch.isfinite(features).all()


def test_encoder_has_two_learnable_states():
    encoder = EdgeStateEncoder(d_model=8)
    assert encoder.embedding.num_embeddings == 2
    assert encoder.embedding.weight.requires_grad


def test_selection_is_a_gather_no_grad_needed(manual_batch):
    """edge state 是离散的，展开过程本身不产生梯度。"""
    z = manual_batch.target_candidate.clone().detach()
    state = expand_to_edge_state(
        z,
        manual_batch.candidate_is_null,
        manual_batch.branch_edge_ids,
        manual_batch.branch_edge_owner,
        manual_batch.branch_edge_lengths,
        manual_batch.msg_to_phys_edge,
        manual_batch.num_physical_edges,
    )
    assert state.dtype == torch.long
    assert state.numel() == manual_batch.edge_index.shape[1]


def test_physical_edge_ids_are_offset_across_graphs(tiny_batches):
    """多图 batch：物理边 ID 必须整体平移，否则第二张图会污染第一张图的状态。"""
    samples, batch = tiny_batches
    expected = sum(sample.segments.num_physical_edges for sample in samples)
    assert batch.num_physical_edges == expected

    offsets = []
    running = 0
    for sample in samples:
        offsets.append(running)
        running += sample.segments.num_physical_edges
    assert offsets[1] == samples[0].segments.num_physical_edges

    # 每张图的 branch 物理边必须落在自己那张图的全局编号区间内
    candidate_cursor = 0
    for graph_index, sample in enumerate(samples):
        lo = offsets[graph_index]
        hi = lo + sample.segments.num_physical_edges
        for branch in sample.field.candidates.candidate_branch:
            edges = batch.branch_edge_ids[candidate_cursor]
            length = int(batch.branch_edge_lengths[candidate_cursor])
            for edge in edges[:length].tolist():
                assert lo <= edge < hi, (graph_index, edge, lo, hi)
            candidate_cursor += 1
    assert candidate_cursor == batch.num_candidates
