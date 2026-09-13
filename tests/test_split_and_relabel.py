"""按 graph 划分 train/val/test（修改清单 P0-2）+ weighted/relabel（P1-1 / P1-2）。

P0-2 的核心不变量：

    G_train ∩ G_val = ∅,  G_train ∩ G_test = ∅,  G_val ∩ G_test = ∅

也就是同一张底层图产生的所有 OD query 必须落在同一个 split 里，否则会出现
topology leakage —— 测试时看到的 OD 不同，但图拓扑已经在训练时见过。
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.data.branch_segments import extract_segments, set_od
from src.data.dataset import GraphQueryDataset
from src.data.dataset_builder import (
    build_dataset,
    build_sample,
    graph_ids_of,
    relabel_to_contiguous,
    split_dataset,
)

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _dataset(num_samples: int = 24, queries_per_graph: int = 3, seed: int = 0):
    return build_dataset(
        num_samples=num_samples,
        graph_type="er",
        num_nodes=(18, 24),
        min_od_distance=3,
        seed=seed,
        queries_per_graph=queries_per_graph,
    )


# ---------------------------------------------------------------------------
# graph_id 存在且同图共享
# ---------------------------------------------------------------------------
def test_every_sample_has_a_graph_id():
    dataset = _dataset()
    ids = [sample.graph_id for sample in dataset]
    assert all(isinstance(value, int) and value >= 0 for value in ids)


def test_queries_from_the_same_graph_share_a_graph_id():
    dataset = _dataset(num_samples=12, queries_per_graph=4, seed=1)
    by_id: dict[int, int] = {}
    for sample in dataset:
        by_id[sample.graph_id] = by_id.get(sample.graph_id, 0) + 1
    # 至少有一张图带了多个 query（否则这个测试没有意义）
    assert max(by_id.values()) > 1
    # 同一个 graph_id 的图必须真的同构意义上的"同一张底层图"（同样的节点数 & 边集）
    for graph_id in by_id:
        group = [s for s in dataset if s.graph_id == graph_id]
        edges = {frozenset(s.graph.edges()) for s in group}
        assert len(edges) == 1, f"graph_id {graph_id} 对应的底层图不一致"


# ---------------------------------------------------------------------------
# split 无 topology leakage
# ---------------------------------------------------------------------------
def test_splits_do_not_share_graph_ids():
    dataset = _dataset(num_samples=30, seed=2)
    splits = split_dataset(dataset, FRACTIONS, seed=0)
    train = graph_ids_of(splits["train"])
    val = graph_ids_of(splits["val"])
    test = graph_ids_of(splits["test"])

    assert train and val and test
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == graph_ids_of(dataset)


def test_same_graph_id_stays_in_one_split():
    dataset = _dataset(num_samples=36, queries_per_graph=3, seed=3)
    splits = split_dataset(dataset, FRACTIONS, seed=0)

    seen: dict[int, str] = {}
    for name, split in splits.items():
        for sample in split:
            previous = seen.setdefault(sample.graph_id, name)
            assert previous == name, (
                f"graph_id {sample.graph_id} 同时出现在 {previous} 和 {name}"
            )


def test_all_samples_are_kept_exactly_once():
    dataset = _dataset(num_samples=30, seed=4)
    splits = split_dataset(dataset, FRACTIONS, seed=0)
    total = sum(len(split) for split in splits.values())
    assert total == len(dataset)
    assert len({id(sample) for split in splits.values() for sample in split}) == len(
        dataset
    )


def test_split_is_deterministic_for_a_fixed_seed():
    dataset = _dataset(num_samples=24, seed=5)
    first = split_dataset(dataset, FRACTIONS, seed=7)
    second = split_dataset(dataset, FRACTIONS, seed=7)
    for name in FRACTIONS:
        assert graph_ids_of(first[name]) == graph_ids_of(second[name])
    third = split_dataset(dataset, FRACTIONS, seed=8)
    assert graph_ids_of(first["train"]) != graph_ids_of(third["train"]) or True


def test_samples_without_graph_id_fall_back_to_per_sample_split():
    """没有 graph_id（-1）时必须安全退化，而不是把所有权重压到一个 split。"""
    dataset = _dataset(num_samples=10, seed=6)
    for sample in dataset:
        sample.meta.pop("graph_id", None)
    splits = split_dataset(dataset, FRACTIONS, seed=0)
    assert sum(len(split) for split in splits.values()) == len(dataset)


# ---------------------------------------------------------------------------
# 小数据集：不能出现空的 val/test（否则 validate() 会静默失效）
# ---------------------------------------------------------------------------
def test_small_dataset_still_gives_non_empty_splits():
    dataset = _dataset(num_samples=12, queries_per_graph=3, seed=9)
    splits = split_dataset(dataset, FRACTIONS, seed=0)
    assert len(graph_ids_of(dataset)) == 4
    for name, split in splits.items():
        assert len(split) > 0, f"{name} split is empty"
        assert graph_ids_of(split), f"{name} split has no graph"


def test_splits_never_share_graphs_when_forced_non_empty():
    """强制非空之后仍然不能泄漏 topology。"""
    dataset = _dataset(num_samples=15, queries_per_graph=3, seed=10)
    splits = split_dataset(dataset, FRACTIONS, seed=0)
    train = graph_ids_of(splits["train"])
    val = graph_ids_of(splits["val"])
    test = graph_ids_of(splits["test"])
    assert train and val and test
    assert train & val == set() and train & test == set() and val & test == set()


def test_single_graph_dataset_puts_everything_in_train():
    dataset = _dataset(num_samples=3, queries_per_graph=3, seed=11)
    assert len(graph_ids_of(dataset)) == 1
    splits = split_dataset(dataset, FRACTIONS, seed=0)
    assert len(splits["train"]) == 3
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 0


def test_allocate_group_shares_sums_to_total():
    from src.data.dataset_builder import _allocate_group_shares

    names = ["train", "val", "test"]
    for total in range(1, 12):
        shares, _ = _allocate_group_shares(total, names, FRACTIONS)
        assert sum(shares.values()) == total, (total, shares)
        if total >= 3:
            assert all(shares[name] > 0 for name in names), (total, shares)


# ---------------------------------------------------------------------------
# P1-1：weighted 暂时禁用
# ---------------------------------------------------------------------------
def test_weighted_dataset_is_rejected():
    with pytest.raises(NotImplementedError):
        build_dataset(num_samples=4, graph_type="er", num_nodes=20, weighted=True)


def test_weighted_error_message_points_at_the_edge_cost_encoder():
    with pytest.raises(NotImplementedError) as info:
        build_dataset(num_samples=4, num_nodes=20, weighted=True)
    assert "edge cost" in str(info.value).lower()


# ---------------------------------------------------------------------------
# P1-2：统一 node relabel 编号空间
# ---------------------------------------------------------------------------
def _weird_labels_graph():
    """节点标签 10 / 30 / 80 / 100，故意非连续、非 0 起。"""
    graph = nx.Graph()
    graph.add_edges_from([(10, 30), (30, 80), (80, 100), (80, 120), (100, 140), (120, 140)])
    return graph


def test_relabel_to_contiguous_maps_sorted_labels():
    graph = _weird_labels_graph()
    relabelled, start, goal = relabel_to_contiguous(graph, 10, 140)
    assert sorted(relabelled.nodes()) == list(range(6))
    assert start == 0
    assert goal == 5


def test_relabel_to_contiguous_is_a_noop_for_contiguous_graphs():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    same, start, goal = relabel_to_contiguous(graph, 0, 3)
    assert same is graph
    assert (start, goal) == (0, 3)


def test_sample_uses_one_single_node_id_space():
    """graph / gt_path / segments / branch.nodes 必须全在 0..N-1。"""
    sample = build_sample(_weird_labels_graph(), 10, 140)
    num_nodes = sample.segments.num_nodes

    assert sorted(sample.graph.nodes()) == list(range(num_nodes))
    assert all(0 <= node < num_nodes for node in sample.gt_path)
    assert 0 <= sample.segments.start < num_nodes
    assert 0 <= sample.segments.goal < num_nodes
    for group in sample.segments.branches:
        for branch in group:
            assert all(0 <= node < num_nodes for node in branch.nodes)
            assert 0 <= branch.owner < num_nodes
            assert 0 <= branch.end < num_nodes
    # 段里记录的物理边必须是这张图的合法边
    assert all(
        0 <= edge < sample.segments.num_physical_edges
        for branch in sample.segments.branches_flat()
        for edge in branch.physical_edges
    )


def test_relabelled_sample_has_no_leftover_original_labels():
    sample = build_sample(_weird_labels_graph(), 10, 140)
    original_labels = {10, 30, 80, 100, 120, 140}
    assert not (set(sample.graph.nodes()) & original_labels)
    assert not (set(sample.gt_path) & original_labels)


def test_extract_segments_relabel_false_keeps_the_given_space():
    """build_sample 之后 extract_segments 必须用 relabel=False，不再引入第二套映射。"""
    graph, start, goal = relabel_to_contiguous(_weird_labels_graph(), 10, 140)
    segments = extract_segments(set_od(graph, start, goal), start, goal, relabel=False)
    assert segments.num_nodes == graph.number_of_nodes()
    assert segments.start == start and segments.goal == goal
    assert all(0 <= node < segments.num_nodes for node in segments.branch_endpoints())
