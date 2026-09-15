"""DiDi 真实数据适配层单测（实施方案第 17-K 节）。

覆盖方案明确点名的检查：

 1. road sequence 能正确转 junction sequence；
 2. 无连续性的 road sequence 被拒绝；
 3. GT 每一步都是真实 graph edge；
 4. weighted edge 全部正且 finite；
 5. 同一物理边两个 message direction 仍共享 cost；
 6. observed GT 不会被 ``build_sample_from_observed_path()`` 换成 Dijkstra；
 7. corridor builder 的函数输入里不使用 GT path；
 8. （见 tests/test_real_path_metrics.py）
 9. train/val/test 的轨迹键无交集；
10. ``flow_steps=1`` 配置实际进入模型。

大部分用例只用**手工造的微型路网**，不依赖 600MB 的原始 CSV；需要真实文件的
端到端用例在文件缺失时 skip，而不是让整个测试文件报错。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import didi_dataset as didi  # noqa: E402
from src.data.branch_segments import (  # noqa: E402
    GOAL,
    ORDINARY,
    START,
    Branch,
    FlatCandidates,
    GraphSegments,
)
from src.data.branch_segments import build_decision_nodes as bs_build_decision_nodes  # noqa: E402
from src.data.dataset import (  # noqa: E402
    GraphQueryDataset,
    target_null_fraction,
    target_null_statistics,
    validate_sample,
)
from src.evaluation import real_path_metrics as rpm  # noqa: E402
from src.data.dataset_builder import (  # noqa: E402
    build_sample,
    build_sample_from_observed_path,
    relabel_graph_and_path_to_contiguous,
)
from src.data.decision_field import (  # noqa: E402
    DecisionField,
    is_closed_cycle_branch,
    validate_decision_field,
)
from src.utils.config import load_config  # noqa: E402

DIDI_ROOT = (
    PROJECT_ROOT / "data/DiDiChengduXian/didi_datasets/datasets/didi_chengdu"
)
DIDI_CONFIG = PROJECT_ROOT / "configs/graph_flow_didi_weighted.yaml"

needs_didi = pytest.mark.skipif(
    not DIDI_ROOT.exists(), reason="DiDi raw data is not present in data/"
)


# ---------------------------------------------------------------------------
# 手工微型路网
# ---------------------------------------------------------------------------
def make_idx2edge(edges):
    return {index: (u, v, 0) for index, (u, v) in enumerate(edges)}


def make_graph(edges, weights=None):
    graph = nx.Graph()
    for index, (u, v) in enumerate(edges):
        weight = 1.0 if weights is None else float(weights[index])
        graph.add_edge(u, v, weight=weight)
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    return graph


#: 一条简单的链式 road 序列：10-11-12-13-14
CHAIN_EDGES = [(10, 11), (11, 12), (12, 13), (13, 14)]


# ---------------------------------------------------------------------------
# 1. road -> junction
# ---------------------------------------------------------------------------
def test_road_path_converts_to_junction_path():
    idx2edge = make_idx2edge(CHAIN_EDGES)
    graph = make_graph(CHAIN_EDGES)
    result = didi.road_path_to_junction_path([0, 1, 2, 3], idx2edge, graph)
    assert result.ok
    assert result.path == [10, 11, 12, 13, 14]
    assert result.method == didi.CONVERSION_STORED_DIRECTION


def test_duplicate_road_ids_are_collapsed():
    idx2edge = make_idx2edge(CHAIN_EDGES)
    graph = make_graph(CHAIN_EDGES)
    result = didi.road_path_to_junction_path([0, 0, 1, 1, 2, 3], idx2edge, graph)
    assert result.ok
    assert result.path == [10, 11, 12, 13, 14]
    assert result.dropped_duplicate_roads == 2


def test_reversed_stored_direction_falls_back_to_endpoint_walk():
    """存储方向不连续时，端点集合兜底仍然能得到正确的 junction 序列。

    真实成都数据实测是 100% 存储方向连续的，所以这条分支只是保险；但它必须
    正确，否则一旦遇到反向存储的 road 就会静默丢样本。
    """
    # 走过 10 -> 11 -> 12 -> 13，但 road 1 存成 (12, 11)
    edges = [(10, 11), (12, 11), (12, 13)]
    idx2edge = make_idx2edge(edges)
    graph = make_graph(edges)

    assert didi._stored_direction_walk(
        [(0, 10, 11), (1, 12, 11), (2, 12, 13)]
    ) is None

    result = didi.road_path_to_junction_path([0, 1, 2], idx2edge, graph)
    assert result.ok, result.reason
    assert result.method == didi.CONVERSION_ENDPOINT_WALK
    assert result.path == [10, 11, 12, 13]


def test_parallel_u_turn_keeps_the_repeated_junction():
    """平行路段掉头（a -> b -> a）必须保留重复 junction，而不是被当成重复删掉。

    真实成都数据里掉头的写法就是这样：相邻两条 road 是同一对 junction 的平行线、
    存储方向相反。约 1.6% 的相邻对属于这一类。
    """
    edges = [(10, 11), (11, 10), (10, 12)]
    idx2edge = make_idx2edge(edges)
    graph = make_graph(edges)
    result = didi.road_path_to_junction_path([0, 1, 2], idx2edge, graph)
    assert result.ok, result.reason
    assert result.path == [10, 11, 10, 12]
    assert result.u_turns == 1
    assert not didi.is_simple_path(result.path)
    # 端点集合兜底在平行掉头上给出同样的序列
    assert didi._endpoint_walk([(0, 10, 11), (1, 11, 10), (2, 10, 12)]) == [
        10, 11, 10, 12,
    ]


# ---------------------------------------------------------------------------
# 2. 拒绝非法序列
# ---------------------------------------------------------------------------
def test_discontinuous_road_path_is_rejected():
    idx2edge = make_idx2edge(CHAIN_EDGES)
    graph = make_graph(CHAIN_EDGES)
    # road 0 与 road 2 不共享任何端点
    result = didi.road_path_to_junction_path([0, 2, 3], idx2edge, graph)
    assert not result.ok
    assert result.path is None
    assert result.reason == "discontinuous"


def test_unknown_road_id_is_rejected():
    idx2edge = make_idx2edge(CHAIN_EDGES)
    graph = make_graph(CHAIN_EDGES)
    result = didi.road_path_to_junction_path([0, 1, 99], idx2edge, graph)
    assert not result.ok
    assert result.reason == "unknown_road_id"


def test_non_edge_transition_is_rejected():
    """相邻 road 共享一个端点，但 pair 拼出来的 junction edge 不在图里。"""
    idx2edge = make_idx2edge(CHAIN_EDGES)
    graph = make_graph([(10, 11), (11, 12), (12, 13)])  # 12-13 存在，13-14 不存在
    result = didi.road_path_to_junction_path([0, 1, 2, 3], idx2edge, graph)
    assert not result.ok
    assert result.reason == "non_edge_transition"


def test_parse_road_path_never_uses_eval():
    assert didi.parse_road_path("[1, 2, 3]") == [1, 2, 3]
    with pytest.raises(Exception):
        didi.parse_road_path("__import__('os').system('ls')")


# ---------------------------------------------------------------------------
# 4. 图构造：weighted edge 正且 finite，平行路段取最小
# ---------------------------------------------------------------------------
def test_parallel_road_segments_fold_to_min_length():
    # road 0/1 是同一对 junction 的平行线（20 与 5），road 2 是另一条
    idx2edge = make_idx2edge([(10, 11), (10, 11), (11, 12)])
    lengths = {0: 20.0, 1: 5.0, 2: 7.0}
    graph, stats = didi.build_global_weighted_graph(idx2edge, lengths)
    assert stats.parallel_segments_merged == 1
    assert stats.unique_junction_edges == 2
    assert graph.edges[10, 11]["weight"] == pytest.approx(5.0)
    assert graph.edges[11, 12]["weight"] == pytest.approx(7.0)


def test_self_loop_roads_are_dropped_and_counted():
    idx2edge = make_idx2edge([(10, 10), (10, 11)])
    graph, stats = didi.build_global_weighted_graph(idx2edge, {0: 3.0, 1: 4.0})
    assert stats.self_loop_segments == 1
    assert graph.number_of_edges() == 1


def test_all_edge_weights_are_positive_and_finite():
    idx2edge = make_idx2edge(CHAIN_EDGES)
    graph, stats = didi.build_global_weighted_graph(
        idx2edge, {index: 5.0 for index in idx2edge}
    )
    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    assert weights
    assert all(w > 0 and w == w and w != float("inf") for w in weights)
    assert stats.edge_length_min > 0


def test_non_positive_length_is_rejected():
    idx2edge = make_idx2edge(CHAIN_EDGES)
    with pytest.raises(didi.DidiDataError):
        didi.build_global_weighted_graph(idx2edge, {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0})


def test_length_column_must_be_declared(tmp_path):
    """方案第 3.2 节：不许猜列名，也不许静默退化成 weight=1。"""
    csv_path = tmp_path / "edge_features.csv"
    csv_path.write_text("road_id,length\n0,10.5\n", encoding="utf-8")
    with pytest.raises(didi.DidiDataError):
        didi.load_edge_lengths(csv_path, None)
    columns, lengths = didi.load_edge_lengths(csv_path, "length")
    assert columns == ["road_id", "length"]
    assert lengths == {0: 10.5}


def test_missing_length_column_is_rejected(tmp_path):
    csv_path = tmp_path / "edge_features.csv"
    csv_path.write_text("road_id,foo\n0,10.5\n", encoding="utf-8")
    with pytest.raises(didi.DidiDataError):
        didi.load_edge_lengths(csv_path, "length")


# ---------------------------------------------------------------------------
# 6. observed GT 不能被 Dijkstra 换掉
# ---------------------------------------------------------------------------
def _detour_graph():
    """0-1-4 是最短路（cost 2），0-2-3-4 是真实司机走的绕路（cost 4）。"""
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 4, weight=1.0)
    graph.add_edge(0, 2, weight=1.0)
    graph.add_edge(2, 3, weight=1.0)
    graph.add_edge(3, 4, weight=2.0)
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    return graph


def test_observed_gt_is_not_replaced_by_dijkstra():
    graph = _detour_graph()
    shortest = [int(v) for v in nx.shortest_path(graph, 0, 4, weight="weight")]
    assert shortest == [0, 1, 4]

    observed = [0, 2, 3, 4]
    sample = build_sample_from_observed_path(graph, observed)
    assert list(sample.gt_path) == observed
    assert list(sample.gt_path) != shortest
    assert sample.meta["gt_source"] == "observed"
    assert sample.start == 0 and sample.goal == 4
    validate_sample(sample)


def test_build_sample_still_uses_shortest_path():
    """旧函数语义不变：synthetic pipeline 完全可复现。"""
    graph = _detour_graph()
    sample = build_sample(graph, 0, 4)
    assert list(sample.gt_path) == [0, 1, 4]


def test_observed_path_relabel_keeps_graph_and_path_in_one_id_space():
    graph = nx.Graph()
    for u, v in [(100, 200), (200, 300), (300, 400)]:
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True
    path = [100, 200, 300, 400]

    relabelled, new_path, mapping = relabel_graph_and_path_to_contiguous(graph, path)
    assert set(relabelled.nodes()) == set(range(relabelled.number_of_nodes()))
    assert new_path == [mapping[node] for node in path]
    assert all(node in relabelled for node in new_path)

    sample = build_sample_from_observed_path(graph, path)
    assert sample.gt_path == new_path
    for node in sample.gt_path:
        assert 0 <= node < sample.num_nodes


def test_observed_path_with_unknown_node_is_rejected():
    graph = _detour_graph()
    with pytest.raises(ValueError):
        relabel_graph_and_path_to_contiguous(graph, [0, 999])


def test_observed_path_must_be_a_real_path():
    graph = _detour_graph()
    with pytest.raises(Exception):
        build_sample_from_observed_path(graph, [0, 4])  # 0-4 不是边


# ---------------------------------------------------------------------------
# local -> global 编号反查表（KLEV/JSEV 的全局 id 空间）
# ---------------------------------------------------------------------------
def _chain_with_junction(offset):
    """offset+0 - offset+1 - offset+2，外加 offset+1 伸出去一条 offset+3。"""
    graph = nx.Graph()
    for u, v in (
        (offset + 0, offset + 1),
        (offset + 1, offset + 2),
        (offset + 1, offset + 3),
    ):
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    return graph


def test_local_to_global_is_stored_on_observed_samples():
    graph = _chain_with_junction(100)
    sample = build_sample_from_observed_path(graph, [100, 101, 102])
    assert sample.gt_path == [0, 1, 2]  # 局部编号
    assert sample.meta["local_to_global"] == [100, 101, 102, 103]


def test_distribution_metrics_use_global_node_ids():
    """两个样本 relabel 后都是 [0,1,2]，但它们是**不同的城市道路**。

    直接拿局部编号做 edge visit 分布，KLEV/JSEV 会把这两条轨迹当成同一条路
    （只有 2 条边）；映射回全局 OSM id 后才是 4 条不同的边。
    """
    sample_a = build_sample_from_observed_path(_chain_with_junction(100), [100, 101, 102])
    sample_b = build_sample_from_observed_path(_chain_with_junction(700), [700, 701, 702])
    assert sample_a.gt_path == sample_b.gt_path == [0, 1, 2]

    naive = rpm.edge_visit_distribution([sample_a.gt_path, sample_b.gt_path])
    assert len(naive) == 2  # ← 错的口径：只有两条"边"

    global_a = rpm.to_global_path(sample_a.gt_path, sample_a.meta["local_to_global"])
    global_b = rpm.to_global_path(sample_b.gt_path, sample_b.meta["local_to_global"])
    assert global_a == [100, 101, 102]
    assert global_b == [700, 701, 702]

    fixed = rpm.edge_visit_distribution([global_a, global_b])
    assert len(fixed) == 4
    assert set(fixed) == {(100, 101), (101, 102), (700, 701), (701, 702)}
    # 映射后两条完全不同的轨迹，JSEV 必须 > 0（局部口径下会是 0）
    assert rpm.jsev(rpm.edge_visit_distribution([global_a]),
                    rpm.edge_visit_distribution([global_b])) > 0.0
    assert rpm.jsev(rpm.edge_visit_distribution([sample_a.gt_path]),
                    rpm.edge_visit_distribution([sample_b.gt_path])) == pytest.approx(0.0)


def test_to_global_path_without_mapping_is_identity():
    assert rpm.to_global_path([0, 1, 2], None) == [0, 1, 2]
    assert rpm.to_global_path([5, 6], []) == [5, 6]


# ---------------------------------------------------------------------------
# decision 计数：Start 不能重复计
# ---------------------------------------------------------------------------
def test_count_decisions_does_not_double_count_start():
    """deg(start) >= 3 时，start 已经属于 {v: deg(v) >= 3}，不能再 +1。"""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4)])
    # start=0 的度是 3 -> 它就是 junction，集合里已经有一个
    assert didi.count_decisions(graph, 0, 4) == 1
    assert len(bs_build_decision_nodes(graph, 0, 4)) == 1
    assert didi.corridor_decision_count(graph, set(graph.nodes()), 0, 4) == 1

    # start 度 == 2（不是 junction）时才需要额外加进去
    graph2 = nx.Graph()
    graph2.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4)])
    assert didi.count_decisions(graph2, 0, 3) == 1
    assert len(bs_build_decision_nodes(graph2, 0, 3)) == 1
    assert didi.corridor_decision_count(graph2, set(graph2.nodes()), 0, 3) == 1

    # goal 无论度多大都不是 decision
    graph3 = nx.Graph()
    graph3.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4)])
    assert didi.count_decisions(graph3, 0, 1) == 0
    assert len(bs_build_decision_nodes(graph3, 0, 1)) == 0


def test_corridor_decision_count_matches_built_sample():
    """扫描用的快速版必须和真正建样本得到的 decision 数一致。"""
    graph = _triangle_graph()
    corridor = didi.build_od_corridor(graph, 0, 4, rho=2.0)
    sample = build_sample_from_observed_path(corridor.graph, [0, 3, 4])
    keep, _cost = didi.corridor_node_set(graph, 0, 4, rho=2.0)
    assert (
        didi.corridor_decision_count(graph, keep, 0, 4) == sample.num_decisions
    )


# ---------------------------------------------------------------------------
# target_null_fraction（标签级）≠ candidate_null_fraction（候选级）
# ---------------------------------------------------------------------------
def _two_decision_graph():
    """0 是 start/junction 且在 GT 上；5 是另一个 junction，GT 不经过它。

        GT = 0 -> 1 -> 4
        5 伸向 2（回到 0）和两个死胡同 7/8
    """
    graph = nx.Graph()
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (5, 7), (5, 8)]:
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    return graph


def test_target_null_fraction_is_decision_level():
    graph = _two_decision_graph()
    sample = build_sample_from_observed_path(graph, [0, 1, 4])
    assert sample.num_decisions == 2  # 0 与 5

    candidates = sample.field.candidates
    assert len(candidates.target_candidate) == 2
    targets = [candidates.candidate_is_null[i] for i in candidates.target_candidate]
    # decision 0 在 GT 上 -> 选 branch；decision 5 不在 -> NULL
    assert targets.count(True) == 1

    assert target_null_fraction(sample) == pytest.approx(0.5)

    # candidate 级口径完全不同（NULL 候选只有 1 个，候选总数 3 + 4 = 7）
    candidate_level = float(np.mean(candidates.candidate_is_null))
    assert candidate_level == pytest.approx(1 / 7)
    assert candidate_level != pytest.approx(0.5)

    stats = target_null_statistics([sample, sample])
    assert stats["num_decisions"] == 4
    assert stats["num_null_targets"] == 2
    assert stats["target_null_fraction"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# closed-cycle branch（corridor 裁剪造成的 O -> X -> Y -> O）的受控放宽
# ---------------------------------------------------------------------------
def _triangle_graph():
    """0 与 1/2 构成一个三角形，另加一条开放分支 0-3-4。

    度：0 -> 3，1 -> 2，2 -> 2，3 -> 2，4 -> 1。endpoint = {0, 4}，
    所以 0 是唯一的 decision node，它有三条 branch：

        0 -> 1 -> 2 -> 0     closed cycle（corridor 裁剪的产物）
        0 -> 2 -> 1 -> 0     closed cycle
        0 -> 3 -> 4          开放 branch，终点是 goal
    """
    graph = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)]:
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    return graph


def test_closed_cycle_branch_allowed_but_not_gt():
    """允许 closed-cycle branch 当候选，但 z_0 绝对不能落在它上面。"""
    graph = _triangle_graph()
    sample = build_sample_from_observed_path(graph, [0, 3, 4])
    validate_sample(sample)

    closed = [b for b in sample.segments.branches_flat() if is_closed_cycle_branch(b)]
    assert len(closed) == 2, [b.nodes for b in sample.segments.branches_flat()]
    for branch in closed:
        assert branch.nodes[0] == branch.nodes[-1] == branch.owner
        assert len(set(branch.nodes)) == len(branch.nodes) - 1  # 内部不重复
        assert branch.end == branch.owner

    chosen = sample.field.decision_branch[0]
    assert chosen is not None
    assert not is_closed_cycle_branch(chosen)
    assert chosen.end == sample.goal

    # 关掉 GT 检查后仍然要拦住"闭合环当正标签"
    field = DecisionField(
        candidates=FlatCandidates(
            candidate_owner=[0],
            candidate_is_null=[False],
            candidate_branch=[closed[0]],
            target_candidate=[0],
            num_candidates=1,
        ),
        decision_branch=[closed[0]],
    )
    with pytest.raises(AssertionError, match="closed-cycle"):
        validate_decision_field(sample.segments, field)


def test_internal_repeat_branch_still_rejected():
    """内部节点重复的 branch 仍然是结构错误，放宽只覆盖"首尾相同"。"""
    open_branch = Branch(owner=0, nodes=[0, 1, 2, 3], physical_edges=[0, 1, 2], end=3)
    closed = Branch(owner=0, nodes=[0, 1, 2, 0], physical_edges=[0, 1, 2], end=0)
    internal_repeat = Branch(
        owner=0, nodes=[0, 1, 2, 1, 3], physical_edges=[0, 1, 2, 3], end=3
    )
    assert not is_closed_cycle_branch(open_branch)
    assert is_closed_cycle_branch(closed)
    assert not is_closed_cycle_branch(internal_repeat)

    segments = GraphSegments(
        num_nodes=5,
        start=0,
        goal=3,
        node_type={0: START, 1: ORDINARY, 2: ORDINARY, 3: GOAL, 4: ORDINARY},
        edge_index=[],
        msg_to_phys_edge=[],
        num_physical_edges=0,
        decision_nodes=[0],
        branches=[[internal_repeat]],
        endpoints=[0, 3],
    )
    field = DecisionField(
        candidates=FlatCandidates(
            candidate_owner=[0],
            candidate_is_null=[False],
            candidate_branch=[internal_repeat],
            target_candidate=[0],
            num_candidates=1,
        ),
        decision_branch=[internal_repeat],
    )
    with pytest.raises(AssertionError, match="visits a node twice"):
        validate_decision_field(segments, field)


# ---------------------------------------------------------------------------
# 7. corridor 不使用 GT
# ---------------------------------------------------------------------------
def test_corridor_builder_signature_has_no_gt_path():
    for function in (didi.corridor_node_set, didi.build_od_corridor):
        parameters = list(inspect.signature(function).parameters)
        for name in parameters:
            assert "gt" not in name.lower(), (function.__name__, parameters)
            assert "path" not in name.lower(), (function.__name__, parameters)


def test_corridor_source_does_not_touch_gt():
    """源码层面守住泄漏点：函数体里不许出现 gt / observed path。"""
    source = inspect.getsource(didi.corridor_node_set)
    source += inspect.getsource(didi.build_od_corridor)
    for forbidden in ("gt_path", "observed", "junction_path"):
        assert forbidden not in source


def test_corridor_keeps_od_and_drops_distant_nodes():
    # 0-1-2-3 是一条直链，99 挂在很远的 98 上
    graph = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 98), (98, 99)]:
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True

    corridor = didi.build_od_corridor(graph, 0, 3, rho=1.0)
    assert corridor is not None
    assert 99 not in corridor.graph
    assert 0 in corridor.graph and 3 in corridor.graph

    wide = didi.build_od_corridor(graph, 0, 3, rho=3.0)
    assert 99 in wide.graph


def test_corridor_respects_max_nodes():
    graph = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 98), (98, 99)]:
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True
    assert didi.build_od_corridor(graph, 0, 3, rho=3.0, max_nodes=3) is None
    assert didi.build_od_corridor(graph, 0, 3, rho=3.0, max_nodes=100) is not None


def test_distance_cache_matches_networkx():
    graph = nx.Graph()
    for u, v, w in [(0, 1, 2.0), (1, 2, 3.0), (0, 2, 9.0)]:
        graph.add_edge(u, v, weight=w)
    cache = didi.ShortestDistanceCache(graph)
    assert cache.distance(0, 2) == pytest.approx(5.0)
    assert cache.distance(1, 0) == pytest.approx(2.0)
    masks = cache.corridor_mask(0, 2, [1.0, 2.0])
    assert set(masks) == {1.0, 2.0}
    assert masks[1.0] == {0, 1, 2}


# ---------------------------------------------------------------------------
# 过滤 / 去重 / 划分（检查 9）
# ---------------------------------------------------------------------------
def test_length_filter_is_an_open_interval():
    cfg = didi.TrajectoryFilter(min_road_segments=10, max_road_segments=100)
    assert not cfg.road_length_ok(10)
    assert cfg.road_length_ok(11)
    assert cfg.road_length_ok(99)
    assert not cfg.road_length_ok(100)


def test_trajectory_reject_reasons():
    cfg = didi.TrajectoryFilter()
    assert didi.trajectory_reject_reason(
        num_road_segments=5, junction_path=[0, 1, 2], filter_cfg=cfg
    ) == "length_invalid"
    assert didi.trajectory_reject_reason(
        num_road_segments=20, junction_path=[0, 1, 0, 2], filter_cfg=cfg
    ) == "non_simple_gt"
    assert didi.trajectory_reject_reason(
        num_road_segments=20, junction_path=[0, 1, 2, 3], filter_cfg=cfg
    ) is None


def _candidate(path, order_id="x"):
    return didi.TrajectoryCandidate(
        order_id=order_id,
        date="20161010",
        junction_path=list(path),
        raw_road_len=len(path),
        junction_len=len(path),
        raw_road_cost=1.0,
        gt_cost=1.0,
        dijkstra_cost=1.0,
        conversion=didi.CONVERSION_STORED_DIRECTION,
    )


def test_dedup_then_split_is_disjoint():
    candidates = [_candidate([0, 1, 2, 3], f"a{i}") for i in range(4)]
    candidates += [_candidate([0, 1, 2, 4], f"b{i}") for i in range(4)]
    candidates += [_candidate([5, 6, 7, 8], f"c{i}") for i in range(4)]

    kept, dropped = didi.deduplicate_candidates(candidates)
    assert dropped == 9
    assert len(kept) == 3

    splits = didi.split_real_paths(
        kept, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0
    )
    keys = {
        name: {candidate.dedup_key() for candidate in rows}
        for name, rows in splits.items()
    }
    assert not (keys["train"] & keys["val"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["val"] & keys["test"])
    assert sum(len(rows) for rows in splits.values()) == len(kept)


def test_split_is_deterministic_for_a_seed():
    candidates = [_candidate([index, index + 1, index + 2]) for index in range(20)]
    first = didi.split_real_paths(candidates, {"train": 0.5, "val": 0.5}, seed=7)
    second = didi.split_real_paths(candidates, {"train": 0.5, "val": 0.5}, seed=7)
    assert [c.order_id for c in first["train"]] == [
        c.order_id for c in second["train"]
    ]


def test_sample_subset_is_deterministic_and_bounded():
    candidates = [_candidate([index, index + 1, index + 2]) for index in range(50)]
    subset = didi.sample_subset(candidates, 10, seed=3)
    assert len(subset) == 10
    assert subset == didi.sample_subset(candidates, 10, seed=3)
    assert didi.sample_subset(candidates, 100, seed=3) == candidates


# ---------------------------------------------------------------------------
# 分层抽样（date × GT 长度）
# ---------------------------------------------------------------------------
def _dated_candidate(date, junction_len, order_id):
    return didi.TrajectoryCandidate(
        order_id=order_id,
        date=date,
        junction_path=list(range(int(junction_len))),
        raw_road_len=int(junction_len),
        junction_len=int(junction_len),
        raw_road_cost=1.0,
        gt_cost=1.0,
        dijkstra_cost=1.0,
        conversion=didi.CONVERSION_STORED_DIRECTION,
    )


def _dated_pool(days=10, per_day=100):
    pool = []
    for day in range(days):
        for index in range(per_day):
            length = 10 + index  # 10 .. 109
            pool.append(_dated_candidate(f"201610{day + 10:02d}", length, f"{day}-{index}"))
    return pool


def test_stratified_subsample_covers_every_date_and_length_bucket():
    pool = _dated_pool()
    subset, stats = didi.stratified_subsample(pool, 300, seed=0, num_length_buckets=3)

    assert len(subset) == 300
    assert len({id(c) for c in subset}) == 300  # 不重复
    assert stats["num_dates_covered"] == stats["num_dates_total"] == 10
    assert len(stats["length_bucket_edges"]) == 2
    # 每个长度层都有样本，且长度分布没有被截断破坏
    for name, row in stats["per_length_bucket"].items():
        assert row["selected"] > 0, name
    selected_lengths = sorted(c.junction_len for c in subset)
    assert selected_lengths[0] < 30 and selected_lengths[-1] > 90


def test_stratified_subsample_is_deterministic_and_proportional():
    pool = _dated_pool()
    first, stats_first = didi.stratified_subsample(pool, 250, seed=7)
    second, stats_second = didi.stratified_subsample(pool, 250, seed=7)
    assert [c.order_id for c in first] == [c.order_id for c in second]
    assert stats_first["per_date"] == stats_second["per_date"]

    # 各日期规模相同 -> 入选数也应该基本相同（不是"前 N 条"）
    counts = [row["selected"] for row in stats_first["per_date"].values()]
    assert max(counts) - min(counts) <= 1, counts


def test_stratified_subsample_without_effect_returns_everything():
    pool = _dated_pool(days=2, per_day=5)
    subset, stats = didi.stratified_subsample(pool, 100, seed=0)
    assert subset == pool
    assert stats["selected"] == len(pool)


# ---------------------------------------------------------------------------
# 5. 同一物理边两个方向共享 cost
# ---------------------------------------------------------------------------
def test_message_edge_cost_shared_between_directions():
    torch = pytest.importorskip("torch")
    from src.data.collate import collate_samples

    graph = _detour_graph()
    sample = build_sample_from_observed_path(graph, [0, 2, 3, 4])
    batch = collate_samples([sample], device="cpu")

    cost = batch.message_edge_cost(normalized=False).tolist()
    physical = list(batch.msg_to_phys_edge)
    by_physical = {}
    for index, physical_id in enumerate(physical):
        by_physical.setdefault(int(physical_id), set()).add(round(cost[index], 9))
    for physical_id, values in by_physical.items():
        assert len(values) == 1, (physical_id, values)
    assert batch.has_edge_cost


# ---------------------------------------------------------------------------
# 10. flow_steps=1 真的进入模型
# ---------------------------------------------------------------------------
def test_didi_config_declares_flow_steps_one():
    config = load_config(str(DIDI_CONFIG))
    assert int(config.get("model.flow_steps")) == 1
    assert bool(config.get("data.weighted", False)) is True
    assert bool(config.get("model.use_edge_cost", False)) is True
    assert bool(config.get("split.split_by_graph", False)) is False
    assert str(config.get("training.selection_metric")) == "path_similarity_score"
    assert str(config.get("data.source")) == "didi_chengdu"


def test_didi_config_flow_steps_reaches_the_model():
    pytest.importorskip("torch")
    from src.training.setup import build_model, model_kwargs

    config = load_config(str(DIDI_CONFIG))
    assert model_kwargs(config)["flow_steps"] == 1
    model = build_model(config)
    assert model.flow_steps == 1
    assert model.max_flow_steps == 1


# ---------------------------------------------------------------------------
# 端到端（需要真实数据）
# ---------------------------------------------------------------------------
@needs_didi
def test_real_didi_end_to_end():
    """真实文件上跑一遍：建图 -> 转换 -> corridor -> GraphSample。"""
    idx2edge = didi.load_dicts(DIDI_ROOT / "dicts.pkl")
    _columns, lengths = didi.load_edge_lengths(DIDI_ROOT / "edge_features.csv", "length")
    graph, stats = didi.build_global_weighted_graph(idx2edge, lengths)

    assert stats.num_nodes > 1000
    assert stats.parallel_segments_merged > 0
    assert stats.num_connected_components == 1

    files = didi.discover_trajectory_files(DIDI_ROOT, "201610*.csv")
    assert len(files) >= 1

    filter_cfg = didi.TrajectoryFilter()
    distance = didi.ShortestDistanceCache(graph)
    built = 0
    for _index, row in didi.iter_trajectory_csv(files[0], max_rows=400):
        try:
            roads = didi.parse_road_path(row["path"])
        except (ValueError, SyntaxError):
            continue
        conversion = didi.road_path_to_junction_path(roads, idx2edge, graph)
        if not conversion.ok or not didi.is_simple_path(conversion.path):
            continue
        if not filter_cfg.road_length_ok(len(roads)):
            continue
        path = conversion.path
        corridor = didi.build_od_corridor(graph, path[0], path[-1], rho=1.5)
        if corridor is None or not didi.path_contained(corridor.graph, path):
            continue
        sample = build_sample_from_observed_path(corridor.graph, path, meta={
            "gt_source": "observed",
            "gt_cost_ratio": didi.path_cost(graph, path)
            / distance.distance(path[0], path[-1]),
        })
        validate_sample(sample)
        assert sample.meta["gt_source"] == "observed"
        # GT 必须是真实路径：从 OSM 编号 relabel 过来之后仍与转换结果一一对应
        assert len(sample.gt_path) == len(path)
        assert sample.gt_path[0] == 0 or sample.gt_path[0] >= 0
        assert len(set(sample.gt_path)) == len(sample.gt_path)
        # Dijkstra 只能当标尺，不能等于 GT（数据里 ~90% 的轨迹都不是最短路）
        shortest = nx.shortest_path_length(
            sample.graph, sample.start, sample.goal, weight="weight"
        )
        assert shortest <= didi.path_cost(sample.graph, sample.gt_path) + 1e-6
        built += 1
        if built >= 3:
            break
    assert built >= 1, "no usable trajectory found in the first CSV"
