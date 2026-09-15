"""DiDi 真实数据链路自检（不需要 pytest / torch）。

``tests/test_didi_dataset.py`` 与 ``tests/test_real_path_metrics.py`` 是正式单测，
但它们依赖 ``tests/conftest.py`` -> ``import torch``，在没有 torch 的机器上一条都
跑不起来。这个脚本用**纯 Python assert** 复刻同一批检查（第 5、10 条需要 torch 的
用例单独 skip），用来在数据准备阶段就地验证整条链路：

    python tools/verify_didi_pipeline.py
    python tools/verify_didi_pipeline.py --data data/didi/graph/chengdu

退出码 0 = 全部通过。
"""

from __future__ import annotations

import argparse
import inspect
import math
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import networkx as nx
import numpy as np

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
from src.data.dataset import (  # noqa: E402
    GraphQueryDataset,
    target_null_statistics as null_statistics,
    validate_sample,
)
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
from src.evaluation import real_path_metrics as rpm  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DIDI_ROOT = PROJECT_ROOT / "data/didi/raw/chengdu"
DIDI_CONFIG = PROJECT_ROOT / "configs/didi_chengdu.yaml"

CHECKS: List[Tuple[str, Callable[[], None]]] = []


def check(name: str):
    def decorator(function: Callable[[], None]) -> Callable[[], None]:
        CHECKS.append((name, function))
        return function

    return decorator


# ---------------------------------------------------------------------------
def make_idx2edge(edges):
    return {index: (u, v, 0) for index, (u, v) in enumerate(edges)}


def make_graph(edges, weights=None):
    graph = nx.Graph()
    for index, (u, v) in enumerate(edges):
        graph.add_edge(u, v, weight=1.0 if weights is None else float(weights[index]))
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    return graph


CHAIN = [(10, 11), (11, 12), (12, 13), (13, 14)]


def detour_graph():
    """0-1-4 最短路（cost 2），0-2-3-4 是真实司机路线（cost 4）。"""
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 4, weight=1.0)
    graph.add_edge(0, 2, weight=1.0)
    graph.add_edge(2, 3, weight=1.0)
    graph.add_edge(3, 4, weight=2.0)
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    return graph


# ---------------------------------------------------------------------------
@check("1. road path -> junction path（存储方向）")
def check_stored_direction():
    idx2edge = make_idx2edge(CHAIN)
    result = didi.road_path_to_junction_path([0, 1, 2, 3], idx2edge, make_graph(CHAIN))
    assert result.ok, result.reason
    assert result.path == [10, 11, 12, 13, 14], result.path
    assert result.method == didi.CONVERSION_STORED_DIRECTION


@check("1b. 端点集合兜底（存储方向反向）")
def check_endpoint_fallback():
    edges = [(10, 11), (12, 11), (12, 13)]
    idx2edge = make_idx2edge(edges)
    assert didi._stored_direction_walk([(0, 10, 11), (1, 12, 11), (2, 12, 13)]) is None
    result = didi.road_path_to_junction_path([0, 1, 2], idx2edge, make_graph(edges))
    assert result.ok, result.reason
    assert result.method == didi.CONVERSION_ENDPOINT_WALK
    assert result.path == [10, 11, 12, 13], result.path


@check("1c. 平行路段掉头保留重复 junction")
def check_u_turn():
    # road 0 与 road 1 是同一对 junction 的平行线、存储方向相反（真实数据里
    # 掉头就是这个样子），随后继续沿 road 2 走
    edges = [(10, 11), (11, 10), (10, 12)]
    result = didi.road_path_to_junction_path(
        [0, 1, 2], make_idx2edge(edges), make_graph(edges)
    )
    assert result.ok, result.reason
    assert result.path == [10, 11, 10, 12], result.path
    assert result.u_turns == 1
    assert not didi.is_simple_path(result.path)
    # 端点集合兜底在平行掉头上给出同样的序列
    fallback = didi._endpoint_walk([(0, 10, 11), (1, 11, 10), (2, 10, 12)])
    assert fallback == [10, 11, 10, 12], fallback


@check("2. 非连续 road 序列被拒绝")
def check_discontinuous():
    result = didi.road_path_to_junction_path(
        [0, 2, 3], make_idx2edge(CHAIN), make_graph(CHAIN)
    )
    assert not result.ok and result.reason == "discontinuous", result


@check("2b. 未知 road id 被拒绝")
def check_unknown_road():
    result = didi.road_path_to_junction_path(
        [0, 1, 99], make_idx2edge(CHAIN), make_graph(CHAIN)
    )
    assert not result.ok and result.reason == "unknown_road_id", result


@check("3. GT 每一步都是真实 graph edge")
def check_gt_edges_are_real():
    graph = detour_graph()
    sample = build_sample_from_observed_path(graph, [0, 2, 3, 4])
    for u, v in zip(sample.gt_path[:-1], sample.gt_path[1:]):
        assert sample.graph.has_edge(u, v), (u, v)
    validate_sample(sample)


@check("4. edge weight 全部 > 0 且 finite")
def check_weights():
    graph, stats = didi.build_global_weighted_graph(
        make_idx2edge(CHAIN), {index: 5.0 for index in range(len(CHAIN))}
    )
    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    assert weights and all(w > 0 and w == w and abs(w) != float("inf") for w in weights)
    assert stats.edge_length_min > 0
    try:
        didi.build_global_weighted_graph(
            make_idx2edge(CHAIN), {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}
        )
    except didi.DidiDataError:
        pass
    else:
        raise AssertionError("non-positive length must be rejected")


@check("4b. 平行 road segment 折叠成最短长度")
def check_parallel_fold():
    idx2edge = make_idx2edge([(10, 11), (10, 11), (11, 12)])
    graph, stats = didi.build_global_weighted_graph(
        idx2edge, {0: 20.0, 1: 5.0, 2: 7.0}
    )
    assert stats.parallel_segments_merged == 1
    assert stats.unique_junction_edges == 2
    assert graph.edges[10, 11]["weight"] == 5.0


@check("4c. 长度列必须显式声明")
def check_length_column_required(tmp: Path):
    csv_path = tmp / "edge_features.csv"
    csv_path.write_text("road_id,length\n0,10.5\n", encoding="utf-8")
    try:
        didi.load_edge_lengths(csv_path, None)
    except didi.DidiDataError:
        pass
    else:
        raise AssertionError("missing length_column must raise")
    _columns, lengths = didi.load_edge_lengths(csv_path, "length")
    assert lengths == {0: 10.5}


@check("5. closed-cycle branch 允许存在但绝不能是 GT")
def check_closed_cycle_branch():
    # 0 与 1/2 构成三角形（corridor 裁剪的产物），另加一条开放分支 0-3-4
    graph = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4)]:
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"

    sample = build_sample_from_observed_path(graph, [0, 3, 4])
    validate_sample(sample)
    branches = sample.segments.branches_flat()
    closed = [b for b in branches if is_closed_cycle_branch(b)]
    assert len(closed) == 2, [b.nodes for b in branches]
    for branch in closed:
        assert branch.nodes[0] == branch.nodes[-1] == branch.owner
        assert len(set(branch.nodes)) == len(branch.nodes) - 1
    chosen = sample.field.decision_branch[0]
    assert chosen is not None and not is_closed_cycle_branch(chosen)
    assert chosen.end == sample.goal

    # 闭合环当正标签 -> 必须被拦下（关掉 GT 匹配检查后仍然要拦）
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
    try:
        validate_decision_field(sample.segments, field)
    except AssertionError as error:
        assert "closed-cycle" in str(error), error
    else:
        raise AssertionError("a closed-cycle branch must never be a GT target")

    # 内部节点重复仍然是结构错误（放宽只覆盖"首尾相同"）
    internal_repeat = Branch(
        owner=0, nodes=[0, 1, 2, 1, 3], physical_edges=[0, 1, 2, 3], end=3
    )
    assert not is_closed_cycle_branch(internal_repeat)
    assert is_closed_cycle_branch(
        Branch(owner=0, nodes=[0, 1, 2, 0], physical_edges=[0, 1, 2], end=0)
    )
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
    bad_field = DecisionField(
        candidates=FlatCandidates(
            candidate_owner=[0],
            candidate_is_null=[False],
            candidate_branch=[internal_repeat],
            target_candidate=[0],
            num_candidates=1,
        ),
        decision_branch=[internal_repeat],
    )
    try:
        validate_decision_field(segments, bad_field)
    except AssertionError as error:
        assert "visits a node twice" in str(error), error
    else:
        raise AssertionError("an internally-repeating branch must still be rejected")


@check("9b. 分层抽样覆盖每个日期与每个长度层")
def check_stratified_subsample():
    def make(date, length, order_id):
        return didi.TrajectoryCandidate(
            order_id=order_id,
            date=date,
            junction_path=list(range(int(length))),
            raw_road_len=int(length),
            junction_len=int(length),
            raw_road_cost=1.0,
            gt_cost=1.0,
            dijkstra_cost=1.0,
            conversion=didi.CONVERSION_STORED_DIRECTION,
        )

    pool = [
        make(f"201610{day + 10:02d}", 10 + index, f"{day}-{index}")
        for day in range(10)
        for index in range(100)
    ]
    subset, stats = didi.stratified_subsample(pool, 300, seed=0)
    assert len(subset) == 300
    assert stats["num_dates_covered"] == stats["num_dates_total"] == 10, stats["per_date"]
    for name, row in stats["per_length_bucket"].items():
        assert row["selected"] > 0, name
    again, _ = didi.stratified_subsample(pool, 300, seed=0)
    assert [c.order_id for c in again] == [c.order_id for c in subset]
    counts = [row["selected"] for row in stats["per_date"].values()]
    assert max(counts) - min(counts) <= 1, counts


@check("6. observed GT 不会被 Dijkstra 替换")
def check_observed_gt():
    graph = detour_graph()
    shortest = [int(v) for v in nx.shortest_path(graph, 0, 4, weight="weight")]
    assert shortest == [0, 1, 4], shortest

    observed = [0, 2, 3, 4]
    sample = build_sample_from_observed_path(graph, observed)
    assert list(sample.gt_path) == observed, sample.gt_path
    assert list(sample.gt_path) != shortest
    assert sample.meta["gt_source"] == "observed"
    validate_sample(sample)

    # 旧函数语义不变
    assert list(build_sample(graph, 0, 4).gt_path) == [0, 1, 4]


@check("6b. 非边的 observed path 被拒绝")
def check_bad_observed_path():
    try:
        build_sample_from_observed_path(detour_graph(), [0, 4])
    except Exception:
        return
    raise AssertionError("a path with a non-edge must be rejected")


@check("6c. relabel 把 graph 与整条 GT 放进同一编号空间")
def check_relabel():
    graph = nx.Graph()
    for u, v in [(100, 200), (200, 300), (300, 400)]:
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True
    path = [100, 200, 300, 400]
    relabelled, new_path, mapping = relabel_graph_and_path_to_contiguous(graph, path)
    assert set(relabelled.nodes()) == set(range(relabelled.number_of_nodes()))
    assert new_path == [mapping[node] for node in path]
    sample = build_sample_from_observed_path(graph, path)
    assert sample.gt_path == new_path, (sample.gt_path, new_path)
    assert all(0 <= node < sample.num_nodes for node in sample.gt_path)
    try:
        relabel_graph_and_path_to_contiguous(graph, [0, 999])
    except ValueError:
        return
    raise AssertionError("unknown gt_path node must raise")


@check("7. corridor builder 的输入里没有 GT path")
def check_corridor_signature():
    for function in (didi.corridor_node_set, didi.build_od_corridor):
        for name in inspect.signature(function).parameters:
            assert "gt" not in name.lower(), (function.__name__, name)
            assert "path" not in name.lower(), (function.__name__, name)
    source = inspect.getsource(didi.corridor_node_set) + inspect.getsource(
        didi.build_od_corridor
    )
    for forbidden in ("gt_path", "observed", "junction_path"):
        assert forbidden not in source, forbidden


@check("7b. corridor 只保留 OD 附近的节点")
def check_corridor_content():
    graph = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 98), (98, 99)]:
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True
    narrow = didi.build_od_corridor(graph, 0, 3, rho=1.0)
    assert narrow is not None and 99 not in narrow.graph
    wide = didi.build_od_corridor(graph, 0, 3, rho=3.0)
    assert 99 in wide.graph
    assert didi.build_od_corridor(graph, 0, 3, rho=3.0, max_nodes=3) is None


@check("7c. 距离缓存与 networkx 一致")
def check_distance_cache():
    graph = nx.Graph()
    for u, v, w in [(0, 1, 2.0), (1, 2, 3.0), (0, 2, 9.0)]:
        graph.add_edge(u, v, weight=w)
    cache = didi.ShortestDistanceCache(graph)
    assert cache.distance(0, 2) == 5.0
    assert cache.distance(1, 0) == 2.0
    masks = cache.corridor_mask(0, 2, [1.0, 2.0])
    assert set(masks) == {1.0, 2.0}
    assert masks[1.0] == {0, 1, 2}


@check("8. LCS / Edge F1 手算样例")
def check_lcs_edges():
    assert rpm.lcs_length([1, 2, 3, 4, 5], [2, 4, 5, 6]) == 3
    assert rpm.lcs_length([1, 2, 3], [4, 5, 6]) == 0
    assert rpm.normalized_lcs([1, 2, 3, 4, 5], [2, 4, 5, 6]) == 3 / 4
    precision, recall, f1 = rpm.paired_edge_prf([0, 1, 2, 3], [0, 1, 4, 3])
    assert abs(precision - 1 / 3) < 1e-12
    assert abs(recall - 1 / 3) < 1e-12
    assert abs(f1 - 1 / 3) < 1e-12
    assert rpm.edge_set([0, 1, 2]) == rpm.edge_set([2, 1, 0])
    assert rpm.paired_edge_prf([0, 1, 2], [0, 1, 2]) == (1.0, 1.0, 1.0)


@check("8b. KLEV / JSEV 手算样例")
def check_divergences():
    import math

    gt = {("a", "b"): 3.0, ("b", "c"): 1.0}
    pred = {("a", "b"): 1.0, ("b", "c"): 1.0}
    expected_klev = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert abs(rpm.klev(gt, pred) - expected_klev) < 1e-9
    assert abs(expected_klev - 0.1308120) < 1e-6
    assert abs(rpm.jsev(gt, pred) - 0.0338220) < 1e-6
    assert abs(rpm.klev(gt, dict(gt))) < 1e-9
    assert abs(rpm.jsev(gt, dict(gt))) < 1e-9
    swapped = rpm.jsev(pred, gt)
    assert abs(swapped - rpm.jsev(gt, pred)) < 1e-9
    # pred 一条边都没有 -> 平滑后 q 退化成均匀分布 (0.5, 0.5)，KLEV 仍是有限值
    assert abs(rpm.klev(gt, {}) - expected_klev) < 1e-9
    # 没到 goal 的 query 记 0
    missed = rpm.pair_record([0, 1, 2, 3], [0, 1, 2, 3], goal_hit=False)
    assert missed.normalized_lcs == 1.0 and missed.normalized_lcs_selection == 0.0


@check("8c. 分桶（length / decision）")
def check_buckets():
    class Stub:
        def __init__(self, gt_length, num_decisions):
            self.gt_length = gt_length
            self.num_decisions = num_decisions
            self.meta = {}

    dataset = [Stub(index + 1, value) for index, value in enumerate([1, 2, 3, 4, 5, 6])]
    groups = rpm.length_buckets(dataset, 3)
    assert groups["short"] == [0, 1] and groups["medium"] == [2, 3]
    assert groups["long"] == [4, 5]

    decisions = [Stub(20, value) for value in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]]
    buckets = rpm.decision_buckets(decisions)
    assert buckets["1-3"] == [0, 1, 2] and buckets["4-6"] == [3, 4, 5]
    assert buckets["7-9"] == [6, 7, 8] and buckets[">=10"] == [9, 10]

    records = [
        rpm.pair_record([0, 1, 2], [0, 1, 2], goal_hit=True).to_dict(),
        None,
        rpm.pair_record([0, 1, 9], [0, 1, 2], goal_hit=False).to_dict(),
        None,
    ]
    report = rpm.bucket_report(dataset, records, {"all": [0, 1, 2, 3]})
    assert report["all"]["real_num_queries"] == 2
    assert abs(report["all"]["path_similarity_score"] - 0.5) < 1e-12


@check("8d. PathSimilarityScore 对失败样本记 0 + Pred/GT cost ratio")
def check_path_similarity_and_cost_ratio():
    missed = rpm.pair_record([0, 1, 2, 3], [0, 1, 2, 3], goal_hit=False)
    assert missed.normalized_lcs == 1.0 and missed.normalized_lcs_selection == 0.0

    # "50% 到达但每次都完美" 不该赢过 "100% 到达但只像一半"
    model_a = [
        rpm.pair_record([0, 1, 2, 3], [0, 1, 2, 3], goal_hit=True),
        rpm.pair_record([0, 1, 2, 3], [0, 1, 2, 3], goal_hit=False),
    ]
    model_b = [
        rpm.pair_record([0, 1, 9, 3], [0, 1, 2, 3], goal_hit=True),
        rpm.pair_record([0, 1, 9, 3], [0, 1, 2, 3], goal_hit=True),
    ]
    score_a = rpm.aggregate_pair_records(model_a)["path_similarity_score"]
    score_b = rpm.aggregate_pair_records(model_b)["path_similarity_score"]
    assert abs(score_a - 0.5) < 1e-12 and abs(score_b - 0.75) < 1e-12
    assert score_b > score_a

    # C(P_pred) / C(P_GT)
    record = rpm.pair_record(
        [0, 1, 2], [0, 1, 2], goal_hit=True, gt_cost_ratio=1.2, pred_cost_ratio=1.5
    )
    assert abs(record.pred_over_gt_cost_ratio - 1.25) < 1e-12
    aggregate = rpm.aggregate_pair_records([record])
    assert abs(aggregate["pred_over_gt_cost_ratio"] - 1.25) < 1e-12
    assert math.isnan(
        rpm.pair_record([0, 1, 2], [0, 1, 2], goal_hit=True).pred_over_gt_cost_ratio
    )


@check("8e. KLEV/JSEV 必须在全局 OSM id 空间统计")
def check_global_id_distribution():
    def chain(offset):
        graph = nx.Graph()
        for u, v in ((offset, offset + 1), (offset + 1, offset + 2),
                     (offset + 1, offset + 3)):
            graph.add_edge(u, v, weight=1.0)
        graph.graph["weighted"] = True
        graph.graph["weight"] = "weight"
        return graph

    a = build_sample_from_observed_path(chain(100), [100, 101, 102])
    b = build_sample_from_observed_path(chain(700), [700, 701, 702])
    assert a.gt_path == b.gt_path == [0, 1, 2], (a.gt_path, b.gt_path)
    assert a.meta["local_to_global"] == [100, 101, 102, 103]
    assert b.meta["local_to_global"] == [700, 701, 702, 703]

    naive = rpm.edge_visit_distribution([a.gt_path, b.gt_path])
    assert len(naive) == 2, naive          # 错的口径：两条不同的路被当成同一条
    ga = rpm.to_global_path(a.gt_path, a.meta["local_to_global"])
    gb = rpm.to_global_path(b.gt_path, b.meta["local_to_global"])
    fixed = rpm.edge_visit_distribution([ga, gb])
    assert len(fixed) == 4, fixed
    assert rpm.jsev(rpm.edge_visit_distribution([ga]),
                    rpm.edge_visit_distribution([gb])) > 0.0
    assert rpm.to_global_path([5, 6], None) == [5, 6]


@check("8f. decision 计数不重复算 Start")
def check_decision_count():
    from src.data.branch_segments import build_decision_nodes as reference

    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4)])   # deg(start)=3
    assert didi.count_decisions(graph, 0, 4) == len(reference(graph, 0, 4)) == 1
    assert didi.corridor_decision_count(graph, set(graph.nodes()), 0, 4) == 1

    graph2 = nx.Graph()
    graph2.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4)])  # deg(start)=2
    assert didi.count_decisions(graph2, 0, 3) == len(reference(graph2, 0, 3)) == 1

    graph3 = nx.Graph()
    graph3.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4)])  # goal 是 junction
    assert didi.count_decisions(graph3, 0, 1) == len(reference(graph3, 0, 1)) == 0


@check("8g. target_null_fraction 是标签级（不是候选级）")
def check_target_null_fraction():
    from src.data.dataset import target_null_fraction, target_null_statistics

    graph = nx.Graph()
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (5, 7), (5, 8)]:
        graph.add_edge(u, v, weight=1.0)
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    sample = build_sample_from_observed_path(graph, [0, 1, 4])
    assert sample.num_decisions == 2
    assert abs(target_null_fraction(sample) - 0.5) < 1e-12
    candidate_level = float(np.mean(sample.field.candidates.candidate_is_null))
    assert abs(candidate_level - 1 / 7) < 1e-12
    assert abs(candidate_level - 0.5) > 0.1      # 两个口径必须明显不同
    stats = target_null_statistics([sample, sample])
    assert stats["num_decisions"] == 4 and stats["num_null_targets"] == 2
    assert abs(stats["target_null_fraction"] - 0.5) < 1e-12


@check("8h. km-based DTW 手算样例")
def check_dtw():
    def meridian(count, step=0.001):
        return [(104.0, 30.0 + index * step) for index in range(count)]

    points = meridian(4)
    coords = {index: point for index, point in enumerate(points)}
    same = rpm.dtw_distance_km([0, 1, 2, 3], [0, 1, 2, 3], coords)
    assert same.ok and abs(same.total_km) < 1e-12 and same.warping_steps == 4

    pred = ["p0", "p1", "p2"]
    coords.update({"p0": points[0], "p1": (104.0, 30.0021), "p2": points[3]})
    result = rpm.dtw_distance_km(pred, [0, 1, 2, 3], coords)
    expected = (
        rpm.haversine_km(104.0, 30.0000, 104.0, 30.0010)
        + rpm.haversine_km(104.0, 30.0021, 104.0, 30.0020)
    )
    assert abs(result.total_km - expected) < 1e-9, (result.total_km, expected)
    assert result.warping_steps == 4
    assert abs(result.mean_km - expected / 4) < 1e-9
    naive = (
        rpm.haversine_km(104.0, 30.0021, 104.0, 30.0010)
        + rpm.haversine_km(104.0, 30.0030, 104.0, 30.0020)
    )
    assert result.total_km < naive
    assert not rpm.dtw_distance_km([0, 1], [0, 1, 99], coords).ok
    assert abs(rpm.haversine_km(104.0, 30.0, 104.0, 30.01) - 1.11195) < 1e-3


@check("9. 去重先于 split，且 train/val/test 无交集")
def check_split_disjoint():
    def candidate(path, order_id):
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

    candidates = [candidate([0, 1, 2, 3], f"a{i}") for i in range(4)]
    candidates += [candidate([0, 1, 2, 4], f"b{i}") for i in range(4)]
    candidates += [candidate([5, 6, 7, 8], f"c{i}") for i in range(4)]

    kept, dropped = didi.deduplicate_candidates(candidates)
    assert dropped == 9 and len(kept) == 3, (dropped, len(kept))
    splits = didi.split_real_paths(kept, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    keys = {name: {c.dedup_key() for c in rows} for name, rows in splits.items()}
    assert not (keys["train"] & keys["val"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["val"] & keys["test"])
    again = didi.split_real_paths(kept, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    assert [c.order_id for c in again["train"]] == [c.order_id for c in splits["train"]]


@check("10. flow_steps=1 / weighted / split_by_graph=false 进入配置")
def check_config():
    config = load_config(str(DIDI_CONFIG))
    assert int(config.get("model.flow_steps")) == 1
    assert bool(config.get("data.weighted", False)) is True
    assert bool(config.get("model.use_edge_cost", False)) is True
    assert bool(config.get("split.split_by_graph", False)) is False
    assert str(config.get("training.selection_metric")) == "path_similarity_score"
    assert str(config.get("data.source")) == "didi_chengdu"
    assert str(config.get("data.length_column")) == "length"
    assert float(config.get("data.corridor.rho")) > 1.0
    try:
        import torch  # noqa: F401
    except ImportError:
        print("      (skip: torch not installed, cannot build the model here)")
        return
    from src.training.setup import build_model, model_kwargs

    assert model_kwargs(config)["flow_steps"] == 1
    model = build_model(config)
    assert model.flow_steps == 1 and model.max_flow_steps == 1


# ---------------------------------------------------------------------------
# 真实数据端到端
# ---------------------------------------------------------------------------
def check_real_data(limit: int = 4000) -> None:
    if not DIDI_ROOT.exists():
        print("[skip] DiDi raw data not present")
        return
    idx2edge = didi.load_dicts(DIDI_ROOT / "dicts.pkl")
    _columns, lengths = didi.load_edge_lengths(DIDI_ROOT / "edge_features.csv", "length")
    graph, stats = didi.build_global_weighted_graph(idx2edge, lengths)
    assert stats.num_connected_components == 1
    assert stats.parallel_segments_merged > 0
    print(
        f"      graph: {stats.num_nodes} nodes / {stats.unique_junction_edges} edges "
        f"({stats.parallel_segments_merged} parallel merged, "
        f"{stats.self_loop_segments} self loops dropped)"
    )

    files = didi.discover_trajectory_files(DIDI_ROOT, "201610*.csv")
    filter_cfg = didi.TrajectoryFilter()
    distance = didi.ShortestDistanceCache(graph)
    funnel = didi.FunnelStats()
    built = 0
    for file_path in files[:1]:
        for _index, row in didi.iter_trajectory_csv(file_path, max_rows=limit):
            funnel.add("raw_rows")
            try:
                roads = didi.parse_road_path(row["path"])
            except (ValueError, SyntaxError, TypeError):
                funnel.add("parse_failed")
                continue
            funnel.add("parse_valid")
            conversion = didi.road_path_to_junction_path(roads, idx2edge, graph)
            if not conversion.ok:
                funnel.add(f"reject_{conversion.reason}")
                continue
            funnel.add("continuous_valid")
            if not didi.is_simple_path(conversion.path):
                funnel.add("reject_non_simple_gt")
                continue
            if not filter_cfg.road_length_ok(len(roads)):
                funnel.add("reject_length_invalid")
                continue
            funnel.add("length_valid")
            path = conversion.path
            corridor = didi.build_od_corridor(graph, path[0], path[-1], rho=1.5)
            if corridor is None:
                funnel.add("corridor_too_large")
                continue
            if not didi.path_contained(corridor.graph, path):
                funnel.add("corridor_miss")
                continue
            sample = build_sample_from_observed_path(
                corridor.graph,
                path,
                meta={
                    "gt_source": "observed",
                    "gt_cost_ratio": didi.path_cost(graph, path)
                    / distance.distance(path[0], path[-1]),
                },
            )
            validate_sample(sample)
            assert sample.meta["gt_source"] == "observed"
            assert len(sample.gt_path) == len(path) == len(set(path))
            assert didi.path_cost(sample.graph, sample.gt_path) >= (
                nx.shortest_path_length(
                    sample.graph, sample.start, sample.goal, weight="weight"
                )
                - 1e-6
            )
            funnel.add("built")
            built += 1
            if built >= 50:
                break
        if built >= 50:
            break
    assert built >= 10, f"only {built} samples built; funnel={dict(funnel.counts)}"
    print(f"      built {built} real samples; funnel={dict(funnel.counts)}")


def check_built_dataset(data_dir: Path) -> None:
    if not (data_dir / "train.pkl").exists():
        print(f"[skip] {data_dir}/train.pkl not found")
        return
    from src.evaluation import real_path_metrics as metrics

    splits = {}
    for name in ("train", "val", "test"):
        dataset = GraphQueryDataset.load(data_dir / f"{name}.pkl")
        splits[name] = dataset
        assert len(dataset) > 0, name
        for sample in dataset:
            assert sample.meta.get("gt_source") == "observed", name
            assert float(sample.meta["gt_cost_ratio"]) >= 1.0 - 1e-9
            assert sample.gt_path[0] == sample.start
            assert sample.gt_path[-1] == sample.goal
            assert len(set(sample.gt_path)) == len(sample.gt_path)
            # KLEV/JSEV 的全局 id 空间依赖这张反查表，缺了会静默算错
            mapping = sample.meta.get("local_to_global")
            assert isinstance(mapping, list), f"{name}: missing local_to_global"
            assert len(mapping) == sample.num_nodes, (
                f"{name}: local_to_global has {len(mapping)} entries but the "
                f"corridor has {sample.num_nodes} nodes"
            )
            assert len(set(mapping)) == len(mapping), f"{name}: duplicate global ids"
        null_stats = null_statistics(list(dataset))
        candidate_level = sum(
            float(np.mean(s.field.candidates.candidate_is_null)) for s in dataset
        ) / len(dataset)
        print(
            f"      {name}: n={len(dataset)} decisions(mean)="
            f"{sum(s.num_decisions for s in dataset) / len(dataset):.0f} "
            f"candidate_null={candidate_level:.4f} "
            f"target_null={null_stats['target_null_fraction']:.4f} (标签级)"
        )
        assert 0.0 <= null_stats["target_null_fraction"] <= 1.0

    # split 之间不允许有重复轨迹
    keys = {
        name: {(s.start, s.goal, tuple(s.gt_path)) for s in dataset}
        for name, dataset in splits.items()
    }
    assert not (keys["train"] & keys["val"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["val"] & keys["test"])

    # test_1000 / shuffled_od
    test_1000 = GraphQueryDataset.load(data_dir / "test_1000.pkl")
    assert len(test_1000) > 0
    assert all(s.meta.get("gt_source") == "observed" for s in test_1000)
    shuffled = GraphQueryDataset.load(data_dir / "shuffled_od_1000.pkl")
    assert all(s.meta.get("no_real_gt") for s in shuffled)
    assert all(s.meta.get("gt_source") == "dijkstra_placeholder" for s in shuffled)
    print(f"      test_1000: {len(test_1000)}  shuffled_od: {len(shuffled)}")

    # shuffled OD 必须是**新 OD**：不能是 test_1000 原题的副本
    # （实现时踩过的坑：置换作用在下标而不是 goals 上，配出来的还是原 OD）
    test_od = {(s.meta["start_node"], s.meta["goal_node"]) for s in test_1000}
    shuffled_od = {(s.meta["start_node"], s.meta["goal_node"]) for s in shuffled}
    overlap = test_od & shuffled_od
    assert not overlap, (
        f"{len(overlap)} shuffled OD pairs are actually original test ODs, e.g. "
        f"{sorted(overlap)[:3]}"
    )
    assert len(shuffled_od) == len(shuffled), "shuffled OD contains duplicate pairs"
    print(f"      shuffled OD is disjoint from test OD ({len(test_od)} pairs)")

    # manifest 可复现 & 统计齐全
    manifest = data_dir / "split_manifest.csv"
    assert manifest.exists()
    header = manifest.read_text(encoding="utf-8").splitlines()[0]
    for column in (
        "sample_id", "order_id", "date", "split", "raw_road_len", "junction_len",
        "num_corridor_nodes", "num_decisions", "gt_cost", "dijkstra_cost",
        "gt_cost_ratio",
    ):
        assert column in header, column
    print(f"      manifest columns ok: {header[:80]}...")

    # GT cost ratio 真的不是 1（否则说明 GT 被 Dijkstra 换掉了）
    ratios = [float(s.meta["gt_cost_ratio"]) for s in splits["test"]]
    above_one = sum(1 for value in ratios if value > 1.0 + 1e-6) / len(ratios)
    assert above_one > 0.3, f"only {above_one:.2%} of GT paths are non-optimal"
    print(
        f"      GT/Dijkstra cost ratio: mean={sum(ratios) / len(ratios):.4f} "
        f"max={max(ratios):.4f} non-optimal={above_one:.1%}"
    )


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="verify the DiDi real-data pipeline")
    parser.add_argument("--data", default="data/didi/graph/chengdu")
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()

    import tempfile

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, function in CHECKS:
            try:
                if function.__code__.co_argcount:
                    function(tmp_path)
                else:
                    function()
            except Exception as error:  # noqa: BLE001 - 自检脚本要打完所有用例
                failures += 1
                print(f"[FAIL] {name}: {type(error).__name__}: {error}")
                traceback.print_exc()
            else:
                print(f"[ ok ] {name}")

    if not args.skip_real:
        for name, function in (
            ("11. 真实 CSV 端到端（建图/转换/corridor/GT）", lambda: check_real_data(args.limit)),
            ("12. 已生成数据集自检", lambda: check_built_dataset(Path(args.data))),
        ):
            try:
                function()
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"[FAIL] {name}: {type(error).__name__}: {error}")
                traceback.print_exc()
            else:
                print(f"[ ok ] {name}")

    print()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print(f"all {len(CHECKS)} unit checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
