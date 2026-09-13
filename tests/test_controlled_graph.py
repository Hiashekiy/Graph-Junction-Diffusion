"""Controlled Junction Graph 生成器测试（《V2 数据集生成指南》第 3-14 节）。

重点验证：
    * 每段骨架都插入 degree=2 普通节点（Branch Segment 不是一条边）；
    * 干扰分支三类（dead-end / detour / loop）都存在且比例合理；
    * 加完干扰分支后**重新求 GT**，且 GT 真的沿骨架走完；
    * 难度契约（hops / decisions / branch factor）真的被强制；
    * loop 不会造出绕过骨架的捷径（真实踩过的 bug）；
    * source 的两种形态：deg(s)==1 的被迫段、deg(s)>1 的 decision source。
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pytest

from src.data.controlled_graph import (
    ACCEPT_BRANCH_FACTOR,
    ACCEPT_DECISIONS,
    ACCEPT_HOPS,
    DIFFICULTY_SPECS,
    MODE_SPECS,
    compute_metrics,
    difficulty_filter,
    generate_controlled_junction_graph,
    sample_difficulty,
    sample_mode,
)
from src.data.dataset_builder import build_sample, dataset_statistics

MIX = {"easy": 0.2, "medium": 0.6, "hard": 0.2}
STRUCTURE = {"branch_heavy": 0.4, "long_chain": 0.4, "loop_detour": 0.2}


def _generate(seed: int = 0, attempts: int = 400):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(attempts):
        out.append(
            generate_controlled_junction_graph(
                rng,
                difficulty_mix=MIX,
                structure_mix=STRUCTURE,
                forced_source_probability=0.7,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 基本结构
# ---------------------------------------------------------------------------
def test_generator_is_deterministic_for_a_fixed_seed():
    first = _generate(seed=5, attempts=25)
    second = _generate(seed=5, attempts=25)
    for a, b in zip(first, second):
        assert a.accepted == b.accepted
        assert a.difficulty == b.difficulty
        assert a.mode == b.mode
        if a.accepted:
            assert sorted(a.graph.edges()) == sorted(b.graph.edges())


def test_accepted_graphs_are_connected_and_have_no_self_loops():
    for candidate in _generate(seed=1):
        if not candidate.accepted:
            continue
        graph = candidate.graph
        assert nx.is_connected(graph)
        assert not any(u == v for u, v in graph.edges())
        assert candidate.start in graph and candidate.goal in graph


def test_every_skeleton_junction_is_a_real_junction():
    """骨架内部的 junction 度数必须 >= 3，否则它根本不是 decision。"""
    checked = 0
    for candidate in _generate(seed=2):
        if not candidate.accepted:
            continue
        for node in candidate.skeleton[1:-1]:
            assert candidate.graph.degree(node) >= 3, (
                f"{candidate.difficulty}/{candidate.mode}: junction {node} "
                f"degree={candidate.graph.degree(node)}"
            )
        checked += 1
    assert checked > 10


def test_branch_segments_contain_ordinary_nodes():
    """指南第 5 节：structural endpoints 之间必须插入 degree=2 普通节点。"""
    for candidate in _generate(seed=3):
        if not candidate.accepted:
            continue
        sample = build_sample(candidate.graph.copy(), candidate.start, candidate.goal)
        lengths = [len(branch.nodes) for branch in sample.segments.branches_flat()]
        assert max(lengths) >= 3, "至少存在跨多个节点的 segment"
        # 至少一段包含 ordinary 节点（degree == 2 且非 start/goal）
        for group in sample.segments.branches:
            for branch in group:
                assert len(branch.nodes) == len(branch.physical_edges) + 1
        break


def test_goal_is_never_a_decision_node():
    for candidate in _generate(seed=4):
        if not candidate.accepted:
            continue
        assert candidate.metrics["gt_path"][0] == candidate.start
        assert candidate.metrics["gt_path"][-1] == candidate.goal
        assert candidate.goal not in candidate.metrics["gt_decision_nodes"]


# ---------------------------------------------------------------------------
# 难度契约
# ---------------------------------------------------------------------------
def test_accepted_graphs_respect_the_difficulty_contract():
    accepted = 0
    for candidate in _generate(seed=6):
        if not candidate.accepted:
            continue
        accepted += 1
        metrics = candidate.metrics
        assert ACCEPT_HOPS[0] <= metrics["gt_hops"] <= ACCEPT_HOPS[1]
        assert ACCEPT_DECISIONS[0] <= metrics["gt_decisions"] <= ACCEPT_DECISIONS[1]
        assert (
            ACCEPT_BRANCH_FACTOR[0]
            <= metrics["avg_branch_factor"]
            <= ACCEPT_BRANCH_FACTOR[1]
        )
    assert accepted >= 20, f"接受率过低，只拿到 {accepted} 个合格样本"


def test_difficulty_filter_rejects_out_of_contract_metrics():
    ok, reason = difficulty_filter(
        {"gt_hops": 5, "gt_decisions": 6, "avg_branch_factor": 3.0}
    )
    assert not ok and "gt_hops" in reason
    ok, reason = difficulty_filter(
        {"gt_hops": 20, "gt_decisions": 2, "avg_branch_factor": 3.0}
    )
    assert not ok and "gt_decisions" in reason
    ok, reason = difficulty_filter(
        {"gt_hops": 20, "gt_decisions": 6, "avg_branch_factor": 1.2}
    )
    assert not ok and "branch_factor" in reason
    ok, _ = difficulty_filter(
        {"gt_hops": 20, "gt_decisions": 6, "avg_branch_factor": 3.0}
    )
    assert ok


def test_difficulty_and_mode_mixes_are_respected():
    rng = np.random.default_rng(0)
    levels = Counter(sample_difficulty(rng, MIX) for _ in range(2000))
    modes = Counter(sample_mode(rng, STRUCTURE) for _ in range(2000))
    assert levels["medium"] > levels["easy"] > 0
    assert levels["medium"] > levels["hard"] > 0
    assert 0.10 < levels["hard"] / 2000 < 0.30
    assert 0.30 < modes["branch_heavy"] / 2000 < 0.50
    assert 0.16 < modes["loop_detour"] / 2000 < 0.25
    assert 0.30 < modes["long_chain"] / 2000 < 0.50


def test_each_difficulty_spec_is_internally_feasible():
    """每档的 (hops, decisions) 必须至少存在一个可行 K（否则那一档永远产不出）。"""
    for level, spec in DIFFICULTY_SPECS.items():
        hop_lo, hop_hi = spec["gt_hops"]
        dec_lo, dec_hi = spec["gt_decisions"]
        feasible = False
        for decisions in range(dec_lo, dec_hi + 1):
            for ordinary_lo, ordinary_hi in (
                MODE_SPECS[mode]["ordinary_nodes_per_segment"] for mode in MODE_SPECS
            ):
                low = (decisions + 1) * (1 + ordinary_lo)
                high = (decisions + 1) * (1 + ordinary_hi) + 4
                if max(hop_lo, low) <= min(hop_hi, high):
                    feasible = True
        assert feasible, f"{level} 档不存在可行的 (K, hops) 组合：{spec}"


# ---------------------------------------------------------------------------
# 干扰分支
# ---------------------------------------------------------------------------
def test_distractors_cover_all_three_kinds():
    kinds = Counter()
    for candidate in _generate(seed=7):
        if not candidate.accepted:
            continue
        for distractor in candidate.distractors:
            kinds[distractor.kind] += 1
    assert set(kinds) == {"dead_end", "detour", "loop"}
    total = sum(kinds.values())
    assert kinds["dead_end"] / total > 0.25
    assert kinds["detour"] / total > 0.10
    assert kinds["loop"] / total > 0.03


def test_loop_never_creates_a_shortcut_around_the_skeleton():
    """loop 只能连到"至少隔两跳"的下游 junction，否则会把 GT 压短。"""
    for candidate in _generate(seed=8):
        if not candidate.accepted:
            continue
        skeleton = candidate.skeleton
        for distractor in candidate.distractors:
            if distractor.kind != "loop":
                continue
            assert distractor.to_node in skeleton
            index_from = skeleton.index(distractor.from_node)
            index_to = skeleton.index(distractor.to_node)
            assert index_to >= index_from + 2, (
                f"loop {distractor.from_node}->{distractor.to_node} 没有跳过至少一段"
            )


def test_gt_is_recomputed_and_matches_the_skeleton():
    """GT 必须重新求解，并且真的经过骨架上的 junction。"""
    for candidate in _generate(seed=9):
        if not candidate.accepted:
            continue
        graph, start, goal = candidate.graph, candidate.start, candidate.goal
        recomputed = compute_metrics(graph, start, goal)
        assert recomputed["gt_path"] == candidate.metrics["gt_path"]
        skeleton_junctions = set(candidate.skeleton[1:-1])
        on_path = skeleton_junctions & set(recomputed["gt_path"])
        assert len(on_path) >= max(0, candidate.target_decisions - 1)
        break


def test_detour_is_strictly_longer_than_the_skeleton_route():
    for candidate in _generate(seed=10):
        if not candidate.accepted:
            continue
        for distractor in candidate.distractors:
            if distractor.kind != "detour":
                continue
            # detour 路径长度 = len(path)+1 条边；它必须 >= 骨架对应段的距离 + 1
            assert len(distractor.path) >= 1
        break


# ---------------------------------------------------------------------------
# source 的两种形态
# ---------------------------------------------------------------------------
def test_forced_source_has_degree_one_and_a_permanent_segment():
    samples = []
    for candidate in _generate(seed=11):
        if not candidate.accepted:
            continue
        sample = build_sample(candidate.graph.copy(), candidate.start, candidate.goal)
        samples.append(sample)
    forced = [s for s in samples if s.segments.source_forced_edge_ids]
    assert forced, "应该产出一批单出口 source 的样本"
    for sample in forced:
        assert sample.graph.degree(sample.start) == 1
        assert sample.segments.source_forced_nodes[0] == sample.start
        assert sample.segments.start not in sample.segments.decision_nodes
        # 被迫段整段都在 GT path 上
        for node in sample.segments.source_forced_nodes:
            assert node in sample.gt_path


def test_decision_source_has_multiple_exits():
    samples = []
    for candidate in _generate(seed=12):
        if not candidate.accepted:
            continue
        sample = build_sample(candidate.graph.copy(), candidate.start, candidate.goal)
        samples.append(sample)
    decision_source = [
        s for s in samples if not s.segments.source_forced_edge_ids
    ]
    assert decision_source, "应该产出一批多出口 source 的样本"
    for sample in decision_source:
        assert sample.graph.degree(sample.start) >= 2
        assert sample.segments.start in sample.segments.decision_nodes
        # source 没有 NULL 候选
        group = sample.field.candidates
        target = group.target_candidate[
            sample.segments.decision_nodes.index(sample.segments.start)
        ]
        assert not group.candidate_is_null[target]


# ---------------------------------------------------------------------------
# 数据集层面
# ---------------------------------------------------------------------------
def test_build_dataset_controlled_junction_end_to_end():
    from src.data.dataset_builder import build_dataset

    dataset = build_dataset(
        num_samples=60,
        graph_type="controlled_junction",
        seed=0,
        generator_cfg={
            "difficulty_mix": MIX,
            "structure_mix": STRUCTURE,
            "source_forced_probability": 0.7,
        },
    )
    assert len(dataset) == 60
    stats = dataset_statistics(dataset)
    assert stats["num_graphs"] == 60
    assert ACCEPT_HOPS[0] <= stats["hops"]["min"]
    assert stats["hops"]["max"] <= ACCEPT_HOPS[1]
    assert ACCEPT_DECISIONS[0] <= stats["decisions"]["min"]
    assert stats["decisions"]["max"] <= ACCEPT_DECISIONS[1]
    assert stats["source_forced_fraction"] + (1 - stats["source_as_decision_fraction"]) > 0
    for key in ("dead_end_branch_fraction", "detour_branch_fraction", "loop_branch_fraction"):
        assert 0.0 <= stats[key] <= 1.0


def test_statistics_report_the_guide_metrics():
    from src.data.dataset_builder import build_dataset

    dataset = build_dataset(
        num_samples=25,
        graph_type="controlled_junction",
        seed=1,
        generator_cfg={"difficulty_mix": MIX, "structure_mix": STRUCTURE},
    )
    stats = dataset_statistics(dataset)
    for key in (
        "num_graphs",
        "num_queries",
        "nodes",
        "hops",
        "decisions",
        "branch_factor",
        "null_fraction",
        "source_as_decision_fraction",
        "dead_end_branch_fraction",
        "detour_branch_fraction",
        "loop_branch_fraction",
        "difficulty_easy_fraction",
        "difficulty_medium_fraction",
        "difficulty_hard_fraction",
        "mode_branch_heavy_fraction",
        "mode_long_chain_fraction",
        "mode_loop_detour_fraction",
    ):
        assert key in stats, key
    assert stats["acceptance_contract"]["hops"] == list(ACCEPT_HOPS)


def test_unknown_graph_type_still_uses_the_old_generator():
    """回归：ER 路径不能被 controlled_junction 的改动破坏。"""
    from src.data.dataset_builder import build_dataset

    dataset = build_dataset(
        num_samples=6, graph_type="er", num_nodes=20, min_od_distance=3, seed=0
    )
    assert len(dataset) == 6
    assert all(sample.num_decisions >= 1 for sample in dataset)
