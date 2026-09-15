"""多分支（存活路径表）解码测试。

手工图（与 test_soft_goal 一致）：

    0=s 1=a 2=J1 3=b 4=J2 5=c 6=g
    J1 额外两条出口：->S(经 1)、->7(dead)
    J2 额外两条出口：->J1(经 3)、->8(dead)

candidate 组：J1 = [NULL, ->S, ->J2, ->dead7]，J2 = [NULL, ->J1, ->g, ->dead8]
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from src.data.collate import collate_samples
from src.data.dataset_builder import build_sample
from src.evaluation.evaluator import optimal_coverage
from src.evaluation.multi_path_decoder import decode_multi_path

S, A, J1, B, J2, C, G, D1, D2 = range(9)
EDGES = [
    (0, 1), (1, 2),
    (2, 3), (3, 4),
    (2, 7),
    (4, 5), (5, 6),
    (4, 8),
]


@pytest.fixture
def sample():
    graph = nx.Graph()
    graph.add_edges_from(EDGES)
    return build_sample(graph, S, G)


@pytest.fixture
def batch(sample):
    return collate_samples([sample], device="cpu")


def _cand(sample, owner, end):
    candidates = sample.field.candidates
    for index, (cand_owner, branch) in enumerate(
        zip(candidates.candidate_owner, candidates.candidate_branch)
    ):
        if cand_owner != owner:
            continue
        if end is None and branch is None:
            return index
        if branch is not None and branch.end == end:
            return index
    raise AssertionError(f"no candidate owned by {owner} ending at {end}")


def _prob(batch, table):
    prob = torch.zeros(batch.num_candidates)
    for index, value in table.items():
        prob[index] = float(value)
    for decision in range(batch.num_decisions):
        rows = batch.candidate_owner == decision
        total = float(prob[rows].sum())
        # 没写到的 decision 组允许全 0（测试只关心被指定的那个路口）
        assert total == 0.0 or abs(total - 1.0) < 1e-6, total
    return prob


def _decisions(sample):
    index = {node: i for i, node in enumerate(sample.segments.decision_nodes)}
    return index[J1], index[J2]


# ---------------------------------------------------------------------------
def test_top_k_one_matches_the_greedy_single_path(sample, batch):
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, J2): 0.7,
            _cand(sample, j1, D1): 0.3,
            _cand(sample, j2, G): 0.9,
            _cand(sample, j2, D2): 0.1,
        },
    )
    result = decode_multi_path(sample, prob, top_k=1, beam_width=8)
    assert len(result.finished) == 1
    best = result.best
    assert best.status == "goal"
    assert best.nodes == [S, A, J1, B, J2, C, G]
    assert result.coverage


def test_top_k_two_rescues_a_path_that_top_k_one_loses(sample, batch):
    """J1 的最高概率出口是 dead-end，只留一条会失败；留两条就能到终点。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, D1): 0.5,      # 概率最高，但通向 dead-end
            _cand(sample, j1, J2): 0.4,
            _cand(sample, j1, S): 0.1,
            _cand(sample, j2, G): 0.9,
            _cand(sample, j2, D2): 0.1,
        },
    )
    greedy = decode_multi_path(sample, prob, top_k=1, beam_width=8)
    assert not greedy.coverage
    assert greedy.best.status == "broken"

    wide = decode_multi_path(sample, prob, top_k=2, beam_width=8)
    assert wide.coverage
    assert wide.best.status == "broken"                  # 概率最高的那条仍然是死路
    assert wide.best_goal.nodes == [S, A, J1, B, J2, C, G]
    assert wide.best_goal_cost == 6


def test_null_policy_stop_kills_the_path_but_skip_continues(sample, batch):
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, None): 0.6,    # NULL 概率最高
            _cand(sample, j1, J2): 0.4,
            _cand(sample, j2, G): 1.0,
        },
    )
    stopped = decode_multi_path(sample, prob, top_k=1, null_policy="stop")
    assert not stopped.coverage
    assert stopped.best.status == "broken"
    assert "NULL selected" in stopped.best.reason

    skipped = decode_multi_path(sample, prob, top_k=1, null_policy="skip")
    assert skipped.coverage
    assert skipped.best.status == "goal"


def test_loop_paths_are_pruned_and_do_not_block_other_paths(sample, batch):
    """J1 的最高概率出口绕回 S（判 loop），第二条通向终点 —— 两条都要在表里。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, S): 0.6,       # 回到 start -> loop
            _cand(sample, j1, J2): 0.4,
            _cand(sample, j2, G): 1.0,
        },
    )
    result = decode_multi_path(sample, prob, top_k=2, beam_width=8)
    statuses = [path.status for path in result.finished]
    assert "loop" in statuses and "goal" in statuses
    assert result.coverage
    assert result.best.status == "loop"          # 概率更高的那条是 loop
    loop_path = result.best
    assert loop_path.nodes[0] == S and loop_path.nodes[-1] == S


def test_dead_end_branch_is_recorded_as_broken(sample, batch):
    j1, _ = _decisions(sample)
    prob = _prob(batch, {_cand(sample, j1, D1): 1.0})
    result = decode_multi_path(sample, prob, top_k=1)
    assert not result.coverage
    assert result.best.status == "broken"
    assert "no decision variable" in result.best.reason


def test_beam_width_truncates_the_frontier():
    """两个并行分支都存活时，beam_width=1 会丢掉概率较低的那条。"""
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1), (1, 2),          # s - a - J1
            (2, 3), (3, 4),          # J1 - b - J2
            (2, 7), (7, 8),          # J1 - d - J3
            (4, 5), (5, 6),          # J2 - c - g
            (4, 9),                  # J2 - dead
            (8, 6),                  # J3 - g
            (8, 10),                 # J3 - dead
        ]
    )
    wide_sample = build_sample(graph, 0, 6)
    wide_batch = collate_samples([wide_sample], device="cpu")
    j1 = wide_sample.segments.decision_nodes.index(2)
    j2 = wide_sample.segments.decision_nodes.index(4)
    j3 = wide_sample.segments.decision_nodes.index(8)
    prob = _prob(
        wide_batch,
        {
            _cand(wide_sample, j1, 8): 0.6,     # -> J3（较快到 goal）
            _cand(wide_sample, j1, 4): 0.4,     # -> J2
            _cand(wide_sample, j2, 6): 1.0,     # J2 -> goal
            _cand(wide_sample, j3, 6): 1.0,     # J3 -> goal
        },
    )
    wide = decode_multi_path(wide_sample, prob, top_k=2, beam_width=64)
    assert wide.pruned == 0
    assert wide.coverage

    narrow = decode_multi_path(wide_sample, prob, top_k=2, beam_width=1)
    assert narrow.pruned >= 1
    # 剪枝只丢低概率分支，最高概率那条不受影响
    assert narrow.best.log_prob == pytest.approx(wide.best.log_prob)
    assert narrow.best.goal_hit


def test_finished_paths_are_sorted_by_probability(sample, batch):
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, J2): 0.6,
            _cand(sample, j1, D1): 0.4,
            _cand(sample, j2, G): 0.8,
            _cand(sample, j2, D2): 0.2,
        },
    )
    result = decode_multi_path(sample, prob, top_k=2, beam_width=8)
    log_probs = [path.log_prob for path in result.finished]
    assert log_probs == sorted(log_probs, reverse=True)
    assert result.best.status == "goal"


def test_result_is_deterministic(sample, batch):
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, J2): 0.5,
            _cand(sample, j1, D1): 0.5,
            _cand(sample, j2, G): 0.5,
            _cand(sample, j2, D2): 0.5,
        },
    )
    first = decode_multi_path(sample, prob, top_k=2)
    second = decode_multi_path(sample, prob, top_k=2)
    assert [p.nodes for p in first.finished] == [p.nodes for p in second.finished]
    assert [p.log_prob for p in first.finished] == [p.log_prob for p in second.finished]


def test_invalid_arguments_are_rejected(sample, batch):
    j1, _ = _decisions(sample)
    prob = _prob(batch, {_cand(sample, j1, J2): 1.0})
    with pytest.raises(ValueError):
        decode_multi_path(sample, prob, top_k=0)
    with pytest.raises(ValueError):
        decode_multi_path(sample, prob, beam_width=0)
    with pytest.raises(ValueError):
        decode_multi_path(sample, prob, null_policy="nope")


def test_to_decode_result_keeps_the_metric_semantics(sample, batch):
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, J2): 0.8,
            _cand(sample, j1, D1): 0.2,
            _cand(sample, j2, G): 0.9,
            _cand(sample, j2, D2): 0.1,
        },
    )
    result = decode_multi_path(sample, prob, top_k=1)
    decoded = result.best.to_decode_result()
    assert decoded.status == "goal"
    assert decoded.path == [S, A, J1, B, J2, C, G]
    assert decoded.num_branches == 2


# ---------------------------------------------------------------------------
# 《Multi-Path Decoder 增强修改指南》第 11 节的 11 个测试
# ---------------------------------------------------------------------------
def _rank_prob(batch, table):
    """把任意正分数表归一化成每个 decision 组内和为 1 的概率（只关心排序）。

    指南的示例写的是 p=0.8 / 0.6 / 0.5（本身不归一），这里按组归一，排序不变。
    """
    prob = torch.zeros(batch.num_candidates)
    for index, value in table.items():
        prob[index] = float(value)
    for decision in range(batch.num_decisions):
        rows = batch.candidate_owner == decision
        total = float(prob[rows].sum())
        if total > 0:
            prob[rows] = prob[rows] / total
    return prob


def _cand_via(sample, owner, first_hop):
    """按 branch 的**第二跳**找候选（两条 branch 可能 end 相同，光看 end 分不开）。"""
    candidates = sample.field.candidates
    for index, (cand_owner, branch) in enumerate(
        zip(candidates.candidate_owner, candidates.candidate_branch)
    ):
        if cand_owner != owner or branch is None:
            continue
        if len(branch.nodes) > 1 and int(branch.nodes[1]) == int(first_hop):
            return index
    raise AssertionError(f"no candidate owned by {owner} leaving via {first_hop}")


def _dead_vs_decision_vs_goal(sample, batch):
    """J2 上的三选一：dead p=0.8 / decision p=0.6 / goal p=0.5（指南 Test 1）。

    先把路径送到 J2（J1 只走 ->J2），再在 J2 比较。
    """
    j1, j2 = _decisions(sample)
    return _rank_prob(
        batch,
        {
            _cand(sample, j1, J2): 1.0,
            _cand(sample, j2, D2): 0.8,     # 普通终止节点：必死
            _cand(sample, j2, J1): 0.6,     # 下一个 decision（会判 loop）
            _cand(sample, j2, G): 0.5,      # goal
        },
    )


def test_dead_branch_does_not_waste_a_top_k_slot(sample, batch):
    """Test 1：dead 分数最高也不该占掉 top-k；开筛选后必须是 decision + goal。"""
    prob = _dead_vs_decision_vs_goal(sample, batch)

    # 用 skip 让 NULL 不参与排名，这样断言只反映"必死 branch 是否占名额"这一件事
    without = decode_multi_path(
        sample, prob, top_k=2, beam_width=8, null_policy="skip"
    )
    assert {path.status for path in without.finished} == {"broken", "loop"}
    assert not without.coverage
    assert without.num_filtered_dead_branches == 0

    with_filter = decode_multi_path(
        sample, prob, top_k=2, beam_width=8, null_policy="skip",
        filter_dead_branches=True,
    )
    # J1 的 ->S / ->dead7 与 J2 的 ->dead8 都被剔除（计数是整个搜索累加的）
    assert with_filter.num_filtered_dead_branches == 3
    assert with_filter.coverage
    assert with_filter.best_goal is not None
    assert {path.status for path in with_filter.finished} == {"loop", "goal"}


def test_filter_off_reproduces_the_historical_decoder(sample, batch):
    """Test 2：默认（不传参）与显式 False 必须逐位一致，且与旧行为同形。"""
    prob = _dead_vs_decision_vs_goal(sample, batch)
    default = decode_multi_path(sample, prob, top_k=2, beam_width=8, null_policy="skip")
    explicit_off = decode_multi_path(
        sample, prob, top_k=2, beam_width=8, null_policy="skip",
        filter_dead_branches=False,
    )
    assert [(p.nodes, p.log_prob) for p in default.finished] == [
        (p.nodes, p.log_prob) for p in explicit_off.finished
    ]
    assert default.num_filtered_dead_branches == 0
    # 旧行为：分数最高的必死 branch 仍然被选中，并占掉一个 top-k 名额
    assert default.best.status == "broken"
    assert default.best.nodes[-1] == D2


def test_all_candidates_filtered_out_is_broken_not_a_crash(sample, batch):
    """Test 3：全部候选都是必死 branch 时判 broken，不能崩。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {_cand(sample, j1, J2): 1.0, _cand(sample, j2, D2): 1.0},
    )
    # 连通图里这个分支是**防御性**的：一个 decision 至少有一条 branch 通向别的
    # decision 或 Goal（否则 s->g 的 GT 根本不存在）。所以这里显式把 J2 的候选终点
    # 全部改写成叶子节点，专门验证守卫分支不崩。
    for index, owner in enumerate(sample.field.candidates.candidate_owner):
        branch = sample.field.candidates.candidate_branch[index]
        if int(owner) == j2 and branch is not None:
            branch.end = D2

    skipped = decode_multi_path(
        sample, prob, top_k=2, null_policy="skip", filter_dead_branches=True
    )
    assert not skipped.coverage
    assert skipped.best.status == "broken"
    assert "no viable candidate" in skipped.best.reason
    # J1 的 ->S / ->dead7 + J2 被改写成叶子的 3 条候选 = 5
    assert skipped.num_filtered_dead_branches == 5

    # stop 策略下 NULL 仍然参与排名（NULL 不参与预筛选），同样不许崩
    stopped = decode_multi_path(
        sample, prob, top_k=2, null_policy="stop", filter_dead_branches=True
    )
    assert stopped.best.status == "broken"


def test_loop_handling_is_unchanged_by_the_filter(sample, batch):
    """Test 4：loop 不在预筛选里处理；开/关筛选走同一条 loop 逻辑。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, J2): 0.5,
            _cand(sample, j1, S): 0.5,
            _cand(sample, j2, J1): 0.6,
            _cand(sample, j2, G): 0.4,
        },
    )
    off = decode_multi_path(sample, prob, top_k=2, beam_width=8)
    on = decode_multi_path(
        sample, prob, top_k=2, beam_width=8, filter_dead_branches=True
    )

    def loop_into_j1(result):
        # J2 -> J1（end 是 decision）这条一定会被判 loop，且不受预筛选影响
        return [p.nodes for p in result.finished if p.status == "loop" and p.nodes[-1] == J1]

    assert loop_into_j1(off) and loop_into_j1(off) == loop_into_j1(on)
    assert on.coverage


def _wide_case():
    """J1 有两条都通向 goal 的分支（一条经 J2、一条经 J3），用来测 beam 与排序。"""
    graph = nx.Graph()
    graph.add_edges_from(
        [
            (0, 1), (1, 2),          # s - a - J1
            (2, 3), (3, 4),          # J1 - b - J2
            (2, 7), (7, 8),          # J1 - d - J3
            (4, 5), (5, 6),          # J2 - c - g
            (4, 9),                  # J2 - dead
            (8, 6),                  # J3 - g
            (8, 10),                 # J3 - dead
        ]
    )
    sample = build_sample(graph, 0, 6)
    batch = collate_samples([sample], device="cpu")
    j1 = sample.segments.decision_nodes.index(2)
    j2 = sample.segments.decision_nodes.index(4)
    j3 = sample.segments.decision_nodes.index(8)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, 8): 0.6,
            _cand(sample, j1, 4): 0.4,
            _cand(sample, j2, 6): 1.0,
            _cand(sample, j3, 6): 1.0,
        },
    )
    return sample, batch, prob


def test_beam_pruning_is_unchanged_by_the_filter():
    """Test 5：合法候选集合相同时，beam 仍严格按累计 log 概率剪枝。"""
    sample, _, prob = _wide_case()
    off = decode_multi_path(sample, prob, top_k=2, beam_width=1)
    on = decode_multi_path(
        sample, prob, top_k=2, beam_width=1, filter_dead_branches=True
    )
    assert off.pruned == on.pruned >= 1
    assert off.best.log_prob == pytest.approx(on.best.log_prob)
    assert [p.nodes for p in off.finished] == [p.nodes for p in on.finished]


def test_weighted_path_cost_accumulates_forced_walk_and_branches():
    """Test 6：真实 cost = sum of edge weights（含被迫段）。"""
    # s(0) -3- a(1) -5- J1(2) -7- b(3) -11- g(4)，J1 另有一条 -2- dead(5)
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)])
    for (u, v), weight in {
        (0, 1): 3.0, (1, 2): 5.0, (2, 3): 7.0, (3, 4): 11.0, (2, 5): 2.0,
    }.items():
        graph.edges[u, v]["weight"] = weight
    graph.graph["weighted"] = True
    sample = build_sample(graph, 0, 4)
    batch = collate_samples([sample], device="cpu")
    j1 = sample.segments.decision_nodes.index(2)
    prob = _prob(batch, {_cand(sample, j1, 4): 1.0})

    result = decode_multi_path(sample, prob, top_k=1)
    goal = result.best_goal
    assert goal.path_cost == pytest.approx(3.0 + 5.0 + 7.0 + 11.0)
    assert goal.cost == 4                     # 历史字段：跳数
    assert result.best_goal_path_cost == pytest.approx(26.0)
    assert result.best_goal_cost == 4         # 历史字段：最少跳数


def test_weighted_path_cost_on_a_pure_forced_walk():
    """Test 6b：没有 decision、整条被迫走的路径也要累计真实 cost。"""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    graph.edges[0, 1]["weight"] = 3.0
    graph.edges[1, 2]["weight"] = 7.0
    graph.graph["weighted"] = True
    sample = build_sample(graph, 0, 2)
    result = decode_multi_path(sample, torch.zeros(sample.num_candidates), top_k=1)
    assert result.best.status == "goal"
    assert result.best.path_cost == pytest.approx(10.0)
    assert result.best.cost == 2


def test_unweighted_path_cost_equals_hop_count(sample, batch):
    """Test 7：没有 weight 属性时 path_cost 自然退化成跳数（旧数据兼容）。"""
    j1, j2 = _decisions(sample)
    prob = _prob(batch, {_cand(sample, j1, J2): 1.0, _cand(sample, j2, G): 1.0})
    result = decode_multi_path(sample, prob, top_k=2)
    best = result.best_goal
    assert best.path_cost == pytest.approx(float(best.cost))
    assert best.path_cost == pytest.approx(float(sample.gt_length))
    assert best.to_dict()["hops"] == best.cost


def test_best_keeps_the_historical_semantics(sample, batch):
    """Test 8：最高概率路径即使是 broken，multi.best 仍然是它。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, D1): 0.9,     # 最高概率 -> dead-end
            _cand(sample, j1, J2): 0.1,
            _cand(sample, j2, G): 1.0,
        },
    )
    result = decode_multi_path(sample, prob, top_k=2)
    assert result.best.status == "broken"
    assert result.best.log_prob > result.best_goal.log_prob
    assert result.best_goal.status == "goal"


def _two_route_weighted_case():
    """s-1-J1；J1 直连 goal（2 跳 / cost 21）与 J1-J2-g（3 跳 / cost 6）。"""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 3)])
    for (u, v), weight in {
        (0, 1): 1.0, (1, 2): 2.0, (2, 3): 3.0, (1, 3): 20.0,
    }.items():
        graph.edges[u, v]["weight"] = weight
    graph.graph["weighted"] = True
    sample = build_sample(graph, 0, 3)
    batch = collate_samples([sample], device="cpu")
    j1 = sample.segments.decision_nodes.index(1)
    prob = _prob(
        batch,
        {
            _cand_via(sample, j1, 3): 0.6,     # 贵的直连，概率更高
            _cand_via(sample, j1, 2): 0.4,     # 便宜的绕路
        },
    )
    return sample, batch, prob


def test_best_goal_is_the_highest_probability_goal_path():
    """Test 9：best_goal 取 Goal 路径里累计 log 概率最高者（这里正是贵的那条）。"""
    sample, _, prob = _two_route_weighted_case()
    result = decode_multi_path(sample, prob, top_k=2)
    assert len(result.goal_paths) == 2
    best_goal = result.best_goal
    assert best_goal.path_cost == pytest.approx(21.0)
    assert best_goal.cost == 2
    assert result.goal_paths_by_prob()[0] is best_goal


def test_best_goal_cost_path_prefers_the_cheaper_route():
    """Test 10：2 跳 cost=21 vs 3 跳 cost=6 —— 必须返回后者。"""
    sample, _, prob = _two_route_weighted_case()
    result = decode_multi_path(sample, prob, top_k=2)
    cheapest = result.best_goal_cost_path
    assert cheapest is not None
    assert cheapest.path_cost == pytest.approx(6.0)
    assert cheapest.cost == 3
    assert result.best_goal_path_cost == pytest.approx(6.0)
    assert result.goal_paths_by_cost()[0] is cheapest
    assert [p.path_cost for p in result.goal_paths_by_cost()] == [6.0, 21.0]

    # 同 cost 时取概率更高者：把两条路的 cost 调成一样再验证一次
    sample_equal, _, prob_equal = _two_route_weighted_case()
    for (u, v), weight in {
        (0, 1): 1.0, (1, 2): 2.0, (2, 3): 3.0, (1, 3): 5.0,
    }.items():
        sample_equal.graph.edges[u, v]["weight"] = weight      # 1+5 == 1+2+3 == 6
    equal = decode_multi_path(sample_equal, prob_equal, top_k=2)
    costs = {round(p.path_cost, 6) for p in equal.goal_paths}
    assert len(costs) == 1
    assert equal.best_goal_cost_path is equal.best_goal


def test_weighted_optimal_coverage_needs_a_true_minimum_cost_path():
    """Test 11：只有表里真的存在 Dijkstra 最小 cost 路时才算 optimal coverage。"""
    sample, _, prob = _two_route_weighted_case()
    wide = decode_multi_path(sample, prob, top_k=2)
    assert wide.coverage and optimal_coverage(sample, wide)

    narrow = decode_multi_path(sample, prob, top_k=1)   # 只留概率最高的贵直连
    assert narrow.coverage                              # 能到终点
    assert not optimal_coverage(sample, narrow)         # 但不是最小 cost


def test_optimal_coverage_degrades_to_hops_on_unweighted_graphs(sample, batch):
    """无权图：path_cost == 跳数，所以与旧的 cost <= gt_length 口径等价。"""
    j1, j2 = _decisions(sample)
    prob = _prob(batch, {_cand(sample, j1, J2): 1.0, _cand(sample, j2, G): 1.0})
    result = decode_multi_path(sample, prob, top_k=1)
    assert optimal_coverage(sample, result)
    assert result.best_goal.path_cost == pytest.approx(float(sample.gt_length))

    # 一条只有 dead-end 的表：coverage 与 optimal coverage 都是 False
    dead = decode_multi_path(
        sample,
        _prob(batch, {_cand(sample, j1, D1): 1.0}),
        top_k=1,
    )
    assert not dead.coverage and not optimal_coverage(sample, dead)


def test_goal_path_export_helpers_are_json_friendly():
    """指南第 5.3 节：每条路径至少能导出 nodes / log_prob / path_cost / hops / status。"""
    sample, _, prob = _two_route_weighted_case()
    result = decode_multi_path(sample, prob, top_k=2)
    rows = result.goal_paths_by_cost_dicts(1)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "nodes", "log_prob", "path_cost", "hops", "num_branches", "status", "reason",
    }
    assert row["status"] == "goal" and row["path_cost"] == pytest.approx(6.0)
    assert result.summary()["num_filtered_dead_branches"] == 0


# ---------------------------------------------------------------------------
# evaluator 接线：一次搜索 -> 三条口径 + 集合语义指标
# ---------------------------------------------------------------------------
def _tiny_report(sample, decode="multi", **kwargs):
    """用随机初始化的极小模型跑一次 evaluate_dataset（只验接线与口径，不看数字）。"""
    from src.data.dataset import GraphQueryDataset
    from src.diffusion.categorical import CategoricalDiffusion
    from src.diffusion.schedule import NoiseSchedule
    from src.evaluation.evaluator import evaluate_dataset
    from src.models.denoiser import GraphFlowDenoiser
    from src.utils.seed import make_generator, set_seed

    set_seed(0)
    torch.manual_seed(0)
    model = GraphFlowDenoiser(
        d_model=8, num_node_types=4, num_edge_states=2, ffn_hidden=16,
        flow_steps=1, slot_embedding=False,
    )
    diffusion = CategoricalDiffusion(
        NoiseSchedule(T=6, schedule="linear", beta_start=0.05, beta_end=0.5)
    )
    return evaluate_dataset(
        model,
        diffusion,
        GraphQueryDataset([sample]),
        batch_size=1,
        device="cpu",
        generator=make_generator(0, device="cpu"),
        max_steps=6,
        progress=False,
        decode=decode,
        **kwargs,
    )


def test_evaluate_dataset_exposes_the_three_multi_modes(sample):
    report = _tiny_report(sample, top_k=2, beam_width=8)
    assert set(report.multi) == {
        "multi_best", "multi_best_goal", "multi_best_goal_cost", "info",
    }
    for label in ("multi_best", "multi_best_goal", "multi_best_goal_cost"):
        row = report.multi[label]
        for key in ("goal_hit_rate", "optimal_path_rate", "success_cost_ratio"):
            assert key in row
    # 历史口径：metrics 仍然是 multi.best 的那一套
    assert report.metrics["goal_hit_rate"] == report.multi["multi_best"]["goal_hit_rate"]
    info = report.multi["info"]
    assert info["filter_dead_branches"] is False
    assert info["null_policy"] == "stop"
    for key in (
        "coverage_rate", "optimal_coverage_rate", "mean_goal_paths",
        "mean_finished_paths", "mean_filtered_dead_branches",
    ):
        assert key in report.metrics
    # 无权数据不暴露 weighted 别名（旧 eval json 的结构保持不变）
    assert "weighted_optimal_coverage_rate" not in report.metrics


def test_evaluate_dataset_accepts_the_dead_branch_filter(sample):
    report = _tiny_report(sample, top_k=2, beam_width=8, filter_dead_branches=True)
    assert report.multi["info"]["filter_dead_branches"] is True
    assert "mean_filtered_dead_branches" in report.metrics


def test_evaluate_dataset_reports_weighted_optimal_coverage_on_weighted_data():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 3)])
    for (u, v), weight in {
        (0, 1): 1.0, (1, 2): 2.0, (2, 3): 3.0, (1, 3): 20.0,
    }.items():
        graph.edges[u, v]["weight"] = weight
    graph.graph["weighted"] = True
    sample = build_sample(graph, 0, 3)
    report = _tiny_report(sample, top_k=2, beam_width=8)
    assert report.multi["info"]["dataset_is_weighted"] is True
    assert report.metrics["weighted_optimal_coverage_rate"] == pytest.approx(
        report.metrics["optimal_coverage_rate"]
    )


# ---------------------------------------------------------------------------
# strict 三池语义（src/evaluation/strict_beam_decoder.py）
#
# 这一组测试钉死的是"失败路径淘汰、最终候选只有完整到 Goal 的路径"这条规格。
# 历史路径（strict=False）的 27 个测试全部保持不变。
# ---------------------------------------------------------------------------
def test_strict_null_is_masked_instead_of_killing_the_path(sample, batch):
    """J1 的 top-1 是 NULL：历史口径当场 broken，strict 会 mask 掉改走第二名。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, None): 0.6,     # NULL 概率最高 -> 历史口径在这里死
            _cand(sample, j1, J2): 0.4,
            _cand(sample, j2, G): 0.9,
            _cand(sample, j2, D2): 0.1,
        },
    )
    historical = decode_multi_path(sample, prob, top_k=2, beam_width=3, null_policy="stop")
    assert historical.best.status == "broken"
    assert historical.best.reason.startswith("NULL selected")
    # 完整路线其实就在同一张表里 —— 历史口径只是把它排到了残骸后面
    assert historical.best_goal is not None
    assert historical.best_goal.nodes == [S, A, J1, B, J2, C, G]

    strict = decode_multi_path(sample, prob, top_k=2, beam_width=3, strict=True)
    assert strict.strict is True
    assert strict.best.status == "goal"
    assert strict.best.nodes == [S, A, J1, B, J2, C, G]
    assert strict.coverage
    assert strict.num_masked_null == 2          # J1 一个、J2 一个
    assert strict.discarded == []               # mask 掉候选不等于路径死亡


def test_strict_loop_branch_is_masked_and_reselected(sample, batch):
    """J1 的 top-1 是回到 S 的 loop branch：mask 掉它，重新在剩下的里选。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, S): 0.7,        # -> 回到已访问节点，非法
            _cand(sample, j1, J2): 0.3,
            _cand(sample, j2, G): 0.9,
            _cand(sample, j2, D2): 0.1,
        },
    )
    historical = decode_multi_path(sample, prob, top_k=2, beam_width=3)
    assert historical.best.status == "loop"      # 历史口径把这条 loop 当候选留着

    strict = decode_multi_path(sample, prob, top_k=2, beam_width=3, strict=True)
    assert strict.best.status == "goal"
    assert strict.num_masked_loop >= 1
    assert all(path.status == "goal" for path in strict.finished)


def test_strict_dead_end_branch_is_masked_before_top_k(sample, batch):
    """J1 的 top-1 通向 dead-end：strict 在 top-k **之前**就把它 mask 掉。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, D1): 0.55,      # dead-end，但概率最高
            _cand(sample, j1, J2): 0.45,
            _cand(sample, j2, G): 0.9,
            _cand(sample, j2, D2): 0.1,
        },
    )
    strict = decode_multi_path(sample, prob, top_k=2, beam_width=3, strict=True)
    assert strict.num_masked_dead_end >= 2       # J1 的 D1、J2 的 D2
    assert strict.best.status == "goal"


def test_strict_finished_is_the_success_pool(sample, batch):
    """``finished`` 在 strict 下就是 success 池：只有完整路线，没有残骸。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, None): 0.5,
            _cand(sample, j1, J2): 0.5,
            _cand(sample, j2, None): 0.5,
            _cand(sample, j2, G): 0.5,
        },
    )
    strict = decode_multi_path(sample, prob, top_k=2, beam_width=2, strict=True)
    assert strict.finished is strict.success
    assert all(path.status == "goal" for path in strict.finished)
    assert strict.goal_paths == strict.finished
    assert strict.alive_left == 0

    historical = decode_multi_path(sample, prob, top_k=2, beam_width=2, null_policy="stop")
    # 历史口径的 finished 里混着 NULL 残骸
    assert any(path.status != "goal" for path in historical.finished)


def test_strict_picks_the_long_complete_route_over_the_short_corpse(sample, batch):
    """这条就是那个荒谬现象的回归测试。

    历史口径下，一条 3 个节点就撞 NULL 的残骸因为累计 log 概率更大（负数加得少），
    排到了真正 7 个节点走到 Goal 的完整路线前面，成为 ``best``。
    strict 下残骸根本不进候选集，``best`` 必然是完整路线。
    """
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            # J1 的第一名是 NULL（高概率）-> 残骸 log_prob = log 0.9，只有 3 个节点
            _cand(sample, j1, None): 0.9,
            # 第二名通向 J2，再走到 Goal：链更长，负数累加更多，log 概率更低
            _cand(sample, j1, J2): 0.1,
            _cand(sample, j2, G): 0.9,
            _cand(sample, j2, D2): 0.1,
        },
    )
    historical = decode_multi_path(sample, prob, top_k=2, beam_width=4, null_policy="stop")
    corpse = historical.best
    assert corpse.status == "broken" and len(corpse.nodes) == 3
    assert historical.best_goal is not None            # 完整路线就在同一个表里
    assert len(historical.best_goal.nodes) == 7
    assert corpse.log_prob > historical.best_goal.log_prob   # 残骸概率更高 -> 历史口径选它

    strict = decode_multi_path(sample, prob, top_k=2, beam_width=4, strict=True)
    assert strict.best.status == "goal"
    assert strict.best.nodes == [S, A, J1, B, J2, C, G]
    assert all(len(path.nodes) == 7 for path in strict.finished)


def test_strict_legal_candidates_returns_empty_when_everything_is_illegal(sample, batch):
    """mask 规则的单测：四条合法性判定全不通过时返回空表。

    这张手工图每个 decision 恰好只有一条合法 branch，所以没法靠概率造出"无路可走"，
    直接测 mask 函数本身。
    """
    from src.evaluation.multi_path_decoder import MultiDecodeResult
    from src.evaluation.strict_beam_decoder import _legal_candidates

    candidates = sample.field.candidates
    decision_of = {int(n): i for i, n in enumerate(sample.segments.decision_nodes)}
    group = [i for i, owner in enumerate(candidates.candidate_owner) if int(owner) == 0]
    assert len(group) == 4
    probs = [0.4, 0.3, 0.2, 0.1]
    result = MultiDecodeResult(strict=True)

    # seen = {0,1,2}（真实初始状态）：->J2 的 branch 合法
    legal = _legal_candidates(candidates, group, probs, {0, 1, 2}, G, decision_of, result)
    assert len(legal) == 1

    # seen = {0,1,2,3}：->J2 的 branch 经过已访问的 3，也被 mask -> 一条合法 branch 都没有
    result = MultiDecodeResult(strict=True)
    legal = _legal_candidates(candidates, group, probs, {0, 1, 2, 3}, G, decision_of, result)
    assert legal == []
    assert result.num_masked_null == 1        # cand0
    assert result.num_masked_loop == 2        # cand1 回到 0、cand2 经过 3
    assert result.num_masked_dead_end == 1    # cand3 -> 7
    assert result.num_masked == 4


def test_strict_step_limit_path_goes_to_discarded(sample, batch):
    """路径死亡时进 discarded（这里用 max_branches 触发），且不进最终候选集。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, J2): 0.6,
            _cand(sample, j1, D1): 0.4,
            _cand(sample, j2, G): 1.0,
        },
    )
    strict = decode_multi_path(
        sample, prob, top_k=2, beam_width=3, max_branches=1, strict=True
    )
    assert not strict.coverage
    assert strict.finished == [] and strict.success == []
    assert len(strict.discarded) == 1
    assert strict.discarded[0].reason == "step limit exceeded"
    assert strict.alive_left == 0


def test_strict_ignores_null_policy_and_dead_branch_filter(sample, batch):
    """strict 内置 NULL mask 与 dead-end mask，两个旧开关不再起作用。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, None): 0.5,
            _cand(sample, j1, J2): 0.5,
            _cand(sample, j2, G): 1.0,
        },
    )
    a = decode_multi_path(sample, prob, top_k=2, beam_width=3, null_policy="stop", strict=True)
    b = decode_multi_path(sample, prob, top_k=2, beam_width=3, null_policy="skip", strict=True)
    c = decode_multi_path(
        sample, prob, top_k=2, beam_width=3, null_policy="stop",
        filter_dead_branches=True, strict=True,
    )
    for other in (b, c):
        assert [p.nodes for p in other.finished] == [p.nodes for p in a.finished]
        assert other.num_masked_null == a.num_masked_null


def test_strict_beam_width_bounds_alive_paths_only(sample, batch):
    """beam_width 限制的是"同时活着的路径数"，不是最终候选数。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, None): 0.3,
            _cand(sample, j1, J2): 0.4,
            _cand(sample, j1, D1): 0.3,
            _cand(sample, j2, None): 0.3,
            _cand(sample, j2, G): 0.5,
            _cand(sample, j2, D2): 0.2,
        },
    )
    narrow = decode_multi_path(sample, prob, top_k=2, beam_width=1, strict=True)
    wide = decode_multi_path(sample, prob, top_k=2, beam_width=8, strict=True)
    assert narrow.coverage and wide.coverage
    assert narrow.alive_left == 0 and wide.alive_left == 0
    assert len(wide.success) >= len(narrow.success)


def test_strict_summary_exposes_the_three_pools(sample, batch):
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, None): 0.5,
            _cand(sample, j1, J2): 0.5,
            _cand(sample, j2, G): 1.0,
        },
    )
    payload = decode_multi_path(sample, prob, top_k=2, beam_width=3, strict=True).summary()
    assert payload["strict"] is True
    assert payload["num_success"] == 1
    assert payload["num_discarded"] == 0
    assert payload["num_masked_null"] == 2
    assert payload["num_masked"] == 6          # null 2 + loop 2 + dead-end 2
    assert payload["alive_left"] == 0
    assert payload["best_status"] == "goal"


def test_strict_rejects_bad_arguments(sample, batch):
    prob = torch.zeros(batch.num_candidates)
    with pytest.raises(ValueError):
        decode_multi_path(sample, prob, top_k=0, strict=True)
    with pytest.raises(ValueError):
        decode_multi_path(sample, prob, beam_width=0, strict=True)


def test_strict_matches_greedy_when_there_is_exactly_one_legal_route(sample, batch):
    """退化检查：整张图只有一条合法 s->g 路线时，strict 必须逐位走出它。"""
    j1, j2 = _decisions(sample)
    prob = _prob(
        batch,
        {
            _cand(sample, j1, None): 0.4,
            _cand(sample, j1, S): 0.3,       # loop
            _cand(sample, j1, D1): 0.2,      # dead-end
            _cand(sample, j1, J2): 0.1,      # 唯一的合法出口
            _cand(sample, j2, None): 0.4,
            _cand(sample, j2, D2): 0.35,
            _cand(sample, j2, G): 0.25,      # 唯一的合法出口
        },
    )
    strict = decode_multi_path(sample, prob, top_k=2, beam_width=1, strict=True)
    assert strict.coverage
    assert strict.best.nodes == [S, A, J1, B, J2, C, G]
    assert strict.num_masked_null == 2
    assert strict.num_masked_loop == 2       # J1 的 ->S、J2 的 ->J1（经已访问的 3）
    assert strict.num_masked_dead_end == 2   # J1 的 ->7、J2 的 ->8


def test_evaluate_dataset_supports_strict_decode(sample):
    """evaluator 层要能把 strict 一路传下去（并且只有当 strict 时才报 mask 指标）。"""
    report = _tiny_report(sample, top_k=2, beam_width=3, strict_decode=True)
    info = report.multi["info"]
    assert info["strict"] is True
    assert info["best_readout"] == "argmax_{P in success} log_prob"
    assert "mean_masked_null" in report.metrics
    assert "mean_discarded_paths" in report.metrics

    legacy = _tiny_report(sample, top_k=2, beam_width=3)
    assert legacy.multi["info"]["strict"] is False
    assert "mean_masked_null" not in legacy.metrics
