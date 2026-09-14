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
