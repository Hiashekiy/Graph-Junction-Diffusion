"""Soft Goal Reachability 测试（第二轮修订 B 项）.

覆盖指南第十四节要求的验收点：

1. ``0.8 x 0.7`` 的图必须得到 ``P_goal = 0.56``；
2. NULL / dead-end 对 Goal value 的贡献为 0；
3. 两条不同路径都能到 Goal 时概率相加；
4. SoftGoalLoss.backward() 能给 Branch 概率（进而 logits）非零梯度；
5. degree-1 Source 从 forced segment 的终点开始算，deg(s) > 1 时从 source 自己算；
6. 多图 batch 之间 reachability 不能串图；
7. ``goal_reach_weight=0`` 时总 loss 与纯 CE 完全一致。

外加拓扑字段本身（candidate_next_decision / candidate_hits_goal）的正确性检查。
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from src.data.collate import collate_samples
from src.data.dataset_builder import build_sample
from src.diffusion.categorical import CategoricalDiffusion
from src.diffusion.schedule import NoiseSchedule
from src.models.denoiser import GraphFlowDenoiser
from src.training.losses import LossWeights, recurrent_reverse_loss
from src.training.soft_goal import soft_goal_loss, soft_goal_reachability

# ---------------------------------------------------------------------------
# 手工图 1：s - a - J1 - b - J2 - c - g，J1 / J2 各带一个 dead-end 出口
#
#   0=s 1=a 2=J1 3=b 4=J2 5=c 6=g 7=dead 8=dead
#
# candidate 顺序（NULL 永远在组内第一个）：
#   J1: [NULL, ->s, ->J2, ->dead]
#   J2: [NULL, ->J1, ->g,  ->dead]
# ---------------------------------------------------------------------------
CHAIN_EDGES = [
    (0, 1), (1, 2),          # s - a - J1
    (2, 3), (3, 4),          # J1 - b - J2
    (2, 7),                  # J1 - dead
    (4, 5), (5, 6),          # J2 - c - g
    (4, 8),                  # J2 - dead
]
CHAIN_S, CHAIN_A, CHAIN_J1, CHAIN_B, CHAIN_J2, CHAIN_C, CHAIN_G, CHAIN_D1, CHAIN_D2 = range(9)


def make_sample(edges, start, goal):
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return build_sample(graph, start, goal)


@pytest.fixture
def chain_sample():
    return make_sample(CHAIN_EDGES, CHAIN_S, CHAIN_G)


@pytest.fixture
def chain_batch(chain_sample):
    return collate_samples([chain_sample], device="cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _candidate_index(sample, owner, end):
    """(decision 局部编号, branch 终点) -> flat candidate index。"""
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
    raise AssertionError(f"no candidate owned by decision {owner} ending at {end}")


def _prob_from_table(batch, table):
    """table: {flat candidate index: 概率} -> [C]；每个 decision 组内必须和为 1。"""
    prob = torch.zeros(batch.num_candidates)
    for index, value in table.items():
        prob[index] = float(value)
    for decision_index in range(batch.num_decisions):
        rows = batch.candidate_owner == decision_index
        assert abs(float(prob[rows].sum()) - 1.0) < 1e-6, "候选组内概率必须和 = 1"
    return prob


def _decision_index(sample, node):
    return sample.segments.decision_nodes.index(node)


# ---------------------------------------------------------------------------
# 拓扑字段
# ---------------------------------------------------------------------------
def test_topology_fields_of_chain_graph(chain_sample, chain_batch):
    batch = chain_batch
    j1 = _decision_index(chain_sample, CHAIN_J1)
    j2 = _decision_index(chain_sample, CHAIN_J2)

    expected_next = {
        _candidate_index(chain_sample, j1, None): -1,
        _candidate_index(chain_sample, j1, CHAIN_S): -1,
        _candidate_index(chain_sample, j1, CHAIN_J2): j2,
        _candidate_index(chain_sample, j1, CHAIN_D1): -1,
        _candidate_index(chain_sample, j2, None): -1,
        _candidate_index(chain_sample, j2, CHAIN_J1): j1,
        _candidate_index(chain_sample, j2, CHAIN_G): -1,
        _candidate_index(chain_sample, j2, CHAIN_D2): -1,
    }
    for index, target in expected_next.items():
        assert int(batch.candidate_next_decision[index]) == int(target)

    # 只有 J2 -> g 这条 branch 直接命中 Goal
    assert batch.candidate_hits_goal.sum().item() == 1
    assert bool(batch.candidate_hits_goal[_candidate_index(chain_sample, j2, CHAIN_G)])

    # deg(s) == 1：从 forced segment 的终点（J1）开始算
    assert int(batch.reach_start_decision[0]) == int(j1)
    assert not bool(batch.reach_start_is_goal[0])


# ---------------------------------------------------------------------------
# 1. 0.8 x 0.7 = 0.56
# ---------------------------------------------------------------------------
def test_soft_goal_is_the_product_of_branch_probabilities(chain_sample, chain_batch):
    batch = chain_batch
    j1 = _decision_index(chain_sample, CHAIN_J1)
    j2 = _decision_index(chain_sample, CHAIN_J2)
    prob = _prob_from_table(
        batch,
        {
            _candidate_index(chain_sample, j1, None): 0.0,
            _candidate_index(chain_sample, j1, CHAIN_S): 0.0,
            _candidate_index(chain_sample, j1, CHAIN_J2): 0.8,
            _candidate_index(chain_sample, j1, CHAIN_D1): 0.2,
            _candidate_index(chain_sample, j2, None): 0.0,
            _candidate_index(chain_sample, j2, CHAIN_J1): 0.0,
            _candidate_index(chain_sample, j2, CHAIN_G): 0.7,
            _candidate_index(chain_sample, j2, CHAIN_D2): 0.3,
        },
    )
    p_goal = soft_goal_reachability(prob, batch)
    assert p_goal.shape == (1,)
    assert abs(float(p_goal[0]) - 0.56) < 1e-6, float(p_goal[0])


# ---------------------------------------------------------------------------
# 2. NULL / dead-end 的贡献为 0
# ---------------------------------------------------------------------------
def test_null_and_dead_end_contribute_nothing(chain_sample, chain_batch):
    batch = chain_batch
    j1 = _decision_index(chain_sample, CHAIN_J1)
    j2 = _decision_index(chain_sample, CHAIN_J2)
    # J2 把概率全部放在 dead-end 上（NULL 也是 0）：Goal 完全不可达
    prob = _prob_from_table(
        batch,
        {
            _candidate_index(chain_sample, j1, CHAIN_S): 0.0,
            _candidate_index(chain_sample, j1, CHAIN_J2): 1.0,
            _candidate_index(chain_sample, j2, CHAIN_J1): 0.0,
            _candidate_index(chain_sample, j2, CHAIN_G): 0.0,
            _candidate_index(chain_sample, j2, CHAIN_D2): 1.0,
        },
    )
    assert abs(float(soft_goal_reachability(prob, batch)[0])) < 1e-6

    # 反过来：概率放在 NULL 上（J1 直接 NULL）也必须是 0
    prob_null = _prob_from_table(
        batch,
        {
            _candidate_index(chain_sample, j1, CHAIN_J2): 0.0,
            _candidate_index(chain_sample, j1, None): 1.0,
            _candidate_index(chain_sample, j2, CHAIN_G): 1.0,
        },
    )
    assert abs(float(soft_goal_reachability(prob_null, batch)[0])) < 1e-6


# ---------------------------------------------------------------------------
# 3. 多条路径的概率相加
# ---------------------------------------------------------------------------
def test_two_paths_to_goal_add_up():
    """J1 有两条 branch 分别到 J2 / J3，二者都 100% 到 Goal。"""
    edges = [
        (0, 1), (1, 2),          # s - a - J1
        (2, 3), (3, 4),          # J1 - b - J2
        (2, 7), (7, 8),          # J1 - d - J3
        (4, 5), (5, 6),          # J2 - c - g
        (4, 9),                  # J2 - dead
        (8, 6),                  # J3 - g
        (8, 10),                 # J3 - dead
    ]
    sample = make_sample(edges, 0, 6)
    batch = collate_samples([sample], device="cpu")

    j1 = _decision_index(sample, 2)
    j2 = _decision_index(sample, 4)
    j3 = _decision_index(sample, 8)
    assert batch.num_decisions == 3

    prob = _prob_from_table(
        batch,
        {
            _candidate_index(sample, j1, 4): 0.3,
            _candidate_index(sample, j1, 8): 0.5,
            _candidate_index(sample, j1, 0): 0.2,
            _candidate_index(sample, j2, 6): 1.0,
            _candidate_index(sample, j3, 6): 1.0,
        },
    )
    p_goal = soft_goal_reachability(prob, batch)
    # 0.3 * 1 + 0.5 * 1 = 0.8
    assert abs(float(p_goal[0]) - 0.8) < 1e-6, float(p_goal[0])


# ---------------------------------------------------------------------------
# 4. 可微 + 梯度能回到 Branch logits
# ---------------------------------------------------------------------------
def test_soft_goal_is_differentiable_wrt_candidate_prob(chain_sample, chain_batch):
    batch = chain_batch
    j1 = _decision_index(chain_sample, CHAIN_J1)
    j2 = _decision_index(chain_sample, CHAIN_J2)
    table = {
        _candidate_index(chain_sample, j1, CHAIN_J2): 0.6,
        _candidate_index(chain_sample, j1, CHAIN_D1): 0.4,
        _candidate_index(chain_sample, j2, CHAIN_G): 0.5,
        _candidate_index(chain_sample, j2, CHAIN_D2): 0.5,
    }
    prob = _prob_from_table(batch, table).requires_grad_(True)
    p_goal = soft_goal_reachability(prob, batch)
    assert p_goal.requires_grad
    soft_goal_loss(p_goal).backward()
    assert prob.grad is not None and torch.isfinite(prob.grad).all()
    # 只有"能带向 Goal"的两个候选有非零梯度
    for index in (
        _candidate_index(chain_sample, j1, CHAIN_J2),
        _candidate_index(chain_sample, j2, CHAIN_G),
    ):
        assert prob.grad[index] != 0.0, f"candidate {index} 没有梯度"
    # NULL / dead-end / 与 Goal 无关的候选梯度必须为 0
    assert prob.grad[_candidate_index(chain_sample, j1, None)] == 0.0
    assert prob.grad[_candidate_index(chain_sample, j1, CHAIN_D1)] == 0.0
    assert prob.grad[_candidate_index(chain_sample, j2, CHAIN_D2)] == 0.0


def test_goal_term_backprops_into_branch_scorer(chain_sample):
    batch = collate_samples([chain_sample], device="cpu")
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = CategoricalDiffusion(
        NoiseSchedule(T=4, schedule="linear", beta_start=0.05, beta_end=0.5)
    )
    out = recurrent_reverse_loss(
        model,
        diffusion,
        batch,
        weights=LossWeights(goal_reach_weight=1.0),
        max_steps=4,
        record=True,
    )
    assert torch.isfinite(out.goal_loss)
    assert len(out.per_step_goal_loss) == 4
    assert len(out.per_step_soft_goal) == 4
    out.goal_loss.backward()
    assert model.branch_scorer.branch_mlp[0].weight.grad is not None
    assert model.graph_flow.k_node_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# 5. Source 语义
# ---------------------------------------------------------------------------
def test_degree_one_source_starts_at_the_forced_endpoint(chain_sample, chain_batch):
    j1 = _decision_index(chain_sample, CHAIN_J1)
    assert int(chain_batch.reach_start_decision[0]) == int(j1)
    assert chain_sample.segments.source_forced_nodes == [CHAIN_S, CHAIN_A, CHAIN_J1]


def test_degree_one_source_forced_segment_can_hit_goal_directly():
    """s - a - b - g：整张图没有 decision，Soft Goal 直接是 1。"""
    sample = make_sample([(0, 1), (1, 2), (2, 3)], 0, 3)
    batch = collate_samples([sample], device="cpu")
    assert batch.num_decisions == 0
    assert bool(batch.reach_start_is_goal[0])
    assert float(soft_goal_reachability(torch.zeros(0), batch)[0]) == 1.0


def test_multi_exit_source_starts_at_itself():
    """deg(s) > 1 时 source 自己是 decision，必须从它开始算。"""
    edges = [
        (0, 1), (1, 2),          # s - a - J
        (0, 3),                  # s - dead
        (2, 4), (4, 5),          # J - b - g
        (2, 6),                  # J - dead
    ]
    sample = make_sample(edges, 0, 5)
    batch = collate_samples([sample], device="cpu")

    source_decision = _decision_index(sample, 0)
    assert int(batch.reach_start_decision[0]) == int(source_decision)
    assert not bool(batch.reach_start_is_goal[0])
    assert sample.segments.source_forced_nodes == []

    # source: ->J = 0.6, ->dead = 0.4；J: ->g = 1.0
    prob = _prob_from_table(
        batch,
        {
            _candidate_index(sample, source_decision, 2): 0.6,
            _candidate_index(sample, source_decision, 3): 0.4,
            _candidate_index(sample, _decision_index(sample, 2), 5): 1.0,
        },
    )
    assert abs(float(soft_goal_reachability(prob, batch)[0]) - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# 6. 多图 batch 不串图
# ---------------------------------------------------------------------------
def test_batch_members_do_not_leak_into_each_other(chain_sample):
    second = make_sample(CHAIN_EDGES, CHAIN_S, CHAIN_G)
    samples = [chain_sample, second]
    batch = collate_samples(samples, device="cpu")
    counts = [s.num_candidates for s in samples]

    prob = torch.zeros(batch.num_candidates)

    def fill(sample, offset, j1_to_j2, j2_to_goal):
        j1 = _decision_index(sample, CHAIN_J1)
        j2 = _decision_index(sample, CHAIN_J2)
        prob[offset + _candidate_index(sample, j1, CHAIN_J2)] = j1_to_j2
        prob[offset + _candidate_index(sample, j1, CHAIN_D1)] = 1.0 - j1_to_j2
        prob[offset + _candidate_index(sample, j2, CHAIN_G)] = j2_to_goal
        prob[offset + _candidate_index(sample, j2, CHAIN_D2)] = 1.0 - j2_to_goal

    # 第一张图：0.3 * 0.4 = 0.12；第二张图：J1 全部走 dead-end -> 0
    fill(samples[0], 0, 0.3, 0.4)
    fill(samples[1], counts[0], 0.0, 1.0)

    p_batch = soft_goal_reachability(prob, batch)
    assert abs(float(p_batch[0]) - 0.3 * 0.4) < 1e-6
    assert abs(float(p_batch[1])) < 1e-6

    # 两张图单独算必须得到同样的数（否则就是把邻居图的信息算进来了）
    single0 = collate_samples([samples[0]], device="cpu")
    p0 = soft_goal_reachability(prob[: counts[0]], single0)
    assert abs(float(p0[0]) - float(p_batch[0])) < 1e-6
    single1 = collate_samples([samples[1]], device="cpu")
    p1 = soft_goal_reachability(prob[counts[0] :], single1)
    assert abs(float(p1[0]) - float(p_batch[1])) < 1e-6


def test_batch_of_two_different_graphs_uses_own_horizon():
    """决策数不同的两张图同 batch，短的那张不会被长的图多迭代出错误的值。"""
    long_sample = make_sample(CHAIN_EDGES, CHAIN_S, CHAIN_G)
    short_sample = make_sample([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)], 0, 4)
    batch = collate_samples([short_sample, long_sample], device="cpu")
    assert batch.num_decisions == short_sample.num_decisions + long_sample.num_decisions
    prob = torch.zeros(batch.num_candidates)
    # 短图：J1 -> g = 0.4, -> dead = 0.6
    offset = 0
    j_short = _decision_index(short_sample, 2)
    prob[offset + _candidate_index(short_sample, j_short, 4)] = 0.4
    prob[offset + _candidate_index(short_sample, j_short, 5)] = 0.6
    # 长图：J1 -> J2 = 1.0，J2 -> g = 0.5
    offset = short_sample.num_candidates
    j1 = _decision_index(long_sample, CHAIN_J1)
    j2 = _decision_index(long_sample, CHAIN_J2)
    prob[offset + _candidate_index(long_sample, j1, CHAIN_J2)] = 1.0
    prob[offset + _candidate_index(long_sample, j2, CHAIN_G)] = 0.5
    prob[offset + _candidate_index(long_sample, j2, CHAIN_D2)] = 0.5

    p = soft_goal_reachability(prob, batch)
    assert abs(float(p[0]) - 0.4) < 1e-6
    assert abs(float(p[1]) - 0.5) < 1e-6


def test_mixed_batch_with_a_decision_free_graph():
    """整张图没有 decision 的样本（source 被迫段直接到 Goal）与别的图同 batch。"""
    path_sample = make_sample([(0, 1), (1, 2), (2, 3)], 0, 3)
    chain = make_sample(CHAIN_EDGES, CHAIN_S, CHAIN_G)
    assert path_sample.num_decisions == 0
    assert path_sample.num_candidates == 0

    batch = collate_samples([path_sample, chain], device="cpu")
    assert batch.num_decisions == 2
    assert bool(batch.reach_start_is_goal[0])

    j1 = _decision_index(chain, CHAIN_J1)
    j2 = _decision_index(chain, CHAIN_J2)
    prob = _prob_from_table(
        batch,
        {
            _candidate_index(chain, j1, CHAIN_J2): 0.8,
            _candidate_index(chain, j1, CHAIN_D1): 0.2,
            _candidate_index(chain, j2, CHAIN_G): 0.7,
            _candidate_index(chain, j2, CHAIN_D2): 0.3,
        },
    )
    p = soft_goal_reachability(prob, batch)
    assert float(p[0]) == 1.0
    assert abs(float(p[1]) - 0.56) < 1e-6


# ---------------------------------------------------------------------------
# 7. goal_reach_weight = 0 严格退化成纯 CE
# ---------------------------------------------------------------------------
def test_zero_goal_weight_reproduces_the_pure_ce_loss(chain_batch):
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = CategoricalDiffusion(
        NoiseSchedule(T=4, schedule="linear", beta_start=0.05, beta_end=0.5)
    )
    out = recurrent_reverse_loss(
        model,
        diffusion,
        chain_batch,
        weights=LossWeights(goal_reach_weight=0.0),
        max_steps=4,
    )
    assert torch.equal(out.loss, out.ce_loss)
    # 逐 step CE 的均值就是 ce_loss（旧实现的口径）
    assert abs(float(out.loss.detach()) - float(out.ce_loss.detach())) < 1e-12


def test_zero_goal_weight_skips_the_value_iteration(chain_batch, monkeypatch):
    """``goal_reach_weight = 0`` 时必须**整段跳过** soft_goal_reachability。

    它是一段 Python for-loop 的 value iteration（每个 reverse step 迭代
    min(decision 数, horizon_cap) 轮）。真实 DiDi corridor 有 200~970 个 decision，
    实测占单步训练耗时 60%+；乘 0 与跳过在梯度上等价，所以只乘 0 是纯浪费。
    这条测试钉住"真的没算"，而不是只看 loss 数值相等。
    """
    import src.training.losses as losses_module

    calls = {"n": 0}
    original = losses_module.soft_goal_reachability

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(losses_module, "soft_goal_reachability", counting)
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = CategoricalDiffusion(
        NoiseSchedule(T=4, schedule="linear", beta_start=0.05, beta_end=0.5)
    )

    off = recurrent_reverse_loss(
        model, diffusion, chain_batch, LossWeights(goal_reach_weight=0.0), max_steps=4
    )
    assert calls["n"] == 0, "goal_reach_weight=0 时不该调用 value iteration"
    assert float(off.goal_loss.detach()) == 0.0
    # NaN 表示"没算"，与"算出来是 0"区分开
    assert off.soft_goal_mean != off.soft_goal_mean

    recurrent_reverse_loss(
        model, diffusion, chain_batch, LossWeights(goal_reach_weight=0.1), max_steps=4
    )
    assert calls["n"] == 4, "每个 reverse step 应该各调用一次"


def test_positive_goal_weight_changes_the_loss(chain_batch):
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = CategoricalDiffusion(
        NoiseSchedule(T=4, schedule="linear", beta_start=0.05, beta_end=0.5)
    )
    zero = recurrent_reverse_loss(
        model, diffusion, chain_batch, LossWeights(goal_reach_weight=0.0), max_steps=4
    )
    nonzero = recurrent_reverse_loss(
        model, diffusion, chain_batch, LossWeights(goal_reach_weight=0.1), max_steps=4
    )
    assert float(nonzero.goal_loss.detach()) > 0.0
    assert abs(float(nonzero.loss.detach()) - float(zero.loss.detach())) > 1e-9


def test_soft_goal_loss_is_monotone_in_p_goal():
    good = soft_goal_loss(torch.tensor([1.0, 0.9]))
    bad = soft_goal_loss(torch.tensor([0.5, 0.1]))
    assert float(good) < float(bad)


def test_invalid_goal_timestep_weighting_is_rejected(chain_batch):
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = CategoricalDiffusion(
        NoiseSchedule(T=4, schedule="linear", beta_start=0.05, beta_end=0.5)
    )
    with pytest.raises(ValueError):
        recurrent_reverse_loss(
            model,
            diffusion,
            chain_batch,
            LossWeights(goal_timestep_weighting="nope"),
            max_steps=4,
        )


# ---------------------------------------------------------------------------
# 8. horizon cap（混入"路口很多、路径很短"的数据时控制 cost）
# ---------------------------------------------------------------------------
def _long_chain_sample():
    """s-a-J1-b-J2-c-J3-d-g，每个 J 带一个 dead-end：值需要传 3 轮才到 J1。"""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (2, 9), (4, 10), (6, 11),
    ]
    return make_sample(edges, 0, 8)


def test_horizon_cap_does_not_change_small_graphs(chain_sample, chain_batch):
    """decision 数 <= cap 的图，加不加 cap 结果必须逐位相同。"""
    j1 = _decision_index(chain_sample, CHAIN_J1)
    j2 = _decision_index(chain_sample, CHAIN_J2)
    prob = _prob_from_table(
        chain_batch,
        {
            _candidate_index(chain_sample, j1, CHAIN_J2): 0.8,
            _candidate_index(chain_sample, j1, CHAIN_D1): 0.2,
            _candidate_index(chain_sample, j2, CHAIN_G): 0.7,
            _candidate_index(chain_sample, j2, CHAIN_D2): 0.3,
        },
    )
    assert torch.equal(
        soft_goal_reachability(prob, chain_batch),
        soft_goal_reachability(prob, chain_batch, horizon_cap=24),
    )


def test_horizon_cap_truncates_the_propagation():
    sample = _long_chain_sample()
    batch = collate_samples([sample], device="cpu")
    j1, j2, j3 = (_decision_index(sample, node) for node in (2, 4, 6))
    prob = _prob_from_table(
        batch,
        {
            _candidate_index(sample, j1, 4): 1.0,
            _candidate_index(sample, j2, 6): 1.0,
            _candidate_index(sample, j3, 8): 1.0,
        },
    ).requires_grad_(True)

    full = soft_goal_reachability(prob, batch)
    assert abs(float(full[0].detach()) - 1.0) < 1e-6      # 3 轮之后 J1 才拿到 1.0
    assert abs(float(soft_goal_reachability(prob, batch, horizon_cap=3)[0].detach()) - 1.0) < 1e-6
    # cap 太小时传播被截断：J1 的值还是 0
    assert float(soft_goal_reachability(prob, batch, horizon_cap=1)[0].detach()) == 0.0
    assert float(soft_goal_reachability(prob, batch, horizon_cap=2)[0].detach()) == 0.0

    soft_goal_loss(soft_goal_reachability(prob, batch, horizon_cap=2)).backward()
    assert prob.grad is not None and torch.isfinite(prob.grad).all()


def test_invalid_horizon_cap_is_rejected(chain_batch):
    model = GraphFlowDenoiser(d_model=16, ffn_hidden=32)
    diffusion = CategoricalDiffusion(
        NoiseSchedule(T=4, schedule="linear", beta_start=0.05, beta_end=0.5)
    )
    with pytest.raises(ValueError):
        recurrent_reverse_loss(
            model, diffusion, chain_batch, LossWeights(goal_horizon_cap=0), max_steps=4
        )
