"""Weighted 扩展的验收测试（实施方案第 16 节 + 向后兼容）。

覆盖：

3. weighted GT 必须是 Dijkstra 解（S-A-G 而不是 1 跳直连）；
4. 一条物理边的两个 message 方向拿到**相同**的 normalized cost；
5. per-graph mean 归一化的数值与 argmin 不变性；
6. cost 真的影响网络（改一条边的 cost，weighted 模型输出必须变；无权模型不受影响）；
7. edge_cost_encoder / k_cost_proj / v_cost_proj 必须拿到非零梯度；
8. Dijkstra baseline 是完美 oracle（GoalHit=Optimal=1，CostRatio=1）；
9. weighted 数据集的 conflict rate / bfs cost ratio 统计；
10. 向后兼容：weighted=false 时参数集合里没有任何 cost 参数，旧 checkpoint 照常加载。
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
import torch

from src.data.collate import collate_samples
from src.data.dataset import GraphQueryDataset
from src.data.dataset_builder import build_sample, dataset_statistics
from src.evaluation.baselines import baseline_summary, shortest_path
from src.evaluation.metrics import evaluate_sample
from src.evaluation.path_decoder import DecodeResult
from src.models.denoiser import GraphFlowDenoiser
from src.training.checkpoint import load_checkpoint
from src.training.setup import build_model
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CHECKPOINT = REPO_ROOT / "outputs" / "runs" / "v2_rev2_mixed" / "best.pt"

# conftest 里的手工图（节点编号固定，见 tests/conftest.py 的 docstring）：
#     s(0) - c(1) - J1(2) - a(3) - b(4) - x(5) - J2(6) - g(7)
#                   |                              |
#                   h(8)-i(9)          d(10)-e(11)-f(12) 绕行回到 J2
S, C, J1, A, B, X, J2, G, H, I, D, E, F = range(13)
MANUAL_EDGES = [
    (S, C), (C, J1), (J1, H), (H, I), (J1, A), (A, B), (B, X), (X, J2),
    (J1, D), (D, E), (E, F), (F, J2), (J2, G),
]
# 主干上的"桥"边：把它改贵，加权最短路就会改走 J1-D-E-F-J2 的绕行段
BRIDGE_EDGE = (J1, A)
DETOUR_COST = 4.0     # J1-D-E-F-J2 四条边各 1.0
CHEAP_TOTAL = 7.0     # S-C-J1 + J1-A-B-X-J2 + J2-G = 1+1+3+1
EXPENSIVE_TOTAL = 1.0 + 1.0 + DETOUR_COST + 1.0


def make_weighted_graph(bridge_weight: float = 1.0) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(MANUAL_EDGES)
    for u, v in graph.edges():
        weight = float(bridge_weight) if {u, v} == set(BRIDGE_EDGE) else 1.0
        graph.edges[u, v]["weight"] = weight
    graph.graph["weighted"] = True
    return graph


def make_weighted_sample(bridge_weight: float = 1.0):
    return build_sample(make_weighted_graph(bridge_weight), S, G)


def make_model(use_edge_cost: bool, seed: int = 0) -> GraphFlowDenoiser:
    torch.manual_seed(seed)
    return GraphFlowDenoiser(
        d_model=16,
        num_node_types=4,
        num_edge_states=2,
        ffn_hidden=32,
        flow_steps=1,
        slot_embedding=False,
        use_edge_cost=use_edge_cost,
        edge_cost_hidden=8,
    )


# ---------------------------------------------------------------------------
# 3. weighted GT = Dijkstra
# ---------------------------------------------------------------------------
def test_weighted_gt_prefers_the_cheap_two_hop_route():
    """S -1- A -1- G 与 S -10- G：加权 GT 必须是 S-A-G，而不是跳数更少的直连。"""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
    graph.edges[0, 1]["weight"] = 1.0
    graph.edges[1, 2]["weight"] = 1.0
    graph.edges[0, 2]["weight"] = 10.0
    graph.graph["weighted"] = True
    assert build_sample(graph, 0, 2).gt_path == [0, 1, 2]

    plain = nx.Graph()
    plain.add_edges_from([(0, 1), (1, 2), (0, 2)])
    # 同一张拓扑、无权时 GT 就是最少跳数的直连（旧行为逐字不变）
    assert build_sample(plain, 0, 2).gt_path == [0, 2]


def _path_cost(sample) -> float:
    return sum(
        float(sample.graph.edges[u, v]["weight"])
        for u, v in zip(sample.gt_path[:-1], sample.gt_path[1:])
    )


def test_expensive_bridge_is_avoided_by_the_weighted_gt():
    cheap = make_weighted_sample(0.5)
    expensive = make_weighted_sample(9.0)
    assert A in cheap.gt_path and D not in cheap.gt_path
    assert A not in expensive.gt_path and D in expensive.gt_path
    # 两条路线的**跳数相同**（都是 7 跳 / 6 条边），区别只在 cost —— 这正是
    # weighted 任务与 BFS 任务的本质区别：只看跳数根本分不出哪条更优。
    assert cheap.gt_length == expensive.gt_length
    assert _path_cost(cheap) == pytest.approx(1.0 + 1.0 + 0.5 + 1.0 + 1.0 + 1.0 + 1.0)
    assert _path_cost(expensive) == pytest.approx(EXPENSIVE_TOTAL)


# ---------------------------------------------------------------------------
# 4 / 5. cost 映射与归一化
# ---------------------------------------------------------------------------
def test_both_message_directions_share_the_same_normalized_cost():
    batch = collate_samples([make_weighted_sample(9.0)])
    cost = batch.message_edge_cost()
    assert cost.shape[0] == batch.edge_index.shape[1]
    assert torch.allclose(cost, batch.physical_edge_cost_norm[batch.msg_to_phys_edge])
    for physical in range(batch.num_physical_edges):
        values = cost[batch.msg_to_phys_edge == physical]
        assert values.numel() == 2
        assert torch.allclose(values, values[0].expand_as(values))


def test_graph_mean_normalization_is_purely_proportional():
    """weights = [2, 4, 6] -> mean 4 -> [0.5, 1.0, 1.5]，argmin 不变。"""
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    for (u, v), weight in zip(graph.edges(), (2.0, 4.0, 6.0)):
        graph.edges[u, v]["weight"] = weight
    graph.graph["weighted"] = True
    sample = build_sample(graph, 0, 3)
    batch = collate_samples([sample])
    assert torch.allclose(
        batch.physical_edge_cost, torch.tensor([2.0, 4.0, 6.0])
    )
    assert torch.allclose(
        batch.physical_edge_cost_norm, torch.tensor([0.5, 1.0, 1.5])
    )
    # argmin_P sum w_e 只依赖比例关系：整体缩放不改变最短路
    scaled = graph.copy()
    for (u, v), weight in zip(scaled.edges(), (0.5, 1.0, 1.5)):
        scaled.edges[u, v]["weight"] = weight
    assert nx.shortest_path(scaled, 0, 3, weight="weight") == nx.shortest_path(
        graph, 0, 3, weight="weight"
    )


def test_branch_cost_is_the_sum_of_its_physical_edges():
    sample = make_weighted_sample(9.0)
    batch = collate_samples([sample])
    assert batch.candidate_branch_cost.shape[0] == sample.num_candidates
    for index, branch in enumerate(sample.field.candidates.candidate_branch):
        if branch is None:
            # C(NULL) = 0
            assert float(batch.candidate_branch_cost[index]) == 0.0
            continue
        expected = sum(
            float(sample.graph.edges[u, v]["weight"])
            for u, v in zip(branch.nodes[:-1], branch.nodes[1:])
        )
        assert float(batch.candidate_branch_cost[index]) == pytest.approx(expected)


def test_unweighted_batch_costs_are_all_one(manual_sample):
    """旧数据 / 无权数据：cost 缺省为 1.0，归一化后恒为 1.0。"""
    batch = collate_samples([manual_sample])
    assert batch.is_weighted is False
    assert torch.allclose(
        batch.physical_edge_cost, torch.ones(batch.num_physical_edges)
    )
    assert torch.allclose(
        batch.physical_edge_cost_norm, torch.ones(batch.num_physical_edges)
    )


# ---------------------------------------------------------------------------
# 6 / 7. cost 真的进入网络，并且梯度能回传
# ---------------------------------------------------------------------------
def _step(model, batch, z):
    return model.step(batch, model.init_nodes(batch), z, 9)


def test_edge_cost_changes_the_weighted_model_but_not_the_unweighted_one():
    weighted = make_model(use_edge_cost=True)
    unweighted = make_model(use_edge_cost=False)   # 同一个 seed -> 同一套 backbone 初值
    batch_cheap = collate_samples([make_weighted_sample(1.0)])
    batch_expensive = collate_samples([make_weighted_sample(9.0)])
    # 拓扑 / start / goal / z_t 完全相同，唯一变化的是那一条边的 cost
    z = batch_cheap.target_candidate
    with torch.no_grad():
        out_cheap = _step(weighted, batch_cheap, z)
        out_expensive = _step(weighted, batch_expensive, z)
        ref_cheap = _step(unweighted, batch_cheap, z)
        ref_expensive = _step(unweighted, batch_expensive, z)

    assert not torch.allclose(out_cheap.attn, out_expensive.attn)
    assert not torch.allclose(out_cheap.H_next, out_expensive.H_next)
    assert not torch.allclose(
        out_cheap.candidate_logits, out_expensive.candidate_logits
    )
    # 无权模型看到的是完全相同的输入，输出必须逐位一致
    assert torch.allclose(ref_cheap.attn, ref_expensive.attn)
    assert torch.allclose(ref_cheap.H_next, ref_expensive.H_next)
    assert torch.allclose(ref_cheap.candidate_logits, ref_expensive.candidate_logits)


def test_cost_parameters_receive_non_zero_gradients():
    model = make_model(use_edge_cost=True)
    batch = collate_samples([make_weighted_sample(9.0)])
    out = _step(model, batch, batch.target_candidate)
    (-out.candidate_log_prob.sum()).backward()

    encoder_grad = model.edge_cost_encoder.net[0].weight.grad
    assert encoder_grad is not None and float(encoder_grad.abs().sum()) > 0.0
    for name in ("k_cost_proj", "v_cost_proj"):
        projection = getattr(model.graph_flow, name)
        assert projection is not None
        assert projection.weight.grad is not None
        assert float(projection.weight.grad.abs().sum()) > 0.0


def test_weighted_model_rejects_a_batch_without_edge_cost(manual_sample):
    """拿不到 cost 时必须显式报错，而不是静默喂 0。"""
    batch = collate_samples([manual_sample])
    batch.physical_edge_cost = torch.zeros(0)
    batch.physical_edge_cost_norm = torch.zeros(0)
    model = make_model(use_edge_cost=True)
    with pytest.raises(ValueError):
        _step(model, batch, batch.target_candidate)


def test_unweighted_graph_flow_rejects_an_edge_cost_feature(manual_sample):
    model = make_model(use_edge_cost=False)
    batch = collate_samples([manual_sample])
    with pytest.raises(ValueError):
        model.graph_flow(
            H_t=model.init_nodes(batch),
            edge_index=batch.edge_index,
            edge_feat=model.edge_state_encoder(batch, batch.target_candidate),
            tau_t=model.time_embedding(5, batch.num_graphs),
            fixed_mask=batch.start_goal_mask,
            graph_node_ptr=batch.graph_node_ptr,
            edge_cost_feat=torch.zeros(batch.edge_index.shape[1], model.d_model),
        )


# ---------------------------------------------------------------------------
# 8. Dijkstra baseline 是完美 oracle
# ---------------------------------------------------------------------------
def test_dijkstra_baseline_is_a_perfect_oracle_on_weighted_graphs():
    sample = make_weighted_sample(9.0)
    dataset = GraphQueryDataset([sample])
    stats = baseline_summary(dataset)
    assert stats["shortest_path"]["goal_hit_rate"] == 1.0
    assert stats["shortest_path"]["mean_cost_ratio"] == pytest.approx(1.0)
    # 按跳数贪心的 baseline 在带权图上必然退化（这正是 weighted 任务的意义）
    assert stats["greedy_bfs"]["mean_cost_ratio"] > 1.0

    path = shortest_path(sample.graph, sample.start, sample.goal)
    record = evaluate_sample(sample, DecodeResult("goal", path, 0, ""))
    assert record.goal_hit and record.optimal
    assert record.cost_ratio == pytest.approx(1.0)
    assert record.optimal_cost == pytest.approx(EXPENSIVE_TOTAL)


# ---------------------------------------------------------------------------
# 9. 数据集 sanity check
# ---------------------------------------------------------------------------
def test_weighted_dataset_statistics_expose_the_required_fields():
    dataset = GraphQueryDataset([make_weighted_sample(1.0), make_weighted_sample(9.0)])
    stats = dataset_statistics(dataset)
    weighted = stats["weighted"]
    for key in (
        "weight_mean", "weight_std", "weight_min", "weight_max",
        "weighted_conflict_rate", "bfs_cost_ratio", "dijkstra_cost_ratio",
        "gt_path_is_weighted_optimal_fraction",
    ):
        assert key in weighted
    # 一半样本的加权最优与跳数最优不同
    assert weighted["weighted_conflict_rate"] == pytest.approx(0.5)
    assert weighted["bfs_cost_ratio"] > 1.0
    assert weighted["dijkstra_cost_ratio"] == 1.0
    assert weighted["gt_path_is_weighted_optimal_fraction"] == 1.0
    assert weighted["warnings"] == []


def test_unweighted_dataset_statistics_have_no_weighted_section(manual_sample):
    stats = dataset_statistics(GraphQueryDataset([manual_sample]))
    assert "weighted" not in stats


# ---------------------------------------------------------------------------
# 10. 向后兼容
# ---------------------------------------------------------------------------
def test_unweighted_model_has_no_cost_parameters():
    model = make_model(use_edge_cost=False)
    assert model.edge_cost_encoder is None
    assert model.graph_flow.k_cost_proj is None
    assert model.graph_flow.v_cost_proj is None
    assert not any("cost" in name for name in model.state_dict())


def test_unweighted_config_defaults_to_no_edge_cost():
    config = load_config(REPO_ROOT / "configs" / "graph_flow.yaml")
    assert config.section("model").get("use_edge_cost", False) is False
    model = build_model(config)
    assert model.edge_cost_encoder is None
    assert not any("cost" in name for name in model.state_dict())


def test_weighted_config_enables_the_edge_cost_path():
    config = load_config(REPO_ROOT / "configs" / "graph_flow_weighted.yaml")
    assert config.section("data").get("weighted") is True
    assert config.section("model").get("use_edge_cost") is True
    model = build_model(config)
    assert model.use_edge_cost
    assert model.edge_cost_encoder is not None
    assert "graph_flow.k_cost_proj.weight" in model.state_dict()
    assert "graph_flow.v_cost_proj.weight" in model.state_dict()
    assert "edge_cost_encoder.net.0.weight" in model.state_dict()


def test_only_graph_mean_normalization_is_implemented(tmp_path):
    config = load_config(
        REPO_ROOT / "configs" / "graph_flow_weighted.yaml",
        ["model.edge_cost.normalization=minmax"],
    )
    with pytest.raises(NotImplementedError):
        build_model(config)


@pytest.mark.skipif(
    not LEGACY_CHECKPOINT.exists(), reason="legacy checkpoint is not available"
)
def test_legacy_checkpoint_still_loads_with_the_unweighted_config():
    """旧 checkpoint 的参数集合必须与新代码的无权模型逐 key 相同。"""
    model = build_model(load_config(REPO_ROOT / "configs" / "graph_flow.yaml"))
    payload = load_checkpoint(LEGACY_CHECKPOINT, model=model)
    assert set(payload["model"]) == set(model.state_dict())
    assert not any("cost" in key for key in payload["model"])
