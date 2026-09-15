from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dashboard.server import (
    DashboardService,
    _route_payload,
    discover_datasets,
    discover_metrics,
    discover_runs,
)


class _TinyGraph:
    def __init__(self, edges):
        self.edges = {frozenset(edge) for edge in edges}

    def has_edge(self, source, target):
        return frozenset((source, target)) in self.edges


def test_dashboard_discovers_nested_runs_and_datasets(tmp_path: Path) -> None:
    run = tmp_path / "outputs" / "runs" / "family" / "run-a"
    run.mkdir(parents=True)
    (run / "run_config.json").write_text("{}", encoding="utf-8")
    (run / "best.pt").write_bytes(b"checkpoint")
    data = tmp_path / "data"
    data.mkdir()
    (data / "controlled_test.pkl").write_bytes(b"data")

    runs = discover_runs(tmp_path)
    datasets = discover_datasets(tmp_path)

    assert list(runs) == ["family/run-a"]
    assert list(datasets) == ["controlled_test.pkl"]


def test_dashboard_discovers_grouped_datasets(tmp_path: Path) -> None:
    """数据集按来源分组后仍要被发现，而且 id 保持文件名（历史产物按文件名记录）。"""
    for group, name in (
        ("controlled", "controlled_test.pkl"),
        ("mixed", "mixed_oldv1_train.pkl"),
    ):
        directory = tmp_path / "data" / group
        directory.mkdir(parents=True)
        (directory / name).write_bytes(b"data")

    datasets = discover_datasets(tmp_path)

    assert list(datasets) == ["controlled_test.pkl", "mixed_oldv1_train.pkl"]
    assert datasets["mixed_oldv1_train.pkl"].path.parent.name == "mixed"
    assert datasets["mixed_oldv1_train.pkl"].group == "mixed"
    assert datasets["controlled_test.pkl"].group == "controlled"
    assert datasets["controlled_test.pkl"].relative == "data/controlled/controlled_test.pkl"


def test_catalog_exposes_dataset_group_for_the_picker(tmp_path: Path) -> None:
    """面板靠 group 把数据集按目录分组、靠 relative 显示文件在哪。"""
    for group, name in (
        ("controlled", "controlled_test.pkl"),
        ("mixed", "mixed_oldv1_train.pkl"),
    ):
        directory = tmp_path / "data" / group
        directory.mkdir(parents=True)
        (directory / name).write_bytes(b"data")

    entry = next(
        item
        for item in DashboardService(tmp_path).catalog()["datasets"]
        if item["id"] == "mixed_oldv1_train.pkl"
    )

    assert entry["group"] == "mixed"
    assert entry["relative"] == "data/mixed/mixed_oldv1_train.pkl"
    assert entry["label"] == "mixed oldv1 train"


def test_dashboard_reads_compact_eval_and_multipath_metrics(tmp_path: Path) -> None:
    run = tmp_path / "outputs" / "runs" / "model"
    run.mkdir(parents=True)
    (run / "run_config.json").write_text("{}", encoding="utf-8")
    (run / "best.pt").write_bytes(b"checkpoint")
    (run / "eval_test.json").write_text(
        json.dumps({"metrics": {"num_queries": 10, "goal_hit_rate": 0.8}}),
        encoding="utf-8",
    )
    (run / "mp_test_k2.json").write_text(
        json.dumps(
            {
                "data": "data/controlled/controlled_test.pkl",
                "top_k": 2,
                "null_policy": "skip",
                "results": [
                    {
                        "multi_best": {"num_queries": 10, "goal_hit_rate": 1.0},
                        "coverage_rate": 1.0,
                        "optimal_coverage_rate": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = discover_metrics(discover_runs(tmp_path), tmp_path)

    assert [(row["decoding"], row["goal_hit_rate"]) for row in rows] == [
        ("multi k=2 / skip", 1.0),
        ("single", 0.8),
    ]
    assert rows[0]["coverage_rate"] == 1.0


def test_route_payload_uses_and_validates_physical_edges() -> None:
    decoded = SimpleNamespace(path=[5, 8, 13], status="goal", reason="")
    graph = _TinyGraph([(5, 8), (8, 13)])

    payload = _route_payload(decoded, 1, graph)

    assert payload["nodes"] == [5, 8, 13]
    assert payload["edges"] == [[5, 8], [8, 13]]

    with pytest.raises(ValueError, match="不属于原图"):
        _route_payload(decoded, 1, _TinyGraph([(5, 8)]))


def test_single_decode_contrast_explains_readout_vs_chain_state(manual_sample) -> None:
    """single 解码用的是 readout（默认=最终概率组内 argmax），链状态只作诊断。

    实测 #92：readout 18 跳到达，采样链在最后一个路口抽到 NULL 判 broken —— 面板要把
    这两件事同时摊开，别让用户以为 single 还跟着采样状态走。
    """
    from dashboard.server import _single_decode_contrast
    from src.data.collate import collate_samples, null_candidate_of_decision
    from src.evaluation.path_decoder import decode_flat

    batch = collate_samples([manual_sample], device="cpu")
    readout = decode_flat(manual_sample, batch.target_candidate, decision_offset=0, candidate_offset=0)
    null_state = null_candidate_of_decision(batch)
    chain_state = torch.where(null_state >= 0, null_state, batch.target_candidate)
    frames = [{"clean_path": readout.path, "clean_status": "goal", "clean_reason": ""}]

    contrast = _single_decode_contrast(
        manual_sample, frames, readout, chain_state, "final_prob_argmax", True
    )

    assert contrast["readout_mode"] == "final_prob_argmax"
    assert contrast["chain_stochastic"] is True
    assert contrast["readout"]["status"] == "goal"
    assert contrast["readout"]["hops"] == 7
    assert contrast["readout"]["path_cost"] == 7.0      # 无权图退化成跳数
    assert contrast["chain_state"]["status"] == "broken"
    assert "NULL" in contrast["chain_state"]["reason"]
    assert contrast["clean_prediction"]["status"] == "goal"
    assert contrast["clean_path"] == [int(node) for node in readout.path]
    # 没有扩散帧（未请求 include_diffusion）时不返回对照块
    assert _single_decode_contrast(manual_sample, [], readout, chain_state, "single", True) is None


def test_route_lanes_collapse_the_common_prefix() -> None:
    """所有路线共用的前缀必须画成一条主干线，而不是一束平行线。

    实测反馈：8 条存活路线在"还没分叉"的公共前缀上各自画一条平行偏移线，看起来
    就像"一开始就分叉了"。规则：一条边被**当前显示的全部**路线经过时不携带区分
    信息 -> 不铺车道、用中性色 TRUNK_COLOR 画一条；只有真正分叉之后的边才着色。
    """
    script = (
        Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"
    ).read_text(encoding="utf-8")

    # 判定：uniqueRoutes == totalRoutes -> shared（公共前缀）
    assert "(usage.uniqueRoutes.get(key)?.size || 0) === usage.totalRoutes" in script
    assert "shared: true" in script
    # 绘制：公共前缀用中性主干色
    assert "TRUNK_COLOR" in script
    assert "geometry.shared ? TRUNK_COLOR : color" in script


def test_route_payload_reports_real_cost_on_weighted_graphs() -> None:
    """加权图上 ``cost``（跳数）与 ``path_cost``（边权和）必须分开报。"""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=3.0)
    graph.add_edge(1, 2, weight=7.0)

    # 形状对齐 PathCandidate：nodes / status / reason / log_prob / cost / path_cost
    weighted = SimpleNamespace(
        nodes=[0, 1, 2], status="goal", reason="", log_prob=-0.5, cost=2, path_cost=10.0
    )
    payload = _route_payload(weighted, 1, graph)
    assert payload["path_cost"] == 10.0
    assert payload["weighted"] is True
    assert payload["cost"] == 2

    # 单路径解码结果没有 path_cost 字段，面板要自己按边权算
    plain = SimpleNamespace(path=[0, 1, 2], status="goal", reason="")
    assert _route_payload(plain, 1, graph)["path_cost"] == 10.0

    # 无权图：path_cost 退化成跳数
    bare = nx.Graph()
    bare.add_edges_from([(0, 1), (1, 2)])
    bare_payload = _route_payload(plain, 1, bare)
    assert bare_payload["path_cost"] == 2.0
    assert bare_payload["weighted"] is False


def test_route_reveal_is_measured_not_dashed() -> None:
    """Route drawing must not depend on dash geometry.

    Chromium resolves ``stroke-dasharray`` in *screen* pixels as soon as a
    stroke uses ``vector-effect: non-scaling-stroke``, while
    ``getTotalLength()`` keeps reporting user units.  A dash as long as an edge
    therefore covered only ``1 / renderScale`` of it (measured: 84% of every
    edge at the dashboard's usual 1.19 scale) and the missing tail landed right
    in front of the next node.  The reveal now samples real arc length, so the
    route strokes stay in user space and no dash pattern is involved.
    """
    static = Path(__file__).resolve().parents[1] / "dashboard" / "static"
    script = (static / "app.js").read_text(encoding="utf-8")
    styles = (static / "styles.css").read_text(encoding="utf-8")

    assert "strokeDasharray" not in script
    assert "getPointAtLength" in script and "getTotalLength" in script
    for selector in (".route-casing {", ".route {"):
        rule = styles.split(selector, 1)[1].split("}", 1)[0]
        assert "vector-effect" not in rule
