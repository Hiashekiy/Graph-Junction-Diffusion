from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dashboard.server import (
    DashboardService,
    GeoLayout,
    _graph_payload,
    _letterbox,
    _route_payload,
    discover_datasets,
    discover_runs,
    geo_payload,
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
        ("weighted", "weighted_test.pkl"),
        ("unweighted", "unweighted_train.pkl"),
    ):
        directory = tmp_path / "data" / group
        directory.mkdir(parents=True)
        (directory / name).write_bytes(b"data")

    datasets = discover_datasets(tmp_path)

    assert list(datasets) == ["unweighted_train.pkl", "weighted_test.pkl"]
    assert datasets["unweighted_train.pkl"].path.parent.name == "unweighted"
    assert datasets["unweighted_train.pkl"].group == "unweighted"
    assert datasets["weighted_test.pkl"].group == "weighted"
    assert datasets["weighted_test.pkl"].relative == "data/weighted/weighted_test.pkl"


def test_catalog_exposes_dataset_group_for_the_picker(tmp_path: Path) -> None:
    """面板靠 group 把数据集按目录分组、靠 relative 显示文件在哪。"""
    for group, name in (
        ("weighted", "weighted_test.pkl"),
        ("unweighted", "unweighted_train.pkl"),
    ):
        directory = tmp_path / "data" / group
        directory.mkdir(parents=True)
        (directory / name).write_bytes(b"data")

    entry = next(
        item
        for item in DashboardService(tmp_path).catalog()["datasets"]
        if item["id"] == "unweighted_train.pkl"
    )

    assert entry["group"] == "unweighted"
    assert entry["relative"] == "data/unweighted/unweighted_train.pkl"
    assert entry["label"] == "unweighted train"


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


# ---------------------------------------------------------------------------
# 地理底图（滴滴成都真实路网）
# ---------------------------------------------------------------------------
def _fake_geo_sample(nodes, edges, mapping, start=0, goal=1, decisions=()):
    """最小可用样本。

    ``geo_payload`` 只用到 ``graph`` 与 ``meta['local_to_global']``；
    ``_graph_payload`` 还要 ``start`` / ``goal`` / ``segments.decision_nodes``。
    """
    import networkx as nx
    from types import SimpleNamespace

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return SimpleNamespace(
        graph=graph,
        meta={"local_to_global": mapping},
        start=start,
        goal=goal,
        segments=SimpleNamespace(decision_nodes=list(decisions)),
    )


def _square_layout(scale: float = 1.0):
    """一个 2x2 的"城市"：全局 id 0..3 在 (0,0)-(1,1)，外加一条街边。"""
    positions = {
        0: (0.0, 0.0), 1: (1.0 * scale, 0.0),
        2: (0.0, 1.0 * scale), 3: (1.0 * scale, 1.0 * scale),
    }
    return GeoLayout(
        positions=positions,
        street_edges=[(0, 1), (1, 3), (3, 2), (2, 0), (0, 3)],
        lat0=30.69,
        coverage=0.962,
        source="data/didi/raw/chengdu/ChengDu.pkl",
    )


def test_letterbox_preserves_the_panel_aspect_ratio() -> None:
    from dashboard.server import PANEL_ASPECT

    # 方形输入 -> 补成宽屏（左右撑开），高度不变
    x0, x1, y0, y1 = _letterbox((0.0, 1.0, 0.0, 1.0), PANEL_ASPECT)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(PANEL_ASPECT)
    assert (y1 - y0) == pytest.approx(1.0)
    assert ((x0 + x1) / 2, (y0 + y1) / 2) == pytest.approx((0.5, 0.5))

    # 又高又窄的输入 -> 补成宽屏（上下不动，左右撑开）
    x0, x1, y0, y1 = _letterbox((0.0, 0.2, 0.0, 3.0), PANEL_ASPECT)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(PANEL_ASPECT)
    assert (y1 - y0) == pytest.approx(3.0)


def test_geo_payload_normalises_corridor_and_streets_into_the_unit_box() -> None:
    from dashboard.server import PANEL_ASPECT

    sample = _fake_geo_sample([0, 1], [(0, 1)], mapping=[0, 1])
    payload = geo_payload(sample, _square_layout())

    xs = [point[0] for point in payload["node_positions"].values()]
    ys = [point[1] for point in payload["node_positions"].values()]
    # corridor 必须落在 [0,1]（前端直接线性映射到画布，越界就画到框外了）
    assert 0.0 <= min(xs) and max(xs) <= 1.0
    assert 0.0 <= min(ys) and max(ys) <= 1.0
    # 形状不能被拉伸：corridor 在两个方向上的"占幅比"要反映出 letterbox 后的框
    assert payload["geo"] is True
    assert payload["coverage"] == pytest.approx(0.962)
    assert payload["lat0"] == pytest.approx(30.69)
    # 比例尺：整幅宽度 = crop_km[0]，且 x/y 同尺度（等距圆柱，已按 cos(lat0) 校正）
    assert payload["km_per_x_unit"] == pytest.approx(payload["crop_km"][0])
    assert payload["crop_km"][0] / payload["crop_km"][1] == pytest.approx(PANEL_ASPECT)
    assert payload["km_per_x_unit"] > 0


def test_geo_payload_keeps_street_edges_as_bare_coordinate_pairs() -> None:
    """街道底图回传的是"算好的两端坐标"，不是节点 id。

    底图是纯装饰层；把全局 OSM id 混进面板只会和 corridor 的**样本内**编号空间撞车
    （两套编号都以 0 开头，但含义完全不同）。
    """
    sample = _fake_geo_sample([0, 1], [(0, 1)], mapping=[0, 1])
    payload = geo_payload(sample, _square_layout())

    assert payload["street_edges"]
    for edge in payload["street_edges"]:
        assert len(edge) == 4
        assert all(isinstance(value, float) for value in edge)
    assert "street_truncated" in payload


def _strip_layout(count: int = 6) -> GeoLayout:
    """一条横向走廊 + 两侧各一条平行街：条条都落在裁剪框里，用来验证上限。"""
    positions = {}
    edges = []
    for index in range(count):
        positions[index] = (index * 0.1, 0.0)
        positions[100 + index] = (index * 0.1, 0.05)
        edges.append((index, 100 + index))
        if index:
            edges.append((index - 1, index))
            edges.append((100 + index - 1, 100 + index))
    return GeoLayout(
        positions=positions, street_edges=edges, lat0=30.69, coverage=1.0,
        source="data/didi/raw/chengdu/ChengDu.pkl",
    )


def test_geo_payload_caps_street_edges() -> None:
    # 走廊取整条 strip 的两端（映射到全局 0 与 5），否则裁剪框比街道还小
    sample = _fake_geo_sample([0, 1], [(0, 1)], mapping=[0, 5])
    layout = _strip_layout()
    uncapped = geo_payload(sample, layout)
    assert len(uncapped["street_edges"]) > 2
    assert uncapped["street_truncated"] is False

    capped = geo_payload(sample, layout, max_street_edges=2)
    assert len(capped["street_edges"]) == 2
    assert capped["street_truncated"] is True


def test_geo_payload_returns_none_without_a_local_to_global_map() -> None:
    """没有反查表就不能猜 —— 猜错会把"别的城市的坐标"画上来。"""
    sample = _fake_geo_sample([0, 1], [(0, 1)], mapping=None)
    assert geo_payload(sample, _square_layout()) is None


def test_geo_payload_returns_none_when_a_node_has_no_coordinate() -> None:
    """缺一个坐标就整张退回弹簧布局，而不是画一条穿过半个城市的假边。"""
    layout = _square_layout()
    broken = GeoLayout(
        positions={0: layout.positions[0]},       # 少了全局节点 1
        street_edges=layout.street_edges,
        lat0=layout.lat0,
        coverage=layout.coverage,
        source=layout.source,
    )
    sample = _fake_geo_sample([0, 1], [(0, 1)], mapping=[0, 1])
    assert geo_payload(sample, broken) is None


def test_graph_payload_switches_between_geo_and_spring_layout(manual_sample) -> None:
    plain = _graph_payload(manual_sample)
    assert plain["geo"] is False
    assert "street_edges" not in plain
    # 合成图退回弹簧布局，坐标归一化到 [0,1]
    for node in plain["nodes"]:
        assert 0.0 <= node["x"] <= 1.0 and 0.0 <= node["y"] <= 1.0

    sample = _fake_geo_sample([0, 1], [(0, 1)], mapping=[0, 1])
    geo = geo_payload(sample, _square_layout())
    payload = _graph_payload(sample, geo)
    assert payload["geo"] is True
    assert payload["street_edges"] == geo["street_edges"]
    assert payload["crop_km"] == geo["crop_km"]
    # node_positions 是内部字段，不该泄漏到前端
    assert "node_positions" not in payload


def test_panel_geometry_matches_between_backend_and_frontend() -> None:
    """后端 letterbox 与前端映射必须用**同一套**画布常量。

    两边一漂移，真实路网就会被拉伸：成都经度方向已经按 cos(30.7°) 缩短了 19%，
    再被前端按 [0,1] 各自拉伸一次就完全不像地图了。这条只能靠静态比对钉住。
    """
    import re

    from dashboard.server import PANEL_HEIGHT, PANEL_MARGIN, PANEL_WIDTH

    script = (
        Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    for name, expected in (
        ("GRAPH_MARGIN", PANEL_MARGIN),
        ("GRAPH_WIDTH", PANEL_WIDTH),
        ("GRAPH_HEIGHT", PANEL_HEIGHT),
    ):
        match = re.search(rf"const {name} = ([0-9.]+);", script)
        assert match, f"app.js 里找不到 {name}"
        assert float(match.group(1)) == pytest.approx(expected), name


def test_street_basemap_bypasses_node_ids() -> None:
    """街道层必须直接用后端给的坐标，不能过 graphCoordinates。"""
    script = (
        Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"
    ).read_text(encoding="utf-8")
    body = script.split("function renderStreetBasemap", 1)[1].split("\n}", 1)[0]
    assert "panelX(edge[0])" in body and "panelY(edge[1])" in body
    assert "coordinates" not in body


# ---------------------------------------------------------------------------
# run / 数据集发现：名字变动 + 过期快照
# ---------------------------------------------------------------------------
def _write_run(root: Path, name: str, config: dict, live_config: str = None) -> None:
    run = root / "outputs" / "runs" / name
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "best.pt").write_bytes(b"checkpoint")
    if live_config is not None:
        (root / "configs").mkdir(exist_ok=True)
        (root / "configs" / f"{name}.yaml").write_text(live_config, encoding="utf-8")


def test_discover_runs_prefers_the_directory_name_over_a_stale_snapshot(tmp_path: Path) -> None:
    """合并/改名过的 run 目录里留着旧快照，live config 必须按**目录名**去找。

    实测 outputs/runs/didi_chengdu/run_config.json 里写的还是
    paths.run_name=didi_chengdu_flow1_weighted_new、paths.data_dir=data/didi_chengdu_gjd、
    data.coords_file=data/DiDiChengduXian/...（三个都已失效）。按快照去找配置会一个都
    找不到，然后静默退回一份过期快照 —— 面板就再也画不出街道底图了。
    """
    (tmp_path / "data" / "didi" / "raw" / "chengdu").mkdir(parents=True)
    (tmp_path / "data" / "didi" / "raw" / "chengdu" / "ChengDu.pkl").write_bytes(b"coords")
    _write_run(
        tmp_path,
        "didi_chengdu",
        {
            "paths": {"run_name": "didi_chengdu_flow1_weighted_new",
                      "data_dir": "data/didi_chengdu_gjd"},
            "data": {"source": "didi_chengdu", "weighted": True,
                     "coords_file": "data/DiDiChengduXian/dead/ChengDu.pkl"},
            "model": {"use_edge_cost": True, "flow_steps": 1},
        },
        live_config=(
            "data:\n"
            "  source: didi_chengdu\n"
            "  coords_file: data/didi/raw/chengdu/ChengDu.pkl\n"
            "model:\n"
            "  use_edge_cost: true\n"
            "paths:\n"
            "  run_name: didi_chengdu\n"
            "  data_dir: data/didi/graph/chengdu\n"
        ),
    )

    run = discover_runs(tmp_path)["didi_chengdu"]

    assert run.data_dir == "data/didi/graph/chengdu"          # 不是过期的 data/didi_chengdu_gjd
    assert run.coords_file == "data/didi/raw/chengdu/ChengDu.pkl"
    assert run.live_config == "configs/didi_chengdu.yaml"
    assert run.source == "didi_chengdu"
    assert run.is_geo is True
    assert run.kind == "didi"
    assert run.kind_label == "滴滴·带权"


def test_discover_runs_clears_coords_when_the_file_is_gone(tmp_path: Path) -> None:
    """坐标文件不存在时必须显式置空，而不是把死路径传下去。"""
    _write_run(
        tmp_path,
        "didi_chengdu",
        {"paths": {"data_dir": "data/didi/graph/chengdu"}, "data": {"source": "didi_chengdu"}},
        live_config=(
            "data:\n"
            "  source: didi_chengdu\n"
            "  coords_file: data/didi/raw/chengdu/GONE.pkl\n"
            "paths:\n"
            "  data_dir: data/didi/graph/chengdu\n"
        ),
    )
    run = discover_runs(tmp_path)["didi_chengdu"]
    assert run.coords_file == ""
    assert run.is_geo is False


def test_discover_runs_falls_back_to_the_snapshot_without_a_live_config(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "controlled_unweighted",
        {"paths": {"data_dir": "data/unweighted"}, "data": {"weighted": False}, "model": {}},
    )
    run = discover_runs(tmp_path)["controlled_unweighted"]
    assert run.data_dir == "data/unweighted"
    assert run.coords_file == ""
    assert run.live_config == ""
    assert run.kind == "unweighted"


def test_discover_datasets_skips_raw_inputs_and_caches(tmp_path: Path) -> None:
    """``raw/`` 下的 OSMnx 图、``_`` 前缀的缓存、``graph_global.pkl`` 都不是数据集。

    它们用 ``GraphQueryDataset.load`` 打开会直接抛异常，而面板只会在用户点下"生成"
    之后才报一句"推理未完成" —— 这种失败必须在发现阶段就挡掉。
    """
    (tmp_path / "data" / "didi" / "raw" / "chengdu").mkdir(parents=True)
    (tmp_path / "data" / "didi" / "raw" / "chengdu" / "ChengDu.pkl").write_bytes(b"x")
    (tmp_path / "data" / "didi" / "raw" / "chengdu" / "dicts.pkl").write_bytes(b"x")
    graph_dir = tmp_path / "data" / "didi" / "graph" / "chengdu"
    graph_dir.mkdir(parents=True)
    (graph_dir / "graph_global.pkl").write_bytes(b"x")
    (graph_dir / "_didi_candidates.pkl").write_bytes(b"x")
    (graph_dir / "test_1000.pkl").write_bytes(b"x")

    datasets = discover_datasets(tmp_path)

    assert list(datasets) == ["test_1000.pkl"]
    assert datasets["test_1000.pkl"].group == "didi/graph/chengdu"
    assert datasets["test_1000.pkl"].relative == "data/didi/graph/chengdu/test_1000.pkl"


# ---------------------------------------------------------------------------
# 指标页已经删掉 / 旧名字没有残留
# ---------------------------------------------------------------------------
def _static(name: str) -> str:
    return (Path(__file__).resolve().parents[1] / "dashboard" / "static" / name).read_text(
        encoding="utf-8"
    )


def test_metrics_tab_is_gone_from_the_frontend() -> None:
    html = _static("index.html")
    script = _static("app.js")
    styles = _static("styles.css")

    for banned in ("metrics-panel", "metrics-tab", "metric-chart", "model-filters",
                   "metric-summary", "metric-dataset"):
        assert banned not in html, banned
    for banned in ("renderMetrics", "renderMetricChart", "selectedModels",
                   "visibleMetrics", "metric-dataset", "model-filters"):
        assert banned not in script, banned
    assert "metric-summary" not in styles
    assert "model-filter" not in styles
    # 指标页之外的样式不能被误删
    assert ".street-edge" in styles and ".report-body" in styles


def test_backend_no_longer_serves_metrics() -> None:
    import dashboard.server as server

    assert not hasattr(server, "discover_metrics")
    assert not hasattr(server, "_metric_row")
    catalog = DashboardService(Path(__file__).resolve().parents[1]).catalog()
    assert "metrics" not in catalog
    assert sorted(catalog) == ["datasets", "models", "reports"]


def test_deleted_report_is_not_offered() -> None:
    """被删掉的报告不能继续挂在清单里（会变成一个恒 404 的死链接）。"""
    import dashboard.server as server

    paths = {relative for _, _, relative in server.REPORT_ARTIFACTS}
    assert "docs/REPORT_multipath_and_weighted.md" not in paths
    # 清单里的 id 必须唯一，否则报告页会出现两个同名按钮
    ids = [report_id for report_id, _, _ in server.REPORT_ARTIFACTS]
    assert len(ids) == len(set(ids))


def test_frontend_defaults_use_the_current_names() -> None:
    """默认模型/数据集不能再用改名前的字符串。

    旧代码写的是 ``v2_rev2_mixed`` + ``controlled_test.pkl``，改完之后它俩都不存在，
    ``preferred()`` 会静默退回列表第一项 —— 面板照样能开，但默认组合是错的。
    """
    script = _static("app.js")
    for stale in ("v2_rev2_mixed", "controlled_test.pkl", "v2_weighted_controlled",
                  "didi_chengdu_flow1_weighted"):
        assert stale not in script, stale
    assert "controlled_unweighted" in script
    assert "unweighted_test.pkl" in script
