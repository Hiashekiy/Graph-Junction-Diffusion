from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
