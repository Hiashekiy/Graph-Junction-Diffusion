"""把 DiDi 真实数据集的样本画出来看（不需要模型 / 不需要 GPU）。

回答的问题是："这份数据集里的一条样本到底长什么样？"

每个 panel：

    浅灰细线   整张成都路网（**真实经纬度底图**，2891 节点 / 4403 边）
    蓝色细线   该样本的 OD corridor 子图（模型真正看到的图）
    红粗线     GT —— **真实司机走过的历史路径**
    绿虚线     Dijkstra 最短路（同一个 OD）—— 用来直观看出"GT 不是最短路"
    绿星/红星  start / goal
    橙色小点   decision node（模型要在这里做选择）

坐标来源：DiDi 附带的 ``ChengDu.pkl`` / ``graph.pkl`` 是 OSMnx 1.1.1 导出的
``MultiDiGraph``（``crs = epsg:4326``），**节点自带 x=经度 / y=纬度**，节点 id 与
``dicts.pkl`` 的 ``(u, v)`` 是同一套 OSM node id。所以底图是真实地理形状，不是
拓扑示意。默认用等距圆柱投影（``x = lon·cos(lat0)``，``y = lat``），在这个
9km × 10km 的范围内形变可以忽略。

用法::

    python tools/visualize_didi_samples.py
    python tools/visualize_didi_samples.py --per-split 4 --out outputs/figures/didi_samples.png
    python tools/visualize_didi_samples.py --zoom --out outputs/figures/didi_zoom.png
    python tools/visualize_didi_samples.py --topology   # 退回拓扑 spring 布局
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import didi_dataset as didi  # noqa: E402
from src.data.dataset import GraphQueryDataset  # noqa: E402
from src.evaluation import real_path_metrics as rpm  # noqa: E402

#: OSMnx 导出的路网（节点带经纬度）。相对项目根。
DEFAULT_COORDS = "data/didi/raw/chengdu/ChengDu.pkl"

# 中文字体（Windows 上一定有；没有就退回英文标签，绝不让画图挂掉）
for _font in ("Microsoft YaHei", "SimHei", "SimSun"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [_font] + list(
            matplotlib.rcParams["font.sans-serif"]
        )
        break
    except Exception:  # pragma: no cover
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

COLOR_FULL_EDGE = "#d9d9d9"
COLOR_FULL_NODE = "#eeeeee"
COLOR_CORRIDOR = "#9ecae1"
COLOR_GT = "#d62728"
COLOR_DIJKSTRA = "#2ca02c"
COLOR_DECISION = "#ff7f0e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="visualize DiDi dataset samples")
    parser.add_argument("--data", default="data/didi/graph/chengdu")
    parser.add_argument("--coords", default=DEFAULT_COORDS)
    parser.add_argument("--per-split", type=int, default=3)
    parser.add_argument("--out", default="outputs/figures/didi_samples.png")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--topology",
        action="store_true",
        help="不用真实经纬度，退回 spring 拓扑布局（对比用；两者形状完全不同）",
    )
    parser.add_argument("--layout", default="spring", choices=["spring", "kamada_kawai"],
                        help="仅 --topology 时生效")
    parser.add_argument(
        "--pick",
        default="quantile",
        choices=["quantile", "random", "first"],
        help="每个 split 里怎么挑样本：按 GT 长度分位（默认）/ 随机 / 前 N 条",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="只画 corridor 本身（放大看 decision node 与 GT 拐弯）",
    )
    parser.add_argument("--no-cache-layout", action="store_true")
    parser.add_argument(
        "--no-crop",
        dest="crop",
        action="store_false",
        help="地理模式下不要裁到样本范围（看它在整座城市里的位置；细节会糊）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
def load_global_graph(data_dir: Path) -> nx.Graph:
    with open(data_dir / "graph_global.pkl", "rb") as handle:
        payload = pickle.load(handle)
    return payload["graph"]


def geographic_layout(
    graph: nx.Graph, coords_path: Path
) -> Tuple[Dict, Dict[str, Any]]:
    """把经纬度投影成画图坐标（等距圆柱）。

    Returns:
        ``(pos, stats)``；pos 是 ``node -> (x, y)``，单位是"投影后的度"，
        比例已经按 ``cos(lat0)`` 校正，所以图上 1:1 就是真实距离比例。
    """
    coordinates = didi.load_node_coordinates(coords_path)
    graph, stats = didi.attach_coordinates(graph, coordinates, fill_missing=True)
    latitudes = [graph.nodes[n]["y"] for n in graph]
    latitude0 = float(np.mean(latitudes))
    scale = math.cos(math.radians(latitude0))
    pos = {
        node: (graph.nodes[node]["x"] * scale, graph.nodes[node]["y"])
        for node in graph.nodes()
    }
    # 真实尺度：1 度纬度 ≈ 110.57 km
    stats["center"] = (float(np.mean([p[0] for p in pos.values()])),
                       float(np.mean([p[1] for p in pos.values()])))
    stats["wx_km"] = (max(p[0] for p in pos.values()) - min(p[0] for p in pos.values())) * 110.57
    stats["wy_km"] = (max(p[1] for p in pos.values()) - min(p[1] for p in pos.values())) * 110.57
    stats["lat0"] = latitude0
    return pos, stats


def topological_layout(
    graph: nx.Graph, method: str, cache: Path, use_cache: bool
) -> Dict:
    if use_cache and cache.exists():
        try:
            with open(cache, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if raw.get("num_nodes") == graph.number_of_nodes() and raw.get("method") == method:
                return {int(k): tuple(v) for k, v in raw["positions"].items()}
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    print(f"computing {method} layout for {graph.number_of_nodes()} nodes ...", flush=True)
    if method == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=0, iterations=200, k=1.0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": method,
                "num_nodes": graph.number_of_nodes(),
                "positions": {str(node): [float(v[0]), float(v[1])] for node, v in pos.items()},
            },
            handle,
        )
    return pos


def pick_indices(dataset: GraphQueryDataset, count: int, how: str, seed: int) -> List[int]:
    if how == "first":
        return list(range(min(count, len(dataset))))
    if how == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.permutation(len(dataset))[:count].tolist())
    # quantile：按 GT 长度取分位点，保证短/中/长都能看到
    lengths = np.asarray([sample.gt_length for sample in dataset], dtype=float)
    order = np.argsort(lengths)
    picked: List[int] = []
    for position in range(count):
        fraction = (position + 0.5) / count
        index = int(order[min(int(fraction * len(order)), len(order) - 1)])
        if index not in picked:
            picked.append(index)
    return sorted(picked)


def corridor_positions(
    full_graph: nx.Graph, sample, full_pos: Dict
) -> Optional[Dict[int, Tuple[float, float]]]:
    """把 corridor 的局部编号映射回全图坐标。

    ``build_sample_from_observed_path`` 用 ``sorted(labels)`` 把 corridor 节点重编号成
    0..N-1，所以反查就是"把 corridor 的 OSM 节点集合排序"。corridor 节点集合可以只
    用 OD + rho 重新算出来（``corridor_node_set`` 不含 GT，没有泄漏问题）。
    """
    start = int(sample.meta["start_node"])
    goal = int(sample.meta["goal_node"])
    rho = float(sample.meta["rho"])
    keep, _cost = didi.corridor_node_set(full_graph, start, goal, rho)
    ids = sorted(keep)
    if len(ids) != sample.num_nodes:
        return None
    return {index: full_pos[ids[index]] for index in range(len(ids))}


def dijkstra_path(sample) -> Optional[List[int]]:
    try:
        return [
            int(v)
            for v in nx.shortest_path(
                sample.graph, sample.start, sample.goal, weight="weight"
            )
        ]
    except (nx.NetworkXNoPath, nx.NodeNotFound):  # pragma: no cover
        return None


def draw_panel(
    ax,
    sample,
    full_graph: nx.Graph,
    full_pos: Dict,
    title: str,
    zoom: bool,
    geo: bool,
    crop: bool = True,
) -> Dict[str, Any]:
    pos = corridor_positions(full_graph, sample, full_pos)
    local = bool(pos)
    if not local:
        # 兜底：万一编号对不上（不该发生），至少把 corridor 自己画出来
        pos = nx.spring_layout(sample.graph, seed=0, iterations=100)

    gt = list(sample.gt_path)
    shortest = dijkstra_path(sample)
    decision_nodes = list(sample.segments.decision_nodes)

    if not zoom:
        # 全图底图：地理模式下就是真实的成都路网形状
        nx.draw_networkx_edges(
            full_graph, full_pos, ax=ax, edge_color=COLOR_FULL_EDGE, width=0.35
        )
        nx.draw_networkx_nodes(
            full_graph, full_pos, ax=ax, node_size=1.0,
            node_color=COLOR_FULL_NODE, linewidths=0,
        )
        # 高亮：整张图里这一条样本用到的部分
        nx.draw_networkx_edges(
            sample.graph, pos, ax=ax, edge_color=COLOR_CORRIDOR, width=0.9, alpha=0.9
        )

    if shortest is not None:
        nx.draw_networkx_edges(
            sample.graph, pos, edgelist=list(zip(shortest[:-1], shortest[1:])),
            ax=ax, edge_color=COLOR_DIJKSTRA, width=2.0, style="--", alpha=0.9,
        )
    nx.draw_networkx_edges(
        sample.graph, pos, edgelist=list(zip(gt[:-1], gt[1:])),
        ax=ax, edge_color=COLOR_GT, width=2.6, alpha=0.95,
    )
    if decision_nodes:
        nx.draw_networkx_nodes(
            sample.graph, pos, nodelist=decision_nodes, ax=ax,
            node_size=9, node_color=COLOR_DECISION, linewidths=0,
        )
    nx.draw_networkx_nodes(
        sample.graph, pos, nodelist=[sample.start], ax=ax,
        node_size=110, node_color="#2ca02c", node_shape="*", linewidths=0,
    )
    nx.draw_networkx_nodes(
        sample.graph, pos, nodelist=[sample.goal], ax=ax,
        node_size=110, node_color="#d62728", node_shape="*", linewidths=0,
    )

    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    # 地理模式下必须锁等比例，否则 9 个 panel 会各自缩放、看不出真实长度
    ax.set_aspect("equal" if geo else "auto")
    if crop and geo:
        # 裁到该样本自己的范围，否则整城 9×10 km 铺满一格，街道和 GT 都糊成一团
        xs = [pos[node][0] for node in sample.graph.nodes() if node in pos]
        ys = [pos[node][1] for node in sample.graph.nodes() if node in pos]
        if xs and ys:
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            margin = 0.12 * span + 1e-4
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ratio = float(sample.meta.get("gt_cost_ratio", float("nan")))
    return {
        "index": title,
        "order_id": sample.meta.get("order_id"),
        "date": sample.meta.get("date"),
        "junction_len": int(sample.meta.get("junction_len", len(gt))),
        "num_corridor_nodes": int(sample.num_nodes),
        "num_decisions": int(sample.num_decisions),
        "num_candidates": int(sample.num_candidates),
        "gt_cost": float(sample.meta.get("gt_cost", float("nan"))),
        "dijkstra_cost": float(sample.meta.get("dijkstra_cost", float("nan"))),
        "gt_cost_ratio": ratio,
        "null_fraction": float(np.mean(sample.field.candidates.candidate_is_null)),
        "gt_is_simple": len(set(gt)) == len(gt),
        "dijkstra_len": len(shortest) if shortest else None,
        "dijkstra_equals_gt": (shortest == gt) if shortest else None,
        "layout_ok": local,
    }


def main() -> int:
    args = parse_args()
    data_dir = PROJECT_ROOT / args.data
    full_graph = load_global_graph(data_dir)
    geo = not args.topology
    geo_stats: Dict[str, Any] = {}
    if geo:
        coords_path = PROJECT_ROOT / args.coords
        full_pos, geo_stats = geographic_layout(full_graph, coords_path)
        print(
            f"geographic layout from {args.coords}: "
            f"{geo_stats['with_coordinates']}/{geo_stats['nodes']} nodes carry "
            f"lon/lat ({geo_stats['coverage']:.1%}), {geo_stats['filled']} filled "
            f"from neighbours; extent ≈ {geo_stats['wx_km']:.1f} km × "
            f"{geo_stats['wy_km']:.1f} km"
        )
    else:
        full_pos = topological_layout(
            full_graph, args.layout,
            PROJECT_ROOT / "outputs/figures/_didi_full_layout.json",
            not args.no_cache_layout,
        )
        print("topological layout (--topology): shapes are NOT geographic")

    splits = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.pkl"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        splits[name] = GraphQueryDataset.load(path)

    if not splits:
        print("no dataset found")
        return 2

    names = list(splits)
    per_split = int(args.per_split)
    fig, axes = plt.subplots(
        len(names), per_split,
        figsize=(5.0 * per_split, 5.0 * len(names)),
        squeeze=False,
    )
    summary: List[Dict[str, Any]] = []
    for row, name in enumerate(names):
        dataset = splits[name]
        indices = pick_indices(dataset, per_split, args.pick, args.seed + row)
        for column in range(per_split):
            ax = axes[row][column]
            if column >= len(indices):
                ax.set_axis_off()
                continue
            index = indices[column]
            sample = dataset[index]
            ratio = float(sample.meta.get("gt_cost_ratio", float("nan")))
            title = (
                f"{name} #{index} | {sample.meta.get('date')}\n"
                f"junc={sample.meta.get('junction_len')} "
                f"dec={sample.num_decisions} cand={sample.num_candidates} "
                f"corridor={sample.num_nodes}\n"
                f"GT/Dijkstra cost = {ratio:.3f}"
            )
            info = draw_panel(
                ax, sample, full_graph, full_pos, title, args.zoom, geo, args.crop
            )
            info["split"] = name
            info["index"] = index
            summary.append(info)

    legend = [
        plt.Line2D([], [], color=COLOR_GT, lw=2.6, label="GT = 真实司机历史路径"),
        plt.Line2D([], [], color=COLOR_DIJKSTRA, lw=2.0, ls="--",
                   label="Dijkstra 最短路（同一 OD）"),
        plt.Line2D([], [], color=COLOR_CORRIDOR, lw=1.6, label="OD corridor（模型看到的图）"),
        plt.Line2D([], [], color=COLOR_FULL_EDGE, lw=1.6,
                   label="整张成都路网（真实经纬度底图）" if geo else "整张成都路网（拓扑布局，非地理）"),
        plt.Line2D([], [], marker="*", color="w", markerfacecolor="#2ca02c",
                   markersize=12, label="start"),
        plt.Line2D([], [], marker="*", color="w", markerfacecolor="#d62728",
                   markersize=12, label="goal"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=COLOR_DECISION,
                   markersize=5, label="decision node"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False)
    suptitle = "DiDi 成都真实数据样本（GT = 真实车辆历史路径，不是最短路）"
    if geo:
        suptitle += (
            "\n底图 = 真实经纬度（OSMnx epsg:4326，"
            f"{geo_stats['coverage']:.0%} 节点有坐标）；范围 ≈ "
            f"{geo_stats['wx_km']:.1f} km × {geo_stats['wy_km']:.1f} km"
        )
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"saved -> {out_path}")

    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
    print(f"saved -> {summary_path}")

    print(f"\n{'split':<6}{'idx':>6}{'date':>10}{'junc':>6}{'dec':>6}{'cand':>7}"
          f"{'corr':>6}{'GTratio':>9}{'GT==Dij':>9}")
    for row in summary:
        print(
            f"{row['split']:<6}{row['index']:>6}{str(row['date']):>10}"
            f"{row['junction_len']:>6}{row['num_decisions']:>6}"
            f"{row['num_candidates']:>7}{row['num_corridor_nodes']:>6}"
            f"{row['gt_cost_ratio']:>9.3f}{str(row['dijkstra_equals_gt']):>9}"
        )
    equal = sum(1 for row in summary if row["dijkstra_equals_gt"])
    print(f"\n其中 GT == Dijkstra 的样本: {equal}/{len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
