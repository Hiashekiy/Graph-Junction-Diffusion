"""从数据集里抽一条样本，渲染一张（偏展示用的）城市街道图。

不是评测图，是**展示图**：深色底 + 路网 + 高亮 GT 路径 + S/G + 比例尺。

    深色细线    路网（真实经纬度投影，米制）
    浅色小圈    路口节点
    亮黄绿粗线  GT —— 该样本真实司机走过的历史路径
    绿点 / 橙点 Start / Goal
    左下角      比例尺（米）

取景两种（``--view``）::

    route  （默认）裁到 GT 路径的范围。城市路网 9.2km x 9.9km 接近正方，
                    塞进 16:9 画布会左右空掉一大片，所以默认跟着路径走。
    city            整个路网，此时画布比例**自动**按数据算，保证铺满不留白。

坐标：``data/didi/raw/chengdu/ChengDu.pkl`` 是 OSMnx 导出的 MultiDiGraph
（crs=epsg:4326，节点自带 x=经度 / y=纬度），只覆盖 2891 个节点里的 2780 个；
缺的 111 个用 ``graph_global.pkl`` 的邻居坐标迭代补齐（覆盖率 100%）。

投影用等距圆柱：``x = (lon-lon0)·111320·cos(lat0)``、``y = (lat-lat0)·110540``，
单位是**米**，所以比例尺可以直接按米画。9km × 10km 范围内形变可忽略。

注意 ``GraphSample.gt_path`` 用的是 **corridor 内的局部索引**，不是全局 OSM id，
必须经 ``sample.meta["local_to_global"]`` 映射后才能落在全城路网上。

用法::

    python tools/render_street_map.py                       # 默认：val 里最长那条，裁到路径
    python tools/render_street_map.py --view city           # 整城
    python tools/render_street_map.py --data data/didi/graph/chengdu_long/test_1000.pkl --index 7
    python tools/render_street_map.py --corridor --caption  # 叠加 corridor + 样本信息

换城市时序三件套都要跟着换（默认都是成都）：::

    python tools/render_street_map.py --data data/didi/graph/xian/test_1000.pkl \n        --graph data/didi/graph/xian/graph_global.pkl \n        --coords data/didi/raw/xian/XiAn.pkl
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import didi_dataset as didi  # noqa: E402
from src.data.dataset import GraphQueryDataset  # noqa: E402

# ---------------------------------------------------------------------------
# 配色：深色底 + 低对比路网 + 高饱和高亮路径
# ---------------------------------------------------------------------------
BG = "#070a11"
ROAD = "#31415a"
NODE_FACE = "#0b1220"
NODE_EDGE = "#a9bdd2"
CORRIDOR = "#3f6ea8"
ROUTE = "#c8f22e"
START_COLOR = "#35d07f"
GOAL_COLOR = "#ffb020"
TEXT = "#e8eef6"
MUTED = "#8fa3ba"
#: 备选路径的配色（第 0 个给"最短路"，其余给 k-最短路里挑出来的）
ALT_COLORS = ["#38bdf8", "#f472b6", "#fbbf24", "#a78bfa", "#34d399", "#fb7185"]
DASH = (0, (5, 2))

DEFAULT_DATA = "data/didi/graph/chengdu_large_finetune/val.pkl"
DEFAULT_GRAPH = "data/didi/graph/chengdu_large_finetune/graph_global.pkl"
DEFAULT_COORDS = "data/didi/raw/chengdu/ChengDu.pkl"

M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="render one dataset sample as a dark street map")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--index", type=int, default=0, help="样本下标（--pick index 时生效）")
    parser.add_argument("--pick",
                        choices=("index", "longest", "shortest", "zoom", "diverse"),
                        default="zoom",
                        help="diverse（默认）= 挑一个真有多条不同路径的 OD；zoom = 挑包围盒最接近 "
                             "--target-span-m 的样本；longest / shortest / index")
    parser.add_argument("--target-span-m", type=float, default=2600.0,
                        help="--pick zoom 的目标视野跨度（米）")
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--coords", default=DEFAULT_COORDS)
    parser.add_argument("--out", default="outputs/figures/street_map.png")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--width", type=float, default=15.0, help="画布宽度（英寸）")
    parser.add_argument("--aspect", type=float, default=0.0,
                        help="强行指定画布宽高比；0（默认）= 跟着取景走，四边不留白。"
                             "给了值就会在短边补空白（只扩不裁）")
    parser.add_argument("--view", choices=("route", "city"), default="route",
                        help="route（默认）= 只画样本周边的街区；city = 整个路网")
    parser.add_argument("--margin", type=float, default=None,
                        help="取景留白比例；默认 route=0.22（多带一圈周边街道）/ city=0.03")
    parser.add_argument("--title", default=None)
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--corridor", action="store_true",
                        help="额外把模型看到的 corridor 画亮一档")
    #: scatter 的 s 是**面积**(pt^2)。d = sqrt(4s/pi) pt；200dpi 下 s=16 约 12px 直径，
    #: 和参考图里"小圆环"的观感差不多。
    parser.add_argument("--node-size", type=float, default=16.0)
    parser.add_argument("--road-width", type=float, default=1.05)
    parser.add_argument("--route-width", type=float, default=3.2)
    parser.add_argument("--scale-bar-m", type=float, default=100.0)
    parser.add_argument("--paths", type=int, default=5,
                        help="一共画几条 S->G 路径（含 GT）。1 = 只画 GT")
    parser.add_argument("--overlap", type=float, default=0.90,
                        help="备选路与已有路径的节点重合度上限；超过就跳过，" 
                             "否则 k-最短路出来的几条几乎叠在一起，看着还是一条")
    return parser.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ---------------------------------------------------------------------------
def load_network(graph_path: Path, coords_path: Path):
    """返回 (graph, node -> (x_m, y_m))，坐标是米制局部等距投影。"""
    with open(graph_path, "rb") as handle:
        payload = pickle.load(handle)
    graph: nx.Graph = payload["graph"] if isinstance(payload, dict) else payload

    lonlat, stats = didi.load_node_coordinates_filled(coords_path, graph_path)
    print(f"[map] coords: {stats['with_coordinates']}/{stats['total_nodes']} real + "
          f"{stats['filled']} filled by neighbours -> coverage {stats['coverage']:.3f}")

    known = {n: lonlat[n] for n in graph.nodes() if n in lonlat}
    lat0 = sum(v[1] for v in known.values()) / len(known)
    lon0 = sum(v[0] for v in known.values()) / len(known)
    kx = M_PER_DEG_LON * math.cos(math.radians(lat0))
    xy = {n: ((lon - lon0) * kx, (lat - lat0) * M_PER_DEG_LAT) for n, (lon, lat) in known.items()}
    missing = graph.number_of_nodes() - len(xy)
    if missing:
        print(f"[map] WARNING: {missing} graph node(s) have no coordinates and are skipped")
    return graph, xy


def sample_route(sample, xy: Dict[Any, Tuple[float, float]]) -> List[Tuple[float, float]]:
    """把 gt_path（corridor 局部索引）映射回全局节点，再取米制坐标。"""
    l2g = sample.meta.get("local_to_global")
    if l2g is None:
        raise KeyError(
            "sample.meta['local_to_global'] is missing: gt_path uses corridor-local "
            "indices and cannot be placed on the city map without it."
        )
    nodes = [l2g[i] for i in sample.gt_path]
    unknown = [n for n in nodes if n not in xy]
    if unknown:
        raise KeyError(f"{len(unknown)} gt_path node(s) have no coordinates, e.g. {unknown[:3]}")
    return [xy[n] for n in nodes]


def corridor_segments(sample, xy) -> List[List[Tuple[float, float]]]:
    """corridor 的物理边（同样要经 local_to_global 映射）。"""
    l2g = sample.meta["local_to_global"]
    edge_index = getattr(sample.segments, "edge_index", None)
    if not edge_index:
        return []
    out = []
    for u, v in edge_index[::2]:            # 物理边取每个头的第一份 (u, v)
        gu, gv = l2g[u], l2g[v]
        if gu in xy and gv in xy:
            out.append([xy[gu], xy[gv]])
    return out


def compute_view(args, xy, route) -> Tuple[float, float, float, float]:
    """取景框。只**扩**不裁，所以 GT 永远不会被切掉。

    ``--aspect`` 默认 0 = **跟着取景走**。这一点很关键：城市路网 9.2km x 9.9km
    接近正方，某条 GT 的包围盒可能是竖的（实测那条最长的 103 跳是 7250 x 7897），
    强行套 16:9 会在左右各补出 3.7km 的纯黑，图就"空"了。
    """
    pts = list(xy.values()) if args.view == "city" else route
    margin = args.margin if args.margin is not None else (0.03 if args.view == "city" else 0.22)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = x1 - x0, y1 - y0
    x0 -= dx * margin
    x1 += dx * margin
    y0 -= dy * margin
    y1 += dy * margin

    aspect = args.aspect if args.aspect > 0 else (1.6 if args.view == "route" else 0.0)
    if aspect > 0:
        dx, dy = x1 - x0, y1 - y0
        if dx / dy < aspect:                 # 太窄 -> 左右补
            pad = (dy * aspect - dx) / 2.0
            x0, x1 = x0 - pad, x1 + pad
        else:                                 # 太高 -> 上下补
            pad = (dx / aspect - dy) / 2.0
            y0, y1 = y0 - pad, y1 + pad
    return x0, x1, y0, y1


def path_cost(graph, nodes, weight: str = "weight") -> float:
    """路径的米数（graph 的 edge weight 就是道路长度）。"""
    return float(sum(graph[u][v].get(weight, 1.0) for u, v in zip(nodes[:-1], nodes[1:])))


def overlap_ratio(a, b) -> float:
    """两条路径的节点重合度（相对较短的那条）。"""
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def route_diversity(graph, gt_nodes, want: int, overlap_max: float,
                    cap: int = 40) -> List[List]:
    """GT + 最多 want 条**节点重合度不超过 overlap_max** 的备选路。

    直接取 ``nx.shortest_simple_paths`` 的前 k 条是不够的：Yen 逐条只差一条边，
    实测某个 OD 前 300 条的成本全落在 3098~3150 m（差 1.7%），画出来完全叠在一起。
    所以必须按重合度过滤，否则"多画几条"只是把同一条线描三遍。
    """
    kept = [list(gt_nodes)]
    tried = 0
    try:
        for nodes in nx.shortest_simple_paths(graph, gt_nodes[0], gt_nodes[-1],
                                              weight="weight"):
            if tried >= cap or len(kept) >= want:
                break
            tried += 1
            if any(overlap_ratio(nodes, other) > overlap_max for other in kept):
                continue
            kept.append(list(nodes))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return kept


def compute_routes(graph, gt_nodes, total: int, overlap_max: float) -> List[dict]:
    """把 route_diversity 的结果配上颜色 / 线型。"""
    kept = route_diversity(graph, gt_nodes, max(total, 1), overlap_max)
    if not kept:
        kept = [list(gt_nodes)]
    routes = []
    for index, nodes in enumerate(kept):
        if index == 0:
            routes.append({"kind": "GT（观测）", "nodes": nodes,
                           "cost": path_cost(graph, nodes), "color": ROUTE,
                           "style": "-", "width": 3.4})
        else:
            routes.append({
                "kind": "最短路" if index == 1 else f"备选 {index - 1}",
                "nodes": nodes, "cost": path_cost(graph, nodes),
                "color": ALT_COLORS[(index - 1) % len(ALT_COLORS)],
                "style": DASH, "width": 2.5 if index == 1 else 2.2,
            })
    return routes


def pick_diverse(dataset, graph, want: int, overlap_max: float) -> int:
    """挑一个**真的存在多条不同路径**的 OD。

    固定用同一个样本时，"多画几条路径"经常落空 —— 实测 val[409] 只有 2 条真正不同
    （3.51km 的 GT 和 3.10km 的直行），第 3、4 条只差几十米，必然叠在一起。
    所以按"能凑出几条互不重合的路"来挑样本，同分时优先 GT 与最短路差得多的
    （那个才是有故事可讲的 OD：司机没有走最短路）。
    """
    best = None
    for i in range(len(dataset)):
        l2g = dataset[i].meta.get("local_to_global")
        if not l2g:
            continue
        gt_nodes = [l2g[j] for j in dataset[i].gt_path]
        kept = route_diversity(graph, gt_nodes, want, overlap_max)
        if len(kept) < 2:
            continue
        gt_cost = path_cost(graph, gt_nodes)
        alt_cost = min(path_cost(graph, nodes) for nodes in kept[1:])
        gain = (gt_cost - alt_cost) / max(gt_cost, 1.0)
        score = (len(kept), round(gain, 4))
        if best is None or score > best[0]:
            best = (score, i, len(kept), gt_cost, alt_cost)
    if best is None:
        print("[map] --pick diverse: no sample offers >1 distinct route; falling back to longest")
        return max(range(len(dataset)), key=lambda i: dataset[i].gt_length)
    _, index, count, gt_cost, alt_cost = best
    print(f"[map] --pick diverse: sample {index}  {count} distinct route(s)  "
          f"GT {gt_cost:.0f} m vs best {alt_cost:.0f} m  (GT/best = {gt_cost / alt_cost:.3f})")
    return index


def draw_route_legend(ax, view, routes) -> None:
    x0, x1, y0, y1 = view
    dx, dy = x1 - x0, y1 - y0
    row = dy * 0.056
    pad = dy * 0.022
    swatch = dx * 0.042
    left = x1 - dx * 0.345
    top = y1 - dy * 0.030
    height = row * len(routes) + pad * 2.0
    width = dx * 0.315

    # 底板：地图上的街线会从图例文字里穿过去，不铺底没法看
    ax.add_patch(FancyBboxPatch(
        (left - pad * 1.2, top - height), width, height,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=BG, edgecolor="#3d4d63", linewidth=1.1,
        alpha=0.92, zorder=39))

    y = top - pad - row * 0.5
    for item in routes:
        ax.plot([left, left + swatch], [y, y], color=item["color"],
                linewidth=item["width"], linestyle=item["style"], zorder=40,
                solid_capstyle="round")
        ax.text(left + swatch + dx * 0.012, y,
                f'{item["kind"]}   {item["cost"] / 1000:.2f} km',
                color=TEXT, fontsize=12.5, ha="left", va="center", zorder=40)
        y -= row


def pick_by_span(dataset, xy, target_w: float, target_aspect: float) -> int:
    """挑一条 GT 包围盒最接近**目标画框**的样本。

    只按"最大跨度"挑是不够的：实测挑出过 2588m x 500m 的轨迹，裁出来是一条
    3.2:1 的长条，两边全是空街。所以宽和高都要匹配：

        score = |ln(w / target_w)| + |ln(h / target_h)|,   target_h = target_w / aspect

    这样选中的轨迹会大致铺满画框，四周再各带一圈周边街道。

    为什么需要它：``--pick longest`` 会选中那条 103 跳、横跨 7.2km x 7.9km 的轨迹，
    裁到它等于把整座城又画了一遍 —— 街区细节全糊在一起。
    """
    target_h = target_w / max(target_aspect, 0.1)
    best = None
    for i in range(len(dataset)):
        route = sample_route(dataset[i], xy)
        xs = [p[0] for p in route]
        ys = [p[1] for p in route]
        w = max(max(xs) - min(xs), 1.0)
        h = max(max(ys) - min(ys), 1.0)
        score = abs(math.log(w / target_w)) + abs(math.log(h / target_h))
        if best is None or score < best[0]:
            best = (score, i, w, h)
    _, index, w, h = best
    print(f"[map] --pick zoom: sample {index}  route bbox {w:.0f} m x {h:.0f} m  "
          f"(target frame {target_w:.0f} m x {target_h:.0f} m)")
    return index


# ---------------------------------------------------------------------------
def draw_scale_bar(ax, view, length_m: float, label: str) -> None:
    x0, x1, y0, y1 = view
    dx, dy = x1 - x0, y1 - y0
    px = x0 + dx * 0.045
    py = y0 + dy * 0.055
    tick = dy * 0.016
    ax.plot([px, px + length_m], [py, py], color=TEXT, linewidth=1.8,
            solid_capstyle="butt", zorder=40)
    for x in (px, px + length_m):
        ax.plot([x, x], [py - tick, py + tick], color=TEXT, linewidth=1.8,
                solid_capstyle="butt", zorder=40)
    ax.text(px, py + tick * 1.5, label, color=TEXT, fontsize=13,
            ha="left", va="bottom", zorder=40)


def draw_endpoints(ax, view, start_xy, goal_xy) -> None:
    x0, x1, y0, y1 = view
    dx, dy = x1 - x0, y1 - y0
    for point, color, text, sign in (
        (start_xy, START_COLOR, "S", -1.0),
        (goal_xy, GOAL_COLOR, "G", 1.0),
    ):
        ax.scatter([point[0]], [point[1]], s=190, facecolor=color,
                   edgecolor="#ffffff", linewidths=1.9, zorder=30)
        ax.text(point[0] + dx * 0.011, point[1] + sign * dy * 0.022, text,
                color=color, fontsize=17, weight="bold", ha="left", va="center",
                zorder=31)


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    matplotlib.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })

    graph_path, coords_path = resolve(args.graph), resolve(args.coords)
    data_path = resolve(args.data)
    graph, xy = load_network(graph_path, coords_path)

    dataset = GraphQueryDataset.load(data_path)
    if args.pick == "longest":
        index = max(range(len(dataset)), key=lambda i: dataset[i].gt_length)
    elif args.pick == "shortest":
        index = min(range(len(dataset)), key=lambda i: dataset[i].gt_length)
    elif args.pick == "zoom":
        index = pick_by_span(dataset, xy, args.target_span_m,
                             args.aspect if args.aspect > 0 else 1.6)
    elif args.pick == "diverse":
        index = pick_diverse(dataset, graph, max(args.paths, 2), args.overlap)
    else:
        index = args.index
    if not 0 <= index < len(dataset):
        raise SystemExit(f"--index {index} out of range (dataset has {len(dataset)} samples)")
    sample = dataset[index]

    route = sample_route(sample, xy)
    view = compute_view(args, xy, route)
    x0, x1, y0, y1 = view
    dx, dy = x1 - x0, y1 - y0

    # 画布比例 == 取景比例，并让坐标区铺满整张画布 -> 四边不留白
    fig = plt.figure(figsize=(args.width, args.width * dy / dx))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(BG)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")

    # (1) 路网
    segments = [[xy[u], xy[v]] for u, v in graph.edges() if u in xy and v in xy]
    ax.add_collection(LineCollection(segments, colors=ROAD, linewidths=args.road_width,
                                     alpha=0.95, zorder=1))

    # (2) 模型看到的 corridor（可选）
    if args.corridor:
        seg = corridor_segments(sample, xy)
        if seg:
            ax.add_collection(LineCollection(seg, colors=CORRIDOR, linewidths=1.5,
                                             alpha=0.85, zorder=2))

    # (3) 节点小圆环
    xs = [xy[n][0] for n in graph.nodes() if n in xy]
    ys = [xy[n][1] for n in graph.nodes() if n in xy]
    ax.scatter(xs, ys, s=args.node_size, facecolors=NODE_FACE, edgecolors=NODE_EDGE,
               linewidths=0.9, alpha=0.95, zorder=3)

    # (4) 多条 S->G 路径。倒序画，GT 落在最上层；每条先铺一层光晕再压实线。
    gt_nodes = [sample.meta["local_to_global"][i] for i in sample.gt_path]
    routes = compute_routes(graph, gt_nodes, args.paths, args.overlap)
    for rank, item in enumerate(reversed(routes)):
        pts = [xy[n] for n in item["nodes"] if n in xy]
        if len(pts) < 2:
            continue
        px = [q[0] for q in pts]
        py = [q[1] for q in pts]
        z = 4.0 + rank * 0.15
        scale = item["width"] / max(args.route_width, 0.1)
        ax.plot(px, py, color=item["color"], linewidth=item["width"] * 3.2,
                alpha=0.12, solid_capstyle="round", solid_joinstyle="round", zorder=z)
        ax.plot(px, py, color=item["color"], linewidth=item["width"],
                linestyle=item["style"], alpha=0.98, solid_capstyle="round",
                solid_joinstyle="round", zorder=z + 0.05)

    # (5) S / G
    # S/G 用 GT 的两端（所有备选路都共享同一对 OD）
    draw_endpoints(ax, view, route[0], route[-1])
    draw_route_legend(ax, view, routes)

    # (6) 比例尺 / 标题 / 说明
    draw_scale_bar(ax, view, args.scale_bar_m, f"{args.scale_bar_m:g} m")
    if args.title:
        ax.text(x0 + dx * 0.045, y1 - dy * 0.045, args.title, color=TEXT,
                fontsize=21, weight="bold", ha="left", va="top", zorder=40)
    if args.caption:
        ax.text(x0 + dx * 0.045, y0 + dy * 0.115,
                f"{data_path.name}[{index}]    gt_length={sample.gt_length}    "
                f"corridor nodes={sample.num_nodes}    "
                f"date={sample.meta.get('date', '?')}",
                color="#a9bdd2", fontsize=13, ha="left", va="bottom", zorder=40)

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[map] wrote {out}")
    print(f"[map] view      : {args.view}  {dx:.0f} m x {dy:.0f} m  ratio {dx/dy:.2f}"
          f"  (canvas {args.width:.1f} x {args.width * dy / dx:.1f} in)")
    print(f"[map] sample    : {data_path.name}[{index}]  ({args.pick})")
    print(f"[map] gt_length : {sample.gt_length} hops   corridor nodes={sample.num_nodes}")
    print(f"[map] S={sample.meta.get('start_node')}  G={sample.meta.get('goal_node')}")
    for item in routes:
        print(f"[map]   {item['kind']:<10} {item['cost']:8.0f} m   "
              f"{len(item['nodes'])} nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
