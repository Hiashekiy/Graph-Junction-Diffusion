"""把 ``data/public_graphs/`` 里的公开图数据画出来（规模 + 全貌 + 局部结构 + 分布）。

用法::

    E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_public_graphs.py
    E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_public_graphs.py --only rome99 luxembourg
    E:/CondaEnvData/envs/GGMPC/python.exe tools/visualize_public_graphs.py --out-dir outputs/figures --dpi 150

输入是 ``scripts/download_graph_datasets.py`` 下载的原始文件，直接读压缩包，不落中间文件：

* DIMACS9 ``*.gr`` / ``*.gr.gz``：``c`` 注释 + ``p sp N M`` + ``a u v w``（有向，w = 弧长）；
  配套 ``*.co`` / ``*.co.gz`` 给出 ``v <id> <lon> <lat>``（1e-6 度），有它才能画成地图；
* DIMACS10 ``*.graph.bz2``：``%`` 注释 + ``N M`` + M 行边表 + N 行坐标。

输出两张图（默认放 ``outputs/figures/``）：

* ``public_graphs_map.png``：每份数据一行。左 = 整张图（有坐标就按经纬度画，等比例并做
  ``cos(lat)`` 修正，否则退化成力导向布局并标注）；右 = 全图最高度数节点周围 2 跳的放大图，
  节点按度数上色、大小随度数增大，左图里的红框就是这块放大范围。
* ``public_graphs_stats.png``：度数分布（log-log）+ 边权分布 + 规模统计表。

注意：这里画的是"数据本身长什么样"，和模型预测路径无关；后者看
``tools/visualize_paths.py``（项目自己的 Controlled Junction 数据）或 dashboard。
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "public_graphs"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "figures"


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
@dataclass
class GraphData:
    name: str
    path: Path
    num_nodes: int
    edges: np.ndarray                      # (m, 2) int32，0-based
    weights: Optional[np.ndarray]          # (m,) float64
    coords: Optional[np.ndarray]           # (n, 2) float64
    coord_source: str
    kind: str
    seconds: float = 0.0

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])


def _open_text(path: Path):
    """按后缀透明地打开 .gz / .bz2 / 纯文本（DIMACS 里都是 ASCII，用 latin-1 最稳）。"""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="latin-1")
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="latin-1")
    return open(path, "rt", encoding="latin-1")


def read_dimacs9_gr(path: Path) -> Tuple[int, np.ndarray, np.ndarray]:
    """``p sp N M`` + ``a u v w`` -> (N, edges(0-based), weights)。"""
    num_nodes = 0
    sources: List[int] = []
    targets: List[int] = []
    weights: List[float] = []
    with _open_text(path) as handle:
        for line in handle:
            head = line[:1]
            if head == "a":
                parts = line.split()
                sources.append(int(parts[1]) - 1)
                targets.append(int(parts[2]) - 1)
                weights.append(float(parts[3]))
            elif head == "p":
                parts = line.split()
                num_nodes = int(parts[2])
    edges = np.empty((len(sources), 2), dtype=np.int32)
    edges[:, 0] = sources
    edges[:, 1] = targets
    return num_nodes, edges, np.asarray(weights, dtype=np.float64)


def read_dimacs9_co(path: Path) -> np.ndarray:
    """``p aux sp co N`` + ``v id lon lat``（1e-6 度）-> (N, 2) 经纬度。"""
    ids: List[int] = []
    xs: List[float] = []
    ys: List[float] = []
    with _open_text(path) as handle:
        for line in handle:
            if line[:1] == "v":
                parts = line.split()
                ids.append(int(parts[1]) - 1)
                xs.append(float(parts[2]) * 1e-6)
                ys.append(float(parts[3]) * 1e-6)
    coords = np.empty((len(ids), 2), dtype=np.float64)
    coords[ids, 0] = xs
    coords[ids, 1] = ys
    return coords


def read_dimacs10_graph(path: Path) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
    """``%`` 注释 + ``N M`` + 数据行。

    DIMACS10 归档里有两个变体，按数据行数自动判定：

    * **邻接表**（streets 归档，如 luxembourg）：``N`` 行，第 i 行是点 i 的邻居；
    * **边表**：``M`` 行 ``u v``，后面可能再跟 ``N`` 行坐标。

    返回 ``(N, edges(0-based), coords 或 None)``。
    """
    with _open_text(path) as handle:
        header: Optional[List[str]] = None
        for line in handle:
            if line.strip() and not line.startswith("%"):
                header = line.split()
                break
        if not header:
            raise ValueError(f"{path.name}: 没有数据行")
        num_nodes, num_edges = int(header[0]), int(header[1])
        rows = [line.split() for line in handle if line.strip() and not line.startswith("%")]

    if len(rows) == num_nodes:                      # 邻接表
        sources: List[int] = []
        targets: List[int] = []
        for index, neighbours in enumerate(rows):
            for neighbour in neighbours:
                sources.append(index)
                targets.append(int(neighbour) - 1)
        edges = np.column_stack([sources, targets]).astype(np.int32)
        edges = np.unique(np.sort(edges, axis=1), axis=0)   # 无向，两个方向都列了
        return num_nodes, edges, None

    edges = np.array([[int(row[0]), int(row[1])] for row in rows[:num_edges]], dtype=np.int32)
    coords = None
    if len(rows) >= num_edges + num_nodes:          # 边表后面跟坐标
        coords = np.array(
            [[float(row[0]), float(row[1])] for row in rows[num_edges : num_edges + num_nodes]],
            dtype=np.float64,
        )
    if int(edges.min()) > 0:                        # 有的归档 0-based、有的 1-based
        edges -= 1
    return num_nodes, edges, coords


def read_dimacs10_xyz(path: Path) -> np.ndarray:
    """streets 归档的坐标文件：每个节点一行 ``x y z``（UTM 米）。"""
    rows: List[Tuple[float, float]] = []
    with _open_text(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
    return np.asarray(rows, dtype=np.float64)


def load_dataset(path: Path) -> GraphData:
    started = time.time()
    name = path.name
    for suffix in (".gr.gz", ".gr", ".graph.bz2", ".graph"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    is_dimacs9 = path.name.endswith(".gr") or path.name.endswith(".gr.gz")
    if is_dimacs9:
        num_nodes, edges, weights = read_dimacs9_gr(path)
        stem = path.name[: -len(".gr.gz")] if path.name.endswith(".gr.gz") else path.name[: -len(".gr")]
        co_path = None
        for candidate in (path.with_name(stem + ".co.gz"), path.with_name(stem + ".co")):
            if candidate.exists():
                co_path = candidate
                break
        coords = read_dimacs9_co(co_path) if co_path else None
        coord_source = co_path.name if co_path else "无（力导向布局）"
        kind = "DIMACS9 shortest-path instance"
    else:
        num_nodes, edges, coords = read_dimacs10_graph(path)
        weights = None
        stem = path.name[: -len(".graph.bz2")] if path.name.endswith(".graph.bz2") else path.name[: -len(".graph")]
        xyz_path = None
        for candidate in (path.with_name(stem + ".xyz.bz2"), path.with_name(stem + ".xyz")):
            if candidate.exists():
                xyz_path = candidate
                break
        if coords is None and xyz_path is not None:
            coords = read_dimacs10_xyz(xyz_path)
        coord_source = xyz_path.name if (coords is not None and xyz_path is not None) else (
            path.name if coords is not None else "无（力导向布局）"
        )
        kind = "DIMACS10 street network"

    return GraphData(
        name=name,
        path=path,
        num_nodes=int(num_nodes),
        edges=edges,
        weights=weights,
        coords=coords,
        coord_source=coord_source,
        kind=kind,
        seconds=time.time() - started,
    )


# --------------------------------------------------------------------------- #
# 结构统计
# --------------------------------------------------------------------------- #
def degrees(num_nodes: int, edges: np.ndarray) -> np.ndarray:
    """无向化之后的度（DIMACS9 是有向弧表，这里按"路网"看）。"""
    flat = np.concatenate([edges[:, 0], edges[:, 1]])
    return np.bincount(flat, minlength=num_nodes).astype(np.int64)


def weak_components(num_nodes: int, edges: np.ndarray) -> int:
    """并查集数弱连通分量（只数数量，不需要 label）。"""
    parent = np.arange(num_nodes, dtype=np.int64)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:          # 路径压缩
            parent[node], node = root, parent[node]
        return root

    for source, target in edges:
        root_a, root_b = find(int(source)), find(int(target))
        if root_a != root_b:
            parent[root_a] = root_b
    roots = np.array([find(node) for node in range(num_nodes)])
    return int(np.unique(roots).size)


def adjacency(num_nodes: int, edges: np.ndarray):
    """CSR 邻接表（无向），供 BFS 取局部子图。"""
    source = np.concatenate([edges[:, 0], edges[:, 1]])
    target = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(source, kind="stable")
    source, target = source[order], target[order]
    indptr = np.searchsorted(source, np.arange(num_nodes + 1))
    return target, indptr


def k_hop_nodes(target: np.ndarray, indptr: np.ndarray, start: int, hops: int, cap: int) -> np.ndarray:
    visited = {int(start)}
    frontier = [int(start)]
    for _ in range(hops):
        nxt: List[int] = []
        for node in frontier:
            for neighbour in target[indptr[node] : indptr[node + 1]]:
                neighbour = int(neighbour)
                if neighbour not in visited:
                    visited.add(neighbour)
                    nxt.append(neighbour)
        frontier = nxt
        if len(visited) >= cap:
            break
    return np.fromiter(sorted(visited), dtype=np.int32, count=len(visited))


# --------------------------------------------------------------------------- #
# 画
# --------------------------------------------------------------------------- #
def project(coords: np.ndarray) -> np.ndarray:
    """经纬度 -> 等距离近似（x 乘 cos(平均纬度)），否则北美/欧洲地图会被横向拉长。"""
    if coords is None:
        return None
    out = np.array(coords, dtype=np.float64, copy=True)
    if np.abs(out[:, 0]).max() <= 360 and np.abs(out[:, 1]).max() <= 90:
        out[:, 0] *= np.cos(np.deg2rad(float(np.mean(out[:, 1]))))
    return out


def force_layout(num_nodes: int, edges: np.ndarray, seed: int = 0) -> np.ndarray:
    """没有坐标文件时（rome99）用 networkx 的 spring layout 兜底。"""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(map(tuple, edges.tolist()))
    pos = nx.spring_layout(graph, seed=seed, iterations=80)
    return np.asarray([pos[node] for node in range(num_nodes)], dtype=np.float64)


def draw_edges(ax, points: np.ndarray, edges: np.ndarray, linewidth: float, color: str, alpha: float, zorder: int = 1):
    from matplotlib.collections import LineCollection

    segments = points[edges]
    collection = LineCollection(segments, linewidths=linewidth, colors=color, alpha=alpha, zorder=zorder)
    ax.add_collection(collection)
    return collection


def draw_map_panel(ax, data: GraphData, points: np.ndarray, zoom_box: Optional[Tuple[float, float, float, float]],
                   title: str, linewidth: float):
    draw_edges(ax, points, data.edges, linewidth=linewidth, color="#4a5568", alpha=0.8)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c9ced8")
    ax.set_title(title, fontsize=10.5, pad=6)
    if zoom_box is not None:
        from matplotlib.patches import Rectangle

        ax.add_patch(
            Rectangle(
                (zoom_box[0], zoom_box[1]), zoom_box[2], zoom_box[3],
                fill=False, edgecolor="#e53e3e", linewidth=1.4, zorder=5,
            )
        )


def pick_hub(
    num_nodes: int,
    edges: np.ndarray,
    degree: np.ndarray,
    hops: int,
    cap: int,
    candidates: int = 25,
) -> Tuple[int, np.ndarray]:
    """挑一个"最热闹的路口"来放大。

    只看度数最高的点，在近似树状的路网（如 Luxembourg，平均度 2.09）上会挑到一条长链中间，
    放大出来只有十几个点。所以在度数最高的前 ``candidates`` 个点里，取 k 跳邻域最大的那个。
    """
    target, indptr = adjacency(num_nodes, edges)
    order = np.argsort(degree)[::-1][:candidates]
    best_node = int(order[0])
    best_nodes = k_hop_nodes(target, indptr, best_node, hops, cap)
    for node in order[1:]:
        nodes = k_hop_nodes(target, indptr, int(node), hops, cap)
        if len(nodes) > len(best_nodes):
            best_node, best_nodes = int(node), nodes
    return best_node, best_nodes


def draw_zoom_panel(ax, data: GraphData, points: np.ndarray, nodes: np.ndarray, hub: int, degree: np.ndarray, title: str):
    node_set = set(int(node) for node in nodes)
    mask = np.isin(data.edges[:, 0], nodes) & np.isin(data.edges[:, 1], nodes)
    local_edges = data.edges[mask]
    draw_edges(ax, points, local_edges, linewidth=1.5, color="#9aa5b1", alpha=0.9, zorder=1)

    local_degree = degree[nodes]
    sizes = 12.0 + 3.0 * np.sqrt(local_degree.astype(np.float64))
    scatter = ax.scatter(
        points[nodes, 0], points[nodes, 1],
        s=sizes, c=local_degree, cmap="viridis", linewidths=0.4,
        edgecolors="#1a202c", zorder=3,
    )
    hub_xy = points[hub]
    ax.scatter([hub_xy[0]], [hub_xy[1]], s=140, facecolors="none", edgecolors="#e53e3e", linewidths=1.6, zorder=4)
    ax.annotate(
        f"中心点 #{hub}（度 {int(degree[hub])}）",
        xy=(hub_xy[0], hub_xy[1]), xytext=(8, 8), textcoords="offset points",
        fontsize=8, color="#2d3748",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e0", lw=0.6, alpha=0.85),
        zorder=6,
    )
    # 正方形取景窗：等比例又不会因为邻域细长而在面板里留下一大片空白。
    xs, ys = points[nodes, 0], points[nodes, 1]
    centre_x, centre_y = (float(xs.min()) + float(xs.max())) / 2, (float(ys.min()) + float(ys.max())) / 2
    half = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2 * 1.16 + 1e-9
    ax.set_aspect("equal")
    ax.set_xlim(centre_x - half, centre_x + half)
    ax.set_ylim(centre_y - half, centre_y + half)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c9ced8")
    ax.set_title(title, fontsize=10.5, pad=6)
    return scatter


def draw_stats_panel_degrees(ax, datasets: Sequence[GraphData], deg_by_name: Dict[str, np.ndarray]):
    for data in datasets:
        values = deg_by_name[data.name]
        counts = np.bincount(values)                  # 度是整数：直接精确计数，不做分箱
        degree_axis = np.arange(counts.size)
        keep = counts > 0
        ax.plot(degree_axis[keep], counts[keep], marker="o", markersize=3.4, linewidth=1.3, label=data.name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("度（无向化）")
    ax.set_ylabel("节点数")
    ax.set_title("度数分布（log-log，整数度精确计数）", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, frameon=False)


def draw_stats_panel_weights(ax, datasets: Sequence[GraphData]):
    drew = False
    for data in datasets:
        if data.weights is None or data.weights.size == 0:
            continue
        values = data.weights[data.weights > 0]
        if values.size == 0:
            continue
        lo, hi = float(values.min()), float(values.max())
        bins = np.logspace(np.log10(lo), np.log10(hi), 40) if hi > lo else np.array([lo, lo * 1.01])
        ax.hist(values, bins=bins, histtype="step", linewidth=1.4, density=True, label=data.name)
        drew = True
    if drew:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("弧长 w（DIMACS9 距离权，单位随数据源）")
        ax.set_ylabel("密度")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "所选数据没有边权", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("边权分布", fontsize=11)
    ax.grid(alpha=0.25, which="both")


def draw_stats_table(ax, rows: Sequence[Sequence[str]], headers: Sequence[str]):
    ax.axis("off")
    table = ax.table(cellText=[list(row) for row in rows], colLabels=list(headers), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#dfe3ea")
        if row == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(weight="bold")


def human_int(value: float) -> str:
    return f"{int(round(value)):,}"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="visualize the public graph datasets under data/public_graphs")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", nargs="+", default=None, help="只画名字里含这些子串的数据集")
    parser.add_argument("--hops", type=int, default=3, help="放大图取最高度数节点周围几跳（默认 3）")
    parser.add_argument("--zoom-cap", type=int, default=600, help="放大图节点数上限（默认 600）")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--seed", type=int, default=0, help="无坐标数据集的布局随机种子")
    parser.add_argument("--skip-components", action="store_true", help="跳过连通分量统计（大图能省几秒）")
    return parser.parse_args()


def discover(data_dir: Path, only: Optional[Sequence[str]]) -> List[Path]:
    paths: List[Path] = []
    for pattern in ("*.gr", "*.gr.gz", "*.graph", "*.graph.bz2"):
        paths.extend(sorted(data_dir.rglob(pattern)))
    if only:
        paths = [path for path in paths if any(token in path.name for token in only)]
    return paths


def main() -> int:
    args = parse_args()
    if not args.data_dir.exists():
        print(f"找不到数据目录：{args.data_dir}", file=sys.stderr)
        print("先跑：E:/CondaEnvData/envs/GGMPC/python.exe scripts/download_graph_datasets.py --with-coords", file=sys.stderr)
        return 2

    paths = discover(args.data_dir.resolve(), args.only)
    if not paths:
        print(f"{args.data_dir} 下没有匹配的图文件", file=sys.stderr)
        return 2

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if candidate in available:
            matplotlib.rcParams["font.sans-serif"] = [candidate]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

    datasets: List[GraphData] = []
    for path in paths:
        print(f"读取 {path.relative_to(PROJECT_ROOT)} ...", flush=True)
        datasets.append(load_dataset(path))
    # 小的排前面，图上不至于被大图的细节淹没。
    datasets.sort(key=lambda item: item.num_nodes)

    degree_by_name: Dict[str, np.ndarray] = {}
    stats_rows: List[List[str]] = []
    for data in datasets:
        degree = degrees(data.num_nodes, data.edges)
        degree_by_name[data.name] = degree
        components = "—" if args.skip_components else human_int(weak_components(data.num_nodes, data.edges))
        weights = "—"
        if data.weights is not None and data.weights.size:
            weights = f"{data.weights.min():.0f} / {data.weights.mean():.1f} / {data.weights.max():.0f}"
        stats_rows.append([
            data.name,
            human_int(data.num_nodes),
            human_int(data.num_edges),
            f"{2 * data.num_edges / max(data.num_nodes, 1):.2f}",
            human_int(int(degree.max())),
            components,
            weights,
            data.coord_source if data.coords is not None else "无坐标",
        ])
        print(
            f"  {data.name:22s} N={human_int(data.num_nodes):>10s}  M={human_int(data.num_edges):>10s}  "
            f"max_deg={int(degree.max()):>4d}  {data.seconds:.1f}s",
            flush=True,
        )

    # ---------------- 图 1：全貌 + 局部放大 ----------------
    rows = len(datasets)
    fig, axes = plt.subplots(rows, 2, figsize=(13.6, 4.2 * rows), gridspec_kw={"width_ratios": [1.25, 1.0]})
    axes = np.atleast_2d(axes)
    for row, data in enumerate(datasets):
        points = project(data.coords) if data.coords is not None else force_layout(data.num_nodes, data.edges, args.seed)
        degree = degree_by_name[data.name]
        hub, nodes = pick_hub(data.num_nodes, data.edges, degree, args.hops, args.zoom_cap)

        span_x = float(points[nodes, 0].max() - points[nodes, 0].min())
        span_y = float(points[nodes, 1].max() - points[nodes, 1].min())
        pad = 0.06
        zoom_box = (
            float(points[nodes, 0].min()) - pad * span_x,
            float(points[nodes, 1].min()) - pad * span_y,
            span_x * (1 + 2 * pad),
            span_y * (1 + 2 * pad),
        )

        layout_note = "经纬度" if data.coords is not None else "力导向布局（无坐标文件）"
        ax = axes[row, 0]
        draw_map_panel(
            ax, data, points, zoom_box,
            f"{data.name} · {human_int(data.num_nodes)} 节点 / {human_int(data.num_edges)} 条"
            f"{'弧' if data.weights is not None else '边'} · {layout_note}",
            linewidth=0.75 if data.num_edges < 200_000 else 0.28,
        )
        ax = axes[row, 1]
        scatter = draw_zoom_panel(
            ax, data, points, nodes, hub, degree,
            f"局部放大：{data.name} 最热闹的路口周围 {args.hops} 跳（{len(nodes)} 个节点，中心度 {int(degree[hub])}）",
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("度", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle("公开图数据全貌（data/public_graphs/）", fontsize=14, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    map_path = args.out_dir / "public_graphs_map.png"
    fig.savefig(map_path, dpi=args.dpi)
    plt.close(fig)
    print(f"\n全貌图 -> {map_path}")

    # ---------------- 图 2：分布 + 统计表 ----------------
    fig = plt.figure(figsize=(13.6, 8.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62], hspace=0.28, wspace=0.16)
    draw_stats_panel_degrees(fig.add_subplot(grid[0, 0]), datasets, degree_by_name)
    draw_stats_panel_weights(fig.add_subplot(grid[0, 1]), datasets)
    draw_stats_table(
        fig.add_subplot(grid[1, :]),
        stats_rows,
        ["数据集", "节点", "边/弧", "平均度", "最大度", "弱连通分量", "弧长 min/mean/max", "坐标来源"],
    )
    fig.suptitle("公开图数据规模与分布（data/public_graphs/）", fontsize=14, y=0.98)
    stats_path = args.out_dir / "public_graphs_stats.png"
    fig.savefig(stats_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"统计图 -> {stats_path}")

    print("\n=== 规模统计 ===")
    header = ["数据集", "节点", "边/弧", "平均度", "最大度", "分量", "弧长 min/mean/max", "坐标来源"]
    print(" | ".join(f"{h}" for h in header))
    for row in stats_rows:
        print(" | ".join(str(cell) for cell in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
