"""图结构术语标注图（v2）：网格路网 + J1 的全部 branch + junction 怎么选 branch。

与 v1 的差别（按使用反馈改）：

    * 图结构更复杂：3x3 网格路网 + s/g 两个接出点，13 节点 / 16 边（v1 是 11/10）
    * **不出现 dead-end**：除 s、g 外所有节点 deg >= 2，所以没有死胡同要标注
    * **J1 的 4 条 branch 全部画出来**，各自上色并标出终点
    * 右侧改成"一个 junction 是怎么选 branch 的"：候选组概率 -> mask 非法 -> top-k

关键：**图上的结构不是手写的，是用仓库真实算法算出来的**。
脚本用 ``src/data/branch_segments.py`` 在示例路网上跑一遍
``node_types / build_endpoints / build_decision_nodes / build_branches``，
再按结果上色标注。所以图不可能和代码语义不一致。

    endpoint A = {v : deg(v) != 2} ∪ {s, g}
    decision D = (J(G) ∪ {s | deg(s) > 1}) \\ {g}
    branch     = 从 decision 沿一个邻居走到下一个 endpoint（中间可穿过 deg=2 节点）

    s --- J2 ---- J3 ---- c
           |       |       |
          J4 -- a--J1 -- b-- J5
           |       |       |
          d  ---- J6 ---- J7 --- g

    GT path : s -> J2 -> J4 -> a -> J1 -> b -> J5 -> J7 -> g
    J1（deg = 4，4 条 branch 全画出来）
    命名：J* = junction（deg >= 3），a/b/c/d = ordinary（deg == 2），s/g = OD

输出（默认 ``outputs/interview/``）::

    graph_concepts.png / .svg

用法::

    python tools/visualize_graph_concepts.py
    python tools/visualize_graph_concepts.py --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import branch_segments as bs  # noqa: E402


# ---------------------------------------------------------------------------
# 与 tools/generate_interview_showcase.py 同一套配色
# ---------------------------------------------------------------------------
BG = "#F7F9FC"
INK = "#172033"
MUTED = "#64748B"
BLUE = "#2563EB"
ORANGE = "#F97316"
GREEN = "#22C55E"
DARK_GREEN = "#15803D"
RED = "#EF4444"
PURPLE = "#7C3AED"
TEAL = "#0D9488"
GREY = "#94A3B8"
LIGHT_GREY = "#E2E8F0"
PINK = "#DB2777"

# ---------------------------------------------------------------------------
# 示例路网：3x3 网格 + s / g 两个接出点
#
# 命名规则**只有一条**，按节点**类型**走 —— 不再按行用 T/M/L，
# 那种命名读图前得先解码"这是第几行"，而且同一张图混三套前缀看着很散：
#
#     J1 .. J7      junction（deg >= 3），焦点那个固定叫 J1
#     a, b, c, d    ordinary（deg == 2）
#     s / g         Start / Goal
# ---------------------------------------------------------------------------
NODE_POS: Dict[str, Tuple[float, float]] = {
    # 上排
    "s": (-2.5, 2.9),
    "J2": (0.0, 2.9),
    "J3": (3.2, 2.9),
    "c": (6.4, 2.9),
    # 中排（a / b 是 J1 两条横向 branch 内部的 deg=2 普通节点）
    "J4": (0.0, 0.0),
    "a": (1.6, 0.0),
    "J1": (3.2, 0.0),
    "b": (4.8, 0.0),
    "J5": (6.4, 0.0),
    # 下排
    "d": (0.0, -2.9),
    "J6": (3.2, -2.9),
    "J7": (6.4, -2.9),
    "g": (8.9, -2.9),
}

EDGES: List[Tuple[str, str]] = [
    ("s", "J2"), ("J2", "J3"), ("J3", "c"),
    ("J2", "J4"), ("J3", "J1"), ("c", "J5"),
    ("J4", "a"), ("a", "J1"), ("J1", "b"), ("b", "J5"),
    ("J4", "d"), ("J1", "J6"), ("J5", "J7"),
    ("d", "J6"), ("J6", "J7"), ("J7", "g"),
]

GT_PATH: List[str] = ["s", "J2", "J4", "a", "J1", "b", "J5", "J7", "g"]
START, GOAL = "s", "g"
FOCUS = "J1"          # 被放大的那个 junction

#: J1 的 4 条 branch 上色（顺序固定，便于对照右图与图例）。
#: 第 4 条是 GT 所走的那条 —— 刻意用深绿色，和 GT path 同一套语义：
#: "绿色 = GT 的选择"，另外三条才是模型可以改选的备选。
#: （原来给的 teal 和 GT 的绿太接近，看图上分不出来。）
BRANCH_COLORS = [ORANGE, PURPLE, PINK, DARK_GREEN]

PANEL_A_TITLE_1 = "A · 路网结构：node / junction / branch"
PANEL_A_TITLE_2 = "命名：J* = junction（deg ≥ 3），a–d = ordinary（deg = 2），s / g = OD"
PANEL_B_TITLE = "B · J1 的候选 branch"

NODE_RADIUS = {
    "start": 0.30, "goal": 0.30, "junction": 0.255, "ordinary": 0.135,
}
JUNCTION_FONT = 7.6


# ---------------------------------------------------------------------------
def configure_style() -> str:
    chosen = "DejaVu Sans"
    for font in ("Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"):
        if font in {item.name for item in matplotlib.font_manager.fontManager.ttflist}:
            chosen = font
            matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            break
    matplotlib.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "text.color": INK,
            "font.size": 12,
        }
    )
    return chosen


# ---------------------------------------------------------------------------
# 用真实算法推结构（图上的标注全部来自这里，不手写）
# ---------------------------------------------------------------------------
class Structure:
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_edges_from(EDGES)
        bs.set_od(self.graph, START, GOAL)
        self.types = bs.node_types(self.graph, START, GOAL)
        self.endpoints = set(bs.build_endpoints(self.graph, START, GOAL))
        self.decisions = list(bs.build_decision_nodes(self.graph, START, GOAL))
        self.groups = bs.build_branches(
            self.graph, self.decisions, start=START, goal=GOAL, endpoints=self.endpoints
        )
        self.branches: List[Any] = []
        for owner, group in zip(self.decisions, self.groups):
            if owner == FOCUS:
                self.branches = list(group)
        if len(self.branches) != 4:
            raise RuntimeError(
                f"{FOCUS} should own 4 branches for this figure, got {len(self.branches)}"
            )

    def kind(self, node: str) -> str:
        if node == START:
            return "start"
        if node == GOAL:
            return "goal"
        if self.types[node] == bs.JUNCTION:
            return "junction"
        return "ordinary"

    def branch_label(self, index: int) -> str:
        branch = self.branches[index]
        inner = list(branch.nodes[1:-1])
        chain = " → ".join([FOCUS] + inner + [str(branch.end)])
        return f"B{index + 1}  {chain}"

    def branch_index_on_gt(self) -> int:
        """GT 在 FOCUS 处走的是哪条 branch（= z_0）。"""
        position = GT_PATH.index(FOCUS)
        nxt = GT_PATH[position + 1]
        for index, branch in enumerate(self.branches):
            if branch.nodes[1] == nxt:
                return index
        raise RuntimeError(f"GT leaves {FOCUS} via {nxt}, which is not a branch")

    def branch_backtracks(self, index: int) -> bool:
        """这条 branch 是否回头（非首节点里有 GT 已经访问过的节点）。"""
        return self.backtrack_node(index) is not None

    def backtrack_node(self, index: int) -> str | None:
        """回头分支里第一个"已访问"的节点名（解码时 mask 的直接原因）。"""
        visited = set(GT_PATH[: GT_PATH.index(FOCUS) + 1])
        for node in self.branches[index].nodes[1:]:
            if node in visited:
                return str(node)
        return None


STRUCTURE: Structure


# ---------------------------------------------------------------------------
def draw_edge(ax, u: str, v: str, **kwargs) -> None:
    (x0, y0), (x1, y1) = NODE_POS[u], NODE_POS[v]
    ax.plot([x0, x1], [y0, y1], solid_capstyle="round", **kwargs)


def draw_chain(ax, nodes: List[str], **kwargs) -> None:
    ax.plot([NODE_POS[n][0] for n in nodes], [NODE_POS[n][1] for n in nodes],
            solid_capstyle="round", solid_joinstyle="round", **kwargs)


def label(ax, x, y, text, **kwargs) -> None:
    opts = {"ha": "center", "va": "center", "fontsize": 11, "color": INK, "zorder": 30}
    opts.update(kwargs)
    ax.text(x, y, text, **opts)


def draw_node(ax, name: str, *, halo: bool = False, focus: bool = False,
              zorder: float = 8.0) -> None:
    x, y = NODE_POS[name]
    kind = STRUCTURE.kind(name)
    radius = NODE_RADIUS[kind]

    if halo:
        ax.add_patch(Circle((x, y), radius * 1.42, facecolor="none", edgecolor=MUTED,
                            linewidth=1.0, linestyle=(0, (2.0, 2.0)), alpha=0.75,
                            zorder=zorder - 2.0))
    if focus:
        ax.add_patch(Circle((x, y), radius * 1.95, facecolor="none", edgecolor=BLUE,
                            linewidth=2.0, linestyle=(0, (1.8, 1.6)), alpha=0.95,
                            zorder=zorder - 1.5))

    face, edge, lw = {
        "start": (GREEN, DARK_GREEN, 2.2),
        "goal": (RED, "#B91C1C", 2.2),
        "junction": (BLUE, "#1D4ED8", 2.0),
        "ordinary": ("#FFFFFF", GREY, 1.6),
    }[kind]
    ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=edge,
                        linewidth=lw, zorder=zorder))


def rounded_box(ax, x, y, w, h, *, face="#FFFFFF", edge=LIGHT_GREY, radius=0.10,
                zorder=2, lw=1.2):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={radius}",
                       facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder)
    )


# ---------------------------------------------------------------------------
# Panel A：路网 + J1 的全部 branch
# ---------------------------------------------------------------------------
def draw_panel_a(ax) -> None:
    ax.set_xlim(-3.60, 9.90)
    ax.set_ylim(-3.60, 3.95)
    ax.set_aspect("equal")
    ax.axis("off")

    # (1) GT 光晕（整条路线，垫在最底下）
    draw_chain(ax, GT_PATH, color=GREEN, linewidth=11.0, alpha=0.16, zorder=1.0)

    # (2) 全部物理边
    for u, v in EDGES:
        draw_edge(ax, u, v, color=LIGHT_GREY, linewidth=3.4, zorder=2.5)

    # (3) J1 的 4 条 branch 全部画出来（在普通边之上）
    for index, branch in enumerate(STRUCTURE.branches):
        color = BRANCH_COLORS[index]
        on_gt = index == STRUCTURE.branch_index_on_gt()
        draw_chain(ax, list(branch.nodes), color=color,
                   linewidth=6.4 if on_gt else 5.0, alpha=0.92, zorder=4.0 + index * 0.01)

        # 标签放在 chain 中点、沿垂直方向偏出去：横向 branch 偏上/下，纵向 branch 偏右。
        # 放在终点外侧会压到 x=0 / x=6.4 那两条竖边和节点，所以按几何推导而不是硬编码。
        end = str(branch.end)
        chain = [NODE_POS[n] for n in branch.nodes]
        mx = sum(p[0] for p in chain) / len(chain)
        my = sum(p[1] for p in chain) / len(chain)
        ox, oy = NODE_POS[FOCUS]
        dx, dy = NODE_POS[end][0] - ox, NODE_POS[end][1] - oy
        if abs(dx) >= abs(dy):          # 横向：西侧偏上、东侧偏下
            tx, ty, ha = mx, my + (0.60 if dx < 0 else -0.60), "center"
        else:                            # 纵向：一律偏右
            tx, ty, ha = mx + 0.82, my, "left"
        text = STRUCTURE.branch_label(index)
        label(ax, tx, ty, text, fontsize=9.8, color=color, weight="bold",
              ha=ha, va="center",
              bbox=dict(boxstyle="round,pad=0.26", facecolor="#FFFFFF",
                        edgecolor=color, linewidth=1.2, alpha=0.97))

    # (4) GT 细虚线压在 branch 之上：回头 branch 会和 GT 走同一段物理边
    #     （本例 B3 = J1→a→J4 正是 GT 来路），不这样画 GT 会被 branch 盖住看不见。
    draw_chain(ax, GT_PATH, color=DARK_GREEN, linewidth=1.9, alpha=0.95,
               linestyle=(0, (4.0, 2.6)), zorder=6.5)

    # (5) 节点
    for name in NODE_POS:
        draw_node(ax, name, halo=name in STRUCTURE.endpoints, focus=name == FOCUS)

    # (6) 节点名
    label(ax, *NODE_POS["s"], "s", color="white", fontsize=11, weight="bold", zorder=30)
    label(ax, *NODE_POS["g"], "g", color="white", fontsize=11, weight="bold", zorder=30)
    for name in NODE_POS:
        if name in (START, GOAL):
            continue
        kind = STRUCTURE.kind(name)
        if kind == "junction":
            label(ax, *NODE_POS[name], name, color="white", fontsize=JUNCTION_FONT,
                  weight="bold", zorder=30)
        else:
            label(ax, *NODE_POS[name], name, color="#475569", fontsize=9.8,
                  weight="bold", zorder=30)

    # (7) s / g 说明 + J1 强调
    label(ax, -2.62, 3.42, "Start s", fontsize=10.5, weight="bold",
          color=DARK_GREEN, ha="left")
    label(ax, 9.05, -2.34, "Goal g", fontsize=10.5, weight="bold",
          color="#B91C1C", ha="right")
    label(ax, 3.2, 0.0, "", fontsize=1)
    label(ax, 3.2, 3.42, "GT path（真实观测轨迹，非最短路）",
          fontsize=10.2, color=DARK_GREEN, weight="bold")
    # J1 说明放在网格空单元格里（原来放在 (3.2,-1.9) 会压在 B1 那条竖 branch 上）
    ax.annotate(
        "J1\njunction（deg = 4）\n4 条 branch 全部画出",
        xy=(3.06, 0.30), xytext=(1.95, 1.30),
        ha="center", va="center", fontsize=10.2, color=BLUE, weight="bold", zorder=30,
        arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=1.2, alpha=0.7,
                        shrinkA=4, shrinkB=3),
    )

    # (8) 图例
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=GREEN,
               markeredgecolor=DARK_GREEN, markersize=11, label="Start node（type = 2）"),
        Line2D([], [], marker="o", color="none", markerfacecolor=RED,
               markeredgecolor="#B91C1C", markersize=11, label="Goal node（type = 3）"),
        Line2D([], [], marker="o", color="none", markerfacecolor=BLUE,
               markeredgecolor="#1D4ED8", markersize=10,
               label="Junction node（type = 1，deg ≥ 3）"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=GREY, markersize=7,
               label="Ordinary node（type = 0，deg = 2）"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=MUTED, markeredgewidth=1.0, markersize=12,
               linestyle=(0, (2.0, 2.0)),
               label="endpoint：A = {v : deg(v) ≠ 2} ∪ {s, g}"),
        Line2D([], [], color=DARK_GREEN, linewidth=3.0,
               label="GT path：训练监督 z_0 的来源"),
    ]
    for index in range(4):
        branch = STRUCTURE.branches[index]
        tag = "（GT 所走 → z_0）" if index == STRUCTURE.branch_index_on_gt() else ""
        handles.append(
            Line2D([], [], color=BRANCH_COLORS[index], linewidth=3.4,
                   label=f"branch B{index + 1} = {STRUCTURE.branch_label(index).split('  ')[1]}"
                         f"  从 J1 到下一个 endpoint{tag}")
        )
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.012),
        ncol=2, frameon=True, facecolor="#FFFFFF", edgecolor=LIGHT_GREY,
        framealpha=1.0, fontsize=9.4, labelspacing=0.56, borderpad=0.85,
        handletextpad=0.7, columnspacing=1.6,
    )


# ---------------------------------------------------------------------------
# Panel B：J1 的候选 branch —— 只列举，不画概率/排序/mask 流程
# ---------------------------------------------------------------------------
def draw_panel_b(ax) -> None:
    ax.set_xlim(0.0, 10.5)
    ax.set_ylim(0.0, 5.70)
    ax.set_aspect("equal")
    ax.axis("off")

    gt_index = STRUCTURE.branch_index_on_gt()

    rows: List[Dict[str, Any]] = []
    for index, branch in enumerate(STRUCTURE.branches):
        rows.append({
            "name": f"B{index + 1}",
            "chain": "  →  ".join(str(n) for n in branch.nodes),
            "color": BRANCH_COLORS[index],
            "is_gt": index == gt_index,
        })
    rows.append({"name": "NULL", "chain": "停止（不激活）", "color": MUTED,
                 "is_gt": False})

    first_y, step = 5.28, 0.90
    for rank, row in enumerate(rows):
        y = first_y - rank * step
        color = row["color"]
        if row["is_gt"]:
            rounded_box(ax, 0.25, y - 0.36, 10.00, 0.72, face="#F0FDF4",
                        edge=GREEN, radius=0.10, lw=1.5, zorder=1.0)
        ax.add_patch(Circle((0.70, y), 0.145, facecolor=color, edgecolor="none",
                            zorder=3))
        label(ax, 1.06, y, row["name"], fontsize=12.5, weight="bold",
              color=color, ha="left")
        label(ax, 2.72, y, row["chain"], fontsize=12.5, color=INK, ha="left")
        if row["is_gt"]:
            label(ax, 10.18, y, "← GT 所走（z_0）", fontsize=10.8, weight="bold",
                  color=DARK_GREEN, ha="right")

    label(ax, 0.25, 0.62, "候选组 = [ NULL , B1 , B2 , B3 , B4 ]",
          fontsize=11.5, weight="bold", color=MUTED, ha="left")



# ---------------------------------------------------------------------------
def main() -> int:
    global STRUCTURE
    parser = argparse.ArgumentParser(description="draw the graph-structure concept figure")
    parser.add_argument("--out-dir", default="outputs/interview")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--name", default="graph_concepts")
    args = parser.parse_args()

    configure_style()
    STRUCTURE = Structure()

    # 自检：图上的结构性结论必须和代码一致，否则直接报错而不是画错图
    gt_index = STRUCTURE.branch_index_on_gt()
    assert STRUCTURE.endpoints == {n for n in NODE_POS
                                   if STRUCTURE.kind(n) in ("start", "goal", "junction")}
    assert FOCUS in STRUCTURE.decisions

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(19.0, 7.9),
                             gridspec_kw={"width_ratios": [1.58, 1.0], "wspace": 0.04})
    for axis in axes:
        axis.set_anchor("N")
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    fig.suptitle("Graph-Junction Diffusion · 图结构术语标注",
                 fontsize=18.5, weight="bold", x=0.5, y=0.985)

    # 两个小标题画在**同一条水平线**上：A 的标题是两行、B 的是一行，
    # 各自用 ax.set_title 的话 B 会明显偏低，并排看很散。
    fig.canvas.draw()
    top = max(axes[0].get_position().y1, axes[1].get_position().y1) + 0.030
    xa = axes[0].get_position().x0
    xb = axes[1].get_position().x0
    fig.text(xa, top, PANEL_A_TITLE_1, ha="left", va="bottom",
             fontsize=14.5, weight="bold", color=INK)
    fig.text(xa, top - 0.040, PANEL_A_TITLE_2, ha="left", va="bottom",
             fontsize=10.8, color=MUTED)
    fig.text(xb, top, PANEL_B_TITLE, ha="left", va="bottom",
             fontsize=14.5, weight="bold", color=INK)

    png = out_dir / f"{args.name}.png"
    svg = out_dir / f"{args.name}.svg"
    for path in (png, svg):
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[figure] wrote {png}")
    print(f"[figure] wrote {svg}")
    print()
    print(f"路网        : {STRUCTURE.graph.number_of_nodes()} nodes / "
          f"{STRUCTURE.graph.number_of_edges()} edges")
    print(f"junction(J) : {STRUCTURE.decisions}")
    print(f"endpoint(A) : {sorted(STRUCTURE.endpoints)}")
    print(f"degree      : {dict(sorted(STRUCTURE.graph.degree(), key=lambda kv: str(kv[0])))}")
    print(f"GT path     : {' -> '.join(GT_PATH)}")
    print()
    print(f"{FOCUS} 的 {len(STRUCTURE.branches)} 条 branch：")
    for index, branch in enumerate(STRUCTURE.branches):
        flags = []
        if index == gt_index:
            flags.append("GT 所走 -> z_0")
        if STRUCTURE.branch_backtracks(index):
            flags.append("回头 -> 解码时 mask")
        print(f"  B{index+1}  {' -> '.join(str(n) for n in branch.nodes)}"
              f"   end={branch.end}   {'  '.join(flags)}")
    print("  NULL  (停止 / 不激活)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
