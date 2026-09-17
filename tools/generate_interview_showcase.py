"""Generate the compact interview showcase requested in the project guide.

Outputs (under ``outputs/interview``): four PNG figures, one MP4 demo and
one Markdown handout.  Evaluation numbers are always read from the existing
formal JSON artifacts; the demo runs the saved Chengdu checkpoint once on a
small deterministic pool and visualizes a successful strict-beam route.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_DIR = ROOT / "outputs/runs/didi_chengdu_loss_improved"
OUT_DIR = ROOT / "outputs/interview"

CHENGDU_JSON = RUN_DIR / "eval_test_1000.json"
XIAN_JSON = RUN_DIR / "eval_xian_test_1000.json"
BEAM_JSONS = {
    3: RUN_DIR / "eval_test_1000_long.json",
    8: RUN_DIR / "eval_test_1000_long_beam8.json",
    16: RUN_DIR / "eval_test_1000_long_beam16.json",
}

BG = "#F7F9FC"
INK = "#172033"
MUTED = "#64748B"
BLUE = "#2563EB"
LIGHT_BLUE = "#DBEAFE"
ORANGE = "#F97316"
GREEN = "#22C55E"
RED = "#EF4444"
PURPLE = "#7C3AED"


def configure_style() -> None:
    for font in ("Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"):
        if font in {item.name for item in matplotlib.font_manager.fontManager.ttflist}:
            matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            break
    matplotlib.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 12,
        }
    )


def load_metric(path: Path) -> Tuple[float, dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metric = float(payload["metrics"]["goal_hit_rate"])
    inference = payload.get("inference", {})
    return metric, inference


def validate_sources() -> Tuple[Dict[str, float], Dict[int, float]]:
    required = [CHENGDU_JSON, XIAN_JSON, *BEAM_JSONS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing formal evaluation JSON: " + ", ".join(missing))

    city = {}
    for name, path in (("chengdu", CHENGDU_JSON), ("xian", XIAN_JSON)):
        value, inference = load_metric(path)
        if not (
            bool(inference.get("strict"))
            and int(inference.get("top_k", -1)) == 2
            and int(inference.get("beam_width", -1)) == 3
        ):
            raise ValueError(f"{path.name} is not strict top-2/beam-3")
        city[name] = value

    beam = {}
    for width, path in BEAM_JSONS.items():
        value, inference = load_metric(path)
        if not bool(inference.get("strict")) or int(inference.get("beam_width", -1)) != width:
            raise ValueError(f"{path.name} does not match strict beam={width}")
        beam[width] = value
    return city, beam


def save_figure(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_background() -> None:
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(-0.7, 10.8)
    ax.set_ylim(-0.8, 6.8)
    ax.axis("off")

    ax.text(0, 6.25, "Conditional Routing Probability Field", fontsize=27, weight="bold")
    ax.text(
        0,
        5.78,
        "Graph + Start/Goal  →  Junction-level branch probabilities  →  Structured route",
        fontsize=13,
        color=MUTED,
    )

    pos = {
        "Start": (0.8, 2.8),
        "Junction": (3.2, 2.8),
        "A": (6.0, 4.65),
        "B": (6.0, 2.8),
        "C": (6.0, 0.95),
        "Goal": (9.3, 4.65),
    }
    edges = [
        ("Start", "Junction", 0.62),
        ("Junction", "A", 0.62),
        ("Junction", "B", 0.27),
        ("Junction", "C", 0.11),
        ("A", "Goal", 0.62),
        ("B", "Goal", 0.27),
        ("C", "Goal", 0.11),
    ]
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    for u, v, probability in edges:
        selected = (u, v) in {("Start", "Junction"), ("Junction", "A"), ("A", "Goal")}
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(u, v)],
            ax=ax,
            width=2.0 + 7.0 * probability,
            edge_color=ORANGE if selected else "#CBD5E1",
            alpha=1.0 if selected else 0.9,
            arrows=False,
        )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=["Start"],
        node_color=GREEN,
        edgecolors="white",
        linewidths=2.5,
        node_size=1200,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=["Goal"],
        node_color=RED,
        edgecolors="white",
        linewidths=2.5,
        node_size=1200,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=["Junction"],
        node_color="white",
        edgecolors=INK,
        linewidths=2.5,
        node_size=1350,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=["A", "B", "C"],
        node_color="#EEF2F7",
        edgecolors="#94A3B8",
        linewidths=1.5,
        node_size=570,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={"Start": "Start", "Goal": "Goal", "Junction": "J", "A": "A", "B": "B", "C": "C"},
        font_size=11,
        font_color=INK,
        ax=ax,
    )
    ax.text(pos["Junction"][0], pos["Junction"][1] - 0.56, "Junction", ha="center", va="top", fontsize=11.5, weight="bold", color=INK)
    for node, probability, dy in (("A", 0.62, 0.52), ("B", 0.27, 0.45), ("C", 0.11, -0.52)):
        x, y = pos[node]
        ax.text(
            x - 1.35,
            y + dy,
            f"Branch {node}   {probability:.2f}",
            fontsize=13,
            weight="bold" if node == "A" else "normal",
            color=ORANGE if node == "A" else MUTED,
        )

    note = (
        "模型输出的是各 Junction 上不同 Branch 的概率分布，"
        "而不是直接输出一条路径。"
    )
    ax.text(
        0,
        -0.18,
        note,
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="white", edgecolor="#E2E8F0"),
    )
    ax.text(7.25, 5.55, r"$\Pi_\theta(G,s,g,W)=\{\pi_i(B)\}$", fontsize=18, color=PURPLE)
    save_figure(fig, "01_background.png")


def rounded_box(
    ax,
    xy,
    width,
    height,
    title,
    subtitle="",
    fc="white",
    ec="#CBD5E1",
    lw=1.6,
    title_size=13,
    subtitle_size=9.5,
):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.12",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(box)
    cx = xy[0] + width / 2
    ax.text(cx, xy[1] + height * (0.59 if subtitle else 0.5), title, ha="center", va="center", fontsize=title_size, weight="bold")
    if subtitle:
        ax.text(cx, xy[1] + height * 0.28, subtitle, ha="center", va="center", fontsize=subtitle_size, color=MUTED)
    return box


def arrow(ax, start, end, color="#94A3B8", lw=1.8):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2))


def _draw_framework_detailed() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.35, 8.55, "Graph-Junction Diffusion · Network Architecture", fontsize=26, weight="bold")
    ax.text(
        0.35,
        8.16,
        "Graph encoding  →  categorical reverse diffusion  →  per-Junction branch probability field",
        fontsize=13,
        color=MUTED,
    )

    def panel(x, y, width, height, title, color):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.03,rounding_size=0.13",
                facecolor="white", edgecolor="#D7DEE9", linewidth=1.6,
            )
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y + height - 0.58), width, 0.58,
                boxstyle="round,pad=0.03,rounding_size=0.13",
                facecolor=color, edgecolor=color, linewidth=0,
            )
        )
        ax.add_patch(patches.Rectangle((x, y + height - 0.36), width, 0.36, facecolor=color, edgecolor=color))
        ax.text(x + 0.18, y + height - 0.29, title, va="center", fontsize=13, weight="bold", color="white")

    def small_box(x, y, width, height, title, subtitle="", fc="#F8FAFC", ec="#CBD5E1", title_size=9.4, subtitle_size=7.6):
        rounded_box(
            ax, (x, y), width, height, title, subtitle, fc, ec, 1.25,
            title_size=title_size, subtitle_size=subtitle_size,
        )

    # ------------------------------------------------------------------
    # A. Graph encoding and structured categorical state.
    # ------------------------------------------------------------------
    panel(0.32, 0.62, 4.15, 7.15, "A · Graph & Structured State", BLUE)
    ax.text(0.55, 6.93, "Road Graph  G, Start s, Goal g", fontsize=11.2, weight="bold")

    graph_pos = {
        "s": (0.8, 6.3), "j1": (1.65, 6.3), "a": (2.45, 6.68),
        "b": (2.45, 5.93), "j2": (3.22, 6.3), "g": (3.96, 6.3),
    }
    graph_edges = [("s", "j1"), ("j1", "a"), ("j1", "b"), ("a", "j2"), ("b", "j2"), ("j2", "g")]
    for u, v in graph_edges:
        ax.plot([graph_pos[u][0], graph_pos[v][0]], [graph_pos[u][1], graph_pos[v][1]], color="#AFC7E8", lw=3, zorder=1)
    for node, (xx, yy) in graph_pos.items():
        color = GREEN if node == "s" else RED if node == "g" else "white"
        edge = "white" if node in {"s", "g"} else "#64748B"
        ax.add_patch(patches.Circle((xx, yy), 0.14, facecolor=color, edgecolor=edge, linewidth=1.3, zorder=3))
        if node in {"s", "g", "j1", "j2"}:
            ax.text(xx, yy, node.upper(), ha="center", va="center", fontsize=7.4, weight="bold", zorder=4)
    ax.text(2.38, 5.75, "Branch decomposition:  J1 → {B1, B2, NULL}", ha="center", fontsize=8.2, color=MUTED)

    rows = [
        (4.82, "Node type", "normal / Start / Goal / Junction", "Embedding", r"$H_T\;(N\!\times\!128)$", "#ECFDF5", GREEN),
        (3.84, r"Decision state  $z_t$", "one category at each Junction", "Branch→Edge", r"$E_t^{state}\;(E_{msg}\!\times\!128)$", "#F5F3FF", PURPLE),
        (2.86, r"Road cost  $w_{uv}$", "normalized physical-edge length", "Cost MLP", r"$E^{cost}\;(E_{msg}\!\times\!128)$", "#FFF7ED", ORANGE),
        (1.88, r"Diffusion step  $t$", "sinusoidal encoding", "Time MLP", r"$\tau_t\;(1\!\times\!128)$", "#EFF6FF", BLUE),
    ]
    for yy, left, note, encoder, out, fc, ec in rows:
        small_box(0.55, yy, 1.45, 0.72, left, note, fc, ec, title_size=8.4, subtitle_size=6.8)
        small_box(2.25, yy + 0.08, 0.82, 0.56, encoder, "", "white", ec, title_size=8.1)
        small_box(3.31, yy + 0.08, 0.88, 0.56, out, "", "#F8FAFC", ec, title_size=7.8)
        arrow(ax, (2.02, yy + 0.36), (2.23, yy + 0.36), color=ec, lw=1.3)
        arrow(ax, (3.09, yy + 0.36), (3.29, yy + 0.36), color=ec, lw=1.3)
    ax.text(0.58, 1.45, r"Forward:  $q(z_t|z_0)$ keeps the clean branch with $\bar\alpha_t$", fontsize=8.7, color=INK)
    ax.text(0.58, 1.12, "otherwise resamples uniformly inside that Junction's candidate set", fontsize=8.0, color=MUTED)
    arrow(ax, (4.20, 4.42), (4.76, 4.42), color=PURPLE, lw=2.4)

    # ------------------------------------------------------------------
    # B. Actual one-step GraphFlow denoiser.
    # ------------------------------------------------------------------
    panel(4.78, 0.62, 6.65, 7.15, r"B · One Reverse Step  $(H_t,z_t,t)\rightarrow(H_{t-1},z_{t-1})$", PURPLE)
    ax.text(5.05, 6.95, "Persistent node state + edge-conditioned graph attention", fontsize=11.2, weight="bold")

    for xx, label, color in (
        (5.02, r"$H_t$", PURPLE),
        (6.13, r"$E_t^{state}$", BLUE),
        (7.52, r"$E^{cost}$", ORANGE),
        (8.79, r"$\tau_t$", GREEN),
    ):
        small_box(xx, 6.22, 0.95 if xx == 5.02 else 1.12, 0.48, label, "", "white", color, title_size=9)

    small_box(5.03, 5.38, 2.1, 0.62, "AdaLN / FiLM", r"$\hat H_t=(1+\gamma_t)LN(H_t)+\beta_t$", "#F5F3FF", PURPLE, title_size=9.5)
    arrow(ax, (5.49, 6.20), (5.49, 6.02), color=PURPLE, lw=1.4)
    arrow(ax, (9.34, 6.20), (7.10, 5.76), color=GREEN, lw=1.25)

    # Attention cell with the exact Q/K/V roles used in graph_flow.py.
    ax.add_patch(patches.FancyBboxPatch((5.02, 3.43), 5.95, 1.62, boxstyle="round,pad=0.04,rounding_size=0.10", facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.5))
    ax.text(5.22, 4.79, "Edge-conditioned Graph Attention", fontsize=11.2, weight="bold")
    ax.text(5.24, 4.39, r"Receiver:   $Q_v=W_Q\hat h_v$", fontsize=9.2)
    ax.text(5.24, 4.03, r"Sender+edge: $K_{uv}=W_{K,n}\hat h_u+W_{K,e}e^{state}_{uv}+W_{K,c}e^{cost}_{uv}$", fontsize=8.7)
    ax.text(5.24, 3.67, r"Message:    $V_{uv}=W_V\hat h_u+W_{V,c}e^{cost}_{uv}$", fontsize=9.0)
    ax.text(8.25, 4.40, r"$\alpha_{uv}=softmax_{u\in\mathcal{N}(v)}(Q_v^\top K_{uv}/\sqrt{d})$", fontsize=8.5, color=BLUE)
    ax.text(8.25, 3.79, r"$m_v=\sum_u\alpha_{uv}V_{uv}$", fontsize=10.0, color=BLUE)
    arrow(ax, (6.67, 5.37), (6.67, 5.07), color=PURPLE, lw=1.5)
    arrow(ax, (6.68, 6.20), (7.33, 5.07), color=BLUE, lw=1.2)
    arrow(ax, (8.08, 6.20), (8.08, 5.07), color=ORANGE, lw=1.2)

    small_box(5.25, 2.55, 2.45, 0.62, "Residual + LayerNorm", r"$\tilde H=LN(H_t+W_Om)$", "#EFF6FF", BLUE, title_size=9.4)
    small_box(8.10, 2.55, 2.45, 0.62, "FFN + Residual + LayerNorm", r"$H_{t-1}=LN(\tilde H+FFN(\tilde H))$", "#EFF6FF", BLUE, title_size=8.8)
    arrow(ax, (7.02, 3.41), (6.48, 3.19), color=BLUE, lw=1.45)
    arrow(ax, (7.72, 2.86), (8.08, 2.86), color=BLUE, lw=1.45)
    ax.text(5.16, 2.15, "Hard constraint: Start / Goal receive no messages and are clamped to their previous state", fontsize=8.4, color=MUTED)

    # Persistent reverse chain ribbon.
    ax.add_patch(patches.FancyBboxPatch((5.02, 1.03), 5.98, 0.82, boxstyle="round,pad=0.03,rounding_size=0.10", facecolor="#FAF5FF", edgecolor="#D8B4FE", linewidth=1.3))
    ax.text(5.23, 1.61, "Persistent reverse chain", fontsize=9.2, weight="bold", color=PURPLE)
    chain = [r"$(H_T,z_T)$", r"$(H_{T-1},z_{T-1})$", r"$\cdots$", r"$(H_1,z_1)$", r"$(H_0,z_0)$"]
    chain_x = [5.30, 6.56, 8.15, 8.74, 9.77]
    for xx, label in zip(chain_x, chain):
        ax.text(xx, 1.28, label, fontsize=8.2, color=INK)
    for a, b in zip(chain_x[:-1], chain_x[1:]):
        arrow(ax, (a + 0.72, 1.31), (b - 0.08, 1.31), color=PURPLE, lw=1.0)

    # ------------------------------------------------------------------
    # C. Candidate scoring and per-Junction normalization.
    # ------------------------------------------------------------------
    panel(11.75, 0.62, 3.93, 7.15, "C · Branch Probability Field", ORANGE)
    ax.text(11.98, 6.96, "For every Junction i", fontsize=11.1, weight="bold")

    # One junction and its variable-length branch candidate set.
    center = (12.35, 6.29)
    ax.add_patch(patches.Circle(center, 0.16, facecolor="white", edgecolor=INK, linewidth=1.7))
    ax.text(*center, "Ji", ha="center", va="center", fontsize=7.3, weight="bold")
    branch_ends = [(13.42, 6.72), (13.42, 6.29), (13.42, 5.86)]
    for idx, end in enumerate(branch_ends, 1):
        ax.plot([center[0] + 0.16, end[0] - 0.13], [center[1], end[1]], color=ORANGE if idx == 1 else "#CBD5E1", lw=3.2 - idx * 0.35)
        ax.add_patch(patches.Circle(end, 0.12, facecolor="#FFF7ED", edgecolor=ORANGE, linewidth=1.1))
        ax.text(end[0] + 0.22, end[1], f"B{idx}", va="center", fontsize=8.3)
    ax.text(14.30, 6.29, "+ NULL", fontsize=8.5, color=MUTED, va="center")

    small_box(11.99, 5.03, 3.43, 0.62, "Branch mean pooling", r"$\bar h_{ik}=mean_{v\in B_{ik}\setminus\{i\}}h_v^{t-1}$", "#FFF7ED", ORANGE, title_size=9.4)
    small_box(11.99, 4.15, 2.14, 0.62, "Branch scorer MLP", r"$[h_i^{t-1},\bar h_{ik},\tau_t]\rightarrow \ell_{ik}$", "#EFF6FF", BLUE, title_size=9.0)
    small_box(14.30, 4.15, 1.12, 0.62, "NULL MLP", r"$[h_i,\tau_t]$", "#F5F3FF", PURPLE, title_size=8.4)
    arrow(ax, (13.70, 5.01), (13.28, 4.79), color=ORANGE, lw=1.3)
    arrow(ax, (14.86, 5.01), (14.86, 4.79), color=PURPLE, lw=1.3)

    small_box(12.35, 3.27, 2.72, 0.58, "Grouped Softmax inside each Junction", r"$\pi_i(c)=softmax_{c\in\mathcal{C}_i}(\ell_{ic})$", "#ECFDF5", GREEN, title_size=8.9)
    arrow(ax, (13.08, 4.13), (13.08, 3.87), color=BLUE, lw=1.3)
    arrow(ax, (14.86, 4.13), (14.56, 3.87), color=PURPLE, lw=1.3)
    ax.text(13.70, 3.08, r"$p_\theta(z_0\mid z_t,G,s,g,W)=\{\pi_i\}_{i=1}^{M}$", ha="center", fontsize=8.3, color=ORANGE)

    ax.add_patch(patches.FancyBboxPatch((12.08, 2.05), 3.24, 0.90, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2))
    ax.text(12.30, 2.69, "J1  [0.62, 0.27, 0.11]", fontsize=8.8, family="monospace")
    ax.text(12.30, 2.39, "J2  [0.18, 0.71, 0.11]", fontsize=8.8, family="monospace")
    ax.text(12.30, 2.09, "J3  [0.09, 0.23, 0.68]", fontsize=8.8, family="monospace")

    small_box(11.98, 1.10, 1.70, 0.62, "Reverse posterior", r"$p_\theta(z_{t-1}|z_t)$", "#F5F3FF", PURPLE, title_size=8.8)
    small_box(13.95, 1.10, 1.48, 0.62, "Strict Beam", "legal route", "#ECFDF5", GREEN, title_size=9.0)
    arrow(ax, (12.88, 2.03), (12.88, 1.74), color=PURPLE, lw=1.3)
    arrow(ax, (14.55, 2.03), (14.55, 1.74), color=GREEN, lw=1.3)
    ax.text(11.98, 0.80, "During diffusion: update z.  After t=1: decode the probability field.", fontsize=7.8, color=MUTED)

    # Cross-panel connections: updated H feeds the scorer; posterior returns z_{t-1}.
    arrow(ax, (10.56, 2.84), (11.72, 4.47), color=BLUE, lw=2.0)
    ax.text(10.86, 3.67, r"$H_{t-1}$", fontsize=9.0, color=BLUE, weight="bold")
    arrow(ax, (11.98, 1.38), (10.98, 1.38), color=PURPLE, lw=1.8)
    ax.text(11.13, 1.54, r"$z_{t-1}$", fontsize=8.5, color=PURPLE)

    ax.text(
        0.40, 0.16,
        "Current Chengdu checkpoint: d=128 · T=50 · weighted edge cost · 1 GraphFlow round per reverse step · shared parameters across all t",
        fontsize=9.5, color=MUTED,
    )
    save_figure(fig, "02_framework_detailed.png")


def _draw_framework_three_stage() -> None:
    """Draw the interview-facing architecture: three readable stages, no clutter."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#F5F7FB")
    ax.set_facecolor("#F5F7FB")

    navy = "#13213C"
    blue = "#246BFD"
    violet = "#7447F5"
    orange = "#FF7A1A"
    green = "#18A66A"
    line = "#D9E1EE"
    soft = "#F8FAFD"

    ax.text(0.55, 8.50, "Graph-Junction Diffusion", fontsize=28, weight="bold", color=navy)
    ax.text(
        0.56, 8.08,
        "从道路图编码，到离散反向扩散，再到每个 Junction 的 Branch 概率分布",
        fontsize=13.5, color=MUTED,
    )

    def stage(x, width, number, title, subtitle, color):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, 0.70), width, 6.95,
                boxstyle="round,pad=0.035,rounding_size=0.16",
                facecolor="white", edgecolor=line, linewidth=1.6,
            )
        )
        ax.add_patch(patches.Circle((x + 0.38, 7.25), 0.22, facecolor=color, edgecolor="none"))
        ax.text(x + 0.38, 7.25, str(number), ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(x + 0.72, 7.30, title, fontsize=16, weight="bold", color=navy, va="center")
        ax.text(x + 0.72, 6.96, subtitle, fontsize=9.8, color=MUTED, va="center")

    def card(x, y, width, height, title, subtitle="", edge=line, fill=soft, title_size=10.5, subtitle_size=8.2):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.025,rounding_size=0.10",
                facecolor=fill, edgecolor=edge, linewidth=1.35,
            )
        )
        cy = y + height / 2
        ax.text(x + width / 2, cy + (0.13 if subtitle else 0), title, ha="center", va="center", fontsize=title_size, weight="bold", color=navy)
        if subtitle:
            ax.text(x + width / 2, cy - 0.17, subtitle, ha="center", va="center", fontsize=subtitle_size, color=MUTED)

    def flow(start, end, color="#9AA9BF", width=1.8, style="-|>"):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle=style, color=color, lw=width, shrinkA=1, shrinkB=1))

    # Three clean stages.
    stage(0.45, 4.30, 1, "Graph Encoding", "把结构、路径状态、道路代价和时间编码成向量", blue)
    stage(5.05, 5.70, 2, "Categorical Reverse Diffusion", "共享 GraphFlow Cell，持续更新 (Hₜ, zₜ)", violet)
    stage(11.05, 4.50, 3, "Junction-wise Branch Head", "逐分叉点生成变长候选集合上的概率分布", orange)

    # ------------------------------------------------------------------
    # Stage 1: road graph, branch decomposition and four encoders.
    # ------------------------------------------------------------------
    graph_pos = {
        "S": (0.88, 6.16), "J1": (1.72, 6.16), "a": (2.55, 6.48),
        "b": (2.55, 5.84), "J2": (3.38, 6.16), "G": (4.22, 6.16),
    }
    for u, v in (("S", "J1"), ("J1", "a"), ("J1", "b"), ("a", "J2"), ("b", "J2"), ("J2", "G")):
        ax.plot([graph_pos[u][0], graph_pos[v][0]], [graph_pos[u][1], graph_pos[v][1]], color="#AFC8F8", lw=4, solid_capstyle="round")
    for name, (xx, yy) in graph_pos.items():
        fill = green if name == "S" else "#F04452" if name == "G" else "white"
        border = fill if name in {"S", "G"} else "#60708A"
        ax.add_patch(patches.Circle((xx, yy), 0.16, facecolor=fill, edgecolor=border, linewidth=1.5, zorder=3))
        if name in {"S", "G", "J1", "J2"}:
            ax.text(xx, yy, name, ha="center", va="center", fontsize=7.6, weight="bold", color="white" if name in {"S", "G"} else navy, zorder=4)
    ax.text(2.55, 5.47, "Branch Decomposition", ha="center", fontsize=10.5, weight="bold", color=blue)
    ax.text(2.55, 5.20, "只在真正的分叉点建立 categorical variable", ha="center", fontsize=8.8, color=MUTED)

    encoder_rows = [
        (4.48, "Node Type", "Embedding", r"$H_T$", blue),
        (3.69, r"Decision $z_t$", "Branch → Edge", r"$E_t^{state}$", violet),
        (2.90, "Edge Cost", "Cost MLP", r"$E^{cost}$", orange),
        (2.11, "Timestep t", "Time Encoder", r"$\tau_t$", green),
    ]
    for yy, source, encoder, output, color in encoder_rows:
        card(0.72, yy, 1.18, 0.55, source, edge=color, fill="white", title_size=9.1)
        flow((1.92, yy + 0.275), (2.15, yy + 0.275), color=color, width=1.6)
        card(2.17, yy, 1.28, 0.55, encoder, edge=color, fill=soft, title_size=8.7)
        flow((3.47, yy + 0.275), (3.69, yy + 0.275), color=color, width=1.6)
        card(3.71, yy, 0.73, 0.55, output, edge=color, fill="white", title_size=9.4)
    card(0.79, 1.14, 3.55, 0.58, "Encoded graph state", r"$\{H_t, E_t^{state}, E^{cost}, \tau_t\}$", edge="#B7C4D8", fill="#F1F5FA", title_size=10.2)
    for yy, *_ in encoder_rows:
        flow((4.45, yy + 0.275), (4.60, 1.43), color="#A8B5C8", width=1.0)
    flow((4.36, 1.43), (5.02, 1.43), color=blue, width=3.0)

    # ------------------------------------------------------------------
    # Stage 2: one reusable reverse cell and the temporal recurrence.
    # ------------------------------------------------------------------
    ax.text(5.42, 6.36, "Reverse chain", fontsize=10.5, weight="bold", color=violet)
    chain_x = [6.10, 7.35, 8.30, 9.18, 10.10]
    chain_labels = [r"$t=50$", r"$t=49$", r"$\cdots$", r"$t=2$", r"$t=1$"]
    for idx, (xx, label) in enumerate(zip(chain_x, chain_labels)):
        if label == r"$\cdots$":
            ax.text(xx, 6.13, label, ha="center", va="center", fontsize=14, color=MUTED)
        else:
            ax.add_patch(patches.Circle((xx, 6.13), 0.27, facecolor="#F4F0FF", edgecolor=violet, linewidth=1.5))
            ax.text(xx, 6.13, label, ha="center", va="center", fontsize=8.3, color=navy)
        if idx < len(chain_x) - 1:
            flow((xx + 0.29, 6.13), (chain_x[idx + 1] - 0.30, 6.13), color=violet, width=1.5)

    # Main GraphFlow cell.
    ax.add_patch(
        patches.FancyBboxPatch(
            (5.52, 2.30), 4.78, 3.30,
            boxstyle="round,pad=0.035,rounding_size=0.14",
            facecolor="#FBFAFF", edgecolor="#BBA7FF", linewidth=1.8,
        )
    )
    ax.text(5.82, 5.27, "GraphFlow Denoiser Cell", fontsize=14.5, weight="bold", color=navy)
    ax.text(9.96, 5.29, "shared × 50", ha="right", fontsize=9.0, color=violet, weight="bold")

    card(5.83, 4.43, 1.48, 0.55, r"$H_t$ + $\tau_t$", "AdaLN / FiLM", edge=violet, fill="white", title_size=9.5)
    flow((7.33, 4.70), (7.65, 4.70), color=violet, width=1.8)
    card(7.68, 4.24, 2.20, 0.92, "Edge-aware Attention", "Q: receiver node\nK/V: sender + state + cost", edge=blue, fill="#EFF5FF", title_size=10.5, subtitle_size=7.6)
    flow((8.78, 4.21), (8.78, 3.91), color=blue, width=1.8)
    card(7.43, 3.24, 2.70, 0.64, "Residual + FFN + LayerNorm", edge=blue, fill="white", title_size=9.8)
    flow((8.78, 3.22), (8.78, 2.87), color=blue, width=1.8)
    card(7.88, 2.54, 1.80, 0.52, r"Persistent $H_{t-1}$", edge=violet, fill="#F4F0FF", title_size=9.4)

    # Edge inputs enter attention directly.
    card(5.80, 3.50, 1.10, 0.46, r"$E_t^{state}$", edge=violet, fill="white", title_size=9.0)
    card(5.80, 2.86, 1.10, 0.46, r"$E^{cost}$", edge=orange, fill="white", title_size=9.0)
    flow((6.92, 3.73), (7.66, 4.39), color=violet, width=1.45)
    flow((6.92, 3.09), (7.66, 4.28), color=orange, width=1.45)
    flow((7.31, 4.70), (7.66, 4.70), color=violet, width=1.7)

    # Categorical reverse update is distinct from graph encoding.
    card(5.58, 1.10, 4.62, 0.76, "Categorical posterior update", r"$\pi_\theta(z_0\mid z_t,G,s,g,W)+z_t\;\rightarrow\;p_\theta(z_{t-1}\mid z_t)\;\rightarrow\;z_{t-1}$", edge=violet, fill="#F4F0FF", title_size=10.5, subtitle_size=8.4)
    flow((8.78, 2.52), (8.78, 1.88), color=violet, width=1.7)
    ax.annotate("", xy=(5.58, 2.68), xytext=(5.58, 1.47), arrowprops=dict(arrowstyle="-|>", color=violet, lw=1.7, connectionstyle="arc3,rad=-0.35"))
    ax.text(5.42, 2.02, "next t", fontsize=8.2, color=violet, rotation=90, va="center")

    # ------------------------------------------------------------------
    # Stage 3: branch pooling, logits, grouped softmax, decoder.
    # ------------------------------------------------------------------
    center = (11.62, 6.05)
    ax.add_patch(patches.Circle(center, 0.17, facecolor="white", edgecolor=navy, linewidth=1.7))
    ax.text(*center, "Ji", ha="center", va="center", fontsize=7.6, weight="bold", color=navy)
    branch_y = [6.46, 6.05, 5.64]
    for idx, yy in enumerate(branch_y, 1):
        ax.plot([11.79, 12.73], [6.05, yy], color=orange if idx == 1 else "#CDD6E5", lw=3.5 - idx * 0.35, solid_capstyle="round")
        ax.add_patch(patches.Circle((12.78, yy), 0.12, facecolor="#FFF5EC", edgecolor=orange, linewidth=1.2))
        ax.text(13.02, yy, f"Branch {idx}", va="center", fontsize=8.8, color=navy)
    ax.text(14.02, 6.05, "+ NULL", fontsize=9.0, color=MUTED, va="center")

    card(11.42, 4.69, 3.75, 0.67, "Branch Mean Pool", r"$\bar h_{ik}=mean\{h_v:v\in B_{ik}\setminus i\}$", edge=orange, fill="#FFF7F0", title_size=10.3)
    flow((13.30, 5.62), (13.30, 5.38), color=orange, width=1.7)

    card(11.42, 3.76, 2.45, 0.68, "Branch Scorer", r"$[h_i,\bar h_{ik},\tau_t]\rightarrow \ell_{ik}$", edge=blue, fill="#EFF5FF", title_size=10.1)
    card(14.05, 3.76, 1.12, 0.68, "NULL", r"$[h_i,\tau_t]$", edge=violet, fill="#F4F0FF", title_size=9.5)
    flow((13.30, 4.67), (12.67, 4.46), color=orange, width=1.5)
    flow((14.62, 4.67), (14.62, 4.46), color=violet, width=1.5)

    card(11.70, 2.84, 3.20, 0.60, "Grouped Softmax per Junction", r"$\sum_{c\in\mathcal{C}_i}\pi_i(c)=1$", edge=green, fill="#EDFBF5", title_size=10.0)
    flow((12.67, 3.74), (12.67, 3.46), color=blue, width=1.5)
    flow((14.62, 3.74), (14.20, 3.46), color=violet, width=1.5)

    # A compact, visual probability distribution instead of another text table.
    ax.text(11.42, 2.47, "Junction i", fontsize=9.2, weight="bold", color=navy)
    probs = [("B1", 0.62, orange), ("B2", 0.27, blue), ("NULL", 0.11, "#9AA9BF")]
    for idx, (label, prob, color) in enumerate(probs):
        yy = 2.22 - idx * 0.36
        ax.text(11.42, yy, label, fontsize=8.4, color=MUTED, va="center")
        ax.add_patch(patches.FancyBboxPatch((11.95, yy - 0.09), 2.30, 0.18, boxstyle="round,pad=0.0,rounding_size=0.06", facecolor="#EDF1F7", edgecolor="none"))
        ax.add_patch(patches.FancyBboxPatch((11.95, yy - 0.09), 2.30 * prob, 0.18, boxstyle="round,pad=0.0,rounding_size=0.06", facecolor=color, edgecolor="none"))
        ax.text(14.40, yy, f"{prob:.2f}", fontsize=8.4, color=navy, va="center", ha="right")

    card(11.42, 0.91, 1.68, 0.58, "Probability Field", r"$\{\pi_i\}_{i=1}^{M}$", edge=orange, fill="#FFF7F0", title_size=9.2)
    card(13.38, 0.91, 1.78, 0.58, "Strict Beam Search", "legal route", edge=green, fill="#EDFBF5", title_size=8.8)
    flow((13.12, 1.20), (13.36, 1.20), color=green, width=2.0)

    # Cross-stage semantics: H goes to scorer; probabilities return to posterior.
    flow((10.32, 2.80), (11.03, 4.08), color=blue, width=2.5)
    ax.text(10.45, 3.53, r"$H_{t-1}$", fontsize=9.0, color=blue, weight="bold")
    ax.annotate("", xy=(10.22, 1.45), xytext=(11.40, 1.20), arrowprops=dict(arrowstyle="-|>", color=violet, lw=2.2, connectionstyle="arc3,rad=0.10"))
    ax.text(10.56, 1.13, r"$\pi_\theta(z_0|\cdot)$", fontsize=8.4, color=violet)

    ax.text(
        0.55, 0.20,
        "成都 checkpoint：d=128 · T=50 · Edge Cost enabled · 1 GraphFlow round / reverse step · 参数跨 t 共享，隐藏状态 H 持续传递",
        fontsize=10.0, color=MUTED,
    )
    save_figure(fig, "02_framework.png")


def draw_framework() -> None:
    """Integrated end-to-end architecture for the interview slide."""
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#F6F8FC")
    ax.set_facecolor("#F6F8FC")

    navy = "#14213D"
    blue = "#2764E7"
    violet = "#7047EB"
    orange = "#F97316"
    green = "#16A66A"
    gray = "#65758B"
    border = "#D7DFEB"

    ax.text(0.52, 8.50, "Graph-Junction Diffusion · Network Architecture", fontsize=27, weight="bold", color=navy)
    ax.text(0.53, 8.10, "Graph + Start/Goal  →  persistent categorical reverse diffusion  →  Junction-wise branch probabilities  →  Route", fontsize=12.5, color=gray)

    def box(x, y, w, h, title, subtitle="", edge=border, fill="white", title_size=10.2, subtitle_size=8.0, lw=1.4):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.10", facecolor=fill, edgecolor=edge, linewidth=lw))
        ax.text(x + w / 2, y + h / 2 + (0.13 if subtitle else 0), title, ha="center", va="center", fontsize=title_size, weight="bold", color=navy)
        if subtitle:
            ax.text(x + w / 2, y + h / 2 - 0.17, subtitle, ha="center", va="center", fontsize=subtitle_size, color=gray)

    def arrow(start, end, color="#93A2B8", lw=1.8, connection=None):
        props = dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2)
        if connection:
            props["connectionstyle"] = connection
        ax.annotate("", xy=end, xytext=start, arrowprops=props)

    # ------------------------------------------------------------------
    # Continuous top pipeline: graph -> structured state -> unrolled reverse chain.
    # ------------------------------------------------------------------
    ax.text(0.55, 7.55, "INPUT GRAPH", fontsize=9.0, weight="bold", color=blue)
    graph_pos = {
        "S": (0.83, 6.52), "J1": (1.50, 6.52), "a": (2.12, 6.82),
        "b": (2.12, 6.22), "J2": (2.75, 6.52), "G": (3.38, 6.52),
    }
    for u, v in (("S", "J1"), ("J1", "a"), ("J1", "b"), ("a", "J2"), ("b", "J2"), ("J2", "G")):
        ax.plot([graph_pos[u][0], graph_pos[v][0]], [graph_pos[u][1], graph_pos[v][1]], color="#AFC7F4", lw=4, solid_capstyle="round")
    for name, (xx, yy) in graph_pos.items():
        fill = green if name == "S" else "#EF4444" if name == "G" else "white"
        edge = fill if name in {"S", "G"} else gray
        ax.add_patch(patches.Circle((xx, yy), 0.145, facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=3))
        if name in {"S", "G", "J1", "J2"}:
            ax.text(xx, yy, name, ha="center", va="center", fontsize=7.0, weight="bold", color="white" if name in {"S", "G"} else navy, zorder=4)
    ax.text(2.10, 5.90, "node type · edge cost · topology", ha="center", fontsize=8.4, color=gray)

    arrow((3.60, 6.50), (3.94, 6.50), color=blue, lw=2.2)
    box(3.97, 5.92, 1.74, 1.18, "Branch Decomposition", "degree-2 chains → Branches\ndecisions only at Junctions", edge=blue, fill="#EEF4FF", title_size=8.9, subtitle_size=7.3, lw=1.7)
    arrow((5.73, 6.50), (6.06, 6.50), color=blue, lw=2.2)

    box(6.10, 5.86, 1.62, 1.30, "Initial State", r"$H_T=Emb(type)$" + "\n" + r"$z_T\sim Uniform(\mathcal{C}_i)$", edge=violet, fill="#F4F0FF", title_size=10.5, subtitle_size=8.1, lw=1.7)
    arrow((7.74, 6.50), (8.02, 6.50), color=violet, lw=2.3)

    # Unrolled shared cell. Each box intentionally has the same visual identity.
    cell_specs = [(8.05, "t = 50"), (9.70, "t = 49"), (11.82, "t = 1")]
    for xx, label in cell_specs:
        box(xx, 5.75, 1.42, 1.52, "GraphFlow θ", label + "\nshared cell", edge=violet, fill="white", title_size=10.3, subtitle_size=7.7, lw=1.8)
    arrow((9.49, 6.50), (9.68, 6.50), color=violet, lw=2.0)
    ax.text(11.12, 6.50, "···", ha="center", va="center", fontsize=20, color=gray)
    arrow((10.98, 6.50), (11.78, 6.50), color=violet, lw=2.0)
    ax.text(10.75, 7.48, "same parameters θ · persistent H", ha="center", fontsize=8.7, color=violet, weight="bold")
    arrow((9.12, 7.30), (12.52, 7.30), color=violet, lw=1.5, connection="arc3,rad=-0.12")

    arrow((13.26, 6.50), (13.52, 6.50), color=orange, lw=2.3)
    box(13.55, 5.74, 1.90, 1.54, "Probability Field", r"$\{\pi_i(B)\}_{i=1}^{M}$", edge=orange, fill="#FFF6EE", title_size=10.5, subtitle_size=9.0, lw=1.8)

    # ------------------------------------------------------------------
    # Enlarged view of the single shared GraphFlow cell. One integrated flow.
    # ------------------------------------------------------------------
    ax.plot([8.22, 8.22, 4.00], [5.73, 5.36, 5.36], color="#B9C3D2", lw=1.2, ls=(0, (4, 3)))
    ax.plot([9.30, 9.30, 12.76], [5.73, 5.36, 5.36], color="#B9C3D2", lw=1.2, ls=(0, (4, 3)))
    ax.text(4.02, 5.08, "ONE SHARED REVERSE-DIFFUSION CELL", fontsize=9.2, weight="bold", color=violet)

    # Inputs on the left of the expanded cell.
    input_items = [
        (4.02, 4.42, r"$H_t$", violet),
        (4.02, 3.72, r"$z_t$", violet),
        (4.02, 3.02, r"$w_{uv}$", orange),
        (4.02, 2.32, r"$t$", blue),
    ]
    for xx, yy, label, color in input_items:
        box(xx, yy, 0.78, 0.46, label, edge=color, fill="white", title_size=9.2, lw=1.4)

    # Encoders are a continuous fan-in, not separate stages.
    box(5.20, 4.27, 1.55, 0.72, "AdaLN", r"$H_t + \tau_t$", edge=violet, fill="#F4F0FF", title_size=10.0)
    box(5.20, 3.43, 1.55, 0.62, "Branch → Edge", r"$z_t\rightarrow E_t^{state}$", edge=violet, fill="#F4F0FF", title_size=9.4)
    box(5.20, 2.65, 1.55, 0.62, "Cost Encoder", r"$w_{uv}\rightarrow E^{cost}$", edge=orange, fill="#FFF6EE", title_size=9.4)
    box(5.20, 1.87, 1.55, 0.62, "Time Encoder", r"$t\rightarrow\tau_t$", edge=blue, fill="#EEF4FF", title_size=9.4)
    arrow((4.82, 4.65), (5.18, 4.65), color=violet, lw=1.6)
    arrow((4.82, 3.95), (5.18, 3.75), color=violet, lw=1.6)
    arrow((4.82, 3.25), (5.18, 2.96), color=orange, lw=1.6)
    arrow((4.82, 2.55), (5.18, 2.18), color=blue, lw=1.6)

    box(7.18, 3.10, 2.26, 1.47, "Graph Attention", "Q: receiver node\nK: sender + state + cost\nV: sender + edge cost", edge=blue, fill="#EEF4FF", title_size=10.8, subtitle_size=7.5, lw=1.7)
    ax.text(8.31, 4.39, "edge-conditioned", ha="center", fontsize=7.7, color=blue, weight="bold")
    for yy, color in ((4.63, violet), (3.74, violet), (2.96, orange), (2.18, blue)):
        arrow((6.77, yy), (7.16, 3.84), color=color, lw=1.35)

    box(7.40, 2.05, 1.82, 0.62, "Residual + FFN", r"$\rightarrow H_{t-1}$", edge=blue, fill="white", title_size=9.6)
    arrow((8.31, 3.08), (8.31, 2.69), color=blue, lw=1.8)

    # Branch head is in the denoiser step, producing clean-state probabilities.
    box(9.88, 3.77, 1.72, 0.77, "Branch Mean Pool", r"$\bar h_{ik}=mean(B_{ik})$", edge=orange, fill="#FFF6EE", title_size=9.4)
    box(9.88, 2.84, 1.72, 0.68, "Branch / NULL MLP", r"$[h_i,\bar h_{ik},\tau_t]\rightarrow\ell_{ik}$", edge=orange, fill="white", title_size=8.8)
    box(9.88, 1.93, 1.72, 0.67, "Grouped Softmax", r"$\pi_i(B)$", edge=green, fill="#EDFBF5", title_size=9.5)
    arrow((9.24, 2.36), (9.86, 4.13), color=blue, lw=1.6)
    arrow((10.74, 3.75), (10.74, 3.54), color=orange, lw=1.5)
    arrow((10.74, 2.82), (10.74, 2.62), color=green, lw=1.5)

    box(12.05, 2.45, 1.85, 0.86, "Categorical Posterior", r"$\pi_i + z_t\rightarrow z_{t-1}$", edge=violet, fill="#F4F0FF", title_size=9.4)
    arrow((11.62, 2.26), (12.03, 2.74), color=green, lw=1.7)
    box(11.66, 3.22, 0.38, 0.32, r"$z_t$", edge=violet, fill="white", title_size=7.8, lw=1.2)
    arrow((11.86, 3.20), (12.18, 3.02), color=violet, lw=1.4)
    arrow((13.92, 2.88), (14.35, 2.88), color=violet, lw=1.8)
    box(14.38, 2.41, 1.08, 0.94, r"$z_{t-1}$", "next step", edge=violet, fill="white", title_size=10.5)

    # Final probability field: one Junction's variable-length candidate distribution.
    ax.text(12.15, 4.70, "每个 Junction 独立归一化", fontsize=9.5, weight="bold", color=orange)
    probs = [("B1", 0.62, orange), ("B2", 0.27, blue), ("NULL", 0.11, "#9AA9BF")]
    for idx, (label, prob, color) in enumerate(probs):
        yy = 4.38 - idx * 0.34
        ax.text(12.15, yy, label, fontsize=8.0, color=gray, va="center")
        ax.add_patch(patches.FancyBboxPatch((12.65, yy - 0.08), 2.10, 0.16, boxstyle="round,pad=0,rounding_size=0.05", facecolor="#E9EEF5", edgecolor="none"))
        ax.add_patch(patches.FancyBboxPatch((12.65, yy - 0.08), 2.10 * prob, 0.16, boxstyle="round,pad=0,rounding_size=0.05", facecolor=color, edgecolor="none"))
        ax.text(14.95, yy, f"{prob:.2f}", fontsize=8.0, color=navy, va="center", ha="right")

    # Final route readout.
    arrow((14.52, 5.72), (14.52, 5.30), color=green, lw=2.0)
    box(13.38, 4.87, 2.08, 0.52, "Strict Beam Search", "Complete legal route", edge=green, fill="#EDFBF5", title_size=9.0, subtitle_size=7.2)

    ax.text(0.55, 0.42, "成都模型：d=128 · T=50 · Edge Cost enabled · GraphFlow 参数跨 timestep 共享 · H 在 reverse chain 中持续传递", fontsize=9.7, color=gray)
    save_figure(fig, "02_framework.png")


def draw_cross_city(city: Dict[str, float]) -> None:
    labels = ["成都 / 同分布", "西安 / Zero-shot"]
    values = [city["chengdu"], city["xian"]]
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    bars = ax.bar(labels, values, color=[BLUE, PURPLE], width=0.5, edgecolor="white", linewidth=1.5)
    ax.set_ylim(0.94, 1.006)
    ax.set_ylabel("GoalHit")
    ax.set_title("Cross-city Zero-shot Generalization", fontsize=22, weight="bold", pad=22)
    ax.grid(axis="y", color="#E2E8F0", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0013, f"{value:.1%}", ha="center", va="bottom", fontsize=19, weight="bold")
    fig.text(0.5, 0.01, "成都训练模型无需重新训练即可迁移到西安，同规模场景下到达率基本保持。", ha="center", fontsize=12.5, color=MUTED)
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.96))
    save_figure(fig, "03_cross_city.png")


def draw_beam(beam: Dict[int, float]) -> None:
    widths = sorted(beam)
    values = [beam[width] for width in widths]
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.plot(widths, values, color=ORANGE, linewidth=4, marker="o", markersize=11, markerfacecolor="white", markeredgewidth=3)
    ax.fill_between(widths, [min(values) - 0.02] * len(values), values, color="#FFEDD5", alpha=0.55)
    ax.set_xticks(widths)
    ax.set_xlabel("Beam Width")
    ax.set_ylabel("GoalHit")
    ax.set_ylim(0.74, 1.01)
    ax.set_title("Large-graph Beam Ablation", fontsize=22, weight="bold", pad=22)
    ax.grid(color="#E2E8F0", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for width, value in zip(widths, values):
        ax.text(width, value + 0.013, f"{value:.1%}", ha="center", va="bottom", fontsize=15, weight="bold", color=INK)
    ax.text(4.2, 0.765, "Same model, only decoder search budget changes.\n模型参数不变，仅扩大 Decoder 搜索预算。", fontsize=11.5, color=MUTED, bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor="#E2E8F0"))
    fig.text(0.5, 0.01, "大图中小 Beam 容易提前剪掉正确前缀；扩大搜索预算后 GoalHit 显著恢复。", ha="center", fontsize=12.5, color=MUTED)
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.96))
    save_figure(fig, "04_beam_ablation.png")


def edge_list(graph: nx.Graph, path: Sequence[int]) -> List[Tuple[int, int]]:
    return [(int(u), int(v)) for u, v in zip(path[:-1], path[1:]) if graph.has_edge(u, v)]


def decode_demo_case():
    import tools.visualize_didi_paths as vdp
    import tools.visualize_didi_samples as vds
    from src.data.dataset import GraphQueryDataset
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_diffusion, build_model, get_device
    from src.utils.config import load_config
    from src.utils.seed import make_generator, set_seed

    run_config = RUN_DIR / "run_config.json"
    checkpoint = RUN_DIR / "best.pt"
    data_path = ROOT / "data/didi/graph/chengdu/test_1000.pkl"
    config = load_config(run_config)
    device = get_device("cpu")
    seed = int(config.get("seed", 0))
    set_seed(seed)
    model = build_model(config, device)
    load_checkpoint(checkpoint, model=model, map_location=device)
    model = model.to(device)
    diffusion = build_diffusion(config)
    dataset = GraphQueryDataset.load(data_path)

    args = SimpleNamespace(
        batch_size=8,
        deterministic=True,
        multi_k=2,
        beam_width=3,
        null_policy="stop",
        strict=True,
    )
    # GoalHit is 99.6%; a small fixed pool is enough while remaining reproducible.
    indices = list(range(8))
    rows = vdp.decode_pool(
        model,
        diffusion,
        [(index, dataset[index]) for index in indices],
        device,
        make_generator(seed, device="cpu"),
        args,
    )
    successful = [row for row in rows if row.get("multi_best_goal")]
    if not successful:
        raise RuntimeError("No strict-beam success in deterministic demo pool")
    row = max(successful, key=lambda item: item["gt_hops"])
    sample = dataset[row["index"]]

    full_graph = vds.load_global_graph(ROOT / "data/didi/graph/chengdu")
    full_pos, geo_stats = vds.geographic_layout(full_graph, ROOT / "data/didi/raw/chengdu/ChengDu.pkl")
    pos_local = vdp.local_positions(sample, full_graph, full_pos)
    if not pos_local:
        raise RuntimeError("Failed to map demo corridor to Chengdu coordinates")
    return sample, row, full_graph, full_pos, pos_local, geo_stats


def crop_limits(pos_local: Dict[int, Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = [point[0] for point in pos_local.values()]
    ys = [point[1] for point in pos_local.values()]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    pad_x = max(dx * 0.09, 1e-5)
    pad_y = max(dy * 0.09, 1e-5)
    return (min(xs) - pad_x, max(xs) + pad_x), (min(ys) - pad_y, max(ys) + pad_y)


def render_demo_frame(
    path: Path,
    sample,
    row: dict,
    full_graph: nx.Graph,
    full_pos: dict,
    pos_local: dict,
    stage: int,
    progress: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    nx.draw_networkx_edges(full_graph, full_pos, ax=ax, edge_color="#D7DDE5", width=0.55, alpha=0.95)
    nx.draw_networkx_edges(sample.graph, pos_local, ax=ax, edge_color="#9EC5E8", width=1.25, alpha=0.95)

    prediction = [int(node) for node in row["multi_best"]]
    gt_path = [int(node) for node in row["gt_path"]]
    if stage >= 1:
        nx.draw_networkx_nodes(sample.graph, pos_local, nodelist=[sample.start], node_color=GREEN, edgecolors="white", linewidths=1.5, node_shape="*", node_size=260, ax=ax)
        nx.draw_networkx_nodes(sample.graph, pos_local, nodelist=[sample.goal], node_color=RED, edgecolors="white", linewidths=1.5, node_shape="*", node_size=260, ax=ax)
    if stage >= 2:
        junctions = [int(node) for node in sample.segments.decision_nodes if node in pos_local]
        nx.draw_networkx_nodes(sample.graph, pos_local, nodelist=junctions, node_color="white", edgecolors="#475569", linewidths=0.55, node_shape="s", node_size=13, ax=ax)
    if stage >= 3:
        alternatives = sorted(row.get("multi_paths") or [], key=lambda item: item["log_prob"], reverse=True)
        shown = 0
        for item in alternatives:
            nodes = [int(node) for node in item["nodes"]]
            if nodes == prediction or item.get("status") != "goal":
                continue
            edges = edge_list(sample.graph, nodes)
            if edges:
                nx.draw_networkx_edges(sample.graph, pos_local, edgelist=edges, edge_color="#CBD5E1", width=1.6, alpha=0.65, ax=ax)
                shown += 1
            if shown >= 6:
                break
        count = max(2, min(len(prediction), int(math.ceil(progress * len(prediction)))))
        nx.draw_networkx_edges(sample.graph, pos_local, edgelist=edge_list(sample.graph, prediction[:count]), edge_color=ORANGE, width=4.6, alpha=0.98, ax=ax)
    if stage >= 4:
        nx.draw_networkx_edges(sample.graph, pos_local, edgelist=edge_list(sample.graph, gt_path), edge_color=BLUE, width=2.7, style=(0, (3.2, 2.0)), alpha=1.0, ax=ax)
        nx.draw_networkx_edges(sample.graph, pos_local, edgelist=edge_list(sample.graph, prediction), edge_color=ORANGE, width=4.6, alpha=0.95, ax=ax)

    labels = [
        ("1 · Road Graph / OD Corridor", "真实成都街道与模型看到的局部图"),
        ("2 · Start / Goal", "绿色起点 · 红色终点"),
        ("3 · Decision Junctions", "只在真实分叉点建立决策变量"),
        ("4 · Strict Beam Search", "概率场中的候选路线竞争，逐步扩展合法前缀"),
        ("5 · Final Route", "蓝色虚线：GT · 橙色实线：Prediction"),
    ]
    title, subtitle = labels[stage]
    fig.text(0.055, 0.93, title, fontsize=22, weight="bold", color=INK)
    fig.text(0.055, 0.885, subtitle, fontsize=12.5, color=MUTED)
    fig.text(0.945, 0.93, "Graph + OD  →  Probability Field  →  Structured Decoder  →  Route", ha="right", fontsize=11.5, color=PURPLE)
    if stage >= 4:
        legend = [
            Line2D([], [], color="#D7DDE5", lw=1.5, label="Road graph"),
            Line2D([], [], color="#9EC5E8", lw=2, label="OD corridor"),
            Line2D([], [], color=BLUE, lw=2.7, ls="--", label="GT"),
            Line2D([], [], color=ORANGE, lw=4, label="Prediction"),
            Line2D([], [], marker="*", color="white", markerfacecolor=GREEN, markersize=12, label="Start"),
            Line2D([], [], marker="*", color="white", markerfacecolor=RED, markersize=12, label="Goal"),
        ]
        ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=6, frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9.5)
    xlim, ylim = crop_limits(pos_local)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(rect=(0.025, 0.04, 0.975, 0.86))
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_demo() -> dict:
    sample, row, full_graph, full_pos, pos_local, geo_stats = decode_demo_case()
    fps = 10
    stages = [(0, 10), (1, 10), (2, 10), (3, 28), (4, 22)]
    with tempfile.TemporaryDirectory(prefix="gjd_interview_demo_") as temp_name:
        temp = Path(temp_name)
        frame_index = 0
        for stage, count in stages:
            for local_index in range(count):
                progress = (local_index + 1) / count
                render_demo_frame(
                    temp / f"frame_{frame_index:04d}.png",
                    sample,
                    row,
                    full_graph,
                    full_pos,
                    pos_local,
                    stage,
                    progress,
                )
                frame_index += 1
        output = OUT_DIR / "demo.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(temp / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)
    return {
        "index": int(row["index"]),
        "gt_hops": int(row["gt_hops"]),
        "pred_hops": len(row["multi_best"]) - 1,
        "num_goal_paths": int(row["num_goal_paths"]),
        "street_nodes": int(full_graph.number_of_nodes()),
        "street_edges": int(full_graph.number_of_edges()),
        "geo_coverage": float(geo_stats["coverage"]),
    }


def write_showcase(city: Dict[str, float], beam: Dict[int, float], demo: dict) -> None:
    content = f"""# Graph-Junction Diffusion

## Page 1｜项目背景与核心思想

真实司机路径不严格等于最短路，同一 OD 下也可能存在多条合理路线。
本项目不直接预测一条节点序列，而是在关键 Junction 上学习不同 Branch 的条件选择概率，形成 Routing Probability Field，再通过 Strict Beam Search 提取完整路线。

![background](01_background.png)

**核心：**

Graph + Start/Goal → Routing Probability Field → Structured Decoder → Route

---

## Page 2｜网络框架

![framework](02_framework.png)

- Branch Decomposition：只在真实分叉点建立决策变量
- Categorical Diffusion：在 Decision Field 上执行扩散与恢复
- Persistent Graph Flow：传播 Start/Goal、当前 Edge State 与 Road Cost
- Branch Scorer：输出每个 Junction 的 Branch Probability
- Strict Beam Search：屏蔽 Loop、Dead-end 等非法扩展并提取到达 Goal 的完整路线

训练目标：

Path NLL + Saturating NULL + Trajectory Set Loss

---

## Page 3｜实验结果

### 跨城市泛化

![cross-city](03_cross_city.png)

成都训练模型无需重训直接迁移到西安，同规模场景下 GoalHit 基本保持（成都 {city['chengdu']:.1%}，西安 Zero-shot {city['xian']:.1%}）。

### 大图搜索预算

![beam](04_beam_ablation.png)

同一个模型仅扩大 Beam，即可将大图场景 GoalHit 从 Beam3 的 {beam[3]:.1%} 恢复至 Beam16 的 {beam[16]:.1%}，说明 Routing Probability Field 中仍保留有效候选。

### Demo

[播放 demo.mp4](demo.mp4)

成都真实案例 #{demo['index']}：真实经纬度街道底图，deterministic + strict top2/beam3；动画依次显示 Road Graph、Start/Goal、Decision Junction、搜索候选与最终 GT/Prediction。

---

## 实验口径与来源

- 成都同分布：`../runs/didi_chengdu_loss_improved/eval_test_1000.json`
- 西安 Zero-shot：`../runs/didi_chengdu_loss_improved/eval_xian_test_1000.json`
- 大图 Beam 消融：`eval_test_1000_long*.json`
- 主评测统一使用 deterministic + strict + top-k 2；跨城市主图使用 beam 3
- Demo 使用成都 `test_1000.pkl`、成都 `graph_global.pkl`、`ChengDu.pkl` 与 `best.pt`，未混用西安坐标

---

## 面试讲法

### Page 1｜30 秒

> 真实司机路线通常不是严格最短路，而且同一个 OD 可以有多条合理路线。所以我没有让 Diffusion 直接生成节点序列，而是让模型在每个关键分叉点预测不同 Branch 的条件概率，形成一张 Routing Probability Field，再通过结构化搜索得到最终路线。

### Page 2｜60 秒

> 首先把连续度为 2 的道路压缩成 Branch，只在真正有选择的 Junction 建立决策变量。真实路径被表示为 Decision Field，并在这个离散场上做 Categorical Diffusion。每一步去噪时，Graph Flow 会结合当前 Edge State、道路长度、Start/Goal 和时间条件传播全局信息，最终输出每个 Junction 的 Branch Probability。最后 Strict Beam Search 负责屏蔽 Loop、Dead-end 等非法扩展并提取完整路径。

### Page 3｜45 秒

> 成都同分布测试 GoalHit 为 {city['chengdu']:.1%}，成都训练出的模型不重新训练直接迁移到西安仍为 {city['xian']:.1%}。在大图场景下，strict beam3 的固定搜索预算下 GoalHit 为 {beam[3]:.1%}，同一个模型把 Beam 扩大到 16 后恢复到 {beam[16]:.1%}。因此大图问题不只是模型会不会判断 Branch，也和长时域组合搜索中的 Beam 剪枝有关。
"""
    (OUT_DIR / "SHOWCASE.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate interview showcase assets")
    parser.add_argument("--skip-demo", action="store_true", help="Generate only figures and Markdown")
    parser.add_argument(
        "--regenerate-framework-code",
        action="store_true",
        help="Overwrite the curated model-generated framework image with the code fallback",
    )
    args = parser.parse_args()

    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    city, beam = validate_sources()
    draw_background()
    framework_path = OUT_DIR / "02_framework.png"
    if args.regenerate_framework_code or not framework_path.exists():
        draw_framework()
    draw_cross_city(city)
    draw_beam(beam)
    demo = {
        "index": -1,
        "gt_hops": 0,
        "pred_hops": 0,
        "num_goal_paths": 0,
        "street_nodes": 0,
        "street_edges": 0,
        "geo_coverage": 0.0,
    }
    if not args.skip_demo:
        demo = draw_demo()
    write_showcase(city, beam, demo)

    print("Interview showcase completed:")
    print("- background diagram")
    print("- framework diagram")
    print("- cross-city metrics")
    print("- beam ablation")
    print("- demo video" if not args.skip_demo else "- demo video (skipped)")
    print("- SHOWCASE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
