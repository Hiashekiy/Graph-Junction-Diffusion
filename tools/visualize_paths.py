"""把模型的预测路径与 GT 路径画在一起（直观看效果）。

用法::

    python tools/visualize_paths.py \
        --run outputs/runs/v2_controlled_100ep_flow3 \
        --data data/controlled_test.pkl \
        --num 8 --out outputs/figures/paths_flow3.png

面板里画什么：

* 浅灰 = 图里所有边（含干扰分支 dead-end / detour / loop）
* **橙色粗实线 = 模型预测路径**（broken / loop 时画到断掉的那一段为止）
* **蓝色虚线 = GT 最短路径**
* 绿色星 = start，红色星 = goal，小方块 = decision node（模型真正要做选择的地方）
* GT 路径上的节点按顺序标了序号，方便顺着看

样本选择（``--select``）：

* ``auto``（默认）：先跑完整个 pool，再按 (难度, 结局) 分桶轮询抽样，尽量覆盖
  "easy/medium/hard × goal/broken/loop"，而不是随机抽 8 个全是同一种情况；
* ``indices``：``--indices 3,17,42`` 指定样本下标，便于复现同一组图；
* ``goal`` / ``loop`` / ``broken``：只看某一类结局；
* ``optimal``：只看预测恰好等于最短路的。

布局：把 GT 路径上的节点固定成一条从左到右的"脊柱"，其余节点用 spring 布局挂在
周围 —— Controlled Junction Graph 本身就是"骨架 + 干扰分支"，这样画出来的图能直接
看出模型在哪一段拐错了。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIFFICULTY_RANK = {"hard": 0, "medium": 1, "easy": 2}
STATUS_RANK = {"goal": 0, "broken": 1, "loop": 2}
# auto 抽样里成功案例的目标占比：模型的真实成功率在 0.7~0.9 之间，全画失败案例会
# 严重误导，全画成功案例又看不出错在哪，所以按 ~6:4 混。
GOAL_FRACTION = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="draw predicted vs ground-truth paths")
    parser.add_argument("--run", required=True, help="run 目录（含 run_config.json）")
    parser.add_argument("--checkpoint", default=None, help="默认 <run>/best.pt")
    parser.add_argument("--data", default="data/controlled_test.pkl")
    parser.add_argument("--num", type=int, default=8, help="画几张")
    parser.add_argument(
        "--cols", type=int, default=2,
        help="每行几张。默认为 2：脊柱+梳子的图偏宽，两列才看得清路径",
    )
    parser.add_argument(
        "--select",
        default="auto",
        help="auto | random | optimal | goal | loop | broken | indices | all",
    )
    parser.add_argument("--indices", default=None, help="逗号分隔的样本下标（--select indices）")
    parser.add_argument(
        "--select-seed",
        type=int,
        default=None,
        help="--select random 用的抽样种子；不给就用系统熵（每次不同），"
             "给了就固定下来便于复现同一组图",
    )
    parser.add_argument("--out", default="outputs/figures/paths.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--deterministic", action="store_true", help="用 posterior argmax 采样")
    parser.add_argument(
        "--layout",
        default="spring",
        choices=["spring", "forceatlas2", "kamada_kawai", "spine"],
        help="图布局。默认 spring（正常图结构，等比例不拉伸）；spine 会把 GT 压成一条"
             "直线，只适合盯着单条路径看，会扭曲真实结构",
    )
    parser.add_argument("--labels", action="store_true", help="在 GT 路径节点上标顺序号")
    parser.add_argument(
        "--only-difficulty", default=None,
        help="只在某个难度里抽（easy/medium/hard，逗号分隔）",
    )
    parser.add_argument(
        "--only-mode", default=None,
        help="只在某个结构模式里抽（branch_heavy/long_chain/loop_detour，逗号分隔）",
    )
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--title", default=None)
    parser.add_argument("--json-out", default=None, help="把每张图的路径写成 json")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 解码
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_pool(model, diffusion, dataset, device, generator, args) -> List[Dict[str, Any]]:
    """在整个 pool 上解码，返回每个样本的预测路径与判定。"""
    import networkx as nx

    from src.data.collate import collate_samples
    from src.diffusion.sampler import sample_reverse_chain
    from src.evaluation.path_decoder import (
        candidate_offsets,
        decision_offsets,
        decode_flat,
    )

    model.eval()
    results: List[Dict[str, Any]] = []
    samples = list(dataset)
    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start : start + args.batch_size]
        batch = collate_samples(chunk, device=device)
        chain = sample_reverse_chain(
            diffusion,
            model,
            batch,
            generator=generator,
            stochastic=not args.deterministic,
        )
        z0 = chain["z0"]
        offsets = decision_offsets(chunk)
        candidate_starts = candidate_offsets(chunk)
        for index, sample in enumerate(chunk):
            result = decode_flat(
                sample,
                z0,
                decision_offset=offsets[index],
                candidate_offset=candidate_starts[index],
            )
            optimal_cost = float(
                nx.shortest_path_length(sample.graph, sample.start, sample.goal)
            )
            pred_cost = float(len(result.path) - 1)
            results.append(
                {
                    "index": start + index,
                    "status": result.status,
                    "reason": result.reason,
                    "path": list(result.path),
                    "pred_hops": pred_cost,
                    "optimal_hops": optimal_cost,
                    "optimal": result.status == "goal" and abs(pred_cost - optimal_cost) < 1e-9,
                    "difficulty": sample.meta.get("difficulty", "n/a"),
                    "mode": sample.meta.get("mode", "n/a"),
                    "decisions": int(sample.num_decisions),
                    "nodes": int(sample.num_nodes),
                    "graph_id": sample.meta.get("graph_id"),
                }
            )
    return results


def apply_filters(results: Sequence[Dict[str, Any]], args) -> List[int]:
    """按 --only-difficulty / --only-mode 缩小候选池（返回 pool 内位置）。"""
    allowed = list(range(len(results)))
    if args.only_difficulty:
        wanted = {part.strip() for part in args.only_difficulty.split(",") if part.strip()}
        allowed = [i for i in allowed if results[i]["difficulty"] in wanted]
    if args.only_mode:
        wanted = {part.strip() for part in args.only_mode.split(",") if part.strip()}
        allowed = [i for i in allowed if results[i]["mode"] in wanted]
    return allowed


def resolve_select_seed(args) -> Optional[int]:
    """``--select random`` 用哪个抽样种子。

    不给 ``--select-seed`` 就用系统熵（每次跑都不一样，这是"随机看样本"想要的），
    并把实际用的数字打印出来 —— 想复现同一组图时把它填回 ``--select-seed`` 即可。
    其它 ``--select`` 模式是确定性的，返回 None。
    """
    if args.select != "random":
        return None
    if args.select_seed is not None:
        return int(args.select_seed)
    return int.from_bytes(os.urandom(4), "little")


def select_indices(results: Sequence[Dict[str, Any]], args) -> List[int]:
    """从解码结果里挑要画的下标（返回 pool 内的位置）。"""
    allowed = apply_filters(results, args)
    if not allowed:
        raise SystemExit(
            f"filters removed every sample "
            f"(--only-difficulty {args.only_difficulty}, --only-mode {args.only_mode})"
        )
    allowed_set = set(allowed)
    if args.select == "indices":
        if not args.indices:
            raise SystemExit("--select indices 需要 --indices 3,17,42")
        wanted = [int(part) for part in args.indices.split(",") if part.strip()]
        by_index = {row["index"]: position for position, row in enumerate(results)}
        missing = [index for index in wanted if index not in by_index]
        if missing:
            raise SystemExit(f"these sample indices are not in the pool: {missing}")
        positions = [by_index[index] for index in wanted]
        filtered = [p for p in positions if p not in allowed_set]
        if filtered:
            raise SystemExit(
                f"these samples are excluded by the --only-* filters: "
                f"{[results[p]['index'] for p in filtered]}"
            )
        return positions

    if args.select == "all":
        return allowed
    if args.select == "random":
        import random as _random

        seed = getattr(args, "resolved_select_seed", None)
        rng = _random.Random(seed)
        count = min(int(args.num), len(allowed))
        return rng.sample(allowed, count)
    if args.select == "optimal":
        return [i for i in allowed if results[i]["optimal"]]
    if args.select in ("goal", "loop", "broken"):
        return [i for i in allowed if results[i]["status"] == args.select]

    # auto：先按 (难度, 结局) 分桶，再按"成功率真实分布"混着抽 —— 优先抽到 goal 的
    # 桶（覆盖 hard/medium/easy），剩余名额给失败案例（broken / loop），这样面板里既有
    # 正确行为也有典型错误。同桶内挑决策数居中的，避免全挑最极端的长样本。
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for position in allowed:
        row = results[position]
        buckets.setdefault((row["difficulty"], row["status"]), []).append(position)
    for key, positions in buckets.items():
        positions.sort(key=lambda p: (results[p]["decisions"], results[p]["pred_hops"]))

    def order(keys):
        return sorted(keys, key=lambda key: (DIFFICULTY_RANK.get(key[0], 9), key[1]))

    goal_keys = order([key for key in buckets if key[1] == "goal"])
    fail_keys = order([key for key in buckets if key[1] != "goal"])

    chosen: List[int] = []
    seen = set()

    def take(keys, quota):
        round_index = 0
        while len(chosen) < quota and round_index < 20:
            progressed = False
            for key in keys:
                positions = buckets[key]
                if round_index < len(positions):
                    candidate = positions[round_index]
                    if candidate not in seen:
                        chosen.append(candidate)
                        seen.add(candidate)
                        progressed = True
                        if len(chosen) >= quota:
                            break
            if not progressed:
                break
            round_index += 1

    take(goal_keys, int(round(args.num * GOAL_FRACTION)))
    take(fail_keys, args.num)
    if len(chosen) < args.num:  # 失败案例不够就用更多成功案例补齐
        take(goal_keys, args.num)
    return chosen[: args.num]


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------
def layout_graph(graph, gt_path: Sequence[int], seed: int = 0, method: str = "spring"):
    """给整张图算坐标（**正常图布局**，不把 GT 压成一条线）。

    ``method``：

    * ``spring``（默认）：经典 Fruchterman-Reingold，正常图结构、各向同性，
      面板里用等比例坐标画，不会拉伸变形；
    * ``forceatlas2``：networkx 的 ForceAtlas2，链状分支多的图会被拉成长条；
    * ``kamada_kawai``：距离保持布局，形状最"正"，节点多时稍慢；
    * ``spine``：把 GT 路径摊成一条水平脊柱、干扰分支垂直伸出。**只用于"盯一条具体
      路径怎么走"**，会严重扭曲图的真实结构，所以不是默认值。

    所有力导向布局都用 ``seed`` 固定，保证同一张图每次画出来一样。
    """
    import networkx as nx

    if method == "spine":
        return _spine_layout(graph, gt_path, seed)
    if method == "spring":
        return nx.spring_layout(graph, seed=seed, iterations=300, k=1.1)
    if method == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if method == "forceatlas2":
        return nx.forceatlas2_layout(graph, seed=seed, max_iter=400)
    raise SystemExit(f"unknown --layout {method!r}")


def _spine_layout(graph, gt_path: Sequence[int], seed: int = 0):
    """（非默认）把 GT 路径摊成水平脊柱，干扰分支垂直伸出。会扭曲真实结构。"""
    from collections import deque

    spine = list(dict.fromkeys(int(node) for node in gt_path))
    pos: Dict[int, Tuple[float, float]] = {
        node: (float(index), 0.0) for index, node in enumerate(spine)
    }
    known = set(spine)

    depth_y = 0.62
    branch_dx = 0.34
    for spine_index, node in enumerate(spine):
        branches = [n for n in graph.neighbors(node) if n not in known]
        count = len(branches)
        if not count:
            continue
        branches.sort()
        for order, start in enumerate(branches):
            # 左右交错 + 在 junction 附近水平展开，避免多条分支叠在一条竖线上
            side = 1.0 if order % 2 == 0 else -1.0
            offset = (order - (count - 1) / 2.0) * branch_dx
            queue = deque([(start, node, (spine_index + offset, 0.0), 1, side)])
            known.add(start)
            while queue:
                current, _parent, (px, py), step, sign = queue.popleft()
                pos[current] = (px, py + sign * depth_y)
                # 分支内部的继续延伸：同一条分支保持同一个 x
                for nxt in graph.neighbors(current):
                    if nxt not in known:
                        known.add(nxt)
                        queue.append((nxt, current, (px, py + sign * depth_y), step + 1, sign))

    # 兜底：环状干扰（loop distractor）可能让个别节点没被 BFS 覆盖
    leftover = [n for n in graph.nodes if n not in pos]
    if leftover:
        rng = np.random.default_rng(seed)
        for node in leftover:
            pos[node] = (float(rng.integers(0, max(len(spine), 1))), float(rng.normal(0, 1.2)))
    return pos


def draw_panel(ax, sample, row: Dict[str, Any], args, seed: int, show_legend: bool) -> None:
    import networkx as nx

    graph = sample.graph
    pos = layout_graph(graph, sample.gt_path, seed=seed + row["index"], method=args.layout)

    def edge_list(path: Sequence[int]) -> List[Tuple[int, int]]:
        return [(int(u), int(v)) for u, v in zip(path[:-1], path[1:]) if graph.has_edge(u, v)]

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#d9d9d9", width=0.9)
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=14, node_color="#bdbdbd", linewidths=0
    )

    predicted = edge_list(row["path"])
    ground_truth = edge_list(sample.gt_path)
    # 画法约定：**预测路径是橙色粗实线**（它是这张图的主角），**GT 最短路是蓝色细虚线**
    # 盖在上面。两者一致时看到"橙底蓝虚线"；分岔之后橙色继续走模型选的路、蓝色虚线
    # 继续走 GT，两条线分开，一眼能看出模型在哪一跳拐错了。
    if predicted:
        nx.draw_networkx_edges(
            graph, pos, edgelist=predicted, ax=ax, edge_color="#ff8c00",
            width=5.0, alpha=0.9,
        )
    if ground_truth:
        nx.draw_networkx_edges(
            graph, pos, edgelist=ground_truth, ax=ax, edge_color="#1f4fd8",
            width=2.4, style=(0, (3, 2.4)), alpha=1.0,
        )

    decision_nodes = list(getattr(sample.segments, "decision_nodes", []) or [])
    if decision_nodes:
        nx.draw_networkx_nodes(
            graph, pos, nodelist=decision_nodes, ax=ax, node_size=26,
            node_color="#ffffff", edgecolors="#333333", linewidths=0.8, node_shape="s",
        )

    nx.draw_networkx_nodes(
        graph, pos, nodelist=[sample.start], ax=ax, node_size=210,
        node_color="#2ca02c", edgecolors="black", linewidths=0.6, node_shape="*",
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=[sample.goal], ax=ax, node_size=210,
        node_color="#d62728", edgecolors="black", linewidths=0.6, node_shape="*",
    )

    # 分岔点：预测与 GT 最后一个相同的节点。看错误案例时直接盯这个点。
    gt_path = [int(node) for node in sample.gt_path]
    pred_path = [int(node) for node in row["path"]]
    common = 0
    for node_gt, node_pred in zip(gt_path, pred_path):
        if node_gt != node_pred:
            break
        common += 1
    diverged = common < len(pred_path) and pred_path != gt_path
    if diverged and common >= 1:
        ax.plot(*pos[gt_path[common - 1]], marker="X", markersize=9, color="#8b0000", zorder=6)

    if args.labels:
        labels = {int(node): str(order + 1) for order, node in enumerate(sample.gt_path)}
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=5.5)

    status_text = {
        "goal": "到达 goal",
        "loop": "走进环（重复节点）",
        "broken": "断掉（选到 NULL / 死路）",
    }.get(row["status"], row["status"])
    same_as_gt = pred_path == gt_path
    if row["optimal"] and same_as_gt:
        extra = " · 与标注 GT 完全一致"
    elif row["optimal"]:
        # 图里可能有多条等长的最短路（实测 sample 170 有 2 条），模型走了另一条：
        # 代价一样，但和标注的 GT 不是同一条序列。
        extra = " · 同代价最优（与标注 GT 是另一条等长路）"
    elif row["status"] == "goal":
        extra = f" · 比 GT 多 {int(row['pred_hops']) - int(row['optimal_hops'])} 跳"
    else:
        extra = ""
    if not same_as_gt and common >= 1:
        extra += f" · 第 {common} 跳与 GT 分开（X）"
    ax.set_title(
        f"#{row['index']}  {row['difficulty']}/{row['mode']}\n"
        f"GT {int(row['optimal_hops'])} 跳 / {row['decisions']} 决策 · "
        f"预测 {int(row['pred_hops'])} 跳 · {status_text}{extra}",
        fontsize=9,
    )
    # 等比例：布局坐标是各向同性的，拉成椭圆会让人误判"哪条路更近"
    ax.set_aspect("equal")
    ax.set_axis_off()
    if show_legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], color="#ff8c00", lw=5.0, label="模型预测路径（橙实线）"),
            Line2D([], [], color="#1f4fd8", lw=2.4, linestyle=(0, (3, 2.4)),
                   label="GT 最短路（蓝虚线）"),
            Line2D([], [], color="#dcdcdc", lw=1.0, label="干扰分支"),
            Line2D([], [], color="#2ca02c", marker="*", lw=0, markersize=12, label="start"),
            Line2D([], [], color="#d62728", marker="*", lw=0, markersize=12, label="goal"),
            Line2D([], [], color="#333333", marker="s", lw=0, markersize=7,
                   markerfacecolor="white", label="decision node（要做选择的地方）"),
            Line2D([], [], color="#8b0000", marker="X", lw=0, markersize=9,
                   label="分岔点（预测与 GT 最后一次相同的位置）"),
        ]
        return handles
    return None


def main() -> int:
    args = parse_args()

    from src.data.dataset import GraphQueryDataset
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_diffusion, build_model, get_device
    from src.utils.config import load_config
    from src.utils.seed import make_generator, set_seed

    run_dir = Path(args.run)
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise SystemExit(f"missing {config_path}")
    config = load_config(config_path)
    checkpoint = args.checkpoint or str(run_dir / "best.pt")
    device = get_device(args.device or str(config.get("training.device", "auto")))
    seed = int(config.get("seed", args.seed))

    set_seed(seed)
    model = build_model(config, device)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    model = model.to(device)
    diffusion = build_diffusion(config)
    dataset = GraphQueryDataset.load(args.data)

    print(f"checkpoint : {checkpoint} (epoch={payload.get('epoch')})")
    print(f"model      : {model.flow_steps_label}")
    print(f"data       : {args.data} ({len(dataset)} queries)")
    print(f"sample seed: {seed}（推理采样；改 --seed 会改变 pool 结果）")

    results = decode_pool(
        model, diffusion, dataset, device, make_generator(seed, device="cpu"), args
    )
    hits = sum(1 for row in results if row["status"] == "goal")
    print(
        f"pool result: goal={hits}/{len(results)}  "
        f"loop={sum(1 for r in results if r['status'] == 'loop')}  "
        f"broken={sum(1 for r in results if r['status'] == 'broken')}"
    )

    args.resolved_select_seed = resolve_select_seed(args)
    positions = select_indices(results, args)
    if not positions:
        raise SystemExit("no sample matched the selection; try --select auto")
    if args.select == "random":
        print(
            f"select     : random  选样种子 select_seed={args.resolved_select_seed}"
            f"（想复现这一组就加 --select-seed {args.resolved_select_seed}）"
        )
    else:
        print(f"select     : {args.select}（确定性规则）")
    print(f"selected   : {[results[p]['index'] for p in positions]}")

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

    cols = min(args.cols, len(positions))
    rows = int(np.ceil(len(positions) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    table: List[Dict[str, Any]] = []
    legend_handles = None
    for panel, position in enumerate(positions):
        row = results[position]
        sample = dataset[row["index"]]
        handles = draw_panel(
            axes[panel], sample, row, args, seed, show_legend=(panel == 0)
        )
        if handles:
            legend_handles = handles
        table.append(
            {
                **{key: row[key] for key in ("index", "difficulty", "mode", "decisions",
                                             "optimal_hops", "pred_hops", "status",
                                             "optimal", "reason")},
                "path": row["path"],
                "gt_path": list(sample.gt_path),
            }
        )
    for spare in range(len(positions), len(axes)):
        axes[spare].set_axis_off()

    select_note = (
        f"select=random(select_seed={args.resolved_select_seed})"
        if args.select == "random"
        else f"select={args.select}"
    )
    title = args.title or (
        f"{run_dir.name}  ·  {model.flow_steps_label}  ·  "
        f"Checkpoint epoch {payload.get('epoch')}  ·  {Path(args.data).name}  ·  "
        f"{select_note}  ·  sample_seed={seed}"
    )
    fig.suptitle(title, fontsize=13, y=0.995)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=8.5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"\nfigure     : {out_path}")

    header = (
        f"{'idx':>4}  {'difficulty':<10} {'mode':<12} {'dec':>3} {'gt_hops':>7} "
        f"{'pred_hops':>9}  {'status':<7} {'optimal':<7} note"
    )
    lines = [title, "", header, "-" * len(header)]
    for row in table:
        lines.append(
            f"{row['index']:>4}  {row['difficulty']:<10} {row['mode']:<12} "
            f"{row['decisions']:>3} {int(row['optimal_hops']):>7} {int(row['pred_hops']):>9}  "
            f"{row['status']:<7} {'yes' if row['optimal'] else 'no':<7} {row['reason']}"
        )
    report = "\n".join(lines)
    print(report)
    text_path = out_path.with_suffix(out_path.suffix + ".txt")
    text_path.write_text(report + "\n", encoding="utf-8")
    print(f"\ntable      : {text_path}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(table, handle, indent=1, ensure_ascii=False)
        print(f"written    : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
