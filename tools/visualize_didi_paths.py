"""把训练好的模型在 **DiDi 成都真实数据** 上跑一遍，画出"路径规划图"。

和 :mod:`tools.visualize_paths.py` 的关系：

    visualize_paths.py       controlled junction graph，**spring 拓扑布局**，
                             图里没有地理含义（"街道背景"是拓扑骨架）
    visualize_didi_paths.py  DiDi 真实路网，**真实经纬度布局**，底图就是成都
                             的街道（2891 节点 / 4403 条边），GT 是真实司机
                             历史路径而不是 Dijkstra

面板里画什么：

    浅灰细线   整张成都路网 = **街道背景**（真实经纬度，OSMnx epsg:4326）
    浅蓝细线   该样本的 OD corridor 子图（模型真正看到的图）
    蓝色粗虚线 GT —— **真实司机走过的历史路径**（不是最短路）
    橙色粗实线 **模型最终选中的那条路径**
    绿星/红星  start / goal
    白色方块   decision node（模型要在这里做选择）

两种解码各出一张图（``--mode both``，默认）：

    分支解码（single）    posterior argmax 单路径解码：一个 decision 只留一条
                          branch，走到底就是"模型最终选中的路径"
    多分支解码（multi）   存活路径表解码：每个 decision 保留 top-k 条 branch，
                          整张表画出来 —— 粗橙线是**累计 log 概率最高的那条**
                          （也就是主口径的"最终选中路径"），第 2~5 名上不同颜色
                          并在"与主路径的分叉点"标同色圆圈编号，其余淡灰细线。
                          标题给出 `路径表 N 条（到终点 M 条）· coverage ✓/✗`

用法::

    python tools/visualize_didi_paths.py --run outputs/runs/didi_chengdu_flow1_weighted
    python tools/visualize_didi_paths.py --run <run> --num 6 --select goal
    python tools/visualize_didi_paths.py --run <run> --mode multi --multi-k 3
    python tools/visualize_didi_paths.py --run <run> --checkpoint last.pt --deterministic

坐标来源与 ``tools/visualize_didi_samples.py`` 完全一致（同一个
``geographic_layout``），所以两张图的形状可以直接对照。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 复用数据集可视化那份地理布局 / 底图加载代码，保证两张图的形状一模一样
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import visualize_didi_samples as vds  # noqa: E402

#: 路径表按名次上色：第 1 名是粗橙线（最终选中），2..N 名用这些颜色
_RANK_COLORS = [
    "#ff8c00",  # 1（主路径，粗线）
    "#1f77b4",  # 2
    "#2ca02c",  # 3
    "#d62728",  # 4
    "#9467bd",  # 5
    "#8c564b",
    "#e377c2",
    "#17becf",
]

COLOR_STREET = "#c4c4c4"
COLOR_CORRIDOR = "#7fb2dd"
COLOR_GT = "#1f4fd8"
COLOR_PRED = "#ff8c00"
COLOR_ALT = "#c9c9c9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="draw DiDi path-planning figures (street background + chosen path)"
    )
    parser.add_argument("--run", required=True, help="run 目录（含 run_config.json）")
    parser.add_argument("--checkpoint", default=None, help="默认 <run>/best.pt")
    parser.add_argument(
        "--data", default=None,
        help="默认取 run_config 里的 paths.data_dir + test_1000.pkl",
    )
    parser.add_argument("--coords", default=vds.DEFAULT_COORDS)
    parser.add_argument("--num", type=int, default=6, help="画几张")
    parser.add_argument("--cols", type=int, default=2, help="每行几张")
    parser.add_argument(
        "--select", default="mixed",
        choices=["mixed", "random", "indices", "goal", "broken", "loop", "all"],
        help="mixed（默认）：按 --goal-fraction 混成功/失败案例（只看成功会严重高估模型）；"
             "random：无偏随机；indices / goal / broken / loop / all",
    )
    parser.add_argument(
        "--goal-fraction", type=float, default=0.6,
        help="--select mixed 里成功案例占多少（0.6 = 6 张图里 4 张到达 goal、2 张失败）",
    )
    parser.add_argument("--indices", default=None, help="逗号分隔（--select indices）")
    parser.add_argument(
        "--pool-limit", type=int, default=0,
        help=">0 时只解码前 N 条（冒烟用；默认解码整个 test split）",
    )
    parser.add_argument(
        "--select-seed", type=int, default=20260915,
        help="--select random 的抽样种子（固定下来便于复现同一组图）",
    )
    parser.add_argument(
        "--mode", default="both", choices=["single", "multi", "both"],
        help="single=分支解码；multi=多分支（存活路径表）解码；both=各出一张图",
    )
    parser.add_argument("--multi-k", type=int, default=2, help="多分支每个 decision 留 top-k")
    parser.add_argument("--beam-width", type=int, default=64, help="存活路径表上限")
    parser.add_argument(
        "--null-policy", default="stop", choices=["stop", "skip"],
        help="stop：NULL 参与排名、选中即终止；skip：NULL 不停",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="多分支改用**三池**语义（src.evaluation.strict_beam_decoder）：NULL / loop / "
             "dead-end 在 top-k 之前 mask 掉、失败路径淘汰，路径表里只剩完整到 Goal 的"
             "路径。strict 下 --null-policy 失效。",
    )
    parser.add_argument("--multi-max-draw", type=int, default=40, help="最多画多少条细线")
    parser.add_argument("--multi-highlight", type=int, default=4, help="前 N 名上不同颜色")
    parser.add_argument(
        "--multi-goal-only", action="store_true",
        help="多分支图里只画**到终点**的备选路径（连第 2~5 名也只保留到终点的）",
    )
    parser.add_argument("--out-prefix", default="outputs/figures/didi_paths")
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="posterior argmax rollout（默认是随机采样链 + 最终 argmax readout）",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="推理采样种子")
    parser.add_argument(
        "--no-crop", dest="crop", action="store_false",
        help="不要裁到样本范围（看它在整座城市里的位置；细节会糊）",
    )
    parser.add_argument(
        "--no-street", dest="street", action="store_false",
        help="不画整城街道底图（只剩 corridor，画得快但看不出街道背景）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 数据 / 坐标
# ---------------------------------------------------------------------------
def resolve_data_path(args, config) -> Path:
    if args.data:
        path = Path(args.data)
        return path if path.is_absolute() else PROJECT_ROOT / path
    data_dir = PROJECT_ROOT / str(config.get("paths.data_dir", "data/didi_chengdu_gjd"))
    for name in ("test_1000.pkl", "test.pkl"):
        if (data_dir / name).exists():
            return data_dir / name
    raise SystemExit(f"no test split under {data_dir}")


def local_positions(
    sample, full_graph, full_pos: Dict
) -> Optional[Dict[int, Tuple[float, float]]]:
    """局部编号 -> 地理坐标。

    首选 ``meta['local_to_global']``（``build_sample_from_observed_path`` 写的权威
    反查表）；缺了就退回 ``visualize_didi_samples`` 那套"重新算一遍 corridor"的兜底
    （corridor_node_set 只用 OD + rho，不含 GT，所以没有泄漏问题）。
    """
    mapping = sample.meta.get("local_to_global")
    if mapping and len(mapping) >= sample.num_nodes:
        pos = {index: full_pos.get(int(mapping[index])) for index in range(sample.num_nodes)}
        if all(value is not None for value in pos.values()):
            return pos

    from src.data import didi_dataset as didi

    try:
        keep, _cost = didi.corridor_node_set(
            full_graph,
            int(sample.meta["start_node"]),
            int(sample.meta["goal_node"]),
            float(sample.meta["rho"]),
        )
        ids = sorted(keep)
        if len(ids) != sample.num_nodes:
            return None
        pos = {index: full_pos.get(ids[index]) for index in range(len(ids))}
        return pos if all(value is not None for value in pos.values()) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
def path_cost(graph, path: Sequence[int]) -> Tuple[float, int]:
    """返回 ``(weighted cost, missing_hops)``。

    ``missing_hops`` 是"在 graph 里不存在的边"的条数 —— 正常情况下应该是 0。
    它可以不为 0 的唯一原因是解码出的路径不是一条合法游走；那时不能默默把这条边
    当成 0 米，否则会**低估**预测路径的代价、把它算得比 GT 还短。
    """
    total = 0.0
    missing = 0
    for u, v in zip(path[:-1], path[1:]):
        if graph.has_edge(u, v):
            total += float(graph[u][v].get("weight", 1.0))
        else:
            missing += 1
    return total, missing


@torch.no_grad()
def decode_pool(model, diffusion, pairs, device, generator, args) -> List[Dict[str, Any]]:
    """对 ``pairs``（``(全局下标, sample)`` 列表）逐 batch 跑 reverse chain 并解码。

    ``--select random / indices`` 时调用方**只传要画的那几条**，所以这里的 "pool"
    可能只有 6 条样本 —— 选择规则不需要看全量结果，就没必要解码 1000 条。
    """
    from src.data.collate import collate_samples
    from src.diffusion.sampler import sample_reverse_chain
    from src.evaluation.multi_path_decoder import decode_multi_path
    from src.evaluation.path_decoder import candidate_offsets, decision_offsets, decode_flat
    from src.evaluation.readout import single_path_state

    model.eval()
    results: List[Dict[str, Any]] = []
    for start in range(0, len(pairs), args.batch_size):
        chunk_pairs = pairs[start : start + args.batch_size]
        chunk = [sample for _index, sample in chunk_pairs]
        batch = collate_samples(chunk, device=device)
        chain = sample_reverse_chain(
            diffusion, model, batch, generator=generator,
            stochastic=not args.deterministic,
        )
        z0 = chain["z0"] if args.deterministic else single_path_state(chain, batch, "single")
        offsets = decision_offsets(chunk)
        candidate_starts = candidate_offsets(chunk)

        for index, sample in enumerate(chunk):
            global_index = int(chunk_pairs[index][0])
            result = decode_flat(
                sample, z0,
                decision_offset=offsets[index],
                candidate_offset=candidate_starts[index],
            )
            local = chain["candidate_prob"][
                candidate_starts[index] : candidate_starts[index] + sample.num_candidates
            ]
            multi = decode_multi_path(
                sample, local, top_k=args.multi_k, beam_width=args.beam_width,
                null_policy=args.null_policy, strict=bool(args.strict),
            )
            multi_best = None
            if multi.best is not None:
                multi_best = list(multi.best.nodes)
            gt_path = [int(node) for node in sample.gt_path]
            pred_path = [int(node) for node in result.path]
            pred_cost, missing_hops = path_cost(sample.graph, pred_path)
            gt_cost, _ = path_cost(sample.graph, gt_path)
            # 走廊里的**加权最短路**。GT 是真实司机路径、不是最短路，所以"预测比 GT 便宜"
            # 本身不说明模型错了 —— 必须同时看 GT/最短：如果 GT/最短 ≈ 2，那是司机绕了
            # 远路，模型只是走了更近的路。
            optimal_cost = float("nan")
            try:
                optimal_path = nx.shortest_path(
                    sample.graph, sample.start, sample.goal, weight="weight"
                )
                optimal_cost, _ = path_cost(sample.graph, optimal_path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            results.append(
                {
                    "index": global_index,
                    "status": result.status,
                    "reason": result.reason,
                    "path": pred_path,
                    "gt_path": gt_path,
                    "pred_hops": len(pred_path) - 1,
                    "gt_hops": len(gt_path) - 1,
                    "pred_cost": pred_cost,
                    "gt_cost": gt_cost,
                    "optimal_cost": optimal_cost,
                    "missing_hops": missing_hops,
                    "goal_hit": result.status == "goal",
                    "multi_best": multi_best,
                    "multi_best_goal": (
                        multi_best is not None and multi_best[-1] == int(sample.goal)
                    ),
                    "num_paths": len(multi.finished),
                    "num_goal_paths": len(multi.goal_paths),
                    "coverage": bool(multi.coverage),
                    "best_goal_hops": multi.best_goal_cost,
                    "multi_paths": [
                        {
                            "nodes": [int(v) for v in path.nodes],
                            "status": path.status,
                            "log_prob": float(path.log_prob),
                            "hops": int(path.cost),
                        }
                        for path in multi.finished
                    ],
                    "decisions": int(sample.num_decisions),
                    "nodes": int(sample.num_nodes),
                    "candidates": int(sample.num_candidates),
                }
            )
        print(
            f"  decoded {min(start + args.batch_size, len(pairs))}/{len(pairs)}",
            end="\r", flush=True,
        )
    print()
    return results


def plan_indices(total: int, args) -> Optional[List[int]]:
    """``random`` / ``indices`` 两条规则**不需要看模型输出**，可以在解码前就定下来。

    返回 ``None`` 表示这条选择规则依赖全量解码结果（``goal`` / ``broken`` / ``loop``
    / ``all``），必须先解码整个 pool。返回列表时调用方只解码这几条 —— 6 张图就不必
    跑 1000 条样本的 reverse chain。
    """
    if args.select == "random":
        rng = np.random.default_rng(int(args.select_seed))
        count = min(int(args.num), total)
        return sorted(rng.permutation(total)[:count].tolist())
    if args.select == "indices":
        if not args.indices:
            raise SystemExit("--select indices 需要 --indices 3,17,42")
        wanted = [int(part) for part in args.indices.split(",") if part.strip()]
        bad = [index for index in wanted if not 0 <= index < total]
        if bad:
            raise SystemExit(f"these sample indices are out of range: {bad}")
        return wanted
    return None


def print_cost_summary(results: Sequence[Dict[str, Any]]) -> None:
    """回答"预测怎么比 GT 便宜那么多"这个问题。

    GT 是真实司机路径，**不是**最短路（方案第 5.2 节），所以

        pred/GT < 1  不等于"模型错了"，可能只是司机绕了远路；
        GT/最短      才是"这条 GT 到底绕了多少"的直接度量。

    两个比例一起看才判得出来：如果 GT/最短 ≈ 2 同时 pred/最短 ≈ 1，那是**司机绕路、
    模型走近路**；如果 pred/最短 > 1.5，那才是模型自己走歪了。
    """
    import statistics as st

    def med(values):
        values = [v for v in values if v == v]
        return st.median(values) if values else float("nan")

    pred_gt = [r["pred_cost"] / r["gt_cost"] for r in results if r["gt_cost"]]
    gt_opt = [r["gt_cost"] / r["optimal_cost"] for r in results if r.get("optimal_cost")]
    # 只有**到达 goal**的路径才谈得上"预测/最短"：没到 goal 的只有前半段，
    # 除以"起点到 goal 的最短路"会得到一个 <1 的荒谬比例。
    pred_opt = [
        r["pred_cost"] / r["optimal_cost"]
        for r in results
        if r.get("optimal_cost") and r["status"] == "goal"
    ]
    cheaper = sum(1 for value in pred_gt if value < 1.0)
    print(
        f"cost check : 预测/GT 中位数 {med(pred_gt):.3f}"
        f"（{cheaper}/{len(pred_gt)} 条比 GT 便宜）"
    )
    print(
        f"             GT/走廊最短路 中位数 {med(gt_opt):.3f}（GT 本身就不是最短路）"
    )
    print(
        f"             预测/走廊最短路 中位数 {med(pred_opt):.3f}"
        f"（{sum(1 for v in pred_opt if v < 1.05)}/{len(pred_opt)} 条到达 goal 的路径"
        f" ≤ 最短路的 1.05 倍 —— 模型基本在走最短路）"
    )


def print_multi_summary(results: Sequence[Dict[str, Any]], args) -> None:
    """多分支解码的池级小结。

    关键的一列是 **coverage 但主路径失败**：说明"能到终点的正确分支确实活在路径表里、
    只是没被排到第 1 名"。这一列大 = 该调 readout / 概率校准；这一列 ≈ 0 而 coverage
    也低 = beam 或 top_k 太小，正确分支在分叉那一刻就被剪掉了。
    """
    import statistics as st

    def med(values):
        values = [v for v in values if v is not None]
        return st.median(values) if values else float("nan")

    covered = [r for r in results if r.get("coverage")]
    best_goal = [r for r in results if r.get("multi_best_goal")]
    miss = [r for r in results if r.get("coverage") and not r.get("multi_best_goal")]
    print(
        f"multi      : top_k={args.multi_k} beam={args.beam_width} "
        f"null={args.null_policy}  ·  "
        f"coverage(表里至少一条到终点)={len(covered)}/{len(results)}  ·  "
        f"主路径到终点={len(best_goal)}/{len(results)}"
    )
    print(
        f"             主路径失败但表里有正确分支={len(miss)}/{len(results)}"
        f"（这些属于「排序没排对」，不是「压根没生成」）  ·  "
        f"路径表条数中位数 {med([r.get('num_paths') for r in results]):.0f}  ·  "
        f"到终点条数中位数 {med([r.get('num_goal_paths') for r in results]):.0f}"
    )


def select_positions(results: Sequence[Dict[str, Any]], args) -> List[int]:
    if args.select == "indices":
        if not args.indices:
            raise SystemExit("--select indices 需要 --indices 3,17,42")
        by_index = {row["index"]: position for position, row in enumerate(results)}
        wanted = [int(part) for part in args.indices.split(",") if part.strip()]
        missing = [index for index in wanted if index not in by_index]
        if missing:
            raise SystemExit(f"these sample indices are not in the pool: {missing}")
        return [by_index[index] for index in wanted]
    if args.select == "all":
        return list(range(len(results)))
    if args.select == "random":
        rng = np.random.default_rng(int(args.select_seed))
        count = min(int(args.num), len(results))
        return sorted(rng.permutation(len(results))[:count].tolist())
    rng = np.random.default_rng(int(args.select_seed))
    if args.select == "mixed":
        # 只看成功案例会严重高估模型（这个 run 的 val goal_hit 是 0.686，不是 1.0），
        # 只看失败案例又看不出它对在哪 —— 按 --goal-fraction 混着抽。
        goals = [i for i, row in enumerate(results) if row["goal_hit"]]
        fails = [i for i, row in enumerate(results) if not row["goal_hit"]]
        want = min(int(args.num), len(results))
        want_goal = min(int(round(want * float(args.goal_fraction))), len(goals))
        pick = list(rng.permutation(goals)[:want_goal]) if want_goal else []
        remaining = want - len(pick)
        if remaining > 0:
            pick += list(rng.permutation(fails)[:remaining])
        if len(pick) < want:  # 失败案例不够就用成功案例补齐
            extra = [i for i in rng.permutation(goals) if i not in set(pick)]
            pick += list(extra[: want - len(pick)])
        return sorted(int(i) for i in pick)
    if args.select == "goal":
        pool = [i for i, row in enumerate(results) if row["goal_hit"]]
    elif args.select == "broken":
        pool = [i for i, row in enumerate(results) if row["status"] == "broken"]
    else:
        pool = [i for i, row in enumerate(results) if row["status"] == "loop"]
    if not pool:
        raise SystemExit(f"--select {args.select}: pool 里一条都没有")
    return pool[: int(args.num)]


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------
def _edge_list(graph, path: Sequence[int]) -> List[Tuple[int, int]]:
    return [(int(u), int(v)) for u, v in zip(path[:-1], path[1:]) if graph.has_edge(u, v)]


def _first_divergence(path: Sequence[int], reference: Sequence[int]) -> int:
    for index, (node, other) in enumerate(zip(path, reference)):
        if node != other:
            return int(path[index - 1]) if index > 0 else int(path[0])
    return int(path[min(len(path), len(reference)) - 1])


def _crop(ax, sample, pos_local) -> None:
    xs = [pos_local[node][0] for node in sample.graph.nodes() if node in pos_local]
    ys = [pos_local[node][1] for node in sample.graph.nodes() if node in pos_local]
    if not xs or not ys:
        return
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    margin = 0.10 * span + 1e-4
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def draw_panel(ax, sample, row, full_graph, full_pos, pos_local, args, mode, title) -> None:
    graph = sample.graph
    if args.street:
        nx.draw_networkx_edges(
            full_graph, full_pos, ax=ax, edge_color=COLOR_STREET, width=0.65,
        )
    # corridor 画成一条细蓝线就够了。**不要**给 corridor 边加宽光晕：corridor 本身
    # 是 100~350 个节点的密子图，粗光晕会互相叠成一片蓝，把底下的街道全糊掉 ——
    # 那就没有"街道背景"可看了。
    nx.draw_networkx_edges(
        graph, pos_local, ax=ax, edge_color=COLOR_CORRIDOR, width=1.2, alpha=0.9,
    )

    pred_path = [int(v) for v in row["path"]]
    gt_path = [int(v) for v in row["gt_path"]]

    if mode == "multi":
        ordered = sorted(
            row.get("multi_paths") or [], key=lambda item: item["log_prob"], reverse=True
        )
        if args.multi_goal_only:
            ordered = [item for item in ordered if item["status"] == "goal"]
        highlight = max(0, int(args.multi_highlight))
        drawn = 0
        forks: List[Tuple[int, int]] = []
        for rank, item in enumerate(ordered):
            nodes = [int(v) for v in item["nodes"]]
            if nodes == pred_path:
                continue
            if drawn >= int(args.multi_max_draw):
                break
            edges = _edge_list(graph, nodes)
            if not edges:
                continue
            if 1 <= rank <= highlight:
                color = _RANK_COLORS[rank % len(_RANK_COLORS)]
                width, alpha = 3.0, 0.95
                forks.append((rank, _first_divergence(nodes, pred_path)))
            elif item["status"] == "goal":
                # 到终点、但排不进前 N 名的备选：淡灰细线（它们是**完整备选路线**）
                color, width, alpha = COLOR_ALT, 1.6, 0.55
            else:
                # 路径表里绝大多数条目是被 NULL 提前终止的 1~2 跳短支，全画出来只会
                # 把上面那几条真正有意义的备选路线糊掉，所以直接跳过。
                continue
            nx.draw_networkx_edges(
                graph, pos_local, edgelist=edges, ax=ax,
                edge_color=color, width=width, alpha=alpha,
            )
            drawn += 1
        for rank, node in forks:
            if node in pos_local:
                ax.text(
                    pos_local[node][0], pos_local[node][1], str(rank + 1),
                    fontsize=6.5, color="white", ha="center", va="center", zorder=8,
                    bbox=dict(
                        boxstyle="circle,pad=0.16",
                        facecolor=_RANK_COLORS[rank % len(_RANK_COLORS)],
                        edgecolor="white", linewidth=0.6,
                    ),
                )

    # GT（真实司机历史路径）在下、模型选中路径在上
    gt_edges = _edge_list(graph, gt_path)
    if gt_edges:
        nx.draw_networkx_edges(
            graph, pos_local, edgelist=gt_edges, ax=ax, edge_color=COLOR_GT,
            width=2.6, style=(0, (3.2, 2.2)), alpha=1.0,
        )
    pred_edges = _edge_list(graph, pred_path)
    if pred_edges:
        nx.draw_networkx_edges(
            graph, pos_local, edgelist=pred_edges, ax=ax, edge_color=COLOR_PRED,
            width=4.4, alpha=0.95,
        )

    decision_nodes = [n for n in list(sample.segments.decision_nodes) if n in pos_local]
    if decision_nodes:
        nx.draw_networkx_nodes(
            graph, pos_local, nodelist=decision_nodes, ax=ax, node_size=12,
            node_color="#ffffff", edgecolors="#555555", linewidths=0.5, node_shape="s",
        )
    for node, color in ((sample.start, "#2ca02c"), (sample.goal, "#d62728")):
        if node in pos_local:
            nx.draw_networkx_nodes(
                graph, pos_local, nodelist=[node], ax=ax, node_size=180,
                node_color=color, edgecolors="black", linewidths=0.6, node_shape="*",
            )

    # 分叉点：预测路径与 GT 最后一次相同的节点
    common = 0
    for node_gt, node_pred in zip(gt_path, pred_path):
        if node_gt != node_pred:
            break
        common += 1
    if common < len(pred_path) and pred_path != gt_path and common >= 1:
        node = gt_path[common - 1]
        if node in pos_local:
            ax.plot(*pos_local[node], marker="X", markersize=8, color="#8b0000", zorder=9)

    ax.set_title(title, fontsize=8.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if args.crop:
        _crop(ax, sample, pos_local)


def render(
    mode, positions, results, dataset, full_graph, full_pos, args, payload, data_path
) -> Path:
    cols = min(int(args.cols), len(positions))
    rows = int(np.ceil(len(positions) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.4 * cols, 6.2 * rows), squeeze=False)
    axes = axes.ravel()
    table: List[Dict[str, Any]] = []
    for panel, position in enumerate(positions):
        row = results[position]
        sample = dataset[row["index"]]
        pos_local = local_positions(sample, full_graph, full_pos)
        if not pos_local:
            pos_local = nx.spring_layout(sample.graph, seed=0, iterations=100)
        ratio = row["pred_cost"] / row["gt_cost"] if row["gt_cost"] else float("nan")
        optimal = row.get("optimal_cost") or float("nan")
        gt_over_opt = row["gt_cost"] / optimal if optimal else float("nan")
        pred_over_opt = row["pred_cost"] / optimal if optimal else float("nan")
        status_text = {
            "goal": "到达 goal", "loop": "走进环", "broken": "断掉（NULL / 死路）",
        }.get(row["status"], row["status"])
        extra = ""
        if mode == "multi":
            # 不用 ✓/✗：Microsoft YaHei 缺 U+2713，会画成豆腐块（√/× 是有的）
            cover = "√" if row.get("coverage") else "×"
            extra = (
                f"\n路径表 {row.get('num_paths', 0)} 条"
                f"（到终点 {row.get('num_goal_paths', 0)}）· coverage {cover}"
            )
            if row.get("best_goal_hops") is not None:
                extra += f" · 最优 {int(row['best_goal_hops'])} 跳"
        cost_note = ""
        if optimal == optimal and optimal > 0:  # not NaN
            cost_note = f"\n走廊最短路 {optimal:.0f} m  ·  GT/最短 = {gt_over_opt:.2f}"
            if row["status"] == "goal":
                cost_note += f"  ·  预测/最短 = {pred_over_opt:.2f}"
            else:
                # 没到 goal 的路径只有前半段，拿它除以"起点到 goal 的最短路"是错的
                # （会算出一个 <1 的荒谬数字），所以这里不给比例。
                cost_note += (
                    f"  ·  预测未到 goal，只走了 {row['pred_cost']:.0f} m"
                    f"（不参与 预测/最短）"
                )
        title = (
            f"#{row['index']}  {sample.meta.get('date')}  ·  "
            f"junc={sample.meta.get('junction_len')} "
            f"dec={row['decisions']} corridor={row['nodes']}\n"
            f"GT {row['gt_hops']} 跳 / {row['gt_cost']:.0f} m  ·  "
            f"预测 {row['pred_hops']} 跳 / {row['pred_cost']:.0f} m "
            f"(×{ratio:.2f})  ·  {status_text}{cost_note}{extra}"
            + (f"\n⚠ 非法跳 {row['missing_hops']} 条" if row.get("missing_hops") else "")
        )
        draw_panel(
            axes[panel], sample, row, full_graph, full_pos, pos_local, args, mode, title
        )
        table.append(
            {
                "index": row["index"],
                "date": sample.meta.get("date"),
                "order_id": sample.meta.get("order_id"),
                "status": row["status"],
                "reason": row["reason"],
                "gt_hops": row["gt_hops"],
                "pred_hops": row["pred_hops"],
                "gt_cost_m": round(row["gt_cost"], 1),
                "pred_cost_m": round(row["pred_cost"], 1),
                "pred_over_gt_cost": round(ratio, 4),
                "decisions": row["decisions"],
                "corridor_nodes": row["nodes"],
                "candidates": row["candidates"],
                "num_paths": row["num_paths"],
                "num_goal_paths": row["num_goal_paths"],
                "coverage": row["coverage"],
                "best_goal_hops": row["best_goal_hops"],
                "path": row["path"],
                "gt_path": row["gt_path"],
                "missing_hops": row.get("missing_hops", 0),
            }
        )
    for spare in range(len(positions), len(axes)):
        axes[spare].set_axis_off()

    legend = [
        plt.Line2D([], [], color=COLOR_STREET, lw=1.6, label="成都路网 = 街道背景（真实经纬度）"),
        plt.Line2D([], [], color=COLOR_CORRIDOR, lw=1.6, label="OD corridor（模型看到的图）"),
        plt.Line2D([], [], color=COLOR_GT, lw=2.6, ls=(0, (3.2, 2.2)),
                   label="GT（真实司机历史路径）"),
        plt.Line2D([], [], color=COLOR_PRED, lw=4.4, label="模型最终选中的路径"),
        plt.Line2D([], [], marker="*", color="w", markerfacecolor="#2ca02c",
                   markersize=12, label="start"),
        plt.Line2D([], [], marker="*", color="w", markerfacecolor="#d62728",
                   markersize=12, label="goal"),
        plt.Line2D([], [], marker="s", color="w", markerfacecolor="white",
                   markeredgecolor="#555555", markersize=5, label="decision node"),
        plt.Line2D([], [], marker="X", color="w", markerfacecolor="#8b0000",
                   markersize=9, label="预测与 GT 的分叉点"),
    ]
    if mode == "multi":
        legend.insert(
            4,
            plt.Line2D([], [], color=_RANK_COLORS[1], lw=2.8,
                       label="第 2~5 名备选（圆圈编号 = 与主路径的分叉点）"),
        )
        legend.insert(5, plt.Line2D([], [], color=COLOR_ALT, lw=1.5,
                                    label="路径表里的其它路径（淡灰细线）"))
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8.6, frameon=False)
    mode_cn = "分支解码（single，posterior argmax 单路径）" if mode == "single" else (
        f"多分支解码（严格存活路径竞争 top-{args.multi_k}, beam={args.beam_width}）"
        if args.strict else
        f"多分支解码（存活路径表 top-{args.multi_k}, beam={args.beam_width}, "
        f"null={args.null_policy}）"
    )
    fig.suptitle(
        f"{Path(args.run).name}  ·  DiDi 成都真实数据  ·  {mode_cn}\n"
        f"checkpoint {Path(args.checkpoint).name}（epoch {payload.get('epoch')}）  ·  "
        f"{data_path.name}  ·  "
        f"{'deterministic' if args.deterministic else f'stochastic(seed={args.seed})'}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.955))
    out_path = PROJECT_ROOT / f"{args.out_prefix}_{mode}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"figure     : {out_path}")

    text_path = out_path.with_suffix(".txt")
    header = (
        f"{'idx':>5} {'date':>9} {'dec':>5} {'gt_hops':>8} {'pred':>5} "
        f"{'gt_m':>8} {'pred_m':>8} {'xGT':>6} {'status':<8} {'paths':>5} {'2goal':>6} cov"
    )
    lines = [out_path.name, "", header, "-" * len(header)]
    for row in table:
        lines.append(
            f"{row['index']:>5} {str(row['date']):>9} {row['decisions']:>5} "
            f"{row['gt_hops']:>8} {row['pred_hops']:>5} {row['gt_cost_m']:>8.0f} "
            f"{row['pred_cost_m']:>8.0f} {row['pred_over_gt_cost']:>6.2f} "
            f"{row['status']:<8} {row['num_paths']:>5} {row['num_goal_paths']:>6} "
            f"{'yes' if row['coverage'] else 'no'}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"table      : {text_path}")
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=1, ensure_ascii=False)
    return out_path


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
    args.checkpoint = args.checkpoint or str(run_dir / "best.pt")
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")

    device = get_device(args.device or str(config.get("training.device", "auto")))
    seed = int(config.get("seed", args.seed))
    set_seed(seed)

    model = build_model(config, device)
    payload = load_checkpoint(args.checkpoint, model=model, map_location=device)
    model = model.to(device)
    diffusion = build_diffusion(config)

    data_path = resolve_data_path(args, config)
    dataset = GraphQueryDataset.load(str(data_path))
    total = len(dataset)
    if int(args.pool_limit) > 0:
        total = min(total, int(args.pool_limit))

    print(f"checkpoint : {args.checkpoint} (epoch={payload.get('epoch')})")
    print(f"model      : {model.flow_steps_label}")
    print(f"data       : {data_path} ({len(dataset)} queries)")
    print(f"device     : {device}   T={diffusion.T}")

    full_graph = vds.load_global_graph(PROJECT_ROOT / str(config.get("paths.data_dir")))
    full_pos, geo_stats = vds.geographic_layout(full_graph, PROJECT_ROOT / args.coords)
    print(
        f"street bg  : {full_graph.number_of_nodes()} nodes / "
        f"{full_graph.number_of_edges()} edges, "
        f"{geo_stats['coverage']:.1%} carry lon/lat, "
        f"extent ≈ {geo_stats['wx_km']:.1f} × {geo_stats['wy_km']:.1f} km"
    )

    print("running reverse chain ...")
    planned = plan_indices(total, args)
    if planned is None:
        pairs = [(index, dataset[index]) for index in range(total)]
        print(f"  {args.select}: 需要全量 pool，解码 {len(pairs)} 条")
    else:
        pairs = [(index, dataset[index]) for index in planned]
        print(f"  {args.select}: 只解码选中的 {len(pairs)} 条（不必跑全量）")
    results = decode_pool(
        model, diffusion, pairs, device, make_generator(seed, device="cpu"), args
    )
    hits = sum(1 for row in results if row["goal_hit"])
    covered = sum(1 for row in results if row["coverage"])
    invalid = sum(1 for row in results if row.get("missing_hops"))
    print(
        f"pool result: goal={hits}/{len(results)}  "
        f"loop={sum(1 for r in results if r['status'] == 'loop')}  "
        f"broken={sum(1 for r in results if r['status'] == 'broken')}  "
        f"multi coverage={covered}/{len(results)}"
    )
    if invalid:
        print(
            f"WARNING    : {invalid}/{len(results)} 条解码路径含**图里不存在的边**，"
            f"代价被低估；图标题里标了 ⚠ 非法跳"
        )
    print_cost_summary(results)
    print_multi_summary(results, args)

    positions = (
        list(range(len(results))) if planned is not None else select_positions(results, args)
    )
    if not positions:
        raise SystemExit("no sample matched the selection")
    print(f"selected   : {[results[p]['index'] for p in positions]}")

    modes = ["single", "multi"] if args.mode == "both" else [args.mode]
    for mode in modes:
        render(
            mode, positions, results, dataset, full_graph, full_pos, args,
            payload, data_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
