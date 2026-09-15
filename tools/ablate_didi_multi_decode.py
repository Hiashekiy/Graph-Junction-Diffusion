"""多分支解码参数的消融：**反向链只跑一次**，然后在一堆 (top_k, beam, null_policy) 上复用。

为什么要单独有这么一个脚本：``decode_multi_path`` 是**纯 CPU 的解码后处理**，它的输入
只是每个样本的 local candidate 概率（``chain["candidate_prob"]`` 的一段）。所以换
top_k / beam / null_policy 根本不需要重新跑模型 —— 把 candidate_prob 留在内存里，
想比多少组合就比多少组合。直接反复调 ``visualize_didi_paths.py`` 会把 1000 条样本的
reverse chain 重跑一遍，绝大部分时间都在算一模一样的东西。

用法::

    python tools/ablate_didi_multi_decode.py --run outputs/runs/didi_chengdu
    python tools/ablate_didi_multi_decode.py --run <run> --combos 2:64:stop,2:3:skip,3:64:stop
    python tools/ablate_didi_multi_decode.py --run <run> --data data/didi/graph/chengdu/test.pkl

输出的每一行是一个组合，列的含义：

    coverage          路径表里**至少有一条**到终点（"生成出来了"）
    rank1→goal        路径表里累计 log 概率最高的那条（主口径的"最终选中路径"）到终点
    cover&rank1✗      表里有正确分支、但排名第 1 的那条没到终点 —— **纯排序问题**
    cov-ok/rank1      上面两个的差：coverage 减去 rank1→goal
    paths / goal      路径表条数 / 其中到终点的条数的中位数 (p50)
    goal_hops         表里到终点路径的跳数中位数 (p50)

``coverage`` 高而 ``rank1→goal`` 低 = 正确分支被生成出来了、只是没排到第一，该动的是
readout / 概率校准；``coverage`` 本身低 = beam 或 top_k 把正确分支剪掉了，该动的是搜索。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: 默认对比的组合：``top_k:beam_width:null_policy``。
#: 第一行是历史基线（可视化/评测的默认值），第二行是"beam 砍到 3 + NULL 不停"。
DEFAULT_COMBOS = "2:64:stop,2:64:skip,2:3:stop,2:3:skip,3:64:stop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ablate multi-path decode settings")
    parser.add_argument("--run", required=True, help="run 目录（含 run_config.json）")
    parser.add_argument("--checkpoint", default=None, help="默认 <run>/best.pt")
    parser.add_argument("--data", default=None, help="默认 <data_dir>/test_1000.pkl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pool-limit", type=int, default=0, help=">0 时只跑前 N 条")
    parser.add_argument(
        "--combos", default=DEFAULT_COMBOS,
        help="逗号分隔的 top_k:beam_width:null_policy",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="posterior argmax rollout。**建议开**：随机采样下 candidate_prob 依赖 "
             "batch 组成，组合之间的差异会混进采样噪声",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strict", action="store_true",
        help="改用三池语义（strict_beam_decoder）：失败路径淘汰、最终候选只有完整到 "
             "Goal 的路径。**这个开关才是「严格存活路径竞争」**；不加就是历史口径。",
    )
    parser.add_argument("--out", default=None, help="把结果写成 json")
    return parser.parse_args()


def parse_combos(spec: str) -> List[Tuple[int, int, str]]:
    combos: List[Tuple[int, int, str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 3:
            raise SystemExit(f"--combos 项必须是 top_k:beam:null_policy，收到 {part!r}")
        top_k, beam = int(fields[0]), int(fields[1])
        policy = fields[2].strip()
        if policy not in ("stop", "skip"):
            raise SystemExit(f"null_policy 只能是 stop/skip，收到 {policy!r}")
        combos.append((top_k, beam, policy))
    if not combos:
        raise SystemExit("--combos 是空的")
    return combos


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def med(values):
        values = [v for v in values if v is not None]
        return float(st.median(values)) if values else float("nan")

    total = len(rows)
    covered = [r for r in rows if r["coverage"]]
    rank1_goal = [r for r in rows if r["rank1_goal"]]
    miss = [r for r in covered if not r["rank1_goal"]]
    optimal_cov = [r for r in rows if r["optimal_coverage"]]
    goal_ratios = [r["best_goal_over_optimal"] for r in rows]
    rank1_ratios = [r["rank1_over_optimal"] for r in rows if r["rank1_goal"]]
    return {
        "num_samples": total,
        "coverage": len(covered),
        "coverage_rate": len(covered) / total if total else float("nan"),
        "optimal_coverage": len(optimal_cov),
        "optimal_coverage_rate": len(optimal_cov) / total if total else float("nan"),
        "rank1_goal": len(rank1_goal),
        "rank1_goal_rate": len(rank1_goal) / total if total else float("nan"),
        "coverage_but_rank1_fail": len(miss),
        "median_paths": med([r["num_paths"] for r in rows]),
        "median_goal_paths": med([r["num_goal_paths"] for r in rows]),
        "median_best_goal_hops": med([r["best_goal_hops"] for r in rows]),
        "median_rank1_hops": med([r["rank1_hops"] for r in rows]),
        "mean_masked": med([r["num_masked"] for r in rows]),
        "mean_discarded": med([r["num_discarded"] for r in rows]),
        "median_best_goal_over_optimal": med(goal_ratios),
        "median_rank1_over_optimal": med(rank1_ratios),
        "best_goal_within_1.05": sum(1 for v in goal_ratios if v is not None and v <= 1.05),
    }


def path_cost(graph, path: Sequence[int]) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if graph.has_edge(u, v):
            total += float(graph[u][v].get("weight", 1.0))
    return total


def optimal_cost(graph, start, goal) -> float:
    import networkx as nx

    try:
        return path_cost(
            graph, nx.shortest_path(graph, start, goal, weight="weight")
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):  # pragma: no cover
        return float("nan")


def main() -> int:
    args = parse_args()
    combos = parse_combos(args.combos)

    # Windows 控制台默认是 GBK，不能编码 ✓/✗ 这类符号会直接抛 UnicodeEncodeError
    # 把整个消融打断（表格都还没打出来）。这里降级成替换字符，**绝不因为打不出一个
    # 符号就崩**；同时输出里只用 GBK 覆盖得到的字符。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover
            pass

    from src.data.collate import collate_samples
    from src.data.dataset import GraphQueryDataset
    from src.diffusion.sampler import sample_reverse_chain
    from src.evaluation.multi_path_decoder import decode_multi_path
    from src.evaluation.path_decoder import candidate_offsets, decision_offsets, decode_flat
    from src.evaluation.readout import single_path_state
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_diffusion, build_model, get_device
    from src.utils.config import load_config
    from src.utils.seed import make_generator, set_seed

    run_dir = Path(args.run)
    config = load_config(run_dir / "run_config.json")
    checkpoint = args.checkpoint or str(run_dir / "best.pt")
    device = get_device(args.device or str(config.get("training.device", "auto")))
    seed = int(config.get("seed", args.seed))
    set_seed(seed)

    model = build_model(config, device)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    model = model.to(device)
    diffusion = build_diffusion(config)

    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
    else:
        data_dir = PROJECT_ROOT / str(config.get("paths.data_dir"))
        data_path = next(
            (data_dir / name for name in ("test_1000.pkl", "test.pkl")
             if (data_dir / name).exists()),
            None,
        )
        if data_path is None:
            raise SystemExit(f"no test split under {data_dir}")
    dataset = GraphQueryDataset.load(str(data_path))
    total = len(dataset)
    if int(args.pool_limit) > 0:
        total = min(total, int(args.pool_limit))

    print(f"checkpoint : {checkpoint} (epoch={payload.get('epoch')})")
    print(f"data       : {data_path}  ({total} queries)")
    print(
        f"mode       : {'deterministic' if args.deterministic else f'stochastic(seed={seed})'}"
        f"  ·  {'STRICT 三池语义' if args.strict else '历史口径（finished 混装）'}"
    )
    print("running the reverse chain ONCE for all combos ...")

    samples = [dataset[index] for index in range(total)]
    # 每条样本只留"本地 candidate 概率"和 single 解码结果 —— 这就是多分支解码的全部输入
    local_probs: List[np.ndarray] = []
    single_goal = 0
    statuses: List[str] = []
    with torch.no_grad():
        model.eval()
        for start in range(0, total, args.batch_size):
            chunk = samples[start : start + args.batch_size]
            batch = collate_samples(chunk, device=device)
            chain = sample_reverse_chain(
                diffusion, model, batch, generator=make_generator(seed, device="cpu"),
                stochastic=not args.deterministic,
            )
            z0 = (
                chain["z0"] if args.deterministic
                else single_path_state(chain, batch, "single")
            )
            offsets = decision_offsets(chunk)
            starts = candidate_offsets(chunk)
            for index, sample in enumerate(chunk):
                local_probs.append(
                    chain["candidate_prob"][
                        starts[index] : starts[index] + sample.num_candidates
                    ].detach().cpu().numpy()
                )
                result = decode_flat(
                    sample, z0, decision_offset=offsets[index],
                    candidate_offset=starts[index],
                )
                statuses.append(result.status)
                single_goal += int(result.status == "goal")
            print(f"  chained {min(start + args.batch_size, total)}/{total}", end="\r", flush=True)
    print()

    print(
        f"\n参考（single 解码，不是多分支）：goal={single_goal}/{total}  "
        f"loop={sum(1 for s in statuses if s == 'loop')}  "
        f"broken={sum(1 for s in statuses if s == 'broken')}"
    )

    table: Dict[str, Any] = {}
    header = (
        f"{'top_k':>5} {'beam':>5} {'null':>5} | {'coverage':>10} {'opt_cov':>10} | "
        f"{'multi_best':>11} {'mb/opt':>7} | {'paths':>7} {'goal':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for top_k, beam, policy in combos:
        rows: List[Dict[str, Any]] = []
        for sample, probability in zip(samples, local_probs):
            multi = decode_multi_path(
                sample, probability, top_k=top_k, beam_width=beam,
                null_policy=policy, strict=bool(args.strict),
            )
            best = multi.best
            best_goal = multi.best_goal
            optimal = optimal_cost(sample.graph, sample.start, sample.goal)
            rank1_goal = best is not None and int(best.nodes[-1]) == int(sample.goal)
            # 与 evaluator.py:69 完全同口径：表里有没有一条 path_cost == Dijkstra 的
            # goal 路径（rel_tol/abs_tol 都是 1e-6）
            optimal_coverage = any(
                math.isclose(path.path_cost, optimal, rel_tol=1e-6, abs_tol=1e-6)
                for path in multi.goal_paths
            ) if optimal == optimal else False
            rows.append(
                {
                    "coverage": bool(multi.coverage),
                    "optimal_coverage": optimal_coverage,
                    "rank1_goal": rank1_goal,
                    "num_paths": len(multi.finished),
                    "num_masked": int(multi.num_masked),
                    "num_discarded": len(multi.discarded),
                    "num_goal_paths": len(multi.goal_paths),
                    "best_goal_hops": multi.best_goal_cost,
                    "rank1_hops": None if best is None else int(best.cost),
                    "rank1_over_optimal": (
                        path_cost(sample.graph, best.nodes) / optimal
                        if rank1_goal and optimal == optimal and optimal > 0 else None
                    ),
                    "best_goal_over_optimal": (
                        path_cost(sample.graph, best_goal.nodes) / optimal
                        if best_goal is not None and optimal == optimal and optimal > 0
                        else None
                    ),
                }
            )
        stats = summarize(rows)
        key = f"top{top_k}_beam{beam}_{policy}"
        table[key] = stats
        print(
            f"{top_k:>5} {beam:>5} {policy:>5} | "
            f"{stats['coverage']:>4}/{total:<5} {stats['optimal_coverage']:>4}/{total:<5} | "
            f"{stats['rank1_goal']:>5}/{total:<5} "
            f"{stats['median_rank1_over_optimal']:>7.3f} | "
            f"{stats['median_paths']:>7.0f} {stats['median_goal_paths']:>6.0f}"
        )

    if args.out:
        out = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "checkpoint": str(checkpoint),
                    "epoch": payload.get("epoch"),
                    "data": str(data_path),
                    "num_samples": total,
                    "deterministic": bool(args.deterministic),
                    "single_goal": single_goal,
                    "combos": table,
                },
                handle, indent=1, ensure_ascii=False,
            )
        print(f"\nwritten    : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
