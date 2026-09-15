"""多分支（存活路径表）解码评测：单路径 vs 多路径。

用法::

    python tools/evaluate_multipath.py --run outputs/runs/controlled_unweighted \
        --data data/unweighted/unweighted_test.pkl --top-k 2 --beam-width 64

它跑三套口径，全部用**同一套指标定义**（直接复用 evaluate_sample / aggregate）：

    1. 单路径（现状）：按采样出来的 z_0 解码，就是现在所有 eval json 的口径；
    2. 多分支 best：存活路径表里累计 log 概率最高的那条（不管有没有到终点）；
    3. 多分支 coverage：集合语义 —— 只要**任意一条**路径到终点就算成功，
       另外单独报告 "至少有一条最短路" 的比例。

第 3 项回答的是："模型的分布里到底有没有一条能到终点的路"（对应"NULL 提前停不等于
走不到"这个问题），它给出的是单路径口径的上界。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import collate_samples, iter_batches  # noqa: E402
from src.data.dataset import GraphQueryDataset  # noqa: E402
from src.diffusion.sampler import sample_reverse_chain  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    SampleRecord,
    aggregate,
    evaluate_sample,
)
from src.evaluation.evaluator import optimal_coverage  # noqa: E402
from src.evaluation.multi_path_decoder import decode_multi_path  # noqa: E402
from src.evaluation.readout import single_path_state  # noqa: E402
from src.evaluation.path_decoder import (  # noqa: E402
    candidate_offsets,
    decision_offsets,
    decode_flat,
)
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.setup import build_diffusion, build_model, get_device  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import make_generator, set_seed  # noqa: E402

_METRICS = ("goal_hit_rate", "optimal_path_rate", "success_cost_ratio", "loop_rate", "broken_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="single-path vs multi-path (beam) decoding")
    parser.add_argument("--run", required=True, help="run 目录（含 run_config.json / best.pt）")
    parser.add_argument("--checkpoint", default=None, help="默认 <run>/best.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--top-k", type=int, default=2, help="每个 decision 保留几条 branch")
    parser.add_argument("--beam-width", type=int, default=64, help="存活路径表的最大长度")
    parser.add_argument(
        "--null-policy", default="stop", choices=["stop", "skip"],
        help="stop：NULL 参与排名、选中即该路径终止（现状语义）；skip：NULL 不停，"
             "只在非 NULL 候选里取 top-k",
    )
    parser.add_argument(
        "--filter-dead-branches",
        action="store_true",
        help="top-k 之前剔除「终点既不是 Goal 也不是 decision node」的非 NULL branch"
        "（默认关 = 历史行为逐位可复现）",
    )
    parser.add_argument("--seeds", default="0", help="逗号分隔的采样种子（模型分布随种子变）")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条（0 = 全部）")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="结果 json")
    parser.add_argument("--show", type=int, default=0, help="打印前 N 条多分支详情")
    return parser.parse_args()


def evaluate_seed(
    model,
    diffusion,
    dataset,
    samples: Sequence[Any],
    seed: int,
    args,
    device,
) -> Dict[str, Any]:
    set_seed(seed)
    generator = make_generator(seed, device="cpu")

    single_records: List[SampleRecord] = []
    single_sampled_records: List[SampleRecord] = []
    best_records: List[SampleRecord] = []
    best_goal_records: List[SampleRecord] = []
    best_goal_cost_records: List[SampleRecord] = []
    coverage = 0
    optimal_coverage_hits = 0
    weighted_seen = False
    filtered_total = 0
    goal_path_counts: List[int] = []
    finished_counts: List[int] = []
    pruned_total = 0
    expanded_total = 0
    depth_max = 0
    details: List[Dict[str, Any]] = []
    start_time = time.time()

    for batch_samples in iter_batches(list(samples), batch_size=args.batch_size, shuffle=False):
        batch = collate_samples(batch_samples, device=device)
        chain = sample_reverse_chain(
            diffusion, model, batch, generator=generator, stochastic=True
        )
        # single 的最终 readout：默认取最终候选概率的组内 argmax（不是采样 z0）；
        # 采样 z0 仍然解一遍，作为诊断口径 single_sampled 单独报告。
        z0 = single_path_state(chain, batch, "single")
        z0_sampled = chain["z0"]
        prob = chain["candidate_prob"]
        d_offsets = decision_offsets(batch_samples)
        c_offsets = candidate_offsets(batch_samples)

        for index, sample in enumerate(batch_samples):
            offsets = dict(
                decision_offset=d_offsets[index], candidate_offset=c_offsets[index]
            )
            single = decode_flat(sample, z0, **offsets)
            single_records.append(evaluate_sample(sample, single))
            sampled = decode_flat(sample, z0_sampled, **offsets)
            single_sampled_records.append(evaluate_sample(sample, sampled))

            local = prob[c_offsets[index] : c_offsets[index] + sample.num_candidates]
            multi = decode_multi_path(
                sample,
                local,
                top_k=args.top_k,
                beam_width=args.beam_width,
                null_policy=args.null_policy,
                filter_dead_branches=args.filter_dead_branches,
            )
            best = multi.best
            if best is None:  # pragma: no cover - 理论上不会发生
                from src.evaluation.path_decoder import DecodeResult

                best_result = DecodeResult("broken", [sample.start], 0, "empty frontier")
            else:
                best_result = best.to_decode_result()
            best_records.append(evaluate_sample(sample, best_result))

            # 指南第 8 节：Goal 路径里"概率最高"与"真实 cost 最低"各自单独评测
            for records, path in (
                (best_goal_records, multi.best_goal),
                (best_goal_cost_records, multi.best_goal_cost_path),
            ):
                if path is None:
                    from src.evaluation.path_decoder import DecodeResult

                    records.append(
                        evaluate_sample(
                            sample,
                            DecodeResult(
                                "broken",
                                [sample.start],
                                0,
                                "no goal path in the surviving table",
                            ),
                        )
                    )
                else:
                    records.append(evaluate_sample(sample, path.to_decode_result()))

            hit = multi.coverage
            coverage += int(hit)
            # 集合语义：路径表里至少有一条**真实最优**（weighted = 最小 cost；
            # 无权图退化成跳数，与旧口径等价）
            optimal_coverage_hits += int(optimal_coverage(sample, multi))
            weighted_seen = weighted_seen or bool(sample.graph.graph.get("weighted", False))
            filtered_total += int(multi.num_filtered_dead_branches)
            goal_path_counts.append(len(multi.goal_paths))
            finished_counts.append(len(multi.finished))
            pruned_total += multi.pruned
            expanded_total += multi.num_expanded
            depth_max = max(depth_max, multi.max_depth)

            if len(details) < args.show:
                details.append(
                    {
                        "index": index,
                        "num_decisions": sample.num_decisions,
                        "gt_length": sample.gt_length,
                        "single": single.status,
                        "best": best.status if best else None,
                        "best_hops": best.cost if best else None,
                        "best_path_cost": best.path_cost if best else None,
                        "best_goal_path_cost": multi.best_goal_path_cost,
                        "coverage": hit,
                        "goal_paths_by_prob": multi.goal_paths_by_prob_dicts(3),
                        "goal_paths_by_cost": multi.goal_paths_by_cost_dicts(3),
                        **multi.summary(),
                    }
                )

    n = max(len(single_records), 1)
    out: Dict[str, Any] = {
        "seed": seed,
        "num_queries": len(single_records),
        # 方案里的 "single"：最终分支分布的 top-1 / 组内 argmax 单路径解码
        "single_path": aggregate(single_records),
        # 诊断口径：直接解码 reverse chain 采样出来的 z0（旧默认行为）
        "single_sampled": aggregate(single_sampled_records),
        "multi_best": aggregate(best_records),
        "multi_best_goal": aggregate(best_goal_records),
        "multi_best_goal_cost": aggregate(best_goal_cost_records),
        "coverage_rate": coverage / n,
        "optimal_coverage_rate": optimal_coverage_hits / n,
        "mean_goal_paths": statistics.fmean(goal_path_counts) if goal_path_counts else 0.0,
        "mean_finished_paths": statistics.fmean(finished_counts) if finished_counts else 0.0,
        "mean_filtered_dead_branches": filtered_total / n,
        "mean_pruned": pruned_total / n,
        "mean_expanded": expanded_total / n,
        "max_depth": depth_max,
        "filter_dead_branches": bool(args.filter_dead_branches),
        "dataset_is_weighted": bool(weighted_seen),
        "wall_time": time.time() - start_time,
    }
    if weighted_seen:
        # 名字点明按真实 cost 判定；无权图上它与 optimal_coverage_rate 是同一件事
        out["weighted_optimal_coverage_rate"] = optimal_coverage_hits / n
    if args.show:
        out["details"] = details
    return out


def _line(label: str, metrics: Dict[str, float]) -> str:
    parts = [f"{label:<26s}"]
    for key in _METRICS:
        value = metrics.get(key)
        parts.append(f"{key.split('_')[0][:7]}={value:6.4f}" if isinstance(value, float) else "")
    return "  ".join(parts)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run)
    config = load_config(run_dir / "run_config.json")
    device = get_device(args.device if args.device != "auto" else str(config.get("training.device", "auto")))

    model = build_model(config, device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pt"
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    model = model.to(device)
    model.eval()
    diffusion = build_diffusion(config)

    dataset = GraphQueryDataset.load(args.data)
    samples = list(dataset)
    if args.limit:
        samples = samples[: args.limit]
    seeds = [int(part) for part in str(args.seeds).split(",") if part.strip()]

    print(f"checkpoint   : {checkpoint} (epoch={payload.get('epoch')})")
    print(f"data         : {args.data}  ({len(samples)} queries)")
    print(f"mode         : top_k={args.top_k} beam_width={args.beam_width} "
          f"null_policy={args.null_policy} "
          f"filter_dead_branches={bool(args.filter_dead_branches)} seeds={seeds}")
    print()

    results = []
    for seed in seeds:
        result = evaluate_seed(model, diffusion, dataset, samples, seed, args, device)
        results.append(result)
        print(f"--- seed {seed} ({result['wall_time']:.1f}s)")
        print(_line("  single(最终 argmax, 新默认)", result["single_path"]))
        print(_line("  single_sampled(诊断口径)", result["single_sampled"]))
        print(_line("  多分支 best(累计概率最高)", result["multi_best"]))
        print(_line("  best_goal(Goal 里概率最高)", result["multi_best_goal"]))
        print(_line("  best_goal_cost(Goal 里 cost 最低)", result["multi_best_goal_cost"]))
        print(f"  多分支 coverage(>=1 条到终点) : {result['coverage_rate']:.4f}")
        coverage_label = (
            "  多分支 weighted optimal coverage(>=1 条最小 cost 路)"
            if result.get("dataset_is_weighted")
            else "  多分支 optimal coverage(>=1 条最短路)"
        )
        print(f"{coverage_label} : {result['optimal_coverage_rate']:.4f}")
        print(f"  平均：终止路径 {result['mean_finished_paths']:.1f} 条 / 到终点 "
              f"{result['mean_goal_paths']:.1f} 条 / 剪枝 {result['mean_pruned']:.1f} 条 / "
              f"被预筛选的必死 branch {result['mean_filtered_dead_branches']:.1f} 条 / "
              f"最大深度 {result['max_depth']}")
        if args.show:
            print("  样例：")
            for row in result.get("details", []):
                print(f"    #{row['index']:<5d} dec={row['num_decisions']:<3d} gt={row['gt_length']:<3d} "
                      f"单路径={row['single']:<7s} best={row['best']:<7s} hops={row['best_hops']} path_cost={row['best_path_cost']} "
                      f"coverage={row['coverage']} goal_paths={row['num_goal_paths']}")
        print()

    if len(results) > 1:
        print("=== 多种子均值 ===")
        for key in ("coverage_rate", "optimal_coverage_rate"):
            values = [r[key] for r in results]
            print(f"  {key:<24s} {statistics.fmean(values):.4f} ± {statistics.pstdev(values):.4f}")
        for label, name in (("单路径", "single_path"), ("多分支 best", "multi_best")):
            for key in ("goal_hit_rate", "optimal_path_rate"):
                values = [r[name][key] for r in results]
                print(f"  {label} {key:<16s} {statistics.fmean(values):.4f} ± "
                      f"{statistics.pstdev(values):.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run": str(run_dir),
                    "checkpoint": str(checkpoint),
                    "data": args.data,
                    "top_k": args.top_k,
                    "beam_width": args.beam_width,
                    "null_policy": args.null_policy,
                    "results": results,
                },
                handle,
                indent=1,
                ensure_ascii=False,
            )
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
