"""评测入口（实施指南第 24 节）。

用法::

    python scripts/evaluate.py --config configs/graph_flow.yaml \
        --checkpoint outputs/runs/graph_flow/best.pt --data data/controlled/controlled_test.pkl

输出主指标：Goal Hit / Optimal Path / Success Cost Ratio / Loop / Broken / 推理时间。
另外会打印一个 **debug 用** 的 teacher-forced 单步 decision accuracy。

``--decode multi`` 时，同一个存活路径表会按《Multi-Path Decoder 增强修改指南》第 8 节
并排输出三条口径（都走同一套 ``evaluate_sample``）：

    multi_best            累计 log 概率最高（历史口径，= 主指标）
    multi_best_goal       Goal 路径里概率最高
    multi_best_goal_cost  Goal 路径里真实 cost 最低

额外的集合语义指标：``coverage_rate`` / ``optimal_coverage_rate``
（weighted 数据集上后者按 Dijkstra 最小 cost 判定，并额外暴露
``weighted_optimal_coverage_rate`` 这个名字）/ ``mean_goal_paths`` /
``mean_filtered_dead_branches``。结果 JSON 里放在 ``multi`` 键下。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GraphQueryDataset  # noqa: E402
from src.evaluation.baselines import baseline_summary  # noqa: E402
from src.evaluation.evaluator import evaluate_dataset, records_to_dicts  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.setup import build_diffusion, build_model, get_device  # noqa: E402
from src.utils.config import flatten_overrides, load_config  # noqa: E402
from src.utils.seed import make_generator, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="evaluate the V2 Graph Flow denoiser")
    parser.add_argument("--config", default="configs/graph_flow.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None, help="结果 json 的输出路径")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--eval-flow-steps",
        type=int,
        default=None,
        help=(
            "推理时每个 reverse step 的图信息交流轮数（必须 <= 训练时的 round 数）。"
            "用于做'训练多轮、推理提前退出'的 ablation；默认用配置里的值。"
        ),
    )
    parser.add_argument("--stochastic", action="store_true", help="强制随机采样")
    parser.add_argument("--deterministic", action="store_true", help="posterior argmax 采样")
    parser.add_argument(
        "--decode",
        default="single",
        choices=["single", "multi"],
        help="single：按采样 z_0 单路径解码（历史口径）；"
        "multi：存活路径表解码（每个 decision 保留 top-k 条 branch，主指标取累计概率"
        "最高的那条，并额外输出 multi_best_goal / multi_best_goal_cost 两条口径与"
        "coverage_rate / optimal_coverage_rate）",
    )
    parser.add_argument("--top-k", type=int, default=2, help="--decode multi 时每个路口的 branch 数")
    parser.add_argument(
        "--filter-dead-branches",
        action="store_true",
        help="--decode multi 时，top-k 之前剔除「终点既不是 Goal 也不是 decision "
        "node」的非 NULL branch（默认关 = 历史多分支结果逐位可复现）",
    )
    parser.add_argument("--beam-width", type=int, default=64, help="--decode multi 时存活路径表上限")
    parser.add_argument(
        "--null-policy",
        default="stop",
        choices=["stop", "skip"],
        help="stop：NULL 参与排名、选中即该路径终止；skip：NULL 不停，只在非 NULL 候选里取 top-k",
    )
    parser.add_argument("--baselines", action="store_true", help="顺带跑 shortest/greedy baseline")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置项，可重复：--set training.lr=1e-4 --set data.num_samples=32",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = flatten_overrides(args.overrides)
    if args.device:
        overrides.append(f"training.device={args.device}")
    config = load_config(args.config, overrides)
    if args.stochastic:
        config = load_config(args.config, overrides + ["evaluation.stochastic_sampling=true"])
    if args.deterministic:
        config = load_config(args.config, overrides + ["evaluation.stochastic_sampling=false"])

    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = get_device(str(config.get("training.device", "auto")))
    generator = make_generator(seed, device="cpu")

    model = build_model(config, device)
    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint, model=model, map_location=device)
        model = model.to(device)
        print(f"loaded checkpoint {args.checkpoint} (epoch={payload.get('epoch')})")
    if args.eval_flow_steps is not None:
        previous = model.set_inference_flow_steps(args.eval_flow_steps)
        print(
            f"inference flow_steps overridden: {previous} -> {model.flow_steps} "
            f"(trained with {model.max_flow_steps})"
        )
    print(f"model        : {model.flow_steps_label}")
    if args.decode == "multi":
        print(
            f"decode       : multi (top_k={args.top_k}, beam_width={args.beam_width}, "
            f"null_policy={args.null_policy})  —— 主指标取累计概率最高的路径，"
            "额外报告 coverage_rate / optimal_coverage_rate"
        )
    else:
        print("decode       : single（按采样 z_0 解码，历史口径）")

    diffusion = build_diffusion(config)
    dataset = GraphQueryDataset.load(args.data)
    print(f"evaluating {len(dataset)} queries on {device}")

    report = evaluate_dataset(
        model,
        diffusion,
        dataset,
        batch_size=int(config.get("evaluation.batch_size", 8)),
        stochastic=bool(config.get("evaluation.stochastic_sampling", True)),
        device=device,
        generator=generator,
        max_steps=int(config.get("evaluation.max_steps", 0)) or None,
        progress=not args.no_progress,
        weights=None,
        decode=args.decode,
        top_k=args.top_k,
        beam_width=args.beam_width,
        null_policy=args.null_policy,
        filter_dead_branches=args.filter_dead_branches,
    )

    print("main metrics :", report.summary())
    if report.multi:
        # 指南第 8 节：同一个存活路径表，三条口径并排看（multi_best 是历史口径）
        for label in ("multi_best", "multi_best_goal", "multi_best_goal_cost"):
            row = report.multi.get(label, {})
            if not row:
                continue
            print(f"  {label:<20s} goal_hit={row.get('goal_hit_rate', float('nan')):.4f}"
                  f"  optimal={row.get('optimal_path_rate', float('nan')):.4f}"
                  f"  cost_ratio={row.get('success_cost_ratio', float('nan')):.4f}"
                  f"  broken={row.get('broken_rate', float('nan')):.4f}")
    if report.debug:
        print(f"debug metric : one_step_x0_acc={report.debug.get('accuracy', float('nan')):.4f} "
              f"(只作诊断，不作模型选择)")

    payload = {
        "metrics": report.metrics,
        "debug": report.debug,
        "inference": {
            "flow_steps": int(model.flow_steps),
            "trained_flow_steps": int(model.max_flow_steps),
            "label": model.flow_steps_label,
            "decode": args.decode,
            "top_k": int(args.top_k) if args.decode == "multi" else None,
            "beam_width": int(args.beam_width) if args.decode == "multi" else None,
            "null_policy": args.null_policy if args.decode == "multi" else None,
            "filter_dead_branches": bool(args.filter_dead_branches)
            if args.decode == "multi"
            else None,
        },
        "records": records_to_dicts(report.records),
    }
    if report.multi:
        payload["multi"] = report.multi
    if args.baselines:
        payload["baselines"] = baseline_summary(dataset)
        print("baselines    :", json.dumps(payload["baselines"], ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)
        print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
