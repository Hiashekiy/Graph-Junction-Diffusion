"""推理轮数 ablation：训练用 K 轮，推理分别跑 1..K 轮，量准确率/成本的取舍。

多轮交流的代价在推理时是线性的（实测 flow_steps=3 约 0.06 s/query，1 轮约
0.03 s/query）。而"训练时见识过多轮、推理时少跑几轮"完全合法（第 k 轮用的
slot embedding 就是训练时那一行），所以值得量化"多少轮才够"：

    python tools/ablation_eval.py outputs/runs/v2_controlled_100ep_flow3 \
        --data data/controlled_test.pkl

对 k = 1..trained_flow_steps 各评一遍，写
``<run_dir>/eval_test_flow{k}.json``，并把汇总表打印出来 + 写
``<run_dir>/ablation_flow_steps.json``。若某个 k 的结果文件已存在且 ``--skip-existing``，
就跳过（避免重复跑几十分钟）。

实现上直接改模型自己的 ``flow_steps``（``set_inference_flow_steps``），
**不重建模型**，所以它和 checkpoint 的结构始终匹配。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inference flow_steps ablation")
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default=None, help="默认 <run_dir>/best.pt")
    parser.add_argument("--data", default="data/controlled_test.pkl")
    parser.add_argument("--config", default=None, help="默认 <run_dir>/run_config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, nargs="*", default=None,
                        help="要评的轮数，默认 1..训练时的轮数")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    from src.data.dataset import GraphQueryDataset
    from src.evaluation.evaluator import evaluate_dataset, records_to_dicts
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_diffusion, build_model, get_device
    from src.utils.config import load_config
    from src.utils.seed import make_generator, set_seed

    run_dir = Path(args.run_dir)
    config_path = Path(args.config) if args.config else run_dir / "run_config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config: {config_path}")
    config = load_config(config_path)
    checkpoint = args.checkpoint or str(run_dir / "best.pt")

    seed = int(config.get("seed", 0))
    device = get_device(args.device or str(config.get("training.device", "auto")))
    dataset = GraphQueryDataset.load(args.data)

    # 先建一次模型只为读出"训练时是几轮"（max_flow_steps = slot embedding 的行数）。
    probe = build_model(config, device)
    probe_payload = load_checkpoint(checkpoint, model=probe, map_location=device)
    trained_steps = int(probe.max_flow_steps)
    del probe, torch
    steps_list = args.steps or list(range(1, trained_steps + 1))
    for steps in steps_list:
        if steps < 1 or steps > trained_steps:
            raise SystemExit(
                f"requested flow_steps={steps} but the checkpoint was trained with "
                f"{trained_steps} round(s); inference can only use fewer rounds"
            )

    print(f"run        : {run_dir}")
    print(f"checkpoint : {checkpoint} (epoch={probe_payload.get('epoch')})")
    print(f"data       : {args.data} ({len(dataset)} queries)")
    print(f"rounds     : trained={trained_steps}, evaluating {steps_list}")

    # 每次都从同一个 checkpoint 重建模型，避免上一次改过 flow_steps 之后影响下一次。
    summary: List[Dict[str, Any]] = []
    for steps in steps_list:
        out_path = run_dir / f"eval_test_flow{steps}.json"
        if args.skip_existing and out_path.exists():
            with open(out_path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)["metrics"]
            print(f"[skip] flow_steps={steps} -> {out_path}")
        else:
            set_seed(seed)
            model = build_model(config, device)
            payload = load_checkpoint(checkpoint, model=model, map_location=device)
            model = model.to(device)
            model.set_inference_flow_steps(steps)
            diffusion = build_diffusion(config)
            batch_size = args.batch_size or int(config.get("evaluation.batch_size", 16))
            report = evaluate_dataset(
                model,
                diffusion,
                dataset,
                batch_size=batch_size,
                stochastic=bool(config.get("evaluation.stochastic_sampling", True)),
                device=device,
                generator=make_generator(seed, device="cpu"),
                max_steps=int(config.get("evaluation.max_steps", 0)) or None,
                progress=not args.no_progress,
                weights=None,
            )
            metrics = report.metrics
            out_payload = {
                "metrics": metrics,
                "debug": report.debug,
                "inference": {
                    "flow_steps": int(steps),
                    "trained_flow_steps": int(model.max_flow_steps),
                    "checkpoint_epoch": payload.get("epoch"),
                    "label": model.flow_steps_label,
                },
                "records": records_to_dicts(report.records),
            }
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(out_payload, handle, indent=1, ensure_ascii=False)
            print(f"[done] flow_steps={steps} -> {out_path}")
        summary.append(
            {
                "flow_steps": int(steps),
                "goal_hit_rate": metrics.get("goal_hit_rate"),
                "optimal_path_rate": metrics.get("optimal_path_rate"),
                "success_cost_ratio": metrics.get("success_cost_ratio"),
                "broken_rate": metrics.get("broken_rate"),
                "mean_elapsed": metrics.get("mean_elapsed"),
            }
        )

    header = (
        f"{'flow_steps':>10}  {'goal_hit':>8}  {'optimal':>8}  {'cost':>6}  "
        f"{'broken':>7}  {'sec/query':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['flow_steps']:>10}  {row['goal_hit_rate']:>8.4f}  "
            f"{row['optimal_path_rate']:>8.4f}  {row['success_cost_ratio']:>6.3f}  "
            f"{row['broken_rate']:>7.4f}  {row['mean_elapsed']:>9.4f}"
        )
    with open(run_dir / "ablation_flow_steps.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
    print(f"\nwritten: {run_dir / 'ablation_flow_steps.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
