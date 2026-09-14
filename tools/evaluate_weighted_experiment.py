"""Weighted 实验收尾：训练结束后一次性出四套口径的对照（方案第 17 节）。

    1. Ours weighted                 —— weighted GT + Edge Cost Encoder
    2. Ours weighted (cost ablated)  —— 同一份 weighted GT，但模型看不到 edge cost
    3. Greedy-BFS                    —— 按跳数贪心（在带权图上必然退化）
    4. Dijkstra oracle               —— 加权最优（cost_ratio 恒等于 1.0）

用法::

    # 等两个 run 都训完再评测（轮询 history.json 的 epoch，不需要看进程表）
    python tools/evaluate_weighted_experiment.py --wait

    # 立刻评测（训练已经结束）
    python tools/evaluate_weighted_experiment.py

输出默认写到 ``outputs/weighted_experiment.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GraphQueryDataset  # noqa: E402
from src.evaluation.baselines import baseline_summary  # noqa: E402
from src.evaluation.evaluator import evaluate_dataset  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.setup import build_diffusion, build_model, get_device  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import make_generator, set_seed  # noqa: E402

METRIC_KEYS = (
    "num_queries",
    "goal_hit_rate",
    "optimal_path_rate",
    "success_cost_ratio",
    "loop_rate",
    "broken_rate",
    "soft_goal_reachability",
    "mean_pred_cost",
    "mean_optimal_cost",
    "mean_elapsed",
)


def training_finished(run_dir: Path) -> bool:
    """history.json 的最后一个 epoch 是否已经到达 run_config 里的 epochs。"""
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not history:
        return False
    target = int(load_config(run_dir / "run_config.json").get("training.epochs", 100))
    return int(history[-1].get("epoch", 0)) >= target


def evaluate_run(run_dir: Path, dataset, device, seed: int) -> dict:
    config = load_config(run_dir / "run_config.json")
    set_seed(seed)
    generator = make_generator(seed, device="cpu")
    model = build_model(config, device)
    payload = load_checkpoint(run_dir / "best.pt", model=model, map_location=device)
    model = model.to(device)
    diffusion = build_diffusion(config)
    report = evaluate_dataset(
        model,
        diffusion,
        dataset,
        batch_size=int(config.get("evaluation.batch_size", 8)),
        stochastic=bool(config.get("evaluation.stochastic_sampling", True)),
        device=device,
        generator=generator,
        max_steps=int(config.get("evaluation.max_steps", 0)) or None,
        progress=False,
        weights=None,
    )
    metrics = {key: float(report.metrics[key]) for key in METRIC_KEYS if key in report.metrics}
    return {
        "run_dir": str(run_dir),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "use_edge_cost": bool(config.get("model.use_edge_cost", False)),
        "edge_cost_label": model.edge_cost_label,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="evaluate the weighted experiment")
    parser.add_argument("--main-run", default="outputs/runs/v2_weighted_controlled")
    parser.add_argument(
        "--ablation-run", default="outputs/runs/v2_weighted_controlled_cost_ablated"
    )
    parser.add_argument("--data", default="data/weighted_controlled/weighted_controlled_test.pkl")
    parser.add_argument("--out", default="outputs/weighted_experiment.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wait", action="store_true", help="等两个 run 都训完再评测")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()

    main_dir = PROJECT_ROOT / args.main_run
    ablation_dir = PROJECT_ROOT / args.ablation_run

    if args.wait:
        deadline = time.time() + args.timeout_hours * 3600
        while True:
            done = training_finished(main_dir) and training_finished(ablation_dir)
            if done:
                print("[wait] both runs finished", flush=True)
                break
            if time.time() > deadline:
                print("[wait] timeout: evaluating whatever is on disk now", flush=True)
                break
            print(
                f"[wait] main={training_finished(main_dir)} "
                f"ablation={training_finished(ablation_dir)}",
                flush=True,
            )
            time.sleep(max(10, int(args.poll_seconds)))

    device = get_device(args.device)
    dataset = GraphQueryDataset.load(PROJECT_ROOT / args.data)
    print(f"evaluating {len(dataset)} weighted queries on {device}", flush=True)

    set_seed(args.seed)
    baselines = baseline_summary(dataset)
    print("baselines:", json.dumps(baselines, ensure_ascii=False), flush=True)

    weight_stats = {}
    summary_path = PROJECT_ROOT / args.data.replace("_test.pkl", "_summary.json")
    if summary_path.exists():
        weight_stats = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "dataset", {}
        ).get("weighted", {})

    payload = {
        "data": args.data,
        "num_queries": len(dataset),
        "seed": int(args.seed),
        "dataset_weighted_stats": weight_stats,
        "baselines": baselines,
        "runs": {},
    }
    for name, run_dir in (("ours_weighted", main_dir), ("ours_cost_ablated", ablation_dir)):
        if not (run_dir / "best.pt").exists():
            print(f"[skip] {run_dir} has no best.pt", flush=True)
            continue
        print(f"--- {name}: {run_dir}", flush=True)
        payload["runs"][name] = evaluate_run(run_dir, dataset, device, args.seed)
        print(json.dumps(payload["runs"][name]["metrics"], indent=1), flush=True)

    ours = payload["runs"].get("ours_weighted", {}).get("metrics", {})
    ablated = payload["runs"].get("ours_cost_ablated", {}).get("metrics", {})
    if ours and ablated:
        payload["comparison"] = {
            key: {
                "with_edge_cost": ours.get(key),
                "cost_ablated": ablated.get(key),
                "delta": (ours.get(key, 0.0) - ablated.get(key, 0.0)),
            }
            for key in ("goal_hit_rate", "optimal_path_rate", "success_cost_ratio")
        }

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
