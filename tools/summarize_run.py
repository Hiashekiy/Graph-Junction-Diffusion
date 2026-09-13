"""汇总一次训练的结果：曲线 + best checkpoint 的测试集评测。

用法::

    python tools/summarize_run.py outputs/runs/v2_controlled_100ep \
        --config configs/graph_flow.yaml --test-data data/controlled_test.pkl

输出 ``<run_dir>/summary.json`` 与 ``<run_dir>/summary.txt``，内容包括：
    * 每 epoch 的 train loss / x0 accuracy；
    * 每次验证的 Goal Hit / Optimal / Cost Ratio / Loop / Broken；
    * best checkpoint 在测试集上的完整指标（含 baseline 对照）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_history(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"no history.json in {run_dir}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def curve(history: List[Dict[str, Any]], key: str) -> List[Dict[str, float]]:
    """从 history 抽一条曲线。

    epoch 号优先用记录里的 ``epoch`` 字段（resume 之后列表下标 ≠ epoch 号，
    老版本 history 里没有这个字段，就退化成下标 +1）。
    """
    out = []
    for index, record in enumerate(history, start=1):
        if key in record and record[key] == record[key]:  # 过滤 NaN
            out.append(
                {"epoch": int(record.get("epoch", index)), "value": float(record[key])}
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="summarize a training run")
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="configs/graph_flow.yaml")
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    import torch

    from src.data.dataset import GraphQueryDataset
    from src.evaluation.baselines import baseline_summary
    from src.evaluation.evaluator import evaluate_dataset, records_to_dicts
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_diffusion, build_model, get_device
    from src.utils.config import load_config
    from src.utils.seed import make_generator, set_seed

    run_dir = Path(args.run_dir)
    history = load_history(run_dir)
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "epochs_completed": len(history),
        "curves": {
            key: curve(history, key)
            for key in (
                "train_loss",
                "train_x0_acc",
                "goal_hit_rate",
                "optimal_path_rate",
                "success_cost_ratio",
                "loop_rate",
                "broken_rate",
                "val_x0_acc",
            )
        },
    }

    validation = [r for r in history if "goal_hit_rate" in r]
    if validation:
        best = max(validation, key=lambda r: r["goal_hit_rate"])
        best_epoch = int(
            best.get("epoch", history.index(best) + 1)
        )
        summary["best_validation"] = {
            "epoch": best_epoch,
            "goal_hit_rate": best["goal_hit_rate"],
            "optimal_path_rate": best.get("optimal_path_rate"),
            "avg_goal_hit_rate": sum(r["goal_hit_rate"] for r in validation)
            / len(validation),
            "avg_optimal_path_rate": sum(
                r.get("optimal_path_rate", 0.0) for r in validation
            )
            / len(validation),
        }

    if args.test_data and not args.skip_eval:
        overrides = [item for group in args.overrides for item in group]
        config = load_config(args.config, overrides)
        seed = int(config.get("seed", 0))
        set_seed(seed)
        device = get_device(args.device or str(config.get("training.device", "auto")))
        checkpoint = args.checkpoint or str(run_dir / "best.pt")

        model = build_model(config, device)
        payload = load_checkpoint(checkpoint, model=model, map_location=device)
        model = model.to(device)
        diffusion = build_diffusion(config)
        dataset = GraphQueryDataset.load(args.test_data)
        report = evaluate_dataset(
            model,
            diffusion,
            dataset,
            batch_size=int(config.get("evaluation.batch_size", 16)),
            stochastic=bool(config.get("evaluation.stochastic_sampling", True)),
            device=device,
            generator=make_generator(seed, device="cpu"),
            progress=False,
            weights=None,
        )
        summary["test"] = {
            "checkpoint": checkpoint,
            "checkpoint_epoch": payload.get("epoch"),
            "num_queries": len(dataset),
            "metrics": report.metrics,
            "debug": report.debug,
            "baselines": baseline_summary(dataset),
        }
        with open(run_dir / "test_records.json", "w", encoding="utf-8") as handle:
            json.dump(records_to_dicts(report.records), handle, indent=1)
        del torch

    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)

    lines = [f"run: {run_dir}", f"epochs completed: {len(history)}"]
    if "best_validation" in summary:
        best = summary["best_validation"]
        lines.append(
            f"best validation: epoch {best['epoch']}  goal_hit={best['goal_hit_rate']:.4f}  "
            f"optimal={best['optimal_path_rate']:.4f}"
        )
        lines.append(
            f"validation average: goal_hit={best['avg_goal_hit_rate']:.4f}  "
            f"optimal={best['avg_optimal_path_rate']:.4f}"
        )
    if "test" in summary:
        metrics = summary["test"]["metrics"]
        lines.append(f"test ({summary['test']['num_queries']} queries, "
                     f"checkpoint epoch {summary['test']['checkpoint_epoch']}):")
        for key in (
            "goal_hit_rate",
            "optimal_path_rate",
            "success_cost_ratio",
            "loop_rate",
            "broken_rate",
            "mean_elapsed",
        ):
            if key in metrics:
                lines.append(f"  {key}: {metrics[key]:.4f}")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {run_dir / 'summary.json'} , {run_dir / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
