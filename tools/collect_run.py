"""整合一次训练的全部可用证据，输出统一的 summary.json / summary.txt。

数据来源：
  * ``history.json``        逐 epoch 的 train loss / accuracy（本 run 覆盖 21..100）
  * ``val_records_epoch*.json``  每次验证的逐 query 记录（文件名带**真实** epoch 号，
    所以即使 history.json 被续训覆盖，验证曲线仍然完整）
  * checkpoint               记录 epoch / best_metric

用法：python tools/collect_run.py outputs/runs/v2_controlled_100ep [--test-eval <json>]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List


def load_history(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "history.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_validation_curves(run_dir: Path) -> Dict[int, Dict[str, float]]:
    """从 val_records_epoch*.json 复原每次验证的汇总指标。"""
    curves: Dict[int, Dict[str, float]] = {}
    for path in sorted(run_dir.glob("val_records_epoch*.json")):
        match = re.search(r"epoch(\d+)\.json$", path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not records:
            continue
        total = len(records)
        hits = [r for r in records if r.get("goal_hit")]
        ratios = [r["cost_ratio"] for r in hits if r.get("cost_ratio") not in (None, float("inf"))]
        curves[epoch] = {
            "num_queries": total,
            "goal_hit_rate": len(hits) / total,
            "optimal_path_rate": sum(1 for r in records if r.get("optimal")) / total,
            "loop_rate": sum(1 for r in records if r.get("status") == "loop") / total,
            "broken_rate": sum(1 for r in records if r.get("status") == "broken") / total,
            "success_cost_ratio": (sum(ratios) / len(ratios)) if ratios else float("nan"),
            "mean_elapsed": sum(r.get("elapsed", 0.0) for r in records) / total,
        }
    return curves


def main() -> int:
    parser = argparse.ArgumentParser(description="collect run evidence into one summary")
    parser.add_argument("run_dir")
    parser.add_argument("--test-eval", default=None, help="scripts/evaluate.py 输出的 json")
    parser.add_argument("--dataset-summary", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history = load_history(run_dir)
    validation = load_validation_curves(run_dir)
    checkpoint_meta = {}
    for name in ("best.pt", "last.pt"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_meta[name] = {
                "epoch": payload.get("epoch"),
                "global_step": payload.get("global_step"),
                "best_metric": payload.get("best_metric"),
            }
        except Exception as error:  # noqa: BLE001
            checkpoint_meta[name] = {"error": str(error)}

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "epochs_in_history": len(history),
        "history_epoch_range": (
            [history[0].get("epoch", 1), history[-1].get("epoch", len(history))]
            if history
            else None
        ),
        "checkpoints": checkpoint_meta,
        "validation": {str(epoch): values for epoch, values in sorted(validation.items())},
    }

    if history:
        summary["train_curve"] = [
            {
                "epoch": int(record.get("epoch", index + 1)),
                "train_loss": record.get("train_loss"),
                "train_x0_acc": record.get("train_x0_acc"),
            }
            for index, record in enumerate(history)
        ]
        summary["train_first_last"] = {
            "loss": [history[0].get("train_loss"), history[-1].get("train_loss")],
            "x0_acc": [history[0].get("train_x0_acc"), history[-1].get("train_x0_acc")],
        }

    if validation:
        best_epoch = max(validation, key=lambda epoch: validation[epoch]["goal_hit_rate"])
        hits = [validation[epoch]["goal_hit_rate"] for epoch in validation]
        summary["best_validation"] = {
            "epoch": best_epoch,
            **validation[best_epoch],
        }
        summary["validation_average"] = {
            "num_checks": len(hits),
            "goal_hit_rate": statistics.fmean(hits),
            "optimal_path_rate": statistics.fmean(
                validation[epoch]["optimal_path_rate"] for epoch in validation
            ),
            "loop_rate": statistics.fmean(
                validation[epoch]["loop_rate"] for epoch in validation
            ),
            "broken_rate": statistics.fmean(
                validation[epoch]["broken_rate"] for epoch in validation
            ),
        }
        # 后 30% 验证轮次的均值（更能反映收敛后的水平）
        tail = sorted(validation)[-max(1, len(validation) // 3):]
        summary["validation_tail_average"] = {
            "epochs": tail,
            "goal_hit_rate": statistics.fmean(
                validation[epoch]["goal_hit_rate"] for epoch in tail
            ),
            "optimal_path_rate": statistics.fmean(
                validation[epoch]["optimal_path_rate"] for epoch in tail
            ),
        }

    if args.test_eval and Path(args.test_eval).exists():
        with open(args.test_eval, "r", encoding="utf-8") as handle:
            summary["test"] = json.load(handle)
    if args.dataset_summary and Path(args.dataset_summary).exists():
        with open(args.dataset_summary, "r", encoding="utf-8") as handle:
            summary["dataset"] = json.load(handle)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)

    lines = [f"run: {run_dir}"]
    if checkpoint_meta.get("best.pt"):
        lines.append(
            f"best checkpoint: epoch {checkpoint_meta['best.pt'].get('epoch')} "
            f"(metric {checkpoint_meta['best.pt'].get('best_metric')})"
        )
    if "best_validation" in summary:
        best = summary["best_validation"]
        lines.append(
            f"best validation: epoch {best['epoch']}  goal_hit={best['goal_hit_rate']:.4f}  "
            f"optimal={best['optimal_path_rate']:.4f}  cost_ratio={best['success_cost_ratio']:.4f}"
        )
        avg = summary["validation_average"]
        lines.append(
            f"validation average over {avg['num_checks']} checks: "
            f"goal_hit={avg['goal_hit_rate']:.4f}  optimal={avg['optimal_path_rate']:.4f}  "
            f"loop={avg['loop_rate']:.4f}  broken={avg['broken_rate']:.4f}"
        )
        if "validation_tail_average" in summary:
            tail = summary["validation_tail_average"]
            lines.append(
                f"validation tail average (epochs {tail['epochs'][0]}..{tail['epochs'][-1]}): "
                f"goal_hit={tail['goal_hit_rate']:.4f}  optimal={tail['optimal_path_rate']:.4f}"
            )
    if "train_first_last" in summary:
        first_last = summary["train_first_last"]
        lines.append(
            f"train loss first->last: {first_last['loss'][0]:.4f} -> {first_last['loss'][-1]:.4f}"
        )
        lines.append(
            f"train x0 acc first->last: {first_last['x0_acc'][0]:.4f} -> {first_last['x0_acc'][-1]:.4f}"
        )
    if "test" in summary:
        metrics = summary["test"].get("metrics", {})
        lines.append("test set:")
        for key in ("num_queries", "goal_hit_rate", "optimal_path_rate", "success_cost_ratio",
                    "loop_rate", "broken_rate", "mean_elapsed"):
            if key in metrics:
                lines.append(f"  {key}: {metrics[key]}")

    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {run_dir / 'summary.json'}, {run_dir / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
