"""汇总一个 run 的最终统计（按 README 里的报告格式）。

用法：python tools/final_report.py outputs/runs/v2_controlled_100ep_flow3 \
        --baseline outputs/runs/v2_controlled_100ep

只读工具：从 run 目录里已经落盘的产物（history.json / summary.json /
breakdown_test.json / multiseed_goal_hit.json / ablation_flow_steps.json /
val_records_epoch*.json）算出一份可粘贴的统计报告，并把 markdown 写到
<run_dir>/report_final.md。不加载模型，训练中也能跑。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.history import load_history  # noqa: E402


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def val_checks(run_dir: Path) -> List[Dict[str, Any]]:
    """所有 val_records_epoch*.json 的 (epoch, goal_hit, optimal)。"""
    rows = []
    for path in sorted(
        run_dir.glob("val_records_epoch*.json"),
        key=lambda p: int(p.stem.replace("val_records_epoch", "")),
    ):
        epoch = int(path.stem.replace("val_records_epoch", ""))
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not records:
            continue
        hits = sum(1 for r in records if r["goal_hit"])
        opt = sum(1 for r in records if r["optimal"])
        rows.append(
            {
                "epoch": epoch,
                "n": len(records),
                "goal_hit": hits / len(records),
                "optimal": opt / len(records),
            }
        )
    return rows


def fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="final statistics for a run")
    parser.add_argument("run_dir")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history = load_history(run_dir)
    summary = read_json(run_dir / "summary.json") or {}
    breakdown = read_json(run_dir / "breakdown_test.json") or {}
    multiseed = read_json(run_dir / "multiseed_goal_hit.json")
    ablation = read_json(run_dir / "ablation_flow_steps.json")
    config = read_json(run_dir / "run_config.json") or {}

    model_info = summary.get("model", {})
    test = summary.get("test", {})
    metrics = test.get("metrics", {})

    lines: List[str] = []
    add = lines.append

    add(f"# {run_dir.name} 最终统计")
    add("")
    add("## 配置与训练")
    add("")
    training = config.get("training", {})
    diffusion = config.get("diffusion", {})
    add(
        f"- 模型：`flow_steps={model_info.get('flow_steps')}`"
        f"（{model_info.get('label')}），参数 **{model_info.get('parameters'):,}**"
    )
    add(
        f"- 配置：d_model={model_info.get('d_model')}、T={model_info.get('diffusion_T')}、"
        f"batch={model_info.get('batch_size')}、AMP={model_info.get('amp')}、"
        f"lr={model_info.get('lr')}、optimizer={training.get('optimizer')}、"
        f"epochs={training.get('epochs')}"
    )
    seconds = [record.get("train_seconds", 0.0) for record in history]
    if seconds:
        add(
            f"- 训练耗时：{len(history)} epoch，每 epoch 中位数 "
            f"{statistics.median(seconds):.1f}s（min {min(seconds):.1f}s / "
            f"max {max(seconds):.1f}s），总计 **{sum(seconds) / 60:.1f} 分钟**"
        )
    losses = [record["train_loss"] for record in history if "train_loss" in record]
    x0s = [record["train_x0_acc"] for record in history if "train_x0_acc" in record]
    if losses:
        add(
            f"- train loss：{losses[0]:.3f} → {losses[-1]:.3f}"
            + (f"（x0 accuracy {x0s[0]:.3f} → {x0s[-1]:.3f}）" if x0s else "")
        )
    best = summary.get("best_validation", {})
    if best:
        add(
            f"- **best checkpoint = epoch {best.get('epoch')}**"
            f"（val goal_hit {fmt(best.get('goal_hit_rate'))} / "
            f"optimal {fmt(best.get('optimal_path_rate'))}）"
        )
    add(f"- diffusion：T={diffusion.get('T')}、schedule={diffusion.get('schedule')}、"
        f"base_noise={diffusion.get('base_noise')}")

    checks = val_checks(run_dir)
    if checks:
        add("")
        add("## 验证集（每 5 epoch 一次）")
        add("")
        all_goal = [row["goal_hit"] for row in checks]
        all_opt = [row["optimal"] for row in checks]
        add(
            f"- 全部 {len(checks)} 次（epoch {checks[0]['epoch']}–{checks[-1]['epoch']}）："
            f"平均 goal_hit **{fmt(statistics.fmean(all_goal), 3)}** / "
            f"optimal **{fmt(statistics.fmean(all_opt), 3)}**"
        )
        tail = [row for row in checks if row["epoch"] >= 75]
        if tail:
            add(
                f"- 后段 {len(tail)} 次（epoch {tail[0]['epoch']}–{tail[-1]['epoch']}）："
                f"平均 goal_hit **{fmt(statistics.fmean([r['goal_hit'] for r in tail]), 3)}** / "
                f"optimal **{fmt(statistics.fmean([r['optimal'] for r in tail]), 3)}**"
            )
        add("")
        add("| epoch | " + " | ".join(str(row["epoch"]) for row in checks) + " |")
        add("|---" * (len(checks) + 1) + "|")
        add("| goal_hit | " + " | ".join(f"{row['goal_hit']:.3f}" for row in checks) + " |")
        add("| optimal | " + " | ".join(f"{row['optimal']:.3f}" for row in checks) + " |")

    if metrics:
        add("")
        add(f"## 测试集（{test.get('num_queries')} queries，best.pt epoch "
            f"{test.get('checkpoint_epoch')}）")
        add("")
        add("| 指标 | 值 |")
        add("|---|---|")
        for key, label in (
            ("goal_hit_rate", "Goal Hit Rate"),
            ("optimal_path_rate", "Optimal Path Rate"),
            ("success_cost_ratio", "Success Cost Ratio"),
            ("loop_rate", "Loop Rate"),
            ("broken_rate", "Broken Rate"),
            ("mean_elapsed", "单 query 推理（秒）"),
        ):
            if key in metrics:
                digits = 4 if key == "mean_elapsed" else 3
                add(f"| {label} | {fmt(metrics[key], digits)} |")

    if breakdown:
        add("")
        add("## 分层拆解")
        add("")
        for prefix, title in (
            ("difficulty", "难度档（生成时采样的标签）"),
            ("gt_decisions", "GT 决策数（GT 路径上要做几次选择）"),
            ("mode", "结构模式"),
        ):
            rows = {k: v for k, v in breakdown.items() if k.startswith(prefix + "=")}
            if not rows:
                continue
            add(f"**{title}**")
            add("")
            add("| 分组 | n | goal_hit | optimal | goal_hit−optimal |")
            add("|---|---|---|---|---|")
            for name in sorted(rows):
                stats = rows[name]
                add(
                    f"| {name.split('=', 1)[1]} | {int(stats['num_queries'])} | "
                    f"{fmt(stats['goal_hit_rate'], 3)} | {fmt(stats['optimal_path_rate'], 3)} | "
                    f"{stats['goal_hit_rate'] - stats['optimal_path_rate']:+.3f} |"
                )
            add("")

        # 反推"单步决策准确率"：a = goal_hit ^ (1/平均决策数)
        add("**反推单步决策准确率**（把 goal_hit 近似成 a^k，k = 该组平均决策数）")
        add("")
        add("| 决策数分组 | 平均 k | goal_hit | 反推 a |")
        add("|---|---|---|---|")
        dataset_decisions = mean_decisions_by_bucket()
        for name in sorted(k for k in breakdown if k.startswith("gt_decisions=")):
            stats = breakdown[name]
            key = name.split("=", 1)[1]
            mean_k = dataset_decisions.get(key)
            rate = stats["goal_hit_rate"]
            implied = rate ** (1.0 / mean_k) if mean_k and rate > 0 else float("nan")
            add(
                f"| {key} | {fmt(mean_k, 2)} | {fmt(rate, 3)} | {fmt(implied, 4)} |"
            )

    if multiseed:
        add("")
        add("## 多种子配对检验（判断差异是否真实）")
        add("")
        add(
            f"- 种子 {multiseed['seeds']}：A={fmt(multiseed['rate_a'], 4)}、"
            f"B={fmt(multiseed['rate_b'], 4)}、Δ=**{multiseed['delta']:+.4f}**"
        )
        add(
            f"- 配对 t 检验 p={fmt(multiseed['paired_t_p'], 4)}、"
            f"Wilcoxon p={fmt(multiseed['wilcoxon_p'], 4)}、"
            f"bootstrap 95% CI [{multiseed['bootstrap_ci'][0]:+.4f}, "
            f"{multiseed['bootstrap_ci'][1]:+.4f}]"
        )
        deltas = [
            b - a for a, b in zip(multiseed["per_seed_a"], multiseed["per_seed_b"])
        ]
        add(
            f"- 单种子 delta 区间 [{min(deltas):+.4f}, {max(deltas):+.4f}]"
            f"（std {statistics.pstdev(deltas):.4f}）"
        )
        add(f"- 判定：{'显著' if multiseed['paired_t_p'] < 0.05 else '不显著'}")

    if ablation:
        add("")
        add("## 推理轮数 ablation（同一个 checkpoint）")
        add("")
        add("| 推理轮数 | goal_hit | optimal | broken | 秒/query |")
        add("|---|---|---|---|---|")
        for row in ablation:
            add(
                f"| {row['flow_steps']} | {fmt(row['goal_hit_rate'], 3)} | "
                f"{fmt(row['optimal_path_rate'], 3)} | {fmt(row['broken_rate'], 3)} | "
                f"{fmt(row['mean_elapsed'], 4)} |"
            )

    if args.baseline:
        baseline_dir = Path(args.baseline)
        b_checks = val_checks(baseline_dir)
        b_summary = read_json(baseline_dir / "summary.json") or {}
        b_metrics = (b_summary.get("test") or {}).get("metrics", {})
        add("")
        add(f"## 与 {baseline_dir.name} 对照")
        add("")
        add("| | " + f"{baseline_dir.name} | {run_dir.name} |")
        add("|---|---|---|")
        add(
            f"| flow_steps | {(b_summary.get('model') or {}).get('flow_steps')} | "
            f"{model_info.get('flow_steps')} |"
        )
        add(
            f"| 参数量 | {(b_summary.get('model') or {}).get('parameters'):,} | "
            f"{model_info.get('parameters'):,} |"
        )
        b_best = b_summary.get("best_validation", {})
        add(
            f"| best val（epoch）| {fmt(b_best.get('goal_hit_rate'), 3)}"
            f"（{b_best.get('epoch')}）| {fmt(best.get('goal_hit_rate'), 3)}"
            f"（{best.get('epoch')}）|"
        )
        for key, label in (
            ("goal_hit_rate", "test goal_hit"),
            ("optimal_path_rate", "test optimal"),
            ("success_cost_ratio", "test cost ratio"),
            ("broken_rate", "test broken"),
            ("mean_elapsed", "秒/query"),
        ):
            if key in b_metrics and key in metrics:
                digits = 4 if key == "mean_elapsed" else 3
                add(
                    f"| {label} | {fmt(b_metrics[key], digits)} | "
                    f"{fmt(metrics[key], digits)} |"
                )
        if b_checks:
            add(
                f"| 验证全部均值 goal_hit | "
                f"{fmt(statistics.fmean([r['goal_hit'] for r in b_checks]), 3)} | "
                f"{fmt(statistics.fmean([r['goal_hit'] for r in checks]), 3)} |"
            )

    report = "\n".join(lines) + "\n"
    print(report)
    out_path = Path(args.out) if args.out else run_dir / "report_final.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"written: {out_path}")
    return 0


def mean_decisions_by_bucket() -> Dict[str, float]:
    """按 gt_decisions 的 3 个一档分组，算每组的平均决策数（需要 test 集）。"""
    from src.data.dataset import GraphQueryDataset

    dataset = GraphQueryDataset.load("data/controlled/controlled_test.pkl")
    buckets: Dict[str, List[int]] = {}
    for sample in dataset:
        low = sample.num_decisions // 3 * 3
        buckets.setdefault(f"{low}-{low + 2}", []).append(sample.num_decisions)
    return {key: statistics.fmean(values) for key, values in buckets.items()}


if __name__ == "__main__":
    raise SystemExit(main())
