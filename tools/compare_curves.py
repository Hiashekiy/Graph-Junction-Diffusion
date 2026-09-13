"""按**真实 epoch** 对齐比较两次训练的曲线（train loss / 验证 goal hit 等）。

用法::

    python tools/compare_curves.py \
        --a outputs/runs/v2_controlled_100ep \
        --b outputs/runs/v2_controlled_100ep_flow3 \
        --label-a "flow_steps=1" --label-b "flow_steps=3" \
        --key train_loss --key goal_hit_rate

为什么要专门写一个：

* 老 run 的 ``history.json`` 可能缺前半段且没有 ``epoch`` 字段（基线只剩
  epoch 21-100），直接按下标对齐会拿 A 的 epoch 5 去比 B 的 epoch 25；
* 曲线只在**共同 epoch** 上比较才有意义，所以这里取交集；
* 顺便打印两个 run 各自的 epoch 偏移与模型配置（flow_steps / 参数量）。

只读工具：不加载模型，训练在跑时也能用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.history import (  # noqa: E402
    aligned_curves,
    load_history,
    matched_values,
    validation_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="align two runs by true epoch")
    parser.add_argument("--a", required=True, help="run 目录")
    parser.add_argument("--b", required=True, help="run 目录")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument(
        "--key",
        dest="keys",
        action="append",
        default=None,
        help="要比较的曲线，可重复；默认 train_loss 与 goal_hit_rate",
    )
    parser.add_argument("--every", type=int, default=1, help="每 n 个共同 epoch 打一行")
    parser.add_argument("--out", default=None, help="可选：把对齐结果写成 json")
    return parser.parse_args()


def describe(run_dir: Path) -> dict:
    info = {"run_dir": str(run_dir)}
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        from src.utils.config import load_config

        config = load_config(config_path)
        info["flow_steps"] = int(config.get("model.flow_steps", 1))
        info["run_name"] = str(config.get("paths.run_name", "?"))
    history = load_history(run_dir)
    info["records"] = len(history)
    return info


def main() -> int:
    args = parse_args()
    keys: List[str] = args.keys or ["train_loss", "goal_hit_rate"]
    run_a, run_b = Path(args.a), Path(args.b)

    curves: dict = {}
    for label, run_dir in ((args.label_a, run_a), (args.label_b, run_b)):
        info = describe(run_dir)
        run_curves, offset = aligned_curves(run_dir)
        curves[label] = run_curves
        print(
            f"{label}: {run_dir}  flow_steps={info.get('flow_steps', '?')}  "
            f"records={info['records']}  epoch_offset={offset}"
        )
        history = load_history(run_dir)
        checks = validation_records(history, offset or 0)
        if checks:
            print(
                f"    validation checks: {len(checks)} "
                f"(epoch {checks[0][0]}..{checks[-1][0]})"
            )

    curves_a = curves[args.label_a]
    curves_b = curves[args.label_b]

    payload = {"a": args.label_a, "b": args.label_b, "curves": {}}
    for key in keys:
        pairs = matched_values(curves_a, curves_b, key)
        print(f"\n== {key}  (common epochs: {len(pairs)})")
        if not pairs:
            print("  no common epochs")
            continue
        print(f"{'epoch':>6}  {args.label_a:>14}  {args.label_b:>14}  {'delta':>10}")
        rows = []
        for index, (epoch, value_a, value_b) in enumerate(pairs):
            rows.append({"epoch": epoch, "a": value_a, "b": value_b, "delta": value_b - value_a})
            if index % args.every == 0 or index == len(pairs) - 1:
                print(f"{epoch:>6}  {value_a:>14.4f}  {value_b:>14.4f}  {value_b - value_a:>+10.4f}")
        payload["curves"][key] = rows
        if len(pairs) >= 2:
            mean_a = sum(row["a"] for row in rows) / len(rows)
            mean_b = sum(row["b"] for row in rows) / len(rows)
            # 单点比对噪声很大（验证集只有 300 条，goal_hit 的标准差约 ±0.03），
            # 所以同时给出共同 epoch 上的窗口均值：这才是能下结论的量。
            print(
                f"  windowed mean over {len(rows)} common epochs: "
                f"{args.label_a}={mean_a:.4f}  {args.label_b}={mean_b:.4f}  "
                f"delta={mean_b - mean_a:+.4f}"
            )

    if args.out:
        import json

        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
