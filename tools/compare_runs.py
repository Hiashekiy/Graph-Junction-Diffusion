"""把两次 evaluate.py 的结果并排比较（逐 bucket + 整体指标）。

用法::

    python tools/compare_runs.py \
        --a outputs/runs/controlled_unweighted/eval_test.json \
        --b outputs/runs/controlled_weighted/eval_test.json \
        --data data/unweighted/unweighted_test.pkl

`--data` 可选：给了就按难度 / 结构模式 / source / gt_decisions 分组比较
（分桶口径来自 ``src/evaluation/buckets.py``，与 tools/breakdown_eval.py 共用），
不给就只比整体指标。

只读工具：不写任何文件，也不加载模型，因此可以在训练还在跑的时候安全使用。
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

from src.evaluation.buckets import METRICS, bucket_indices, summarize_records  # noqa: E402


def load_eval(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def fmt(value: float) -> str:
    if value != value:  # NaN
        return "   n/a"
    return f"{value:6.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="side-by-side comparison of two eval runs")
    parser.add_argument("--a", required=True, help="baseline eval json")
    parser.add_argument("--b", required=True, help="new eval json")
    parser.add_argument("--data", default=None, help="same test split, for bucketing")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out", default=None, help="可选：把逐 bucket 结果写成 json")
    args = parser.parse_args()

    payload_a = load_eval(args.a)
    payload_b = load_eval(args.b)
    records_a = payload_a["records"]
    records_b = payload_b["records"]
    if len(records_a) != len(records_b):
        raise SystemExit(
            f"record count differs: {len(records_a)} vs {len(records_b)}; "
            "the two eval runs must use the same split"
        )

    print(f"A = {args.label_a}  ({args.a})")
    print(f"B = {args.label_b}  ({args.b})")
    for name, payload in (("A", payload_a), ("B", payload_b)):
        metrics = payload.get("metrics") or {}
        print(
            f"  {name}: goal_hit={fmt(metrics.get('goal_hit_rate', float('nan')))} "
            f"optimal={fmt(metrics.get('optimal_path_rate', float('nan')))} "
            f"cost={fmt(metrics.get('success_cost_ratio', float('nan')))} "
            f"loop={fmt(metrics.get('loop_rate', float('nan')))} "
            f"broken={fmt(metrics.get('broken_rate', float('nan')))}"
        )

    if not args.data:
        return 0

    from src.data.dataset import GraphQueryDataset

    dataset = GraphQueryDataset.load(args.data)
    if len(dataset) != len(records_a):
        raise SystemExit(
            f"dataset size {len(dataset)} != record count {len(records_a)}; "
            "pass the same test split that produced both eval jsons"
        )

    groups = bucket_indices(dataset)
    header = (
        f"{'bucket':<28}{'n':>5}  "
        f"{'goal_hit A':>10}{'B':>8}{'delta':>8}   "
        f"{'optimal A':>10}{'B':>8}{'delta':>8}   "
        f"{'cost A':>9}{'B':>8}{'delta':>8}"
    )
    print()
    print(header)
    print("-" * len(header))
    rows: List[Dict[str, Any]] = []
    for name in sorted(groups, key=lambda key: (key != "ALL", key)):
        index = groups[name]
        stats_a = summarize_records([records_a[i] for i in index])
        stats_b = summarize_records([records_b[i] for i in index])
        row = {"bucket": name, "n": len(index)}
        for key in METRICS:
            row[f"{key}_a"] = stats_a[key]
            row[f"{key}_b"] = stats_b[key]
            row[f"{key}_delta"] = stats_b[key] - stats_a[key]
        rows.append(row)
        print(
            f"{name:<28}{len(index):>5}  "
            f"{fmt(stats_a['goal_hit_rate']):>10}{fmt(stats_b['goal_hit_rate']):>8}"
            f"{stats_b['goal_hit_rate'] - stats_a['goal_hit_rate']:>+8.3f}   "
            f"{fmt(stats_a['optimal_path_rate']):>10}{fmt(stats_b['optimal_path_rate']):>8}"
            f"{stats_b['optimal_path_rate'] - stats_a['optimal_path_rate']:>+8.3f}   "
            f"{fmt(stats_a['success_cost_ratio']):>9}{fmt(stats_b['success_cost_ratio']):>8}"
            f"{stats_b['success_cost_ratio'] - stats_a['success_cost_ratio']:>+8.3f}"
        )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "a": {"path": args.a, "label": args.label_a},
                    "b": {"path": args.b, "label": args.label_b},
                    "buckets": rows,
                },
                handle,
                indent=1,
                ensure_ascii=False,
            )
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
