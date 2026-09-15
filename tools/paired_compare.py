"""配对比较两次评测（同一批 query 逐条对比 + McNemar 精确检验）。

为什么需要它：验证/测试集只有 300 条，goal_hit 的标准差约 ±0.03，单看两个率
（0.6153 vs 0.6833）分不清是"真的更好"还是随机波动。两次评测跑的是**同一批
query**，可以逐条配对，只有"只有 A 达标 / 只有 B 达标"这两格携带信息。

统计口径在 ``src/evaluation/paired.py``（有单元测试）。

用法::

    python tools/paired_compare.py \
        --a outputs/runs/controlled_unweighted/eval_test.json \
        --b outputs/runs/controlled_weighted/eval_test.json \
        --data data/unweighted/unweighted_test.pkl --metric goal_hit \
        --label-a "flow_steps=1" --label-b "flow_steps=3"

``--metric`` 可选 ``goal_hit`` / ``optimal`` / ``broken`` / ``loop``。给 ``--data``
就同时按难度 / 结构模式 / source / 决策数分组做配对比较。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.paired import METRICS, paired_counts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="paired (per-query) comparison of two evals")
    parser.add_argument("--a", required=True, help="baseline eval json")
    parser.add_argument("--b", required=True, help="new eval json")
    parser.add_argument("--data", default=None, help="同一个 split 的 pkl，用于分组")
    parser.add_argument("--metric", default="goal_hit", choices=list(METRICS))
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _print_row(name: str, stats: Dict[str, Any], label_a: str, label_b: str) -> None:
    print(
        f"{name:<28} n={stats['n']:>3}  "
        f"{label_a}={stats['rate_a']:.3f}  {label_b}={stats['rate_b']:.3f}  "
        f"delta={stats['delta']:+.3f}  "
        f"only_{label_a}={stats['only_a']:>2}  only_{label_b}={stats['only_b']:>2}  "
        f"both={stats['both']:>3}  neither={stats['neither']:>3}  "
        f"p={stats['p_value']:.4f}"
    )


def _load_records(path: str):
    """兼容两种 json：``eval_test.json``（``{"records": [...]}`` 或带 metrics 的字典）
    与 ``val_records_epoch*.json``（裸的列表）。"""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise SystemExit(f"cannot find a record list in {path}")


def _assert_same_queries(records_a, records_b, path_a: str, path_b: str) -> None:
    """配对检验的前提：两次评测的第 i 条记录必须是**同一张图、同一个 OD**。

    ``gt_length`` 与 ``optimal_cost`` 只由 query 决定（与模型无关），所以它们逐条
    相等才说明顺序一致。顺序不一致时配对计数毫无意义，必须直接报错而不是给一个
    看起来很正常的 p 值。
    """
    keys = ("gt_length", "optimal_cost")
    for key in keys:
        values_a = [record.get(key) for record in records_a]
        values_b = [record.get(key) for record in records_b]
        if values_a != values_b:
            raise SystemExit(
                f"the two evals are not query-aligned: {key} differs elementwise.\n"
                f"  A = {path_a}\n  B = {path_b}\n"
                "  paired testing requires the same split loaded in the same order; "
                "re-run both evaluations on the same --data file."
            )


def main() -> int:
    args = parse_args()
    records_a = _load_records(args.a)
    records_b = _load_records(args.b)
    if len(records_a) != len(records_b):
        raise SystemExit(
            f"record count differs: {len(records_a)} vs {len(records_b)}; "
            "the two evals must use the same split"
        )
    _assert_same_queries(records_a, records_b, args.a, args.b)

    overall = paired_counts(records_a, records_b, args.metric, list(range(len(records_a))))
    print(f"metric = {args.metric}   (paired McNemar exact test on discordant pairs)")
    print(f"A = {args.label_a}  ({args.a})")
    print(f"B = {args.label_b}  ({args.b})\n")
    _print_row("ALL", overall, args.label_a, args.label_b)

    result: Dict[str, Any] = {"metric": args.metric, "ALL": overall, "buckets": {}}
    if args.data:
        from src.data.dataset import GraphQueryDataset
        from src.evaluation.buckets import bucket_indices

        dataset = GraphQueryDataset.load(args.data)
        if len(dataset) != len(records_a):
            raise SystemExit(f"dataset size {len(dataset)} != record count {len(records_a)}")
        print()
        groups = bucket_indices(dataset)
        for name in sorted(groups, key=lambda key: (key != "ALL", key)):
            if name == "ALL":
                continue
            stats = paired_counts(records_a, records_b, args.metric, groups[name])
            result["buckets"][name] = stats
            _print_row(name, stats, args.label_a, args.label_b)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, ensure_ascii=False)
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
