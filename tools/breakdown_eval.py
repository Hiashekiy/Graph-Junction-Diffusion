"""按难度 / 结构模式拆分测试结果（用 eval 的逐 query 记录 + 数据集 meta）。

用法：python tools/breakdown_eval.py outputs/runs/controlled_unweighted/eval_test.json \
        --data data/unweighted/unweighted_test.pkl

分桶口径集中在 ``src/evaluation/buckets.py``（tools/compare_runs.py 用的是同一套）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="difficulty / mode breakdown of an eval run")
    parser.add_argument("eval_json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from src.data.dataset import GraphQueryDataset
    from src.evaluation.buckets import breakdown

    with open(args.eval_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["records"]
    dataset = GraphQueryDataset.load(args.data)
    result = breakdown(records, dataset)

    lines = []
    for name in sorted(result, key=lambda key: (key != "ALL", key)):
        stats = result[name]
        lines.append(
            "{0:<28} n={1:>3}  goal_hit={2:.3f}  optimal={3:.3f}  cost={4:.3f}  "
            "loop={5:.3f}  broken={6:.3f}".format(
                name,
                int(stats["num_queries"]),
                stats["goal_hit_rate"],
                stats["optimal_path_rate"],
                stats["success_cost_ratio"],
                stats["loop_rate"],
                stats["broken_rate"],
            )
        )
    print("\n".join(lines))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, ensure_ascii=False)
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
