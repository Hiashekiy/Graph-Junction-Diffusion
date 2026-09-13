"""按难度 / 结构模式拆分测试结果（用 eval 的逐 query 记录 + 数据集 meta）。

用法：python tools/breakdown_eval.py outputs/runs/v2_controlled_100ep/eval_test.json \
        --data data/controlled_test.pkl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def summarize(records):
    total = len(records)
    if total == 0:
        return {}
    hits = [r for r in records if r["goal_hit"]]
    ratios = [r["cost_ratio"] for r in hits if r["cost_ratio"] not in (None, float("inf"))]
    return {
        "num_queries": total,
        "goal_hit_rate": len(hits) / total,
        "optimal_path_rate": sum(1 for r in records if r["optimal"]) / total,
        "success_cost_ratio": (statistics.fmean(ratios) if ratios else float("nan")),
        "loop_rate": sum(1 for r in records if r["status"] == "loop") / total,
        "broken_rate": sum(1 for r in records if r["status"] == "broken") / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="difficulty / mode breakdown of an eval run")
    parser.add_argument("eval_json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from src.data.dataset import GraphQueryDataset

    with open(args.eval_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["records"]
    dataset = GraphQueryDataset.load(args.data)
    if len(records) != len(dataset):
        raise SystemExit(
            f"record count {len(records)} != dataset size {len(dataset)}; "
            "rerun evaluate.py on the same split"
        )

    groups = defaultdict(list)
    for sample, record in zip(dataset, records):
        groups["ALL"].append(record)
        groups[f"difficulty={sample.meta.get('difficulty', 'n/a')}"].append(record)
        groups[f"mode={sample.meta.get('mode', 'n/a')}"].append(record)
        groups[
            "source=" + ("forced" if sample.segments.source_forced_edge_ids else "decision")
        ].append(record)
        groups[f"gt_decisions={sample.num_decisions // 3 * 3}-{sample.num_decisions // 3 * 3 + 2}"].append(
            record
        )

    result = {}
    lines = []
    for name in sorted(groups, key=lambda key: (key != "ALL", key)):
        stats = summarize(groups[name])
        result[name] = stats
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
