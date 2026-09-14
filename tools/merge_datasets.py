"""把多份 dataset pkl 合并成一份（用于给训练集"补长链样本"）。

背景：现有训练集里决策数 >= 9 的样本只占 5%（119/2400），而长链上多轮交流的收益
已经用专门的测试集证实（README §16）。要让模型真的擅长长链，需要把训练分布往长链
偏——做法是**另外生成一批长链样本**再合并，而不是重跑整个生成器（那样会连 test 一起
换掉，之前的结论就不可比了）。

这个工具做的事：

1. 读入多份 ``GraphQueryDataset`` pkl；
2. 把 ``graph_id`` 重新编号成全局唯一（否则不同文件里的 0..N 会撞车）；
3. 用一个固定 seed 打乱顺序（避免"长链样本全挤在文件尾部"）；
4. 打印合并后的决策数直方图与长链占比，并把组成写进 ``<out>_summary.json``。

用法::

    python tools/merge_datasets.py --out data/long/controlled_longmix_train.pkl \
        --input data/controlled/controlled_train.pkl --input data/long/controlled_longpool.pkl \
        --limit 900:1 --seed 0

``--limit N:INDEX`` 表示"第 INDEX 份输入最多取 N 条"（``N:1`` = 第二份最多 900 条），
用于把一份长链池拆成训练补充 + 验证补充（两份不能有重叠样本）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="merge dataset pkl files")
    parser.add_argument("--out", required=True, help="输出 pkl 路径")
    parser.add_argument(
        "--input", action="append", default=[], help="输入 pkl，可重复；按给定顺序处理",
    )
    parser.add_argument(
        "--limit", action="append", default=[],
        help="'N:INDEX' 表示第 INDEX 份（从 0 数）输入最多取 N 条",
    )
    parser.add_argument(
        "--skip", action="append", default=[],
        help="'N:INDEX' 表示第 INDEX 份输入跳过前 N 条（配合 --limit 把一份池子拆两份）",
    )
    parser.add_argument(
        "--sample", action="append", default=[],
        help="'N:INDEX' 表示第 INDEX 份输入**随机抽** N 条（无放回，种子 = --seed + INDEX）。"
             "与 --limit 的区别：--limit 取文件前缀，而按图类型分块存储的数据集"
             "（例如 V1 转换出来的）取前缀会只拿到一种图类型",
    )
    parser.add_argument("--seed", type=int, default=0, help="打乱顺序用的种子")
    parser.add_argument("--min-decisions", type=int, default=9,
                        help="统计报告里『长链』的门槛（默认 9）")
    return parser.parse_args()


def _limits_map(values: Sequence[str]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for item in values:
        count, _, index = str(item).partition(":")
        out[int(index)] = int(count)
    return out

def main() -> int:
    args = parse_args()
    if not args.input:
        raise SystemExit("--input is required (repeatable)")

    from src.data.dataset import GraphQueryDataset

    limits = _limits_map(args.limit)
    skips = _limits_map(args.skip)
    samples_map = _limits_map(args.sample)
    merged: List = []
    composition: List[Dict[str, object]] = []
    next_graph_id = 0

    for index, path in enumerate(args.input):
        dataset = GraphQueryDataset.load(path)
        samples = list(dataset)
        skip = skips.get(index, 0)
        if index in samples_map:
            # 随机抽样（无放回）：按图类型分块存储的数据集不能用前缀取子集
            count = min(int(samples_map[index]), len(samples))
            samples = random.Random(args.seed + index).sample(samples, count)
        elif skip or index in limits:
            samples = samples[skip : skip + limits.get(index, len(samples))]
        elif skip:
            samples = samples[skip:]
        local_ids = {}
        for sample in samples:
            # graph_id 重新编号，保证全局唯一（跨文件不会撞车）
            old = sample.meta.get("graph_id")
            if old not in local_ids:
                local_ids[old] = next_graph_id
                next_graph_id += 1
            sample.meta["graph_id"] = local_ids[old]
            sample.meta["source_file"] = Path(path).name
        merged.extend(samples)
        composition.append(
            {
                "file": Path(path).name,
                "used": len(samples),
                "available": len(dataset),
                "graphs": len(local_ids),
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(merged)

    decisions = Counter(int(sample.num_decisions) for sample in merged)
    total = len(merged)
    long_count = sum(count for value, count in decisions.items() if value >= args.min_decisions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = GraphQueryDataset(merged, name=out_path.stem)
    combined.save(out_path)

    summary = {
        "name": out_path.stem,
        "num_queries": total,
        "num_graphs": next_graph_id,
        "shuffle_seed": args.seed,
        "composition": composition,
        "decisions_histogram": dict(sorted(decisions.items())),
        "long_decisions_threshold": args.min_decisions,
        "long_count": long_count,
        "long_fraction": long_count / total if total else 0.0,
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)

    print(f"merged {total} queries ({next_graph_id} graphs) -> {out_path}")
    for row in composition:
        print(f"  {row['file']:<34} used {row['used']:>5} / {row['available']:>5}  "
              f"graphs {row['graphs']:>4}")
    print(f"decisions histogram: {dict(sorted(decisions.items()))}")
    pct = long_count / total * 100 if total else 0.0
    print(f">= {args.min_decisions} 决策: {long_count}/{total} = {pct:.1f}%")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
