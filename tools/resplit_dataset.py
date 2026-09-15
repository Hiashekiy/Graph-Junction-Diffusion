"""把若干份已生成的数据集**按图**重新划分成 train / val / test。

为什么要单独有这么一个工具：数据集一旦被合并（``tools/merge_datasets.py``），原来的
train/val 划分就被"焊死"在里面了。想换划分比例、或者想要一个原本没有的 test split，
只能重新生成 —— 但重新生成需要上游数据（例如 ``data/processed/v1/*.pt``）还在。
这个工具直接在**成品 pkl** 上重划分，不依赖任何上游。

--------------------------------------------------------------------------
两件必须做对的事

**1. 按图划分，不能按样本。** 项目的生成器是"一张图 + 多个 OD query"，同一张图的
不同 query 共享图结构。按样本随机切会让同一张图同时出现在 train 和 test 里 ——
模型见过测试图的结构，指标虚高。所以这里直接复用
:func:`src.data.dataset_builder.split_dataset`（它按 ``graph_id`` 归组后再切）。
划分完自动跑 :mod:`tools.check_leakage` 的指纹核对。

**2. ``graph_id`` 必须重新编号成全局唯一。** 每份输入文件里的 ``graph_id`` 都是
**各自 0..N 编号**的（``merge_datasets.py`` 只保证单次合并的输出内唯一）。直接拼起来
会让"A 文件的图 0"和"B 文件的图 0"撞成一个键，把两张无关的图绑进同一个 split。
所以这里以 ``(source_file, graph_id)`` 为原始身份重新分配全局 id。

--------------------------------------------------------------------------
分层

``--stratify-by``（默认 ``source_file``）对每个来源分别划分，再拼起来。这样各来源在
三个 split 里的占比与总体完全一致，而不是随机划分下的"期望一致"。来源数很少
（合并链顶端通常 2~4 个），所以不会有分层过细导致的空 split。

用法::

    python tools/resplit_dataset.py \
        --input data/unweighted/unweighted_train.pkl \
        --input data/unweighted/unweighted_val.pkl \
        --out-dir data/unweighted --name unweighted --seed 0

    # 换比例 / 不重编号（调试用）
    python tools/resplit_dataset.py --input a.pkl --input b.pkl \
        --train 0.7 --val 0.15 --test 0.15 --no-rekey
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GraphQueryDataset  # noqa: E402
from src.data.dataset_builder import split_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按图重新划分数据集 train/val/test")
    parser.add_argument("--input", action="append", required=True, help="输入 pkl，可重复")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", required=True, help="输出前缀，例如 unweighted")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stratify-by", default="source_file",
        help="按哪个 meta 键分层（默认 source_file）。给空字符串 '' 表示不分层",
    )
    parser.add_argument(
        "--no-rekey", action="store_true",
        help="不重新编号 graph_id（仅当所有输入的 id 空间本来就不重叠时才安全）",
    )
    parser.add_argument(
        "--drop-summary", action="append", default=[],
        help="顺便删掉过期的 summary json（可重复），例如旧的 *_train_summary.json",
    )
    return parser.parse_args()


def graph_key(sample, index: int) -> Tuple[str, Any]:
    """一张图的原始身份。``graph_id == -1`` 的样本按"一图一样本"处理。"""
    gid = sample.meta.get("graph_id", -1)
    source = str(sample.meta.get("source_file", ""))
    if gid is None or gid == -1:
        return (source, f"sample-{index}")
    return (source, gid)


def main() -> int:
    args = parse_args()
    fractions = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SystemExit(f"train/val/test 必须和为 1，收到 {fractions}")

    # ---- 1) 读入并拼起来，同时建立全局唯一的 graph_id ----
    samples: List[Any] = []
    per_input: List[Tuple[str, int, int]] = []
    for raw in args.input:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise SystemExit(f"missing input {path}")
        dataset = GraphQueryDataset.load(path)
        per_input.append((str(path.relative_to(PROJECT_ROOT)), len(dataset), len(
            {graph_key(s, i) for i, s in enumerate(dataset)}
        )))
        samples.extend(dataset.samples)

    if not args.no_rekey:
        mapping: Dict[Tuple[str, Any], int] = {}
        for index, sample in enumerate(samples):
            key = graph_key(sample, index)
            if key not in mapping:
                mapping[key] = len(mapping)
            sample.meta["graph_id"] = mapping[key]
        print(f"rekeyed graph_id: {len(mapping)} graphs -> 0..{len(mapping) - 1}")

    num_graphs = len({s.meta.get("graph_id") for s in samples})
    print(f"loaded {len(samples)} queries / {num_graphs} graphs from {len(args.input)} file(s)")
    for name, count, graphs in per_input:
        print(f"    {count:6d} queries / {graphs:5d} graphs  <- {name}")

    # ---- 2) 分层：每个来源单独按图划分，再合并 ----
    strata: Dict[Any, List[Any]] = defaultdict(list)
    strat_key = args.stratify_by.strip()
    for index, sample in enumerate(samples):
        key = sample.meta.get(strat_key, "all") if strat_key else "all"
        strata[key].append(sample)
    print(f"strata ({strat_key or 'none'}): "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(strata.items(), key=lambda kv: -len(kv[1]))))

    splits: Dict[str, List[Any]] = {name: [] for name in fractions}
    for key, group in sorted(strata.items(), key=lambda kv: str(kv[0])):
        part = split_dataset(GraphQueryDataset(group, name=str(key)), fractions, seed=args.seed)
        for name in fractions:
            splits[name].extend(part[name].samples)

    # ---- 3) 保存 + 自检 ----
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"input": [p[0] for p in per_input], "seed": args.seed,
                               "stratify_by": strat_key, "fractions": fractions, "splits": {}}
    graph_sets: Dict[str, set] = {}
    for name in fractions:
        dataset = GraphQueryDataset(splits[name], name=f"{args.name}_{name}")
        path = dataset.save(out_dir / f"{args.name}_{name}.pkl")
        graph_sets[name] = {
            s.meta.get("graph_id") for s in splits[name] if s.meta.get("graph_id", -1) != -1
        }
        summary["splits"][name] = dataset.summary()
        summary["splits"][name]["num_graphs"] = len(graph_sets[name])
        summary["splits"][name]["composition"] = dict(
            Counter(str(s.meta.get("source_file", "")) for s in splits[name])
        )
        print(
            f"{name:>5}: {len(splits[name]):>5} queries / {len(graph_sets[name]):>5} graphs "
            f"-> {path}"
        )

    overlap = (
        (graph_sets["train"] & graph_sets["val"])
        | (graph_sets["train"] & graph_sets["test"])
        | (graph_sets["val"] & graph_sets["test"])
    )
    summary["graph_overlap"] = len(overlap)
    print(f"\ntopology leakage check: graph_id 交集 = {len(overlap)} "
          f"{'[OK]' if not overlap else '[FAIL]'}")
    if overlap:
        return 1

    summary_path = out_dir / f"{args.name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
    print(f"summary -> {summary_path}")

    for stale in args.drop_summary:
        path = PROJECT_ROOT / stale
        if path.exists():
            path.unlink()
            print(f"removed stale summary -> {path}")

    print("\n下一步（独立的指纹核对，比 graph_id 更硬）：")
    pairs = " ".join(str(out_dir / f"{args.name}_{n}.pkl") for n in fractions)
    print(f"    python tools/check_leakage.py --all {pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
