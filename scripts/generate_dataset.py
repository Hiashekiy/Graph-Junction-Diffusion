"""生成数据集（实施指南第 2 阶段 + 《V2 数据集生成指南》第 13-18 节）。

用法::

    # Controlled Junction Graph（推荐，指南第 17 节的默认配置）
    python scripts/generate_dataset.py --config configs/controlled_unweighted.yaml

    # 小规模调试
    python scripts/generate_dataset.py --config configs/controlled_unweighted.yaml \
        --name debug --set data.num_samples=300

输出目录默认取配置里的 ``paths.data_dir``（``configs/controlled_unweighted.yaml`` 指向 ``data/controlled``）::

    <data-dir>/<name>_train.pkl / _val.pkl / _test.pkl
    <data-dir>/<name>_summary.json      # 指南第 16 节的全部指标

数据集按来源分组存放（``data/controlled`` / ``long`` / ``oldv1`` / ``mixed`` / ``smoke``），
生成别的族时用 ``--data-dir`` 指定，例如长链集 ``--data-dir data/long``。

``--split-by-graph``（默认开）保证同一张底层图的所有 OD query 落在同一个 split。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import (  # noqa: E402
    build_dataset,
    dataset_statistics,
    graph_ids_of,
    split_dataset,
)
from src.data.dataset_builder import CONTROLLED_JUNCTION  # noqa: E402
from src.training.setup import edge_weight_config, generator_config  # noqa: E402
from src.utils.config import flatten_overrides, load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate V2 graph-query dataset")
    parser.add_argument("--config", default="configs/controlled_unweighted.yaml")
    parser.add_argument("--name", default=None, help="dataset name (default: <graph_type>_<n>)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--min-decisions",
        type=int,
        default=0,
        help="只保留 GT 决策数 >= N 的样本（造长决策链专项评测集用；默认 0 = 不过滤）",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="不划分 train/val/test，整份存成 data/<name>.pkl（专项评测集用）",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置项，可重复：--set training.lr=1e-4 --set data.num_samples=32",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # --set 可重复出现，每次一个值；用 flatten_overrides 兼容 append / nargs='*' 两种写法
    overrides = flatten_overrides(args.overrides)
    config = load_config(args.config, overrides)
    seed = int(config.get("seed", 0))
    set_seed(seed)

    data_cfg = config.section("data")
    num_nodes = data_cfg.get("num_nodes", [20, 40])
    num_nodes = (
        [int(num_nodes[0]), int(num_nodes[1])]
        if isinstance(num_nodes, (list, tuple))
        else int(num_nodes)
    )
    num_samples = int(data_cfg.get("num_samples", 3000))
    graph_type = str(data_cfg.get("graph_type", "er"))

    start = time.time()
    print(
        f"generating {num_samples} samples (graph_type={graph_type}) ...",
        flush=True,
    )
    dataset = build_dataset(
        num_samples=num_samples,
        graph_type=graph_type,
        num_nodes=num_nodes,
        min_od_distance=int(data_cfg.get("min_od_distance", 5)),
        seed=seed,
        queries_per_graph=int(data_cfg.get("num_queries_per_graph", 1)),
        generator_cfg=generator_config(data_cfg),
        weighted=bool(data_cfg.get("weighted", False)),
        component_fallback=bool(data_cfg.get("component_fallback", True)),
        progress_every=max(200, num_samples // 10),
        min_decisions=int(args.min_decisions),
        # Weighted 扩展：data.edge_weight（分布 + 范围）；无权配置这里是 {}
        edge_weight=edge_weight_config(data_cfg),
    )
    elapsed = time.time() - start
    print(
        f"generated {len(dataset)} samples in {elapsed:.1f}s "
        f"({len(dataset) / max(elapsed, 1e-6):.0f} samples/s)",
        flush=True,
    )

    stats = dataset_statistics(dataset)
    print("=== dataset statistics（指南第 16 节）===", flush=True)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    # Weighted sanity check（方案第 11 节）：数据集太容易必须当场喊出来，
    # 否则"模型其实无视了 edge weight"这件事要等到实验做完才会被发现。
    for warning in stats.get("weighted", {}).get("warnings", []):
        print(f"[weighted WARNING] {warning}", flush=True)

    data_dir = Path(args.data_dir or str(config.get("paths.data_dir", "data")))
    name = args.name or f"{graph_type}_{num_samples}"

    if args.no_split:
        # 专项评测集：整份一个文件，不做划分（先于 split 计算，避免白算一遍）
        path = dataset.save(data_dir / f"{name}.pkl")
        filter_info = getattr(dataset, "filter_info", {})
        payload = {
            "dataset": stats,
            "seed": seed,
            "split": "none (dedicated evaluation set)",
            "filter": filter_info,
            "graph_ids": len(graph_ids_of(dataset)),
        }
        summary_path = data_dir / f"{name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)
        print(f"dataset -> {path} ({len(dataset)} queries, no split)")
        print(f"filter  -> {filter_info}")
        print(f"summary -> {summary_path}")
        return 0

    split_cfg = config.get("split", {})
    split_by_graph = bool(split_cfg.get("split_by_graph", True))
    if split_by_graph:
        splits = split_dataset(
            dataset,
            {
                "train": float(split_cfg.get("train", 0.8)),
                "val": float(split_cfg.get("val", 0.1)),
                "test": float(split_cfg.get("test", 0.1)),
            },
            seed=seed,
        )
    else:
        # 明确选择按 query 随机划分（会引入 topology leakage，仅供对照）
        print("WARNING: split_by_graph=false -> topology leakage is possible", flush=True)
        splits = split_dataset(
            dataset,
            {
                "train": float(split_cfg.get("train", 0.8)),
                "val": float(split_cfg.get("val", 0.1)),
                "test": float(split_cfg.get("test", 0.1)),
            },
            seed=seed,
        )

    data_dir = Path(args.data_dir or str(config.get("paths.data_dir", "data")))
    name = args.name or f"{graph_type}_{num_samples}"
    split_summary = {}
    for split_name, split in splits.items():
        path = split.save(data_dir / f"{name}_{split_name}.pkl")
        split_summary[split_name] = split.summary()
        print(
            f"{split_name:>5}: {len(split):>5} samples, {len(graph_ids_of(split)):>4} graphs "
            f"-> {path}",
            flush=True,
        )

    payload = {"dataset": stats, "splits": split_summary, "seed": seed}
    summary_path = data_dir / f"{name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    print(f"summary -> {summary_path}")
    if graph_type == CONTROLLED_JUNCTION:
        train_ids = graph_ids_of(splits["train"])
        val_ids = graph_ids_of(splits["val"])
        test_ids = graph_ids_of(splits["test"])
        leakage = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
        print(f"topology leakage check: {sorted(leakage) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
