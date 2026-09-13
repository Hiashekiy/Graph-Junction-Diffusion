"""生成数据集（实施指南第 2 阶段）。

用法::

    python scripts/generate_dataset.py --config configs/graph_flow.yaml
    python scripts/generate_dataset.py --config configs/graph_flow.yaml \
        --set data.num_samples=64 --name tiny

输出（默认 ``data/``）::

    data/<name>_train.pkl
    data/<name>_val.pkl
    data/<name>_test.pkl
    data/<name>_summary.json
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

from src.data.dataset_builder import build_dataset, split_dataset  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate V2 graph-query dataset")
    parser.add_argument("--config", default="configs/graph_flow.yaml")
    parser.add_argument("--name", default=None, help="dataset name (default: <graph_type>_<n>)")
    parser.add_argument("--data-dir", default=None)
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
    # --set 可以重复出现，每次一个值；这里统一摊平（nargs="*" 会丢掉前面的值）
    overrides = [item for group in args.overrides for item in group]
    config = load_config(args.config, args.overrides)
    set_seed(int(config.get("seed", 0)))

    data_cfg = config.section("data")
    num_nodes = data_cfg.get("num_nodes", [20, 40])
    num_nodes = (
        [int(num_nodes[0]), int(num_nodes[1])]
        if isinstance(num_nodes, (list, tuple))
        else int(num_nodes)
    )
    num_samples = int(data_cfg.get("num_samples", 256))

    start = time.time()
    print(f"generating {num_samples} samples (graph_type={data_cfg.get('graph_type')}) ...", flush=True)
    dataset = build_dataset(
        num_samples=num_samples,
        graph_type=str(data_cfg.get("graph_type", "er")),
        num_nodes=num_nodes,
        min_od_distance=int(data_cfg.get("min_od_distance", 5)),
        seed=int(config.get("seed", 0)),
        queries_per_graph=int(data_cfg.get("num_queries_per_graph", 4)),
        generator_cfg=dict(data_cfg.get("generator", {}) or {}),
        weighted=bool(data_cfg.get("weighted", False)),
        component_fallback=bool(data_cfg.get("component_fallback", True)),
    )
    print(f"generated in {time.time() - start:.1f}s", flush=True)

    split_cfg = config.get("split", {})
    splits = split_dataset(
        dataset,
        {
            "train": float(split_cfg.get("train", 0.8)),
            "val": float(split_cfg.get("val", 0.1)),
            "test": float(split_cfg.get("test", 0.1)),
        },
        seed=int(config.get("seed", 0)),
    )

    data_dir = Path(args.data_dir or str(config.get("paths.data_dir", "data")))
    name = args.name or f"{data_cfg.get('graph_type', 'er')}_{num_samples}"
    summary = {}
    for split_name, split in splits.items():
        path = split.save(data_dir / f"{name}_{split_name}.pkl")
        summary[split_name] = split.summary()
        print(f"{split_name:>5}: {len(split):>4} samples -> {path}", flush=True)

    summary_path = data_dir / f"{name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)
    print(f"summary -> {summary_path}")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
