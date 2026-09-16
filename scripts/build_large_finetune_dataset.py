"""构造"大图最终微调"数据集（LARGE_GRAPH_FINETUNE_PLAN 第 4~11 节）。

这个脚本**不重新跑**昂贵的数据准备链（CSV -> road path -> corridor -> GraphSample），
而是把两份**已经准备好**的数据集重新筛选 / 去重 / 混合：

    data/didi/graph/chengdu/{train,val,test}.pkl        （普通图，72/8/20）
    data/didi/graph/chengdu_long/{train,val,test}.pkl   （大图，55/10/35）
        ↓  防泄漏 + 跨源去重 + 分层抽样
    data/didi/graph/chengdu_large_finetune/
        ├── train.pkl        long : normal = 2 : 1
        ├── val.pkl          long : normal = 2 : 1
        ├── stats.json       计数与分布，可直接进 run 记录
        └── manifest.json    每条样本的来源与规模（排查分布用）

为什么不能直接把两个 train.pkl concat（方案第 5 节）
----------------------------------------------------
两份数据是**两次独立划分**，且长度窗口有重叠：

    normal: 10 < road_len < 100
    long:   59 < road_len < 200

所以同一条真实轨迹完全可能**在 normal/test 里、同时在 long/train 里**。
直接 concat 等于把测试轨迹重新塞回训练集。

防泄漏规则（方案第 6、7 节）
----------------------------
唯一轨迹键与项目现有的去重定义保持一致::

    key = (sample.start, sample.goal, tuple(sample.gt_path))

**不是** (start, goal)：相同 OD + 不同 route 代表真实的多模态路线选择，必须保留。

    holdout = keys(normal_val + normal_test + long_val + long_test)
    new_train = {k in (normal_train + long_train) : k not in holdout}

然后再对 normal_train / long_train 之间按同一个 key 做跨源去重
（重复时**丢 normal 保 long** —— 本次训练的目标就是大图，long 是稀缺侧）。

额外加了第二道闸门：``meta['order_id']``（原始 CSV 里的订单号，即"一条真实车辆
轨迹"）。按 key 去重是**更细**的口径（两个不同订单走出同一条路径时 key 相同），
按 order_id 是**更粗但更直接**的口径。两者都跑、分别记账，任何一侧命中都丢弃，
这样"测试轨迹的回声"不可能以任何形式进入训练集。

最终必须成立（脚本内置 assert，不成立直接报错，不会静默产出坏数据）::

    new_train ∩ normal_val  = ∅
    new_train ∩ normal_test = ∅
    new_train ∩ long_val    = ∅
    new_train ∩ long_test   = ∅
    new_train ∩ new_val     = ∅

normal replay 怎么抽（方案第 9 节）
-----------------------------------
不是简单 ``random.sample``：按 ``num_nodes`` 的 1/3、2/3 分位数分成
small / medium / large-normal 三桶，**每桶按原比例抽样**（最大余数法凑满总数），
这样小图 / 中图 / "普通数据里的大图"三种尺度都被稳定保留。

用法::

    # 正式构造
    python scripts/build_large_finetune_dataset.py

    # 只看统计（不落盘），用于确认池子与重叠量
    python scripts/build_large_finetune_dataset.py --dry-run

    # 冒烟用的小数据集（换个目录，别污染正式数据）
    python scripts/build_large_finetune_dataset.py \
        --out-dir data/didi/graph/chengdu_large_finetune_smoke \
        --long-train-cap 12 --long-val-cap 6

产出后按方案第 20 节训练::

    python scripts/train.py \
      --config configs/didi_chengdu_large_finetune.yaml \
      --name didi_chengdu_large_finetune \
      --data data/didi/graph/chengdu_large_finetune/train.pkl \
      --val-data data/didi/graph/chengdu_large_finetune/val.pkl \
      --init-from outputs/runs/didi_chengdu_loss_improved/best.pt
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GraphQueryDataset, GraphSample  # noqa: E402

# 与项目现有去重定义一致：GT 路径参与 key，所以"同 OD 不同 route"不会被吃掉。
TrajectoryKey = Tuple[int, int, Tuple[int, ...]]

#: normal replay 的 num_nodes 分层桶名（方案第 9 节）
NODE_BUCKETS = ("small", "medium", "large_normal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="build the large-graph fine-tuning dataset (anti-leakage + 2:1 mix)"
    )
    parser.add_argument(
        "--normal-dir", default="data/didi/graph/chengdu", help="普通成都数据集目录"
    )
    parser.add_argument(
        "--long-dir", default="data/didi/graph/chengdu_long", help="大图数据集目录"
    )
    parser.add_argument(
        "--out-dir", default="data/didi/graph/chengdu_large_finetune", help="输出目录"
    )
    # 方案第 8 / 10 节：train 与 val 都用 long : normal = 2 : 1
    parser.add_argument(
        "--ratio", type=float, default=2.0, help="long : normal，train 与 val 共用"
    )
    parser.add_argument("--seed", type=int, default=0)
    # 0 = 不设上限，即"池子里有多少用多少"（正式口径）
    parser.add_argument(
        "--long-train-cap", type=int, default=0, help="0 = 用尽 long train 池（正式口径）"
    )
    parser.add_argument(
        "--normal-train-cap",
        type=int,
        default=0,
        help="0 = 自动 = long_train_selected / ratio",
    )
    parser.add_argument("--long-val-cap", type=int, default=0, help="0 = 用尽 long val 池")
    parser.add_argument(
        "--normal-val-cap", type=int, default=0, help="0 = 自动 = long_val_selected / ratio"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只统计重叠与分布，不写任何文件"
    )
    parser.add_argument("--force", action="store_true", help="覆盖已存在的输出")
    parser.add_argument(
        "--no-copy-graph",
        action="store_true",
        help="不把 graph_global.pkl 复制到输出目录（Trainer 用它补全坐标做 DTW）",
    )
    parser.add_argument(
        "--skip-order-id-check",
        action="store_true",
        help="关闭 order_id 这道额外防泄漏闸门（只按 (start, goal, gt_path) 判重叠）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 读取与键
# ---------------------------------------------------------------------------
def trajectory_key(sample: GraphSample) -> TrajectoryKey:
    """方案第 6 节的唯一轨迹键。"""
    return (int(sample.start), int(sample.goal), tuple(int(node) for node in sample.gt_path))


def key_digest(key: TrajectoryKey) -> str:
    raw = f"{key[0]}:{key[1]}:" + ",".join(str(node) for node in key[2])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def order_id_of(sample: GraphSample) -> str:
    return str(sample.meta.get("order_id", "") or "")


def load_split(directory: Path, split: str, required: bool = True) -> List[GraphSample]:
    path = directory / f"{split}.pkl"
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/prepare_didi.py --config <config> "
                "--build` for that dataset first."
            )
        print(f"[build] {path} missing -> treated as empty", flush=True)
        return []
    start = time.time()
    dataset = GraphQueryDataset.load(path)
    print(
        f"[build] loaded {path} : {len(dataset)} samples in {time.time() - start:.1f}s",
        flush=True,
    )
    return list(dataset.samples)


def keys_of(samples: Iterable[GraphSample]) -> set:
    return {trajectory_key(sample) for sample in samples}


def order_ids_of(samples: Iterable[GraphSample]) -> set:
    return {value for value in (order_id_of(s) for s in samples) if value}


# ---------------------------------------------------------------------------
# 分层抽样（方案第 9 节）
# ---------------------------------------------------------------------------
def node_bucket_edges(pool: Sequence[GraphSample]) -> np.ndarray:
    """num_nodes 的 1/3、2/3 分位数作为桶边界。"""
    nodes = np.asarray([sample.num_nodes for sample in pool], dtype=float)
    return np.quantile(nodes, [1.0 / 3.0, 2.0 / 3.0])


def assign_buckets(
    pool: Sequence[GraphSample], edges: np.ndarray
) -> Dict[int, List[int]]:
    """返回 {bucket_index: [pool 下标]}，bucket 0/1/2 = small/medium/large_normal。"""
    nodes = np.asarray([sample.num_nodes for sample in pool], dtype=float)
    # searchsorted(side='right')：x < e0 -> 0；e0 <= x < e1 -> 1；x >= e1 -> 2
    labels = np.searchsorted(edges, nodes, side="right")
    buckets: Dict[int, List[int]] = {index: [] for index in range(len(NODE_BUCKETS))}
    for position, label in enumerate(labels.tolist()):
        buckets[int(label)].append(position)
    return buckets


def allocate_quota(bucket_sizes: Sequence[int], total: int) -> List[int]:
    """最大余数法：按各桶原比例分配，且各桶之和**恰好**等于 total。"""
    sizes = np.asarray(bucket_sizes, dtype=float)
    pool_size = int(sizes.sum())
    if pool_size == 0 or total <= 0:
        return [0 for _ in sizes]
    total = min(int(total), pool_size)
    raw = [total * float(size) / pool_size for size in sizes]
    quota = [int(value) for value in raw]
    remainder = total - sum(quota)
    # 小数部分大的先补；并列时按下标，保证确定性
    order = sorted(range(len(sizes)), key=lambda index: (-(raw[index] - quota[index]), index))
    for index in order[:remainder]:
        quota[index] += 1
    return quota


def stratified_select(
    pool: Sequence[GraphSample],
    count: int,
    seed: int,
    label: str,
) -> Tuple[List[GraphSample], Dict[str, Any]]:
    """按 num_nodes 三分位分层抽 ``count`` 条；返回 (选中样本, 记账信息)。

    ``pool`` 的顺序由调用方决定且必须确定性，否则同一 seed 抽不出同一批。
    """
    if count <= 0 or not pool:
        return [], {"requested": int(count), "selected": 0, "buckets": {}}
    count = min(int(count), len(pool))
    edges = node_bucket_edges(pool)
    buckets = assign_buckets(pool, edges)
    sizes = [len(buckets[index]) for index in range(len(NODE_BUCKETS))]
    quota = allocate_quota(sizes, count)

    chosen: List[int] = []
    records: Dict[str, Any] = {}
    for index, bucket_name in enumerate(NODE_BUCKETS):
        members = buckets[index]
        take = min(quota[index], len(members))
        # 每个桶一条独立随机流：加桶 / 改桶顺序不会扰动其它桶的抽样结果
        rng = random.Random(seed * 1000003 + index)
        picked = rng.sample(members, take) if take else []
        chosen.extend(picked)
        picked_samples = [pool[position] for position in picked]
        records[bucket_name] = {
            "num_nodes_lo": float(edges[0]) if index == 0 else float(edges[index - 1]),
            "num_nodes_hi": float(edges[index]) if index < len(edges) else None,
            "pool": len(members),
            "quota": int(quota[index]),
            "selected": int(take),
            "gt_length_mean": (
                float(np.mean([sample.gt_length for sample in picked_samples]))
                if picked_samples
                else None
            ),
            "num_nodes_mean": (
                float(np.mean([sample.num_nodes for sample in picked_samples]))
                if picked_samples
                else None
            ),
        }
    chosen.sort()  # 恢复池内顺序，让最终 shuffle 完全由 seed 决定
    selected = [pool[position] for position in chosen]
    if len(selected) < count:
        # 桶内取不满（理论上不会发生）时，从剩余样本里补足，避免静默少给
        taken = set(chosen)
        spare = [position for position in range(len(pool)) if position not in taken]
        rng = random.Random(seed * 1000003 + 97)
        extra = rng.sample(spare, min(count - len(selected), len(spare)))
        selected.extend(pool[position] for position in extra)
        records["topped_up_from_spare"] = len(extra)
    records["edges"] = [float(value) for value in edges]
    records["pool"] = len(pool)
    records["requested"] = int(count)
    records["selected"] = len(selected)
    records["label"] = label
    return selected, records


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def describe(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def split_summary(samples: Sequence[GraphSample]) -> Dict[str, Any]:
    return {
        "num_samples": len(samples),
        "num_nodes": describe([sample.num_nodes for sample in samples]),
        "num_decisions": describe([sample.num_decisions for sample in samples]),
        "num_candidates": describe([sample.num_candidates for sample in samples]),
        "gt_length": describe([sample.gt_length for sample in samples]),
    }


def sample_record(sample: GraphSample, source: str, split: str, index: int) -> Dict[str, Any]:
    key = trajectory_key(sample)
    record: Dict[str, Any] = {
        "index": int(index),
        "source": source,
        "split": split,
        "start": key[0],
        "goal": key[1],
        "gt_length": int(sample.gt_length),
        "num_nodes": int(sample.num_nodes),
        "num_decisions": int(sample.num_decisions),
        "num_candidates": int(sample.num_candidates),
        "key_sha1": key_digest(key),
        "order_id": order_id_of(sample),
        "date": str(sample.meta.get("date", "")),
    }
    ratio = sample.meta.get("gt_cost_ratio")
    if isinstance(ratio, (int, float)):
        record["gt_cost_ratio"] = float(ratio)
    return record


# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    if args.ratio <= 0:
        raise SystemExit(f"--ratio must be positive, got {args.ratio}")

    normal_dir = _resolve(args.normal_dir)
    long_dir = _resolve(args.long_dir)
    out_dir = _resolve(args.out_dir)
    print(f"[build] normal dir : {normal_dir}")
    print(f"[build] long   dir : {long_dir}")
    print(f"[build] output dir : {out_dir}")
    print(f"[build] ratio long:normal = {args.ratio:g} : 1   seed={args.seed}")

    if not args.dry_run and not args.force:
        existing = [name for name in ("train.pkl", "val.pkl") if (out_dir / name).exists()]
        if existing:
            raise SystemExit(
                f"{out_dir} already contains {existing}. Pass --force to overwrite "
                "(refusing to silently replace a dataset that a training run may reference)."
            )

    # ---- 1) val/test：只在键层面需要，拿到键就把样本对象放掉 ------------------
    normal_val = load_split(normal_dir, "val")
    long_val = load_split(long_dir, "val")
    holdout_keys = keys_of(normal_val) | keys_of(long_val)
    holdout_orders = order_ids_of(normal_val) | order_ids_of(long_val)
    for directory, split in ((normal_dir, "test"), (long_dir, "test")):
        samples = load_split(directory, split)
        holdout_keys |= keys_of(samples)
        holdout_orders |= order_ids_of(samples)
        del samples
        gc.collect()
    print(
        f"[build] holdout: {len(holdout_keys)} unique path keys, "
        f"{len(holdout_orders)} unique order_ids (normal+long, val+test)",
        flush=True,
    )

    # ---- 2) train 池：防泄漏 + 跨源去重 --------------------------------------
    normal_train_raw = load_split(normal_dir, "train")
    long_train_raw = load_split(long_dir, "train")
    normal_train_pool_raw = len(normal_train_raw)
    long_train_pool_raw = len(long_train_raw)

    dropped_holdout_key = 0
    dropped_holdout_order = 0
    normal_train: List[GraphSample] = []
    long_train: List[GraphSample] = []
    seen_normal: set = set()
    seen_long: set = set()
    for sample in normal_train_raw:
        if trajectory_key(sample) in holdout_keys:
            dropped_holdout_key += 1
            continue
        if not args.skip_order_id_check and order_id_of(sample) in holdout_orders:
            dropped_holdout_order += 1
            continue
        normal_train.append(sample)
        seen_normal.add(trajectory_key(sample))
    for sample in long_train_raw:
        if trajectory_key(sample) in holdout_keys:
            dropped_holdout_key += 1
            continue
        if not args.skip_order_id_check and order_id_of(sample) in holdout_orders:
            dropped_holdout_order += 1
            continue
        long_train.append(sample)
        seen_long.add(trajectory_key(sample))

    # 跨源去重：同一 key 在两边都有时**丢 normal 保 long**（long 是稀缺侧，
    # 且本次训练的目标就是大图）。order_id 同理 —— 同一订单号不可能既是普通样本
    # 又是大图样本；key 相同是它的充分不必要条件，两道都查。
    duplicate_keys = seen_normal & seen_long
    long_orders = order_ids_of(long_train)
    duplicate_orders = {
        value for value in order_ids_of(normal_train) if value and value in long_orders
    }
    if duplicate_keys or duplicate_orders:
        normal_train = [
            sample
            for sample in normal_train
            if trajectory_key(sample) not in duplicate_keys
            and order_id_of(sample) not in duplicate_orders
        ]
    normal_train_pool = len(normal_train)
    long_train_pool = len(long_train)
    print(
        f"[build] train pool after holdout filter: normal={normal_train_pool} "
        f"long={long_train_pool}; cross-source duplicates dropped: "
        f"{len(duplicate_keys)} path-key + {len(duplicate_orders)} order_id",
        flush=True,
    )

    del normal_train_raw, long_train_raw
    gc.collect()

    # ---- 3) 选 sample -------------------------------------------------------
    if args.long_train_cap > 0:
        long_train_selected, _ = _shuffle_take(long_train, args.long_train_cap, args.seed + 11)
    else:
        long_train_selected = list(long_train)
    n_long_train = len(long_train_selected)
    normal_quota = (
        args.normal_train_cap
        if args.normal_train_cap > 0
        else int(round(n_long_train / args.ratio))
    )
    normal_train_selected, normal_buckets = stratified_select(
        normal_train, normal_quota, args.seed + 1, "normal_train"
    )
    del normal_train
    gc.collect()

    long_train_new = _shuffle_take_ordered(long_train_selected, args.seed + 2)
    normal_train_new = _shuffle_take_ordered(normal_train_selected, args.seed + 3)

    train_samples = long_train_new + normal_train_new
    train_sources = ["long_train"] * len(long_train_new) + ["normal_train"] * len(
        normal_train_new
    )
    order = list(range(len(train_samples)))
    random.Random(args.seed + 4).shuffle(order)
    train_samples = [train_samples[index] for index in order]
    train_sources = [train_sources[index] for index in order]

    # ---- 4) val：long val : normal val = ratio : 1，且只来自 val split ---------
    val_duplicate_keys = keys_of(long_val) & keys_of(normal_val)
    if val_duplicate_keys:
        normal_val = [
            sample for sample in normal_val if trajectory_key(sample) not in val_duplicate_keys
        ]
    if args.long_val_cap > 0:
        long_val_selected = _shuffle_take(long_val, args.long_val_cap, args.seed + 21)
    else:
        long_val_selected = list(long_val)
    n_long_val = len(long_val_selected)
    normal_val_quota = (
        args.normal_val_cap
        if args.normal_val_cap > 0
        else int(round(n_long_val / args.ratio))
    )
    normal_val_selected, normal_val_buckets = stratified_select(
        normal_val, normal_val_quota, args.seed + 5, "normal_val"
    )

    val_long_ordered = _shuffle_take_ordered(long_val_selected, args.seed + 22)
    val_normal_ordered = _shuffle_take_ordered(normal_val_selected, args.seed + 23)
    val_samples = val_long_ordered + val_normal_ordered
    val_sources = ["long_val"] * len(val_long_ordered) + ["normal_val"] * len(
        val_normal_ordered
    )

    # ---- 5) 硬校验：泄漏 / 重复必须为 0 --------------------------------------
    train_keys = keys_of(train_samples)
    val_keys = keys_of(val_samples)
    _assert_no_overlap(train_keys, holdout_keys, "new_train", "holdout (normal+long val/test)")
    _assert_no_overlap(train_keys, val_keys, "new_train", "new_val")
    duplicates_in_train = _duplicates(train_samples)
    if duplicates_in_train:
        raise SystemExit(
            f"new_train contains {len(duplicates_in_train)} duplicated path keys; "
            "the dedup step is broken, refusing to write a dataset with double-weighted routes."
        )
    if not args.skip_order_id_check:
        repeated_orders = _repeated_order_ids(train_samples)
        if repeated_orders:
            raise SystemExit(
                f"new_train reuses {len(repeated_orders)} order_id(s) across samples; "
                "refusing to write. Inspect with --dry-run."
            )

    train_long_fraction = len(long_train_new) / len(train_samples) if train_samples else 0.0
    val_long_fraction = len(long_val_selected) / len(val_samples) if val_samples else 0.0
    target_fraction = args.ratio / (1.0 + args.ratio)
    for label, value in (
        ("train_long_fraction", train_long_fraction),
        ("val_long_fraction", val_long_fraction),
    ):
        if abs(value - target_fraction) > 0.02:
            print(
                f"[build] WARNING: {label}={value:.4f} deviates from the requested "
                f"{args.ratio:g}:1 ratio (target {target_fraction:.4f}); the source pool "
                "was too small to hit the ratio exactly.",
                flush=True,
            )

    # ---- 6) stats / manifest ------------------------------------------------
    stats: Dict[str, Any] = {
        # 方案第 11 节要求的字段
        "normal_train_pool": int(normal_train_pool),
        "long_train_pool": int(long_train_pool),
        "normal_selected": int(len(normal_train_new)),
        "long_selected": int(len(long_train_new)),
        "dropped_due_to_holdout_overlap": int(dropped_holdout_key),
        "dropped_due_to_cross_source_duplicate": int(len(duplicate_keys)),
        "train_total": int(len(train_samples)),
        "val_total": int(len(val_samples)),
        "train_long_fraction": float(train_long_fraction),
        "val_long_fraction": float(val_long_fraction),
        # 扩展：让这份 dataset 自证来源与口径
        "ratio_long_to_normal": float(args.ratio),
        "seed": int(args.seed),
        "normal_dir": str(normal_dir),
        "long_dir": str(long_dir),
        "out_dir": str(out_dir),
        "normal_train_pool_raw": int(normal_train_pool_raw),
        "long_train_pool_raw": int(long_train_pool_raw),
        "dropped_due_to_order_id_overlap": int(dropped_holdout_order),
        "dropped_due_to_cross_source_duplicate_order_id": int(len(duplicate_orders)),
        "dropped_due_to_cross_source_duplicate_val": int(len(val_duplicate_keys)),
        "holdout_unique_keys": int(len(holdout_keys)),
        "holdout_unique_order_ids": int(len(holdout_orders)),
        "long_val_pool": int(len(long_val_selected)),
        "normal_val_pool": int(len(normal_val_selected)),
        "normal_train_buckets": normal_buckets,
        "normal_val_buckets": normal_val_buckets,
        "train_source_counts": dict(Counter(train_sources)),
        "val_source_counts": dict(Counter(val_sources)),
        "train_long": split_summary(long_train_new),
        "train_normal": split_summary(normal_train_new),
        "val_long": split_summary(val_long_ordered),
        "val_normal": split_summary(val_normal_ordered),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    manifest = {
        "description": (
            "large-graph fine-tuning mix: long:normal = "
            f"{args.ratio:g}:1, anti-leakage against normal/long val+test by "
            "(start, goal, gt_path) and by meta['order_id']"
        ),
        "seed": int(args.seed),
        "ratio_long_to_normal": float(args.ratio),
        "train": [
            sample_record(sample, source, "train", index)
            for index, (sample, source) in enumerate(zip(train_samples, train_sources))
        ],
        "val": [
            sample_record(sample, source, "val", index)
            for index, (sample, source) in enumerate(zip(val_samples, val_sources))
        ],
    }

    print()
    print("=" * 68)
    print("large fine-tune dataset summary")
    print("=" * 68)
    print(f"  normal train pool (post-filter) : {normal_train_pool}")
    print(f"  long   train pool (post-filter) : {long_train_pool}")
    print(f"  dropped: holdout path-key ovlp  : {dropped_holdout_key}")
    print(f"  dropped: holdout order_id ovlp  : {dropped_holdout_order}")
    print(f"  dropped: cross-source dup (key) : {len(duplicate_keys)}")
    print(f"  dropped: cross-source dup (oid) : {len(duplicate_orders)}")
    print(f"  train : {len(train_samples)}  (long {len(long_train_new)} / normal {len(normal_train_new)})"
          f"  long fraction {train_long_fraction:.4f}")
    print(f"  val   : {len(val_samples)}  (long {len(long_val_selected)} / normal {len(normal_val_selected)})"
          f"  long fraction {val_long_fraction:.4f}")
    if train_samples:
        print(
            "  train gt_length mean/p50 : "
            f"{stats['train_long']['gt_length']['mean']:.1f} / "
            f"{stats['train_long']['gt_length']['p50']:.0f} (long)  |  "
            f"{stats['train_normal']['gt_length']['mean']:.1f} / "
            f"{stats['train_normal']['gt_length']['p50']:.0f} (normal)"
        )
        print(
            "  train num_nodes mean/p50 : "
            f"{stats['train_long']['num_nodes']['mean']:.0f} / "
            f"{stats['train_long']['num_nodes']['p50']:.0f} (long)  |  "
            f"{stats['train_normal']['num_nodes']['mean']:.0f} / "
            f"{stats['train_normal']['num_nodes']['p50']:.0f} (normal)"
        )
    print("  leak checks              : new_train ∩ holdout = 0, new_train ∩ new_val = 0  OK")
    print("=" * 68)

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return 0

    # ---- 7) 落盘 ------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    GraphQueryDataset(train_samples, name="chengdu_large_finetune_train").save(
        out_dir / "train.pkl"
    )
    GraphQueryDataset(val_samples, name="chengdu_large_finetune_val").save(out_dir / "val.pkl")
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    # Trainer 会用 paths.data_dir/graph_global.pkl 补全坐标（DTW 指标）。不复制也不会
    # 报错（DTW 降级为 NaN），但复制一份能让这个目录自包含、且大图 val 的 DTW 可用。
    graph_file = long_dir / "graph_global.pkl"
    if not args.no_copy_graph and graph_file.exists():
        shutil.copy2(graph_file, out_dir / "graph_global.pkl")
        print(f"[build] copied {graph_file.name} (for DTW coordinates)")

    print(f"[build] wrote {out_dir / 'train.pkl'}")
    print(f"[build] wrote {out_dir / 'val.pkl'}")
    print(f"[build] wrote {out_dir / 'stats.json'}")
    print(f"[build] wrote {out_dir / 'manifest.json'}")
    print()
    print("next:")
    print("  python scripts/train.py \\")
    print("    --config configs/didi_chengdu_large_finetune.yaml \\")
    print("    --name didi_chengdu_large_finetune \\")
    print(f"    --data {_posix(out_dir)}/train.pkl \\")
    print(f"    --val-data {_posix(out_dir)}/val.pkl \\")
    print("    --init-from outputs/runs/didi_chengdu_loss_improved/best.pt")
    return 0


# ---------------------------------------------------------------------------
def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _posix(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def _shuffle_take(pool: Sequence[GraphSample], count: int, seed: int) -> List[GraphSample]:
    """确定性随机取 ``count`` 条（用于给 long 侧限流，正式口径 count=0 不走这里）。"""
    count = min(int(count), len(pool))
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    return [pool[index] for index in order[:count]]


def _shuffle_take_ordered(pool: Sequence[GraphSample], seed: int) -> List[GraphSample]:
    """把池子按 seed 打乱后整体返回（不丢样本）。"""
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    return [pool[index] for index in order]


def _assert_no_overlap(left: set, right: set, left_name: str, right_name: str) -> None:
    overlap = left & right
    if overlap:
        example = next(iter(overlap))
        raise SystemExit(
            f"LEAKAGE: {left_name} ∩ {right_name} = {len(overlap)} path keys "
            f"(e.g. start={example[0]}, goal={example[1]}, gt_len={len(example[2])}). "
            "Refusing to write the dataset: a test/val trajectory would be trained on."
        )


def _duplicates(samples: Sequence[GraphSample]) -> set:
    seen: set = set()
    duplicates: set = set()
    for sample in samples:
        key = trajectory_key(sample)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def _repeated_order_ids(samples: Sequence[GraphSample]) -> set:
    counter = Counter(order_id_of(sample) for sample in samples)
    return {order_id for order_id, count in counter.items() if count > 1 and order_id}


if __name__ == "__main__":
    raise SystemExit(main())
