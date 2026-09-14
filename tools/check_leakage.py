"""数据集之间的图级泄漏检查（合并/补充数据后必跑）。

为什么需要：同一张底层图如果同时出现在 train 和 val/test（哪怕 OD 对不同），模型就
等于见过测试图的结构，指标会虚高。项目的生成器是"一张图 + 多个 OD query"，所以
**必须按图**（而不是按 query）核对。

指纹 = ``hashlib.sha1`` over ``(num_nodes, sorted(edges))``，与 query 的 OD 无关：
同一张图的任意两个 query 指纹相同，不同图（即使节点数一样）几乎必然不同。

用法::

    python tools/check_leakage.py \
        --pair data/mixed/mixed_oldv1_train.pkl data/mixed/mixed_oldv1_val.pkl \
        --pair data/mixed/mixed_oldv1_train.pkl data/oldv1/oldv1_test.pkl \
        --pair data/mixed/mixed_oldv1_train.pkl data/controlled/controlled_test.pkl

    # 或者一次给多份，检查所有两两组合
    python tools/check_leakage.py --all data/mixed/mixed_oldv1_train.pkl data/mixed/mixed_oldv1_val.pkl data/oldv1/oldv1_test.pkl

退出码：有任何一对存在重叠 -> 1（配合 --strict 用于流水线；默认也是 1，便于发现）。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GraphQueryDataset  # noqa: E402


def graph_fingerprints(path: str | Path) -> Dict[str, int]:
    """返回 指纹 -> 该图在文件里出现的 query 次数。"""
    dataset = GraphQueryDataset.load(path)
    per_graph: Dict[str, set] = {}
    for sample in dataset:
        graph = sample.graph
        edges = tuple(sorted(tuple(sorted(edge)) for edge in graph.edges()))
        payload = f"{graph.number_of_nodes()}|{edges}".encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()
        per_graph.setdefault(digest, set()).add(sample.meta.get("graph_id"))
    return {digest: len(ids) for digest, ids in per_graph.items()}


def compare(path_a: str, path_b: str) -> Tuple[int, int]:
    a = graph_fingerprints(path_a)
    b = graph_fingerprints(path_b)
    shared = set(a) & set(b)
    return len(shared), sum(min(a[k], b[k]) for k in shared)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", nargs=2, action="append", default=[],
                        help="要比较的两份数据集，可重复")
    parser.add_argument("--all", nargs="+", default=None,
                        help="一次给多份，检查所有两两组合")
    args = parser.parse_args()

    pairs: List[Sequence[str]] = [tuple(item) for item in args.pair]
    if args.all:
        pairs.extend(itertools.combinations(args.all, 2))
    if not pairs:
        parser.error("至少要给 --pair A B 或 --all A B C")

    bad = 0
    for path_a, path_b in pairs:
        shared_graphs, shared_queries = compare(path_a, path_b)
        flag = "OK  " if shared_graphs == 0 else "泄漏!"
        if shared_graphs:
            bad += 1
        print(f"[{flag}] {Path(path_a).name} <-> {Path(path_b).name}: "
              f"重叠图 {shared_graphs}（涉及 query ≈ {shared_queries}）")
    print("结论：" + ("无可检测的图级重叠" if bad == 0 else f"{bad} 对存在重叠，不能这样训/测"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
