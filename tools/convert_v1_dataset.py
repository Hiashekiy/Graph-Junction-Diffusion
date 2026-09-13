"""把旧版 V1 预处理数据（`data/processed/v1/*.pt`）转成 V2 的 GraphSample 数据集。

V1 与 V2 的差别（详见 README 第 17 节与本次核对）：

    V1  graph record : edge_index(2,2E) 有向 + degree + lap_pe/rwse + directed_edge_lookup
        query record : start/goal/gt_path + decision_nodes + **一跳邻居**候选 + z0
    V2  GraphSample  : networkx 无向图 + GraphSegments（branch **segment** 到下一个
                       structural endpoint）+ DecisionField（z_0）

所以转换的做法是：用 V1 的 `edge_index` 重建无向图，用 `start/goal` 重新计算 V2 的
segments / decision field。GT 路径默认**沿用 V1 的 `gt_path`**（已验证它一定是最短路，
只是多解时可能和 `nx.shortest_path` 选的那条不同；用 V1 自己的路径才能保证
z_0 / active decision 与 V1 完全一致）。V1 的 `decision_nodes/z0` 只用于一致性核对。

用法::

    python tools/convert_v1_dataset.py \
        --input data/processed/v1/test.pt \
        --out data/oldv1_test.pkl [--limit 200] [--shortest-path] [--no-verify]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import branch_segments as bs  # noqa: E402
from src.data.dataset import GraphQueryDataset, GraphSample  # noqa: E402
from src.data.dataset_builder import build_sample, relabel_to_contiguous  # noqa: E402
from src.data.decision_field import (  # noqa: E402
    build_decision_field,
    validate_decision_field,
)


def graph_from_edge_index(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    """V1 的 edge_index 是**有向**的（每条无向边存两个方向），这里去重成无向图。"""
    graph = nx.Graph()
    graph.add_nodes_from(range(int(num_nodes)))
    edges = set()
    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if src == dst:
            continue
        edges.add((src, dst) if src < dst else (dst, src))
    graph.add_edges_from(edges)
    return graph


def build_sample_with_path(
    graph: nx.Graph,
    start: int,
    goal: int,
    gt_path: List[int],
    meta: Dict[str, Any] | None = None,
    graph_id: int | None = None,
) -> GraphSample:
    """与 `dataset_builder.build_sample` 相同，但 GT 路径由调用方给定（V1 的 gt_path）。"""
    graph, start, goal = relabel_to_contiguous(graph, start, goal)
    graph = bs.set_od(graph, start, goal)
    segments = bs.extract_segments(graph, start, goal, relabel=False)
    path = [int(v) for v in gt_path]
    field = build_decision_field(segments, path)
    validate_decision_field(segments, field, path)

    sample_meta = dict(meta or {})
    sample_meta.setdefault("graph_id", -1 if graph_id is None else int(graph_id))
    return GraphSample(
        graph=graph,
        start=int(start),
        goal=int(goal),
        gt_path=path,
        segments=segments,
        field=field,
        meta=sample_meta,
    )


def v1_active_decisions(query: Dict[str, Any]) -> set[int]:
    """V1 记录里"真正 active"的 decision node 集合（NULL 是组内第 0 个候选）。"""
    active = set()
    for index, node in enumerate(query["decision_nodes"]):
        has_null = bool(query["has_null"][index])
        if has_null and int(query["z0"][index]) == 0:
            continue
        active.add(int(node))
    return active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="V1 的 .pt 文件")
    parser.add_argument("--out", default=None, help="输出 GraphQueryDataset pkl")
    parser.add_argument("--limit", type=int, default=0, help="只转前 N 条 query（0 = 全部）")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="随机抽 N 条 query（0 = 全部；V1 的 query 按图类型分块排，取前缀会只拿到 er 图，"
        "所以抽样训练子集必须用这个而不是 --limit）",
    )
    parser.add_argument("--seed", type=int, default=0, help="--sample 用的随机种子")
    parser.add_argument(
        "--shortest-path",
        action="store_true",
        help="用 nx.shortest_path 重算 GT（默认沿用 V1 的 gt_path）",
    )
    parser.add_argument("--no-verify", action="store_true", help="跳过与 V1 字段的一致性核对")
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    graphs = {int(g["graph_id"]): g for g in payload["graphs"]}
    queries: List[Dict[str, Any]] = payload["queries"]
    if args.limit:
        queries = queries[: args.limit]
    if args.sample and args.sample < len(queries):
        queries = random.Random(args.seed).sample(queries, args.sample)

    samples = []
    failures: List[str] = []
    decision_counts: List[int] = []
    candidate_counts: List[int] = []
    by_type: Counter = Counter()
    node_mismatch = 0
    active_mismatch = 0
    compared = 0

    for query in queries:
        graph_record = graphs[int(query["graph_id"])]
        graph = graph_from_edge_index(graph_record["edge_index"], graph_record["num_nodes"])
        meta = {
            "graph_type": query["graph_type"],
            "graph_id": int(query["graph_id"]),
            "v1_query_id": int(query["query_id"]),
            "v1_gt_path": [int(v) for v in query["gt_path"]],
        }
        try:
            if args.shortest_path:
                sample = build_sample(
                    graph, int(query["start"]), int(query["goal"]), meta=meta
                )
            else:
                sample = build_sample_with_path(
                    graph,
                    int(query["start"]),
                    int(query["goal"]),
                    query["gt_path"],
                    meta=meta,
                )
        except Exception as error:  # noqa: BLE001 - 逐条记录，不让整批失败
            failures.append(f"query {query['query_id']}: {type(error).__name__}: {error}")
            continue

        samples.append(sample)
        decision_counts.append(sample.num_decisions)
        candidate_counts.append(sample.num_candidates)
        by_type[query["graph_type"]] += 1

        if not args.no_verify:
            compared += 1
            v2_decisions = set(int(v) for v in sample.segments.decision_nodes)
            if v2_decisions != set(int(v) for v in query["decision_nodes"]):
                node_mismatch += 1
            if v1_active_decisions(query) != set(
                int(v) for v in sample.gt_path if int(v) in v2_decisions
            ):
                active_mismatch += 1

    summary = {
        "input": args.input,
        "gt_path_source": "shortest_path" if args.shortest_path else "v1_gt_path",
        "sampled": args.sample or None,
        "sample_seed": args.seed if args.sample else None,
        "num_queries": len(queries),
        "num_converted": len(samples),
        "num_failed": len(failures),
        "by_graph_type": dict(by_type),
        "decisions_per_query": _stats(decision_counts),
        "candidates_per_query": _stats(candidate_counts),
        "verify": {
            "compared": compared,
            "decision_set_mismatch": node_mismatch,
            "active_set_mismatch": active_mismatch,
        },
        "first_failures": failures[:10],
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))

    if args.out:
        dataset = GraphQueryDataset(samples, name=Path(args.out).stem)
        dataset.save(args.out)
        print(f"saved {len(samples)} samples -> {args.out}")
    return 0 if samples else 1


def _stats(values: List[int]) -> Dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
