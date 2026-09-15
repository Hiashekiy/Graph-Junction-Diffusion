"""检查数据语义（实施指南第 26 节 Milestone A 的验收工具）。

用法::

    python scripts/inspect_dataset.py --data data/unweighted/unweighted_train.pkl --limit 3

对每个样本打印：
    - 节点类型分布 / decision set
    - 每一条 branch 的 nodes 与 physical edges
    - z_0（每个 decision 选了哪条 branch，NULL 会显式标出）
    - 一批样本 collate 之后的 batch 形状
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.branch_segments import NODE_TYPE_NAMES  # noqa: E402
from src.data.collate import collate_samples  # noqa: E402
from src.data.dataset import GraphQueryDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inspect V2 dataset semantics")
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2, help="collate 几个样本做 batch 检查")
    return parser.parse_args()


def describe_sample(sample, index: int) -> None:
    segments = sample.segments
    field = sample.field
    print("=" * 78)
    print(
        f"sample {index}: N={sample.num_nodes} E_phys={segments.num_physical_edges} "
        f"E_msg={len(segments.edge_index)} s={segments.start} g={segments.goal} "
        f"M={sample.num_decisions} C={sample.num_candidates}"
    )
    print(f"  GT path ({sample.gt_length} hops): {sample.gt_path}")
    # node_type 是 {node: type} 的字典，不是按节点顺序排列的 list
    counts: dict[str, int] = {}
    for node in range(segments.num_nodes):
        name = NODE_TYPE_NAMES[segments.node_type[node]]
        counts[name] = counts.get(name, 0) + 1
    print(f"  node types: {counts}")
    print(f"  decisions : {segments.decision_nodes}")
    if segments.source_forced_nodes:
        print(
            f"  source forced segment (P0-1, 永久 selected): "
            f"nodes={segments.source_forced_nodes} "
            f"phys_edges={segments.source_forced_edge_ids}"
        )
    else:
        print("  source forced segment : none (deg(s) > 1, source 是 decision node)")

    candidates = field.candidates
    for decision_index, node in enumerate(segments.decision_nodes):
        target = candidates.target_candidate[decision_index]
        chosen = candidates.candidate_branch[target]
        label = "NULL" if candidates.candidate_is_null[target] else f"branch -> {chosen.end}"
        print(f"  decision {decision_index} (node {node}): z0 = {label}")
        for local, branch in enumerate(segments.branches[decision_index]):
            marker = "  *" if chosen is not None and branch is chosen else "   "
            print(
                f"    {marker} cand={branch.owner}->{branch.end} "
                f"nodes={branch.nodes} phys_edges={branch.physical_edges}"
            )


def main() -> int:
    args = parse_args()
    dataset = GraphQueryDataset.load(args.data)
    print(f"loaded {len(dataset)} samples from {args.data}")
    print(f"summary: {dataset.summary()}")

    for index in range(min(args.limit, len(dataset))):
        describe_sample(dataset[index], index)

    batch_samples = [dataset[i] for i in range(min(args.batch, len(dataset)))]
    batch = collate_samples(batch_samples, device="cpu")
    print("=" * 78)
    print("collated batch:")
    for key, value in batch.describe().items():
        print(f"  {key}: {value}")
    print(f"  node_type           {tuple(batch.node_type.shape)}")
    print(f"  edge_index          {tuple(batch.edge_index.shape)}")
    print(f"  branch_node_ids     {tuple(batch.branch_node_ids.shape)}")
    print(f"  branch_edge_ids     {tuple(batch.branch_edge_ids.shape)}")
    print(f"  candidate_is_null   {int(batch.candidate_is_null.sum())} / {batch.num_candidates}")
    print(f"  source_forced_edges {batch.source_forced_edge_ids.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
