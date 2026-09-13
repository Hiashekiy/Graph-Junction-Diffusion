"""Dataset container (实施指南第 1、6 节).

一个 :class:`GraphSample` = 一张图上的一个 (s, g) query：

    graph       : NetworkX 图（只用于 evaluation / 调试，训练不读它）
    gt_path     : GT 最短路，节点序列 [s, ..., g]
    segments    : Branch Segment 分解结果
    field       : z_0（clean decision field）与扁平 candidate 表

训练与推理阶段的模型只消费这里预先算好的结构，不再做 NetworkX 遍历。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import networkx as nx
import numpy as np

from src.data import branch_segments as bs
from src.data.branch_segments import GraphSegments
from src.data.decision_field import DecisionField


@dataclass
class GraphSample:
    """一张图上的一个 OD query（纯 Python 结构，collate 时才转 tensor）。"""

    graph: nx.Graph
    start: int
    goal: int
    gt_path: List[int]
    segments: GraphSegments
    field: DecisionField
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return self.segments.num_nodes

    @property
    def num_decisions(self) -> int:
        return self.segments.num_decisions

    @property
    def num_candidates(self) -> int:
        return self.field.candidates.num_candidates

    @property
    def gt_length(self) -> int:
        """GT 路径跳数（边数）。"""
        return len(self.gt_path) - 1


def validate_sample(sample: GraphSample) -> None:
    """structural 自检：branch 覆盖的物理边必须与底层图完全一致。"""
    edge_lookup = bs.physical_edge_lookup(sample.graph)
    for branch in sample.segments.branches_flat():
        for index, (u, v) in enumerate(zip(branch.nodes[:-1], branch.nodes[1:])):
            assert edge_lookup.get((u, v)) == branch.physical_edges[index], (
                f"branch at {branch.owner}: edge ({u},{v}) does not match physical "
                f"edge {branch.physical_edges[index]}"
            )
    assert sample.gt_path[0] == sample.start and sample.gt_path[-1] == sample.goal


class GraphQueryDataset:
    """一组 :class:`GraphSample`。"""

    def __init__(self, samples: Sequence[GraphSample], name: str = "dataset"):
        self.samples: List[GraphSample] = list(samples)
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[GraphSample]:
        return iter(self.samples)

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"name": self.name, "samples": self.samples}, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GraphQueryDataset":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(payload["samples"], name=payload.get("name", "dataset"))

    # -- stats ------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"num_samples": 0}
        return {
            "num_samples": len(self.samples),
            "num_nodes": _stats([s.num_nodes for s in self.samples]),
            "num_decisions": _stats([s.num_decisions for s in self.samples]),
            "num_candidates": _stats([s.num_candidates for s in self.samples]),
            "gt_length": _stats([s.gt_length for s in self.samples]),
            "null_fraction": _stats(
                [
                    float(np.mean(s.field.candidates.candidate_is_null))
                    for s in self.samples
                ]
            ),
        }


def _stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }
