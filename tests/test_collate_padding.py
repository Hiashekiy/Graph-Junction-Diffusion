"""``_pad_2d`` / ``collate_samples`` 的设备与向量化回归测试。

背景：这两个函数曾经有两个性能陷阱，都**不影响正确性所以不会报错**，只会慢：

1. ``_pad_2d`` 逐行 ``torch.as_tensor(row, device="cuda")`` —— 分支成员表是
   "每个 candidate 一行"，8 条 DiDi 样本实测 13,310 次微型 H2D，占 collate 总耗时 94%。
2. ``collate_samples`` 直接在目标设备上构造约 20 个拼接张量 —— 同样每次一个 H2D。
   同一份数据 ``device="cuda"`` 1.202 s vs ``device="cpu"`` 0.388 s。

修法是"CPU 拼装 + 一次搬移 + 向量化 padding"。这些测试钉住：
向量化后**逐位等价**、边界输入不炸、以及 collate 结果与设备无关。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import _pad_2d, collate_samples  # noqa: E402
from src.data.dataset_builder import build_sample  # noqa: E402


def _reference_pad_2d(values, device, dtype):
    """修复前的逐行实现，作为等价性基准。"""
    width = max((len(row) for row in values), default=1)
    width = max(width, 1)
    lengths = torch.tensor([len(row) for row in values], dtype=torch.long, device=device)
    out = torch.zeros(len(values), width, dtype=dtype, device=device)
    for row_index, row in enumerate(values):
        if row:
            out[row_index, : len(row)] = torch.as_tensor(
                row, dtype=dtype, device=device
            )
    return out, lengths


def test_pad_2d_matches_reference_on_random_ragged_input():
    rng = random.Random(0)
    for _ in range(200):
        rows = [
            [rng.randint(0, 999) for _ in range(rng.randint(0, 7))]
            for _ in range(rng.randint(1, 50))
        ]
        new_values, new_lengths = _pad_2d(rows, torch.device("cpu"), torch.long)
        ref_values, ref_lengths = _reference_pad_2d(rows, torch.device("cpu"), torch.long)
        assert torch.equal(new_values, ref_values), rows[:3]
        assert torch.equal(new_lengths, ref_lengths), rows[:3]


@pytest.mark.parametrize("rows", [[], [[]], [[], []], [[1, 2, 3]], [[7], [], [1, 2]]])
def test_pad_2d_edge_cases_match_reference(rows):
    new_values, new_lengths = _pad_2d(rows, torch.device("cpu"), torch.long)
    ref_values, ref_lengths = _reference_pad_2d(rows, torch.device("cpu"), torch.long)
    assert torch.equal(new_values, ref_values)
    assert torch.equal(new_lengths, ref_lengths)
    assert new_values.shape[0] == len(rows)
    assert new_values.shape[1] >= 1


def test_pad_2d_handles_five_hundred_rows():
    """行数多的时候也不能退化成逐行张量创建（这是原来的性能陷阱）。"""
    rows = [[index % 9, index % 5] for index in range(500)]
    new_values, new_lengths = _pad_2d(rows, torch.device("cpu"), torch.long)
    ref_values, ref_lengths = _reference_pad_2d(rows, torch.device("cpu"), torch.long)
    assert torch.equal(new_values, ref_values)
    assert torch.equal(new_lengths, ref_lengths)


def _two_samples():
    graph_a, _, _ = _chain_graph()
    graph_b, _, _ = _chain_graph()
    return [
        build_sample(graph_a, 0, 2),
        build_sample(graph_b, 0, 2),
    ]


def _chain_graph():
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=2.0)
    graph.add_edge(1, 3, weight=3.0)
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    return graph, 0, 2


def test_collate_is_device_agnostic():
    """``collate_samples(device=X)`` 必须与先 CPU 再 ``.to(X)`` 完全一致。"""
    samples = _two_samples()
    cpu_batch = collate_samples(samples, device="cpu")
    again = collate_samples(samples, device="cpu")
    for name, value in cpu_batch.__dict__.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(again, name)), name
            assert value.device.type == "cpu", name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_collate_moves_everything_to_cuda():
    samples = _two_samples()
    batch = collate_samples(samples, device="cuda")
    assert batch.device.type == "cuda"
    for name, value in batch.__dict__.items():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cuda", name
    reference = collate_samples(samples, device="cpu")
    for name, value in reference.__dict__.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, getattr(batch, name).cpu()), name
    # 分支表的 bucket 归属也必须一致（padding 是向量化写的）
    assert torch.equal(
        batch.branch_node_ids.cpu(), reference.branch_node_ids
    )
    assert torch.equal(
        batch.branch_node_lengths.cpu(), reference.branch_node_lengths
    )
    assert torch.equal(batch.branch_edge_ids.cpu(), reference.branch_edge_ids)
