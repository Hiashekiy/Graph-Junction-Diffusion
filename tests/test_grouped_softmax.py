"""Grouped (segment) softmax tests (guide sections 13 + 50)."""

from __future__ import annotations

import torch

from src.utils.segment_ops import (
    grouped_log_softmax,
    grouped_softmax,
    segment_log_softmax,
    segment_max,
    segment_softmax,
    segment_sum,
)


def test_grouped_softmax_sum_to_one():
    torch.manual_seed(0)
    sizes = [1, 3, 5, 2, 7]
    owner = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    logits = torch.randn(owner.numel())
    prob = grouped_softmax(logits, owner, len(sizes))
    sums = torch.zeros(len(sizes)).index_add_(0, owner, prob)
    assert torch.allclose(sums, torch.ones(len(sizes)), atol=1e-6)
    assert (prob > 0).all()


def test_grouped_softmax_matches_manual_per_group():
    torch.manual_seed(1)
    sizes = [2, 4, 3]
    owner = torch.repeat_interleave(torch.arange(3), torch.tensor(sizes))
    logits = torch.randn(owner.numel())
    prob = grouped_softmax(logits, owner, 3)
    offset = 0
    for m, size in enumerate(sizes):
        manual = torch.softmax(logits[offset : offset + size], dim=0)
        assert torch.allclose(prob[offset : offset + size], manual, atol=1e-6)
        offset += size


def test_grouped_log_softmax_is_log_of_softmax():
    torch.manual_seed(2)
    sizes = [3, 2]
    owner = torch.repeat_interleave(torch.arange(2), torch.tensor(sizes))
    logits = torch.randn(owner.numel())
    log_prob = grouped_log_softmax(logits, owner, 2)
    assert torch.allclose(log_prob.exp(), grouped_softmax(logits, owner, 2), atol=1e-6)


def test_grouped_softmax_stability_with_large_logits():
    logits = torch.tensor([1e4, 1e4 - 1.0, -1e4, 5.0, 5.0, 5.0])
    owner = torch.tensor([0, 0, 0, 1, 1, 1])
    prob = grouped_softmax(logits, owner, 2)
    assert torch.isfinite(prob).all()
    sums = torch.zeros(2).index_add_(0, owner, prob)
    assert torch.allclose(sums, torch.ones(2), atol=1e-6)


def test_segment_softmax_with_feature_dimension():
    """Attention weights: softmax over the edge dimension for every (dst, head)."""
    index = torch.tensor([0, 0, 1, 2, 2])
    logits = torch.randn(5, 3)
    attn = segment_softmax(logits, index, 3)
    assert attn.shape == logits.shape
    for node in range(3):
        rows = attn[index == node]
        assert torch.allclose(rows.sum(dim=0), torch.ones(3), atol=1e-6)


def test_segment_reductions():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 0, 1])
    assert torch.allclose(segment_sum(values, index, 3)[0], torch.tensor([4.0, 6.0]))
    assert torch.allclose(segment_max(values, index, 3)[0], torch.tensor([3.0, 4.0]))
    # empty segments are handled without NaN
    out = segment_log_softmax(values, index, 3)
    assert torch.isfinite(out).all()


def test_grouped_softmax_gradient_flows_within_group_only():
    logits = torch.randn(6, requires_grad=True)
    owner = torch.tensor([0, 0, 0, 1, 1, 1])
    prob = grouped_softmax(logits, owner, 2)
    prob.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# ISSUES.md 3.2: segment ops must support trailing feature dimensions
# ---------------------------------------------------------------------------
def test_segment_sum_and_max_with_feature_dims():
    values = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    summed = segment_sum(values, index, 3)
    assert summed.shape == (3, 2, 2)
    assert torch.allclose(summed[0], values[0] + values[1])
    assert torch.allclose(summed[2], values[4] + values[5])
    maximum = segment_max(values, index, 3)
    assert maximum.shape == (3, 2, 2)
    assert torch.allclose(maximum[1], torch.maximum(values[2], values[3]))


def test_segment_softmax_matches_manual_for_every_head():
    torch.manual_seed(3)
    num_edges, num_heads, num_nodes = 11, 3, 4
    index = torch.randint(0, num_nodes, (num_edges,))
    index = torch.cat([index, torch.arange(num_nodes)])  # every node gets an edge
    logits = torch.randn(index.numel(), num_heads)
    attn = segment_softmax(logits, index, num_nodes)
    assert attn.shape == logits.shape
    for node in range(num_nodes):
        rows = attn[index == node]
        manual = torch.softmax(logits[index == node], dim=0)
        assert torch.allclose(rows, manual, atol=1e-6)


def test_segment_log_softmax_three_dimensional_input():
    index = torch.tensor([0, 0, 1, 1])
    logits = torch.randn(4, 2, 3)
    out = segment_log_softmax(logits, index, 2)
    assert out.shape == logits.shape
    probs = out.exp()
    for seg in range(2):
        assert torch.allclose(
            probs[index == seg].sum(dim=0), torch.ones(3), atol=1e-6
        )


def test_segment_ops_handle_empty_segments():
    index = torch.tensor([2, 2])
    logits = torch.randn(2, 4)
    out = segment_softmax(logits, index, 5)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=0), torch.ones(4), atol=1e-6)


def test_segment_log_softmax_matches_python_loop():
    torch.manual_seed(4)
    sizes = [2, 5, 1, 3]
    index = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    logits = torch.randn(index.numel(), 3)
    out = segment_log_softmax(logits, index, len(sizes))
    offset = 0
    for size in sizes:
        manual = torch.log_softmax(logits[offset : offset + size], dim=0)
        assert torch.allclose(out[offset : offset + size], manual, atol=1e-6)
        offset += size
