"""Segmented (grouped) reductions and softmax (guide section 13).

Used for two things:

1. `grouped_log_softmax(logits, candidate_owner)` - the variable-cardinality
   classification head.  Every decision node owns a contiguous slice of the
   flattened candidate table and softmax is applied *inside* each slice, so

       for every decision node i:   sum_{c in C_i} p_i(c) = 1

2. edge-softmax inside the edge-aware graph transformer, where the segments are
   the destination nodes of the message-passing edges.

All primitives support trailing feature dimensions (dim 0 is the segment dim)
and are implemented with `scatter_reduce` / `index_add_`, i.e. pure PyTorch, so
no torch_scatter dependency is needed.
"""

from __future__ import annotations

import torch
from torch import Tensor

_NEG_INF = -1e30


def _align_index(index: Tensor, ndim: int) -> Tensor:
    out = index
    while out.dim() < ndim:
        out = out.unsqueeze(-1)
    return out


def segment_max(logits: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Segment-wise maximum, shape [num_segments, *logits.shape[1:]]."""
    shape = (int(num_segments),) + tuple(logits.shape[1:])
    out = torch.full(shape, _NEG_INF, dtype=logits.dtype, device=logits.device)
    if logits.numel() == 0:
        return out
    idx = _align_index(index, logits.dim()).expand_as(logits)
    return out.scatter_reduce(0, idx, logits, reduce="amax", include_self=True)


def segment_sum(values: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Segment-wise sum, shape [num_segments, *values.shape[1:]]."""
    shape = (int(num_segments),) + tuple(values.shape[1:])
    out = torch.zeros(shape, dtype=values.dtype, device=values.device)
    if values.numel() == 0:
        return out
    idx = _align_index(index, values.dim()).expand_as(values)
    return out.scatter_add_(0, idx, values)


def segment_log_softmax(logits: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """log softmax computed independently inside every segment."""
    if logits.numel() == 0:
        return logits
    max_value = segment_max(logits, index, num_segments)
    shifted = logits - max_value[index]
    exp_shifted = shifted.exp()
    denominator = segment_sum(exp_shifted, index, num_segments)
    log_denominator = denominator.clamp_min(1e-30).log()
    return shifted - log_denominator[index]


def segment_softmax(
    logits: Tensor, index: Tensor, num_segments: int, dtype: torch.dtype | None = None
) -> Tensor:
    """softmax computed independently inside every segment (float32 internally)."""
    out = segment_log_softmax(logits.float(), index, num_segments).exp()
    target_dtype = logits.dtype if dtype is None else dtype
    return out.to(target_dtype)


def grouped_log_softmax(logits: Tensor, candidate_owner: Tensor, num_groups: int) -> Tensor:
    """Softmax over the candidate group of every decision node (float32)."""
    return segment_log_softmax(logits.float(), candidate_owner, num_groups)


def grouped_softmax(logits: Tensor, candidate_owner: Tensor, num_groups: int) -> Tensor:
    return segment_softmax(logits.float(), candidate_owner, num_groups)
