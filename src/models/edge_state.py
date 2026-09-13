"""Edge-State Encoder (实施指南第 9 节 / 设计报告 V2.1 第 4 节).

边状态只有两类：

    0 = unselected      1 = selected

关键不是 embedding 本身，而是

    z_t  ->  edge_state_id_t  ->  E_edge[edge_state_id_t]

展开链路（全部向量化，无 Python 循环）：

    1. z_t 里选中的 candidate（NULL 不写入任何 selected 边）
    2. branch_edge membership 中属于"已选中 candidate"的条目
    3. OR 合并到物理边            (amax over physical edge)
    4. 复制到两个 message 方向     (msg_to_phys_edge)
    5. embedding lookup

同一条物理边的两个方向永远拿到同一个 state（设计报告第 4.3 节的并集规则）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.data.branch_segments import NUM_EDGE_STATES, SELECTED, UNSELECTED


def expand_to_edge_state(
    z_t: Tensor,                       # [M] flat candidate index
    candidate_is_null: Tensor,         # [C] bool
    branch_edge_ids: Tensor,           # [C, Le] padded
    branch_edge_owner: Tensor,         # [C, Le] padded
    branch_edge_lengths: Tensor,       # [C]
    msg_to_phys_edge: Tensor,          # [E_msg]
    num_physical_edges: int,
) -> Tensor:
    """z_t -> edge_state_id [E_msg]，取值 0/1。"""
    device = z_t.device
    selected_candidate = torch.zeros(
        candidate_is_null.numel(), dtype=torch.bool, device=device
    )
    valid = ~candidate_is_null[z_t]
    if valid.any():
        selected_candidate[z_t[valid]] = True

    physical_state = torch.full(
        (max(num_physical_edges, 1),),
        UNSELECTED,
        dtype=torch.long,
        device=device,
    )

    if branch_edge_ids.numel() and selected_candidate.any():
        member_active = selected_candidate[branch_edge_owner]           # [C, Le]
        if branch_edge_lengths.numel():
            positions = torch.arange(
                branch_edge_ids.shape[1], device=device
            )[None, :]
            member_active = member_active & (positions < branch_edge_lengths[:, None])
        if member_active.any():
            indices = branch_edge_ids[member_active]
            physical_state.scatter_reduce_(
                0,
                indices,
                torch.full_like(indices, SELECTED),
                reduce="amax",
                include_self=True,
            )

    if num_physical_edges == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    return physical_state[msg_to_phys_edge]


class EdgeStateEncoder(nn.Module):
    """z_t -> edge features [E_msg, d]。"""

    def __init__(self, d_model: int = 128, num_edge_states: int = NUM_EDGE_STATES):
        super().__init__()
        self.d_model = int(d_model)
        self.num_edge_states = int(num_edge_states)
        self.embedding = nn.Embedding(self.num_edge_states, self.d_model)

    def state_ids(self, batch, z_t: Tensor) -> Tensor:
        return expand_to_edge_state(
            z_t=z_t,
            candidate_is_null=batch.candidate_is_null,
            branch_edge_ids=batch.branch_edge_ids,
            branch_edge_owner=batch.branch_edge_owner,
            branch_edge_lengths=batch.branch_edge_lengths,
            msg_to_phys_edge=batch.msg_to_phys_edge,
            num_physical_edges=batch.num_physical_edges,
        )

    def forward(self, batch, z_t: Tensor) -> Tensor:
        return self.embedding(self.state_ids(batch, z_t))
