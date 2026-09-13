"""Batch / Collate：一次把后面需要的 tensor 全部准备好 (实施指南第 6 节).

模型内部**不做任何 NetworkX 遍历**。约定几个编号空间：

    N        全局节点        node_type / node_graph_id
    E_msg    全局有向消息边  edge_index / msg_to_phys_edge
    E_phys   全局物理无向边  （batch 级编号，branch membership 用物理边 ID）
    M        全局 decision    decision_node / decision_graph_id
    C        全局 candidate   candidate_owner / candidate_is_null
    G        batch 中的图     starts / goals / graph_node_ptr

所有编号空间都在 batch 级：节点、decision、candidate 加各自的 offset，
**物理边 ID 同样加 per-graph offset**（`segment` 内部保持局部 ID，collate 时平移）。

Branch membership 采用 **扁平 + 定长 padding** 的布局：

    branch_node_ids      [C, max_nodes_per_branch]  padded
    branch_node_owner    [C, max_nodes_per_branch]  padded
    branch_node_lengths  [C]

这样避免二次 gather，又能直接复用 ``segment_sum`` 做 Branch Mean Pool。
padding 位置用 length mask 清零，不会污染 mean。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor

from src.data.branch_segments import NUM_EDGE_STATES, NUM_NODE_TYPES
from src.data.dataset import GraphSample


# ---------------------------------------------------------------------------
@dataclass
class Batch:
    """一个 batch 里所有图 / decision / candidate 的扁平张量。"""

    # -- graph ------------------------------------------------------------
    node_type: Tensor                 # [N]
    edge_index: Tensor                # [2, E_msg]
    msg_to_phys_edge: Tensor          # [E_msg]
    num_physical_edges: int
    node_graph_id: Tensor             # [N]
    graph_node_ptr: Tensor            # [B+1]
    starts: Tensor                    # [B]
    goals: Tensor                     # [B]
    num_graphs: int
    num_nodes: int

    # -- decision ---------------------------------------------------------
    decision_node: Tensor             # [M]
    decision_graph_id: Tensor         # [M]
    num_decisions: int

    # -- candidates -------------------------------------------------------
    candidate_owner: Tensor           # [C]
    candidate_is_null: Tensor         # [C] bool
    target_candidate: Tensor          # [M]
    num_candidates: int

    # -- branch membership (padded) ---------------------------------------
    branch_node_ids: Tensor           # [C, Ln]
    branch_node_owner: Tensor         # [C, Ln]
    branch_node_lengths: Tensor       # [C]
    branch_edge_ids: Tensor           # [C, Le]
    branch_edge_owner: Tensor         # [C, Le]
    branch_edge_lengths: Tensor       # [C]

    # -- misc -------------------------------------------------------------
    sizes: Dict[str, Any] = field(default_factory=dict)
    device: torch.device = torch.device("cpu")

    # -- derived helpers --------------------------------------------------
    @property
    def start_goal_mask(self) -> Tensor:
        """Start / Goal 只出不进，Graph Flow 需要它们的 node mask。"""
        mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.node_type.device)
        if self.starts.numel():
            mask[self.starts] = True
        if self.goals.numel():
            mask[self.goals] = True
        return mask

    def to(self, device: torch.device | str) -> "Batch":
        device = torch.device(device)
        moved = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                moved[name] = value.to(device)
        for name, value in moved.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "device", device)
        return self

    def describe(self) -> Dict[str, Any]:
        return {
            "num_graphs": self.num_graphs,
            "num_nodes": self.num_nodes,
            "num_message_edges": int(self.edge_index.shape[1]),
            "num_physical_edges": self.num_physical_edges,
            "num_decisions": self.num_decisions,
            "num_candidates": self.num_candidates,
            "num_null_candidates": int(self.candidate_is_null.sum().item()),
            "branch_node_slots": int(self.branch_node_ids.numel()),
            "branch_edge_slots": int(self.branch_edge_ids.numel()),
        }


# ---------------------------------------------------------------------------
def _pad_2d(
    values: Sequence[Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """把 ragged 的 [C, *] 整数表 padding 成 [C, L] + lengths [C]。"""
    width = max((len(row) for row in values), default=1)
    width = max(width, 1)
    lengths = torch.tensor([len(row) for row in values], dtype=torch.long, device=device)
    out = torch.zeros(len(values), width, dtype=dtype, device=device)
    for row_index, row in enumerate(values):
        if row:
            out[row_index, : len(row)] = torch.as_tensor(row, dtype=dtype, device=device)
    return out, lengths


def collate_samples(
    samples: Sequence[GraphSample], device: torch.device | str = "cpu"
) -> Batch:
    """把一组 :class:`GraphSample` 拼成一个 :class:`Batch`。"""
    if not samples:
        raise ValueError("collate_samples() got an empty sample list")
    device = torch.device(device)

    node_type: List[int] = []
    edge_index: List[List[int]] = [[], []]
    msg_to_phys: List[int] = []
    node_graph_id: List[int] = []
    graph_node_ptr: List[int] = [0]
    starts: List[int] = []
    goals: List[int] = []

    decision_node: List[int] = []
    decision_graph_id: List[int] = []
    candidate_owner: List[int] = []
    candidate_is_null: List[bool] = []
    target_candidate: List[int] = []

    branch_node_ids: List[List[int]] = []
    branch_edge_ids: List[List[int]] = []

    # 物理边 ID 在 segments 里是**每图局部**的（0..E_phys(g)-1），batch 里必须整体
    # 平移到 batch 级编号空间，否则第二张图起会写进/读到别的图的物理边槽位。
    physical_edge_offset = 0

    for graph_index, sample in enumerate(samples):
        segments = sample.segments
        node_offset = graph_node_ptr[-1]

        node_type.extend(
            segments.node_type[node] for node in range(segments.num_nodes)
        )
        node_graph_id.extend([graph_index] * segments.num_nodes)
        graph_node_ptr.append(node_offset + segments.num_nodes)
        starts.append(node_offset + segments.start)
        goals.append(node_offset + segments.goal)

        for src, dst in segments.edge_index:
            edge_index[0].append(node_offset + src)
            edge_index[1].append(node_offset + dst)
        msg_to_phys.extend(
            physical_edge_offset + int(phys) for phys in segments.msg_to_phys_edge
        )

        decision_offset = len(decision_node)
        candidate_offset = len(candidate_owner)

        for local_decision, _ in enumerate(segments.decision_nodes):
            decision_node.append(node_offset + segments.decision_nodes[local_decision])
            decision_graph_id.append(graph_index)

        candidates = sample.field.candidates
        for local_owner, is_null in zip(candidates.candidate_owner, candidates.candidate_is_null):
            candidate_owner.append(decision_offset + int(local_owner))
            candidate_is_null.append(bool(is_null))

        for local_target in candidates.target_candidate:
            target_candidate.append(candidate_offset + int(local_target))

        # branch membership（排除 owner；NULL 留空）
        for branch in candidates.candidate_branch:
            if branch is None:
                branch_node_ids.append([])
                branch_edge_ids.append([])
                continue
            branch_node_ids.append([node_offset + int(v) for v in branch.nodes[1:]])
            branch_edge_ids.append(
                [physical_edge_offset + int(edge) for edge in branch.physical_edges]
            )

        physical_edge_offset += segments.num_physical_edges

    num_physical_edges = physical_edge_offset
    num_candidates = len(candidate_owner)

    node_ids, node_lengths = _pad_2d(branch_node_ids, device, torch.long)
    edge_ids, edge_lengths = _pad_2d(branch_edge_ids, device, torch.long)

    # branch_node_owner / branch_edge_owner 就是 candidate 自己（每行 constant）
    candidate_ids = torch.arange(num_candidates, dtype=torch.long, device=device)
    node_owner = candidate_ids[:, None].expand_as(node_ids).contiguous()
    edge_owner = candidate_ids[:, None].expand_as(edge_ids).contiguous()

    batch = Batch(
        node_type=torch.tensor(node_type, dtype=torch.long, device=device),
        edge_index=torch.tensor(edge_index, dtype=torch.long, device=device),
        msg_to_phys_edge=torch.tensor(msg_to_phys, dtype=torch.long, device=device),
        num_physical_edges=num_physical_edges,
        node_graph_id=torch.tensor(node_graph_id, dtype=torch.long, device=device),
        graph_node_ptr=torch.tensor(graph_node_ptr, dtype=torch.long, device=device),
        starts=torch.tensor(starts, dtype=torch.long, device=device),
        goals=torch.tensor(goals, dtype=torch.long, device=device),
        num_graphs=len(samples),
        num_nodes=len(node_type),
        decision_node=torch.tensor(decision_node, dtype=torch.long, device=device),
        decision_graph_id=torch.tensor(decision_graph_id, dtype=torch.long, device=device),
        num_decisions=len(decision_node),
        candidate_owner=torch.tensor(candidate_owner, dtype=torch.long, device=device),
        candidate_is_null=torch.tensor(candidate_is_null, dtype=torch.bool, device=device),
        target_candidate=torch.tensor(target_candidate, dtype=torch.long, device=device),
        num_candidates=num_candidates,
        branch_node_ids=node_ids,
        branch_node_owner=node_owner,
        branch_node_lengths=node_lengths,
        branch_edge_ids=edge_ids,
        branch_edge_owner=edge_owner,
        branch_edge_lengths=edge_lengths,
        sizes={
            "num_node_types": NUM_NODE_TYPES,
            "num_edge_states": NUM_EDGE_STATES,
            "branches_per_decision": [
                len(group) for sample in samples for group in sample.segments.branches
            ],
        },
        device=device,
    )
    return batch


def null_candidate_of_decision(batch: Batch) -> Tensor:
    """每个 decision 的 NULL flat candidate index（没有 NULL 时为 -1）。"""
    out = torch.full(
        (batch.num_decisions,), -1, dtype=torch.long, device=batch.candidate_owner.device
    )
    is_null = batch.candidate_is_null
    if is_null.any():
        owners = batch.candidate_owner[is_null]
        indices = torch.nonzero(is_null, as_tuple=False).squeeze(1)
        out[owners] = indices
    return out


def decision_group_sizes(batch: Batch) -> Tensor:
    return torch.bincount(
        batch.candidate_owner, minlength=batch.num_decisions
    ).to(batch.candidate_owner.device)


# ---------------------------------------------------------------------------
def iter_batches(
    samples: Sequence[GraphSample],
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
) -> List[List[GraphSample]]:
    """把样本切成 mini-batch（返回列表，方便配合 epoch 级进度打印）。"""
    indices = list(range(len(samples)))
    if shuffle:
        rng = torch.Generator().manual_seed(int(seed))
        indices = torch.randperm(len(indices), generator=rng).tolist()

    batches: List[List[GraphSample]] = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        if drop_last and len(chunk) < batch_size:
            continue
        batches.append([samples[int(i)] for i in chunk])
    return batches
