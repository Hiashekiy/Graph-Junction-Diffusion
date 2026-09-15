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

Weighted 扩展（方案第 5 节）：Batch 额外带两类 cost tensor，形状分别是

    physical_edge_cost / physical_edge_cost_norm   [E_phys]
    candidate_branch_cost / candidate_branch_cost_norm   [C]

cost 的读取顺序与 physical edge ID **完全一致**（都是 ``graph.edges()`` 的顺序，
见 ``branch_segments.build_edge_tables``），所以不需要改任何编号系统。归一化用
**每张图的平均边权**（纯比例缩放不改变 argmin sum w_e）。无权数据集的边取默认
1.0，归一化后恒为 1.0；只有 ``use_edge_cost=True`` 的模型才会去读这些 tensor。

Branch membership 采用 **扁平 + 定长 padding** 的布局：

    branch_node_ids      [C, max_nodes_per_branch]  padded
    branch_node_owner    [C, max_nodes_per_branch]  padded
    branch_node_lengths  [C]

这样避免二次 gather，又能直接复用 ``segment_sum`` 做 Branch Mean Pool。
padding 位置用 length mask 清零，不会污染 mean。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch
from torch import Tensor

from src.data.branch_segments import (
    NUM_EDGE_STATES,
    NUM_NODE_TYPES,
    candidate_reach_topology,
    source_reach_start,
)
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

    # -- 单出口 Source 的被迫段（P0-1，物理边永久 selected）-----------------
    source_forced_edge_ids: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.long)
    )                                  # [F] batch 级物理边 ID
    num_source_forced_edges: int = 0

    # -- weighted 扩展（方案第 5 节）：物理边 cost --------------------------
    # 形状都是 [E_phys]，与 branch membership 共用同一套 batch 级物理边编号。
    # 缺省是空 tensor（老代码 / 手工构造的 Batch 仍然能跑）：weighted 模型拿到空
    # cost 时会**显式报错**，而不是静默喂 0。
    physical_edge_cost: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )                                  # [E_phys] 原始 cost
    physical_edge_cost_norm: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )                                  # [E_phys] 除以本图平均边权
    # 每个 candidate branch 的 cost 之和（NULL = 0）。第一版只用于统计 / 诊断，
    # 还没有接进 Branch Scorer（方案第 8 节：一次只改一个模块，便于归因）。
    candidate_branch_cost: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )                                  # [C]
    candidate_branch_cost_norm: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float32)
    )                                  # [C]
    is_weighted: bool = False

    #: 本 batch 的原始 :class:`GraphSample`（**不是 tensor，``to()`` 会跳过它**）。
    #: 只有多轨迹集合损失需要：beam miner 调 ``decode_multi_path(sample, ...)``，而那个
    #: 接口吃的是 GraphSample 而不是扁平张量。不启用该损失时它就不该被读。
    graph_samples: Sequence["GraphSample"] = field(default_factory=tuple)

    # -- 软可达性拓扑（第二轮修订 B 项）-------------------------------------
    # 「从 decision i 出发走 candidate c 会到哪里」在 batch 张量层面直接查表，
    # Soft Goal 的 value iteration 因此不需要任何 NetworkX 遍历。
    candidate_next_decision: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.long)
    )                                  # [C] -1 = 这条 branch 不通向 decision
    candidate_hits_goal: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.bool)
    )                                  # [C] 这条 branch 是否直接到 goal
    reach_start_decision: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.long)
    )                                  # [B] -1 = source 后面没有 decision
    reach_start_is_goal: Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.bool)
    )                                  # [B] source 的被迫段是否直接到 goal

    # -- misc -------------------------------------------------------------
    sizes: Dict[str, Any] = field(default_factory=dict)
    device: torch.device = torch.device("cpu")

    # -- derived helpers --------------------------------------------------
    @property
    def source_decision_mask(self) -> Tensor:
        """哪些 decision 属于 Source（即 deg(s) > 1 的那些 start 节点）。

        Source 没有 NULL 候选；判断"是不是 source decision"必须逐 decision 比较
        它在本图内的局部节点编号与 start，不能拿全局编号去比。
        """
        if not self.num_decisions:
            return torch.zeros(0, dtype=torch.bool, device=self.decision_node.device)
        local = self.decision_node - self.graph_node_ptr[:-1][self.decision_graph_id]
        local_start = self.starts[self.decision_graph_id] - self.graph_node_ptr[:-1][
            self.decision_graph_id
        ]
        return local == local_start

    @property
    def start_goal_mask(self) -> Tensor:
        """Start / Goal 只出不进，Graph Flow 需要它们的 node mask。"""
        mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.node_type.device)
        if self.starts.numel():
            mask[self.starts] = True
        if self.goals.numel():
            mask[self.goals] = True
        return mask

    @property
    def has_edge_cost(self) -> bool:
        """cost tensor 是否真的覆盖了全部物理边。"""
        return int(self.physical_edge_cost.numel()) == int(self.num_physical_edges)

    def message_edge_cost(self, normalized: bool = True) -> Tensor:
        """[E_msg]：把物理边 cost 广播到两个 message 方向。

        ``msg_to_phys_edge`` 是现成的映射，所以同一条无向物理边的 u->v 与 v->u
        永远拿到同一个 cost。weighted 模型拿不到 cost 时**显式报错**：静默喂 0 会
        让"忘了开 weighted / 忘了重新 collate"变成一个查不出来的 bug。
        """
        if not self.has_edge_cost:
            raise ValueError(
                "this Batch carries no edge cost (physical_edge_cost is empty); "
                "it was either built by hand or by an older collate. Re-collate the "
                "dataset before running a model with use_edge_cost=True."
            )
        source = (
            self.physical_edge_cost_norm if normalized else self.physical_edge_cost
        )
        return source[self.msg_to_phys_edge]

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
            "num_source_forced_edges": self.num_source_forced_edges,
            "is_weighted": bool(self.is_weighted),
            "num_physical_edges_with_cost": int(self.physical_edge_cost.numel()),
            "num_goal_candidates": int(self.candidate_hits_goal.sum().item())
            if self.candidate_hits_goal.numel()
            else 0,
        }


# ---------------------------------------------------------------------------
def _pad_2d(
    values: Sequence[Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """把 ragged 的 [C, *] 整数表 padding 成 [C, L] + lengths [C]。

    实现要点：**一次**扁平构造 + index scatter，而不是逐行 ``torch.as_tensor``。

    旧写法是 ``for row in values: out[i, :len(row)] = torch.as_tensor(row, device=...)``，
    而 ``values`` 是"每个 candidate 一行"的分支成员表。在 CUDA 上每行就是一次微型
    H2D 传输：8 条 DiDi 样本（8625 个 candidate）实测 **13,310 次
    ``torch.as_tensor``，占 collate 总耗时的 94%**。改成一次构造后这部分基本消失。
    """
    row_count = len(values)
    lengths = torch.tensor([len(row) for row in values], dtype=torch.long)
    width = max(int(lengths.max()) if row_count else 1, 1)
    out = torch.zeros(row_count, width, dtype=dtype)
    total = int(lengths.sum())
    if total:
        flat = torch.tensor([v for row in values for v in row], dtype=dtype)
        row_index = torch.repeat_interleave(torch.arange(row_count), lengths)
        offsets = torch.cumsum(lengths, 0) - lengths
        column_index = torch.arange(total) - torch.repeat_interleave(offsets, lengths)
        out[row_index, column_index] = flat
    return out.to(device), lengths.to(device)


def collate_samples(
    samples: Sequence[GraphSample], device: torch.device | str = "cpu"
) -> Batch:
    """把一组 :class:`GraphSample` 拼成一个 ``Batch``。

    **实现在 CPU 上拼装，最后整体 ``.to(device)`` 一次**（对外行为完全不变）。

    为什么不能直接在 CUDA 上逐个建小张量：``torch.tensor(list, device="cuda")``
    和 ``torch.as_tensor(row, device="cuda")`` 每调用一次就是一次微型 H2D。
    这里大约有 20 个拼接张量 + 每个 candidate 一行的分支表，8 条 DiDi 样本实测
    **13,310 次逐行 CUDA 构造**。同一份数据：

        collate(device="cuda")  1.202 s
        collate(device="cpu")   0.388 s     ← 只改拼接设备就差 3.1x

    所以这里先在 CPU 拼好（连续内存、一次拷贝），需要时再整体搬过去。
    """
    if not samples:
        raise ValueError("collate_samples() got an empty sample list")
    target_device = torch.device(device)
    # 下面所有张量一律在 CPU 上构造；函数末尾统一 .to(target_device)
    device = torch.device("cpu")

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
    source_forced_edges: List[int] = []

    # weighted：逐图累加（归一化是 per-graph 的，不能跨图做）
    physical_cost: List[float] = []
    physical_cost_norm: List[float] = []
    branch_cost: List[float] = []
    branch_cost_norm: List[float] = []
    any_weighted = False

    # 软可达性拓扑（每图一条 reach_start，每个 candidate 一条 branch endpoint）
    candidate_next_decision: List[int] = []
    candidate_hits_goal: List[bool] = []
    reach_start_decision: List[int] = []
    reach_start_is_goal: List[bool] = []

    # 物理边 ID 在 segments 里是**每图局部**的（0..E_phys(g)-1），batch 里必须整体
    # 平移到 batch 级编号空间，否则第二张图起会写进/读到别的图的物理边槽位。
    physical_edge_offset = 0

    for graph_index, sample in enumerate(samples):
        segments = sample.segments
        node_offset = graph_node_ptr[-1]

        # ---- weighted：物理边 cost -------------------------------------
        # 物理边 ID 就是 graph.edges() 的顺序（branch_segments.build_edge_tables
        # 与 physical_edge_lookup 都按这个顺序编号），所以按同样顺序读一遍即可得到
        # 逐边对齐的 cost。无权图的边没有 weight 属性 -> 1.0。
        raw_cost = [
            float(sample.graph.edges[u, v].get("weight", 1.0))
            for u, v in sample.graph.edges()
        ]
        if len(raw_cost) != segments.num_physical_edges:
            raise ValueError(
                f"graph has {len(raw_cost)} edges but its segments declare "
                f"{segments.num_physical_edges} physical edges; the sample is stale"
            )
        mean_cost = (sum(raw_cost) / len(raw_cost)) if raw_cost else 0.0
        if raw_cost and (not math.isfinite(mean_cost) or mean_cost <= 0.0):
            raise ValueError(
                f"edge weights must be positive and finite; got mean={mean_cost} "
                f"over {len(raw_cost)} edges"
            )
        physical_cost.extend(raw_cost)
        physical_cost_norm.extend(
            [value / mean_cost for value in raw_cost] if raw_cost else []
        )
        any_weighted = any_weighted or bool(sample.graph.graph.get("weighted", False))

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
        # 单出口 Source 的被迫段：这些物理边永久 selected（P0-1）
        source_forced_edges.extend(
            physical_edge_offset + int(edge)
            for edge in segments.source_forced_edge_ids
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

        # 软可达性：每个 candidate 走到的 structural endpoint 是什么
        local_next, local_hits = candidate_reach_topology(
            segments, candidates.candidate_branch
        )
        for target, hits in zip(local_next, local_hits):
            candidate_next_decision.append(
                decision_offset + int(target) if int(target) >= 0 else -1
            )
            candidate_hits_goal.append(bool(hits))

        # source 的起点（deg(s) > 1 时 source 自己是 decision；否则从 forced 段终点开始）
        local_start_decision, start_is_goal = source_reach_start(segments)
        reach_start_decision.append(
            decision_offset + int(local_start_decision)
            if int(local_start_decision) >= 0
            else -1
        )
        reach_start_is_goal.append(bool(start_is_goal))

        # branch membership（排除 owner；NULL 留空）+ branch cost 求和
        for branch in candidates.candidate_branch:
            if branch is None:
                branch_node_ids.append([])
                branch_edge_ids.append([])
                # C(NULL) = 0（方案第 8 节）
                branch_cost.append(0.0)
                branch_cost_norm.append(0.0)
                continue
            branch_node_ids.append([node_offset + int(v) for v in branch.nodes[1:]])
            branch_edge_ids.append(
                [physical_edge_offset + int(edge) for edge in branch.physical_edges]
            )
            # branch.physical_edges 是**本图局部**编号，直接索引 raw_cost
            raw_branch = sum(raw_cost[int(edge)] for edge in branch.physical_edges)
            branch_cost.append(raw_branch)
            branch_cost_norm.append(raw_branch / mean_cost if mean_cost > 0 else 0.0)

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
        source_forced_edge_ids=torch.tensor(
            source_forced_edges, dtype=torch.long, device=device
        ),
        num_source_forced_edges=len(source_forced_edges),
        physical_edge_cost=torch.tensor(physical_cost, dtype=torch.float32, device=device),
        physical_edge_cost_norm=torch.tensor(
            physical_cost_norm, dtype=torch.float32, device=device
        ),
        candidate_branch_cost=torch.tensor(
            branch_cost, dtype=torch.float32, device=device
        ),
        candidate_branch_cost_norm=torch.tensor(
            branch_cost_norm, dtype=torch.float32, device=device
        ),
        is_weighted=bool(any_weighted),
        graph_samples=tuple(samples),
        candidate_next_decision=torch.tensor(
            candidate_next_decision, dtype=torch.long, device=device
        ),
        candidate_hits_goal=torch.tensor(
            candidate_hits_goal, dtype=torch.bool, device=device
        ),
        reach_start_decision=torch.tensor(
            reach_start_decision, dtype=torch.long, device=device
        ),
        reach_start_is_goal=torch.tensor(
            reach_start_is_goal, dtype=torch.bool, device=device
        ),
        sizes={
            "num_node_types": NUM_NODE_TYPES,
            "num_edge_states": NUM_EDGE_STATES,
            "branches_per_decision": [
                len(group) for sample in samples for group in sample.segments.branches
            ],
        },
        device=device,
    )
    # 一次性搬到目标设备（CPU 目标时零拷贝，直接返回）
    return batch if target_device.type == "cpu" else batch.to(target_device)


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
