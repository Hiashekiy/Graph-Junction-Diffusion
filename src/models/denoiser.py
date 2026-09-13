"""GraphFlowDenoiser：单步去噪编排 (实施指南第 12、15、29 节).

接口从设计上强制 persistent state 必须跨 timestep 传递：

    model.init_nodes(batch)            # H_T = TypeEmbedding(V)，只调用一次
    out = model.step(batch, H_t, z_t, t)
    H_t = out.H_next                   # 下一步的输入，不重新初始化

单步顺序是固定的：

    z_t -> E_t -> H_{t-1} -> BranchScore

注意 Branch Scorer 用的是**更新之后**的 ``H_{t-1}``，不是更新之前的 ``H_t``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from src.models.branch_scorer import BranchScorer
from src.models.edge_state import EdgeStateEncoder
from src.models.graph_flow import GraphFlowBlock
from src.models.node_embedding import NodeTypeEmbedding
from src.models.time_encoder import TimeEncoder, broadcast_time_to_decisions


@dataclass
class DenoiserOutput:
    H_next: Tensor                 # [N, d]
    candidate_logits: Tensor       # [C]
    candidate_log_prob: Tensor     # [C]
    candidate_prob: Tensor         # [C]
    tau_graph: Optional[Tensor] = None   # [B, d]
    attn: Optional[Tensor] = None


class GraphFlowDenoiser(nn.Module):
    """Edge-State Conditioned Recurrent Graph Flow Denoiser."""

    def __init__(
        self,
        d_model: int = 128,
        num_node_types: int = 4,
        num_edge_states: int = 2,
        ffn_hidden: int = 256,
        dropout: float = 0.0,
        branch_hidden: Optional[int] = None,
        d_head: Optional[int] = None,
        d_time: Optional[int] = None,
        time_encoding: str = "sinusoidal",
        time_conditioning: str = "adaln",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_node_types = int(num_node_types)
        self.num_edge_states = int(num_edge_states)

        # P2-2：这两个配置项现在就真的控制行为，选到未实现的取值会立刻报错，
        # 而不是被静默忽略。
        if time_encoding != "sinusoidal":
            raise NotImplementedError(
                f"time.encoding={time_encoding!r} is not implemented (only sinusoidal)"
            )
        if time_conditioning != "adaln":
            raise NotImplementedError(
                f"time.conditioning={time_conditioning!r} is not implemented (only adaln)"
            )
        self.time_encoding = time_encoding
        self.time_conditioning = time_conditioning

        # 参数集合 Theta = {E_node, E_edge, TimeEncoder, GraphFlowBlock,
        #                   BranchScorer, NullScorer}
        self.node_encoder = NodeTypeEmbedding(self.d_model, self.num_node_types)
        self.edge_state_encoder = EdgeStateEncoder(self.d_model, self.num_edge_states)
        self.time_encoder = TimeEncoder(d_model=self.d_model, d_time=d_time)
        self.graph_flow = GraphFlowBlock(
            d_model=self.d_model,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
            d_head=d_head,
        )
        self.branch_scorer = BranchScorer(
            d_model=self.d_model, hidden_dim=branch_hidden, dropout=dropout
        )

    # ------------------------------------------------------------------
    def init_nodes(self, batch) -> Tensor:
        """H_T = TypeEmbedding(V)。整个 reverse chain 只调用一次。"""
        return self.node_encoder(batch.node_type)

    def time_embedding(self, t, num_graphs: int) -> Tensor:
        """t: int / [B] tensor -> tau_graph [B, d]。"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=next(self.parameters()).device)
        tau = self.time_encoder(t)
        if tau.dim() == 1:
            tau = tau.unsqueeze(0)
        if tau.shape[0] == 1 and num_graphs > 1:
            tau = tau.expand(num_graphs, -1)
        if tau.shape[0] != num_graphs:
            raise ValueError(
                f"time embedding has {tau.shape[0]} rows but the batch has "
                f"{num_graphs} graphs"
            )
        return tau

    # ------------------------------------------------------------------
    def step(self, batch, H_t: Tensor, z_t: Tensor, t) -> DenoiserOutput:
        """(H_t, z_t, t) -> (H_{t-1}, candidate distribution over z_0)。"""
        tau_graph = self.time_embedding(t, batch.num_graphs)
        tau_decisions = broadcast_time_to_decisions(tau_graph, batch.decision_graph_id)

        edge_feat_t = self.edge_state_encoder(batch, z_t)

        flow_out = self.graph_flow(
            H_t=H_t,
            edge_index=batch.edge_index,
            edge_feat=edge_feat_t,
            tau_t=tau_graph,
            fixed_mask=batch.start_goal_mask,
            graph_node_ptr=batch.graph_node_ptr,
        )
        H_next = flow_out["H_next"]

        scored = self.branch_scorer(H_next, batch, tau_decisions)

        return DenoiserOutput(
            H_next=H_next,
            candidate_logits=scored["candidate_logits"],
            candidate_log_prob=scored["candidate_log_prob"],
            candidate_prob=scored["candidate_prob"],
            tau_graph=tau_graph,
            attn=flow_out["attn"],
        )

    # 兼容旧调用写法：forward == step
    def forward(self, batch, H_t: Tensor, z_t: Tensor, t) -> DenoiserOutput:
        return self.step(batch, H_t, z_t, t)

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
