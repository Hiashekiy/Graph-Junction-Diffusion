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
    attn_per_slot: Optional[list] = None  # 一个 reverse step 内每轮的 attention
    flow_steps: int = 1


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
        flow_steps: int = 1,
        slot_embedding: bool = True,
        slot_scale: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_node_types = int(num_node_types)
        self.num_edge_states = int(num_edge_states)
        self.flow_steps = max(1, int(flow_steps))
        self.slot_scale = float(slot_scale)

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

        # 一个 reverse step 内部要做多轮图信息交流时，用"第几轮"的 embedding 作为
        # 额外条件：同一个 Cell、同一份 E_t 反复作用才不至于退化成求不动点。
        # 它是每个轮次一个向量（与 timestep 无关），所以参数量只增加 flow_steps * d。
        if self.flow_steps > 1 and slot_embedding:
            self.flow_slot_embedding: Optional[nn.Embedding] = nn.Embedding(
                self.flow_steps, self.d_model
            )
            nn.init.normal_(self.flow_slot_embedding.weight, std=0.02)
        else:
            self.flow_slot_embedding = None

    @property
    def flow_steps_label(self) -> str:
        shared = "shared-cell" + ("+slot" if self.flow_slot_embedding is not None else "")
        return f"{self.flow_steps} round(s) per reverse step ({shared})"

    @property
    def max_flow_steps(self) -> int:
        """推理时最多能跑几轮（受 slot embedding 的行数限制）。"""
        if self.flow_slot_embedding is None:
            return 1
        return int(self.flow_slot_embedding.num_embeddings)

    def set_inference_flow_steps(
        self, flow_steps: int, allow_extrapolation: bool = False
    ) -> int:
        """推理阶段的轮数覆盖，返回原来的值。

        多轮交流的代价在推理时是线性的（实测 flow_steps=3 时 0.063 s/query，
        单轮 0.022 s/query），而"训练时见识过多轮、推理时只跑一轮"是完全合法的
        —— 第 0 轮用的 slot embedding 就是训练时第 0 轮那个。所以留一个口子，
        方便做"多少轮才够"的 ablation，而不是硬编码训练时的轮数。

        ``allow_extrapolation=True`` 时允许**超过训练轮数**：多出来的轮次会复用
        最后一行的 slot embedding（``forward_multi`` 里有 clamp）。这是一个
        分布外外推实验，用来回答"继续多跑几轮还会不会涨"，调用方应当在结果里
        标注清楚。
        """
        steps = int(flow_steps)
        if steps < 1:
            raise ValueError(f"flow_steps must be >= 1, got {steps}")
        if steps > self.max_flow_steps and not allow_extrapolation:
            raise ValueError(
                f"flow_steps={steps} exceeds the trained slot embedding rows "
                f"({self.max_flow_steps}); this model was trained with "
                f"flow_steps={self.flow_steps}, inference can only use fewer rounds "
                "(pass allow_extrapolation=True to reuse the last slot embedding)"
            )
        previous = int(self.flow_steps)
        self.flow_steps = steps
        return previous

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
    def step(
        self,
        batch,
        H_t: Tensor,
        z_t: Tensor,
        t,
        flow_steps: Optional[int] = None,
    ) -> DenoiserOutput:
        """(H_t, z_t, t) -> (H_{t-1}, candidate distribution over z_0)。

        ``flow_steps`` 给了就临时覆盖 ``self.flow_steps``（推理 ablation 用），
        不给就用模型自己的设置 —— 训练路径永远走默认值，行为不变。
        """
        steps = self.flow_steps if flow_steps is None else int(flow_steps)
        if steps < 1:
            raise ValueError(f"flow_steps must be >= 1, got {steps}")

        tau_graph = self.time_embedding(t, batch.num_graphs)
        tau_decisions = broadcast_time_to_decisions(tau_graph, batch.decision_graph_id)

        edge_feat_t = self.edge_state_encoder(batch, z_t)

        # 一个 reverse step 内部做 steps 轮图信息交流（默认 1 轮 = 旧行为）。
        # 轮与轮之间 H 继续累积，且每轮带上"第几轮"的 embedding，避免同一个 Cell
        # 反复作用于同一输入退化成求不动点。
        flow_out = self.graph_flow.forward_multi(
            H_t=H_t,
            edge_index=batch.edge_index,
            edge_feat=edge_feat_t,
            tau_t=tau_graph,
            fixed_mask=batch.start_goal_mask,
            graph_node_ptr=batch.graph_node_ptr,
            flow_steps=steps,
            slot_embedding=self.flow_slot_embedding,
            slot_scale=self.slot_scale,
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
            attn_per_slot=flow_out.get("attn_per_slot"),
            flow_steps=int(flow_out.get("flow_steps", 1)),
        )

    # 兼容旧调用写法：forward == step
    def forward(
        self, batch, H_t: Tensor, z_t: Tensor, t, flow_steps: Optional[int] = None
    ) -> DenoiserOutput:
        return self.step(batch, H_t, z_t, t, flow_steps=flow_steps)

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
