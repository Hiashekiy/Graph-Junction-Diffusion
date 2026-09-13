"""Graph Flow Block (实施指南第 10-11 节 / 设计报告 V2.1 第 5-7、11 节).

一个 reverse step 只跑**一次** Graph Flow；这个 Cell 在所有 timestep 上共享参数：

    H_{t-1} = F_theta(H_t, E_t, tau_t)

注意力角色分工（这是 V2 的核心；第二轮修订后 Key 同时吃 sender 节点与边状态）：

    Q_v      = W_Q * h_hat_v                             接收节点  "我需要什么信息"
    K_{uv}   = W_{K,n} * h_hat_u + W_{K,e} * e_uv^t      发送节点 + 边状态
                                                         "你是谁 + 这条通道什么状态"
    V_u      = W_V * h_hat_u                             发送节点  "传播什么信息"

    score    = <q_v, (k_node + k_edge) / sqrt(2)> / sqrt(d)
    alpha    = softmax_{u in N(v)} score
    m_v      = sum_u alpha_uv * V_u

旧实现里 K **只**来自 edge feature，于是同一接收节点的两条入边只要边状态相同，
attention 必然相等，sender 的节点身份（例如 Start / Goal 的独立 embedding）
根本进不了权重。把 node / edge 两路 Key 相加后 attention 才能真正区分 sender。

``1/sqrt(2)`` 只是让 ``k_node + k_edge`` 的初始方差与单路 Key 一致（两路独立、
各自方差相同，相加后方差翻倍，除以 sqrt(2) 抵消），不改变表达能力。

三条硬约束：

1. **Start / Goal 只出不进**：所有 dst 是 start/goal 的 message edge 在 softmax
   和聚合里都被剔除（指南第 10.4 节，不依赖网络自己学）。
2. **必须有 residual**：``H_next = LN(H + W_O m)`` 再 ``LN(H~ + FFN(H~))``，
   而不是 ``H_next = m``（指南第 11 节）。
3. **Start / Goal 的状态被 clamp 回输入**：``H_next[fixed] = H_t[fixed]``，
   用 ``torch.where`` 写（对 autograd 更友好）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional
import torch
from torch import Tensor, nn

from src.models.time_encoder import TimeConditioner, broadcast_time
from src.utils.nn import build_mlp
from src.utils.segment_ops import segment_softmax


class GraphFlowBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        ffn_hidden: int = 256,
        dropout: float = 0.0,
        d_head: Optional[int] = None,
        ffn_layers: int = 2,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.ffn_hidden = int(ffn_hidden)
        # 第一版固定单头，且 d_head == d_model（实施指南第 10.3 节写明 q/k/v 都是
        # [E_msg, d]）。不做 multi-head。
        self.d_head = int(d_head or self.d_model)
        self.attn_scale = float(self.d_head) ** -0.5

        self.node_norm = nn.LayerNorm(self.d_model)
        self.time_conditioner = TimeConditioner(self.d_model, dropout=dropout)

        self.q_proj = nn.Linear(self.d_model, self.d_head)
        # Key 拆成"发送节点"与"边状态"两路，forward 里相加（第二轮修订 A 项）。
        self.k_node_proj = nn.Linear(self.d_model, self.d_head)
        self.k_edge_proj = nn.Linear(self.d_model, self.d_head)
        self.v_proj = nn.Linear(self.d_model, self.d_head)
        self.o_proj = nn.Linear(self.d_head, self.d_model)
        # k = (k_node + k_edge) / sqrt(2)：保持初始 Key 方差与单路一致
        self.k_pair_scale = 1.0 / math.sqrt(2.0)

        # 两个 residual 子层各用一套 LayerNorm 参数（修改清单 P1-3）：
        #   h~      = LN_1(h + W_O m)
        #   h^{t-1} = LN_2(h~ + FFN(h~))
        # 之前两处共用同一个 ffn_norm，等于强制两个 stage 共享仿射参数。
        self.attn_out_norm = nn.LayerNorm(self.d_model)
        self.ffn_out_norm = nn.LayerNorm(self.d_model)
        self.ffn = build_mlp(
            self.d_model,
            self.ffn_hidden,
            self.d_model,
            num_layers=ffn_layers,
            activation="silu",
            normalization="none",
            dropout=dropout,
        )

        # 让 Q / K / V 的初始尺度与 d_model 匹配
        for projection in (
            self.q_proj,
            self.k_node_proj,
            self.k_edge_proj,
            self.v_proj,
        ):
            nn.init.normal_(projection.weight, std=self.d_model ** -0.5)
            nn.init.zeros_(projection.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        H_t: Tensor,                 # [N, d] persistent node state
        edge_index: Tensor,          # [2, E_msg]
        edge_feat: Tensor,           # [E_msg, d] edge-state embedding
        tau_t: Tensor,               # [B, d] 每个图当前 timestep 的 embedding
        fixed_mask: Tensor,          # [N] bool, Start/Goal
        graph_node_ptr: Tensor,      # [B+1]
        slot_tau: Optional[Tensor] = None,   # [B, d] 本轮（step 内第几轮）的 embedding
    ) -> Dict[str, Any]:
        """一次 Graph Flow：``H_next = F_theta(H_t, E_t, tau_t [+ slot_tau])``。

        ``slot_tau`` 是"同一个 reverse step 内部第几轮交流"的 embedding。多轮交流时
        每轮传入不同的 ``slot_tau``，这样即使是同一个 Cell、同一份 ``E_t``，各轮也
        不会退化成"把同一个映射反复应用求不动点"。
        """
        if edge_index.numel():
            src, dst = edge_index[0], edge_index[1]
        else:
            src = dst = torch.zeros(0, dtype=torch.long, device=H_t.device)

        # ---- timestep conditioning (AdaLN / FiLM) ---------------------
        condition = tau_t if slot_tau is None else tau_t + slot_tau
        gamma, beta = self.time_conditioner(condition)             # [B, d]
        gamma = broadcast_time(gamma, graph_node_ptr)              # [N, d]
        beta = broadcast_time(beta, graph_node_ptr)
        H_hat = (1.0 + gamma) * self.node_norm(H_t) + beta

        # ---- Q = receiver, K = sender node + edge state, V = sender ---
        q = self.q_proj(H_hat[dst])                                # [E_msg, d_head]
        k_node = self.k_node_proj(H_hat[src])                      # [E_msg, d_head]
        k_edge = self.k_edge_proj(edge_feat)                       # [E_msg, d_head]
        k = (k_node + k_edge) * self.k_pair_scale                  # [E_msg, d_head]
        v = self.v_proj(H_hat[src])
        score = (q * k).sum(-1) * self.attn_scale                  # [E_msg]

        # ---- Start / Goal 只出不进 -----------------------------------
        valid_msg = ~fixed_mask[dst]

        num_nodes = H_t.shape[0]
        attn = torch.zeros_like(score)
        if valid_msg.any():
            attn[valid_msg] = segment_softmax(
                score[valid_msg], dst[valid_msg], num_nodes
            )

        # ---- message aggregation -------------------------------------
        weighted = attn[:, None] * v                               # [E_msg, d_head]
        message = torch.zeros(
            num_nodes, self.d_head, device=H_t.device, dtype=H_t.dtype
        )
        if dst.numel():
            message.index_add_(0, dst, weighted)

        # ---- residual update（绝不 H_next = m） -----------------------
        H_mid = self.attn_out_norm(H_t + self.o_proj(message))
        H_new = self.ffn_out_norm(H_mid + self.ffn(H_mid))

        # ---- clamp Start / Goal --------------------------------------
        H_next = torch.where(fixed_mask[:, None], H_t, H_new)

        return {
            "H_next": H_next,
            "attn": attn.detach(),
            "gamma": gamma,
            "beta": beta,
        }

    # ------------------------------------------------------------------
    def forward_multi(
        self,
        H_t: Tensor,
        edge_index: Tensor,
        edge_feat: Tensor,
        tau_t: Tensor,
        fixed_mask: Tensor,
        graph_node_ptr: Tensor,
        flow_steps: int = 1,
        slot_embedding: Optional[nn.Embedding] = None,
        slot_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """在一个 reverse step 内部连续做 ``flow_steps`` 轮图信息交流。

        ``H`` 在轮与轮之间**继续累积**（每轮都是 ``H_k = F(H_{k-1})``），所以多轮
        不等于把同一个映射重复作用于同一输入。第 k 轮额外拿到 ``slot_embedding(k)``
        作为条件，用来区分"这是第几轮"。

        返回最后一轮的 ``H_next`` 以及每一轮的 attention（便于诊断）。
        """
        steps = max(1, int(flow_steps))
        H = H_t
        attentions = []
        out: Dict[str, Any] = {}
        for step in range(steps):
            slot_tau = None
            if slot_embedding is not None and steps > 1:
                index = torch.full(
                    (tau_t.shape[0],),
                    min(step, slot_embedding.num_embeddings - 1),
                    dtype=torch.long,
                    device=tau_t.device,
                )
                slot_tau = slot_embedding(index) * float(slot_scale)
            out = self.forward(
                H_t=H,
                edge_index=edge_index,
                edge_feat=edge_feat,
                tau_t=tau_t,
                fixed_mask=fixed_mask,
                graph_node_ptr=graph_node_ptr,
                slot_tau=slot_tau,
            )
            H = out["H_next"]
            attentions.append(out["attn"])
        out["H_next"] = H
        out["attn_per_slot"] = attentions
        out["flow_steps"] = steps
        return out
