"""Graph Flow Block (实施指南第 10-11 节 / 设计报告 V2.1 第 5-7、11 节).

一个 reverse step 只跑**一次** Graph Flow；这个 Cell 在所有 timestep 上共享参数：

    H_{t-1} = F_theta(H_t, E_t, tau_t)

注意力角色分工（这是 V2 的核心）：

    Q_v      = W_Q * h_hat_v        接收节点     "我怎样接收信息"
    K_{uv}   = W_K * e_uv^t         边状态       "这条通道当前是什么状态"
    V_u      = W_V * h_hat_u        发送节点     "传播什么信息"

    score    = <q_v, k_uv> / sqrt(d)
    alpha    = softmax_{u in N(v)} score
    m_v      = sum_u alpha_uv * V_u

三条硬约束：

1. **Start / Goal 只出不进**：所有 dst 是 start/goal 的 message edge 在 softmax
   和聚合里都被剔除（指南第 10.4 节，不依赖网络自己学）。
2. **必须有 residual**：``H_next = LN(H + W_O m)`` 再 ``LN(H~ + FFN(H~))``，
   而不是 ``H_next = m``（指南第 11 节）。
3. **Start / Goal 的状态被 clamp 回输入**：``H_next[fixed] = H_t[fixed]``，
   用 ``torch.where`` 写（对 autograd 更友好）。
"""

from __future__ import annotations

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
        self.k_proj = nn.Linear(self.d_model, self.d_head)
        self.v_proj = nn.Linear(self.d_model, self.d_head)
        self.o_proj = nn.Linear(self.d_head, self.d_model)

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
        for projection in (self.q_proj, self.k_proj, self.v_proj):
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
    ) -> Dict[str, Any]:
        if edge_index.numel():
            src, dst = edge_index[0], edge_index[1]
        else:
            src = dst = torch.zeros(0, dtype=torch.long, device=H_t.device)

        # ---- timestep conditioning (AdaLN / FiLM) ---------------------
        gamma, beta = self.time_conditioner(tau_t)                 # [B, d]
        gamma = broadcast_time(gamma, graph_node_ptr)              # [N, d]
        beta = broadcast_time(beta, graph_node_ptr)
        H_hat = (1.0 + gamma) * self.node_norm(H_t) + beta

        # ---- Q = receiver, K = edge state, V = sender ----------------
        q = self.q_proj(H_hat[dst])                                # [E_msg, d_head]
        k = self.k_proj(edge_feat)
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
