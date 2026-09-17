"""Branch Scorer + NULL Scorer + Grouped Softmax (实施指南第 13-14 节).

Branch Mean Pool：平均的是**整条 branch 上除 owner 以外**的节点

    h_bar_ik = 1/|B_ik \\ {i}| * sum_{v in B_ik \\ {i}} h_v

dataset 里的 ``branch_node_ids`` 已经不包含 owner，所以分母就是 membership 长度。

Branch Scorer 输入：

    c_ik = [h_i^{t-1}, h_bar_ik^{t-1}, tau_t]   in R^{3d}

NULL Scorer 输入：

    c_iNULL = [h_i^{t-1}, tau_t]                in R^{2d}

最后所有 logits 填回 flat ``candidate_logits [C]``，用 grouped softmax 在每个
Junction 自己的 ragged candidate set 内归一化：

    sum_{c in C_i} p_i(c) = 1
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from src.utils.segment_ops import grouped_log_softmax, segment_sum


def branch_mean_pool(
    H: Tensor,                    # [N, d]
    branch_node_ids: Tensor,      # [C, L] padded（不含 owner）
    branch_node_lengths: Tensor,  # [C]
    num_candidates: int,
) -> Tensor:
    """Branch 的 mean-pool 表示，形状 [C, d]；NULL / 空 branch 保持 0。"""
    if branch_node_ids.numel() == 0:
        return H.new_zeros(num_candidates, H.shape[1])

    device = H.device
    dtype = H.dtype
    width = branch_node_ids.shape[1]
    mask = (
        torch.arange(width, device=device)[None, :] < branch_node_lengths[:, None]
    )

    node_feat = H[branch_node_ids.reshape(-1)].reshape(num_candidates, width, -1)
    node_feat = node_feat * mask.unsqueeze(-1).to(node_feat.dtype)

    owner_index = torch.arange(num_candidates, device=device).repeat_interleave(width)
    totals = segment_sum(
        node_feat.reshape(-1, H.shape[1]), owner_index, num_candidates
    )
    counts = mask.sum(dim=1).to(H.dtype)
    # segment_sum 内部按 float32 计算，这里还原成输入 dtype（AMP 下是 fp16）
    return (totals / counts.clamp_min(1.0)[:, None]).to(dtype)


class BranchScorer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        readout: str = "mean_pool",
    ):
        super().__init__()
        self.d_model = int(d_model)
        hidden = int(hidden_dim or 2 * self.d_model)

        if readout not in ("mean_pool", "first_node"):
            raise ValueError(
                f"model.branch_readout={readout!r} is not supported "
                "(mean_pool | first_node)"
            )
        self.readout = readout

        # Linear(3d, 2d) -> SiLU -> Linear(2d, 1)
        self.branch_mlp = nn.Sequential(
            nn.Linear(3 * self.d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        # Linear(2d, d) -> SiLU -> Linear(d, 1)
        self.null_mlp = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )

    # ------------------------------------------------------------------
    def branch_representation(self, H: Tensor, batch, is_branch: Tensor) -> Tensor:
        """非 NULL candidate 的 branch 表示 [C_nonnull, d]。

        ``mean_pool``（默认）：整条 branch 上除 owner 外所有节点取平均 ——
        这是 Full model 的行为，逐位不变。

        ``first_node``（消融）：只取 owner 之后的**第一个节点**。
        候选集合、z_t 语义、edge state 展开、decoder 全部不变，**只改这个
        表示怎么算**；拼接后仍是 3d，所以 Branch MLP 的结构与参数量完全不变
        （消融不引入容量差异）。
        """
        if self.readout == "mean_pool":
            return branch_mean_pool(
                H,
                batch.branch_node_ids,
                batch.branch_node_lengths,
                batch.num_candidates,
            )[is_branch]

        first_ids = batch.branch_node_ids[is_branch][:, 0]
        lengths = batch.branch_node_lengths[is_branch]
        if lengths.numel() and int(lengths.min()) < 1:
            raise ValueError(
                "branch_readout=first_node requires every non-NULL branch to have "
                f"at least one node, got min length {int(lengths.min())}"
            )
        return H[first_ids]

    # ------------------------------------------------------------------
    def branch_logits(
        self, H: Tensor, batch, tau_decisions: Tensor
    ) -> Tensor:
        """所有非 NULL candidate 的 logits，形状 [C_nonnull]。"""
        is_branch = ~batch.candidate_is_null
        if not bool(is_branch.any()):
            return H.new_zeros(0)

        owners = batch.candidate_owner[is_branch]
        pooled = self.branch_representation(H, batch, is_branch)
        junction = H[batch.decision_node[owners]]
        features = torch.cat([junction, pooled, tau_decisions[owners]], dim=-1)
        return self.branch_mlp(features).squeeze(-1)

    def null_logits(self, H: Tensor, batch, tau_decisions: Tensor) -> Tensor:
        """所有 NULL candidate 的 logits，形状 [C_null]。"""
        is_null = batch.candidate_is_null
        if not bool(is_null.any()):
            return H.new_zeros(0)
        owners = batch.candidate_owner[is_null]
        junction = H[batch.decision_node[owners]]
        features = torch.cat([junction, tau_decisions[owners]], dim=-1)
        return self.null_mlp(features).squeeze(-1)

    # ------------------------------------------------------------------
    def forward(
        self, H: Tensor, batch, tau_decisions: Tensor
    ) -> Dict[str, Tensor]:
        is_null = batch.candidate_is_null
        branch = self.branch_logits(H, batch, tau_decisions)
        null = self.null_logits(H, batch, tau_decisions)

        # masked_scatter 要求 self 与 source 的 dtype 完全一致。AMP 下 H 是 fp16
        # 而 Linear 输出可能是 fp32，所以统一取两支 logits 的 dtype。
        dtype = branch.dtype if branch.numel() else null.dtype
        logits = H.new_zeros(batch.num_candidates, dtype=dtype)
        logits = logits.masked_scatter(~is_null, branch.to(dtype))
        logits = logits.masked_scatter(is_null, null.to(dtype))

        log_prob = grouped_log_softmax(logits, batch.candidate_owner, batch.num_decisions)
        return {
            "candidate_logits": logits,
            "candidate_log_prob": log_prob,
            "candidate_prob": log_prob.exp(),
        }
