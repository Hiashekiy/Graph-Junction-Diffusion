"""Edge Cost Encoder（Weighted V2 扩展，方案第 3、6 节）.

无权 V2 里模型只看得到边的 state（selected / unselected）：拓扑和 OD 相同、只有边权
不同的两个样本，在模型眼里是**完全一样**的输入，带权任务在信息上不可辨识 —— 这正是
当初 ``build_dataset(weighted=True)`` 直接抛 NotImplementedError 的原因（P1-1）。

Weighted 扩展补上这一路输入::

    w_hat_e  ->  MLP  ->  e^cost_e in R^d

输入是 **per-graph mean 归一化**之后的 cost::

    w_hat_e = w_e / mean_{e' in G}(w_{e'})

为什么是 graph mean：

* 目标函数是 edge cost 的**累加** ``C(P) = sum_{e in P} w_e``，纯比例缩放
  ``w -> c * w`` 不改变 ``argmin_P C(P)``，所以最优路径的语义不变；
* 不能做 min-max 或 z-score：带**平移**的归一化会让"多一条边"的路径不再必然更贵，
  不同 hop 数路径之间的排序会被改掉。

第一版是连续编码（不做 bucket embedding）：权重本身就是连续量，离散化只会多引入
一个超参数。
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn


class EdgeCostEncoder(nn.Module):
    """``[E_msg] -> [E_msg, d_model]`` 的连续 edge cost 编码器。"""

    def __init__(self, d_model: int = 128, hidden_dim: Optional[int] = None):
        super().__init__()
        self.d_model = int(d_model)
        hidden = int(hidden_dim or self.d_model)
        self.hidden_dim = hidden
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.d_model),
        )

    def forward(self, cost: Tensor) -> Tensor:
        """``cost``: ``[E]``（或 ``[E, 1]``）-> ``[E, d_model]``。"""
        if cost.dim() == 1:
            cost = cost.unsqueeze(-1)
        elif cost.dim() != 2 or cost.shape[-1] != 1:
            raise ValueError(
                "EdgeCostEncoder expects cost of shape [E] or [E, 1], got "
                f"{tuple(cost.shape)}"
            )
        return self.net(cost)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["EdgeCostEncoder"]
