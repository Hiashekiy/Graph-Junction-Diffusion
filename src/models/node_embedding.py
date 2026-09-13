"""Node encoder：实际上只是 Node-Type Embedding (实施指南第 7 节).

    H_T = E_node[type(v)],      E_node in R^{4 x d}

四种类型共享四个可学习向量；网络在 t = T 时完全不知道节点在图中的位置、
到 Start/Goal 的距离、或者节点 ID。节点差异只能由后续的

    图拓扑 + Start/Goal 信息传播 + Edge State + 历史状态

逐渐形成（设计报告 V2.1 第 3.3 节）。

初始化**只执行一次**：``H_t = node_encoder(batch.node_type)``，之后整条 reverse
chain 不再重新初始化。
"""

from __future__ import annotations

from torch import Tensor, nn

from src.data.branch_segments import NUM_NODE_TYPES


class NodeTypeEmbedding(nn.Module):
    def __init__(self, d_model: int = 128, num_node_types: int = NUM_NODE_TYPES):
        super().__init__()
        self.d_model = int(d_model)
        self.num_node_types = int(num_node_types)
        self.embedding = nn.Embedding(self.num_node_types, self.d_model)

    def forward(self, node_type: Tensor) -> Tensor:
        """node_type: [N] long -> [N, d_model]."""
        return self.embedding(node_type)


# 实施指南里的别名，保持两套命名都能用
NodeEncoder = NodeTypeEmbedding
