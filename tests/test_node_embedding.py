"""Node encoding 测试（实施指南第 7、25.2 节）.

    H_T = E_node[type(v)]

初始时所有同类型节点严格使用同一个向量；网络完全不知道节点 ID，也不知道节点
在图中的位置。
"""

from __future__ import annotations

import torch

from src.data.branch_segments import GOAL, JUNCTION, ORDINARY, START
from src.models.node_embedding import NodeTypeEmbedding


def test_two_ordinary_nodes_share_the_same_initial_embedding():
    encoder = NodeTypeEmbedding(d_model=16)
    node_type = torch.tensor([ORDINARY, ORDINARY, JUNCTION, START, GOAL])
    H = encoder(node_type)
    assert torch.allclose(H[0], H[1])
    assert H.shape == (5, 16)


def test_same_type_nodes_share_the_same_vector():
    encoder = NodeTypeEmbedding(d_model=8)
    H = encoder(torch.tensor([JUNCTION, ORDINARY, JUNCTION]))
    assert torch.allclose(H[0], H[2])


def test_start_and_goal_differ_from_ordinary_and_junction():
    encoder = NodeTypeEmbedding(d_model=8)
    H = encoder(torch.tensor([ORDINARY, JUNCTION, START, GOAL]))
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(H[i], H[j], atol=1e-6)


def test_embedding_table_has_four_entries_and_is_learnable():
    encoder = NodeTypeEmbedding(d_model=8)
    assert encoder.embedding.num_embeddings == 4
    assert encoder.embedding.weight.requires_grad
    assert encoder.embedding.weight.shape == (4, 8)


def test_node_encoder_ignores_graph_position_information(manual_batch):
    """同一张图里两个普通节点初始化必须完全相同（没有 LapPE / degree / node id）。"""
    encoder = NodeTypeEmbedding(d_model=12)
    H = encoder(manual_batch.node_type)
    ordinary = (manual_batch.node_type == ORDINARY).nonzero().flatten()
    assert ordinary.numel() >= 2
    reference = H[ordinary[0]]
    for node in ordinary[1:]:
        assert torch.allclose(H[node], reference)
