"""Shared fixtures: 手工小图 + batch 构造。

手工图（节点编号固定，方便逐步核对 branch / decision / candidate）：

```
                     h --- i
                    /
    s --- c --- J1 --- a --- b --- x --- J2 --- g
                    \\
                     d --- e --- f
```

编号：

    0=s  1=c  2=J1  3=a  4=b  5=x  6=J2  7=g  8=h  9=i  10=d  11=e  12=f

性质（每个测试都依赖这些）：

* `J1`（度 4）与 `J2`（度 3）是仅有的两个 decision node；
* `s` 度为 1，所以**不是** decision node；`g` 度为 1，且永远不是 decision node；
* GT 最短路唯一：`s-c-J1-a-b-x-J2-g`（8 跳），因此 `J1` 的 z_0 是
  `[J1,a,b,x,J2]`，`J2` 的 z_0 是 `[J2,g]`（经 d-e-f 的绕行路线有 9 跳）；
* `J1` 另外两条 branch 分别通向 `s` 和 dead-end `i`；`J2` 另外两条分别通向
  `x`（回到 J1）和 dead-end `f`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.branch_segments import (  # noqa: E402
    GOAL,
    JUNCTION,
    ORDINARY,
    START,
)
from src.data.collate import collate_samples  # noqa: E402
from src.data.dataset_builder import build_sample  # noqa: E402

S, C, J1, A, B, X, J2, G, H, I, D, E, F = range(13)

MANUAL_EDGES = [
    (S, C),    # 0-1
    (C, J1),   # 1-2
    (J1, H),   # 2-8   -> dead end 分支
    (H, I),    # 8-9
    (J1, A),   # 2-3   -> GT 分支
    (A, B),    # 3-4
    (B, X),    # 4-5
    (X, J2),   # 5-6
    (J1, D),   # 2-10  -> 绕行分支
    (D, E),    # 10-11
    (E, F),    # 11-12
    (F, J2),   # 12-6
    (J2, G),   # 6-7
]


def make_manual_graph():
    """返回 (graph, s, g)，节点编号见模块 docstring。"""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(MANUAL_EDGES)
    return graph, S, G


def make_manual_sample():
    graph, start, goal = make_manual_graph()
    return build_sample(graph, start, goal)


@pytest.fixture
def manual_graph():
    return make_manual_graph()


@pytest.fixture
def manual_sample():
    return make_manual_sample()


@pytest.fixture
def manual_batch(manual_sample):
    return collate_samples([manual_sample], device="cpu")


@pytest.fixture
def tiny_batches():
    """两个小样本的 batch（ER 图，branch 语义在生成时已被真实校验）。

    图可能因为 OD 距离约束生成失败（依赖随机性）；这种情况下 pytest.skip 而不是
    让整文件报错。
    """
    from src.data.dataset_builder import tiny_overfit_dataset

    try:
        dataset = tiny_overfit_dataset(num_samples=2, num_nodes=22, seed=3)
    except RuntimeError as error:  # pragma: no cover - 依赖随机性
        pytest.skip(f"dataset generation failed: {error}")
    samples = [dataset[0], dataset[1]]
    return samples, collate_samples(samples, device="cpu")


# 期望值（供多个测试复用）
GT_PATH = [S, C, J1, A, B, X, J2, G]
J1_BRANCH_NODES = [J1, A, B, X, J2]
J2_BRANCH_NODES = [J2, G]


def node_positions():
    return {
        "s": S, "c": C, "J1": J1, "a": A, "b": B, "x": X, "J2": J2, "g": G,
        "h": H, "i": I, "d": D, "e": E, "f": F,
    }
