"""Dataset construction (实施指南第 2、5 节 + 修改清单 P0-2 / P1-1 / P1-2).

生成流程：

    G, s, g  ->  (统一 relabel 到 0..N-1)  ->  Branch Segments  ->  z_0

每个 (G, s, g) 都经过语义校验（分支边属于真实图边、source 非 NULL、goal 不是
decision node、off-path junction 为 NULL）。不满足语义的 OD 对直接丢弃。

修改清单落地：
- **P0-2**：同一张底层图的所有 OD query 共享一个 ``graph_id``，``split_dataset``
  按 graph_id 划分，保证 train/val/test 的图集合两两不相交（无 topology leakage）。
- **P1-1**：``weighted=True`` 直接抛 NotImplementedError —— 模型当前只能看到
  selected/unselected，看不到 edge cost，带权任务在信息上不可辨识。
- **P1-2**：``build_sample`` 一进来就把 graph / start / goal 统一 relabel 到
  0..N-1，之后 graph / gt_path / segments 全部共用同一套编号。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from src.data import branch_segments as bs
from src.data.dataset import GraphQueryDataset, GraphSample, validate_sample
from src.data.decision_field import build_decision_field, validate_decision_field
from src.data.graph_generators import generate_connected_graph


# ---------------------------------------------------------------------------
# node relabel（P1-2）
# ---------------------------------------------------------------------------
def relabel_to_contiguous(
    graph: nx.Graph, start: Any, goal: Any
) -> Tuple[nx.Graph, Any, Any]:
    """把节点统一重编号成 0..N-1，返回 (新图, 新 start, 新 goal)。

    编号顺序是 ``sorted(labels)``，对同一张图是确定性的；已经是 0..N-1 的图直接
    原样返回（不复制）。这样 ``graph`` / ``gt_path`` / ``segments`` 只会存在**一套**
    节点编号空间。
    """
    nodes = list(graph.nodes())
    if nodes == list(range(len(nodes))):
        return graph, start, goal

    mapping = {node: index for index, node in enumerate(sorted(nodes))}
    relabelled = nx.relabel_nodes(graph, mapping, copy=True)
    return relabelled, mapping[start], mapping[goal]


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_od_pair(
    graph: nx.Graph,
    rng: np.random.Generator,
    min_distance: int = 0,
    max_attempts: int = 200,
) -> Tuple[int, int]:
    """抽一个满足 d_G(s, g) >= min_distance 的有序 OD 对。

    实现说明：对每个候选 source 只算一次单源最短路，随机挑一个足够远的 goal。
    """
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("graph needs at least two nodes")

    for _ in range(max_attempts):
        start = int(rng.choice(nodes))
        distances = nx.single_source_shortest_path_length(graph, start)
        reachable = [
            node
            for node, distance in distances.items()
            if node != start and distance >= min_distance
        ]
        if reachable:
            goal = int(reachable[int(rng.integers(0, len(reachable)))])
            return start, goal
    raise RuntimeError(
        f"could not sample an OD pair with distance >= {min_distance}"
    )


# ---------------------------------------------------------------------------
# single sample
# ---------------------------------------------------------------------------
def build_sample(
    graph: nx.Graph,
    start: Any,
    goal: Any,
    meta: Optional[Dict[str, Any]] = None,
    graph_id: Optional[int] = None,
) -> GraphSample:
    """从 (G, s, g) 构造完整样本，并校验数据语义。

    P1-2：先把 graph/start/goal 统一 relabel 成 0..N-1，再交给 ``extract_segments``
    （``relabel=False``），保证不存在第二套节点编号空间。
    """
    graph, start, goal = relabel_to_contiguous(graph, start, goal)
    graph = bs.set_od(graph, start, goal)
    segments = bs.extract_segments(graph, start, goal, relabel=False)

    gt_path = [int(v) for v in nx.shortest_path(graph, int(start), int(goal))]
    field = build_decision_field(segments, gt_path)
    validate_decision_field(segments, field, gt_path)

    sample_meta = dict(meta or {})
    # P0-2：同一张底层图的所有 query 共享 graph_id，供 split_dataset 按图划分
    sample_meta.setdefault("graph_id", -1 if graph_id is None else int(graph_id))

    return GraphSample(
        graph=graph,
        start=int(start),
        goal=int(goal),
        gt_path=gt_path,
        segments=segments,
        field=field,
        meta=sample_meta,
    )


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def build_dataset(
    num_samples: int,
    graph_type: str = "er",
    num_nodes: int | Sequence[int] = (20, 40),
    min_od_distance: int = 5,
    seed: int = 0,
    queries_per_graph: int = 1,
    generator_cfg: Optional[Dict[str, Any]] = None,
    weighted: bool = False,
    component_fallback: bool = True,
    max_rejects: int = 5000,
) -> GraphQueryDataset:
    """生成 ``num_samples`` 个 (G, s, g) 样本。

    ``queries_per_graph`` > 1 时同一张图上抽多个 OD 对；这些样本共享同一个
    ``graph_id``，所以按 graph 划分时它们会一起进同一个 split。
    """
    if weighted:
        # P1-1：模型边输入只有 selected/unselected，看不到 edge cost。
        # 拓扑/OD/edge-state 相同但 cost 不同的两个样本，模型输入完全一样而最优路径
        # 可能不同 —— 信息上不可辨识，所以第一版直接禁用。
        raise NotImplementedError(
            "V2 baseline does not encode edge cost yet: "
            "weighted=True would make the task unidentifiable from the model's "
            "inputs. Keep weighted=false until an Edge Cost Encoder is implemented."
        )

    rng = np.random.default_rng(seed)
    if isinstance(num_nodes, (int, np.integer)):
        node_lo = node_hi = int(num_nodes)
    else:
        node_lo, node_hi = int(num_nodes[0]), int(num_nodes[1])

    samples: List[GraphSample] = []
    rejects = 0
    graph_id = 0
    while len(samples) < num_samples:
        num = int(rng.integers(node_lo, node_hi + 1))
        graph, params = generate_connected_graph(
            graph_type,
            num,
            rng,
            generator_cfg=generator_cfg,
            min_od_distance=min_od_distance,
            component_fallback=component_fallback,
            weighted=False,
        )
        accepted = 0
        for _ in range(queries_per_graph * 4):
            if accepted >= queries_per_graph or len(samples) >= num_samples:
                break
            try:
                start, goal = sample_od_pair(graph, rng, min_od_distance)
                sample = build_sample(
                    graph.copy(),
                    start,
                    goal,
                    meta={
                        "graph_type": graph_type,
                        "params": params,
                        "num_nodes": graph.number_of_nodes(),
                    },
                    graph_id=graph_id,
                )
                validate_sample(sample)
            except (ValueError, RuntimeError, AssertionError, nx.NetworkXError):
                rejects += 1
                if rejects > max_rejects:
                    raise RuntimeError(
                        f"rejected {rejects} OD pairs while generating {num_samples} "
                        "samples; the data semantics never validated"
                    )
                continue
            samples.append(sample)
            accepted += 1
        graph_id += 1

    return GraphQueryDataset(samples, name=f"{graph_type}_{num_samples}")


def _allocate_group_shares(
    total: int, names: Sequence[str], fractions: Dict[str, float]
) -> Tuple[Dict[str, int], List[str]]:
    """按比例把 ``total`` 个 graph 分给各 split，并保证尽量每个 split 非空。

    纯按比例取整会让小数据集出事：4 张图、0.8/0.1/0.1 时 val/test 都会取整成 0，
    于是 val 集为空、validate() 什么都不返回、early-stop / best-checkpoint 直接
    失效（这是真实踩过的坑）。这里在 "有足够的图" 时强制给每个 split 至少一个，
    不够时按参数顺序优先前面的 split，并返回被挤掉的名字供调用方打印警告。
    """
    shares = {name: int(round(float(fractions[name]) * total)) for name in names}

    # 把多余/缺失的名额调整到总和 == total
    while sum(shares.values()) > total:
        largest = max(names, key=lambda name: shares[name])
        shares[largest] -= 1
    while sum(shares.values()) < total:
        shares[names[0]] += 1

    squeezed: List[str] = []
    if total >= len(names):
        changed = True
        while changed:
            changed = False
            empty = [name for name in names if shares[name] == 0]
            if not empty:
                break
            for name in empty:
                donors = [n for n in names if shares[n] > 1]
                if not donors:
                    squeezed.extend(empty)
                    return shares, squeezed
                donor = max(donors, key=lambda n: shares[n])
                shares[donor] -= 1
                shares[name] += 1
                changed = True
    else:
        # 图比 split 还少：只保证前 total 个 split 有图
        squeezed = list(names[total:])
        shares = {name: (1 if index < total else 0) for index, name in enumerate(names)}
    return shares, squeezed


def split_dataset(
    dataset: GraphQueryDataset,
    fractions: Dict[str, float],
    seed: int = 0,
) -> Dict[str, GraphQueryDataset]:
    """按 **graph_id** 划分 train / val / test（修改清单 P0-2）。

    先按 graph_id 归组，再打乱 graph 顺序并切分，最后把每组的所有 query 收集到
    对应的 split。这样同一张图不会同时出现在两个 split 里，避免 topology leakage。

    另外保证：只要图的数量够，每个 split 至少分到一张图（否则 val 为空会让
    ``validate()`` 静默失效）。

    没有 graph_id 的样本（``meta['graph_id'] == -1``）按"一图一样本"处理，退化成
    按样本划分 —— 安全，但会牺牲一点统计效率。
    """
    groups: Dict[Any, List[GraphSample]] = {}
    for index, sample in enumerate(dataset):
        key = sample.graph_id
        if key is None or key == -1:
            key = f"sample-{index}"
        groups.setdefault(key, []).append(sample)

    keys = list(groups)
    order = np.random.default_rng(seed).permutation(len(keys)).tolist()
    shuffled = [keys[i] for i in order]

    names = list(fractions)
    shares, squeezed = _allocate_group_shares(len(shuffled), names, fractions)
    if squeezed:
        print(
            f"[split_dataset] warning: only {len(shuffled)} graph(s) for "
            f"{len(names)} splits; {squeezed} will be empty",
            flush=True,
        )

    splits: Dict[str, GraphQueryDataset] = {}
    cursor = 0
    for name in names:
        count = shares[name]
        collected: List[GraphSample] = []
        for key in shuffled[cursor : cursor + count]:
            collected.extend(groups[key])
        cursor += count
        splits[name] = GraphQueryDataset(collected, name=f"{dataset.name}_{name}")
    return splits


def graph_ids_of(dataset: GraphQueryDataset) -> set:
    """该 split 覆盖的 graph_id 集合（用于验收 topology leakage）。"""
    return {
        sample.graph_id
        for index, sample in enumerate(dataset)
        if sample.graph_id not in (None, -1)
    }


def tiny_overfit_dataset(
    num_samples: int = 16,
    num_nodes: int = 24,
    seed: int = 0,
    graph_type: str = "er",
    min_od_distance: int = 4,
) -> GraphQueryDataset:
    """实施指南第 27 节：固定 seed 的小数据集，用来先验证整条链能否过拟合。"""
    return build_dataset(
        num_samples=num_samples,
        graph_type=graph_type,
        num_nodes=num_nodes,
        min_od_distance=min_od_distance,
        seed=seed,
        queries_per_graph=2,
    )
