"""Branch Segment decomposition (实施指南第 2-4 节 / 设计报告 V2.1 第 2 节).

V2 的候选不再是"下一个邻居"，而是从 decision node 出发、穿过普通节点、直到
下一个 structural endpoint 的**整条 segment**：

    J1 ─ a ─ b ─ J2      ->   branch nodes = [J1, a, b, J2]
                              branch physical edges = [(J1,a), (a,b), (b,J2)]

节点只有四类：

    0 = Ordinary   1 = Junction   2 = Start   3 = Goal

其中 Start / Goal 的优先级高于 degree：一个度为 4 的 start 仍然是 START。

Branch endpoint 集合为

    A = {v : deg(v) != 2} ∪ {s, g}

（degree >= 3 是 Junction，degree == 1 是 dead-end，只要不是 2 就停止追溯；
s / g 无条件作为停止点。）

物理边与 message edge 严格分开：

    physical edge : 无向边 {u, v}，唯一 ID，branch 只记录 physical edge ID
    message edge  : 有向边 u -> v 与 v -> u，用 msg_to_phys_edge 映射回物理边

于是 E_msg = 2 * E_phys，且两个方向永远共享同一个 edge-state。

**编号约定**：本模块内部一律使用图的**原始节点标签**（不强制是整数）。
``collate`` 需要连续整数编号，所以 ``extract_segments`` 默认会把节点重编号为
``0..N-1``（顺序是 ``sorted(labels)``，确定性）；也可以传 ``relabel=False``
自行保证编号连续。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# node types
# ---------------------------------------------------------------------------
ORDINARY = 0
JUNCTION = 1
START = 2
GOAL = 3

NODE_TYPE_NAMES = {
    ORDINARY: "ordinary",
    JUNCTION: "junction",
    START: "start",
    GOAL: "goal",
}
NUM_NODE_TYPES = 4

# edge states
UNSELECTED = 0
SELECTED = 1
NUM_EDGE_STATES = 2

Node = Hashable
# (u, v) 有序或无向键都可以；物理边 ID 就是无向边的编号
EdgeKey = Tuple[Node, Node]


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass
class Branch:
    """一条 branch candidate（有 owner 的决策类别）。

    Attributes:
        owner:          发起这条 branch 的 decision node
        nodes:          [owner, ..., endpoint]，至少两个节点
        physical_edges: 这条 branch 覆盖的无向物理边 ID
        end:            nodes[-1]，下一个 structural endpoint
    """

    owner: Node
    nodes: List[Node]
    physical_edges: List[int]
    end: Node


@dataclass
class GraphSegments:
    """一张图的所有结构信息（尚未 batch）。"""

    num_nodes: int
    start: Node
    goal: Node
    node_type: Dict[Node, int]
    edge_index: List[Tuple[Node, Node]]        # message edges (2 * E_phys)
    msg_to_phys_edge: List[int]                # [E_msg]
    num_physical_edges: int
    decision_nodes: List[Node]                 # [M]
    branches: List[List[Branch]]               # [M] 每个 decision 的候选 branch
    endpoints: List[Node] = field(default_factory=list)

    # -- convenience -----------------------------------------------------
    @property
    def num_decisions(self) -> int:
        return len(self.decision_nodes)

    def branches_flat(self) -> List[Branch]:
        return [b for group in self.branches for b in group]

    def branch_endpoints(self) -> List[Node]:
        return [b.end for b in self.branches_flat()]


@dataclass
class FlatCandidates:
    """扁平化的 candidate 表（全局 candidate index 空间）。

    普通 Junction 的候选组是 [NULL, B1..BK]；Source 的候选组是 [B1..BK]，没有 NULL。
    """

    candidate_owner: List[int]      # [C] candidate -> decision index
    candidate_is_null: List[bool]   # [C]
    candidate_branch: List[Optional[Branch]]  # [C] NULL 位置为 None
    target_candidate: List[int]     # [M] z_0 的 flat candidate index
    num_candidates: int


# ---------------------------------------------------------------------------
# node / endpoint classification
# ---------------------------------------------------------------------------
def node_types(graph: nx.Graph, start: Node, goal: Node) -> Dict[Node, int]:
    """四类节点 ID，Start / Goal 优先级高于 degree。

    返回 **node -> type** 的字典，而不是按遍历顺序排列的 list：
    NetworkX 的 ``graph.nodes()`` 顺序是插入顺序，和节点编号无关，用 list 会让
    调用方拿错节点的类型。
    """
    types: Dict[Node, int] = {}
    for node in graph.nodes():
        degree = graph.degree(node)
        if node == start:
            types[node] = START
        elif node == goal:
            types[node] = GOAL
        elif degree >= 3:
            types[node] = JUNCTION
        else:
            types[node] = ORDINARY
    return types


def build_decision_nodes(graph: nx.Graph, start: Node, goal: Node) -> List[Node]:
    """D(G, s, g) = (J(G) ∪ {s | deg(s) > 1}) \\ {g}."""
    decisions = {v for v in graph.nodes() if graph.degree(v) >= 3}
    if graph.degree(start) > 1:
        decisions.add(start)
    decisions.discard(goal)
    return sorted(decisions)


def build_endpoints(graph: nx.Graph, start: Node, goal: Node) -> List[Node]:
    """A = {v : deg(v) != 2} ∪ {s, g}."""
    endpoints = {v for v in graph.nodes() if graph.degree(v) != 2}
    endpoints.add(start)
    endpoints.add(goal)
    return sorted(endpoints)


# ---------------------------------------------------------------------------
# edge tables
# ---------------------------------------------------------------------------
def build_edge_tables(graph: nx.Graph) -> Tuple[List[Tuple[Node, Node]], List[int], int]:
    """物理边 <-> message edge 的映射（两个方向共享同一个物理边 ID）。"""
    edge_index: List[Tuple[Node, Node]] = []
    msg_to_phys: List[int] = []
    for phys_id, (u, v) in enumerate(graph.edges()):
        edge_index.append((u, v))
        msg_to_phys.append(phys_id)
        edge_index.append((v, u))
        msg_to_phys.append(phys_id)
    return edge_index, msg_to_phys, len(msg_to_phys) // 2


def physical_edge_lookup(graph: nx.Graph) -> Dict[EdgeKey, int]:
    lookup: Dict[EdgeKey, int] = {}
    for phys_id, (u, v) in enumerate(graph.edges()):
        lookup[(u, v)] = phys_id
        lookup[(v, u)] = phys_id
    return lookup


# ---------------------------------------------------------------------------
# branch tracing
# ---------------------------------------------------------------------------
def trace_branch(
    graph: nx.Graph,
    owner: Node,
    first_hop: Node,
    endpoints: Iterable[Node],
    edge_of: Dict[EdgeKey, int],
) -> Branch:
    """从一个 decision node 沿某个邻居一直走到下一个 endpoint。"""
    endpoint_set = endpoints if isinstance(endpoints, (set, frozenset)) else set(endpoints)
    path = [owner, first_hop]
    edges = [edge_of[(owner, first_hop)]]

    prev, cur = owner, first_hop
    while cur not in endpoint_set:
        neighbours = list(graph.neighbors(cur))
        if len(neighbours) != 2:
            raise ValueError(
                f"node {cur!r} has degree {len(neighbours)} but is not an endpoint; "
                "the endpoint set and the graph disagree"
            )
        nxt = neighbours[0] if neighbours[1] == prev else neighbours[1]
        edges.append(edge_of[(cur, nxt)])
        path.append(nxt)
        prev, cur = cur, nxt

    return Branch(owner=owner, nodes=path, physical_edges=edges, end=path[-1])


def build_branches(
    graph: nx.Graph,
    decision_nodes: Sequence[Node],
    start: Optional[Node] = None,
    goal: Optional[Node] = None,
    endpoints: Optional[Iterable[Node]] = None,
) -> List[List[Branch]]:
    """每个 decision node 的完整候选 branch 列表（按邻居排序，保证可复现）。

    endpoint 集合必须与调用方使用的 OD 对一致：显式传 ``start``/``goal``
    （或直接传 ``endpoints``）时不会再去读 ``graph.graph`` 里的旧属性。
    """
    if endpoints is None:
        endpoints = build_endpoints(
            graph,
            _graph_start(graph) if start is None else start,
            _graph_goal(graph) if goal is None else goal,
        )
    endpoint_set = set(endpoints)
    edge_of = physical_edge_lookup(graph)

    groups: List[List[Branch]] = []
    for owner in decision_nodes:
        group = [
            trace_branch(graph, owner, neighbour, endpoint_set, edge_of)
            for neighbour in sorted(graph.neighbors(owner))
        ]
        groups.append(group)
    return groups


# start / goal 存成图属性，避免每个函数都传一遍
def set_od(graph: nx.Graph, start: Node, goal: Node) -> nx.Graph:
    graph.graph["start"] = start
    graph.graph["goal"] = goal
    return graph


def _graph_start(graph: nx.Graph) -> Node:
    if "start" not in graph.graph:
        raise KeyError("graph has no 'start' attribute; call set_od(graph, s, g) first")
    return graph.graph["start"]


def _graph_goal(graph: nx.Graph) -> Node:
    if "goal" not in graph.graph:
        raise KeyError("graph has no 'goal' attribute; call set_od(graph, s, g) first")
    return graph.graph["goal"]


# ---------------------------------------------------------------------------
def extract_segments(
    graph: nx.Graph,
    start: Optional[Node] = None,
    goal: Optional[Node] = None,
    relabel: bool = True,
) -> GraphSegments:
    """从一张 NetworkX 图 + OD 对得到完整结构信息。

    Args:
        graph:   无向图；节点标签可以是任意可排序对象
        start:   source 节点（未给出时读 ``graph.graph['start']``）
        goal:    goal 节点（未给出时读 ``graph.graph['goal']``）
        relabel: True 时把节点重编号为 0..N-1（collate 需要连续整数编号）。
                 编号顺序为 ``sorted(labels)``，对同一张图是确定性的。

    ``GraphSegments`` 里所有编号都是**重编号后**的局部编号。
    """
    if start is None:
        start = _graph_start(graph)
    if goal is None:
        goal = _graph_goal(graph)

    if start not in graph or goal not in graph:
        raise ValueError(f"start={start} / goal={goal} must both be nodes of the graph")
    if start == goal:
        raise ValueError("start and goal must differ")

    if relabel:
        nodes = list(graph.nodes())
        if nodes != list(range(len(nodes))):
            mapping = {node: index for index, node in enumerate(sorted(nodes))}
            graph = nx.relabel_nodes(graph, mapping, copy=True)
            start, goal = mapping[start], mapping[goal]

    decisions = build_decision_nodes(graph, start, goal)
    edge_index, msg_to_phys, num_phys = build_edge_tables(graph)
    endpoints = build_endpoints(graph, start, goal)

    return GraphSegments(
        num_nodes=graph.number_of_nodes(),
        start=start,
        goal=goal,
        node_type=node_types(graph, start, goal),
        edge_index=edge_index,
        msg_to_phys_edge=msg_to_phys,
        num_physical_edges=num_phys,
        decision_nodes=decisions,
        branches=build_branches(graph, decisions, start, goal, endpoints),
        endpoints=endpoints,
    )


def branch_lookup(segments: GraphSegments) -> Dict[Tuple[Node, Node], Branch]:
    """(owner, end) -> Branch，用于从 GT path 反查正确候选。

    注意：同一对 (owner, end) 可能对应多条 branch（例如两条不同的路径走到同一个
    junction），此时只有最后一条会被保留；需要精确匹配请用 ``Branch.nodes``。
    """
    lookup: Dict[Tuple[Node, Node], Branch] = {}
    for group in segments.branches:
        for branch in group:
            lookup[(branch.owner, branch.end)] = branch
    return lookup


def branch_covers_path(branch: Branch, path: Sequence[Node]) -> bool:
    """branch.nodes 是否是 ``path`` 的一个前缀子序列（从 path[0] 开始）。"""
    if len(branch.nodes) > len(path):
        return False
    return all(a == b for a, b in zip(branch.nodes, path))


def describe_segments(segments: GraphSegments) -> Dict[str, Any]:
    """给 --inspect 用的可读摘要。"""
    return {
        "num_nodes": segments.num_nodes,
        "num_physical_edges": segments.num_physical_edges,
        "num_message_edges": len(segments.edge_index),
        "start": segments.start,
        "goal": segments.goal,
        "num_decisions": segments.num_decisions,
        "num_candidates": sum(len(g) for g in segments.branches) + _num_nulls(segments),
        "branches": [
            {
                "owner": branch.owner,
                "nodes": list(branch.nodes),
                "edges": list(branch.physical_edges),
                "end": branch.end,
            }
            for branch in segments.branches_flat()
        ],
    }


def _num_nulls(segments: GraphSegments) -> int:
    """普通 Junction（非 source）各有一个 NULL candidate。"""
    return sum(1 for node in segments.decision_nodes if node != segments.start)
