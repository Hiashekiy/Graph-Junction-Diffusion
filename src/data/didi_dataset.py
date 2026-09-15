"""DiDi 成都真实道路数据适配层（实施方案第 3、4、6、7、17-A 节）。

这一层是**整个真实数据接入的唯一入口**，负责把 DiDi 的原始文件翻译成
Graph-Junction-Diffusion 需要的三元组：

    (固定真实路网 G + 道路长度 W, OD, 真实车辆历史 junction path)

它**不**负责构造 ``GraphSample`` —— 那是
:func:`src.data.dataset_builder.build_sample_from_observed_path` 的工作。
把 DiDi 的特殊逻辑（road id、平行路段折叠、corridor）塞进通用 dataset_builder
会让合成数据那条链也一起被污染，所以这里严格分层。

数据文件语义（实施方案 2.1）：

    dicts.pkl              road_id -> (u, v, key)    重建 junction graph
    edge_features.csv      道路静态属性，第一版只取 length 作为 edge weight
    20161010~19.csv        真实车辆轨迹，核心字段 path = road id 序列
    line_graph_edge_idx.npy  只用于校验，**不作为主模型图**

几个关键结论来自对真实文件的实测（写进代码注释，避免后人再猜）：

* ``edge_features.csv`` 的列名是
  ``road_id, oneway, lanes, highway, length, bridge, tunnel, highway_id,
  length_id, road_speed, traj_speed``，道路长度就是 ``length``（米）。
* 成都路网折叠成无向 junction graph 后是 **2891 个节点 / 4403 条边**，单连通分量。
  6639 条 road segment 里 10 条是自环（丢弃）、2226 条是平行路段（按方案第 3.3 节
  折叠成最短的那条），最后得到 4403 个唯一无向 ``(u, v)``。
* 轨迹里的 ``path`` 是**道路 id 序列**，且相邻 road 在 ``idx2edge`` 的**存储方向**
  上是连续的（``u_{i+1} == v_i``）：实测 30000 条轨迹 100% 满足，没有一条需要反向。
  约 1.6% 的相邻对是"平行路段掉头"（``{u,v}`` 相同、方向相反），会形成重复
  junction，由 ``require_simple_gt`` 过滤。

因此 road → junction 转换的**主算法**用存储方向（精确、能正确处理掉头），
**回退算法**用方案第 4.1 节的端点集合交集写法（不依赖方向，用于兜底）。
"""

from __future__ import annotations

import ast
import csv
import math
import pickle
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# 0. 常量 / 异常
# ---------------------------------------------------------------------------
#: ``dicts.pkl`` 里 ``idx2edge`` 的标签
IDX2EDGE = "idx2edge"
EDGE2IDX = "edge2idx"

#: edge_features.csv 里长度列的候选名（只用于**报错提示**，绝不自动选取）
LENGTH_COLUMN_CANDIDATES = ("length", "road_length", "len", "edge_length", "distance")

#: junction path 转换的两条路径
CONVERSION_STORED_DIRECTION = "stored_direction"
CONVERSION_ENDPOINT_WALK = "endpoint_walk"


class DidiDataError(RuntimeError):
    """DiDi 数据层错误（文件缺失 / 列名不对 / 结构不符）。"""


# ---------------------------------------------------------------------------
# 1. 静态文件读取
# ---------------------------------------------------------------------------
def load_dicts(path: str | Path) -> Dict[int, Tuple[int, int, int]]:
    """读取 ``dicts.pkl``，返回 ``road_id -> (u, v, key)``。"""
    path = Path(path)
    if not path.exists():
        raise DidiDataError(f"dicts file not found: {path}")
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or IDX2EDGE not in payload:
        raise DidiDataError(
            f"{path} does not contain {IDX2EDGE!r}; got keys="
            f"{sorted(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )
    idx2edge = payload[IDX2EDGE]
    mapping: Dict[int, Tuple[int, int, int]] = {}
    for key, value in idx2edge.items():
        item = tuple(int(x) for x in value)
        if len(item) != 3:
            raise DidiDataError(
                f"{path}: {IDX2EDGE}[{key}] has {len(item)} entries, expected (u, v, key)"
            )
        mapping[int(key)] = item
    return mapping


def read_edge_features(
    path: str | Path,
) -> Tuple[List[str], Dict[str, Any]]:
    """用标准库 csv 读 ``edge_features.csv``（不依赖 pandas）。

    Returns:
        (columns, rows_as_lists)。行数只有几千，整表放内存没有压力。
    """
    path = Path(path)
    if not path.exists():
        raise DidiDataError(f"edge_features file not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
        except StopIteration as error:  # pragma: no cover - 空文件
            raise DidiDataError(f"{path} is empty") from error
        rows = [row for row in reader if row]
    return [column.strip() for column in columns], rows


def detect_length_columns(columns: Sequence[str]) -> List[str]:
    """猜"哪些列可能是道路长度"——**只用于报错时给人提示**。

    方案第 3.2 节明令禁止静默猜列名，所以这个函数的结果不会自动被采用。
    """
    lowered = {column.lower(): column for column in columns}
    return [lowered[name] for name in LENGTH_COLUMN_CANDIDATES if name in lowered]


def resolve_length_column(columns: Sequence[str], requested: Optional[str]) -> str:
    """确定道路长度列名；无法确定就报错，绝不退化成 ``weight=1``。

    ``requested`` 为空时抛 :class:`DidiDataError`，并把"看起来像长度的列"和
    "实际列名"一起放进消息里，方便人工确认后填进配置。
    """
    if requested is None or str(requested).strip() == "":
        raise DidiDataError(
            "data.length_column is not set. Run `prepare_didi.py --scan-only` to "
            "print edge_features columns, then set data.length_column explicitly. "
            f"actual columns={list(columns)}; length-like columns="
            f"{detect_length_columns(columns)}. Refusing to silently fall back to "
            "weight=1 (implementation plan section 3.2)."
        )
    if requested not in columns:
        raise DidiDataError(
            f"data.length_column={requested!r} is not a column of edge_features; "
            f"actual columns={list(columns)}; length-like columns="
            f"{detect_length_columns(columns)}"
        )
    return str(requested)


def load_edge_lengths(
    path: str | Path,
    length_column: Optional[str],
    road_id_column: str = "road_id",
) -> Tuple[List[str], Dict[int, float]]:
    """读取 ``road_id -> length``，并执行方案第 3.2 节的硬校验。"""
    columns, rows = read_edge_features(path)
    resolved = resolve_length_column(columns, length_column)
    if road_id_column not in columns:
        raise DidiDataError(
            f"{path}: no {road_id_column!r} column; actual columns={columns}"
        )
    road_index = columns.index(road_id_column)
    length_index = columns.index(resolved)

    lengths: Dict[int, float] = {}
    bad: List[str] = []
    for row in rows:
        if len(row) <= max(road_index, length_index):
            bad.append(f"short row {row!r}")
            continue
        try:
            road_id = int(float(row[road_index]))
            value = float(row[length_index])
        except ValueError:
            bad.append(f"non-numeric row {row!r}")
            continue
        lengths[road_id] = value
    if bad:
        raise DidiDataError(
            f"{path}: {len(bad)} malformed rows (first: {bad[0]}); "
            "refusing to build a graph from a partially parsed table"
        )
    if not lengths:
        raise DidiDataError(f"{path}: no usable rows")

    values = np.asarray(list(lengths.values()), dtype=float)
    if not np.isfinite(values).all():
        raise DidiDataError(
            f"{path}: column {resolved!r} contains non-finite values "
            f"({int((~np.isfinite(values)).sum())} of {values.size})"
        )
    if not (values > 0).all():
        raise DidiDataError(
            f"{path}: column {resolved!r} contains non-positive values "
            f"({int((values <= 0).sum())} of {values.size}); edge weights must be > 0"
        )
    return columns, lengths


# ---------------------------------------------------------------------------
# 2. 全局无向有权 Junction Graph
# ---------------------------------------------------------------------------
@dataclass
class GraphBuildStats:
    """方案第 3.3 节要求写进 ``metadata.json`` 的统计。"""

    raw_road_segments: int = 0
    unique_junction_edges: int = 0
    parallel_segments_merged: int = 0
    self_loop_segments: int = 0
    missing_length_roads: int = 0
    num_nodes: int = 0
    num_connected_components: int = 0
    largest_component_nodes: int = 0
    edge_length_min: float = 0.0
    edge_length_mean: float = 0.0
    edge_length_max: float = 0.0
    num_degree_ge3_nodes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def build_global_weighted_graph(
    idx2edge: Dict[int, Tuple[int, int, int]],
    lengths: Dict[int, float],
    stats: Optional[GraphBuildStats] = None,
) -> Tuple[nx.Graph, GraphBuildStats]:
    """把 road segment 表折叠成 **无向 + 有权** 的 junction graph。

    折叠规则（方案第 3.3 节）：同一无向 ``(u, v)`` 上的多条平行 road segment
    只保留**最短**的那条长度

        w_{uv} = min_{r in R(u,v)} l_r

    理由：模型只在 junction 层面决定 ``u -> v`` 这个拓扑选择，不负责区分两条
    平行 road id。自环 road（``u == v``）会被丢弃并计数 —— 它们在
    ``branch_segments`` 的端点/分支语义里没有意义。
    """
    stats = stats if stats is not None else GraphBuildStats()
    stats.raw_road_segments = len(idx2edge)

    graph = nx.Graph()
    graph.graph["weighted"] = True
    graph.graph["weight"] = "weight"
    seen: Dict[Tuple[int, int], int] = {}
    edge_lengths: List[float] = []

    for road_id in sorted(idx2edge):
        u, v, _key = idx2edge[road_id]
        if u == v:
            stats.self_loop_segments += 1
            continue
        length = lengths.get(road_id)
        if length is None:
            stats.missing_length_roads += 1
            continue
        length = float(length)
        if not math.isfinite(length) or length <= 0:
            raise DidiDataError(
                f"road {road_id} has invalid length {length}; every edge weight "
                "must be finite and > 0"
            )
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            stats.parallel_segments_merged += 1
            data = graph.edges[key]
            if length < float(data["weight"]):
                data["weight"] = length
            data["num_road_segments"] = int(data.get("num_road_segments", 1)) + 1
            continue
        seen[key] = road_id
        graph.add_edge(u, v, weight=length, num_road_segments=1)

    if graph.number_of_nodes() == 0:
        raise DidiDataError("the junction graph is empty; check dicts/edge_features")

    edge_lengths = [float(data["weight"]) for _, _, data in graph.edges(data=True)]
    stats.unique_junction_edges = graph.number_of_edges()
    stats.num_nodes = graph.number_of_nodes()
    components = sorted(
        (len(group) for group in nx.connected_components(graph)), reverse=True
    )
    stats.num_connected_components = len(components)
    stats.largest_component_nodes = components[0] if components else 0
    stats.num_degree_ge3_nodes = sum(
        1 for node in graph.nodes() if graph.degree(node) >= 3
    )
    array = np.asarray(edge_lengths, dtype=float)
    stats.edge_length_min = float(array.min()) if array.size else 0.0
    stats.edge_length_mean = float(array.mean()) if array.size else 0.0
    stats.edge_length_max = float(array.max()) if array.size else 0.0
    return graph, stats


# ---------------------------------------------------------------------------
# 3. road path -> junction path
# ---------------------------------------------------------------------------
@dataclass
class RoadConversion:
    """一条 road path 的转换结果。"""

    path: Optional[List[int]]
    method: str = ""
    reason: str = ""
    dropped_duplicate_roads: int = 0
    dropped_self_loop_roads: int = 0
    u_turns: int = 0

    @property
    def ok(self) -> bool:
        return self.path is not None


def parse_road_path(text: Any) -> List[int]:
    """把 CSV 里的 ``path`` 字段变成 road id 列表。

    **禁止 eval()**，统一 ``ast.literal_eval``（方案第 4.2 节）。
    """
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    if not isinstance(text, str):
        raise ValueError(f"path is neither a string nor a sequence: {type(text)!r}")
    value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"path literal is not a list: {type(value)!r}")
    return [int(x) for x in value]


def _drop_consecutive_duplicates(values: Sequence[int]) -> Tuple[List[int], int]:
    out: List[int] = []
    dropped = 0
    for value in values:
        if out and out[-1] == value:
            dropped += 1
            continue
        out.append(value)
    return out, dropped


def _stored_direction_walk(roads: Sequence[Tuple[int, int, int]]) -> Optional[List[int]]:
    """按 ``idx2edge`` 的存储方向走一遍：要求 ``u_{i+1} == v_i``。"""
    path = [roads[0][1]]
    for _road_id, u, v in roads:
        if u != path[-1]:
            return None
        path.append(v)
    return path


def _endpoint_walk(roads: Sequence[Tuple[int, int, int]]) -> Optional[List[int]]:
    """方案第 4.1 节的端点集合写法（不依赖存储方向，作为兜底）。

    相邻两条 road 必须恰好共享一个 junction；共享两个说明是平行路段掉头，
    此时按存储方向走（``a -> b -> a``）。

    注意前两条 road 要一起消费：``pending`` 与它的后继共享的 junction 才是
    "中转路口"，而**后继那条 road 本身也已经被走过**，所以 ``current`` 必须落在
    后继 road 的另一端。只把 ``current`` 设成共享路口、再让下一条 road 去接它，
    会把每条 road 都算错一次（这是实现时真实踩过的坑）。
    """
    path: List[int] = []
    current: Optional[int] = None
    pending: Optional[Tuple[int, int]] = None

    for _road_id, u, v in roads:
        ends = (u, v)
        if current is None:
            if pending is None:
                pending = ends
                continue
            shared = set(pending) & set(ends)
            if len(shared) == 2:
                # 平行路段掉头：方向由 pending 的存储方向决定
                path = [pending[0], pending[1], pending[0]]
                current = pending[0]
                continue
            if len(shared) != 1:
                return None
            joint = shared.pop()
            start = pending[1] if pending[0] == joint else pending[0]
            far = ends[1] if ends[0] == joint else ends[0]
            path = [start, joint, far]
            current = far
            continue
        if current not in ends:
            return None
        nxt = ends[1] if current == ends[0] else ends[0]
        path.append(nxt)
        current = nxt

    if current is None:
        return None
    return path


def road_path_to_junction_path(
    road_ids: Sequence[int],
    idx2edge: Dict[int, Tuple[int, int, int]],
    graph: Optional[nx.Graph] = None,
) -> RoadConversion:
    """真实 road id 序列 -> junction 节点序列（方案第 4.1 / 4.2 节）。

    处理顺序：

        删除连续重复 road id
          -> 丢弃自环 road
          -> 存储方向走一遍（主算法）
          -> 走不通时用端点集合兜底
          -> 合并连续重复 junction
          -> （给了 graph 时）校验每一步都是真实 graph edge
    """
    if not road_ids:
        return RoadConversion(None, reason="empty")

    cleaned, dropped_dup = _drop_consecutive_duplicates([int(x) for x in road_ids])
    roads: List[Tuple[int, int, int]] = []
    dropped_self = 0
    for road_id in cleaned:
        edge = idx2edge.get(road_id)
        if edge is None:
            return RoadConversion(
                None,
                reason="unknown_road_id",
                dropped_duplicate_roads=dropped_dup,
            )
        u, v = int(edge[0]), int(edge[1])
        if u == v:
            dropped_self += 1
            continue
        roads.append((road_id, u, v))

    if len(roads) < 2:
        return RoadConversion(
            None,
            reason="too_short",
            dropped_duplicate_roads=dropped_dup,
            dropped_self_loop_roads=dropped_self,
        )

    method = CONVERSION_STORED_DIRECTION
    path = _stored_direction_walk(roads)
    if path is None:
        method = CONVERSION_ENDPOINT_WALK
        path = _endpoint_walk(roads)
    if path is None:
        return RoadConversion(
            None,
            reason="discontinuous",
            dropped_duplicate_roads=dropped_dup,
            dropped_self_loop_roads=dropped_self,
        )

    path, _ = _drop_consecutive_duplicates(path)
    if len(path) < 2:
        return RoadConversion(
            None,
            reason="degenerate_junction_path",
            dropped_duplicate_roads=dropped_dup,
            dropped_self_loop_roads=dropped_self,
        )

    u_turns = 0
    for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
        if a == c:
            u_turns += 1

    if graph is not None:
        for a, b in zip(path[:-1], path[1:]):
            if not graph.has_edge(a, b):
                return RoadConversion(
                    None,
                    reason="non_edge_transition",
                    dropped_duplicate_roads=dropped_dup,
                    dropped_self_loop_roads=dropped_self,
                    u_turns=u_turns,
                )

    return RoadConversion(
        path,
        method=method,
        dropped_duplicate_roads=dropped_dup,
        dropped_self_loop_roads=dropped_self,
        u_turns=u_turns,
    )


def path_cost(graph: nx.Graph, path: Sequence[int], weight: str = "weight") -> float:
    """路径 cost = 边权之和；出现非边返回 ``inf``。"""
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            return float("inf")
        total += float(graph.edges[u, v].get(weight, 1.0))
    return total


def is_simple_path(path: Sequence[int]) -> bool:
    """GT 里没有重复 junction（``require_simple_gt``）。"""
    return len(set(path)) == len(path)


# ---------------------------------------------------------------------------
# 4. OD Corridor（方案第 6 节）
# ---------------------------------------------------------------------------
@dataclass
class CorridorResult:
    """一个样本的 corridor 结果 + 统计。"""

    graph: nx.Graph
    num_nodes: int
    num_edges: int
    num_decisions: int
    dijkstra_cost: float
    radius: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_corridor_nodes": int(self.num_nodes),
            "num_corridor_edges": int(self.num_edges),
            "num_decisions": int(self.num_decisions),
            "dijkstra_cost": float(self.dijkstra_cost),
            "radius": float(self.radius),
        }


def count_decisions(graph: nx.Graph, start: Any, goal: Any) -> int:
    """corridor 上的 decision 数（与 ``branch_segments.build_decision_nodes`` 同口径）。

    口径是**集合**运算：

        D(G, s, g) = {v : deg(v) >= 3} ∪ {s : deg(s) > 1} \\ {g}

    注意 ``∪``：当 ``deg(start) >= 3`` 时 start **已经**在第一个集合里了，
    再 ``+1`` 就把 Start 重复算了一次（这个 bug 真实存在过，会让
    ``scan_corridor.json`` / ``metadata`` / ``candidate.num_decisions`` 全部偏大 1）。
    这里直接用 set 写，语义上就不可能重复。
    """
    decisions = {node for node in graph.nodes() if graph.degree(node) >= 3}
    if graph.degree(start) > 1:
        decisions.add(start)
    decisions.discard(goal)
    return len(decisions)


def corridor_decision_count(
    graph: nx.Graph, keep: Iterable[Any], start: Any, goal: Any
) -> int:
    """不建子图，直接在邻居集合上算 corridor 的 decision 数。

    ``extract_segments`` 用的是**子图内**的度（邻居被裁掉的节点会掉到度 1/2），
    所以这里必须按 ``keep`` 过滤邻居来数，不能用原图的度。rho 扫描要为每个候选
    算多个 rho，建子图的常数开销会把扫描拖慢一个量级。

    口径与 :func:`count_decisions` / ``build_decision_nodes`` 完全一致（集合语义，
    Start 只算一次），否则扫描出来的 ``mean_decisions`` 会比真实值大 1。
    """
    keep_set = keep if isinstance(keep, (set, frozenset)) else set(keep)
    total = 0
    start_degree = 0
    goal_degree = 0
    for node in keep_set:
        degree = 0
        for neighbour in graph.neighbors(node):
            if neighbour in keep_set:
                degree += 1
        if node == start:
            start_degree = degree
        if node == goal:
            goal_degree = degree
        if degree >= 3:
            total += 1
    # 集合写法：deg(start) >= 3 时它已经在 total 里了，不能再加
    if start_degree > 1 and start_degree < 3:
        total += 1
    if goal_degree >= 3:
        total -= 1
    return int(total)


def corridor_node_set(
    graph: nx.Graph,
    start: Any,
    goal: Any,
    rho: float,
    dijkstra_cost: Optional[float] = None,
) -> Tuple[set, float]:
    """方案第 6.2 节的 corridor 公式，返回值只依赖 OD 与路网。

        d_s(v) = d(s, v)          d_g(v) = d(v, g)          d_sg = d(s, g)
        V_c = { v | d_s(v) + d_g(v) <= rho * d_sg }

    **函数签名里没有 GT path** —— 这是方案第 6.1 节明令禁止的泄漏点，
    测试 ``test_corridor_builder_does_not_take_gt`` 会在源码层面守住它。
    """
    if rho < 1.0:
        raise ValueError(f"rho must be >= 1, got {rho}")
    if dijkstra_cost is None:
        dijkstra_cost = float(
            nx.shortest_path_length(graph, int(start), int(goal), weight="weight")
        )
    if not math.isfinite(dijkstra_cost) or dijkstra_cost <= 0:
        raise ValueError(f"invalid shortest distance {dijkstra_cost}")

    radius = float(rho) * float(dijkstra_cost)
    d_s = nx.single_source_dijkstra_path_length(graph, start, cutoff=radius, weight="weight")
    d_g = nx.single_source_dijkstra_path_length(graph, goal, cutoff=radius, weight="weight")
    keep = {
        node
        for node, value in d_s.items()
        if node in d_g and value + d_g[node] <= radius + 1e-9
    }
    keep.add(start)
    keep.add(goal)
    return keep, float(dijkstra_cost)


def build_od_corridor(
    graph: nx.Graph,
    start: Any,
    goal: Any,
    rho: float,
    dijkstra_cost: Optional[float] = None,
    max_nodes: Optional[int] = None,
) -> Optional[CorridorResult]:
    """构造 OD corridor 子图。

    只使用 ``full graph + edge length + start + goal``，**不使用 GT**（方案第 6.1 节）。
    ``max_nodes`` 命中时返回 ``None``，由调用方记为超限并丢弃。
    """
    keep, cost = corridor_node_set(graph, start, goal, rho, dijkstra_cost)
    if max_nodes is not None and len(keep) > int(max_nodes):
        return None
    subgraph = graph.subgraph(keep).copy()
    # subgraph().copy() 会保留 graph/graph 属性，但显式再设一遍更保险：
    # collate 与 metrics 都靠 graph.graph['weighted'] 决定用不用 Dijkstra。
    subgraph.graph["weighted"] = True
    subgraph.graph["weight"] = "weight"
    subgraph.graph["corridor_rho"] = float(rho)
    if not subgraph.has_node(start) or not subgraph.has_node(goal):
        return None
    if subgraph.degree(start) == 0 or subgraph.degree(goal) == 0:
        return None
    return CorridorResult(
        graph=subgraph,
        num_nodes=subgraph.number_of_nodes(),
        num_edges=subgraph.number_of_edges(),
        num_decisions=count_decisions(subgraph, start, goal),
        dijkstra_cost=cost,
        radius=float(rho) * cost,
    )


def path_contained(graph: nx.Graph, path: Sequence[int]) -> bool:
    """GT 是否完整落在 corridor 里（corridor retention 的判定）。"""
    return all(node in graph for node in path)


class ShortestDistanceCache:
    """``d(s, .)`` 的按源缓存（Dijkstra 结果复用）。

    成都路网只有 2891 节点 / 4403 边，但对 10 万条候选轨迹逐一跑两次 Dijkstra
    仍然是分钟级开销；而候选的 source / goal 去重后远少于候选数。这里按 source
    缓存单源最短路，把 rho 扫描从 O(candidates * Dijkstra) 降到 O(distinct_sources
    * Dijkstra)。

    只缓存**真实用到的源**，缓存规模上限就是图的节点数。
    """

    def __init__(self, graph: nx.Graph, weight: str = "weight"):
        self.graph = graph
        self.weight = weight
        self._cache: Dict[Any, Dict[Any, float]] = {}

    def distances(self, source: Any) -> Dict[Any, float]:
        key = source
        cached = self._cache.get(key)
        if cached is None:
            cached = nx.single_source_dijkstra_path_length(
                self.graph, source, weight=self.weight
            )
            self._cache[key] = cached
        return cached

    def distance(self, source: Any, target: Any) -> float:
        value = self.distances(source).get(target)
        return float(value) if value is not None else float("inf")

    def corridor_mask(
        self,
        start: Any,
        goal: Any,
        rhos: Sequence[float],
    ) -> Dict[float, set]:
        """一次算好 d_s / d_g，再对多个 rho 各自给出 ``V_c``。

        rho 扫描（方案第 6.3 节）要比较多个候选，重复算距离是纯浪费。
        """
        d_s = self.distances(start)
        d_g = self.distances(goal)
        d_sg = d_s.get(goal)
        if d_sg is None or not math.isfinite(float(d_sg)) or float(d_sg) <= 0:
            raise ValueError(f"goal {goal!r} is unreachable from start {start!r}")
        d_sg = float(d_sg)
        combined = {
            node: value + d_g[node]
            for node, value in d_s.items()
            if node in d_g
        }
        out: Dict[float, set] = {}
        for rho in rhos:
            radius = float(rho) * d_sg
            keep = {node for node, value in combined.items() if value <= radius + 1e-9}
            keep.add(start)
            keep.add(goal)
            out[float(rho)] = keep
        return out

    def __len__(self) -> int:  # pragma: no cover - 诊断用
        return len(self._cache)


# ---------------------------------------------------------------------------
# 5. 轨迹过滤（方案第 4.3 节）
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryFilter:
    """第一版过滤规则。``min/max`` 是**工程初始值**，不是论文标准。"""

    min_road_segments: int = 10
    max_road_segments: int = 100
    require_simple_gt: bool = True
    #: road 段数区间是**开区间**：保留 ``min < len(road) < max``（方案 4.3）
    inclusive: bool = False

    def road_length_ok(self, num_road_segments: int) -> bool:
        if self.inclusive:
            return self.min_road_segments <= num_road_segments <= self.max_road_segments
        return self.min_road_segments < num_road_segments < self.max_road_segments

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def trajectory_reject_reason(
    *,
    num_road_segments: int,
    junction_path: Sequence[int],
    filter_cfg: TrajectoryFilter,
) -> Optional[str]:
    """返回 ``None`` 表示通过，否则返回过滤步骤名（用于统计保留率）。"""
    if not filter_cfg.road_length_ok(int(num_road_segments)):
        return "length_invalid"
    if filter_cfg.require_simple_gt and not is_simple_path(junction_path):
        return "non_simple_gt"
    return None


# ---------------------------------------------------------------------------
# 6. 轨迹读入
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryCandidate:
    """一条通过清洗的候选轨迹（尚未建 corridor / 尚未变成 GraphSample）。

    这里**不保留原始 road id 列表**：10 万级候选上它是纯内存负担，而下游需要的
    只有"多少段"和"多少 cost"，两者都在收集阶段一次算好。
    """

    order_id: str
    date: str
    junction_path: List[int]
    raw_road_len: int
    junction_len: int
    raw_road_cost: float
    gt_cost: float
    dijkstra_cost: float
    conversion: str
    u_turns: int = 0
    #: corridor 阶段填
    corridor_nodes: int = 0
    corridor_edges: int = 0
    num_decisions: int = 0
    corridor_miss: bool = False

    @property
    def gt_cost_ratio(self) -> float:
        if self.dijkstra_cost <= 0:
            return float("inf")
        return self.gt_cost / self.dijkstra_cost

    def dedup_key(self) -> Tuple[int, int, Tuple[int, ...]]:
        return (int(self.junction_path[0]), int(self.junction_path[-1]), tuple(self.junction_path))

    def to_manifest_row(self, sample_id: int, split: str) -> Dict[str, Any]:
        return {
            "sample_id": int(sample_id),
            "order_id": self.order_id,
            "date": self.date,
            "split": split,
            "raw_road_len": int(self.raw_road_len),
            "junction_len": int(self.junction_len),
            "num_corridor_nodes": int(self.corridor_nodes),
            "num_corridor_edges": int(self.corridor_edges),
            "num_decisions": int(self.num_decisions),
            "gt_cost": float(self.gt_cost),
            "dijkstra_cost": float(self.dijkstra_cost),
            "gt_cost_ratio": float(self.gt_cost_ratio),
            "corridor_miss": bool(self.corridor_miss),
            "conversion": self.conversion,
            "u_turns": int(self.u_turns),
        }


def iter_trajectory_csv(
    path: str | Path,
    max_rows: Optional[int] = None,
    order_id_column: str = "order_id",
    path_column: str = "path",
) -> Iterator[Tuple[int, Dict[str, str]]]:
    """流式读一个轨迹 CSV，产出 ``(row_index, row)``。

    这些文件每个约 100MB / 16 万行，整表读进内存没有必要。
    """
    path = Path(path)
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for required in (order_id_column, path_column):
            if required not in fields:
                raise DidiDataError(
                    f"{path}: no {required!r} column; actual columns={fields}"
                )
        for index, row in enumerate(reader):
            if max_rows is not None and index >= int(max_rows):
                break
            yield index, row


def discover_trajectory_files(root: str | Path, pattern: str) -> List[Path]:
    """按 glob 找轨迹 CSV，按文件名排序（确定性）。"""
    root = Path(root)
    if not root.exists():
        raise DidiDataError(f"data root not found: {root}")
    files = sorted(root.glob(pattern))
    if not files:
        raise DidiDataError(f"no file matching {pattern!r} under {root}")
    return files


# ---------------------------------------------------------------------------
# 6b. 节点经纬度（真实地理底图 / km-based DTW 用）
# ---------------------------------------------------------------------------
_SHAPELY_STUB_MODULES = (
    "shapely",
    "shapely.geometry",
    "shapely.geometry.linestring",
    "shapely.geometry.point",
    "shapely.geometry.polygon",
)


def _install_shapely_stub() -> None:
    """在没装 shapely 的环境里注入一个"只接住 unpickle、不解析几何"的最小替身。

    为什么需要：DiDi 附带的 ``ChengDu.pkl`` / ``graph.pkl`` 是 OSMnx 1.1.1 导出的
    ``MultiDiGraph``，**边属性里带 ``shapely.geometry.linestring.LineString``**。
    没有 shapely 时 ``pickle.load`` 直接报 ``No module named 'shapely'``，
    于是"这份数据没有经纬度"这个结论会被错误地接受 —— 实际上**节点自带 x/y**
    （``crs = epsg:4326``），只是被一个 optional dependency 挡住了。

    我们只读节点的 ``x`` / ``y``，完全不碰 geometry，所以替身只需要能被 pickle
    的 ``__setstate__`` 路径接住即可。真实 shapely 存在时永远不会走到这里。
    """
    import types

    for name in _SHAPELY_STUB_MODULES:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        if name == "shapely":
            module.__version__ = "0.0.0-stub"
        sys.modules[name] = module
    for class_name, holder in (
        ("LineString", "shapely.geometry.linestring"),
        ("Point", "shapely.geometry.point"),
        ("Polygon", "shapely.geometry.polygon"),
    ):
        if not hasattr(sys.modules[holder], class_name):
            setattr(sys.modules[holder], class_name, type(class_name, (), {
                "__init__": lambda self, *a, **k: setattr(self, "args", a),
                "__setstate__": lambda self, state: setattr(self, "state", state),
            }))
    for attribute in ("linestring", "point", "polygon"):
        if not hasattr(sys.modules["shapely.geometry"], attribute):
            setattr(
                sys.modules["shapely.geometry"], attribute,
                sys.modules[f"shapely.geometry.{attribute}"],
            )


def load_node_coordinates(
    path: str | Path,
    node_x: str = "x",
    node_y: str = "y",
) -> Dict[Any, Tuple[float, float]]:
    """读 OSMnx 图（``ChengDu.pkl`` / ``graph.pkl``），返回 ``node -> (lon, lat)``。

    ``dicts.pkl`` 里的 ``(u, v)`` 就是 OSM node id，与这份图的节点 id 同一套编号，
    所以可以直接把经纬度挂到我们的 junction 图上 —— 于是可以做**真实地理底图**
    和方案第 12.5 节一直没能实现的 **km-based DTW**。

    注意两份 ``dicts.pkl`` 并不相同（``didi_datasets/.../didi_chengdu`` 是 2891 节点，
    ``data/data/cd`` 是 2848 节点），所以坐标对**我们用的那张图**的覆盖率不是 100%
    （实测成都为 2780/2891 = 96.2%）。要 100% 覆盖请用
    :func:`load_node_coordinates_filled`。
    """
    path = Path(path)
    if not path.exists():
        raise DidiDataError(f"OSMnx graph not found: {path}")
    with warnings.catch_warnings():
        # shapely 2.x 会对 1.x 时代存的几何对象发 UserWarning，这里只读节点属性，
        # 不解析几何，噪声警告直接吞掉
        warnings.simplefilter("ignore", UserWarning)
        try:
            with open(path, "rb") as handle:
                graph = pickle.load(handle)
        except ModuleNotFoundError as error:
            if "shapely" not in str(error):
                raise
            _install_shapely_stub()
            with open(path, "rb") as handle:
                graph = pickle.load(handle)

    coordinates: Dict[Any, Tuple[float, float]] = {}
    for node, data in graph.nodes(data=True):
        if node_x in data and node_y in data:
            coordinates[node] = (float(data[node_x]), float(data[node_y]))
    if not coordinates:
        raise DidiDataError(
            f"{path}: nodes carry no {node_x!r}/{node_y!r} attributes; "
            "this is not an OSMnx node-coordinate graph"
        )
    return coordinates


def load_node_coordinates_filled(
    coords_path: str | Path,
    graph_path: Optional[str | Path] = None,
) -> Tuple[Dict[Any, Tuple[float, float]], Dict[str, Any]]:
    """加载坐标并**补齐缺失节点**，返回 ``(node -> (lon, lat), stats)``。

    为什么必须补：OSMnx 图只覆盖我们 2891 个节点里的 2780 个（96.2%），剩下 111 个
    没有坐标。直接拿原始坐标去算 km-based DTW，凡是路径碰到这 111 个节点之一的样本
    都会得到 NaN —— 实测 test 集里确实有这种样本，会静默丢掉一批 DTW 值。

    ``graph_path`` 指向 ``graph_global.pkl`` 时，缺失节点用邻居坐标均值迭代填充，
    覆盖率变成 100%（图越局部补得越准）。
    """
    coordinates = load_node_coordinates(coords_path)
    stats: Dict[str, Any] = {
        "source": str(coords_path),
        "total_nodes": len(coordinates),
        "with_coordinates": len(coordinates),
        "filled": 0,
        "missing": 0,
        "coverage": 1.0,
    }
    if graph_path is None or not Path(graph_path).exists():
        stats["note"] = "no graph_global.pkl given; missing nodes are not filled"
        return coordinates, stats

    with open(graph_path, "rb") as handle:
        payload = pickle.load(handle)
    graph = payload["graph"] if isinstance(payload, dict) else payload
    graph, attach_stats = attach_coordinates(graph, coordinates, fill_missing=True)
    filled = {
        node: (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
        for node in graph.nodes()
        if "x" in graph.nodes[node]
    }
    stats.update(
        {
            "total_nodes": attach_stats["nodes"],
            "with_coordinates": attach_stats["with_coordinates"],
            "filled": attach_stats["filled"],
            "missing": attach_stats["missing"],
            "coverage": attach_stats["coverage"],
        }
    )
    return filled, stats


def attach_coordinates(
    graph: nx.Graph,
    coordinates: Dict[Any, Tuple[float, float]],
    fill_missing: bool = True,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """把经纬度写到图的节点属性上；返回 (graph, 覆盖统计)。

    ``fill_missing`` 打开时，没有坐标的节点用邻居坐标的均值迭代补几轮 —— 这样
    画图时不会出现"孤零零一堆点没位置"，也不会把它们的边整段丢掉。
    """
    stats = {"nodes": graph.number_of_nodes(), "with_coordinates": 0, "filled": 0, "missing": 0}
    for node in graph.nodes():
        if node in coordinates:
            longitude, latitude = coordinates[node]
            graph.nodes[node]["x"] = longitude
            graph.nodes[node]["y"] = latitude
            stats["with_coordinates"] += 1

    if fill_missing:
        missing = [node for node in graph.nodes() if "x" not in graph.nodes[node]]
        for _round in range(8):
            if not missing:
                break
            still: List[Any] = []
            for node in missing:
                neighbours = [
                    n for n in graph.neighbors(node) if "x" in graph.nodes[n]
                ]
                if neighbours:
                    graph.nodes[node]["x"] = float(
                        np.mean([graph.nodes[n]["x"] for n in neighbours])
                    )
                    graph.nodes[node]["y"] = float(
                        np.mean([graph.nodes[n]["y"] for n in neighbours])
                    )
                    stats["filled"] += 1
                else:
                    still.append(node)
            missing = still

    stats["missing"] = sum(1 for node in graph.nodes() if "x" not in graph.nodes[node])
    stats["coverage"] = stats["with_coordinates"] / max(stats["nodes"], 1)
    graph.graph["crs"] = "epsg:4326"
    return graph, stats


# ---------------------------------------------------------------------------
# 7. 去重 / 划分（方案第 7 节）
# ---------------------------------------------------------------------------
def deduplicate_candidates(
    candidates: Sequence[TrajectoryCandidate],
) -> Tuple[List[TrajectoryCandidate], int]:
    """按 ``(start, goal, junction_path)`` 去重，**必须先于 split**。

    完全重复的轨迹只保留第一条。相同 OD 但路径不同是允许的（GDP 的主要实验
    本来就是在同一张城市图上做 unseen path）。
    """
    seen = set()
    kept: List[TrajectoryCandidate] = []
    dropped = 0
    for candidate in candidates:
        key = candidate.dedup_key()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(candidate)
    return kept, dropped


def split_real_paths(
    candidates: Sequence[TrajectoryCandidate],
    fractions: Dict[str, float],
    seed: int = 0,
) -> Dict[str, List[TrajectoryCandidate]]:
    """path-level 固定划分（方案第 7.1 节）。

    真实城市是**一张固定图**，所以不能沿用 synthetic 的 ``split_by_graph``：
    这里按轨迹（样本）划分。顺序由 ``seed`` 决定且**结果会被固化到
    ``split_manifest.csv``**，不会每次 prepare 都变。
    """
    names = list(fractions)
    total = len(candidates)
    order = np.random.default_rng(int(seed)).permutation(total).tolist()

    shares: Dict[str, int] = {}
    assigned = 0
    for index, name in enumerate(names):
        if index == len(names) - 1:
            shares[name] = total - assigned
        else:
            shares[name] = int(round(float(fractions[name]) * total))
            assigned += shares[name]

    splits: Dict[str, List[TrajectoryCandidate]] = {}
    cursor = 0
    for name in names:
        count = max(shares[name], 0)
        splits[name] = [candidates[i] for i in order[cursor : cursor + count]]
        cursor += count
    return splits


def sample_subset(
    candidates: Sequence[TrajectoryCandidate],
    size: int,
    seed: int = 0,
) -> List[TrajectoryCandidate]:
    """从一组候选里**均匀**抽 ``size`` 条（GDP 风格固定 test_1000，方案 7.4）。"""
    if size >= len(candidates):
        return list(candidates)
    order = np.random.default_rng(int(seed)).permutation(len(candidates))[:size]
    return [candidates[int(index)] for index in sorted(order.tolist())]


def stratified_subsample(
    candidates: Sequence[TrajectoryCandidate],
    size: int,
    seed: int = 0,
    num_length_buckets: int = 3,
    length_key: str = "junction_len",
) -> Tuple[List[TrajectoryCandidate], Dict[str, Any]]:
    """按 **(日期 × GT 长度分位)** 分层抽样到 ``size`` 条。

    为什么不能"简单取前 N 条"或"整体均匀随机抽"：候选池来自 10 个日期文件，
    轨迹长度跨度很大（junction 数 12~94）。直接截断会让数据集集中在最初几天；
    整体随机抽虽然无偏，但小样本下仍可能碰巧某天特别少。而 OD 距离 / 路径长度
    恰恰是规划难度的主要因素（GDP 本身就按 short/medium/long 分组报 Hit Ratio），
    所以长度分布必须被控制住。

    做法：

    1. 用池子的 ``length_key`` 分位数把候选切成 ``num_length_buckets`` 个长度层；
    2. 分层键是 ``(date, length_bucket)``；
    3. 按各层规模**等比例**分配名额（最大余额法），某层不够就把它剩下的名额按
       同样规则让给还有余量的层；
    4. 层内用 ``default_rng([seed, stratum_index])`` 独立抽样（确定性、可复现）。

    Returns:
        ``(subset, stats)``；stats 里记录长度分层边界、每个长度层 / 每个日期的
        池子规模与入选规模 —— 抽样结果必须可自证没有偏向某几天或某个长度段。
    """
    if size >= len(candidates):
        return list(candidates), {
            "pool_size": len(candidates),
            "requested": int(size),
            "selected": len(candidates),
            "note": "pool smaller than requested size; no subsampling applied",
        }

    lengths = np.asarray([float(getattr(c, length_key)) for c in candidates], dtype=float)
    quantiles = [index / num_length_buckets for index in range(1, num_length_buckets)]
    edges = [float(np.quantile(lengths, q)) for q in quantiles]
    bucket_of = np.searchsorted(np.asarray(edges, dtype=float), lengths, side="right")

    strata: Dict[Tuple[str, int], List[int]] = {}
    for index, candidate in enumerate(candidates):
        strata.setdefault((str(candidate.date), int(bucket_of[index])), []).append(index)

    sizes = {key: len(rows) for key, rows in strata.items()}
    quota = np.asarray([sizes[key] for key in strata], dtype=float)
    pool_total = float(quota.sum())
    target = int(min(size, pool_total))
    raw = quota * target / pool_total if pool_total > 0 else np.zeros_like(quota)
    alloc = np.minimum(np.floor(raw), quota).astype(int)
    remainder = target - int(alloc.sum())

    keys = list(strata)
    fractional_order = sorted(
        range(len(keys)), key=lambda i: (-(raw[i] - math.floor(raw[i])), str(keys[i]))
    )
    while remainder > 0:
        progressed = False
        for position in fractional_order:
            if remainder == 0:
                break
            if alloc[position] < sizes[keys[position]]:
                alloc[position] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break

    selected: List[int] = []
    per_stratum: Dict[str, Dict[str, int]] = {}
    for position, key in enumerate(keys):
        rows = strata[key]
        take = int(alloc[position])
        if take <= 0:
            continue
        local_rng = np.random.default_rng([int(seed), position])
        picked = local_rng.permutation(len(rows))[:take]
        selected.extend(rows[int(index)] for index in picked.tolist())
        per_stratum[f"{key[0]}|len{key[1]}"] = {"pool": len(rows), "selected": take}

    selected.sort()

    per_date: Dict[str, Dict[str, int]] = {}
    per_bucket: Dict[str, Dict[str, int]] = {}
    for index, candidate in enumerate(candidates):
        date = str(candidate.date)
        bucket = f"len{int(bucket_of[index])}"
        per_date.setdefault(date, {"pool": 0, "selected": 0})["pool"] += 1
        per_bucket.setdefault(bucket, {"pool": 0, "selected": 0})["pool"] += 1
    for index in selected:
        date = str(candidates[index].date)
        bucket = f"len{int(bucket_of[index])}"
        per_date[date]["selected"] += 1
        per_bucket[bucket]["selected"] += 1

    stats = {
        "pool_size": len(candidates),
        "requested": int(size),
        "selected": len(selected),
        "length_key": length_key,
        "num_length_buckets": int(num_length_buckets),
        "length_bucket_edges": edges,
        "length_bucket_pool_range": {
            f"len{b}": [
                float(lengths[bucket_of == b].min()) if (bucket_of == b).any() else 0.0,
                float(lengths[bucket_of == b].max()) if (bucket_of == b).any() else 0.0,
            ]
            for b in range(int(num_length_buckets))
        },
        "per_date": per_date,
        "per_length_bucket": per_bucket,
        "num_strata": len(keys),
        "num_nonempty_strata": int(sum(1 for value in alloc if value > 0)),
        "num_dates_covered": int(sum(1 for value in per_date.values() if value["selected"] > 0)),
        "num_dates_total": len(per_date),
        "per_stratum": per_stratum,
    }
    return [candidates[index] for index in selected], stats


# ---------------------------------------------------------------------------
# 8. 漏斗统计
# ---------------------------------------------------------------------------
@dataclass
class FunnelStats:
    """方案第 4.3 节要求的"任何过滤都不能静默发生"。"""

    counts: Counter = field(default_factory=Counter)

    def add(self, key: str, amount: int = 1) -> None:
        self.counts[key] += int(amount)

    def to_dict(self) -> Dict[str, Any]:
        counts = dict(self.counts)
        raw = max(counts.get("raw_rows", 0), 1)
        out: Dict[str, Any] = dict(counts)
        out["retention"] = {
            key: counts.get(key, 0) / raw
            for key in (
                "parse_valid",
                "road_id_valid",
                "continuous_valid",
                "length_valid",
                "simple_valid",
                "corridor_contained",
                "final_samples",
            )
        }
        return out
