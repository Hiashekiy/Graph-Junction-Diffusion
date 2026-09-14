"""Synthetic graph generators (guide section 8.1 / doc 04 section 4).

Four topology families are supported:

    er         Erdos-Renyi   G(N, p), parameterised by *mean degree* c:
                              p = c / (N - 1)
    ba         Barabasi-Albert preferential attachment
    ws         Watts-Strogatz small world
    geometric  random geometric graph in the unit square, parameterised by a
               multiple of the finite-size connectivity radius
                              r = alpha * sqrt(log N / (pi N))

Engineering note (see IMPLEMENTATION_NOTES.md, 2026-09-11 "generator parameters")
------------------------------------------------------------------------------
The V1 config asks for 20 <= N <= 80 together with d_G(s,g) >= 5.  A dense
Erdős-Rényi graph on 80 nodes has diameter ~3, so a *fixed* p is incompatible
with the distance filter.  Both families are therefore parameterised in a
size-aware way (mean degree for ER, connectivity-scaled radius for geometric)
and every generated graph must additionally admit at least one pair of nodes at
graph distance >= min_od_distance.  Graphs that cannot provide such a pair are
rejected and redrawn.  This changes *how the synthetic pool is parameterised*,
not the algorithm.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np

GRAPH_TYPES = ("er", "ba", "ws", "geometric", "grid")

# 边权的取值分布。uniform 是方案第 10 节的默认实现；loguniform 只是同一个接口上
# 的另一个可选分布（跨度更大 => BFS 路径的 cost ratio 更高、任务更难）。
WEIGHT_DISTRIBUTIONS = ("uniform", "loguniform")


# ----------------------------------------------------------------------------
# single-family generators
# ----------------------------------------------------------------------------
def generate_er_graph(num_nodes: int, p: float, rng: np.random.Generator) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.erdos_renyi_graph(num_nodes, min(max(p, 0.0), 1.0), seed=seed)


def generate_ba_graph(num_nodes: int, m: int, rng: np.random.Generator) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    m = max(1, min(m, max(1, (num_nodes - 2) // 2)))
    return nx.barabasi_albert_graph(num_nodes, m, seed=seed)


def generate_ws_graph(
    num_nodes: int, k: int, p: float, rng: np.random.Generator
) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    k = max(2, min(k, num_nodes - 1))
    if k % 2 == 1:
        k -= 1
    return nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)


def generate_geometric_graph(
    num_nodes: int, radius: float, rng: np.random.Generator
) -> nx.Graph:
    """Random geometric graph; node positions live in node attribute ``pos``."""
    seed = int(rng.integers(0, 2**31 - 1))
    return nx.random_geometric_graph(num_nodes, radius, seed=seed)


def generate_grid_graph(num_nodes: int, side: int, rng: np.random.Generator) -> nx.Graph:
    """2D lattice with ``side^2`` nodes (doc 04 section 4.5).

    A lattice is the cheapest way to get a *long* diameter: the shortest path
    between opposite corners is 2(side - 1) hops, so side=15 already exceeds the
    diameter of every 20..80 node graph in V1.  This is the family used by the
    long-range ablation, where a fixed-receptive-field message passing network
    cannot see the goal from most of the graph.
    """
    grid = nx.grid_2d_graph(side, side)
    order = sorted(grid.nodes())  # (i, j) tuples, lexicographic
    graph = nx.convert_node_labels_to_integers(grid, ordering="sorted")
    for new_label, (i, j) in enumerate(order):
        graph.nodes[new_label]["pos"] = (
            i / max(side - 1, 1),
            j / max(side - 1, 1),
        )
    return graph


_GENERATORS = {
    "er": generate_er_graph,
    "ba": generate_ba_graph,
    "ws": generate_ws_graph,
    "geometric": generate_geometric_graph,
    "grid": generate_grid_graph,
}


def geometric_radius_threshold(num_nodes: int) -> float:
    """Finite-size connectivity radius of a random geometric graph."""
    return math.sqrt(math.log(max(num_nodes, 3)) / (math.pi * max(num_nodes, 3)))


# ----------------------------------------------------------------------------
# parameter sampling
# ----------------------------------------------------------------------------
def sample_generator_params(
    graph_type: str,
    num_nodes: int,
    rng: np.random.Generator,
    generator_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Draw one concrete parameter set for the requested graph family."""
    generator_cfg = generator_cfg or {}

    def _range(key: str, default: Tuple[float, float]) -> Tuple[float, float]:
        value = generator_cfg.get(key, list(default))
        return float(value[0]), float(value[1])

    if graph_type == "er":
        if "p_range" in generator_cfg:
            lo, hi = _range("p_range", (0.06, 0.22))
            return {"p": float(rng.uniform(lo, hi))}
        # mean-degree parameterisation: robust across 20..80 nodes
        c_lo, c_hi = _range("c_range", (3.2, 4.8))
        c = float(rng.uniform(c_lo, c_hi))
        p = c / max(num_nodes - 1, 1)
        return {"p": p, "c": c}

    if graph_type == "ba":
        lo, hi = _range("m_range", (1, 2))
        m_hi = int(hi)
        if num_nodes < 45:  # m >= 2 collapses the diameter below 5 nodes hops
            m_hi = 1
        m_lo = min(int(lo), m_hi)
        return {"m": int(rng.integers(m_lo, m_hi + 1))}

    if graph_type == "ws":
        k_lo, k_hi = _range("k_range", (4, 6))
        p_lo, p_hi = _range("p_range", (0.02, 0.25))
        k_hi = min(int(k_hi), max(2, num_nodes // 4))
        if num_nodes < 45:  # k = 6 already collapses the diameter at N = 20
            k_hi = min(k_hi, 4)
        k_lo = min(int(k_lo), k_hi)
        return {
            "k": int(rng.integers(k_lo, k_hi + 1)),
            "p": float(rng.uniform(p_lo, p_hi)),
        }

    if graph_type == "geometric":
        if "radius_range" in generator_cfg:
            lo, hi = _range("radius_range", (0.16, 0.34))
            return {"radius": float(rng.uniform(lo, hi))}
        a_lo, a_hi = _range("alpha_range", (1.5, 2.0))
        alpha = float(rng.uniform(a_lo, a_hi))
        radius = alpha * geometric_radius_threshold(num_nodes)
        return {"radius": radius, "alpha": alpha}

    if graph_type == "grid":
        # a lattice is fully determined by its side length; keep side^2 as close
        # to the requested node count as an integer square allows
        side = max(3, int(round(math.sqrt(max(num_nodes, 9)))))
        return {"side": side}

    raise ValueError(f"unknown graph_type: {graph_type!r}")


def _build_once(
    graph_type: str, num_nodes: int, params: Dict[str, float], rng: np.random.Generator
) -> nx.Graph:
    fn = _GENERATORS[graph_type]
    if graph_type == "er":
        return fn(num_nodes, params["p"], rng)
    if graph_type == "ba":
        return fn(num_nodes, params["m"], rng)
    if graph_type == "ws":
        return fn(num_nodes, params["k"], params["p"], rng)
    if graph_type == "geometric":
        return fn(num_nodes, params["radius"], rng)
    if graph_type == "grid":
        return fn(num_nodes, params["side"], rng)
    raise ValueError(f"unknown graph_type: {graph_type!r}")


# ----------------------------------------------------------------------------
# OD feasibility
# ----------------------------------------------------------------------------
def graph_admits_distance(
    graph: nx.Graph,
    min_distance: int,
    rng: np.random.Generator,
    probes: Optional[int] = None,
) -> bool:
    """Cheap, sampling-consistent test: is there a node with an eccentricity >= d?

    The OD sampler draws the start node uniformly, so probing a handful of
    uniformly drawn start nodes is a faithful (and much cheaper than computing
    the diameter) feasibility check.
    """
    nodes = list(graph.nodes())
    if probes is None:
        probes = min(len(nodes), 8)
    for start in rng.choice(nodes, size=min(probes, len(nodes)), replace=False):
        distances = nx.single_source_shortest_path_length(graph, int(start))
        if any(d >= min_distance for d in distances.values()):
            return True
    return False


def largest_connected_component(
    graph: nx.Graph, min_fraction: float = 0.7
) -> Optional[nx.Graph]:
    """Largest connected component, relabelled to 0..n-1, or None if too small.

    Erdős-Rényi is the reason this exists.  A connected ER graph needs
    ``c >~ log N``, which for N >= 250 forces a diameter below 5 -- so "same
    family, bigger graph, still d_G(s,g) >= min_od_distance" is *physically*
    unsatisfiable for ER.  Taking the giant component lets the family scale to
    any N while keeping a workable diameter, and is what the connectivity
    requirement is really after.  Node attributes (e.g. ``pos``) survive.
    """
    if nx.is_connected(graph):
        return graph
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if not components or len(components[0]) < min_fraction * graph.number_of_nodes():
        return None
    subgraph = graph.subgraph(components[0]).copy()
    return nx.convert_node_labels_to_integers(subgraph, ordering="sorted")


def generate_connected_graph(
    graph_type: str,
    num_nodes: int,
    rng: np.random.Generator,
    generator_cfg: Optional[Dict[str, Any]] = None,
    max_attempts: int = 200,
    min_od_distance: Optional[int] = None,
    component_fallback: bool = False,
    weighted: bool = False,
    weight_range: Tuple[float, float] = (1.0, 10.0),
    weight_distribution: str = "uniform",
) -> Tuple[nx.Graph, Dict[str, float]]:
    """Generate a connected graph that can host an OD pair of the required length.

    Returns the graph together with the parameters actually used.
    """
    if graph_type not in _GENERATORS:
        raise ValueError(f"unknown graph_type: {graph_type!r}")

    last_params: Dict[str, float] = {}
    for _ in range(max_attempts):
        params = sample_generator_params(graph_type, num_nodes, rng, generator_cfg)
        last_params = params
        graph = _build_once(graph_type, num_nodes, params, rng)
        if graph.number_of_nodes() < 2:
            continue
        if not nx.is_connected(graph):
            if not component_fallback:
                continue
            reduced = largest_connected_component(graph)
            if reduced is None or reduced.number_of_nodes() < 2:
                continue
            graph = reduced
        if weighted:
            graph = attach_edge_weights(
                graph,
                rng,
                weight_range=weight_range,
                from_positions=graph_type in ("geometric", "grid"),
                distribution=weight_distribution,
            )
        if min_od_distance is not None and not graph_admits_distance(
            graph, int(min_od_distance), rng
        ):
            continue
        return graph, params

    raise RuntimeError(
        f"could not generate a usable connected {graph_type} graph with "
        f"{num_nodes} nodes after {max_attempts} attempts (last params={last_params})"
    )


def edge_weight_from_positions(graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    """Euclidean edge weights for geometric graphs (doc 04 section 4.4)."""
    weights: Dict[Tuple[int, int], float] = {}
    for u, v in graph.edges():
        pu = graph.nodes[u]["pos"]
        pv = graph.nodes[v]["pos"]
        weights[(u, v)] = float(math.dist(pu, pv))
    return weights


def attach_edge_weights(
    graph: nx.Graph,
    rng: np.random.Generator,
    weight_range: Tuple[float, float] = (1.0, 10.0),
    from_positions: bool = False,
    distribution: str = "uniform",
) -> nx.Graph:
    """Give every edge a positive cost, turning the task from BFS into Dijkstra.

    This is what makes the Greedy-BFS baseline collapse: with costs drawn
    independently of the topology, "step to the neighbour closer to g in hops"
    no longer produces a minimum-cost route, and the *only* way to answer is to
    actually compare path costs.  ``from_positions`` uses the euclidean edge
    length instead (meaningful for geometric / grid graphs).

    ``distribution``:

    * ``"uniform"``     —— w ~ U(lo, hi)，方案第 10 节的默认实现（[1, 10]）；
    * ``"loguniform"``  —— w = exp(U(log lo, log hi))，跨度更大，weighted 任务明显
      更难（实测 BFS 路径的 cost ratio 从 ~1.03 抬到 ~1.19）。

    权重只乘在边上、不加平移，所以后续的 per-graph mean 归一化不会改变
    ``argmin_P sum w_e``。
    """
    lo, hi = float(weight_range[0]), float(weight_range[1])
    if distribution not in WEIGHT_DISTRIBUTIONS:
        raise ValueError(
            f"unknown weight distribution {distribution!r} "
            f"(supported: {WEIGHT_DISTRIBUTIONS})"
        )
    if lo <= 0.0 or hi < lo:
        raise ValueError(
            f"weight_range must satisfy 0 < lo <= hi, got {(lo, hi)}"
        )
    if from_positions:
        weights = edge_weight_from_positions(graph)
        for edge, value in weights.items():
            # keep weights in the same numerical range as the uniform case
            graph.edges[edge]["weight"] = float(lo + (hi - lo) * value)
    elif distribution == "uniform":
        for u, v in graph.edges():
            graph.edges[u, v]["weight"] = float(rng.uniform(lo, hi))
    else:  # loguniform：跨度更大，weighted 任务更难（bfs_cost_ratio 更高）
        log_lo, log_hi = math.log(lo), math.log(hi)
        for u, v in graph.edges():
            graph.edges[u, v]["weight"] = float(math.exp(rng.uniform(log_lo, log_hi)))
    graph.graph["weighted"] = True
    return graph


def resolve_edge_weight_spec(
    edge_weight_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把 config 的 ``data.edge_weight`` 节翻译成 attach_edge_weights 的 kwargs。

    只在这里做一次校验，生成器、数据集构造、脚本拿到的都是同一份已校验的取值，
    不会出现"配置写错了却被静默忽略"。
    """
    cfg = dict(edge_weight_cfg or {})
    distribution = str(cfg.get("distribution", "uniform")).lower()
    if distribution not in WEIGHT_DISTRIBUTIONS:
        raise ValueError(
            f"data.edge_weight.distribution={distribution!r} is not supported "
            f"(supported: {WEIGHT_DISTRIBUTIONS})"
        )
    bounds = cfg.get("range", (1.0, 10.0)) or (1.0, 10.0)
    lo, hi = float(bounds[0]), float(bounds[1])
    if lo <= 0.0 or hi < lo:
        raise ValueError(
            f"data.edge_weight.range must satisfy 0 < lo <= hi, got {(lo, hi)}"
        )
    return {"weight_range": (lo, hi), "distribution": distribution}
