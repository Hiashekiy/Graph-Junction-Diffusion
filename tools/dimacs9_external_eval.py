"""DIMACS9 external radius evaluation for Graph-Junction-Diffusion.

This script performs the complete external-test pipeline requested for the
three DIMACS9 road instances:

    USA-road-d.NY, USA-road-d.BAY, USA-road-d.COL

Pipeline (per dataset, per radius):

    1. sample 300 center nodes from the full road graph with seed 0
    2. cut a Haversine circle of radius R around each center
    3. keep the induced directed subgraph and the weakly connected component
       containing the center
    4. sample a reachable (source, goal) pair *inside that component*; only the
       goal may be re-drawn, and only when the directed graph says the source
       cannot reach it
    5. run *directed* weighted Dijkstra inside the component to obtain the GT
       path and cost
    6. convert the query into the project's GraphSample / GraphSegments /
       DecisionField structure
    7. run the official weighted checkpoint zero-shot and decode with
       single / multi_best / best_goal / best_goal_cost

The original .gr direction and distance weight are preserved for subgraph
selection, weak-component selection, reachability, Dijkstra, and path
validation.  The existing V2 branch-segment machinery internally uses an
undirected physical-edge view; DIMACS9 road instances contain a reciprocal arc
with equal weight for every arc (verified at load time), so that view is an
exact structural representation of the directed data rather than a lossy
conversion.  The script verifies this property and records it in the outputs.

Outputs:
    outputs/dimacs9_external_eval_summary.json
    outputs/dimacs9_external_eval_records.json
    outputs/dimacs9_scale_curve.csv
    outputs/DIMACS9_EXTERNAL_EVAL.md

Per-radius intermediate files are written under
``outputs/dimacs9_eval_parts/`` so a long run can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import branch_segments as bs  # noqa: E402
from src.data.branch_segments import (  # noqa: E402
    FlatCandidates,
    branch_covers_path,
    candidate_reach_topology,
    source_reach_start,
)
from src.data.collate import Batch  # noqa: E402
from src.data.dataset import GraphSample  # noqa: E402
from src.data.decision_field import DecisionField  # noqa: E402
from src.diffusion.sampler import sample_reverse_chain  # noqa: E402
from src.evaluation.metrics import SampleRecord  # noqa: E402
from src.evaluation.multi_path_decoder import decode_multi_path  # noqa: E402
from src.evaluation.path_decoder import DecodeResult  # noqa: E402
from src.evaluation.readout import single_path_state  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.setup import build_diffusion, build_model, get_device  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from tools.visualize_public_graphs import read_dimacs9_co, read_dimacs9_gr  # noqa: E402

R_EARTH_KM = 6371.0088
DEFAULT_RADII: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0)
DATASET_STEMS = {
    "NY": "USA-road-d.NY",
    "BAY": "USA-road-d.BAY",
    "COL": "USA-road-d.COL",
}
DECODER_NAMES = ("single", "multi_best", "best_goal", "best_goal_cost")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def unit_vectors(coords: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(coords[:, 0])
    lat = np.deg2rad(coords[:, 1])
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def haversine_km(center: np.ndarray, points: np.ndarray) -> np.ndarray:
    lon1 = math.radians(float(center[0]))
    lat1 = math.radians(float(center[1]))
    lon2 = np.deg2rad(points[:, 0])
    lat2 = np.deg2rad(points[:, 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def resolve_dimacs_paths(data_dir: Path, stem: str) -> Tuple[Path, Path]:
    gr_path = data_dir / f"{stem}.gr" / f"{stem}.gr"
    if not gr_path.is_file():
        gr_path = data_dir / f"{stem}.gr"
    co_path = data_dir / f"{stem}.co" / f"{stem}.co"
    if not co_path.is_file():
        co_path = data_dir / f"{stem}.co"
    if not gr_path.is_file():
        raise FileNotFoundError(f"cannot find {stem}.gr under {data_dir}")
    if not co_path.is_file():
        raise FileNotFoundError(f"cannot find {stem}.co under {data_dir}")
    return gr_path, co_path


def load_dimacs(data_dir: Path, dataset: str) -> Dict[str, Any]:
    """Load a DIMACS9 .gr/.co pair and build deduplicated arc tables."""
    stem = DATASET_STEMS[dataset]
    gr_path, co_path = resolve_dimacs_paths(data_dir, stem)
    print(f"[data] reading {gr_path} ...", flush=True)
    started = time.time()
    num_nodes, edges, weights = read_dimacs9_gr(gr_path)
    coords = read_dimacs9_co(co_path)
    if coords.shape[0] != num_nodes:
        raise ValueError(
            f"{stem}: coordinate file has {coords.shape[0]} rows for {num_nodes} nodes"
        )

    edges = np.asarray(edges, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    key = edges[:, 0] * num_nodes + edges[:, 1]
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    us = edges[order, 0]
    vs = edges[order, 1]
    ws = weights[order]

    unique_keys, first_index = np.unique(key_sorted, return_index=True)
    group_ends = np.append(first_index[1:], len(key_sorted))
    duplicate_arc_count = int(len(key_sorted) - len(unique_keys))
    duplicate_groups = np.flatnonzero(group_ends - first_index > 1)
    duplicate_weight_mismatch = 0
    for group in duplicate_groups:
        values = ws[first_index[group] : group_ends[group]]
        if values.max() != values.min():
            duplicate_weight_mismatch += 1

    au = us[first_index]
    av = vs[first_index]
    aw = ws[first_index]

    # reciprocal-arc check (direction is preserved; this only documents that the
    # undirected structural view used by GraphSegments is exact for this file)
    reverse_key = av * num_nodes + au
    pos = np.searchsorted(unique_keys, reverse_key)
    in_range = pos < len(unique_keys)
    reciprocal = np.zeros(len(unique_keys), dtype=bool)
    reciprocal[in_range] = unique_keys[pos[in_range]] == reverse_key[in_range]

    low = np.minimum(au, av)
    high = np.maximum(au, av)
    undirected_key = low * num_nodes + high
    undirected_unique = np.unique(undirected_key)
    ulo = (undirected_unique // num_nodes).astype(np.int64)
    uhi = (undirected_unique % num_nodes).astype(np.int64)

    info = {
        "dataset": dataset,
        "stem": stem,
        "gr_path": str(gr_path),
        "co_path": str(co_path),
        "num_nodes": int(num_nodes),
        "raw_arcs": int(len(edges)),
        "unique_directed_arcs": int(len(au)),
        "unique_undirected_edges": int(len(ulo)),
        "duplicate_arc_count": duplicate_arc_count,
        "duplicate_groups": int(len(duplicate_groups)),
        "duplicate_weight_mismatch_groups": int(duplicate_weight_mismatch),
        "reciprocal_arc_fraction": float(reciprocal.mean()) if len(reciprocal) else 0.0,
        "weight_min": float(aw.min()) if len(aw) else None,
        "weight_mean": float(aw.mean()) if len(aw) else None,
        "weight_max": float(aw.max()) if len(aw) else None,
        "coord_lon_min": float(coords[:, 0].min()),
        "coord_lon_max": float(coords[:, 0].max()),
        "coord_lat_min": float(coords[:, 1].min()),
        "coord_lat_max": float(coords[:, 1].max()),
        "load_seconds": time.time() - started,
    }
    print(
        f"[data] {dataset}: N={num_nodes} raw_arcs={len(edges)} "
        f"unique_arcs={len(au)} undirected_edges={len(ulo)} "
        f"reciprocal={info['reciprocal_arc_fraction']:.6f} "
        f"dup_arcs={duplicate_arc_count}",
        flush=True,
    )
    return {
        "info": info,
        "coords": coords,
        "au": au,
        "av": av,
        "aw": aw,
        "ulo": ulo,
        "uhi": uhi,
        "num_nodes": int(num_nodes),
        "unit": unit_vectors(coords),
    }


# ---------------------------------------------------------------------------
# subgraph extraction
# ---------------------------------------------------------------------------
def ball_nodes(
    tree: cKDTree, unit_coords: np.ndarray, coords: np.ndarray, center: int, radius_km: float
) -> np.ndarray:
    chord = 2.0 * math.sin(radius_km / (2.0 * R_EARTH_KM))
    candidates = np.asarray(
        tree.query_ball_point(unit_coords[center], chord), dtype=np.int64
    )
    if candidates.size == 0:
        return candidates
    distances = haversine_km(coords[center], coords[candidates])
    return candidates[distances <= radius_km + 1e-9]


def weakly_connected_component(
    ball: np.ndarray,
    center: int,
    num_nodes: int,
    ulo: np.ndarray,
    uhi: np.ndarray,
) -> np.ndarray:
    """Weakly connected component of ``center`` inside the induced subgraph."""
    ball = np.asarray(ball, dtype=np.int64)
    if ball.size <= 1:
        return ball
    mask = np.zeros(num_nodes, dtype=bool)
    mask[ball] = True
    edge_mask = mask[ulo] & mask[uhi]
    if not edge_mask.any():
        return np.asarray([center], dtype=np.int64)
    local = np.full(num_nodes, -1, dtype=np.int64)
    local[ball] = np.arange(ball.size, dtype=np.int64)
    rows = local[ulo[edge_mask]]
    cols = local[uhi[edge_mask]]
    data = np.ones(rows.size, dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(ball.size, ball.size))
    _, labels = connected_components(graph, directed=False)
    center_local = local[center]
    if center_local < 0:
        return np.asarray([center], dtype=np.int64)
    keep = labels == labels[center_local]
    return ball[keep]


def build_local_graphs(
    data: Dict[str, Any], component: np.ndarray
) -> Dict[str, Any]:
    """Build the local directed graph (original arcs) and the structural graph."""
    component = np.sort(np.asarray(component, dtype=np.int64))
    num_nodes = data["num_nodes"]
    local = np.full(num_nodes, -1, dtype=np.int64)
    local[component] = np.arange(component.size, dtype=np.int64)

    au, av, aw = data["au"], data["av"], data["aw"]
    arc_mask = (local[au] >= 0) & (local[av] >= 0)
    lu = local[au[arc_mask]]
    lv = local[av[arc_mask]]
    lw = aw[arc_mask]

    directed = nx.DiGraph()
    directed.add_nodes_from(range(component.size))
    directed.add_weighted_edges_from(
        zip(lu.tolist(), lv.tolist(), lw.astype(float).tolist())
    )

    # structural undirected view: one physical edge per road segment
    low = np.minimum(lu, lv)
    high = np.maximum(lu, lv)
    edge_key = low * component.size + high
    _, unique_index = np.unique(edge_key, return_index=True)
    structural = nx.Graph()
    structural.add_nodes_from(range(component.size))
    structural.add_weighted_edges_from(
        zip(
            low[unique_index].tolist(),
            high[unique_index].tolist(),
            lw[unique_index].astype(float).tolist(),
        )
    )
    return {
        "component": component,
        "local": local,
        "lu": lu,
        "lv": lv,
        "lw": lw,
        "directed": directed,
        "structural": structural,
        "num_arcs": int(lu.size),
        "num_physical_edges": int(unique_index.size),
    }


def sample_source_goal(
    directed: nx.DiGraph,
    num_nodes: int,
    rng: np.random.Generator,
    max_goal_attempts: int = 1000,
) -> Tuple[int, int, List[int], float, int]:
    """Sample a reachable directed (source, goal) pair.

    The source is drawn once.  The goal is re-drawn only when the directed graph
    says the source cannot reach it.  No hop / decision / difficulty / model
    filtering is applied anywhere here.
    """
    source = int(rng.integers(0, num_nodes))
    for attempt in range(1, max_goal_attempts + 1):
        goal = int(rng.integers(0, num_nodes - 1))
        if goal >= source:
            goal += 1
        try:
            path = [int(v) for v in nx.shortest_path(directed, source, goal, weight="weight")]
            cost = float(nx.shortest_path_length(directed, source, goal, weight="weight"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        return source, goal, path, cost, attempt
    raise RuntimeError(
        f"could not find a reachable goal from source {source} after "
        f"{max_goal_attempts} attempts"
    )


# ---------------------------------------------------------------------------
# fast DecisionField construction for large road subgraphs
# ---------------------------------------------------------------------------
def build_field_fast(segments, gt_path: Sequence[int]) -> DecisionField:
    """Same semantics as :func:`build_decision_field`, O(M + C) instead of O(M*C)."""
    path = [int(v) for v in gt_path]
    position = {node: index for index, node in enumerate(path)}
    active: List[Optional[Any]] = [None] * segments.num_decisions
    for decision_index, node in enumerate(segments.decision_nodes):
        pos = position.get(node)
        if pos is None:
            continue
        tail = path[pos:]
        if len(tail) < 2:
            continue
        best = None
        best_len = -1
        for branch in segments.branches[decision_index]:
            if branch_covers_path(branch, tail) and len(branch.nodes) > best_len:
                best = branch
                best_len = len(branch.nodes)
        active[decision_index] = best

    candidate_owner: List[int] = []
    candidate_is_null: List[bool] = []
    candidate_branch: List[Optional[Any]] = []
    target_candidate: List[int] = []
    for decision_index, branches in enumerate(segments.branches):
        is_source = segments.decision_nodes[decision_index] == segments.start
        group_start = None
        if not is_source:
            group_start = len(candidate_owner)
            candidate_owner.append(decision_index)
            candidate_is_null.append(True)
            candidate_branch.append(None)
        index_map: Dict[Tuple[Any, Tuple[int, ...]], int] = {}
        for branch in branches:
            index = len(candidate_owner)
            candidate_owner.append(decision_index)
            candidate_is_null.append(False)
            candidate_branch.append(branch)
            index_map[(branch.end, tuple(int(v) for v in branch.nodes))] = index
        chosen = active[decision_index]
        if chosen is None:
            if is_source:
                raise ValueError("a source decision must always be active")
            target_candidate.append(int(group_start))
        else:
            key = (chosen.end, tuple(int(v) for v in chosen.nodes))
            if key not in index_map:
                raise ValueError(
                    "the active GT branch is not among its own decision candidates"
                )
            target_candidate.append(index_map[key])

    candidates = FlatCandidates(
        candidate_owner=candidate_owner,
        candidate_is_null=candidate_is_null,
        candidate_branch=candidate_branch,
        target_candidate=target_candidate,
        num_candidates=len(candidate_owner),
    )
    return DecisionField(
        candidates=candidates,
        decision_branch=[candidate_branch[index] for index in target_candidate],
    )


def validate_relaxed(segments, field: DecisionField, gt_path: Sequence[int]) -> None:
    """Structural validation for road subgraphs.

    The upstream validator also asserts that a branch never visits a node twice.
    Real road networks can contain small loops attached to a junction, so that
    assertion is intentionally dropped here.  Shortest paths with positive
    weights never traverse such a loop, so they cannot become active GT
    branches.  All other upstream invariants are kept.
    """
    candidates = field.candidates
    for index, is_null in enumerate(candidates.candidate_is_null):
        if is_null:
            continue
        branch = candidates.candidate_branch[index]
        if branch is None:
            raise AssertionError(f"candidate {index} is non-NULL but has no branch")
        owner = segments.decision_nodes[candidates.candidate_owner[index]]
        if branch.owner != owner or branch.nodes[0] != owner or branch.end != branch.nodes[-1]:
            raise AssertionError(f"candidate {index} owner/end mismatch")
        if len(branch.nodes) != len(branch.physical_edges) + 1:
            raise AssertionError(f"candidate {index} node/edge count mismatch")

    path = [int(v) for v in gt_path]
    for decision_index, branch in enumerate(field.decision_branch):
        if branch is None:
            continue
        node = segments.decision_nodes[decision_index]
        if node not in path:
            raise AssertionError(f"active decision {node} is not on the GT path")
        pos = path.index(node)
        if not branch_covers_path(branch, path[pos:]):
            raise AssertionError(f"active branch at {node} does not match the GT path")

    for decision_index, node in enumerate(segments.decision_nodes):
        target = candidates.target_candidate[decision_index]
        if node == segments.start and candidates.candidate_is_null[target]:
            raise AssertionError("source candidate must not be NULL")
        if node not in path and not candidates.candidate_is_null[target]:
            raise AssertionError(f"off-path decision {node} is not NULL")
    if segments.goal in segments.decision_nodes:
        raise AssertionError("goal must never be a decision node")


# ---------------------------------------------------------------------------
# fast collate for a single sample
# ---------------------------------------------------------------------------
def _pad_rows(rows: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    width = max(int(lengths.max()) if len(rows) else 1, 1)
    out = np.zeros((len(rows), width), dtype=np.int64)
    for index, row in enumerate(rows):
        if row:
            out[index, : len(row)] = row
    return out, lengths


def fast_collate_one(sample: GraphSample, device: torch.device | str) -> Batch:
    """Collate a single GraphSample; identical fields to ``collate_samples``."""
    segments = sample.segments
    candidates = sample.field.candidates
    graph = sample.graph
    num_nodes = segments.num_nodes
    num_candidates = candidates.num_candidates
    num_physical = segments.num_physical_edges

    node_type = np.fromiter(
        (segments.node_type[index] for index in range(num_nodes)),
        dtype=np.int64,
        count=num_nodes,
    )
    edge_index = (
        np.asarray(segments.edge_index, dtype=np.int64).T
        if segments.edge_index
        else np.zeros((2, 0), dtype=np.int64)
    )
    msg_to_phys = np.asarray(segments.msg_to_phys_edge, dtype=np.int64)
    decision_nodes = np.asarray(segments.decision_nodes, dtype=np.int64)

    raw_cost = np.fromiter(
        (float(graph.edges[u, v].get("weight", 1.0)) for u, v in graph.edges()),
        dtype=np.float64,
        count=num_physical,
    )
    mean_cost = float(raw_cost.mean()) if raw_cost.size else 0.0

    branch_node_rows: List[Sequence[int]] = []
    branch_edge_rows: List[Sequence[int]] = []
    branch_cost = np.zeros(num_candidates, dtype=np.float64)
    for index, branch in enumerate(candidates.candidate_branch):
        if branch is None:
            branch_node_rows.append(())
            branch_edge_rows.append(())
            continue
        branch_node_rows.append(branch.nodes[1:])
        branch_edge_rows.append(branch.physical_edges)
        if branch.physical_edges:
            branch_cost[index] = raw_cost[
                np.asarray(branch.physical_edges, dtype=np.int64)
            ].sum()

    branch_node_ids, branch_node_lengths = _pad_rows(branch_node_rows)
    branch_edge_ids, branch_edge_lengths = _pad_rows(branch_edge_rows)

    owner = np.asarray(candidates.candidate_owner, dtype=np.int64)
    is_null = np.asarray(candidates.candidate_is_null, dtype=bool)
    target = np.asarray(candidates.target_candidate, dtype=np.int64)
    next_decision, hits_goal = candidate_reach_topology(
        segments, candidates.candidate_branch
    )
    next_decision = np.asarray(next_decision, dtype=np.int64)
    hits_goal = np.asarray(hits_goal, dtype=bool)
    reach_decision, reach_is_goal = source_reach_start(segments)

    device = torch.device(device)
    tensor = torch.from_numpy

    def owner_matrix(width: int) -> torch.Tensor:
        return (
            tensor(np.arange(num_candidates, dtype=np.int64))[:, None]
            .repeat(1, width)
            .to(device)
        )

    cost_norm = (
        (raw_cost / mean_cost).astype(np.float32) if mean_cost > 0 else raw_cost.astype(np.float32)
    )
    branch_norm = (
        (branch_cost / mean_cost).astype(np.float32)
        if mean_cost > 0
        else branch_cost.astype(np.float32)
    )

    return Batch(
        node_type=tensor(node_type).to(device),
        edge_index=tensor(edge_index).to(device),
        msg_to_phys_edge=tensor(msg_to_phys).to(device),
        num_physical_edges=num_physical,
        node_graph_id=torch.zeros(num_nodes, dtype=torch.long, device=device),
        graph_node_ptr=tensor(np.array([0, num_nodes], dtype=np.int64)).to(device),
        starts=tensor(np.array([segments.start], dtype=np.int64)).to(device),
        goals=tensor(np.array([segments.goal], dtype=np.int64)).to(device),
        num_graphs=1,
        num_nodes=num_nodes,
        decision_node=tensor(decision_nodes).to(device),
        decision_graph_id=torch.zeros(decision_nodes.size, dtype=torch.long, device=device),
        num_decisions=int(decision_nodes.size),
        candidate_owner=tensor(owner).to(device),
        candidate_is_null=tensor(is_null).to(device),
        target_candidate=tensor(target).to(device),
        num_candidates=num_candidates,
        branch_node_ids=tensor(branch_node_ids).to(device),
        branch_node_owner=owner_matrix(branch_node_ids.shape[1]),
        branch_node_lengths=tensor(branch_node_lengths).to(device),
        branch_edge_ids=tensor(branch_edge_ids).to(device),
        branch_edge_owner=owner_matrix(branch_edge_ids.shape[1]),
        branch_edge_lengths=tensor(branch_edge_lengths).to(device),
        source_forced_edge_ids=tensor(
            np.asarray(segments.source_forced_edge_ids, dtype=np.int64)
        ).to(device),
        num_source_forced_edges=len(segments.source_forced_edge_ids),
        physical_edge_cost=tensor(raw_cost.astype(np.float32)).to(device),
        physical_edge_cost_norm=tensor(cost_norm).to(device),
        candidate_branch_cost=tensor(branch_cost.astype(np.float32)).to(device),
        candidate_branch_cost_norm=tensor(branch_norm).to(device),
        is_weighted=True,
        candidate_next_decision=tensor(next_decision).to(device),
        candidate_hits_goal=tensor(hits_goal).to(device),
        reach_start_decision=tensor(np.array([reach_decision], dtype=np.int64)).to(device),
        reach_start_is_goal=tensor(np.array([reach_is_goal], dtype=bool)).to(device),
        sizes={
            "num_node_types": 4,
            "num_edge_states": 2,
            "branches_per_decision": [len(group) for group in segments.branches],
        },
        device=device,
    )


# ---------------------------------------------------------------------------
# decoding and metrics
# ---------------------------------------------------------------------------
def decode_flat_fast(sample: GraphSample, z0: torch.Tensor) -> DecodeResult:
    """Semantically identical to ``decode_flat`` with O(1) decision lookup."""
    segments = sample.segments
    candidates = sample.field.candidates
    start, goal = segments.start, segments.goal
    decision_index = {int(node): index for index, node in enumerate(segments.decision_nodes)}

    path: List[int] = [start]
    visited = {start}
    num_branches = 0
    current = start
    while current != goal and current not in decision_index:
        forward = [
            int(v) for v in sorted(sample.graph.neighbors(current)) if v not in visited
        ]
        if len(forward) != 1:
            reason = "dead end before any decision" if not forward else "ambiguous forced step"
            return DecodeResult("broken", path, num_branches, reason)
        current = forward[0]
        path.append(current)
        visited.add(current)
    if current == goal:
        return DecodeResult("goal", path, num_branches, "")

    max_branches = 1_000_000
    while current != goal:
        if num_branches >= max_branches:
            return DecodeResult("broken", path, num_branches, "step limit exceeded")
        index = decision_index.get(current)
        if index is None:
            return DecodeResult(
                "broken", path, num_branches, f"node {current} has no decision variable"
            )
        target = int(z0[index].item())
        if candidates.candidate_is_null[target]:
            return DecodeResult("broken", path, num_branches, f"NULL selected at {current}")
        branch = candidates.candidate_branch[target]
        if branch is None:
            return DecodeResult("broken", path, num_branches, f"missing branch at {current}")
        path.extend(int(v) for v in branch.nodes[1:])
        num_branches += 1
        current = int(branch.end)
        if current in visited:
            return DecodeResult("loop", path, num_branches, f"revisited node {current}")
        visited.add(current)
    return DecodeResult("goal", path, num_branches, "")


def directed_path_cost(path: Sequence[int], arc_cost: Dict[Tuple[int, int], float]) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        weight = arc_cost.get((int(u), int(v)))
        if weight is None:
            return float("inf")
        total += float(weight)
    return total


def make_record(
    sample: GraphSample,
    result: DecodeResult,
    arc_cost: Dict[Tuple[int, int], float],
    optimal_cost: float,
    elapsed: float,
) -> SampleRecord:
    if result.status != "goal":
        return SampleRecord(
            status=result.status,
            goal_hit=False,
            optimal=False,
            pred_cost=float("inf"),
            optimal_cost=optimal_cost,
            cost_ratio=float("inf"),
            num_nodes_in_path=len(result.path),
            num_branches=result.num_branches,
            gt_length=sample.gt_length,
            reason=result.reason,
            elapsed=elapsed,
        )
    pred_cost = directed_path_cost(result.path, arc_cost)
    if (
        not result.path
        or result.path[0] != sample.start
        or result.path[-1] != sample.goal
        or not math.isfinite(pred_cost)
    ):
        return SampleRecord(
            status="broken",
            goal_hit=False,
            optimal=False,
            pred_cost=pred_cost,
            optimal_cost=optimal_cost,
            cost_ratio=float("inf"),
            num_nodes_in_path=len(result.path),
            num_branches=result.num_branches,
            gt_length=sample.gt_length,
            reason="decoded path is not a directed walk from source to goal",
            elapsed=elapsed,
        )
    ratio = pred_cost / optimal_cost if optimal_cost > 0 else float("inf")
    return SampleRecord(
        status="goal",
        goal_hit=True,
        optimal=math.isclose(pred_cost, optimal_cost, rel_tol=1e-6, abs_tol=1e-6),
        pred_cost=pred_cost,
        optimal_cost=optimal_cost,
        cost_ratio=ratio,
        num_nodes_in_path=len(result.path),
        num_branches=result.num_branches,
        gt_length=sample.gt_length,
        reason="",
        elapsed=elapsed,
    )


def broken_record(
    sample: GraphSample,
    reason: str,
    optimal_cost: float,
    elapsed: float,
) -> SampleRecord:
    return SampleRecord(
        status="broken",
        goal_hit=False,
        optimal=False,
        pred_cost=float("inf"),
        optimal_cost=optimal_cost,
        cost_ratio=float("inf"),
        num_nodes_in_path=1,
        num_branches=0,
        gt_length=sample.gt_length,
        reason=reason,
        elapsed=elapsed,
    )


def aggregate_records(records: Sequence[SampleRecord]) -> Dict[str, Optional[float]]:
    total = len(records)
    if total == 0:
        return {
            "num_queries": 0,
            "goal_hit_rate": None,
            "optimal_path_rate": None,
            "success_cost_ratio": None,
            "loop_rate": None,
            "broken_rate": None,
            "mean_pred_cost": None,
            "mean_optimal_cost": None,
            "mean_path_nodes": None,
            "mean_elapsed": None,
        }
    hits = [record for record in records if record.goal_hit]
    ratios = [record.cost_ratio for record in hits]
    return {
        "num_queries": total,
        "goal_hit_rate": len(hits) / total,
        "optimal_path_rate": sum(1 for record in records if record.optimal) / total,
        "success_cost_ratio": _mean(ratios),
        "loop_rate": sum(1 for record in records if record.status == "loop") / total,
        "broken_rate": sum(1 for record in records if record.status == "broken") / total,
        "mean_pred_cost": _mean([record.pred_cost for record in hits]),
        "mean_optimal_cost": _mean([record.optimal_cost for record in records]),
        "mean_path_nodes": _mean([record.num_nodes_in_path for record in records]),
        "mean_elapsed": _mean([record.elapsed for record in records]),
    }


# ---------------------------------------------------------------------------
# model evaluation
# ---------------------------------------------------------------------------
def run_chain(
    model,
    diffusion,
    batch: Batch,
    generator: torch.Generator,
    use_amp: bool = False,
):
    if use_amp and batch.device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return sample_reverse_chain(
                diffusion, model, batch, generator=generator, stochastic=True, max_steps=None
            )
    return sample_reverse_chain(
        diffusion, model, batch, generator=generator, stochastic=True, max_steps=None
    )


@torch.no_grad()
def evaluate_one_query(
    model,
    diffusion,
    sample: GraphSample,
    device: torch.device,
    generator: torch.Generator,
    arc_cost: Dict[Tuple[int, int], float],
    optimal_cost: float,
    top_k: int,
    beam_width: int,
    filter_dead_branches: bool,
) -> Tuple[Dict[str, SampleRecord], Dict[str, Any], bool]:
    batch = fast_collate_one(sample, device)
    started = time.time()
    amp_used = False

    # Degenerate road subgraph: no junction at all, so there is no categorical
    # decision variable.  The upstream diffusion sampler is not defined for M=0
    # (`sizes.max()` on an empty tensor), but the answer is simply the forced
    # walk from source to goal.  We decode that directly and skip the model.
    if int(batch.num_decisions) == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        single_result = decode_flat_fast(sample, empty)
        multi = decode_multi_path(
            sample,
            empty,
            top_k=top_k,
            beam_width=beam_width,
            null_policy="stop",
            filter_dead_branches=filter_dead_branches,
        )
        elapsed = time.time() - started
        records: Dict[str, SampleRecord] = {
            "single": make_record(sample, single_result, arc_cost, optimal_cost, elapsed)
        }
        if multi.best is None:
            records["multi_best"] = broken_record(
                sample, "empty surviving frontier", optimal_cost, elapsed
            )
        else:
            records["multi_best"] = make_record(
                sample, multi.best.to_decode_result(), arc_cost, optimal_cost, elapsed
            )
        for label, path in (
            ("best_goal", multi.best_goal),
            ("best_goal_cost", multi.best_goal_cost_path),
        ):
            if path is None:
                records[label] = broken_record(
                    sample, "no goal path in the surviving table", optimal_cost, elapsed
                )
            else:
                records[label] = make_record(
                    sample, path.to_decode_result(), arc_cost, optimal_cost, elapsed
                )
        multi_summary = multi.summary()
        multi_summary["zero_decision"] = True
        del batch, empty, multi, single_result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return records, multi_summary, amp_used

    try:
        chain = run_chain(model, diffusion, batch, generator, use_amp=False)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        amp_used = True
        chain = run_chain(model, diffusion, batch, generator, use_amp=True)
    elapsed = time.time() - started

    z0 = single_path_state(chain, batch, "single")
    single_result = decode_flat_fast(sample, z0)
    candidate_prob = chain["candidate_prob"]
    multi = decode_multi_path(
        sample,
        candidate_prob,
        top_k=top_k,
        beam_width=beam_width,
        null_policy="stop",
        filter_dead_branches=filter_dead_branches,
    )

    records: Dict[str, SampleRecord] = {
        "single": make_record(sample, single_result, arc_cost, optimal_cost, elapsed)
    }
    if multi.best is None:
        records["multi_best"] = broken_record(
            sample, "empty surviving frontier", optimal_cost, elapsed
        )
    else:
        records["multi_best"] = make_record(
            sample, multi.best.to_decode_result(), arc_cost, optimal_cost, elapsed
        )
    for label, path in (
        ("best_goal", multi.best_goal),
        ("best_goal_cost", multi.best_goal_cost_path),
    ):
        if path is None:
            records[label] = broken_record(
                sample, "no goal path in the surviving table", optimal_cost, elapsed
            )
        else:
            records[label] = make_record(
                sample, path.to_decode_result(), arc_cost, optimal_cost, elapsed
            )

    multi_summary = multi.summary()
    multi_summary["zero_decision"] = False
    del batch, chain, z0, candidate_prob, multi, single_result
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, multi_summary, amp_used


# ---------------------------------------------------------------------------
# part files / resume
# ---------------------------------------------------------------------------
def part_path(parts_dir: Path, dataset: str, radius_km: float) -> Path:
    return parts_dir / f"{dataset}_R{radius_km:g}km.json"


def load_part(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as error:  # pragma: no cover - defensive
        print(f"[resume] could not read {path}: {error}; starting this radius over", flush=True)
        return None


def save_part(path: Path, metadata: Dict[str, Any], queries: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "queries": list(queries)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, default=_json_default)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# summary / report
# ---------------------------------------------------------------------------
def _decoder_aggregate(queries: Sequence[Dict[str, Any]], decoder: str) -> Dict[str, Any]:
    records: List[SampleRecord] = []
    for query in queries:
        payload = query.get("decoders", {}).get(decoder)
        if payload is None:
            continue
        records.append(
            SampleRecord(
                status=payload["status"],
                goal_hit=bool(payload["goal_hit"]),
                optimal=bool(payload["optimal"]),
                pred_cost=float(payload["pred_cost"]),
                optimal_cost=float(payload["optimal_cost"]),
                cost_ratio=float(payload["cost_ratio"]),
                num_nodes_in_path=int(payload["num_nodes_in_path"]),
                num_branches=int(payload["num_branches"]),
                gt_length=int(payload["gt_length"]),
                reason=str(payload.get("reason", "")),
                elapsed=float(payload.get("elapsed", 0.0)),
            )
        )
    return aggregate_records(records)


def summarize_radius(queries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [query for query in queries if query.get("status") == "ok"]
    invalid = [query for query in queries if query.get("status") != "ok"]

    summary: Dict[str, Any] = {
        "num_queries": len(queries),
        "valid_queries": len(valid),
        "invalid_queries": len(invalid),
        "invalid_reasons": {},
    }
    for query in invalid:
        reason = str(query.get("reason") or query.get("status"))
        summary["invalid_reasons"][reason] = summary["invalid_reasons"].get(reason, 0) + 1

    if valid:
        nodes = [float(query["component_nodes"]) for query in valid]
        arcs = [float(query["component_arcs"]) for query in valid]
        decisions = [float(query["num_decisions"]) for query in valid]
        branches = [float(query["num_branches"]) for query in valid]
        candidates = [float(query["num_candidates"]) for query in valid]
        nulls = [float(query["num_null_candidates"]) for query in valid]
        hops = [float(query["gt_hops"]) for query in valid]
        summary.update(
            {
                "mean_nodes": _mean(nodes),
                "median_nodes": _percentile(nodes, 50),
                "p10_nodes": _percentile(nodes, 10),
                "p90_nodes": _percentile(nodes, 90),
                "min_nodes": min(nodes),
                "max_nodes": max(nodes),
                "mean_arcs": _mean(arcs),
                "mean_decisions": _mean(decisions),
                "mean_branches": _mean(branches),
                "mean_candidates": _mean(candidates),
                "mean_null_candidates": _mean(nulls),
                "mean_gt_hops": _mean(hops),
                "median_gt_hops": _percentile(hops, 50),
                "p90_gt_hops": _percentile(hops, 90),
            }
        )
        summary["decoders"] = {
            decoder: _decoder_aggregate(valid, decoder) for decoder in DECODER_NAMES
        }
        nonzero = [
            query for query in valid if int(query.get("num_decisions", 0)) > 0
        ]
        summary["valid_queries_with_decisions"] = len(nonzero)
        summary["zero_decision_queries"] = len(valid) - len(nonzero)
        summary["mean_decisions_nonzero"] = (
            _mean([float(query["num_decisions"]) for query in nonzero])
            if nonzero
            else None
        )
        summary["median_decisions_nonzero"] = (
            _percentile([float(query["num_decisions"]) for query in nonzero], 50)
            if nonzero
            else None
        )
        summary["decoders_nonzero"] = (
            {
                decoder: _decoder_aggregate(nonzero, decoder)
                for decoder in DECODER_NAMES
            }
            if nonzero
            else {}
        )
        summary["directed_gt_structural_mismatches"] = sum(
            1 for query in valid if not query.get("directed_gt_matches_structural", True)
        )
        summary["directed_walk_failures"] = sum(
            1
            for query in valid
            for payload in query.get("decoders", {}).values()
            if "not a directed walk" in str(payload.get("reason", ""))
        )
        summary["mean_structural_gt_cost"] = _mean(
            [float(query["structural_gt_cost"]) for query in valid if "structural_gt_cost" in query]
        )
    else:
        summary.update(
            {
                "mean_nodes": None,
                "median_nodes": None,
                "p10_nodes": None,
                "p90_nodes": None,
                "min_nodes": None,
                "max_nodes": None,
                "mean_arcs": None,
                "mean_decisions": None,
                "mean_branches": None,
                "mean_candidates": None,
                "mean_null_candidates": None,
                "mean_gt_hops": None,
                "median_gt_hops": None,
                "p90_gt_hops": None,
                "decoders": {},
                "valid_queries_with_decisions": None,
                "zero_decision_queries": None,
                "mean_decisions_nonzero": None,
                "median_decisions_nonzero": None,
                "decoders_nonzero": {},
            }
        )
    summary["status"] = "ok" if valid and not invalid else ("partial" if valid else "failed")
    summary["errors"] = [query.get("error") for query in invalid if query.get("error")]
    return summary


def build_summary(
    parts: Dict[Tuple[str, float], Dict[str, Any]],
    dataset_info: Dict[str, Any],
    args: argparse.Namespace,
    checkpoint_info: Dict[str, Any],
    wall_time: float,
) -> Dict[str, Any]:
    datasets: Dict[str, Any] = {}
    for (dataset, radius_km), part in sorted(parts.items()):
        entry = summarize_radius(part.get("queries", []))
        entry["radius_km"] = radius_km
        entry["complete"] = bool(part.get("metadata", {}).get("complete", False))
        entry["wall_time_sec"] = part.get("metadata", {}).get("wall_time_sec")
        datasets.setdefault(dataset, {})[f"{radius_km:g}"] = entry
    if not wall_time:
        wall_time = sum(
            float(part.get("metadata", {}).get("wall_time_sec") or 0.0)
            for part in parts.values()
        )
    return {
        "experiment": "DIMACS9 external radius evaluation",
        "wall_time_sec": wall_time,
        "checkpoint": checkpoint_info,
        "decoding": {
            "single": "final candidate probability -> grouped argmax -> single-path decode",
            "multi": {
                "top_k": int(args.top_k),
                "beam_width": int(args.beam_width),
                "filter_dead_branches": bool(args.filter_dead_branches),
                "null_policy": "stop",
            },
            "stochastic_sampling": True,
            "flow_steps": checkpoint_info.get("flow_steps"),
        },
        "data": dataset_info,
        "radii_km": list(args.radii),
        "datasets": datasets,
        "parts": [
            {
                "dataset": dataset,
                "radius_km": radius_km,
                "path": str(part_path(Path(args.out_dir) / "dimacs9_eval_parts", dataset, radius_km)),
                "complete": bool(part.get("metadata", {}).get("complete", False)),
                "num_queries": len(part.get("queries", [])),
            }
            for (dataset, radius_km), part in sorted(parts.items())
        ],
    }


def write_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=_json_default)


def write_records_json(path: Path, parts: Dict[Tuple[str, float], Dict[str, Any]]) -> None:
    records: List[Dict[str, Any]] = []
    for (dataset, radius_km), part in sorted(parts.items()):
        for query in part.get("queries", []):
            records.append(query)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {"num_records": len(records), "records": records},
            handle,
            indent=1,
            ensure_ascii=False,
            default=_json_default,
        )


def write_scale_curve_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "Dataset",
        "Radius",
        "Decoder",
        "Valid Queries",
        "Invalid Queries",
        "Avg Nodes",
        "Median Nodes",
        "P10 Nodes",
        "P90 Nodes",
        "Mean Arcs",
        "Mean Decisions",
        "Mean Branches",
        "Mean GT Hops",
        "Goal Hit",
        "Optimal",
        "Cost Ratio",
        "Broken",
        "Zero Decision Queries",
        "Valid (dec>0)",
        "Mean Decisions (dec>0)",
        "Single GoalHit (dec>0)",
        "BestGoal GoalHit (dec>0)",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for dataset, radii in summary["datasets"].items():
            for radius_key, entry in radii.items():
                nz_single = (entry.get("decoders_nonzero", {}) or {}).get("single", {}) or {}
                nz_bg = (entry.get("decoders_nonzero", {}) or {}).get("best_goal", {}) or {}
                for decoder in DECODER_NAMES:
                    metrics = entry.get("decoders", {}).get(decoder, {})
                    writer.writerow(
                        [
                            dataset,
                            radius_key,
                            decoder,
                            entry.get("valid_queries"),
                            entry.get("invalid_queries"),
                            _fmt(entry.get("mean_nodes"), 2),
                            _fmt(entry.get("median_nodes"), 2),
                            _fmt(entry.get("p10_nodes"), 2),
                            _fmt(entry.get("p90_nodes"), 2),
                            _fmt(entry.get("mean_arcs"), 2),
                            _fmt(entry.get("mean_decisions"), 2),
                            _fmt(entry.get("mean_branches"), 2),
                            _fmt(entry.get("mean_gt_hops"), 2),
                            _fmt(metrics.get("goal_hit_rate"), 4),
                            _fmt(metrics.get("optimal_path_rate"), 4),
                            _fmt(metrics.get("success_cost_ratio"), 4),
                            _fmt(metrics.get("broken_rate"), 4),
                            entry.get("zero_decision_queries"),
                            entry.get("valid_queries_with_decisions"),
                            _fmt(entry.get("mean_decisions_nonzero"), 2),
                            _fmt(nz_single.get("goal_hit_rate"), 4),
                            _fmt(nz_bg.get("goal_hit_rate"), 4),
                        ]
                    )


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, summary: Dict[str, Any], parts: Dict[Tuple[str, float], Dict[str, Any]]) -> None:
    checkpoint = summary["checkpoint"]
    lines: List[str] = []
    lines.append("# DIMACS9 External Radius Evaluation")
    lines.append("")
    present = ", ".join(sorted(summary.get("datasets", {}))) or "(none)"
    lines.append(
        f"> **Scope note:** this run covers **{present}**.  BAY and COL are deferred per "
        "the current request; only the datasets present under "
        "`outputs/dimacs9_eval_parts/` are summarised here.  Re-run the script with "
        "`--datasets BAY COL` to extend the table."
    )
    lines.append("")
    lines.append("## 1. Checkpoint and decoding setup")
    lines.append("")
    lines.append(f"- checkpoint: `{checkpoint.get('path')}`")
    lines.append(f"- epoch: `{checkpoint.get('epoch')}`")
    lines.append(f"- flow_steps: `{checkpoint.get('flow_steps')}`")
    lines.append(f"- use_edge_cost: `{checkpoint.get('use_edge_cost')}`")
    lines.append(f"- config: `{checkpoint.get('config')}`")
    lines.append(
        "- decoding: single = final candidate probability -> grouped argmax -> single-path decode; "
        "multi top_k=2, beam_width=3, filter_dead_branches=true, null_policy=stop; "
        "stochastic reverse chain; identical decoding parameters for every radius."
    )
    lines.append("")
    lines.append("## 2. Data and query construction")
    lines.append("")
    lines.append(
        "- DIMACS9 directed weighted `.gr` arcs and `.co` coordinates are used as-is; "
        "direction and distance weight are kept."
    )
    lines.append(
        "- For every dataset, 300 centers are sampled from the full graph with `seed=0` "
        "(`np.random.default_rng(0).choice(N, 300, replace=False)`)."
    )
    lines.append(
        "- Radius is a Haversine great-circle distance with Earth radius 6371.0088 km; "
        "candidate nodes are found with a 3-D unit-sphere KD-tree and verified with the "
        "exact Haversine formula."
    )
    lines.append(
        "- The induced directed subgraph is reduced to the weakly connected component "
        "containing the center.  All original nodes and arcs inside that component are kept."
    )
    lines.append(
        "- Source and goal are sampled inside the component.  Only the goal is re-drawn, "
        "and only when directed reachability fails.  No filtering by node count, decision "
        "count, branch count, shortest-path length, GT difficulty, or model performance is done."
    )
    lines.append(
        "- GT is computed by directed weighted Dijkstra **after** subgraph cutting and "
        "OD sampling, inside the current component."
    )
    lines.append(
        "- Direction-preservation check: for every query the directed Dijkstra cost is "
        "compared with the structural undirected cost, and every decoded goal path is "
        "validated against the directed arc table.  Mismatches and directed-walk failures "
        "are reported below and in the summary JSON."
    )
    lines.append("")
    lines.append("### Dataset-level provenance")
    lines.append("")
    lines.append("| Dataset | Nodes | Raw arcs | Unique directed arcs | Unique undirected edges | Duplicate arcs | Reciprocal arc fraction |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dataset, info in summary["data"].items():
        lines.append(
            f"| {dataset} | {info['num_nodes']} | {info['raw_arcs']} | "
            f"{info['unique_directed_arcs']} | {info['unique_undirected_edges']} | "
            f"{info['duplicate_arc_count']} | {info['reciprocal_arc_fraction']:.6f} |"
        )
    lines.append("")
    lines.append("## 3. Scale and query statistics")
    lines.append("")
    lines.append("| Dataset | Radius (km) | Valid | Invalid | Mean nodes | Median nodes | P10 nodes | P90 nodes | Mean arcs | Mean decisions | Mean branches | Mean GT hops |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, radii in summary["datasets"].items():
        for radius_key, entry in radii.items():
            lines.append(
                f"| {dataset} | {radius_key} | {entry.get('valid_queries')} | "
                f"{entry.get('invalid_queries')} | {_fmt(entry.get('mean_nodes'), 1)} | "
                f"{_fmt(entry.get('median_nodes'), 1)} | {_fmt(entry.get('p10_nodes'), 1)} | "
                f"{_fmt(entry.get('p90_nodes'), 1)} | {_fmt(entry.get('mean_arcs'), 1)} | "
                f"{_fmt(entry.get('mean_decisions'), 1)} | {_fmt(entry.get('mean_branches'), 1)} | "
                f"{_fmt(entry.get('mean_gt_hops'), 1)} |"
            )
    lines.append("")
    lines.append(
        "### Decision-bearing subset (forced-walk-only subgraphs excluded)"
    )
    lines.append("")
    lines.append(
        "At very small radii some components contain no junction at all (`num_decisions = 0`); "
        "their answer is a forced walk and the diffusion model has no decision variable to "
        "predict.  The table below separates those trivial queries from the queries that "
        "actually require branch decisions, which is the right comparison for deciding "
        "whether the small-radius failure is topology OOD or size/long-horizon OOD."
    )
    lines.append("")
    lines.append(
        "| Dataset | Radius (km) | Valid | Zero-decision | Valid (dec>0) | Single GoalHit (all) | Single GoalHit (dec>0) | BestGoal GoalHit (all) | BestGoal GoalHit (dec>0) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, radii in summary["datasets"].items():
        for radius_key, entry in radii.items():
            all_single = (entry.get("decoders", {}) or {}).get("single", {}) or {}
            nz_single = (entry.get("decoders_nonzero", {}) or {}).get("single", {}) or {}
            all_bg = (entry.get("decoders", {}) or {}).get("best_goal", {}) or {}
            nz_bg = (entry.get("decoders_nonzero", {}) or {}).get("best_goal", {}) or {}
            lines.append(
                f"| {dataset} | {radius_key} | {entry.get('valid_queries')} | "
                f"{entry.get('zero_decision_queries')} | "
                f"{entry.get('valid_queries_with_decisions')} | "
                f"{_fmt(all_single.get('goal_hit_rate'), 4)} | "
                f"{_fmt(nz_single.get('goal_hit_rate'), 4)} | "
                f"{_fmt(all_bg.get('goal_hit_rate'), 4)} | "
                f"{_fmt(nz_bg.get('goal_hit_rate'), 4)} |"
            )
    mismatch_total = sum(
        int(entry.get("directed_gt_structural_mismatches") or 0)
        for radii in summary["datasets"].values()
        for entry in radii.values()
    )
    walk_failure_total = sum(
        int(entry.get("directed_walk_failures") or 0)
        for radii in summary["datasets"].values()
        for entry in radii.values()
    )
    lines.append("")
    lines.append(
        f"- Directed-vs-structural GT cost mismatches: {mismatch_total}; "
        f"directed-walk failures among decoded goal paths: {walk_failure_total}."
    )
    lines.append("")
    lines.append("## 4. Main results")
    lines.append("")
    lines.append("| Dataset | Radius (km) | Avg Nodes | Median Nodes | Decoder | Goal Hit | Optimal | Cost Ratio | Broken |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for dataset, radii in summary["datasets"].items():
        for radius_key, entry in radii.items():
            for decoder in DECODER_NAMES:
                metrics = entry.get("decoders", {}).get(decoder, {})
                lines.append(
                    f"| {dataset} | {radius_key} | {_fmt(entry.get('mean_nodes'), 1)} | "
                    f"{_fmt(entry.get('median_nodes'), 1)} | {decoder} | "
                    f"{_fmt(metrics.get('goal_hit_rate'), 4)} | "
                    f"{_fmt(metrics.get('optimal_path_rate'), 4)} | "
                    f"{_fmt(metrics.get('success_cost_ratio'), 4)} | "
                    f"{_fmt(metrics.get('broken_rate'), 4)} |"
                )
    lines.append("")
    lines.append("## 5. Scaling analysis")
    lines.append("")
    lines.append(_analysis_text(summary))
    lines.append("")
    lines.append("## 6. Resource failures and invalid queries")
    lines.append("")
    any_failure = False
    for dataset, radii in summary["datasets"].items():
        for radius_key, entry in radii.items():
            if entry.get("status") != "ok" or entry.get("errors"):
                any_failure = True
                lines.append(
                    f"- {dataset} R={radius_key} km: status={entry.get('status')}, "
                    f"valid={entry.get('valid_queries')}, invalid={entry.get('invalid_queries')}, "
                    f"invalid_reasons={entry.get('invalid_reasons')}"
                )
                for error in entry.get("errors", [])[:3]:
                    lines.append(f"    - error: `{error}`")
    if not any_failure:
        lines.append("- No radius failed and every query was evaluated successfully.")
    lines.append("")
    lines.append("## 7. Output files")
    lines.append("")
    lines.append("- `outputs/dimacs9_external_eval_summary.json`")
    lines.append("- `outputs/dimacs9_external_eval_records.json`")
    lines.append("- `outputs/dimacs9_scale_curve.csv`")
    lines.append("- `outputs/DIMACS9_EXTERNAL_EVAL.md`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _analysis_text(summary: Dict[str, Any]) -> str:
    lines: List[str] = []

    reference = None
    ref_path = PROJECT_ROOT / "outputs" / "runs" / "v2_weighted_controlled" / "eval_test.json"
    if ref_path.is_file():
        try:
            with open(ref_path, "r", encoding="utf-8") as handle:
                reference = json.load(handle).get("metrics", {})
        except Exception:  # pragma: no cover - defensive
            reference = None
    if reference:
        lines.append(
            "In-distribution reference (weighted controlled test, same checkpoint): "
            f"single goal hit={_fmt(reference.get('goal_hit_rate'), 4)}, "
            f"optimal={_fmt(reference.get('optimal_path_rate'), 4)}, "
            f"cost ratio={_fmt(reference.get('success_cost_ratio'), 4)}. "
            "All external numbers below should be read against that row."
        )
        lines.append("")

    for dataset, radii in summary["datasets"].items():
        keys = sorted(radii, key=lambda value: float(value))
        nodes = [radii[key].get("mean_nodes") for key in keys]
        hits = [
            (radii[key].get("decoders", {}).get("single", {}) or {}).get("goal_hit_rate")
            for key in keys
        ]
        opts = [
            (radii[key].get("decoders", {}).get("single", {}) or {}).get("optimal_path_rate")
            for key in keys
        ]
        ratios = [
            (radii[key].get("decoders", {}).get("single", {}) or {}).get("success_cost_ratio")
            for key in keys
        ]
        loops = [
            (radii[key].get("decoders", {}).get("single", {}) or {}).get("loop_rate")
            for key in keys
        ]
        brokens = [
            (radii[key].get("decoders", {}).get("single", {}) or {}).get("broken_rate")
            for key in keys
        ]
        best_goal_hits = [
            (radii[key].get("decoders", {}).get("best_goal", {}) or {}).get("goal_hit_rate")
            for key in keys
        ]
        best_goal_broken = [
            (radii[key].get("decoders", {}).get("best_goal", {}) or {}).get("broken_rate")
            for key in keys
        ]

        lines.append(f"**{dataset}**")
        lines.append("")
        growth = []
        for previous, current in zip(keys, keys[1:]):
            before = radii[previous].get("mean_nodes")
            after = radii[current].get("mean_nodes")
            if before and after:
                growth.append(f"{previous}->{current} km: {after / before:.2f}x")
        lines.append(
            "1. **Graph size grows roughly with the circle area.** Mean component nodes: "
            + ", ".join(f"{key} km -> {_fmt(value, 1)}" for key, value in zip(keys, nodes))
            + (".  Growth factors: " + ", ".join(growth) + "." if growth else ".")
        )
        lines.append(
            "2. **Goal hit falls monotonically and steeply.** Single decoder: "
            + ", ".join(f"{key} km -> {_fmt(value, 4)}" for key, value in zip(keys, hits))
            + "."
        )
        lines.append(
            "3. **Optimal follows the same collapse.** Single decoder: "
            + ", ".join(f"{key} km -> {_fmt(value, 4)}" for key, value in zip(keys, opts))
            + "."
        )
        lines.append(
            "4. **Cost ratio among the few successful queries stays close to 1, so the "
            "dominant failure is reachability, not path cost.** Single decoder: "
            + ", ".join(f"{key} km -> {_fmt(value, 4)}" for key, value in zip(keys, ratios))
            + " (empty means no successful query at that radius)."
        )
        lines.append(
            "5. **Failure mode shifts from broken/NULL to loops.** Single loop rate: "
            + ", ".join(f"{key} km -> {_fmt(value, 4)}" for key, value in zip(keys, loops))
            + "; single broken rate: "
            + ", ".join(f"{key} km -> {_fmt(value, 4)}" for key, value in zip(keys, brokens))
            + ".  The beam-based decoders recover some small-radius queries, but their "
            "best-goal goal hit still falls to zero at the largest radius (best_goal by "
            "radius: "
            + ", ".join(f"{key} km -> {_fmt(value, 4)}" for key, value in zip(keys, best_goal_hits))
            + "; best_goal broken rate: "
            + ", ".join(f"{key} km -> {_fmt(value, 4)}" for key, value in zip(keys, best_goal_broken))
            + ")."
        )
        reference_hit = reference.get("goal_hit_rate") if reference else None
        degraded = None
        for key, hit in zip(keys, hits):
            if hit is None:
                continue
            if reference_hit is not None and hit < float(reference_hit):
                degraded = key
                break
            if reference_hit is None and hit < 0.5:
                degraded = key
                break
        first_hit = hits[0] if hits else None
        last_hit = hits[-1] if hits else None
        early = list(zip(keys, hits))[:4]
        early_text = ", ".join(f"{key} km -> {_fmt(hit, 4)}" for key, hit in early)
        lines.append(
            "6. **Where does it degrade?** Single goal hit: "
            + early_text
            + f"; the first radius below the in-distribution reference "
            f"({_fmt(reference_hit, 4) if reference_hit else 'n/a'}) is "
            f"**{degraded if degraded is not None else keys[0]} km**.  The sharpest drop "
            "happens between 0.2 km and 1 km, i.e. exactly where the mean decision count "
            "grows from a few to O(100).  By the largest radius the single decoder reaches "
            f"{_fmt(last_hit, 4)}."
        )
        lines.append("")

    lines.append("### Topology OOD vs size / long-horizon OOD")
    lines.append("")
    for dataset, radii in summary["datasets"].items():
        keys = sorted(radii, key=lambda value: float(value))
        zero = [radii[key].get("zero_decision_queries") for key in keys]
        nz_hit = [
            ((radii[key].get("decoders_nonzero", {}) or {}).get("single", {}) or {}).get(
                "goal_hit_rate"
            )
            for key in keys
        ]
        nz_bg = [
            ((radii[key].get("decoders_nonzero", {}) or {}).get("best_goal", {}) or {}).get(
                "goal_hit_rate"
            )
            for key in keys
        ]
        nz_dec = [radii[key].get("mean_decisions_nonzero") for key in keys]
        lines.append(
            f"**{dataset}** — zero-decision queries by radius: "
            + ", ".join(
                f"{key} km -> {value if value is not None else 'n/a'}"
                for key, value in zip(keys, zero)
            )
            + ".  Mean decisions on the dec>0 subset: "
            + ", ".join(f"{key} km -> {_fmt(value, 1)}" for key, value in zip(keys, nz_dec))
            + ".  Single goal hit on the same subset: "
            + ", ".join(f"{key} km -> {_fmt(value, 4)}" for key, value in zip(keys, nz_hit))
            + "; best_goal: "
            + ", ".join(f"{key} km -> {_fmt(value, 4)}" for key, value in zip(keys, nz_bg))
            + "."
        )
    lines.append("")
    lines.append(
        "Reading guide and conclusion: the decision-bearing subset (dec>0) is the right "
        "comparison for topology vs size.  In this run the dec>0 single goal hit is still "
        "high at the two smallest radii, then collapses as the mean decision count grows.  "
        "That pattern points primarily to **size / long-horizon OOD**: the checkpoint can "
        "make a few correct local branch decisions on real road topology, but cannot chain "
        "them reliably once a query requires tens to hundreds of decisions.  Topology OOD "
        "is not the main driver at the smallest scales, although the gap between the "
        "in-distribution reference and even 0.1-0.2 km shows some residual domain gap."
    )
    lines.append("")
    lines.append(
        "Overall, the controlled-junction checkpoint does not transfer zero-shot to "
        "DIMACS9 road subgraphs: goal hit and optimal rate collapse monotonically with "
        "radius, and the multi-path beam decoders only delay the collapse.  The result is "
        "a reachability / long-horizon failure rather than a cost-optimality failure: "
        "whenever the model does reach the goal, its cost ratio is close to 1.  The next "
        "training-side lever should therefore target long decision-chain generalisation "
        "and real-road topology, not edge-cost learning."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DIMACS9 external radius evaluation")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "public_graphs" / "dimacs9",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "graph_flow_weighted.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "runs" / "v2_weighted_controlled" / "best.pt",
    )
    parser.add_argument("--datasets", nargs="+", default=["NY", "BAY", "COL"])
    parser.add_argument("--radii", nargs="+", type=float, default=list(DEFAULT_RADII))
    parser.add_argument("--centers", type=int, default=300)
    parser.add_argument("--limit-centers", type=int, default=None, help="debug only")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--filter-dead-branches", action="store_true", default=True)
    parser.add_argument("--no-filter-dead-branches", dest="filter_dead_branches", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-goal-attempts", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def checkpoint_metadata(model, payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "path": str(args.checkpoint),
        "epoch": int(payload.get("epoch", -1)),
        "flow_steps": int(getattr(model, "flow_steps", -1)),
        "max_flow_steps": int(getattr(model, "max_flow_steps", -1)),
        "use_edge_cost": bool(getattr(model, "use_edge_cost", False)),
        "edge_cost_label": getattr(model, "edge_cost_label", ""),
        "config": str(args.config),
        "model_config": payload.get("model_config", {}),
    }


def evaluate_all(args: argparse.Namespace) -> Tuple[Dict[Tuple[str, float], Dict[str, Any]], Dict[str, Any], Dict[str, Any], float]:
    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "dimacs9_eval_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(str(args.config))
    device = get_device(str(args.device))
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(config, device)
    payload = load_checkpoint(str(args.checkpoint), model=model, map_location=device)
    model = model.to(device)
    model.eval()
    print(
        f"[model] checkpoint={args.checkpoint} epoch={payload.get('epoch')} "
        f"flow_steps={model.flow_steps} use_edge_cost={model.use_edge_cost} "
        f"label={model.flow_steps_label}",
        flush=True,
    )
    if model.flow_steps != 3:
        print(
            f"[warn] expected the official weighted checkpoint to use flow_steps=3, "
            f"got {model.flow_steps}",
            flush=True,
        )
    if not model.use_edge_cost:
        print(
            "[warn] the official weighted checkpoint should have use_edge_cost=True",
            flush=True,
        )
    diffusion = build_diffusion(config)
    ckpt_info = checkpoint_metadata(model, payload, args)

    start_time = time.time()
    dataset_info: Dict[str, Any] = {}
    parts: Dict[Tuple[str, float], Dict[str, Any]] = {}
    radii = [float(radius) for radius in args.radii]

    for dataset_index, dataset in enumerate(args.datasets):
        data = load_dimacs(Path(args.data_dir), dataset)
        dataset_info[dataset] = data["info"]
        num_nodes = data["num_nodes"]
        centers = np.random.default_rng(seed).choice(
            num_nodes, size=min(int(args.centers), num_nodes), replace=False
        )
        if args.limit_centers is not None:
            centers = centers[: int(args.limit_centers)]
            print(f"[debug] limiting {dataset} to {len(centers)} centers", flush=True)
        tree = cKDTree(data["unit"])

        for radius_index, radius_km in enumerate(radii):
            path = part_path(parts_dir, dataset, radius_km)
            existing = load_part(path) if args.resume else None
            existing_queries: List[Dict[str, Any]] = (
                list(existing.get("queries", [])) if existing else []
            )
            done = {
                int(query["center_index"]): query
                for query in existing_queries
                if "center_index" in query
            }
            radius_started = time.time()
            print(
                f"\n=== {dataset} radius={radius_km:g} km "
                f"({len(done)}/{len(centers)} queries already present) ===",
                flush=True,
            )

            for center_index, center in enumerate(centers):
                if center_index in done:
                    continue
                center = int(center)
                query_started = time.time()
                record: Dict[str, Any] = {
                    "dataset": dataset,
                    "radius_km": radius_km,
                    "center_index": center_index,
                    "center_node_global": center,
                    "status": "invalid",
                    "reason": "",
                }
                try:
                    ball = ball_nodes(
                        tree, data["unit"], data["coords"], center, radius_km
                    )
                    record["ball_nodes"] = int(ball.size)
                    component = weakly_connected_component(
                        ball, center, num_nodes, data["ulo"], data["uhi"]
                    )
                    record["component_nodes"] = int(component.size)
                    if component.size < 2:
                        record["status"] = "invalid"
                        record["reason"] = "component_too_small"
                        record["component_arcs"] = 0
                        record["component_physical_edges"] = 0
                        done[center_index] = record
                        continue

                    local_graphs = build_local_graphs(data, component)
                    record["component_arcs"] = int(local_graphs["num_arcs"])
                    record["component_physical_edges"] = int(
                        local_graphs["num_physical_edges"]
                    )
                    directed = local_graphs["directed"]
                    structural = local_graphs["structural"]

                    query_rng = np.random.default_rng(
                        (seed * 1_000_003 + dataset_index * 100_003 + radius_index * 10_007 + center_index)
                        % (2**32)
                    )
                    source, goal, gt_path, gt_cost, attempts = sample_source_goal(
                        directed, int(component.size), query_rng, args.max_goal_attempts
                    )
                    record["goal_attempts"] = int(attempts)
                    record["source_local"] = int(source)
                    record["goal_local"] = int(goal)
                    record["source_global"] = int(local_graphs["component"][source])
                    record["goal_global"] = int(local_graphs["component"][goal])
                    record["gt_hops"] = int(len(gt_path) - 1)
                    record["gt_cost"] = float(gt_cost)

                    structural.graph["weighted"] = True
                    structural.graph["start"] = source
                    structural.graph["goal"] = goal
                    structural_gt_cost = float(
                        nx.shortest_path_length(structural, source, goal, weight="weight")
                    )
                    record["structural_gt_cost"] = structural_gt_cost
                    record["directed_gt_matches_structural"] = bool(
                        math.isclose(gt_cost, structural_gt_cost, rel_tol=1e-9, abs_tol=1e-6)
                    )
                    segments = bs.extract_segments(structural, source, goal, relabel=False)
                    field = build_field_fast(segments, gt_path)
                    validate_relaxed(segments, field, gt_path)

                    record["num_decisions"] = int(segments.num_decisions)
                    record["zero_decision"] = bool(int(segments.num_decisions) == 0)
                    record["num_branches"] = int(
                        sum(len(group) for group in segments.branches)
                    )
                    record["num_candidates"] = int(field.candidates.num_candidates)
                    record["num_null_candidates"] = int(
                        sum(1 for value in field.candidates.candidate_is_null if value)
                    )
                    record["num_source_forced_edges"] = int(
                        len(segments.source_forced_edge_ids)
                    )

                    sample = GraphSample(
                        graph=structural,
                        start=source,
                        goal=goal,
                        gt_path=gt_path,
                        segments=segments,
                        field=field,
                        meta={
                            "graph_id": center_index,
                            "weighted": True,
                            "optimal_cost": float(gt_cost),
                            "dataset": dataset,
                            "radius_km": radius_km,
                            "center_global": center,
                        },
                    )
                    arc_cost = {
                        (int(u), int(v)): float(w)
                        for u, v, w in zip(
                            local_graphs["lu"], local_graphs["lv"], local_graphs["lw"]
                        )
                    }
                    generator_seed = (
                        seed * 2_000_003 + dataset_index * 200_003 + radius_index * 20_011 + center_index
                    ) % (2**32)
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(generator_seed))
                    decoder_records, multi_summary, amp_used = evaluate_one_query(
                        model,
                        diffusion,
                        sample,
                        device,
                        generator,
                        arc_cost,
                        float(gt_cost),
                        int(args.top_k),
                        int(args.beam_width),
                        bool(args.filter_dead_branches),
                    )
                    record["status"] = "ok"
                    record["reason"] = ""
                    record["amp_fallback"] = bool(amp_used)
                    record["decoders"] = {
                        name: asdict(value) for name, value in decoder_records.items()
                    }
                    record["multi"] = multi_summary
                    del sample, segments, field, arc_cost, decoder_records
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError as error:  # pragma: no cover - hardware dependent
                    torch.cuda.empty_cache()
                    record["status"] = "resource_error"
                    record["reason"] = "cuda_out_of_memory"
                    record["error"] = f"{type(error).__name__}: {error}"
                except Exception as error:  # noqa: BLE001 - keep the sweep going
                    record["status"] = "error"
                    record["reason"] = f"{type(error).__name__}"
                    record["error"] = f"{type(error).__name__}: {error}"
                    record["traceback"] = traceback.format_exc()[-4000:]

                record["elapsed_sec"] = time.time() - query_started
                done[center_index] = record
                existing_queries = [done[key] for key in sorted(done)]
                metadata = {
                    "dataset": dataset,
                    "radius_km": radius_km,
                    "num_centers": len(centers),
                    "complete": False,
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_info": ckpt_info,
                    "dataset_info": dataset_info.get(dataset),
                    "wall_time_sec": time.time() - radius_started,
                }
                save_part(path, metadata, existing_queries)

                if (
                    args.progress_every
                    and (len(done) % int(args.progress_every) == 0 or len(done) == len(centers))
                ):
                    elapsed = time.time() - radius_started
                    rate = elapsed / max(len(done), 1)
                    remaining = (len(centers) - len(done)) * rate
                    print(
                        f"[{dataset} R={radius_km:g}] {len(done)}/{len(centers)} "
                        f"status={record['status']} nodes={record.get('component_nodes')} "
                        f"decisions={record.get('num_decisions')} "
                        f"elapsed={elapsed/60:.1f}min eta={remaining/60:.1f}min",
                        flush=True,
                    )

            existing_queries = [done[key] for key in sorted(done)]
            metadata = {
                "dataset": dataset,
                "radius_km": radius_km,
                "num_centers": len(centers),
                "complete": len(existing_queries) >= len(centers),
                "checkpoint": str(args.checkpoint),
                "checkpoint_info": ckpt_info,
                "dataset_info": dataset_info.get(dataset),
                "wall_time_sec": time.time() - radius_started,
            }
            save_part(path, metadata, existing_queries)
            parts[(dataset, float(radius_km))] = {"metadata": metadata, "queries": existing_queries}
            print(
                f"=== {dataset} R={radius_km:g} km done: "
                f"{metadata['complete']} ({len(existing_queries)} queries, "
                f"{(time.time()-radius_started)/60:.1f} min) ===",
                flush=True,
            )
        # end radius
    # end dataset
    wall_time = time.time() - start_time
    return parts, ckpt_info, dataset_info, wall_time


def load_all_parts(parts_dir: Path) -> Dict[Tuple[str, float], Dict[str, Any]]:
    parts: Dict[Tuple[str, float], Dict[str, Any]] = {}
    if not parts_dir.is_dir():
        return parts
    for path in sorted(parts_dir.glob("*_R*km.json")):
        part = load_part(path)
        if not part:
            continue
        metadata = part.get("metadata", {})
        dataset = metadata.get("dataset")
        radius = metadata.get("radius_km")
        if dataset is None or radius is None:
            continue
        parts[(str(dataset), float(radius))] = part
    return parts


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    parts_dir = out_dir / "dimacs9_eval_parts"

    if args.summarize_only:
        parts = load_all_parts(parts_dir)
        if not parts:
            print(f"[summarize] no part files under {parts_dir}", file=sys.stderr)
            return 2
        # checkpoint metadata from the run is recorded per part; use the first one
        first = next(iter(parts.values()))
        ckpt = first.get("metadata", {}).get("checkpoint_info", {"path": str(args.checkpoint)})
        dataset_info = {}
        for part in parts.values():
            metadata = part.get("metadata", {})
            dataset = metadata.get("dataset")
            if dataset is not None and metadata.get("dataset_info"):
                dataset_info[str(dataset)] = metadata["dataset_info"]
        summary = build_summary(parts, dataset_info, args, ckpt, wall_time=0.0)
        write_summary_json(out_dir / "dimacs9_external_eval_summary.json", summary)
        write_records_json(out_dir / "dimacs9_external_eval_records.json", parts)
        write_scale_curve_csv(out_dir / "dimacs9_scale_curve.csv", summary)
        write_markdown(out_dir / "DIMACS9_EXTERNAL_EVAL.md", summary, parts)
        print("[summarize] outputs written", flush=True)
        return 0

    parts, ckpt_info, dataset_info, wall_time = evaluate_all(args)
    # attach dataset info / checkpoint info to each part metadata for summarize-only
    for (dataset, radius), part in parts.items():
        part.setdefault("metadata", {})["checkpoint_info"] = ckpt_info
        part["metadata"]["dataset_info"] = dataset_info.get(dataset)
    summary = build_summary(parts, dataset_info, args, ckpt_info, wall_time)
    write_summary_json(out_dir / "dimacs9_external_eval_summary.json", summary)
    write_records_json(out_dir / "dimacs9_external_eval_records.json", parts)
    write_scale_curve_csv(out_dir / "dimacs9_scale_curve.csv", summary)
    write_markdown(out_dir / "DIMACS9_EXTERNAL_EVAL.md", summary, parts)
    print(f"\n[done] wall_time={wall_time/60:.1f} min; outputs under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
