"""Controlled Junction Graph 生成器（《V2 数据集生成指南》第 3-12 节）。

第一版**不**用"先随机生成图、再碰运气筛长路径"，而是先构造有足够 decision
深度的主结构，再添加随机扰动：

    Start / Goal
        ↓
    构造含 K 个 Junction 的主干骨架（S - J1 - J2 - ... - JK - G）
        ↓
    每段骨架边展开成 Branch Segment（中间插入若干 degree=2 普通节点）
        ↓
    加干扰分支：dead-end / detour / loop
        ↓
    重新计算真实 shortest path（新增边可能造出更短路线）
        ↓
    难度过滤（hops / decisions / branch factor）
        ↓
    合格才返回

难度由三个量描述，而不是节点数：

    L_hop       = GT 路径边数
    L_decision  = GT 路径上的 decision 数
    K_avg       = GT 路径上每个 decision 的平均候选 branch 数

难度档（指南第 10 节的表）与结构模式（第 12 节）都在这里采样，但**最终验收契约**
统一按指南第 9.2 节：hops ∈ [15, 35]、decisions ∈ [5, 12]、branch factor ∈ [2, 5]。

关于 Easy 档的一个必要偏离：指南第 10 节的表中 Easy 写的是
``gt_decisions 2-5``，与第 9.2 节的下界 ``>= 5`` 直接冲突。因为过滤规则必须以
第 9.2 节为准，Easy 的 decisions 下界被抬到 5（hops 相应抬到 12），否则 Easy
样本会被 100% 丢弃、20% 的难度配比形同虚设。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# 难度档 / 结构模式
# ---------------------------------------------------------------------------
DIFFICULTY_LEVELS = ("easy", "medium", "hard")
STRUCTURE_MODES = ("branch_heavy", "long_chain", "loop_detour")

# 难度档的实际采样区间。
#
# 与指南第 10 节的表格有两处必要的收窄，原因是表格里的区间与第 9.2 节的验收契约
# 在数学上几乎无交集（每段至少 1 个 ordinary 节点 => hop >= 2*(K+1)）：
#   * Easy 的 decisions 下界抬到 5（与契约的 decisions >= 5 对齐）；
#   * Hard 的 decisions 上界压到 9、hops 压到 [25, 33]：
#     若 decisions >= 10 则 hop >= 22 已经吃掉大半窗口，decisions >= 12 需要
#     hop >= 26..35 且 hops 上界 35，几乎不可满足（实测接受率 0.5%）。
# 三档仍然保持真实的难度梯度（见 dataset_statistics 的 hops/decisions 均值）。
DIFFICULTY_SPECS: Dict[str, Dict[str, Sequence[int]]] = {
    "easy": {
        "num_nodes": (40, 70),
        "gt_hops": (15, 20),
        "gt_decisions": (5, 7),
    },
    "medium": {
        "num_nodes": (60, 120),
        "gt_hops": (15, 30),
        "gt_decisions": (5, 9),
    },
    "hard": {
        "num_nodes": (100, 180),
        "gt_hops": (25, 33),
        "gt_decisions": (6, 9),
    },
}

# 指南第 9.2 节的统一验收契约
ACCEPT_HOPS = (15, 35)
ACCEPT_DECISIONS = (5, 12)
ACCEPT_BRANCH_FACTOR = (2.0, 5.0)
MIN_BRANCHES_PER_DECISION = 2

# 指南第 12 节：模式影响 branch 数量 / 路径长度 / 回环比例
MODE_SPECS: Dict[str, Dict[str, Any]] = {
    "branch_heavy": {
        "candidates_per_decision": (3, 5),
        "distractors_per_junction": (2, 3),
        "dead_end_prob": 0.55,
        "detour_prob": 0.25,
        "loop_prob": 0.20,
        "ordinary_nodes_per_segment": (1, 3),
        "decision_bias": 0.9,           # 目标 decision 数略偏区间下沿
    },
    "long_chain": {
        "candidates_per_decision": (2, 3),
        "distractors_per_junction": (1, 2),
        "dead_end_prob": 0.25,
        "detour_prob": 0.40,
        "loop_prob": 0.35,
        "ordinary_nodes_per_segment": (2, 4),
        "decision_bias": 1.15,          # 目标 decision 数偏区间上沿
    },
    "loop_detour": {
        "candidates_per_decision": (2, 5),
        "distractors_per_junction": (1, 3),
        "dead_end_prob": 0.30,
        "detour_prob": 0.40,
        "loop_prob": 0.30,
        "ordinary_nodes_per_segment": (1, 4),
        "decision_bias": 1.0,
    },
}


@dataclass
class Distractor:
    """一条干扰 branch。"""

    kind: str                                   # dead_end | detour | loop
    from_node: int
    path: list = field(default_factory=list)     # 从 from_node 之后开始的节点序列
    to_node: Optional[int] = None                # detour/loop 的汇合点；dead-end 为 None

    @property
    def num_nodes(self) -> int:
        return len(self.path) + 1                # 含 from_node


@dataclass
class ControlledGraph:
    """生成结果（纯 Python 结构，便于统计与测试）。"""

    graph: nx.Graph
    start: int
    goal: int
    difficulty: str
    mode: str
    target_decisions: int
    skeleton: list                          # 骨架节点（含 start/goal）
    distractors: list                       # list[Distractor]
    metrics: Dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    reject_reason: str = ""

    def describe(self) -> Dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "mode": self.mode,
            "target_decisions": self.target_decisions,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            **{f"gt_{k}" if not k.startswith("gt_") else k: v
               for k, v in self.metrics.items()},
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
            "distractors": _distractor_counts(self.distractors),
        }


def _distractor_counts(distractors: Sequence[Distractor]) -> Dict[str, int]:
    counts = {"dead_end": 0, "detour": 0, "loop": 0}
    for item in distractors:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    low, high = int(low), int(high)
    if high <= low:
        return low
    return int(rng.integers(low, high + 1))


def _biased_decision_target(
    rng: np.random.Generator, low: int, high: int, bias: float
) -> int:
    """在 [low, high] 内取目标 decision 数，``bias`` > 1 偏上沿、< 1 偏下沿。"""
    if high <= low:
        return low
    u = float(rng.random()) ** (1.0 / max(bias, 1e-6))
    return int(round(low + u * (high - low)))


def sample_difficulty(
    rng: np.random.Generator, mix: Optional[Dict[str, float]] = None
) -> str:
    mix = mix or {"easy": 0.20, "medium": 0.60, "hard": 0.20}
    names = [name for name in DIFFICULTY_LEVELS if mix.get(name, 0.0) > 0]
    weights = np.array([float(mix.get(name, 0.0)) for name in names], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(names, p=weights))


def sample_mode(rng: np.random.Generator, mix: Optional[Dict[str, float]] = None) -> str:
    mix = mix or {"branch_heavy": 0.40, "long_chain": 0.40, "loop_detour": 0.20}
    names = [name for name in STRUCTURE_MODES if mix.get(name, 0.0) > 0]
    weights = np.array([float(mix.get(name, 0.0)) for name in names], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(names, p=weights))


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------
class _GraphBuilder:
    """负责分配节点编号并记录骨架 / 干扰分支。"""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._next = 0

    def new_node(self) -> int:
        node = self._next
        self._next += 1
        self.graph.add_node(node)
        return node

    def chain(self, start: int, num_ordinary: int) -> list:
        """从 ``start`` 接出 ``num_ordinary`` 个普通节点，返回新增节点序列。"""
        nodes = []
        current = start
        for _ in range(int(num_ordinary)):
            nxt = self.new_node()
            self.graph.add_edge(current, nxt)
            nodes.append(nxt)
            current = nxt
        return nodes


def _segment_hop_budget(
    target_hops: int, num_segments: int, forced_len: int, bounds: Tuple[int, int]
) -> Optional[list]:
    """把一个 hop 预算拆成每段的 ordinary 节点数（每段 hop = 1 + ordinary）。

    返回 None 表示这个 (K, target_hops) 组合不可行 —— 调用方应当换一个 K 或 hop
    目标，而不是硬凑出一个自相矛盾的样本。

    ``ordinary`` 只约束**骨架 junction 之间**的段：被迫段（``forced_len``，边数，
    没有 ordinary 节点）是额外开销，会先从预算里扣掉；它的长度由调用方在
    [1, 3] 内单独采样。反过来要求被迫段也落在 ``bounds`` 里会让大量 (K, hops)
    组合变成不可行。
    """
    lo, hi = max(1, int(bounds[0])), max(1, int(bounds[1]))
    budget = int(target_hops) - int(forced_len) - int(num_segments)
    if budget < lo * num_segments or budget > hi * num_segments:
        return None
    base = budget // num_segments
    remainder = budget - base * num_segments
    counts = [base] * num_segments
    for index in range(remainder):
        counts[index] += 1
    return counts


def _build_candidate(
    rng: np.random.Generator,
    difficulty: str,
    mode: str,
    target_decisions: int,
    target_hops: int,
    forced_source: bool,
    ordinary_bounds: Tuple[int, int],
    candidates_bounds: Optional[Tuple[int, int]] = None,
) -> Optional[ControlledGraph]:
    """按给定的 (K, target_hops, forced_source) 构造一张图；不可行时返回 None。"""
    builder = _GraphBuilder(nx.Graph())
    start = builder.new_node()

    forced_len = 0
    if forced_source:
        # deg(s) == 1：start 只接一段被迫段，而这段 chain 的**末端本身就是 J1**。
        # 关键：被迫段用完的 chain 末端必须就是 structural[1]，不能另建一套
        # skeleton 节点，否则被迫段会变成孤立的一条链（曾经真的产生过不连通图）。
        forced_len = _randint(rng, 2, 4)
        chain = builder.chain(start, forced_len)
        real_j1 = chain[-1]
        rest = [builder.new_node() for _ in range(target_decisions)]  # J2..GK(=goal)
        structural = [start, real_j1, *rest]
    else:
        rest = [builder.new_node() for _ in range(target_decisions + 1)]  # J1..goal
        structural = [start, *rest]

    goal = structural[-1]
    # forced_source=True 时 (start, J1) 已经是那条被迫 chain，不能再建一段；
    # forced_source=False 时 start 自己是 decision，必须包含 (start, J1) 这一段，
    # 否则 start 会变成孤立节点、图不连通。
    segments = list(zip(structural[1:-1], structural[2:])) if forced_source else list(
        zip(structural[:-1], structural[1:])
    )

    per_segment = _segment_hop_budget(
        target_hops, len(segments), forced_len, ordinary_bounds
    )
    if per_segment is None:
        return None

    # 顺序按随机顺序展开，避免 ordinary 节点编号总是"前几段多、后几段少"
    order = rng.permutation(len(segments)).tolist()
    for index in order:
        u, v = segments[index]
        chain = builder.chain(u, per_segment[index])
        builder.graph.add_edge(chain[-1] if chain else u, v)

    if not nx.is_connected(builder.graph):
        # 安全网：骨架必须连通，否则 GT 根本不存在。
        return None

    mode_spec = MODE_SPECS[mode]
    if candidates_bounds is not None:
        mode_spec = dict(mode_spec)
        mode_spec["distractors_per_junction"] = (
            max(1, int(candidates_bounds[0]) - 1),
            max(1, int(candidates_bounds[1]) - 1),
        )
    distractors = _add_distractors(builder, structural, rng, mode_spec)
    distractors += _ensure_junction_degree(builder, structural)
    distractors += _add_goal_dead_end(builder, goal, rng)
    if not forced_source:
        # Source 自己是 decision node：给它 2~3 条出口（加上主干那一跳共 3~4 条），
        # 让它真的有"选哪条 Branch"的自由度，而不是只有一条主干 + 一条支路。
        distractors += _add_source_extra_branches(
            builder, start, rng, _randint(rng, 2, 3)
        )

    result = ControlledGraph(
        graph=builder.graph,
        start=start,
        goal=goal,
        difficulty=difficulty,
        mode=mode,
        target_decisions=target_decisions,
        skeleton=list(structural),
        distractors=distractors,
    )
    result.metrics = compute_metrics(builder.graph, start, goal)
    result.metrics["target_hops"] = int(target_hops)
    return result


def _add_distractors(
    builder: _GraphBuilder,
    structural: Sequence[int],
    rng: np.random.Generator,
    mode_spec: Dict[str, Any],
) -> list:
    """给每个**骨架 junction**加 dead-end / detour / loop 干扰分支。

    只加在 ``structural[1:-1]`` 上：start / goal 的额外分支有专门处理（start 的
    多出口会改变 forced-source 语义，goal 的多出口会改变 goal 度数）。
    每个 junction **至少**加 1 条干扰分支，否则它的度数会停在 2、不构成 decision，
    于是"目标 decision 数"根本达不到。
    """
    graph = builder.graph
    distractors: list = []
    junctions = list(structural[1:-1])
    num_min, num_max = mode_spec["distractors_per_junction"]

    for index, junction in enumerate(junctions):
        dead_p = float(mode_spec["dead_end_prob"])
        detour_p = float(mode_spec["detour_prob"])
        # 每个 junction **至少** 1 条干扰分支，保证它的度数 >= 3、真的构成 decision；
        # 类型按模式概率抽，而不是强制 dead-end（强制会让 dead-end 占到 78%，
        # 而指南第 6 节要的是 30~40% / 30~40% / 20~30%）。
        for _ in range(_randint(rng, int(num_min), int(num_max))):
            roll = float(rng.random())
            if roll < dead_p:
                length = _randint(rng, 1, 4)
                path = builder.chain(junction, length)
                distractors.append(Distractor("dead_end", junction, path))
            elif roll < dead_p + detour_p:
                # detour：绕远路后汇合到下游的某个 junction（必须严格更长）
                downstream = list(structural[index + 2 :])
                if not downstream:
                    continue
                join = downstream[_randint(rng, 0, len(downstream) - 1)]
                skeleton_hops = max(1, structural.index(join) - index)
                extra = skeleton_hops + _randint(rng, 1, 3)
                path = builder.chain(junction, extra)
                graph.add_edge(path[-1] if path else junction, join)
                distractors.append(Distractor("detour", junction, path, join))
            else:
                # loop / cross：连回**至少隔两跳**的下游 junction，形成"向前绕一圈再
                # 折回来"的环（指南第 6.3 节的 J1 -> ... -> J3 -> J1）。
                #
                # 两条硬约束（都是实测踩出来的）：
                #  1. 不能连 start 或更早的 junction —— 那会造出绕过整段骨架的捷径
                #     （start -> helper -> J2 把 GT 从 22 跳压到 17 跳）；
                #  2. 不能连紧邻的下一个 junction —— 那会把这一段骨架短路掉。
                downstream = list(structural[index + 3 : -1])
                if not downstream:
                    continue
                target = downstream[_randint(rng, 0, len(downstream) - 1)]
                length = _randint(rng, 1, 4)
                path = builder.chain(junction, length)
                graph.add_edge(path[-1] if path else junction, target)
                distractors.append(Distractor("loop", junction, path, target))
    return distractors


def _ensure_junction_degree(
    builder: _GraphBuilder, structural: Sequence[int]
) -> list:
    """保证每个骨架 junction 的度数 >= 3（否则它根本不是 decision）。

    用"J - x - y"这种**两跳的纯 dead-end** 兜底：它的终点是度 1 的叶子，
    不会连到任何其它节点，因此**不可能**造出捷径、也不会改变 GT 长度。
    """
    graph = builder.graph
    added: list = []
    for junction in structural[1:-1]:
        if graph.degree(junction) >= 3:
            continue
        path = builder.chain(junction, 2)
        added.append(Distractor("dead_end", junction, path))
    return added


def _add_goal_dead_end(
    builder: _GraphBuilder, goal: int, rng: np.random.Generator
) -> list:
    """Goal 也可以带一条 dead-end（选择它会直接走错）。"""
    if float(rng.random()) > 0.5:
        return []
    path = builder.chain(goal, _randint(rng, 1, 3))
    return [Distractor("dead_end", goal, path)]


def _add_source_extra_branches(
    builder: _GraphBuilder, start: int, rng: np.random.Generator, count: int
) -> list:
    """deg(s) > 1 时给 Source 自己加分支（Source 因此成为 decision node）。"""
    distractors: list = []
    for _ in range(int(count)):
        path = builder.chain(start, _randint(rng, 1, 3))
        distractors.append(Distractor("dead_end", start, path))
    return distractors


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def compute_metrics(
    graph: nx.Graph,
    start: int,
    goal: int,
    branches_per_decision: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """重新求真实 shortest path，并计算难度指标（指南第 8、16 节）。"""
    gt_path = [int(v) for v in nx.shortest_path(graph, int(start), int(goal))]
    junctions = {v for v in graph.nodes() if graph.degree(v) >= 3}

    # GT 路径上的 decision：路径上的 junction，外加度数 > 1 的 start
    decisions_on_path = [v for v in gt_path if v in junctions]
    start_is_decision = graph.degree(start) > 1
    if start_is_decision and start not in decisions_on_path:
        decisions_on_path.insert(0, start)
    if goal in decisions_on_path:
        decisions_on_path.remove(goal)

    if branches_per_decision is None:
        branches = [graph.degree(v) for v in decisions_on_path]
    else:
        branches = list(branches_per_decision)

    branch_factor = float(np.mean(branches)) if branches else 0.0
    return {
        "gt_path": gt_path,
        "gt_hops": len(gt_path) - 1,
        "gt_decisions": len(decisions_on_path),
        "gt_decision_nodes": [int(v) for v in decisions_on_path],
        "avg_branch_factor": branch_factor,
        "num_junctions": len(junctions),
        "start_is_decision": bool(start_is_decision),
    }


def difficulty_filter(
    metrics: Dict[str, Any],
    hops_range: Sequence[int] = ACCEPT_HOPS,
    decisions_range: Sequence[int] = ACCEPT_DECISIONS,
    branch_factor_range: Sequence[float] = ACCEPT_BRANCH_FACTOR,
) -> Tuple[bool, str]:
    """指南第 9.2/9.3 节的难度过滤。返回 (是否合格, 拒绝原因)。"""
    hops = int(metrics["gt_hops"])
    decisions = int(metrics["gt_decisions"])
    factor = float(metrics["avg_branch_factor"])

    if hops < int(hops_range[0]) or hops > int(hops_range[1]):
        return False, f"gt_hops={hops} outside {tuple(hops_range)}"
    if decisions < int(decisions_range[0]) or decisions > int(decisions_range[1]):
        return False, f"gt_decisions={decisions} outside {tuple(decisions_range)}"
    if factor < float(branch_factor_range[0]) or factor > float(branch_factor_range[1]):
        return False, f"avg_branch_factor={factor:.2f} outside {tuple(branch_factor_range)}"
    return True, ""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def generate_controlled_junction_graph(
    rng: np.random.Generator,
    difficulty: Optional[str] = None,
    mode: Optional[str] = None,
    difficulty_mix: Optional[Dict[str, float]] = None,
    structure_mix: Optional[Dict[str, float]] = None,
    forced_source_probability: float = 0.70,
    ordinary_nodes_per_segment: Optional[Sequence[int]] = None,
    candidates_per_decision: Optional[Sequence[int]] = None,
    difficulty_specs: Optional[Dict[str, Dict[str, Sequence[int]]]] = None,
    verify_gt: bool = True,
) -> ControlledGraph:
    """生成一张 Controlled Junction Graph（指南第 4-9 节）。

    流程：采样难度档 / 结构模式 → 选一个可行的 decision 数 K → 由 K 推出 hop 目标
    → 构造骨架 → 展开 segment → 加干扰分支 → **重新求真实 GT** → 难度过滤。

    注意 K 与 hop 目标是**耦合**的：每段至少 1 个 ordinary 节点，所以
    hop >= 2(K+1)。二者各自独立随机采样会造出大量自相矛盾的样本（要么 hop 不够、
    要么 decision 不够），这正是之前接受率只有 3% 的原因。这里改为先枚举若干 K
    候选、只接受"hop 预算在每段 bounds 内可行"的组合。

    ``verify_gt=True`` 时只有满足难度契约的图才会被标为 ``accepted``。
    """
    specs = difficulty_specs or DIFFICULTY_SPECS
    difficulty = difficulty or sample_difficulty(rng, difficulty_mix)
    mode = mode or sample_mode(rng, structure_mix)
    spec = specs[difficulty]
    mode_spec = MODE_SPECS[mode]

    ordinary_bounds = tuple(
        int(value)
        for value in (
            ordinary_nodes_per_segment or mode_spec["ordinary_nodes_per_segment"]
        )
    )
    candidates_bounds = (
        tuple(int(value) for value in candidates_per_decision)
        if candidates_per_decision
        else None
    )
    dec_lo, dec_hi = int(spec["gt_decisions"][0]), int(spec["gt_decisions"][1])
    hop_lo, hop_hi = int(spec["gt_hops"][0]), int(spec["gt_hops"][1])

    # 先按模式偏好采一个目标 K，再在区间内枚举其它候选
    preferred = _biased_decision_target(
        rng, dec_lo, dec_hi, float(mode_spec["decision_bias"])
    )
    candidates_k = [preferred] + [
        value for value in range(dec_lo, dec_hi + 1) if value != preferred
    ]
    rng.shuffle(candidates_k)

    last_reason = "infeasible"
    last_result: Optional[ControlledGraph] = None
    for target_decisions in candidates_k:
        # 每次重试都重新抽 forced_source：否则一旦某个 (K, source 类型) 不可行，
        # 整轮都会因为同一个取值而失败（实测会退化成"永远只出 forced source"）。
        forced_source = float(rng.random()) < float(forced_source_probability)
        # hop 目标必须由 K 推出来：每段 hop = 1 + ordinary，
        # 所以 hops 的可行区间是 [segments*(1+lo), segments*(1+hi)] 加上被迫段。
        # 独立随机抽 hop 会让大量样本自相矛盾（实测接受率只有 ~5%）。
        n_seg = target_decisions + 1 - (1 if forced_source else 0)
        if forced_source:
            # 被迫段长度在 [2, 4] 之间随机，这里用期望值 3 保守估计
            seg_hops = (
                n_seg * (1 + ordinary_bounds[0]) + 3,
                n_seg * (1 + ordinary_bounds[1]) + 3,
            )
        else:
            seg_hops = (
                n_seg * (1 + ordinary_bounds[0]),
                n_seg * (1 + ordinary_bounds[1]),
            )
        lo = max(hop_lo, seg_hops[0])
        hi = min(hop_hi, seg_hops[1])
        if hi < lo:
            continue
        target_hops = _randint(rng, lo, hi)

        result = _build_candidate(
            rng,
            difficulty=difficulty,
            mode=mode,
            target_decisions=target_decisions,
            target_hops=target_hops,
            forced_source=forced_source,
            ordinary_bounds=(ordinary_bounds[0], ordinary_bounds[1]),
            candidates_bounds=candidates_bounds,
        )
        if result is None:
            last_reason = "infeasible: hop budget or connectivity"
            continue
        last_result = result
        metrics = result.metrics
        ok, reason = difficulty_filter(metrics)

        # 额外检查：GT 路径必须真的沿骨架走完（防止 shortcut 抢走主路）。
        # 只看"数量够不够"是不够的：一条 S->helper->S 式的假捷径能让 GT 只有几跳。
        if ok:
            skeleton_junctions = set(result.skeleton[1:-1])
            on_path = skeleton_junctions & set(metrics["gt_path"])
            expected = max(0, target_decisions - 1)
            if len(on_path) < expected:
                ok, reason = False, (
                    f"only {len(on_path)}/{len(skeleton_junctions)} skeleton junctions "
                    "lie on the GT path (a shortcut grabbed the main route)"
                )
        result.accepted = ok
        result.reject_reason = reason
        if ok or not verify_gt:
            return result
        last_reason = reason

    if last_result is not None:
        return last_result
    # 连一个可行的 (K, hops) 组合都没有：返回一个带原因的占位结果，调用方会重采样
    placeholder = ControlledGraph(
        graph=nx.Graph(),
        start=0,
        goal=1,
        difficulty=difficulty,
        mode=mode,
        target_decisions=preferred,
        skeleton=[],
        distractors=[],
    )
    placeholder.accepted = False
    placeholder.reject_reason = last_reason
    return placeholder


def generate_accepted_graph(
    rng: np.random.Generator,
    max_attempts: int = 200,
    **kwargs: Any,
) -> ControlledGraph:
    """反复生成直到拿到一张通过难度过滤的图；失败则抛错（不静默补齐）。"""
    last: Optional[ControlledGraph] = None
    for _ in range(max_attempts):
        candidate = generate_controlled_junction_graph(rng, **kwargs)
        if candidate.accepted:
            return candidate
        last = candidate
    raise RuntimeError(
        "could not generate an accepted controlled junction graph after "
        f"{max_attempts} attempts (last: {last.reject_reason if last else 'n/a'})"
    )


__all__ = [
    "DIFFICULTY_LEVELS",
    "STRUCTURE_MODES",
    "DIFFICULTY_SPECS",
    "MODE_SPECS",
    "ACCEPT_HOPS",
    "ACCEPT_DECISIONS",
    "ACCEPT_BRANCH_FACTOR",
    "ControlledGraph",
    "Distractor",
    "compute_metrics",
    "difficulty_filter",
    "generate_accepted_graph",
    "generate_controlled_junction_graph",
    "sample_difficulty",
    "sample_mode",
]
