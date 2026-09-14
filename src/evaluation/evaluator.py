"""Evaluator (实施指南第 24 节).

对每个 query 跑完整 reverse chain，解码，然后统计：

    Goal Hit / Optimal Path / Success Cost Ratio / Loop / Broken / Inference Time

外加一个**只作 debug 用**的 teacher-forced 单步 decision accuracy
（``one_step_accuracy``）——它绝不能用来选模型。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import torch

from src.data.collate import collate_samples, iter_batches
from src.diffusion.categorical import CategoricalDiffusion
from src.diffusion.sampler import sample_reverse_chain
from src.evaluation.metrics import (
    SampleRecord,
    aggregate,
    evaluate_sample,
    format_metrics,
)
from src.evaluation.multi_path_decoder import MultiDecodeResult, decode_multi_path
from src.evaluation.path_decoder import (
    DecodeResult,
    candidate_offsets,
    decode_flat,
    decision_offsets,
)
from src.models.denoiser import GraphFlowDenoiser
from src.training.losses import LossWeights, one_step_clean_state_metrics
from src.training.soft_goal import soft_goal_reachability


@dataclass
class EvaluationReport:
    metrics: Dict[str, float]
    records: List[SampleRecord] = field(default_factory=list)
    debug: Dict[str, float] = field(default_factory=dict)
    #: 仅 ``decode="multi"`` 时非空：三条口径（multi_best / multi_best_goal /
    #: multi_best_goal_cost）各自的 aggregate + 本次搜索的配置与集合语义指标。
    multi: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return format_metrics(self.metrics)


def optimal_coverage(sample, multi: MultiDecodeResult) -> bool:
    """存活路径表里是否至少有一条**真实最优**路径。

    weighted 图按 Dijkstra 的最小 cost 比较；无权图退化成跳数比较（此时
    ``path_cost == 跳数``、``optimal_cost == 最短跳数``，与旧口径**等价**）。
    旧的实现写的是 ``path.cost <= sample.gt_length``，在带权图上是错的
    （跳数最短 != cost 最小）。
    """
    graph = sample.graph
    weight = "weight" if graph.graph.get("weighted", False) else None
    optimal_cost = float(
        nx.shortest_path_length(graph, sample.start, sample.goal, weight=weight)
    )
    return any(
        math.isclose(path.path_cost, optimal_cost, rel_tol=1e-6, abs_tol=1e-6)
        for path in multi.goal_paths
    )


@torch.no_grad()
def evaluate_dataset(
    model: GraphFlowDenoiser,
    diffusion: CategoricalDiffusion,
    dataset,
    batch_size: int = 8,
    stochastic: bool = True,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
    max_steps: Optional[int] = None,
    max_branches: int = 4096,
    progress: bool = True,
    weights: Optional[LossWeights] = None,
    decode: str = "single",
    top_k: int = 2,
    beam_width: int = 64,
    null_policy: str = "stop",
    filter_dead_branches: bool = False,
) -> EvaluationReport:
    """在 dataset 上跑完整评测。

    ``decode``：

    * ``"single"``（默认）：按采样出来的 ``z_0`` 单路径解码，与历史结果口径一致；
    * ``"multi"``：**存活路径表**解码（每个 decision 保留 ``top_k`` 条 branch，
      见 :mod:`src.evaluation.multi_path_decoder`）。``metrics`` 仍然是**历史口径**
      —— 主指标取 ``multi.best``（累计概率最高的那条路径，不管它到不到终点），
      额外报告集合语义的 ``coverage_rate`` / ``optimal_coverage_rate``。

      同一个搜索还会在 ``report.multi`` 里额外给出三条口径的完整指标
      （《Multi-Path Decoder 增强修改指南》第 8 节）：

          multi_best            = multi.best                （历史口径）
          multi_best_goal       = multi.best_goal           （Goal 里概率最高）
          multi_best_goal_cost  = multi.best_goal_cost_path （Goal 里真实 cost 最低）

      ``filter_dead_branches`` 打开时，top-k 之前剔除"终点既不是 Goal 也不是
      decision node"的非 NULL branch（默认关闭 = 历史行为逐位可复现）。
      ``optimal_coverage_rate`` 现在按**真实 cost** 判定：weighted 图上是 Dijkstra
      最小 cost，无权图上等价于原来的跳数口径。
    """
    if decode not in ("single", "multi"):
        raise ValueError(f"decode={decode!r} is not supported (single | multi)")
    model.eval()
    device = torch.device(device)

    records: List[SampleRecord] = []
    # multi 解码的三条口径各自的逐样本记录（顺序与 records 一一对应）
    multi_records: Dict[str, List[SampleRecord]] = {
        "multi_best": [],
        "multi_best_goal": [],
        "multi_best_goal_cost": [],
    }
    debug_accum: Dict[str, float] = {}
    debug_count = 0
    soft_goal_total = 0.0
    soft_goal_graphs = 0
    coverage_hits = 0
    optimal_coverage_hits = 0
    finished_paths_total = 0
    goal_paths_total = 0
    filtered_dead_branches_total = 0
    weighted_seen = False
    start_time = time.time()
    batches = iter_batches(list(dataset), batch_size=batch_size, shuffle=False)

    for batch_index, samples in enumerate(batches):
        batch = collate_samples(samples, device=device)
        batch_start = time.time()
        chain = sample_reverse_chain(
            diffusion,
            model,
            batch,
            generator=generator,
            stochastic=stochastic,
            max_steps=max_steps,
        )
        elapsed = time.time() - batch_start
        z0 = chain["z0"]

        # Soft Goal Reachability：与 Hard Goal Hit 分开报告的可微代理指标。
        # 它是"概率传播到 Goal"的软概率，不是路径解码结果，两者不能互相替代。
        if chain.get("candidate_prob") is not None:
            p_goal = soft_goal_reachability(chain["candidate_prob"], batch)
            soft_goal_total += float(p_goal.sum())
            soft_goal_graphs += int(batch.num_graphs)

        offsets = decision_offsets(samples)
        candidate_starts = candidate_offsets(samples)
        per_sample_time = elapsed / max(len(samples), 1)
        for index, sample in enumerate(samples):
            multi: Optional[MultiDecodeResult] = None
            if decode == "multi":
                local_prob = chain["candidate_prob"][
                    candidate_starts[index] : candidate_starts[index] + sample.num_candidates
                ]
                multi = decode_multi_path(
                    sample,
                    local_prob,
                    top_k=top_k,
                    beam_width=beam_width,
                    null_policy=null_policy,
                    filter_dead_branches=filter_dead_branches,
                )
                best = multi.best
                if best is None:  # pragma: no cover - frontier 非空，理论上不会发生
                    result = DecodeResult("broken", [sample.start], 0, "empty frontier")
                else:
                    result = best.to_decode_result()
                coverage_hits += int(multi.coverage)
                # 集合语义：路径表里至少有一条**真实最优**（weighted = 最小 cost）
                optimal_coverage_hits += int(optimal_coverage(sample, multi))
                finished_paths_total += len(multi.finished)
                goal_paths_total += len(multi.goal_paths)
                filtered_dead_branches_total += int(multi.num_filtered_dead_branches)
                weighted_seen = weighted_seen or bool(getattr(batch, "is_weighted", False))
            else:
                result = decode_flat(
                    sample,
                    z0,
                    decision_offset=offsets[index],
                    candidate_offset=candidate_starts[index],
                    max_branches=max_branches,
                )

            record = evaluate_sample(sample, result, per_sample_time)
            records.append(record)
            if multi is not None:
                # 主口径（multi.best）与 records 共用同一条记录，避免重复算一遍
                multi_records["multi_best"].append(record)
                for label, path in (
                    ("multi_best_goal", multi.best_goal),
                    ("multi_best_goal_cost", multi.best_goal_cost_path),
                ):
                    if path is None:
                        # 表里一条 Goal 路径都没有：这三条口径都算 broken，口径统一
                        multi_records[label].append(
                            evaluate_sample(
                                sample,
                                DecodeResult(
                                    "broken",
                                    [sample.start],
                                    0,
                                    "no goal path in the surviving table",
                                ),
                                per_sample_time,
                            )
                        )
                    else:
                        multi_records[label].append(
                            evaluate_sample(sample, path.to_decode_result(), per_sample_time)
                        )

        # debug metric（teacher-forced 单步），不参与模型选择
        if weights is not None:
            step_metrics = one_step_clean_state_metrics(
                model, diffusion, batch, weights, generator
            )
            for key, value in step_metrics.items():
                debug_accum[key] = debug_accum.get(key, 0.0) + float(value)
            debug_count += 1

        if progress:
            print(
                f"  eval batch {batch_index + 1}/{len(batches)} "
                f"({batch.num_graphs} graphs, {elapsed:.3f}s)",
                flush=True,
            )

    metrics = aggregate(records)
    metrics["wall_time"] = time.time() - start_time
    if soft_goal_graphs:
        metrics["soft_goal_reachability"] = soft_goal_total / soft_goal_graphs
    multi_payload: Dict[str, Any] = {}
    if decode == "multi":
        total = max(len(records), 1)
        metrics["coverage_rate"] = coverage_hits / total
        metrics["optimal_coverage_rate"] = optimal_coverage_hits / total
        metrics["mean_goal_paths"] = goal_paths_total / total
        metrics["mean_finished_paths"] = finished_paths_total / total
        metrics["mean_filtered_dead_branches"] = filtered_dead_branches_total / total
        if weighted_seen:
            # 名字点明"按真实 cost 判定"。无权图上它与 optimal_coverage_rate 是同一件事，
            # 所以只在 weighted 数据集上额外暴露这个别名，旧 JSON 的结构保持不变。
            metrics["weighted_optimal_coverage_rate"] = optimal_coverage_hits / total
        multi_payload = {
            label: aggregate(rows) for label, rows in multi_records.items()
        }
        multi_payload["info"] = {
            "top_k": int(top_k),
            "beam_width": int(beam_width),
            "null_policy": null_policy,
            "filter_dead_branches": bool(filter_dead_branches),
            "dataset_is_weighted": bool(weighted_seen),
            "coverage_rate": metrics["coverage_rate"],
            "optimal_coverage_rate": metrics["optimal_coverage_rate"],
            "mean_goal_paths": metrics["mean_goal_paths"],
            "mean_finished_paths": metrics["mean_finished_paths"],
            "mean_filtered_dead_branches": metrics["mean_filtered_dead_branches"],
        }
    debug = (
        {key: value / max(debug_count, 1) for key, value in debug_accum.items()}
        if debug_count
        else {}
    )
    return EvaluationReport(
        metrics=metrics, records=records, debug=debug, multi=multi_payload
    )


@torch.no_grad()
def one_step_accuracy(
    model: GraphFlowDenoiser,
    diffusion: CategoricalDiffusion,
    dataset,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
    t: Optional[int] = None,
    weights: Optional[LossWeights] = None,
) -> float:
    """teacher-forced 单步 clean-state accuracy（debug metric）。"""
    model.eval()
    device = torch.device(device)
    weights = weights or LossWeights()

    total = 0.0
    count = 0
    for samples in iter_batches(list(dataset), batch_size=batch_size, shuffle=False):
        batch = collate_samples(samples, device=device)
        metrics = one_step_clean_state_metrics(
            model, diffusion, batch, weights, generator, t
        )
        total += metrics["accuracy"] * len(samples)
        count += len(samples)
    return total / max(count, 1)


def records_to_dicts(records: Sequence[SampleRecord]) -> List[Dict[str, Any]]:
    return [record.to_dict() for record in records]
