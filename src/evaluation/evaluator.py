"""Evaluator (实施指南第 24 节).

对每个 query 跑完整 reverse chain，解码，然后统计：

    Goal Hit / Optimal Path / Success Cost Ratio / Loop / Broken / Inference Time

外加一个**只作 debug 用**的 teacher-forced 单步 decision accuracy
（``one_step_accuracy``）——它绝不能用来选模型。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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
from src.evaluation.multi_path_decoder import decode_multi_path
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

    def summary(self) -> str:
        return format_metrics(self.metrics)


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
) -> EvaluationReport:
    """在 dataset 上跑完整评测。

    ``decode``：

    * ``"single"``（默认）：按采样出来的 ``z_0`` 单路径解码，与历史结果口径一致；
    * ``"multi"``：**存活路径表**解码（每个 decision 保留 ``top_k`` 条 branch，
      见 :mod:`src.evaluation.multi_path_decoder`）。主指标取"累计概率最高的那条
      路径"，同时额外报告集合语义的 ``coverage_rate`` / ``optimal_coverage_rate``
      （至少有一条路径到终点 / 至少有一条最短路）。
    """
    if decode not in ("single", "multi"):
        raise ValueError(f"decode={decode!r} is not supported (single | multi)")
    model.eval()
    device = torch.device(device)

    records: List[SampleRecord] = []
    debug_accum: Dict[str, float] = {}
    debug_count = 0
    soft_goal_total = 0.0
    soft_goal_graphs = 0
    coverage_hits = 0
    optimal_coverage_hits = 0
    finished_paths_total = 0
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
                )
                best = multi.best
                if best is None:  # pragma: no cover - frontier 非空，理论上不会发生
                    result = DecodeResult("broken", [sample.start], 0, "empty frontier")
                else:
                    result = best.to_decode_result()
                coverage_hits += int(multi.coverage)
                optimal_coverage_hits += int(
                    any(path.cost <= sample.gt_length for path in multi.goal_paths)
                )
                finished_paths_total += len(multi.finished)
            else:
                result = decode_flat(
                    sample,
                    z0,
                    decision_offset=offsets[index],
                    candidate_offset=candidate_starts[index],
                    max_branches=max_branches,
                )
            records.append(evaluate_sample(sample, result, per_sample_time))

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
    if decode == "multi":
        total = max(len(records), 1)
        metrics["coverage_rate"] = coverage_hits / total
        metrics["optimal_coverage_rate"] = optimal_coverage_hits / total
        metrics["mean_finished_paths"] = finished_paths_total / total
    debug = (
        {key: value / max(debug_count, 1) for key, value in debug_accum.items()}
        if debug_count
        else {}
    )
    return EvaluationReport(metrics=metrics, records=records, debug=debug)


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
