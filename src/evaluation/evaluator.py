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
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
from src.evaluation.readout import (
    SINGLE_READOUTS,
    SINGLE_READOUT_ARGMAX,
    single_path_state,
)
from src.evaluation import real_path_metrics as rpm
from src.models.denoiser import GraphFlowDenoiser
from src.training.losses import LossWeights, one_step_clean_state_metrics
from src.training.soft_goal import soft_goal_reachability


#: ``GraphSample.meta['gt_source']`` 取这个值时，GT 是真实观测的历史车辆路径，
#: 才会计算 normalize LCS / paired edge F1 这类"和真实路线比"的指标。
OBSERVED_GT_SOURCE = "observed"


@dataclass
class EvaluationReport:
    metrics: Dict[str, float]
    records: List[SampleRecord] = field(default_factory=list)
    debug: Dict[str, float] = field(default_factory=dict)
    #: 仅 ``decode="multi"`` 时非空：三条口径（multi_best / multi_best_goal /
    #: multi_best_goal_cost）各自的 aggregate + 本次搜索的配置与集合语义指标。
    multi: Dict[str, Any] = field(default_factory=dict)
    #: 仅当数据集带**真实观测 GT**（``meta['gt_source'] == 'observed'``）时非空。
    #: 结构与旧 JSON 隔离，旧实验的 eval_*.json 一个字段都没被改。
    real: Dict[str, Any] = field(default_factory=dict)

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
    strict_decode: bool = False,
    coordinates: Optional[Mapping[Any, Sequence[float]]] = None,
) -> EvaluationReport:
    """在 dataset 上跑完整评测。

    ``decode``：

    * ``"single"``（默认）：跑完 **stochastic** reverse chain 后，取最后一个 reverse
      step 的 ``candidate_prob`` 做**组内 argmax** 得到 ``z0_argmax``，再单路径解码。
      也就是说：链照旧采样，只有最终 readout 是确定的。
    * ``"single_sampled"``：旧的 single 行为（直接解码采样出来的 ``z0``），只用于诊断；
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

    ``strict_decode``（默认 ``False`` = 上面那套历史口径，逐位可复现）：

      ``True`` 时改走 :mod:`src.evaluation.strict_beam_decoder` 的**三池**语义 ——

          alive      还能继续扩展的路径（全局最多 ``beam_width`` 条）
          success    完整走到 Goal 的路径   ← 最终候选集**只有它**
          discarded  NULL / loop / dead-end / 步数超限，**仅用于统计**

          P* = argmax_{P in success} log P_theta(P)，success 为空才判失败

      与历史口径的本质差别：历史实现把 goal / NULL / loop / dead-end **一起**塞进
      ``finished`` 再统一按累计 log 概率排序，于是"最早被 NULL 打断的残骸"因为负数
      加得少、log 概率反而最大而当选（DiDi test_1000 实测 ``multi.best`` 平均只有
      7.30 个节点，而真正到终点的路径平均 18.49 个节点）。strict 下失败路径直接淘汰，
      永远不会出现在最终候选里。

      同时合法性判定**前置到 top-k 之前**：NULL、会重复经过已访问节点的 branch、
      终点既不是 Goal 也不是 decision node 的 branch 一律 mask，然后在**剩下的合法
      branch**里重新取 top-k。mask 掉一条候选不会让整条路径死掉，只有"一条合法
      branch 都没有"时该路径才进 ``discarded``。

      ``strict=True`` 时 ``null_policy`` 失效（NULL 永远不合法）、
      ``filter_dead_branches`` 恒为真；``report.metrics`` 里的
      ``mean_finished_paths`` 语义变为"平均找到多少条**完整**路线"。
    """
    if decode not in SINGLE_READOUTS + ("multi",):
        raise ValueError(
            f"decode={decode!r} is not supported "
            f"({' | '.join(SINGLE_READOUTS)} | multi)"
        )
    # single 的最终 readout：默认是"最后一个 reverse step 的 candidate probability
    # 组内 argmax"；"single_sampled" 才是旧的"直接解码采样 z0"（诊断用）。
    single_readout = decode if decode in SINGLE_READOUTS else SINGLE_READOUT_ARGMAX
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
    #: 仅 strict 模式非 0：被合法性 mask 剔除的候选 branch / 被淘汰的路径
    masked_null_total = 0
    masked_loop_total = 0
    masked_dead_end_total = 0
    discarded_total = 0
    weighted_seen = False
    # 真实观测 GT（DiDi）才会被填；synthetic 数据集全程为空
    real_records: List[rpm.PathPairRecord] = []
    real_samples: List[Any] = []
    real_skipped = 0
    missing_mapping = 0
    pred_paths: List[List[int]] = []
    gt_paths: List[List[int]] = []
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
        # 采样链本身保持原样（stochastic 由调用方决定）；改的只是 single 的**最终
        # readout**：默认取最终候选概率的组内 argmax，而不是采样出来的 z0。
        z0 = single_path_state(chain, batch, single_readout)

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
                    strict=strict_decode,
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
                masked_null_total += int(multi.num_masked_null)
                masked_loop_total += int(multi.num_masked_loop)
                masked_dead_end_total += int(multi.num_masked_dead_end)
                discarded_total += len(multi.discarded)
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

            # 真实数据指标（方案第 12、17-G 节）：只有 GT 是**观测历史路径**时才
            # 计算。synthetic 数据集 meta 里没有这个标记 -> 整段跳过，旧实验的
            # records / metrics / JSON 结构一个字节都没变。
            gt_source = str(sample.meta.get("gt_source", "shortest"))
            if gt_source == OBSERVED_GT_SOURCE:
                # ⚠️ 跨样本聚合（KLEV / JSEV）必须在**全局 OSM id 空间**里做。
                # 每个样本的 corridor 都被独立 relabel 成 0..N-1，直接混用局部编号
                # 会把"样本 A 的边 (0,1)"和"样本 B 的边 (0,1)"当成同一条城市道路。
                local_to_global = sample.meta.get("local_to_global")
                if not local_to_global:
                    # 缺反查表 -> to_global_path 会退化成恒等映射，KLEV/JSEV 会跑在
                    # **局部编号空间**里（不同样本的 (0,1) 被当成同一条路），结果看着
                    # 正常但完全错。宁可整段不报，也不能报错的数字。
                    missing_mapping += 1
                pred_global = rpm.to_global_path(list(result.path), local_to_global)
                gt_global = rpm.to_global_path(list(sample.gt_path), local_to_global)

                dtw = None
                if coordinates:
                    dtw = rpm.dtw_distance_km(pred_global, gt_global, coordinates)

                paired = rpm.pair_record(
                    list(result.path),
                    list(sample.gt_path),
                    bool(record.goal_hit),
                    gt_cost_ratio=float(sample.meta.get("gt_cost_ratio", float("nan"))),
                    pred_cost_ratio=float(record.cost_ratio),
                    dtw=dtw,
                )
                real_records.append(paired)
                real_samples.append(sample)
                # 分布指标用全局 id
                pred_paths.append(pred_global)
                gt_paths.append(gt_global)
            elif gt_source != "shortest":
                # shuffled OD 之类：GT 是 Dijkstra 占位，绝不能进相似度指标
                real_skipped += 1

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
        if strict_decode:
            # strict 三池语义下才有意义：被 mask 的非法 branch 与被淘汰的路径。
            # 旧口径下这些字段恒为 0，所以只在 strict 时暴露，旧 JSON 结构不变。
            metrics["mean_masked_null"] = masked_null_total / total
            metrics["mean_masked_loop"] = masked_loop_total / total
            metrics["mean_masked_dead_end"] = masked_dead_end_total / total
            metrics["mean_discarded_paths"] = discarded_total / total
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
            "strict": bool(strict_decode),
            "dataset_is_weighted": bool(weighted_seen),
            "coverage_rate": metrics["coverage_rate"],
            "optimal_coverage_rate": metrics["optimal_coverage_rate"],
            "mean_goal_paths": metrics["mean_goal_paths"],
            "mean_finished_paths": metrics["mean_finished_paths"],
            "mean_filtered_dead_branches": metrics["mean_filtered_dead_branches"],
        }
        if strict_decode:
            multi_payload["info"].update(
                {
                    # strict 下 finished == success：mean_finished_paths 就是"找到多少条
                    # 完整路线"；失败路径只进 discarded，不参与最终排名。
                    "mean_success_paths": metrics["mean_finished_paths"],
                    "mean_discarded_paths": metrics["mean_discarded_paths"],
                    "mean_masked_null": metrics["mean_masked_null"],
                    "mean_masked_loop": metrics["mean_masked_loop"],
                    "mean_masked_dead_end": metrics["mean_masked_dead_end"],
                    "best_readout": "argmax_{P in success} log_prob",
                }
            )
    debug = (
        {key: value / max(debug_count, 1) for key, value in debug_accum.items()}
        if debug_count
        else {}
    )

    # ---- 真实数据指标（方案第 12 / 17-G 节）------------------------------
    # 只有存在"观测 GT"样本时才产出。**同时**把主指标里的 path_similarity_score
    # 提到顶层 metrics —— Trainer._maybe_save 是按 record 里的名字选 best.pt 的，
    # 放在 real 里就选不到模型（DiDi 配置 training.selection_metric 用的就是它）。
    real_payload: Dict[str, Any] = {}
    if real_records:
        paired_metrics = rpm.aggregate_pair_records(real_records)
        if missing_mapping:
            print(
                f"[evaluator] WARNING: {missing_mapping} observed sample(s) have no "
                "meta['local_to_global']. KLEV/JSEV can only be computed in the "
                "GLOBAL OSM id space, so they are NOT reported for this run "
                "(computing them on per-sample local ids would silently produce "
                "meaningless numbers). Rebuild the dataset: "
                "python scripts/prepare_didi.py --config <config> --build",
                flush=True,
            )
            distribution: Dict[str, float] = {}
        else:
            distribution = rpm.distribution_metrics(gt_paths, pred_paths)
        real_payload = {
            "metrics": paired_metrics,
            "distribution": distribution,
            "num_paired": len(real_records),
            "num_skipped_placeholder_gt": int(real_skipped),
            "gt_source": OBSERVED_GT_SOURCE,
            "dtw_enabled": bool(coordinates),
            "distribution_node_space": (
                "global_osm_id" if not missing_mapping else "INVALID_missing_local_to_global"
            ),
            "num_missing_local_to_global": int(missing_mapping),
            # 与 records 等长同序；没有观测 GT 的样本是 None，分桶时要跳过
            "records": _align_real_records(dataset, real_samples, real_records),
        }
        metrics["path_similarity_score"] = paired_metrics["path_similarity_score"]
        metrics["normalized_lcs_success"] = paired_metrics["normalized_lcs_success"]
        metrics["edge_f1"] = paired_metrics["edge_f1"]
        metrics["gt_cost_ratio"] = paired_metrics["gt_cost_ratio"]
        metrics["pred_over_gt_cost_ratio"] = paired_metrics["pred_over_gt_cost_ratio"]
        metrics["dtw_km"] = paired_metrics["dtw_km"]
        metrics["dtw_km_success"] = paired_metrics["dtw_km_success"]
        if distribution:
            metrics["klev"] = distribution["klev"]
            metrics["jsev"] = distribution["jsev"]
    elif real_skipped:
        # shuffled OD 集：GT 是 Dijkstra 占位，只报那些不需要真实 GT 的指标
        real_payload = {
            "metrics": {},
            "distribution": {},
            "num_paired": 0,
            "num_skipped_placeholder_gt": int(real_skipped),
            "gt_source": "dijkstra_placeholder",
            "note": (
                "this split has NO real GT path; only Goal Hit / Loop / Broken / "
                "CostRatio / inference time are meaningful"
            ),
        }

    return EvaluationReport(
        metrics=metrics,
        records=records,
        debug=debug,
        multi=multi_payload,
        real=real_payload,
    )


def _align_real_records(
    dataset: Sequence[Any],
    samples_seen: Sequence[Any],
    real_records: Sequence[rpm.PathPairRecord],
) -> List[Optional[Dict[str, Any]]]:
    """把 paired 记录摊回**与 dataset 等长同序**的列表（缺席处为 ``None``）。

    分桶（``length_buckets`` / ``decision_buckets``）是按样本下标取数的，
    所以这里必须对齐；否则桶里会混进别的样本的相似度。
    """
    by_id = {id(sample): record for sample, record in zip(samples_seen, real_records)}
    aligned: List[Optional[Dict[str, Any]]] = []
    for sample in dataset:
        record = by_id.get(id(sample))
        aligned.append(record.to_dict() if record is not None else None)
    return aligned


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
