"""用**已有的合成图模型**在 DiDi 真实样本上跑一遍，量一下跑不跑得动。

背景：``outputs/runs/v2_weighted_controlled`` 是在 controlled_junction 合成图上
训出来的（每个样本 ~6 个 decision / ~35 个候选），而 DiDi 真实 corridor 是
~350 个 decision / ~430 个节点 / ~1800 个候选 —— 大两个数量级。所以这里要回答的
不是"准不准"，而是：

    现有 checkpoint 在真实样本上**能不能跑完一次完整的 reverse chain**，
    要多久，显存峰值多少。

分阶段计时（每个阶段单独测，避免混在一起看不出瓶颈）：

    collate        把一个 batch 的 GraphSample 转成张量
    reverse chain  T 个 reverse step（每步内部 flow_steps 轮图信息交流）—— 主体开销
    decode + metric 扁平候选解码 + 指标计算
    train step     teacher-forced 整条链的 forward + backward（估算训练时间）

用法::

    python tools/benchmark_didi_inference.py --count 4
    python tools/benchmark_didi_inference.py --count 8 --batch-sizes 1 2 4 --flow-steps 3 1
    python tools/benchmark_didi_inference.py --data data/didi_chengdu_smoke/test.pkl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import collate_samples  # noqa: E402
from src.data.dataset import GraphQueryDataset  # noqa: E402
from src.diffusion.sampler import sample_reverse_chain  # noqa: E402
from src.evaluation.metrics import evaluate_sample  # noqa: E402
from src.evaluation.path_decoder import (  # noqa: E402
    candidate_offsets,
    decode_flat,
    decision_offsets,
)
from src.evaluation.readout import SINGLE_READOUT_ARGMAX, single_path_state  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.losses import LossWeights, recurrent_reverse_loss  # noqa: E402
from src.training.setup import (  # noqa: E402
    build_diffusion,
    build_model,
    build_optimizer,
    get_device,
)
from src.utils.config import flatten_overrides, load_config  # noqa: E402
from src.utils.seed import make_generator, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="benchmark a synthetic-graph model on DiDi samples")
    parser.add_argument("--config", default="configs/graph_flow_weighted.yaml",
                        help="必须与 checkpoint 的架构一致")
    parser.add_argument("--checkpoint", default="outputs/runs/v2_weighted_controlled/best.pt")
    parser.add_argument("--data", default="data/didi_chengdu_gjd/test.pkl")
    parser.add_argument("--count", type=int, default=4, help="用多少条样本")
    parser.add_argument("--offset", type=int, default=0, help="从第几条开始取")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--flow-steps", type=int, nargs="+", default=None,
                        help="推理时的图信息交流轮数（默认用配置里的值）")
    parser.add_argument("--repeat", type=int, default=2, help="每个配置重复几次取中位数")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-train-step", action="store_true",
                        help="跳过 train step（它更慢，只是用来估算训练时间）")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated() / 1e9


def main() -> int:
    args = parse_args()
    config = load_config(args.config, flatten_overrides(args.overrides))
    set_seed(0)
    device = get_device(args.device)
    generator = make_generator(0, device="cpu")

    dataset = GraphQueryDataset.load(PROJECT_ROOT / args.data)
    indices = list(range(args.offset, min(args.offset + args.count, len(dataset))))
    samples = [dataset[i] for i in indices]

    print(f"device       : {device}" + (
        f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"checkpoint   : {args.checkpoint}")
    print(f"dataset      : {args.data}  ({len(dataset)} samples, using {len(samples)})")
    print("sample shapes:")
    for index, sample in zip(indices, samples):
        print(
            f"   #{index:<5d} nodes={sample.num_nodes:<5d} decisions={sample.num_decisions:<5d} "
            f"candidates={sample.num_candidates:<5d} branches={sample.segments.num_physical_edges}"
        )

    model = build_model(config, device)
    payload = load_checkpoint(PROJECT_ROOT / args.checkpoint, model=model, map_location=device)
    model = model.to(device).eval()
    print(f"loaded       : epoch={payload.get('epoch')} best_metric={payload.get('best_metric'):.4f}")
    print(f"model        : {model.flow_steps_label}  params={model.num_parameters():,}")
    diffusion = build_diffusion(config)
    T = diffusion.T
    print(f"T            : {T}")

    # ⚠️ 必须按 Trainer 的方式**从配置**构造 LossWeights：
    # LossWeights() 的默认 goal_horizon_cap 是 None，soft-goal 的 value iteration
    # 会迭代到"每张图自己的 decision 数"（DiDi 是几百上千），在真实数据上比
    # cap=24 慢 4~7 倍。用默认值测出来的是最坏情况，不是实际训练成本。
    weights = LossWeights.from_config(config)
    weights.validate()
    print(f"loss weights : goal_reach_weight={weights.goal_reach_weight} "
          f"goal_horizon_cap={weights.goal_horizon_cap}")
    results: List[Dict[str, Any]] = []

    flow_options = args.flow_steps or [model.flow_steps]
    for flow_steps in flow_options:
        previous = model.set_inference_flow_steps(flow_steps)
        if previous != flow_steps:
            print(f"\nflow_steps   : {previous} -> {flow_steps} (推理期覆盖)")
        for batch_size in args.batch_sizes:
            if batch_size > len(samples):
                continue
            batches = [
                samples[start : start + batch_size]
                for start in range(0, len(samples), batch_size)
            ]
            record: Dict[str, Any] = {
                "flow_steps": flow_steps,
                "batch_size": batch_size,
                "num_batches": len(batches),
            }
            outcome_totals: Dict[str, int] = {}
            cost_ratios: List[float] = []
            try:
                with torch.no_grad():
                    # ---- warmup + 计时 ----
                    timings = {"collate": [], "chain": [], "decode": []}
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    for repeat in range(args.repeat + 1):
                        for chunk in batches:
                            sync(device)
                            t0 = time.perf_counter()
                            batch = collate_samples(chunk, device=device)
                            sync(device)
                            t1 = time.perf_counter()
                            chain = sample_reverse_chain(
                                diffusion, model, batch,
                                generator=generator, stochastic=False, max_steps=T,
                            )
                            sync(device)
                            t2 = time.perf_counter()
                            z0 = single_path_state(chain, batch, SINGLE_READOUT_ARGMAX)
                            offsets = decision_offsets(chunk)
                            starts = candidate_offsets(chunk)
                            outcomes = {"goal": 0, "loop": 0, "broken": 0}
                            gt_ratios = []
                            for index, sample in enumerate(chunk):
                                result = decode_flat(
                                    sample, z0,
                                    decision_offset=offsets[index],
                                    candidate_offset=starts[index],
                                    max_branches=8192,
                                )
                                record_row = evaluate_sample(sample, result, 0.0)
                                outcomes[record_row.status] = (
                                    outcomes.get(record_row.status, 0) + 1
                                )
                                if record_row.goal_hit:
                                    gt_ratios.append(record_row.cost_ratio)
                            sync(device)
                            t3 = time.perf_counter()
                            if repeat > 0:  # 第一次是 warmup，不计入
                                timings["collate"].append(t1 - t0)
                                timings["chain"].append(t2 - t1)
                                timings["decode"].append(t3 - t2)
                            for key, value in outcomes.items():
                                outcome_totals[key] = outcome_totals.get(key, 0) + value
                            cost_ratios.extend(gt_ratios)
                    record["outcomes"] = dict(outcome_totals)
                    record["mean_cost_ratio_success"] = (
                        statistics.fmean(cost_ratios) if cost_ratios else float("nan")
                    )
                    record.update(
                        {
                            "collate_s": statistics.median(timings["collate"]),
                            "chain_s": statistics.median(timings["chain"]),
                            "decode_s": statistics.median(timings["decode"]),
                            "peak_memory_gb": peak_memory_gb(device),
                        }
                    )
                    record["total_s_per_batch"] = (
                        record["collate_s"] + record["chain_s"] + record["decode_s"]
                    )
                    record["per_query_s"] = record["total_s_per_batch"] / batch_size

                # ---- 训练一步（forward + backward）----
                # 必须在 no_grad 外面，否则 loss 不带 grad_fn（第一版就踩了这个）
                # 也**必须预热**：第一次 backward 会触发 cuDNN autotune / 显存池增长，
                # 冷启动一次能比稳态慢 5~10 倍（第一版没预热，把 1.5s 报成了 16s）。
                if not args.no_train_step:
                    optimizer = build_optimizer(config, model)
                    model.train()
                    for _ in range(3):
                        warm = collate_samples(batches[0], device=device)
                        warm_out = recurrent_reverse_loss(
                            model, diffusion, warm, weights=weights,
                            generator=generator, max_steps=T,
                        )
                        warm_out.loss.backward()
                        optimizer.zero_grad(set_to_none=True)
                    sync(device)

                    step_times = []
                    for repeat in range(args.repeat):
                        chunk = batches[repeat % len(batches)]
                        batch = collate_samples(chunk, device=device)
                        sync(device)
                        t0 = time.perf_counter()
                        out = recurrent_reverse_loss(
                            model, diffusion, batch, weights=weights,
                            generator=generator, max_steps=T,
                        )
                        out.loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        sync(device)
                        step_times.append(time.perf_counter() - t0)
                    model.eval()
                    record["train_step_s"] = statistics.median(step_times)
                    record["train_step_min_s"] = min(step_times)
                    record["train_step_per_query_s"] = (
                        record["train_step_s"] / batch_size
                    )
            except RuntimeError as error:
                record["error"] = f"{type(error).__name__}: {error}"[:300]
            results.append(record)

    # ---- 输出 ----
    print("\n" + "=" * 96)
    header = (
        f"{'flow':>5}{'batch':>6}{'collate':>9}{'chain':>9}{'decode':>9}"
        f"{'total/batch':>12}{'s/query':>9}{'peakGB':>8}{'trainstep':>10}"
        f"{'  decode outcome (goal/loop/broken)':>36}"
    )
    print(header)
    print("-" * 96)
    for row in results:
        if "error" in row:
            print(f"{row['flow_steps']:>5}{row['batch_size']:>6}  ERROR: {row['error']}")
            continue
        print(
            f"{row['flow_steps']:>5}{row['batch_size']:>6}"
            f"{row['collate_s']:>9.4f}{row['chain_s']:>9.4f}{row['decode_s']:>9.4f}"
            f"{row['total_s_per_batch']:>12.4f}{row['per_query_s']:>9.4f}"
            f"{row['peak_memory_gb']:>8.2f}"
            f"{row.get('train_step_s', float('nan')):>10.4f}"
            f"   {row['outcomes'].get('goal',0)}/{row['outcomes'].get('loop',0)}"
            f"/{row['outcomes'].get('broken',0)}"
        )
    print("=" * 96)

    out_path = PROJECT_ROOT / "outputs/_didi_build/benchmark_inference.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "data": args.data,
                "T": T,
                "indices": indices,
                "results": results,
            },
            handle,
            indent=1,
            ensure_ascii=False,
        )
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
