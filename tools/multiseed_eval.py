"""多随机种子评测 + 逐 query 配对检验（判断"提升是不是真的"）。

为什么需要它：完整 reverse chain 是**随机采样**的（每个 timestep 从 posterior 里采样），
所以同一份 checkpoint、同一批 query，换一个随机种子结果就会差一两个点。300 条测试集上
单次评测的 goal_hit 标准差约 ±0.03，因此"0.893 vs 0.917"这种差距单跑一次根本分不出来。

做法：

1. 对两个 run 的 checkpoint，各跑 K 个随机种子（每个种子跑完整 test split）；
2. 每个 query 得到 K 次成功/失败 -> 取均值，得到该 query 的"成功率"；
3. 在**同一批 query** 上对两个模型的逐 query 成功率做配对 t 检验 / Wilcoxon 符号秩检验
   （scipy），并给出 bootstrap 置信区间；
4. 同时打印"每个种子单跑时两个模型的差值"，让人直接看到种子噪声有多大。

用法::

    python tools/multiseed_eval.py \
        --a outputs/runs/v2_controlled_100ep \
        --b outputs/runs/v2_controlled_100ep_flow3 \
        --data data/controlled/controlled_test.pkl --seeds 0,1,2,3,4 --metric goal_hit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="multi-seed paired evaluation")
    parser.add_argument("--a", required=True, help="基线 run 目录")
    parser.add_argument("--b", required=True, help="新 run 目录")
    parser.add_argument("--checkpoint-a", default=None, help="默认 <a>/best.pt")
    parser.add_argument("--checkpoint-b", default=None, help="默认 <b>/best.pt")
    parser.add_argument("--data", default="data/controlled/controlled_test.pkl")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="逗号分隔的采样种子")
    parser.add_argument(
        "--metric",
        default="goal_hit",
        choices=["goal_hit", "optimal", "broken", "loop"],
    )
    parser.add_argument("--flow-steps-b", type=int, default=None,
                        help="评测 run B 时临时覆盖推理轮数（默认用它的 run_config）")
    parser.add_argument(
        "--decode", default="single", choices=["single", "multi"],
        help="single：按采样 z_0 单路径解码（历史口径）；multi：存活路径表解码"
             "（两个 run 都按同一模式评测，配对检验比的才是同一件事）",
    )
    parser.add_argument("--top-k", type=int, default=2, help="--decode multi 时每个路口的 branch 数")
    parser.add_argument("--beam-width", type=int, default=64, help="--decode multi 时存活路径表上限")
    parser.add_argument("--null-policy", default="stop", choices=["stop", "skip"],
                        help="stop：NULL 参与排名、选中即该路径终止；skip：NULL 不停")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--bucket-data", default=None,
        help="给了就按该 split 的难度 / 决策数 / 结构模式**分组**做多种子配对检验"
             "（用于检验'收益是否集中在长决策链上'这类假设）",
    )
    parser.add_argument(
        "--save-records", default=None,
        help="把逐 seed 的 0/1 命中矩阵存成 json（便于事后换分组/换指标重算）",
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _evaluate_run(
    run_dir: Path,
    checkpoint: Path,
    data_path: str,
    seeds: Sequence[int],
    metric: str,
    device,
    batch_size: int | None,
    flow_steps_override: int | None,
    decode: str = "single",
    top_k: int = 2,
    beam_width: int = 64,
    null_policy: str = "stop",
) -> np.ndarray:
    """返回 [num_seeds, num_queries] 的 0/1 矩阵。"""
    from src.data.dataset import GraphQueryDataset
    from src.evaluation.evaluator import evaluate_dataset
    from src.evaluation.paired import flag
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_diffusion, build_model, get_device
    from src.utils.config import load_config
    from src.utils.seed import make_generator, set_seed

    config = load_config(run_dir / "run_config.json")
    dataset = GraphQueryDataset.load(data_path)
    rows: List[List[float]] = []
    for seed in seeds:
        set_seed(seed)
        model = build_model(config, device)
        payload = load_checkpoint(checkpoint, model=model, map_location=device)
        model = model.to(device)
        if flow_steps_override is not None:
            model.set_inference_flow_steps(
                flow_steps_override, allow_extrapolation=True
            )
        diffusion = build_diffusion(config)
        report = evaluate_dataset(
            model,
            diffusion,
            dataset,
            batch_size=batch_size or int(config.get("evaluation.batch_size", 16)),
            stochastic=bool(config.get("evaluation.stochastic_sampling", True)),
            device=device,
            generator=make_generator(seed, device="cpu"),
            max_steps=int(config.get("evaluation.max_steps", 0)) or None,
            progress=False,
            weights=None,
            decode=decode,
            top_k=top_k,
            beam_width=beam_width,
            null_policy=null_policy,
        )
        rows.append([1.0 if flag(record.to_dict(), metric) else 0.0 for record in report.records])
        print(
            f"  seed {seed}: {metric}={np.mean(rows[-1]):.4f} "
            f"(checkpoint epoch={payload.get('epoch')}, {model.flow_steps_label})",
            flush=True,
        )
        del model, payload
    _ = get_device
    return np.asarray(rows, dtype=float)


def bootstrap_ci(diff: np.ndarray, draws: int, seed: int = 0) -> tuple[float, float]:
    """对逐 query 差值做 bootstrap，返回 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    n = diff.shape[0]
    means = np.array(
        [diff[rng.integers(0, n, n)].mean() for _ in range(draws)], dtype=float
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    args = parse_args()
    from src.training.setup import get_device
    from src.utils.config import load_config

    run_a, run_b = Path(args.a), Path(args.b)
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    device = get_device(args.device)
    checkpoint_a = Path(args.checkpoint_a) if args.checkpoint_a else run_a / "best.pt"
    checkpoint_b = Path(args.checkpoint_b) if args.checkpoint_b else run_b / "best.pt"

    config_a = load_config(run_a / "run_config.json")
    config_b = load_config(run_b / "run_config.json")
    print(
        f"A = {run_a.name}  flow_steps={config_a.get('model.flow_steps', 1)}  "
        f"checkpoint={checkpoint_a.name}"
    )
    print(
        f"B = {run_b.name}  flow_steps={config_b.get('model.flow_steps', 1)}  "
        f"checkpoint={checkpoint_b.name}"
        + (f"  (inference overridden to {args.flow_steps_b})" if args.flow_steps_b else "")
    )
    print(
        f"data = {args.data}   seeds = {seeds}   metric = {args.metric}   "
        f"decode = {args.decode}"
        + (f" (top_k={args.top_k}, beam_width={args.beam_width}, null_policy={args.null_policy})"
           if args.decode == "multi" else "")
        + "\n"
    )

    print("evaluating A:")
    hits_a = _evaluate_run(
        run_a, checkpoint_a, args.data, seeds, args.metric, device,
        args.batch_size, None,
        decode=args.decode, top_k=args.top_k, beam_width=args.beam_width,
        null_policy=args.null_policy,
    )
    print("evaluating B:")
    hits_b = _evaluate_run(
        run_b, checkpoint_b, args.data, seeds, args.metric, device,
        args.batch_size, args.flow_steps_b,
        decode=args.decode, top_k=args.top_k, beam_width=args.beam_width,
        null_policy=args.null_policy,
    )

    per_seed_a = hits_a.mean(axis=1)
    per_seed_b = hits_b.mean(axis=1)
    rate_a, rate_b = float(hits_a.mean()), float(hits_b.mean())
    diff = hits_b.mean(axis=0) - hits_a.mean(axis=0)  # 逐 query 差值

    from scipy import stats

    t_stat, p_value = stats.ttest_rel(hits_b.mean(axis=0), hits_a.mean(axis=0))
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:  # 全为 0
        w_stat, w_p = float("nan"), 1.0
    low, high = bootstrap_ci(diff, args.bootstrap)

    print("\nper-seed rates (same checkpoint, different sampling seeds):")
    for index, seed in enumerate(seeds):
        print(
            f"  seed {seed}: A={per_seed_a[index]:.4f}  B={per_seed_b[index]:.4f}  "
            f"delta={per_seed_b[index] - per_seed_a[index]:+.4f}"
        )
    print(
        f"  single-run delta spread: min={np.min(per_seed_b - per_seed_a):+.4f}  "
        f"max={np.max(per_seed_b - per_seed_a):+.4f}  "
        f"std={np.std(per_seed_b - per_seed_a):.4f}"
    )

    print(f"\n{args.metric}: A={rate_a:.4f}  B={rate_b:.4f}  delta={rate_b - rate_a:+.4f}")
    print(
        f"paired t-test on per-query success rates: t={t_stat:.3f}  p={p_value:.4f}   "
        f"wilcoxon p={w_p:.4f}"
    )
    print(f"bootstrap 95% CI of delta: [{low:+.4f}, {high:+.4f}]")
    verdict = (
        "显著（p<0.05 且 CI 不含 0）"
        if (p_value < 0.05 and (low > 0 or high < 0))
        else "不显著（无法区分于采样噪声）"
    )
    print(f"verdict: {verdict}")

    def paired_stats(mask) -> Dict[str, Any]:
        """在给定 query 子集上做多种子配对检验。"""
        mean_a = hits_a[:, mask].mean(axis=0)
        mean_b = hits_b[:, mask].mean(axis=0)
        subset = mean_b - mean_a
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        t, p = stats.ttest_rel(mean_b, mean_a)
        try:
            _, wp = stats.wilcoxon(subset)
        except ValueError:
            wp = 1.0
        lo, hi = bootstrap_ci(subset, args.bootstrap)
        return {
            "n": n,
            "rate_a": float(hits_a[:, mask].mean()),
            "rate_b": float(hits_b[:, mask].mean()),
            "delta": float(subset.mean()),
            "p_value": float(p),
            "wilcoxon_p": float(wp),
            "bootstrap_ci": [lo, hi],
        }

    buckets: Dict[str, Any] = {}
    if args.bucket_data:
        from src.data.dataset import GraphQueryDataset
        from src.evaluation.buckets import bucket_indices

        dataset = GraphQueryDataset.load(args.bucket_data)
        if len(dataset) != hits_a.shape[1]:
            raise SystemExit(
                f"bucket data has {len(dataset)} queries but eval used {hits_a.shape[1]}"
            )
        print(f"\nper-bucket multi-seed paired test (metric={args.metric}):")
        header = (
            f"{'bucket':<28}{'n':>5}  {'A':>7}{'B':>7}{'delta':>8}{'p':>9}  verdict"
        )
        print(header)
        print("-" * len(header))
        for name, indices in sorted(
            bucket_indices(dataset).items(), key=lambda kv: (kv[0] != "ALL", kv[0])
        ):
            mask = np.zeros(hits_a.shape[1], dtype=bool)
            mask[list(indices)] = True
            stats_row = paired_stats(mask)
            buckets[name] = stats_row
            significant = stats_row["p_value"] < 0.05 and (
                stats_row["bootstrap_ci"][0] > 0 or stats_row["bootstrap_ci"][1] < 0
            )
            print(
                f"{name:<28}{stats_row['n']:>5}  {stats_row['rate_a']:>7.3f}"
                f"{stats_row['rate_b']:>7.3f}{stats_row['delta']:>+8.3f}"
                f"{stats_row['p_value']:>9.4f}  "
                + ("显著" if significant else "-")
            )

    if args.save_records:
        with open(args.save_records, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metric": args.metric,
                    "seeds": seeds,
                    "a": hits_a.tolist(),
                    "b": hits_b.tolist(),
                    "run_a": str(run_a),
                    "run_b": str(run_b),
                },
                handle,
                indent=1,
            )
        print(f"\nper-seed hit matrices written: {args.save_records}")

    if args.out:
        payload: Dict[str, Any] = {
            "a": str(run_a),
            "b": str(run_b),
            "metric": args.metric,
            "seeds": seeds,
            "rate_a": rate_a,
            "rate_b": rate_b,
            "delta": rate_b - rate_a,
            "per_seed_a": per_seed_a.tolist(),
            "per_seed_b": per_seed_b.tolist(),
            "paired_t_p": float(p_value),
            "wilcoxon_p": float(w_p),
            "bootstrap_ci": [low, high],
            "verdict": verdict,
            "buckets": buckets,
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
