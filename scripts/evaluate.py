"""评测入口（实施指南第 24 节）。

用法::

    python scripts/evaluate.py --config configs/controlled_unweighted.yaml \
        --checkpoint outputs/runs/graph_flow/best.pt --data data/unweighted/unweighted_test.pkl

输出主指标：Goal Hit / Optimal Path / Success Cost Ratio / Loop / Broken / 推理时间。
另外会打印一个 **debug 用** 的 teacher-forced 单步 decision accuracy。

``--decode multi`` 时，同一个存活路径表会按《Multi-Path Decoder 增强修改指南》第 8 节
并排输出三条口径（都走同一套 ``evaluate_sample``）：

    multi_best            累计 log 概率最高（历史口径，= 主指标）
    multi_best_goal       Goal 路径里概率最高
    multi_best_goal_cost  Goal 路径里真实 cost 最低

额外的集合语义指标：``coverage_rate`` / ``optimal_coverage_rate``
（weighted 数据集上后者按 Dijkstra 最小 cost 判定，并额外暴露
``weighted_optimal_coverage_rate`` 这个名字）/ ``mean_goal_paths`` /
``mean_filtered_dead_branches``。结果 JSON 里放在 ``multi`` 键下。

**真实数据（DiDi）**：当数据集里的样本带 ``meta['gt_source'] == 'observed'``
（即 GT 是真实车辆历史路径，不是最短路）时，会额外产出并写进 JSON：

    real_path_metrics.metrics       PathSimilarityScore / nLCS / paired Edge F1 /
                                    PredCostRatio / GTCostRatio ...
    real_path_metrics.distribution  KLEV / JSEV（dataset-level）
    buckets.length_buckets          按 GT 长度等量三分（GDP 风格）
    buckets.decision_buckets        按 num_decisions 分桶（长决策链诊断）

这些字段**只在真实数据上出现**，旧实验的 eval_*.json 结构与字段名完全不变。
shuffled OD 集（``meta['no_real_gt']``）没有真实 GT，会自动跳过相似度指标。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import didi_dataset as didi  # noqa: E402
from src.data.dataset import GraphQueryDataset  # noqa: E402
from src.evaluation import real_path_metrics as rpm  # noqa: E402
from src.evaluation.baselines import baseline_summary, real_baseline_summary  # noqa: E402
from src.evaluation.evaluator import evaluate_dataset, records_to_dicts  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.setup import build_diffusion, build_model, get_device  # noqa: E402
from src.utils.config import flatten_overrides, load_config  # noqa: E402
from src.utils.seed import make_generator, set_seed  # noqa: E402


def load_coordinates(config, dataset_path=None, quiet: bool = False):
    """按 config 的 ``data.coords_file`` 加载 ``node -> (lon, lat)``。

    只有真实数据 + 有坐标时才能算 km-based DTW；缺失就返回 None（DTW 记为 NaN），
    绝不让一个可选的地理文件把整次评测搞挂。
    """
    raw = config.get("data.coords_file", None)
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        if not quiet:
            print(f"[warn] data.coords_file not found ({path}) -> DTW disabled")
        return None
    # 顺带把 graph_global.pkl 读进来补缺失节点：原始坐标只覆盖 96.2%，
    # 不补的话碰到缺坐标节点的样本 DTW 会静默变成 NaN。
    graph_path = None
    if dataset_path is not None:
        candidate = Path(dataset_path).parent / "graph_global.pkl"
        if candidate.exists():
            graph_path = candidate
    try:
        coordinates, stats = didi.load_node_coordinates_filled(path, graph_path)
    except Exception as error:  # noqa: BLE001 - 坐标是可选依赖
        if not quiet:
            print(f"[warn] could not load coordinates ({error}) -> DTW disabled")
        return None
    if not quiet:
        print(
            f"coordinates  : {len(coordinates)} nodes from {path.name} "
            f"({stats['with_coordinates']} real @ {stats['coverage']:.1%} "
            f"+ {stats['filled']} filled from neighbours)"
        )
    return coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="evaluate the V2 Graph Flow denoiser")
    parser.add_argument("--config", default="configs/controlled_unweighted.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None, help="结果 json 的输出路径")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--eval-flow-steps",
        type=int,
        default=None,
        help=(
            "推理时每个 reverse step 的图信息交流轮数（必须 <= 训练时的 round 数）。"
            "用于做'训练多轮、推理提前退出'的 ablation；默认用配置里的值。"
        ),
    )
    parser.add_argument("--stochastic", action="store_true", help="强制随机采样")
    parser.add_argument("--deterministic", action="store_true", help="posterior argmax 采样")
    parser.add_argument(
        "--decode",
        default=None,
        choices=["single", "multi"],
        help="single：按采样 z_0 单路径解码（历史口径）；"
        "multi：存活路径表解码（每个 decision 保留 top-k 条 branch，主指标取累计概率"
        "最高的那条，并额外输出 multi_best_goal / multi_best_goal_cost 两条口径与"
        "coverage_rate / optimal_coverage_rate）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="--decode multi 时每个路口的 branch 数（默认取 evaluation.top_k）",
    )
    parser.add_argument(
        "--filter-dead-branches",
        action="store_true",
        default=None,
        help="--decode multi 时，top-k 之前剔除「终点既不是 Goal 也不是 decision "
        "node」的非 NULL branch（默认关 = 历史多分支结果逐位可复现）",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="--decode multi 时存活路径表上限（默认取 evaluation.beam_width）",
    )
    parser.add_argument(
        "--null-policy",
        default=None,
        choices=["stop", "skip"],
        help="stop：NULL 参与排名、选中即该路径终止；skip：NULL 不停，只在非 NULL 候选里取 top-k",
    )
    parser.add_argument(
        "--strict-decode",
        action="store_true",
        default=None,
        help="多分支解码改用**三池**语义（src.evaluation.strict_beam_decoder）："
             "NULL / loop / dead-end 一律在 top-k 之前 mask 掉、失败路径直接淘汰，"
             "最终候选集**只有完整走到 Goal 的路径**，"
             "P* = argmax_{P in success} log_prob；success 为空才判失败。"
             "默认关闭 = 历史口径（goal/loop/NULL/broken 一起排序，逐位可复现）。",
    )
    parser.add_argument("--baselines", action="store_true", help="顺带跑 shortest/greedy baseline")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖配置项，可重复：--set training.lr=1e-4 --set data.num_samples=32",
    )
    return parser.parse_args()


#: CLI 未显式给出时，这些解码口径从 config 的 ``evaluation.*`` 读取。
#: 顺序：CLI > config > 历史默认值（= evaluate_dataset 的形参默认）。
_DECODE_RULER_DEFAULTS = {
    "decode": "single",
    "top_k": 2,
    "beam_width": 64,
    "null_policy": "stop",
    "filter_dead_branches": False,
    "strict_decode": False,
}


def resolve_decode_ruler(args, config) -> None:
    """把 ``evaluation.*`` 里的解码口径填进 ``args``（CLI 显式给了就不动）。

    为什么必须让 config 当默认：``Trainer.validate()``（选 ``best.pt``）读的是
    **同一组键**。如果这里还写死 ``single``/``64``，就会出现"验证按 strict 2/3
    挑 checkpoint、最终评测却按 single 出报表" —— 三把尺子分叉，而 ``best.pt``
    是照最严的那把挑的，报表却拿最松的那把的数字。

    没写这些键的旧 config（``controlled_unweighted`` / ``controlled_weighted``）
    落到的正是 :data:`_DECODE_RULER_DEFAULTS`，与改动前逐位一致。
    """
    # 1) config 自身不能自相矛盾：strict 只在 multi 下存在。这是**配置错误**，
    #    宁可起不来也不要静默按 single 跑完一整套评测。
    config_decode = str(config.get("evaluation.decode", "single")).lower()
    if config_decode not in ("single", "multi"):
        raise SystemExit(f"evaluation.decode={config_decode!r} 不是 single | multi")
    if config_decode != "multi" and bool(config.get("evaluation.strict_decode", False)):
        raise SystemExit(
            "config 自相矛盾：evaluation.strict_decode=true 但 "
            f"evaluation.decode={config_decode!r}；strict 只存在于多分支解码器里"
        )

    # 2) 逐项填默认（CLI 显式给了就不动）
    cli_decode = args.decode
    for key, fallback in _DECODE_RULER_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, config.get(f"evaluation.{key}", fallback))
    args.decode = str(args.decode).lower()
    if args.decode not in ("single", "multi"):
        raise SystemExit(f"evaluation.decode={args.decode!r} 不是 single | multi")

    # 3) CLI 显式要求 single 时**关掉** strict，而不是报错：single 下 strict 没有
    #    意义，用户是在做诊断（想复现历史 single 数字）。危险的方向是反过来的
    #    （以为在用 strict、其实跑了 single），那个已经在第 1 步挡住了。
    if cli_decode is not None and args.decode != "multi":
        args.strict_decode = False


def main() -> int:
    args = parse_args()
    overrides = flatten_overrides(args.overrides)
    if args.device:
        overrides.append(f"training.device={args.device}")
    config = load_config(args.config, overrides)
    if args.stochastic:
        config = load_config(args.config, overrides + ["evaluation.stochastic_sampling=true"])
    if args.deterministic:
        config = load_config(args.config, overrides + ["evaluation.stochastic_sampling=false"])

    resolve_decode_ruler(args, config)
    print(
        f"ruler        : decode={args.decode} strict={bool(args.strict_decode)} "
        f"top_k={args.top_k} beam_width={args.beam_width} "
        f"null_policy={args.null_policy} "
        f"(未显式给的项取自 {args.config} 的 evaluation.*)"
    )

    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = get_device(str(config.get("training.device", "auto")))
    generator = make_generator(seed, device="cpu")

    model = build_model(config, device)
    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint, model=model, map_location=device)
        model = model.to(device)
        print(f"loaded checkpoint {args.checkpoint} (epoch={payload.get('epoch')})")
    if args.eval_flow_steps is not None:
        previous = model.set_inference_flow_steps(args.eval_flow_steps)
        print(
            f"inference flow_steps overridden: {previous} -> {model.flow_steps} "
            f"(trained with {model.max_flow_steps})"
        )
    print(f"model        : {model.flow_steps_label}")
    if args.decode == "multi":
        if args.strict_decode:
            print(
                f"decode       : multi STRICT (top_k={args.top_k}, "
                f"beam_width={args.beam_width})  —— 三池语义：NULL / loop / dead-end 在 "
                "top-k 之前 mask，失败路径直接淘汰，最终候选只有完整到 Goal 的路径，"
                "P* = argmax_{P in success} log_prob"
            )
            print(
                "               null_policy 与 filter_dead_branches 在 strict 下失效"
                "（NULL 永远不合法、dead-end 恒被过滤）"
            )
        else:
            print(
                f"decode       : multi (top_k={args.top_k}, beam_width={args.beam_width}, "
                f"null_policy={args.null_policy})  —— 主指标取累计概率最高的路径，"
                "额外报告 coverage_rate / optimal_coverage_rate"
            )
    else:
        print("decode       : single（按采样 z_0 解码，历史口径）")

    diffusion = build_diffusion(config)
    dataset = GraphQueryDataset.load(args.data)
    coordinates = load_coordinates(config, dataset_path=args.data, quiet=args.no_progress)
    print(f"evaluating {len(dataset)} queries on {device}")

    report = evaluate_dataset(
        model,
        diffusion,
        dataset,
        batch_size=int(config.get("evaluation.batch_size", 8)),
        stochastic=bool(config.get("evaluation.stochastic_sampling", True)),
        device=device,
        generator=generator,
        max_steps=int(config.get("evaluation.max_steps", 0)) or None,
        progress=not args.no_progress,
        weights=None,
        decode=args.decode,
        top_k=args.top_k,
        beam_width=args.beam_width,
        null_policy=args.null_policy,
        filter_dead_branches=args.filter_dead_branches,
        strict_decode=args.strict_decode,
        coordinates=coordinates,
    )

    print("main metrics :", report.summary())
    if report.multi:
        # 指南第 8 节：同一个存活路径表，三条口径并排看（multi_best 是历史口径）
        for label in ("multi_best", "multi_best_goal", "multi_best_goal_cost"):
            row = report.multi.get(label, {})
            if not row:
                continue
            print(f"  {label:<20s} goal_hit={row.get('goal_hit_rate', float('nan')):.4f}"
                  f"  optimal={row.get('optimal_path_rate', float('nan')):.4f}"
                  f"  cost_ratio={row.get('success_cost_ratio', float('nan')):.4f}"
                  f"  broken={row.get('broken_rate', float('nan')):.4f}")
    if report.debug:
        print(f"debug metric : one_step_x0_acc={report.debug.get('accuracy', float('nan')):.4f} "
              f"(只作诊断，不作模型选择)")

    # ---- 真实数据指标（方案第 12、17-J 节）-------------------------------
    real_payload: Dict[str, Any] = {}
    buckets_payload: Dict[str, Any] = {}
    dataset_is_shuffled_od = bool(len(dataset)) and all(
        sample.meta.get("no_real_gt") for sample in dataset
    )
    if report.real:
        real_payload = {
            "metrics": report.real.get("metrics", {}),
            "distribution": report.real.get("distribution", {}),
            "num_paired": report.real.get("num_paired", 0),
            "num_skipped_placeholder_gt": report.real.get("num_skipped_placeholder_gt", 0),
            "gt_source": report.real.get("gt_source"),
        }
        metrics = real_payload["metrics"]
        if metrics:
            print(
                "real metrics : "
                f"goal_hit={metrics.get('goal_hit_rate', float('nan')):.4f} | "
                f"PathSim={metrics.get('path_similarity_score', float('nan')):.4f} | "
                f"nLCS(success)={metrics.get('normalized_lcs_success', float('nan')):.4f} | "
                f"EdgeF1={metrics.get('edge_f1', float('nan')):.4f} | "
                f"PredCostRatio={metrics.get('pred_cost_ratio', float('nan')):.4f} | "
                f"GTCostRatio={metrics.get('gt_cost_ratio', float('nan')):.4f} | "
                f"Pred/GT={metrics.get('pred_over_gt_cost_ratio', float('nan')):.4f} | "
                f"DTW={metrics.get('dtw_km', float('nan')):.4f}km"
                "   (Pred/GT = C(P_pred)/C(P_GT)；DTW = 平均几何偏离 km)"
            )
            distribution = real_payload["distribution"]
            print(
                "distribution : "
                f"KLEV={distribution.get('klev', float('nan')):.6f} | "
                f"JSEV={distribution.get('jsev', float('nan')):.6f} | "
                f"shared_edge_support={distribution.get('shared_edge_support', 0):.0f}"
                "  (dataset-level，不是单样本指标)"
            )
        # 分桶：GDP 风格按 GT 长度等量三分 + 项目特有的 decision 链长度分桶
        aligned = report.real.get("records") or []
        if aligned:
            buckets_payload["length_buckets"] = rpm.bucket_report(
                dataset, aligned, rpm.length_buckets(dataset, 3)
            )
            buckets_payload["decision_buckets"] = rpm.bucket_report(
                dataset, aligned, rpm.decision_buckets(dataset)
            )
            for group_name, table in buckets_payload.items():
                print(f"{group_name}:")
                for bucket, row in table.items():
                    print(
                        f"    {bucket:<10s} n={int(row.get('real_num_queries', 0)):>4d} "
                        f"goal_hit={row.get('goal_hit_rate', float('nan')):.4f} "
                        f"PathSim={row.get('path_similarity_score', float('nan')):.4f} "
                        f"nLCS={row.get('normalized_lcs_success', float('nan')):.4f} "
                        f"EdgeF1={row.get('edge_f1', float('nan')):.4f} "
                        f"CostRatio={row.get('pred_cost_ratio', float('nan')):.4f}"
                    )
    elif dataset_is_shuffled_od:
        # 方案第 7.5 / 16.3 节：shuffled OD 集**没有真实 GT path**，
        # 只报 Goal Hit / Loop / Broken / CostRatio / 推理时间。
        print(
            "real metrics : skipped —— this split has no real GT path "
            "(shuffled OD); only Goal Hit / Loop / Broken / CostRatio / "
            "inference time are meaningful"
        )

    payload = {
        # 评测产物自己记录数据集：看板不再靠文件名猜（加权 run 的 eval_test.json
        # 与无权 run 同名，只靠文件名会认错数据集）。
        "data": args.data,
        "metrics": report.metrics,
        "debug": report.debug,
        "inference": {
            "flow_steps": int(model.flow_steps),
            "trained_flow_steps": int(model.max_flow_steps),
            "label": model.flow_steps_label,
            "decode": args.decode,
            "top_k": int(args.top_k) if args.decode == "multi" else None,
            "beam_width": int(args.beam_width) if args.decode == "multi" else None,
            "null_policy": args.null_policy if args.decode == "multi" else None,
            "filter_dead_branches": bool(args.filter_dead_branches)
            if args.decode == "multi"
            else None,
            "strict": bool(args.strict_decode) if args.decode == "multi" else None,
        },
        "records": records_to_dicts(report.records),
    }
    if real_payload:
        payload["real_path_metrics"] = real_payload
    if buckets_payload:
        payload["buckets"] = buckets_payload
    if real_payload.get("distribution"):
        payload["distribution_metrics"] = real_payload["distribution"]
    if report.multi:
        payload["multi"] = report.multi
    if args.baselines:
        payload["baselines"] = baseline_summary(dataset)
        print("baselines    :", json.dumps(payload["baselines"], ensure_ascii=False))
        # 真实数据上额外给 Dijkstra / greedy 的 nLCS / Edge F1（方案第 13 节）
        try:
            real_baselines = real_baseline_summary(dataset)
        except Exception:  # pragma: no cover - baseline 只是附加产物，不该拖垮评测
            real_baselines = {}
        if real_baselines:
            payload["baselines_real"] = real_baselines
            for name, row in real_baselines.items():
                print(
                    f"baseline {name:<14s} "
                    f"PathSim={row.get('path_similarity_score', float('nan')):.4f} "
                    f"nLCS={row.get('normalized_lcs_success', float('nan')):.4f} "
                    f"EdgeF1={row.get('edge_f1', float('nan')):.4f} "
                    f"CostRatio={row.get('pred_cost_ratio', float('nan')):.4f}"
                )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)
        print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
