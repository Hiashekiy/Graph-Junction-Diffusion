"""评测入口（实施指南第 24 节）。

用法::

    python scripts/evaluate.py --config configs/graph_flow.yaml \
        --checkpoint outputs/runs/graph_flow/best.pt --data data/er_256_test.pkl

输出主指标：Goal Hit / Optimal Path / Success Cost Ratio / Loop / Broken / 推理时间。
另外会打印一个 **debug 用** 的 teacher-forced 单步 decision accuracy。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GraphQueryDataset  # noqa: E402
from src.evaluation.baselines import baseline_summary  # noqa: E402
from src.evaluation.evaluator import evaluate_dataset, records_to_dicts  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.setup import build_diffusion, build_model, get_device  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import make_generator, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="evaluate the V2 Graph Flow denoiser")
    parser.add_argument("--config", default="configs/graph_flow.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None, help="结果 json 的输出路径")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stochastic", action="store_true", help="强制随机采样")
    parser.add_argument("--deterministic", action="store_true", help="posterior argmax 采样")
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


def main() -> int:
    args = parse_args()
    # --set 可以重复出现；nargs='*' 版本的 argparse 会丢掉前面几次的值，
    # 所以这里显式摊平（action='append' 给的是 [[...], [...]]）
    overrides = [item for group in args.overrides for item in group]
    overrides = list(args.overrides)
    if args.device:
        overrides.append(f"training.device={args.device}")
    config = load_config(args.config, overrides)
    if args.stochastic:
        config = load_config(args.config, overrides + ["evaluation.stochastic_sampling=true"])
    if args.deterministic:
        config = load_config(args.config, overrides + ["evaluation.stochastic_sampling=false"])

    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = get_device(str(config.get("training.device", "auto")))
    generator = make_generator(seed, device="cpu")

    model = build_model(config, device)
    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint, model=model, map_location=device)
        model = model.to(device)
        print(f"loaded checkpoint {args.checkpoint} (epoch={payload.get('epoch')})")

    diffusion = build_diffusion(config)
    dataset = GraphQueryDataset.load(args.data)
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
    )

    print("main metrics :", report.summary())
    if report.debug:
        print(f"debug metric : one_step_x0_acc={report.debug.get('accuracy', float('nan')):.4f} "
              f"(只作诊断，不作模型选择)")

    payload = {
        "metrics": report.metrics,
        "debug": report.debug,
        "records": records_to_dicts(report.records),
    }
    if args.baselines:
        payload["baselines"] = baseline_summary(dataset)
        print("baselines    :", json.dumps(payload["baselines"], ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, ensure_ascii=False)
        print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
