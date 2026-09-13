"""训练入口（实施指南第 18-22、28 节）。

用法::

    python scripts/train.py --config configs/graph_flow.yaml --name graph_flow
    python scripts/train.py --config configs/graph_flow.yaml --name tiny \
        --tiny --set training.epochs=200 --set diffusion.T=20

``--tiny`` 会生成实施指南第 27 节要求的 tiny overfit 数据集（固定 seed、小图），
先把"能不能过拟合"验证掉，再扩大数据。

注意：训练脚本**不会**自己决定要不要跑，请确认机器空闲后再执行。
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
from src.data.dataset_builder import tiny_overfit_dataset  # noqa: E402
from src.training.checkpoint import load_checkpoint  # noqa: E402
from src.training.setup import (  # noqa: E402
    build_datasets,
    build_diffusion,
    build_model,
    build_optimizer,
    get_device,
    run_directory,
)
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import make_generator, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train the V2 Graph Flow denoiser")
    parser.add_argument("--config", default="configs/graph_flow.yaml")
    parser.add_argument("--name", default=None, help="run name (outputs/runs/<name>)")
    parser.add_argument("--tiny", action="store_true", help="use the tiny overfit dataset")
    parser.add_argument("--tiny-samples", type=int, default=16)
    parser.add_argument("--tiny-nodes", type=int, default=24)
    parser.add_argument("--data", default=None, help="load a pre-generated dataset pkl (train)")
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--resume", default=None, help="resume from a checkpoint (e.g. last.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="total epoch budget")
    parser.add_argument("--extra-epochs", type=int, default=None, help="在 resume 基础上再训练多少轮")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=None, help="每多少个 batch 打一行")
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
    if args.name:
        overrides.append(f"paths.run_name={args.name}")
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    if args.device:
        overrides.append(f"training.device={args.device}")
    if args.log_every is not None:
        overrides.append(f"training.log_every={args.log_every}")
    config = load_config(args.config, overrides)

    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = get_device(str(config.get("training.device", "auto")))
    generator = make_generator(seed, device="cpu")

    if args.data:
        train_dataset = GraphQueryDataset.load(args.data)
        val_dataset = GraphQueryDataset.load(args.val_data) if args.val_data else None
    elif args.tiny:
        train_dataset = tiny_overfit_dataset(
            num_samples=args.tiny_samples, num_nodes=args.tiny_nodes, seed=seed
        )
        val_dataset = train_dataset
    else:
        splits = build_datasets(config)
        train_dataset = splits["train"]
        val_dataset = splits["val"]

    model = build_model(config, device)
    diffusion = build_diffusion(config)
    optimizer = build_optimizer(config, model)
    run_dir = run_directory(config)

    # 把**解析后**的配置（含 --set 覆盖）落盘：run 目录必须能自证是 flow_steps 几、
    # T 多少、batch 多大跑出来的，否则事后无法区分"单轮 vs 多轮"这类对比。
    try:
        (run_dir / "run_config.json").write_text(
            json.dumps(config.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8"
        )
    except (TypeError, ValueError):  # pragma: no cover - 配置里出现非 JSON 值时跳过
        pass

    print(f"device      : {device}")
    print(f"run dir     : {run_dir}")
    print(f"train/val   : {len(train_dataset)} / {len(val_dataset) if val_dataset else 0}")
    print(f"parameters  : {model.num_parameters():,}")
    print(f"T           : {diffusion.T}")
    print(f"flow steps  : {model.flow_steps_label}")

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        run_dir=run_dir,
        generator=generator,
    )

    if args.resume:
        payload = load_checkpoint(
            args.resume, model=model, optimizer=optimizer, map_location=device
        )
        trainer.start_epoch = int(payload.get("epoch", 0))
        trainer.global_step = int(payload.get("global_step", 0))
        if payload.get("best_metric") is not None:
            trainer.best_metric = float(payload["best_metric"])
        budget = int(config.get("training.epochs", 100))
        if args.extra_epochs is not None:
            budget = trainer.start_epoch + int(args.extra_epochs)
        print(
            f"resumed from {args.resume}: start_epoch={trainer.start_epoch}, "
            f"epoch budget={budget}, best_metric={trainer.best_metric:.4f}",
            flush=True,
        )
        history = trainer.fit(epochs=budget)
    else:
        history = trainer.fit()
    print(json.dumps(history[-1] if history else {}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
