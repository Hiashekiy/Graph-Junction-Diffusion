"""汇总一次训练的结果：曲线 + best checkpoint 的测试集评测。

用法::

    python tools/summarize_run.py outputs/runs/v2_controlled_100ep \
        --config configs/graph_flow.yaml --test-data data/controlled_test.pkl

输出 ``<run_dir>/summary.json`` 与 ``<run_dir>/summary.txt``，内容包括：
    * 每 epoch 的 train loss / x0 accuracy；
    * 每次验证的 Goal Hit / Optimal / Cost Ratio / Loop / Broken；
    * best checkpoint 在测试集上的完整指标（含 baseline 对照）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_history(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "history.json"
    if not path.exists():
        raise FileNotFoundError(f"no history.json in {run_dir}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _val_file_goal_hit(path: Path) -> float:
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not records:
        raise ValueError(f"empty val records: {path}")
    return sum(1 for r in records if r["goal_hit"]) / len(records)


def reconstruct_epoch_offset(
    run_dir: Path, history: List[Dict[str, Any]]
) -> Optional[int]:
    """给"没有 epoch 字段的老 history"找回真实 epoch 号。

    老版本的 trainer 不往记录里写 ``epoch``，而 ``history.json`` 可能只保留了后一段
    （例如 ``v2_controlled_100ep`` 只剩 epoch 21-100 的 80 条，前面 20 条丢了）。
    这时 ``index + 1`` 会凭空少算 20 个 epoch —— 报出来的 "best validation: epoch 60"
    其实是 epoch 80。

    可靠的锚点是 ``val_records_epoch{N}.json``：文件名里带真实 epoch，内容能算出
    goal_hit。做法是从**末尾**对齐：history 里第 j 个带验证的记录，对应最后
    len(val_positions) 个 val 文件里的第 j 个，并用 goal_hit 值逐个核对。核对通过
    才返回偏移量；对不上就返回 None（调用方退回 index+1 并打警告），绝不猜。
    """
    positions = [i for i, record in enumerate(history) if "goal_hit_rate" in record]
    if not positions:
        return None
    files = sorted(
        run_dir.glob("val_records_epoch*.json"),
        key=lambda path: int(path.stem.replace("val_records_epoch", "")),
    )
    if len(files) < len(positions):
        return None
    files = files[-len(positions):]
    for position, path in zip(positions, files):
        expected = float(history[position]["goal_hit_rate"])
        actual = _val_file_goal_hit(path)
        if abs(expected - actual) > 1e-9:
            return None
    first_epoch = int(files[0].stem.replace("val_records_epoch", ""))
    return first_epoch - (positions[0] + 1)


def curve(
    history: List[Dict[str, Any]], key: str, epoch_offset: int = 0
) -> List[Dict[str, float]]:
    """从 history 抽一条曲线。

    epoch 号优先用记录里的 ``epoch`` 字段；老 history 没有这个字段，就用
    ``index + 1 + epoch_offset``（``epoch_offset`` 由 :func:`reconstruct_epoch_offset`
    用 val_records 文件名校准）。
    """
    out = []
    for index, record in enumerate(history, start=1):
        if key in record and record[key] == record[key]:  # 过滤 NaN
            out.append(
                {
                    "epoch": int(record.get("epoch", index + epoch_offset)),
                    "value": float(record[key]),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="summarize a training run")
    parser.add_argument("run_dir")
    parser.add_argument("--config", default="configs/graph_flow.yaml")
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    import torch

    from src.data.dataset import GraphQueryDataset
    from src.evaluation.baselines import baseline_summary
    from src.evaluation.evaluator import evaluate_dataset, records_to_dicts
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_diffusion, build_model, get_device
    from src.utils.config import load_config
    from src.utils.seed import make_generator, set_seed

    run_dir = Path(args.run_dir)
    history = load_history(run_dir)
    # 训练时落盘的 run_config.json 才是这次 run 真正用的配置；没有它才退回 --config
    # （旧 run 目录里没有这个文件，用当前 config 解释它们可能给出错的 flow_steps）。
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        config_source = str(run_config_path)
        overrides: List[str] = []
    else:
        config_source = args.config
        overrides = [item for group in args.overrides for item in group]
    config = load_config(config_source, overrides)
    epoch_offset = reconstruct_epoch_offset(run_dir, history)
    if epoch_offset:
        print(
            f"note: history.json has no 'epoch' field; reconstructed epoch offset "
            f"+{epoch_offset} from val_records_epoch*.json (epoch labels were off by "
            f"{epoch_offset})"
        )
    elif epoch_offset is None and history and "epoch" not in history[0]:
        print(
            "warning: history.json has no 'epoch' field and the val_records_epoch*.json "
            "files could not be aligned; epoch labels below are index+1 and may be wrong"
        )
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "config_source": config_source,
        "epochs_completed": len(history),
        "epoch_offset_reconstructed": epoch_offset,
        "curves": {
            key: curve(history, key, epoch_offset or 0)
            for key in (
                "train_loss",
                "train_x0_acc",
                "goal_hit_rate",
                "optimal_path_rate",
                "success_cost_ratio",
                "loop_rate",
                "broken_rate",
                "val_x0_acc",
            )
        },
    }

    # 一个 run 目录必须能自证自己是哪套模型/超参跑出来的，否则"单轮 vs 多轮"这类
    # 对比只能靠回忆。这里把 config 里真正影响模型的项 + 参数量写进 summary.json。
    device = get_device(args.device or str(config.get("training.device", "auto")))
    probe = build_model(config)
    summary["model"] = {
        "d_model": int(config.get("model.d_model", 128)),
        "ffn_hidden": int(config.get("model.ffn_hidden", 256)),
        "flow_steps": int(config.get("model.flow_steps", 1)),
        "flow_slot_embedding": bool(config.get("model.flow_slot_embedding", True)),
        "flow_slot_scale": float(config.get("model.flow_slot_scale", 1.0)),
        "parameters": int(probe.num_parameters()),
        "diffusion_T": int(config.get("diffusion.T", 50)),
        "batch_size": int(config.get("training.batch_size", 0)),
        "lr": float(config.get("training.lr", 0.0)),
        "amp": bool(config.get("training.amp", False)),
        "device": str(device),
        "label": probe.flow_steps_label,
    }
    del probe

    # 用 enumerate 记住下标：`history.index(best)` 在出现两条完全相同的记录时会指到
    # 前一条，best epoch 就会报错。
    validation = [
        (index, record)
        for index, record in enumerate(history)
        if "goal_hit_rate" in record
    ]
    if validation:
        best_index, best = max(validation, key=lambda item: item[1]["goal_hit_rate"])
        best_epoch = int(
            best.get("epoch", best_index + 1 + (epoch_offset or 0))
        )
        summary["best_validation"] = {
            "epoch": best_epoch,
            "goal_hit_rate": best["goal_hit_rate"],
            "optimal_path_rate": best.get("optimal_path_rate"),
            "avg_goal_hit_rate": sum(r["goal_hit_rate"] for _, r in validation)
            / len(validation),
            "avg_optimal_path_rate": sum(
                r.get("optimal_path_rate", 0.0) for _, r in validation
            )
            / len(validation),
            "tail_goal_hit_rate": sum(
                r["goal_hit_rate"] for _, r in validation[-5:]
            )
            / len(validation[-5:]),
        }

    if args.test_data and not args.skip_eval:
        seed = int(config.get("seed", 0))
        set_seed(seed)
        checkpoint = args.checkpoint or str(run_dir / "best.pt")

        model = build_model(config, device)
        payload = load_checkpoint(checkpoint, model=model, map_location=device)
        model = model.to(device)
        diffusion = build_diffusion(config)
        dataset = GraphQueryDataset.load(args.test_data)
        report = evaluate_dataset(
            model,
            diffusion,
            dataset,
            batch_size=int(config.get("evaluation.batch_size", 16)),
            stochastic=bool(config.get("evaluation.stochastic_sampling", True)),
            device=device,
            generator=make_generator(seed, device="cpu"),
            progress=False,
            weights=None,
        )
        summary["test"] = {
            "checkpoint": checkpoint,
            "checkpoint_epoch": payload.get("epoch"),
            "num_queries": len(dataset),
            "metrics": report.metrics,
            "debug": report.debug,
            "baselines": baseline_summary(dataset),
        }
        with open(run_dir / "test_records.json", "w", encoding="utf-8") as handle:
            json.dump(records_to_dicts(report.records), handle, indent=1)
        del torch

    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, ensure_ascii=False)

    lines = [f"run: {run_dir}", f"epochs completed: {len(history)}"]
    if "config_source" in summary:
        lines.append(f"config: {summary['config_source']}")
    if "model" in summary:
        model_info = summary["model"]
        lines.append(
            f"model: flow_steps={model_info['flow_steps']} "
            f"({model_info['label']})  params={model_info['parameters']}  "
            f"T={model_info['diffusion_T']}  batch={model_info['batch_size']}"
        )
    if "best_validation" in summary:
        best = summary["best_validation"]
        lines.append(
            f"best validation: epoch {best['epoch']}  goal_hit={best['goal_hit_rate']:.4f}  "
            f"optimal={best['optimal_path_rate']:.4f}"
        )
        lines.append(
            f"validation average: goal_hit={best['avg_goal_hit_rate']:.4f}  "
            f"optimal={best['avg_optimal_path_rate']:.4f}"
        )
        if "tail_goal_hit_rate" in best:
            lines.append(
                f"validation tail (last {min(5, summary['epochs_completed'])} checks): "
                f"goal_hit={best['tail_goal_hit_rate']:.4f}"
            )
    if "test" in summary:
        metrics = summary["test"]["metrics"]
        lines.append(f"test ({summary['test']['num_queries']} queries, "
                     f"checkpoint epoch {summary['test']['checkpoint_epoch']}):")
        for key in (
            "goal_hit_rate",
            "optimal_path_rate",
            "success_cost_ratio",
            "loop_rate",
            "broken_rate",
            "mean_elapsed",
        ):
            if key in metrics:
                lines.append(f"  {key}: {metrics[key]:.4f}")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {run_dir / 'summary.json'} , {run_dir / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
