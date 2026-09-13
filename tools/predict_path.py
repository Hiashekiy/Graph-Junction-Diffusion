"""用训练好的 checkpoint 给指定 query 生成路径（纯文本 / JSON 输出）。

和 ``tools/visualize_paths.py`` 的分工：

* 想看**图**（预测路径 vs GT 路径画在一起）用 ``visualize_paths.py``；
* 想要**路径本身**（节点序列、结局、分岔点、逐 decision 选了什么）用这个脚本，
  输出是文本 + 可选 JSON，方便直接看数字或喂给别的脚本。

用法::

    # 单条：看第 170 号 query
    python tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 \
        --data data/controlled_test.pkl --index 170

    # 多条（逗号分隔或重复 --index），并导出 JSON
    python tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 \
        --data data/controlled_test.pkl --index 0,47,170 --out-json paths.json

    # 推理轮数 ablation：同一个 checkpoint 只跑 1 轮
    python tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 \
        --data data/controlled_test.pkl --index 170 --flow-steps 1

    # 换随机种子（采样是随机的，同一 query 换个种子可能走出不同路径）
    python tools/predict_path.py --run outputs/runs/v2_controlled_100ep_flow3 \
        --data data/controlled_test.pkl --index 170 --seed 3

``--index`` 就是数据集里的位置（``dataset[i]``），``tools/visualize_paths.py``
标题里的 ``#170`` 也是这个编号。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate paths with a trained checkpoint")
    parser.add_argument("--run", required=True, help="run 目录（含 run_config.json）")
    parser.add_argument("--checkpoint", default=None, help="默认 <run>/best.pt")
    parser.add_argument("--data", required=True, help="数据集 pkl，如 data/controlled_test.pkl")
    parser.add_argument(
        "--index", action="append", default=None,
        help="query 下标，可重复或写成 0,47,170；默认 0",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="采样随机种子（默认 0）")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="用 posterior argmax 采样（不随机）；默认按配置里的 stochastic 采样",
    )
    parser.add_argument(
        "--flow-steps", type=int, default=None,
        help="临时覆盖推理轮数（只能 <= 训练轮数），用于 ablation",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--max-nodes-print", type=int, default=64,
                        help="打印路径时最多显示多少个节点（0 = 全打）")
    return parser.parse_args()


def parse_indices(values: Optional[Sequence[str]]) -> List[int]:
    if not values:
        return [0]
    indices: List[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                indices.append(int(part))
    return indices


@torch.no_grad()
def main() -> int:
    args = parse_args()

    import networkx as nx

    from src.data.collate import collate_samples
    from src.data.dataset import GraphQueryDataset
    from src.diffusion.sampler import sample_reverse_chain
    from src.evaluation.path_decoder import (
        candidate_offsets,
        decision_offsets,
        decode_flat,
    )
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_diffusion, build_model, get_device
    from src.utils.config import load_config
    from src.utils.seed import make_generator, set_seed

    run_dir = Path(args.run)
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise SystemExit(f"missing {config_path} (run 目录里必须有训练时落盘的配置)")
    config = load_config(config_path)
    checkpoint = args.checkpoint or str(run_dir / "best.pt")
    device = get_device(args.device or str(config.get("training.device", "auto")))
    seed = int(args.seed)

    set_seed(seed)
    model = build_model(config, device)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    model = model.to(device)
    if args.flow_steps is not None:
        previous = model.set_inference_flow_steps(
            args.flow_steps, allow_extrapolation=False
        )
        print(f"[inference flow_steps: {previous} -> {model.flow_steps}]")
    model.eval()

    diffusion = build_diffusion(config)
    dataset = GraphQueryDataset.load(args.data)
    indices = parse_indices(args.index)
    missing = [index for index in indices if not 0 <= index < len(dataset)]
    if missing:
        raise SystemExit(
            f"index out of range: {missing} (dataset has {len(dataset)} queries)"
        )

    print(f"checkpoint : {checkpoint} (epoch={payload.get('epoch')})")
    print(f"model      : {model.flow_steps_label}")
    print(f"data       : {args.data} ({len(dataset)} queries)   seed={seed}")

    samples = [dataset[index] for index in indices]
    batch = collate_samples(samples, device=device)
    chain = sample_reverse_chain(
        diffusion,
        model,
        batch,
        generator=make_generator(seed, device="cpu"),
        stochastic=not args.deterministic,
    )
    z0 = chain["z0"]
    decision_starts = decision_offsets(samples)
    candidate_starts = candidate_offsets(samples)

    results: List[Dict[str, Any]] = []
    for position, sample in enumerate(samples):
        result = decode_flat(
            sample,
            z0,
            decision_offset=decision_starts[position],
            candidate_offset=candidate_starts[position],
        )
        optimal_hops = float(
            nx.shortest_path_length(sample.graph, sample.start, sample.goal)
        )
        pred_path = [int(node) for node in result.path]
        gt_path = [int(node) for node in sample.gt_path]
        common = 0
        for node_pred, node_gt in zip(pred_path, gt_path):
            if node_pred != node_gt:
                break
            common += 1
        diverged = common < len(pred_path) and pred_path != gt_path
        record = {
            "index": indices[position],
            "difficulty": sample.meta.get("difficulty", "n/a"),
            "mode": sample.meta.get("mode", "n/a"),
            "num_nodes": int(sample.num_nodes),
            "decisions": int(sample.num_decisions),
            "start": int(sample.start),
            "goal": int(sample.goal),
            "status": result.status,
            "reason": result.reason,
            "optimal_hops": optimal_hops,
            "pred_hops": len(pred_path) - 1,
            "optimal": result.status == "goal" and (len(pred_path) - 1) == optimal_hops,
            "pred_path": pred_path,
            "gt_path": gt_path,
            "divergence_hop": common if diverged else None,
            "divergence_node": gt_path[common - 1] if diverged and common >= 1 else None,
        }
        results.append(record)

        limit = args.max_nodes_print
        shown = pred_path if not limit or len(pred_path) <= limit else (
            pred_path[: limit // 2] + ["..."] + pred_path[-(limit // 2):]
        )
        status_text = {"goal": "到达 goal", "loop": "走进环", "broken": "断掉"}.get(
            result.status, result.status
        )
        print()
        print(
            f"#{record['index']}  {record['difficulty']}/{record['mode']}  "
            f"{record['num_nodes']} 节点 / {record['decisions']} 决策"
        )
        print(
            f"  结果     : {status_text}"
            + (f"（{result.reason}）" if result.reason else "")
            + ("  [最短路]" if record["optimal"] else "")
        )
        print(
            f"  跳数     : 预测 {record['pred_hops']} / GT {int(optimal_hops)}"
        )
        print(f"  预测路径 : {shown}")
        if record["divergence_hop"]:
            print(
                f"  与 GT 分岔: 第 {record['divergence_hop']} 跳"
                f"（节点 {record['divergence_node']}）"
            )
        else:
            print("  与 GT     : 完全一致（或在 GT 前缀内结束）")
        print(f"  GT 路径  : {gt_path}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1, ensure_ascii=False)
        print(f"\nwritten: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
