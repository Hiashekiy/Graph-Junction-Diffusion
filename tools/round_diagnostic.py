"""诊断"多轮图信息交流到底有没有做事"。

对 ``model.flow_steps = k`` 的 checkpoint，量一个 reverse step 内部每轮对 H 的改动：

    rel_k = ||H_k - H_{k-1}|| / ||H_{k-1}||      （只在自由节点上算，Start/Goal 被 clamp）

用途：

* ``rel`` 接近 0（例如 << 1%）说明 slot embedding 太弱、各轮几乎在做同一件事，
  多轮等于白算 —— 该调大 ``model.flow_slot_scale``，而不是继续等训练；
* ``rel`` 只有第一轮大、后面迅速塌到 0，说明在求不动点；
* 顺便给出 ``flow_steps=1/2/3`` 的**最终** H 差异，判断"少跑几轮"会不会崩。

用法::

    python tools/round_diagnostic.py outputs/runs/v2_controlled_100ep_flow3 \
        --data data/controlled_val.pkl --samples 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inspect per-round changes inside one reverse step")
    parser.add_argument("run_dir", help="含 run_config.json 与 last.pt 的 run 目录")
    parser.add_argument("--checkpoint", default=None, help="默认 <run_dir>/last.pt")
    parser.add_argument("--data", default="data/controlled_val.pkl")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    from src.data.collate import collate_samples
    from src.data.dataset import GraphQueryDataset
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_model
    from src.utils.config import load_config

    run_dir = Path(args.run_dir)
    config_source = run_dir / "run_config.json"
    config = load_config(config_source if config_source.exists() else "configs/graph_flow.yaml")
    device = torch.device(args.device)
    model = build_model(config, device)
    checkpoint = args.checkpoint or str(run_dir / "last.pt")
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    model.eval()
    print(f"checkpoint : {checkpoint} (epoch={payload.get('epoch')})")
    print(f"model      : {model.flow_steps_label}")
    if model.flow_slot_embedding is not None:
        weight = model.flow_slot_embedding.weight
        print(
            f"slot embed : rows={model.max_flow_steps} rows "
            f"std={float(weight.std()):.4f} mean_norm={float(weight.norm(dim=-1).mean()):.4f} "
            f"scale={model.slot_scale}"
        )
    else:
        print("slot embed : none (single-round model; extra rounds would be a fixed-point iteration)")

    dataset = GraphQueryDataset.load(args.data)
    samples = [dataset[index] for index in range(min(args.samples, len(dataset)))]
    batch = collate_samples(samples, device=device)
    H_0 = model.init_nodes(batch)
    z = batch.target_candidate
    t = int(config.get("diffusion.T", 50)) - 1
    free = ~batch.start_goal_mask

    print("\n[per round] change made by round k itself (H is carried across rounds)")
    edge_feat = model.edge_state_encoder(batch, z)
    tau = model.time_embedding(t, batch.num_graphs)
    states = [H_0]
    H = H_0
    for step in range(model.max_flow_steps):
        slot_tau = None
        if model.flow_slot_embedding is not None and model.max_flow_steps > 1:
            index = torch.full((tau.shape[0],), step, dtype=torch.long, device=device)
            slot_tau = model.flow_slot_embedding(index) * model.slot_scale
        out = model.graph_flow(
            H_t=H,
            edge_index=batch.edge_index,
            edge_feat=edge_feat,
            tau_t=tau,
            fixed_mask=batch.start_goal_mask,
            graph_node_ptr=batch.graph_node_ptr,
            slot_tau=slot_tau,
        )
        current = out["H_next"]
        rel_free = float(
            (current[free] - H[free]).norm() / H[free].norm().clamp_min(1e-9)
        )
        print(f"  round {step}: ||dH_free||/||H_free|| = {rel_free:.6f}")
        H = current
        states.append(current)

    if model.max_flow_steps > 1:
        print(
            f"\n[early exit] how far the end state is from the {model.max_flow_steps}-round one"
        )
        final = states[-1]
        for steps in range(1, model.max_flow_steps):
            state = states[steps]
            rel = float(
                (state[free] - final[free]).norm()
                / final[free].norm().clamp_min(1e-9)
            )
            print(f"  {steps} round(s) instead of {model.max_flow_steps}: "
                  f"||H_k - H_K||/||H_K|| = {rel:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
