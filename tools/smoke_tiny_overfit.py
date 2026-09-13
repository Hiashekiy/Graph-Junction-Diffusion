"""Tiny overfit 冒烟脚本（实施指南第 27 节）。

真正跑：
    1. 生成一个固定 seed 的小数据集；
    2. 打印 batch 形状；
    3. 用很小的 T 做若干步 teacher-forced recurrent 训练，打印 loss / x0 acc / goal hit；
    4. 跑一遍完整 reverse chain + decoder，打印主指标。

用法：python outputs/_smoke_tiny.py [epochs] [num_samples]
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data.collate import collate_samples, iter_batches  # noqa: E402
from src.data.dataset_builder import tiny_overfit_dataset  # noqa: E402
from src.diffusion.categorical import CategoricalDiffusion  # noqa: E402
from src.diffusion.schedule import NoiseSchedule  # noqa: E402
from src.evaluation.evaluator import evaluate_dataset  # noqa: E402
from src.models.denoiser import GraphFlowDenoiser  # noqa: E402
from src.training.losses import LossWeights, recurrent_reverse_loss  # noqa: E402
from src.utils.seed import make_generator, set_seed  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
NUM_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 12
T = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(0)
generator = make_generator(0, device="cpu")

t0 = time.time()
dataset = tiny_overfit_dataset(
    num_samples=NUM_SAMPLES, num_nodes=24, seed=0, min_od_distance=3
)
print(f"dataset: {len(dataset)} queries in {time.time() - t0:.1f}s", flush=True)
samples = [dataset[i] for i in range(len(dataset))]
first_batch = collate_samples(samples[:4], device=DEVICE)
print("batch:", json.dumps(first_batch.describe()), flush=True)

# 清单第 11 节验收项之一：multi-graph batch 的 forced physical edge offset 正确。
# （"每个 timestep 都 selected" 的完整性质由 tests/test_source_forced.py 覆盖。）
if first_batch.num_source_forced_edges:
    ids = first_batch.source_forced_edge_ids
    assert int(ids.min()) >= 0
    assert int(ids.max()) < first_batch.num_physical_edges, "physical edge offset 越界"
    print(
        f"source forced edges: {ids.numel()} 条，全部落在 "
        f"[0, {first_batch.num_physical_edges}) 内",
        flush=True,
    )

model = GraphFlowDenoiser(d_model=32, ffn_hidden=64).to(DEVICE)
diffusion = CategoricalDiffusion(
    NoiseSchedule(T=T, schedule="linear", beta_start=0.05, beta_end=0.5)
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
weights = LossWeights()

batches = iter_batches(samples, batch_size=4, shuffle=True, seed=0)
print(f"params: {model.num_parameters():,}  T={T}  batches/epoch={len(batches)}", flush=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    accuracy = 0.0
    for chunk in batches:
        batch = collate_samples(chunk, device=DEVICE)
        out = recurrent_reverse_loss(model, diffusion, batch, weights, generator)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(out.loss.detach())
        accuracy += out.final_accuracy
    if epoch % max(1, EPOCHS // 10) == 0 or epoch == 1:
        print(
            f"epoch {epoch:>4}: loss={total / len(batches):.4f} "
            f"x0_acc={accuracy / len(batches):.3f}",
            flush=True,
        )

model.eval()
report = evaluate_dataset(
    model,
    diffusion,
    dataset,
    batch_size=4,
    stochastic=True,
    device=DEVICE,
    generator=generator,
    progress=False,
    weights=weights,
)
print("full-chain (stochastic):", report.summary(), flush=True)
print("debug x0 acc:", round(report.debug.get("accuracy", float("nan")), 4))
print(
    "statuses:",
    {s: sum(1 for r in report.records if r.status == s) for s in ("goal", "loop", "broken")},
)

# 确定性采样（后验取 argmax）应该更容易命中
deterministic = evaluate_dataset(
    model,
    diffusion,
    dataset,
    batch_size=4,
    stochastic=False,
    device=DEVICE,
    generator=generator,
    progress=False,
    weights=weights,
)
print("full-chain (deterministic):", deterministic.summary(), flush=True)

# 逐样本的 teacher-forced 单步 accuracy（不受 batch 归一化影响）
from src.training.losses import grouped_argmax  # noqa: E402

with torch.no_grad():
    hits = 0
    for sample in dataset:
        batch = collate_samples([sample], device=DEVICE)
        out = model.step(batch, model.init_nodes(batch), batch.target_candidate, 1)
        picks = grouped_argmax(
            out.candidate_log_prob, batch.candidate_owner, batch.num_decisions
        )
        hits += int((picks == batch.target_candidate).all().item())
print(f"per-sample x0 all-correct: {hits}/{len(dataset)}")
