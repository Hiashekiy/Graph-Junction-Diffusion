"""single-path 最终 readout 的验收测试（本次改动的核心）。

新语义（方案要求）：

    stochastic reverse chain（不变）
      -> 最后一个 reverse step 的 candidate probability
      -> 每个 decision 组内 argmax  ->  z0_argmax
      -> decode_flat(sample, z0_argmax)     # 确定性走图

必须与另外两条路径严格区分：

    single_sampled        ：直接解码采样出来的 z0（旧默认，现改为诊断口径）
    deterministic rollout ：``stochastic=False``，每一步 posterior 都取 argmax
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.data.collate import collate_samples, null_candidate_of_decision
from src.data.dataset import GraphQueryDataset
from src.evaluation.evaluator import evaluate_dataset
from src.evaluation.readout import single_path_state, single_readout_state

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTED_RUN = REPO_ROOT / "outputs" / "runs" / "v2_weighted_controlled"
WEIGHTED_DATA = REPO_ROOT / "data" / "weighted_controlled" / "weighted_controlled_test.pkl"


# ---------------------------------------------------------------------------
# 1. grouped argmax：每个 decision 只在自己的候选里取最大
# ---------------------------------------------------------------------------
def test_single_readout_is_per_decision_grouped_argmax() -> None:
    # d0 有 3 个候选（flat 0..2），d1 有 2 个（flat 3..4）
    owner = torch.tensor([0, 0, 0, 1, 1])
    prob = torch.tensor([0.10, 0.70, 0.20, 0.55, 0.45])
    assert single_readout_state(prob, owner, 2).tolist() == [1, 3]

    # 全局 argmax 落在 d1 的候选上：如果写成 flat argmax，d0 会被错误地指到 3
    prob2 = torch.tensor([0.10, 0.20, 0.30, 0.90, 0.05])
    assert single_readout_state(prob2, owner, 2).tolist() == [2, 3]

    # 返回的一定是各 decision 组内的候选编号
    state = single_readout_state(prob2, owner, 2)
    assert int(state[0]) in (0, 1, 2) and int(state[1]) in (3, 4)


def test_single_readout_handles_the_empty_case() -> None:
    empty = single_readout_state(torch.zeros(0), torch.zeros(0, dtype=torch.long), 0)
    assert empty.numel() == 0


# ---------------------------------------------------------------------------
# 2/3/5. evaluator：single 读最终概率、不读采样 z0；采样器调用方式不变
# ---------------------------------------------------------------------------
def _one_hot_at_target(batch) -> torch.Tensor:
    prob = torch.zeros(batch.num_candidates)
    prob[batch.target_candidate] = 1.0
    return prob


def _null_at_first_decision(batch) -> torch.Tensor:
    null_index = null_candidate_of_decision(batch)
    return torch.where(null_index >= 0, null_index, batch.target_candidate)


def _run_with_stub_sampler(monkeypatch, batch, sample, prob, z0):
    """把 reverse chain 换成固定返回，专注检查 single 的 readout 逻辑。"""
    captured = {}

    def fake_chain(diffusion, model, batch_arg, **kwargs):
        captured.update(kwargs)
        return {
            "z0": z0,
            "candidate_prob": prob,
            "candidate_log_prob": None,
            "H0": None,
            "trace": None,
        }

    monkeypatch.setattr("src.evaluation.evaluator.sample_reverse_chain", fake_chain)

    def run(decode: str):
        return evaluate_dataset(
            SimpleNamespace(eval=lambda: None),
            None,
            GraphQueryDataset([sample]),
            batch_size=1,
            device="cpu",
            progress=False,
            decode=decode,
        )

    return run, captured


def test_single_decode_uses_the_final_probability_not_the_sampled_z0(monkeypatch, manual_sample):
    batch = collate_samples([manual_sample], device="cpu")
    prob = _one_hot_at_target(batch)                 # 最终概率的组内 argmax = GT 候选
    z0_sampled = _null_at_first_decision(batch)      # 采样状态：第一个路口选了 NULL
    run, captured = _run_with_stub_sampler(monkeypatch, batch, manual_sample, prob, z0_sampled)

    new = run("single")
    old = run("single_sampled")

    assert new.metrics["goal_hit_rate"] == 1.0, "single 应该按最终概率的 argmax 到达终点"
    assert old.metrics["goal_hit_rate"] == 0.0, "single_sampled 仍然跟随采样状态（NULL -> broken）"
    # 5. reverse chain 仍然是 stochastic 的（evaluator 没有偷偷传 stochastic=False）
    assert captured["stochastic"] is True


def test_single_decode_is_invariant_to_the_sampled_state(monkeypatch, manual_sample):
    """sampled z0 变了、最终 candidate_prob 不变 -> single 结果必须不变。"""
    batch = collate_samples([manual_sample], device="cpu")
    prob = _one_hot_at_target(batch)
    first_z0 = _null_at_first_decision(batch)
    second_z0 = torch.roll(first_z0, shifts=1)       # 另一个完全不同的采样状态

    run_a, _ = _run_with_stub_sampler(monkeypatch, batch, manual_sample, prob, first_z0)
    report_a = run_a("single")
    run_b, _ = _run_with_stub_sampler(monkeypatch, batch, manual_sample, prob, second_z0)
    report_b = run_b("single")

    # wall_time 是耗时，天然会变；其余指标必须逐位相同
    strip = lambda metrics: {k: v for k, v in metrics.items() if k != "wall_time"}
    assert strip(report_a.metrics) == strip(report_b.metrics)
    # 逐样本记录也要一致（elapsed 是耗时，去掉）
    records = lambda report: [
        {k: v for k, v in record.to_dict().items() if k != "elapsed"}
        for record in report.records
    ]
    assert records(report_a) == records(report_b)


def test_multi_path_does_not_depend_on_sampled_state_either(monkeypatch, manual_sample):
    """multi 直接用整组 candidate_prob —— 采样状态换了也不影响（历史口径不变）。"""
    batch = collate_samples([manual_sample], device="cpu")
    prob = _one_hot_at_target(batch)
    run_a, _ = _run_with_stub_sampler(monkeypatch, batch, manual_sample, prob, _null_at_first_decision(batch))
    first = run_a("multi")
    run_b, _ = _run_with_stub_sampler(monkeypatch, batch, manual_sample, prob, torch.zeros_like(batch.target_candidate))
    second = run_b("multi")

    strip = lambda metrics: {k: v for k, v in metrics.items() if k != "wall_time"}
    assert strip(first.metrics) == strip(second.metrics)
    assert first.multi == second.multi


def test_single_path_state_reports_a_clear_error_without_candidate_prob() -> None:
    batch = SimpleNamespace(candidate_owner=torch.tensor([0]), num_decisions=1)
    with pytest.raises(ValueError, match="candidate_prob"):
        single_path_state({"z0": torch.tensor([0])}, batch, "single")
    with pytest.raises(ValueError, match="unknown single readout"):
        single_path_state({"z0": torch.tensor([0])}, batch, "nope")


# ---------------------------------------------------------------------------
# 5b. 采样链本身没有被改成确定性的（签名默认仍是 stochastic=True）
# ---------------------------------------------------------------------------
def test_reverse_chain_default_is_still_stochastic() -> None:
    from src.diffusion.sampler import sample_reverse_chain

    signature = inspect.signature(sample_reverse_chain)
    assert signature.parameters["stochastic"].default is True


# ---------------------------------------------------------------------------
# 6. #92 回归：readout 到达，采样状态仍然是 NULL/broken
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (WEIGHTED_RUN / "best.pt").exists() or not WEIGHTED_DATA.exists(),
    reason="weighted checkpoint / dataset are not available",
)
def test_sample_92_readout_reaches_the_goal_while_the_sample_stays_broken() -> None:
    from src.data.dataset import GraphQueryDataset as _Dataset
    from src.diffusion.sampler import sample_reverse_chain
    from src.evaluation.path_decoder import decode_flat
    from src.training.checkpoint import load_checkpoint
    from src.training.setup import build_diffusion, build_model
    from src.utils.config import load_config
    from src.utils.seed import make_generator, set_seed

    config = load_config(WEIGHTED_RUN / "run_config.json")
    model = build_model(config, torch.device("cpu"))
    load_checkpoint(WEIGHTED_RUN / "best.pt", model=model, map_location="cpu")
    model.eval()
    diffusion = build_diffusion(config)

    sample = _Dataset.load(WEIGHTED_DATA)[92]
    batch = collate_samples([sample], device="cpu")
    set_seed(0)
    chain = sample_reverse_chain(
        diffusion, model, batch, generator=make_generator(0, device="cpu"), stochastic=True
    )

    readout = single_path_state(chain, batch, "single")
    readout_result = decode_flat(sample, readout, decision_offset=0, candidate_offset=0)
    sampled_result = decode_flat(sample, chain["z0"], decision_offset=0, candidate_offset=0)

    # 新默认 single：最终概率 argmax -> node 8 选 ->9 -> 到达 goal
    assert readout_result.status == "goal"
    assert readout_result.path[-1] == int(sample.goal)
    # 采样状态仍然可以是 NULL（它只用于扩散可视化/诊断）
    assert sampled_result.status == "broken"
    assert "NULL" in sampled_result.reason
    assert int(readout[-1]) != int(chain["z0"][-1])

