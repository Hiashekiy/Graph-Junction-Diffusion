"""多轨迹集合损失（``src/training/trajectory_loss.py``）的测试。

这里守的是方案里几条**容易写错、错了还看不出来**的硬约束：

* ``S(P)`` 必须是 **mean** log-prob。用 sum 的话越短的"尸体"分数越高，模型会主动
  去生成最短的失败轨迹 —— 历史 ``best`` readout 就是这个 bug。
* 训练 miner 必须用 ``strict=False`` 的历史 decoder。strict 会在 top-k 之前把
  NULL / loop / dead-end 全 mask 掉，miner 根本看不到失败轨迹，L_fail 恒为 0。
* 候选下标是**样本局部**坐标系（decoder 的 ``candidate_indices`` 与 GT 标签都在
  这个坐标系里）。``batch.target_candidate`` 是全局的，必须减去样本起点。
* ``failure_cost`` 必须真的来自 config。早先 ``MinedTrajectory`` 上挂了一个用模块
  默认值的属性，config 里的 ``failure_cost`` 永远被覆盖、成了死配置。
* batch **等权**平均：候选多的样本不能因为候选多就占更大权重。

两条约定，写断言时必须记住：

1. **所有指标都是跨样本平均的**。``two_sample_batch()`` 里有 2 个样本，所以只给
   其中一个样本设计划会让期望值被 2 整除 —— 用 :func:`plan_all` 给每个样本同一份
   计划，或显式按 /2 写期望。
2. ``pytest.approx(0.0)`` 的相对容差对 ``-1e-8`` 这种"数学上为 0"的量不成立，
   涉及 ``_EPS`` 的零值断言统一写 ``abs=1e-6``。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import collate_samples  # noqa: E402
from src.training import trajectory_loss as tl  # noqa: E402
from src.training.losses import (  # noqa: E402
    SATURATING_NLL,
    LossWeights,
    RecurrentLossOutput,
    saturating_null_loss,
)
from src.training.trajectory_loss import (  # noqa: E402
    DEFAULT_FAILURE_COST,
    METRIC_KEYS,
    MinedTrajectory,
    TrajectoryLossConfig,
    build_gt_trajectory,
    classify_failure,
    mine_success_and_failure,
    trajectory_set_loss,
)

from conftest import make_manual_sample  # noqa: E402

# 数学上为 0、但因为 _EPS 会变成 ±1e-8 的量
ZERO = dict(abs=1e-6)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
class FakeSection(dict):
    """``Config.section()`` 返回的那种"取不到就给默认值"的映射。"""

    def get(self, key, default=None):
        return dict.get(self, key, default)


class FakeConfig:
    """``LossWeights.from_config`` 唯一依赖的接口就是 ``.section(name)``。"""

    def __init__(self, data):
        self._data = dict(data)

    def section(self, name):
        value = self._data.get(name)
        if value is None:
            return FakeSection()
        if not isinstance(value, dict):
            raise TypeError(f"{name!r} is not a config section")
        return FakeSection(value)


def enabled_config(**overrides) -> TrajectoryLossConfig:
    config = TrajectoryLossConfig(enabled=True)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def mined(nodes, indices, status, nlcs=0.0) -> MinedTrajectory:
    return MinedTrajectory(
        nodes=list(nodes), candidate_indices=list(indices), status=status, nlcs=nlcs
    )


class _FakeCandidate:
    """只需要 decoder 返回对象的四个字段。"""

    def __init__(self, nodes, status, reason="", indices=()):
        self.nodes = list(nodes)
        self.status = status
        self.reason = reason
        self.candidate_indices = list(indices)


def two_sample_batch():
    """两个**结构相同但对象不同**的手工样本，用于验证多样本下标与 batch 加权。"""
    first = make_manual_sample()
    second = make_manual_sample()
    return [first, second], collate_samples([first, second], device="cpu")


def log_prob_row(batch, overrides=None, value=-1.0, requires_grad=True) -> torch.Tensor:
    """整表 ``log p`` 常量行，可用 ``overrides`` 覆盖若干**局部**下标。

    先写值再 ``requires_grad_()``：否则对 leaf Variable 的 view 做原地赋值会抛
    "a view of a leaf Variable that requires grad is being used in an in-place
    operation"。
    """
    total = int(batch.candidate_owner.numel())
    row = torch.full((total,), float(value))
    for index, item in (overrides or {}).items():
        row[int(index)] = float(item)
    return row.requires_grad_(requires_grad)


def uniform_log_prob(batch) -> torch.Tensor:
    return log_prob_row(batch)


def value(tensor) -> float:
    """取标量值；避免 ``float(tensor_with_grad)`` 的警告。"""
    return float(tensor.detach())


def plan_all(samples, success, failure):
    """把同一份挖掘结果写给 batch 里的**每个**样本。

    指标是跨样本平均的，只给一个样本设计划会让期望值被样本数整除，读起来容易错
    （例如"NULL 质量 = 1.0"实际报 0.5）。
    """
    return {id(sample): (list(success), list(failure)) for sample in samples}


def gt_active_indices(sample):
    """GT 在**本样本局部**候选表里的 active candidate 下标（NULL decision 不算）。"""
    candidates = sample.field.candidates
    return [
        int(index)
        for index, is_null in zip(
            candidates.target_candidate, candidates.candidate_is_null
        )
        if not is_null
    ]


class FakeResult:
    """``decode_multi_path`` 的返回值里 miner 只用到 ``.finished``。"""

    def __init__(self, finished):
        self.finished = list(finished)


def patch_decoder(monkeypatch, finished):
    """把 ``decode_multi_path`` 换掉，**保留** ``mine_success_and_failure`` 本体。

    miner 里的截断 / 去重 / 类型挑选 / raw_* 统计都长在 miner 这一侧，所以测这些
    行为时必须打桩 decoder 而不是 miner —— 打桩 miner 会把这些逻辑整段绕过。
    """

    def fake(sample, candidate_prob, **kwargs):
        return FakeResult(finished)

    monkeypatch.setattr(
        "src.evaluation.multi_path_decoder.decode_multi_path", fake
    )


def patch_miner(monkeypatch, plan):
    """把 miner 换掉：``plan[id(sample)] = (success, failure)``。

    miner 是唯一"看数据定行为"的部分；换掉它之后候选池完全可控，才能写精确断言。
    """

    def fake(sample, probs, config):
        success, failure = plan.get(id(sample), ([], []))
        stats = {
            "raw_finished": len(success) + len(failure),
            "raw_success": len(success),
            "raw_null": sum(1 for item in failure if item.status == "null"),
            "raw_loop": sum(1 for item in failure if item.status == "loop"),
            "raw_dead_end": sum(1 for item in failure if item.status == "dead_end"),
            "raw_broken": sum(1 for item in failure if item.status == "broken"),
        }
        return list(success), list(failure), stats

    monkeypatch.setattr(tl, "mine_success_and_failure", fake)


# ---------------------------------------------------------------------------
# S(P)：mean 而不是 sum
# ---------------------------------------------------------------------------
def test_trajectory_score_is_mean_log_prob_not_sum():
    """等单步概率下，长轨迹和短轨迹得分**相同**。

    用 sum 的话 2 跳的失败轨迹得分 -2、6 跳的 -6，softmax 会把质量全给最短的尸体。
    """
    row = torch.full((10,), -1.0)
    short = mined([1, 2], [0, 1], "dead_end")
    long = mined([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5], "dead_end")
    scores = tl._trajectory_scores(row, [short, long])
    assert scores.shape == (2,)
    assert value(scores[0]) == pytest.approx(-1.0)
    assert value(scores[1]) == pytest.approx(-1.0)
    assert value(scores[0]) == pytest.approx(value(scores[1]))


def test_trajectory_score_of_empty_index_list_is_zero():
    row = torch.full((4,), -2.0)
    scores = tl._trajectory_scores(row, [mined([1], [], "gt")])
    assert value(scores[0]) == pytest.approx(0.0)


def test_trajectory_score_reads_only_its_own_indices():
    row = torch.tensor([-1.0, -3.0, -5.0])
    scores = tl._trajectory_scores(row, [mined([1], [1], "loop")])
    assert value(scores[0]) == pytest.approx(-3.0)


def test_trajectory_score_keeps_gradient():
    row = torch.tensor([-1.0, -3.0, -5.0], requires_grad=True)
    tl._trajectory_scores(row, [mined([1], [0, 2], "goal")]).sum().backward()
    assert value(row.grad[0]) == pytest.approx(0.5)
    assert value(row.grad[2]) == pytest.approx(0.5)
    assert value(row.grad[1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# classify_failure / build_gt_trajectory
# ---------------------------------------------------------------------------
def test_classify_failure_maps_each_status():
    assert classify_failure(_FakeCandidate([1], "goal")) is None
    assert classify_failure(_FakeCandidate([1], "loop")) == "loop"
    assert classify_failure(
        _FakeCandidate([1], "broken", "NULL selected at 3")
    ) == "null"
    assert classify_failure(
        _FakeCandidate([1], "broken", "no decision variable for 7")
    ) == "dead_end"
    assert classify_failure(_FakeCandidate([1], "broken", "dead end at 9")) == "dead_end"
    assert classify_failure(_FakeCandidate([1], "broken", "step limit exceeded")) == "broken"


def test_build_gt_trajectory_keeps_only_active_decisions():
    sample = make_manual_sample()
    # NULL decision 不进 GT score：它们没有"被选中的 branch"这回事。
    gt = build_gt_trajectory(sample, [0, 1, 2, 3], [False, True, False, True])
    assert gt.candidate_indices == [0, 2]
    assert gt.is_gt and gt.is_success and gt.status == "gt"
    assert gt.nlcs == 1.0
    assert list(gt.nodes) == [int(v) for v in sample.gt_path]


def test_failure_cost_of_success_is_zero():
    assert mined([1], [0], "goal").failure_cost() == 0.0
    # status="gt" 也是 success —— 不能因为忘了给 is_gt=True 就按 1.0 罚
    assert mined([1], [0], "gt").failure_cost() == 0.0
    assert MinedTrajectory([1], [0], "gt").is_success is True
    assert mined([1], [0], "loop").failure_cost() == pytest.approx(1.0)
    assert mined([1], [0], "null").failure_cost() == pytest.approx(1.5)


def test_failure_cost_uses_the_passed_table():
    """代价表必须来自调用方（cfg），不是类上写死的默认值。"""
    assert mined([1], [0], "loop").failure_cost({"loop": 7.0}) == pytest.approx(7.0)
    # 表里没有的类别退回 1.0，而不是崩
    assert mined([1], [0], "broken").failure_cost({"loop": 7.0}) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------
def test_config_defaults_to_disabled():
    assert TrajectoryLossConfig.from_config(None).enabled is False
    assert TrajectoryLossConfig.from_config({}).enabled is False
    # 没有 trajectory 段：默认关闭（两个对照 config 的行为逐位不变）
    assert TrajectoryLossConfig.from_config(FakeSection()).enabled is False
    assert TrajectoryLossConfig().describe() == "trajectory=off"


def test_config_defaults_match_the_report():
    config = TrajectoryLossConfig()
    assert (config.top_k, config.beam_width) == (2, 8)
    assert config.null_policy == "stop"
    assert config.filter_dead_branches is False
    assert config.strict is False
    assert (config.max_success, config.max_failure) == (4, 4)
    assert config.temperature == pytest.approx(1.0)
    assert config.similarity == "nlcs"
    assert config.similarity_beta == pytest.approx(3.0)
    assert config.weight == pytest.approx(0.5)
    assert (config.success_weight, config.similarity_weight) == (1.0, 1.0)
    assert config.failure_weight == pytest.approx(0.5)
    assert config.failure_cost == DEFAULT_FAILURE_COST
    assert config.timestep == 1


def test_config_parses_the_report_block():
    config = TrajectoryLossConfig.from_config(
        FakeSection(
            {
                "trajectory": {
                    "enabled": True,
                    "top_k": 2,
                    "beam_width": 8,
                    "null_policy": "stop",
                    "filter_dead_branches": False,
                    "strict": False,
                    "max_success": 4,
                    "max_failure": 4,
                    "temperature": 1.0,
                    "similarity": "nlcs",
                    "similarity_beta": 3.0,
                    "weight": 0.50,
                    "success_weight": 1.0,
                    "similarity_weight": 1.0,
                    "failure_weight": 0.50,
                    "failure_cost": {
                        "null": 1.50, "loop": 1.0, "dead_end": 1.0, "broken": 1.0
                    },
                    "timestep": 1,
                }
            }
        )
    )
    assert config.enabled is True
    # 训练 miner 必须是历史 decoder：strict=False 才看得见失败轨迹
    assert config.strict is False
    assert config.filter_dead_branches is False
    assert config.failure_cost["null"] == pytest.approx(1.5)
    assert "trajectory[" in config.describe()


def test_config_failure_cost_overrides_one_key_only():
    config = TrajectoryLossConfig.from_config(
        FakeSection({"trajectory": {"enabled": True, "failure_cost": {"null": 9.0}}})
    )
    assert config.failure_cost["null"] == pytest.approx(9.0)
    # 没写的键保留默认，不是被清掉
    assert config.failure_cost["loop"] == pytest.approx(DEFAULT_FAILURE_COST["loop"])


# ---------------------------------------------------------------------------
# 关掉 / 没有样本
# ---------------------------------------------------------------------------
def test_disabled_config_returns_zero_loss_and_no_metrics(monkeypatch):
    samples, batch = two_sample_batch()
    patch_miner(monkeypatch, {})
    loss, metrics = trajectory_set_loss(
        uniform_log_prob(batch), batch, TrajectoryLossConfig()
    )
    assert value(loss) == 0.0
    assert metrics == {}


def test_batch_without_graph_samples_returns_zero():
    """旧调用方直接手搓 Batch（没有 graph_samples）时必须安全退化，不能崩。"""

    class Bare:
        graph_samples = ()
        target_candidate = torch.zeros(0, dtype=torch.long)
        candidate_is_null = torch.zeros(0, dtype=torch.bool)

    loss, metrics = trajectory_set_loss(
        torch.zeros(4, requires_grad=True), Bare(), enabled_config()
    )
    assert value(loss) == 0.0
    assert metrics == {}


# ---------------------------------------------------------------------------
# 候选池 / 三个子 loss
# ---------------------------------------------------------------------------
def test_pool_is_only_gt_when_miner_finds_nothing(monkeypatch):
    """一条 generated 轨迹都没有：三个子 loss 恒为 0，但样本仍进 batch 平均。

    注意 ``num_success`` 仍然是 **1**（GT 恒在 success 里）—— 早先版本在这里
    短路并手写 0，日志上会看到"success_mass = 1 但 num_success = 0"这种自相矛盾。
    """
    samples, batch = two_sample_batch()
    patch_miner(monkeypatch, {})
    loss, metrics = trajectory_set_loss(
        uniform_log_prob(batch), batch, enabled_config(weight=0.5)
    )
    assert value(loss) == pytest.approx(0.0, **ZERO)
    assert metrics["num_candidates"] == pytest.approx(1.0)
    assert metrics["num_success"] == pytest.approx(1.0)
    assert metrics["num_failure"] == pytest.approx(0.0)
    assert metrics["success_mass"] == pytest.approx(1.0)
    assert metrics["failure_mass"] == pytest.approx(0.0)
    assert metrics["success_loss"] == pytest.approx(0.0, **ZERO)
    assert metrics["similarity_loss"] == pytest.approx(0.0, **ZERO)
    assert metrics["failure_loss"] == pytest.approx(0.0)
    assert metrics["mean_success_nlcs"] == pytest.approx(1.0)
    assert metrics["raw_finished"] == pytest.approx(0.0)


def test_pool_dedupes_trajectory_identical_to_gt(monkeypatch):
    """miner 采回一条和 GT 节点完全相同的轨迹 -> 去重后池子只剩 GT。"""
    samples, batch = two_sample_batch()
    gt_nodes = list(samples[0].gt_path)
    patch_miner(
        monkeypatch,
        {id(samples[0]): ([mined(gt_nodes, [0], "goal", 1.0)], [])},
    )
    loss, metrics = trajectory_set_loss(uniform_log_prob(batch), batch, enabled_config())
    assert value(loss) == pytest.approx(0.0, **ZERO)
    assert metrics["num_candidates"] == pytest.approx(1.0)


def test_success_mass_and_failure_mass_sum_to_one(monkeypatch):
    samples, batch = two_sample_batch()
    patch_miner(
        monkeypatch,
        plan_all(
            samples,
            [mined([9, 9], [0, 1], "goal", 0.5)],
            [mined([8, 8], [2, 3], "loop")],
        ),
    )
    loss, metrics = trajectory_set_loss(uniform_log_prob(batch), batch, enabled_config())
    assert torch.isfinite(loss)
    assert metrics["success_mass"] + metrics["failure_mass"] == pytest.approx(1.0)
    assert metrics["num_candidates"] == pytest.approx(3.0)   # GT + 1 success + 1 failure
    assert metrics["num_success"] == pytest.approx(2.0)
    assert metrics["num_failure"] == pytest.approx(1.0)


def test_similarity_loss_is_zero_with_a_single_success(monkeypatch):
    """success 集合只有 GT 时 q = [1]、pi_hat = [1]，L_sim 必须恰好是 0。"""
    samples, batch = two_sample_batch()
    patch_miner(
        monkeypatch,
        plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]),
    )
    _, metrics = trajectory_set_loss(uniform_log_prob(batch), batch, enabled_config())
    assert metrics["similarity_loss"] == pytest.approx(0.0, **ZERO)
    assert metrics["num_success"] == pytest.approx(1.0)
    assert metrics["success_loss"] > 0.0
    assert metrics["failure_loss"] > 0.0


def test_similarity_loss_is_positive_with_two_successes(monkeypatch):
    samples, batch = two_sample_batch()
    patch_miner(
        monkeypatch,
        plan_all(
            samples,
            [
                mined([9, 9], [0, 1], "goal", 0.9),
                mined([7, 7], [2, 3], "goal", 0.1),
            ],
            [],
        ),
    )
    _, metrics = trajectory_set_loss(uniform_log_prob(batch), batch, enabled_config())
    assert metrics["similarity_loss"] > 0.0
    assert metrics["failure_loss"] == pytest.approx(0.0)
    assert metrics["failure_mass"] == pytest.approx(0.0)
    # 池子 = GT(1.0) + 两条 success(0.9, 0.1)，全部是 success
    assert metrics["mean_success_nlcs"] == pytest.approx((1.0 + 0.9 + 0.1) / 3.0)


def test_loss_decreases_when_gt_log_prob_increases(monkeypatch):
    """L_succ 必须把质量推向"到达 goal"：GT 概率越高，loss 越低。"""
    samples, batch = two_sample_batch()
    patch_miner(
        monkeypatch,
        plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]),
    )
    config = enabled_config()
    low, _ = trajectory_set_loss(log_prob_row(batch), batch, config)

    # 只抬 GT 自己的 active candidate（不能抬整行：S 是 mean，整行同抬比例不变）
    boost = {index: 5.0 for index in gt_active_indices(samples[0])}
    assert boost, "手工样本的 GT 必须有 active decision"
    high, _ = trajectory_set_loss(log_prob_row(batch, boost), batch, config)

    assert value(high) < value(low)


def test_loss_increases_when_failures_become_more_likely(monkeypatch):
    samples, batch = two_sample_batch()
    patch_miner(
        monkeypatch,
        plan_all(
            samples,
            [mined([9, 9], [0, 1], "goal", 0.8)],
            [mined([8, 8], [2, 3], "loop")],
        ),
    )
    config = enabled_config()
    _, base = trajectory_set_loss(log_prob_row(batch), batch, config)
    # 把 loop 那条轨迹的候选概率抬起来（下标 2/3 是**样本局部**下标）
    _, shifted = trajectory_set_loss(
        log_prob_row(batch, {2: 5.0, 3: 5.0}), batch, config
    )
    assert shifted["failure_mass"] > base["failure_mass"]
    assert shifted["success_mass"] < base["success_mass"]
    assert shifted["failure_loss"] > base["failure_loss"]


def test_null_termination_costs_more_than_a_loop(monkeypatch):
    """默认代价表：NULL 终止 1.50 > loop / dead-end / broken 1.00。"""
    samples, batch = two_sample_batch()
    config = enabled_config()

    patch_miner(monkeypatch, plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]))
    _, loop_metrics = trajectory_set_loss(log_prob_row(batch), batch, config)

    patch_miner(monkeypatch, plan_all(samples, [], [mined([8, 8], [0, 1], "null")]))
    _, null_metrics = trajectory_set_loss(log_prob_row(batch), batch, config)

    # 两个池子都是 [GT, 失败]，等 log-prob 下 pi = [0.5, 0.5]
    assert null_metrics["failure_mass"] == pytest.approx(0.5)
    assert null_metrics["fail_null_mass"] == pytest.approx(0.5)
    assert null_metrics["fail_loop_mass"] == pytest.approx(0.0)
    assert loop_metrics["fail_loop_mass"] == pytest.approx(0.5)
    assert null_metrics["failure_loss"] == pytest.approx(0.5 * 1.5)
    assert loop_metrics["failure_loss"] == pytest.approx(0.5 * 1.0)


def test_failure_cost_from_config_actually_takes_effect(monkeypatch):
    """回归：``failure_cost`` 曾经被 ``MinedTrajectory`` 上的默认值覆盖成死配置。"""
    samples, batch = two_sample_batch()
    patch_miner(monkeypatch, plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]))
    row = log_prob_row(batch)
    cheap = enabled_config(failure_cost={**DEFAULT_FAILURE_COST, "loop": 0.1})
    pricey = enabled_config(failure_cost={**DEFAULT_FAILURE_COST, "loop": 5.0})
    _, cheap_metrics = trajectory_set_loss(row, batch, cheap)
    _, pricey_metrics = trajectory_set_loss(row, batch, pricey)
    assert cheap_metrics["failure_loss"] == pytest.approx(0.5 * 0.1)
    assert pricey_metrics["failure_loss"] == pytest.approx(0.5 * 5.0)


# ---------------------------------------------------------------------------
# batch 等权平均
# ---------------------------------------------------------------------------
def test_batch_averaging_is_sample_equal_not_candidate_equal(monkeypatch):
    """一个"挖不出东西"的样本 + 一个正常样本 -> loss 恰好是正常样本的一半。

    如果按候选混在一起平均，这个比值不会是 1/2。
    """
    samples, batch = two_sample_batch()
    active = {id(samples[0]): ([], [mined([8, 8], [0, 1], "loop")])}
    patch_miner(monkeypatch, active)
    both, _ = trajectory_set_loss(log_prob_row(batch), batch, enabled_config())

    single = collate_samples([samples[0]], device="cpu")
    patch_miner(monkeypatch, active)
    alone, _ = trajectory_set_loss(log_prob_row(single), single, enabled_config())
    assert value(both) == pytest.approx(value(alone) / 2.0)


def test_weight_scales_the_whole_trajectory_loss(monkeypatch):
    samples, batch = two_sample_batch()
    patch_miner(monkeypatch, plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]))
    half, half_metrics = trajectory_set_loss(
        log_prob_row(batch), batch, enabled_config(weight=0.5)
    )
    full, full_metrics = trajectory_set_loss(
        log_prob_row(batch), batch, enabled_config(weight=1.0)
    )
    assert value(full) == pytest.approx(2.0 * value(half))
    # 权重只缩放 loss，不改指标
    assert full_metrics["failure_loss"] == pytest.approx(half_metrics["failure_loss"])


def test_success_weight_can_be_turned_off(monkeypatch):
    samples, batch = two_sample_batch()
    patch_miner(
        monkeypatch,
        plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]),
    )
    base, _ = trajectory_set_loss(log_prob_row(batch), batch, enabled_config(weight=1.0))
    success_off, metrics = trajectory_set_loss(
        log_prob_row(batch), batch,
        enabled_config(weight=1.0, success_weight=0.0),
    )
    assert value(success_off) < value(base)
    # 关掉 success 项不该改变 L_fail 本身
    assert metrics["failure_loss"] > 0.0


def test_failure_weight_zero_leaves_only_the_success_term(monkeypatch):
    samples, batch = two_sample_batch()
    patch_miner(monkeypatch, plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]))
    no_fail, metrics = trajectory_set_loss(
        log_prob_row(batch), batch, enabled_config(weight=1.0, failure_weight=0.0)
    )
    assert metrics["success_loss"] > 0.0            # success 项还在
    assert value(no_fail) == pytest.approx(metrics["success_loss"])


# ---------------------------------------------------------------------------
# 梯度
# ---------------------------------------------------------------------------
def test_gradient_flows_into_candidate_log_prob(monkeypatch):
    """搜索用 detach 的概率，但打分必须用**未 detach** 的 log-prob。"""
    samples, batch = two_sample_batch()
    patch_miner(monkeypatch, plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]))
    row = log_prob_row(batch)
    loss, _ = trajectory_set_loss(row, batch, enabled_config())
    assert loss.requires_grad
    loss.backward()
    assert row.grad is not None
    assert value(row.grad.abs().sum()) > 0.0


def test_gradient_is_finite_with_zero_log_prob(monkeypatch):
    """全零 log-prob（= 概率 1）下不能出现 inf/nan：_EPS 必须真的兜住。"""
    samples, batch = two_sample_batch()
    patch_miner(monkeypatch, plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]))
    row = log_prob_row(batch, value=0.0)
    loss, _ = trajectory_set_loss(row, batch, enabled_config())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(row.grad).all()


def test_gradient_pushes_towards_the_goal_trajectory():
    """手工样本上不打桩：GT 的 active candidate 应该拿到负梯度（= 概率被推高）。"""
    samples, batch = two_sample_batch()
    row = log_prob_row(batch)
    loss, _ = trajectory_set_loss(row, batch, enabled_config())
    loss.backward()
    indices = gt_active_indices(samples[0])
    assert indices
    for index in indices:
        # d(loss)/d(log p) 为负 = loss 随 log p 上升而下降 = 在鼓励这条轨迹
        assert value(row.grad[index]) < 0.0


# ---------------------------------------------------------------------------
# 指标键与日志字段对齐
# ---------------------------------------------------------------------------
def test_metric_keys_are_all_real_log_fields():
    """每个指标键都要能在 ``RecurrentLossOutput`` 里找到 ``traj_<key>``。

    对不上就会让 ``RecurrentLossOutput(**metrics)`` 抛 TypeError —— 这是纯静态
    契约，不需要跑训练就能守住。
    """
    import dataclasses

    fields = {field.name for field in dataclasses.fields(RecurrentLossOutput)}
    for key in METRIC_KEYS:
        assert f"traj_{key}" in fields, key


def test_no_declared_traj_field_is_missing_from_metric_keys():
    """反向也要守：声明了 ``traj_*`` 字段却没有对应指标键，日志里就永远是 NaN。"""
    import dataclasses

    fields = {field.name for field in dataclasses.fields(RecurrentLossOutput)}
    declared = {name[5:] for name in fields if name.startswith("traj_")}
    assert declared == set(METRIC_KEYS)


def test_trainer_logs_every_declared_metric():
    """trainer 的 ``EXTRA_TRAIN_METRICS`` 不能有拼错的名字。

    ``Trainer`` 用 ``getattr(out, name, None)`` 取值并**静默跳过**取不到的字段，
    所以一个拼错的键不会报错，只会让那条曲线永远缺席 —— 必须静态守住。
    """
    import dataclasses

    from src.training.trainer import (
        EXTRA_TRAIN_METRICS,
        SAMPLED_NULL_METRICS,
        TRAJECTORY_METRICS,
    )

    fields = {field.name for field in dataclasses.fields(RecurrentLossOutput)}
    for name in EXTRA_TRAIN_METRICS:
        assert name in fields, name
    # 两组指标都要在里面（不是只有一组）
    assert set(SAMPLED_NULL_METRICS) <= set(EXTRA_TRAIN_METRICS)
    assert set(TRAJECTORY_METRICS) <= set(EXTRA_TRAIN_METRICS)
    # 多轨迹那组 = 加权后的总项 + METRIC_KEYS 加前缀，
    # 不允许 trainer 和 loss 两边各维护一份不同的清单
    assert set(TRAJECTORY_METRICS) == (
        {"trajectory_loss"} | {f"traj_{key}" for key in METRIC_KEYS}
    )


def test_curve_keys_point_at_real_fields():
    """``CURVE_KEYS`` 里的 traj / NULL 饱和曲线也要真的存在，否则永远画不出来。

    只查本轮新增的这两组：历史上 ``train_x0_acc`` 这类键是 trainer 单独拼出来的
    （字段其实叫 ``final_accuracy``），不能按"去掉前缀就是字段名"来要求。
    """
    import dataclasses

    from src.evaluation.history import CURVE_KEYS

    fields = {field.name for field in dataclasses.fields(RecurrentLossOutput)}
    wanted = [
        key for key in CURVE_KEYS
        if key.startswith("train_traj_") or key == "train_null_saturation_rate"
    ]
    assert wanted, "CURVE_KEYS 里没有本轮新增的曲线"
    for key in wanted:
        assert key[len("train_"):] in fields, key
    # 19 个 trajectory 指标 + 1 个饱和率，一个都不能漏
    assert len(wanted) == len(METRIC_KEYS) + 1


def test_metrics_only_expose_declared_keys(monkeypatch):
    samples, batch = two_sample_batch()
    patch_miner(monkeypatch, plan_all(samples, [], [mined([8, 8], [0, 1], "loop")]))
    _, metrics = trajectory_set_loss(uniform_log_prob(batch), batch, enabled_config())
    assert set(metrics) <= set(METRIC_KEYS)
    assert all(isinstance(item, float) for item in metrics.values())


def test_raw_stats_are_counted_before_truncation(monkeypatch):
    """``raw_*`` 必须是**截断前**的产量，否则 mining budget 消融看不出预算够不够。"""
    samples, batch = two_sample_batch()
    successes = [
        _FakeCandidate([90, 100 + index], "goal", indices=[0]) for index in range(6)
    ]
    failures = [
        _FakeCandidate([80, 200 + index], "loop", indices=[1]) for index in range(5)
    ]
    patch_decoder(monkeypatch, successes + failures)

    success, failure, stats = mine_success_and_failure(
        samples[0], [0.5] * samples[0].num_candidates, enabled_config(
            max_success=2, max_failure=2
        )
    )
    assert len(success) == 2
    assert len(failure) == 2
    # raw_* 记的是截断前的 6 / 5
    assert stats["raw_success"] == 6
    assert stats["raw_loop"] == 5
    assert stats["raw_finished"] == 11

    # 走完整条 loss 路径：池子 = GT + 2 success + 2 failure = 5
    patch_decoder(monkeypatch, successes + failures)
    _, metrics = trajectory_set_loss(
        log_prob_row(batch), batch,
        enabled_config(max_success=2, max_failure=2),
    )
    assert metrics["num_candidates"] == pytest.approx(5.0)
    assert metrics["num_success"] == pytest.approx(3.0)
    assert metrics["num_failure"] == pytest.approx(2.0)
    assert metrics["raw_success"] == pytest.approx(6.0)
    assert metrics["raw_loop"] == pytest.approx(5.0)


def test_failure_pick_prefers_one_of_each_type(monkeypatch):
    """失败候选先按类型各取一条，保证 NULL / loop / dead-end 都进得来。"""
    failures = [
        _FakeCandidate([80, 200], "loop", indices=[0]),
        _FakeCandidate([80, 201], "loop", indices=[1]),
        _FakeCandidate([80, 202], "broken", "NULL selected at 4", indices=[2]),
        _FakeCandidate([80, 203], "broken", "no decision variable for 5", indices=[3]),
        _FakeCandidate([80, 204], "broken", "step limit exceeded", indices=[4]),
    ]
    patch_decoder(monkeypatch, failures)
    samples, batch = two_sample_batch()
    config = enabled_config(max_failure=4)
    success, picked, stats = mine_success_and_failure(
        samples[0], [0.5] * samples[0].num_candidates, config
    )
    assert success == []
    assert len(picked) == 4
    # 类型多样性：四种各一条，第二条 loop 被挤掉
    assert {item.status for item in picked} == {"loop", "null", "dead_end", "broken"}
    assert stats["raw_loop"] == 2

    patch_decoder(monkeypatch, failures)
    _, metrics = trajectory_set_loss(log_prob_row(batch), batch, config)
    assert metrics["num_failure"] == pytest.approx(4.0)
    assert metrics["num_candidates"] == pytest.approx(5.0)   # GT + 4
    for kind in ("null", "loop", "dead", "broken"):
        assert metrics[f"fail_{kind}_mass"] > 0.0


def test_failure_pick_fills_remaining_slots_by_score(monkeypatch):
    """类型各一条之后还没满，就用**同一个** score（mean log p）补高分失败轨迹。"""
    failures = [
        _FakeCandidate([80, 200], "loop", indices=[0]),
        _FakeCandidate([80, 201], "loop", indices=[1]),
        _FakeCandidate([80, 202], "loop", indices=[2]),
    ]
    patch_decoder(monkeypatch, failures)
    sample = make_manual_sample()
    config = enabled_config(max_failure=2)
    probs = [0.5] * sample.num_candidates
    probs[2] = 0.9          # index 2 分数最高
    probs[1] = 0.7          # index 1 次高
    _, picked, _ = mine_success_and_failure(sample, probs, config)
    assert len(picked) == 2
    # 类型代表 = 该类型 mean log p 最高的那条（index 2）；剩下的 1 个名额再按同一个
    # 分数补 index 1。注意 index 2 在 decoder 给出的顺序里排在最后 —— 它当上代表
    # 本身就说明类型内部重排过了。
    assert [item.nodes for item in picked] == [[80, 202], [80, 201]]


def test_failure_type_representative_uses_mean_log_prob(monkeypatch):
    """每种 failure 的第一条必须按 S(P) = mean log p 选，不能继承 decoder 的顺序。

    decoder（搜索阶段）按**累计** prefix log p 排 finished，而累计口径偏好短轨迹：

        短 3 步 × p=0.75 -> 累计 -0.863，平均 -0.288
        长 6 步 × p=0.85 -> 累计 -0.975，平均 -0.163

    两个口径的排序**正好相反**。这里把短的那条先传给 miner（模拟 decoder 的输出
    顺序），只有真正按 mean 重排过，max_failure=1 时留下的才会是长的那条。
    """
    short_loop = _FakeCandidate([90, 100], "loop", indices=[0, 1, 2])
    long_loop = _FakeCandidate([90, 101], "loop", indices=[3, 4, 5, 6, 7, 8])
    # decoder 的累计顺序：短的在前
    patch_decoder(monkeypatch, [short_loop, long_loop])
    sample = make_manual_sample()
    assert sample.num_candidates >= 9, "手工样本的候选数不够构造这个反例"
    # 反例成立的前提：累计口径下短的赢，平均口径下长的赢
    assert 3 * math.log(0.75) > 6 * math.log(0.85)
    assert math.log(0.75) < math.log(0.85)
    probs = [0.5] * sample.num_candidates
    for index in (0, 1, 2):
        probs[index] = 0.75
    for index in (3, 4, 5, 6, 7, 8):
        probs[index] = 0.85

    success, picked, _ = mine_success_and_failure(
        sample, probs, enabled_config(max_failure=1)
    )

    assert success == []
    assert len(picked) == 1
    assert picked[0].status == "loop"
    assert [item.nodes for item in picked] == [[90, 101]], (
        "该类型的代表不是 mean log p 最高的那条 —— 说明 failure 候选仍在继承 "
        "decoder 的累计 log p 顺序（短轨迹偏置又回来了）"
    )


def test_failure_type_representatives_are_truncated_by_score_not_type_order(monkeypatch):
    """``max_failure`` 小于 failure 类型数时，按 mean log p 截断，而不是按类型顺序。

    默认 ``max_failure=4`` 刚好等于类型数，所以这条分支平时走不到；一旦调小
    （消融里很常见），按类型顺序截断会**恒定偏向** NULL / loop，等于偷偷给类型排了
    优先级。这里构造四条分数各异的代表，只有按分数截断才会得到 loop + broken。
    """
    null_lowest = _FakeCandidate([90, 100], "broken", "NULL selected at 5", indices=[0])
    loop_highest = _FakeCandidate([90, 101], "loop", indices=[1])
    dead_middle = _FakeCandidate([90, 102], "broken", "dead end at 6", indices=[2])
    broken_second = _FakeCandidate(
        [90, 103], "broken", "step limit exceeded", indices=[3]
    )
    patch_decoder(
        monkeypatch, [null_lowest, loop_highest, dead_middle, broken_second]
    )
    sample = make_manual_sample()
    probs = [0.5] * sample.num_candidates
    probs[0] = 0.10     # null      -> 最低
    probs[1] = 0.90     # loop      -> 最高
    probs[2] = 0.50     # dead_end  -> 中间
    probs[3] = 0.80     # broken    -> 次高

    _, picked, _ = mine_success_and_failure(
        sample, probs, enabled_config(max_failure=2)
    )

    # 类型顺序是 (null, loop, dead_end, broken)：按顺序截断会得到 [null, loop]，
    # 按 mean log p 截断才是 [loop, broken]
    assert [item.status for item in picked] == ["loop", "broken"]


def test_failure_type_representatives_keep_one_per_type_when_room_allows(monkeypatch):
    """``max_failure >= 类型数`` 时仍然保持"一类至少一条"的多样性设计。"""
    candidates = [
        _FakeCandidate([90, 100], "broken", "NULL selected at 5", indices=[0]),
        _FakeCandidate([90, 101], "loop", indices=[1]),
        _FakeCandidate([90, 102], "broken", "dead end at 6", indices=[2]),
        _FakeCandidate([90, 103], "broken", "step limit exceeded", indices=[3]),
    ]
    patch_decoder(monkeypatch, candidates)
    sample = make_manual_sample()
    probs = [0.5] * sample.num_candidates
    probs[0] = 0.10     # null 分数最低，也必须进（多样性）
    _, picked, _ = mine_success_and_failure(
        sample, probs, enabled_config(max_failure=4)
    )
    assert {item.status for item in picked} == {"null", "loop", "dead_end", "broken"}


def test_success_pool_takes_the_most_probable_first(monkeypatch):
    """成功候选按 mean log 概率降序取前 ``max_success``。"""
    successes = [
        _FakeCandidate([90, 100], "goal", indices=[0]),
        _FakeCandidate([90, 101], "goal", indices=[1]),
        _FakeCandidate([90, 102], "goal", indices=[2]),
    ]
    patch_decoder(monkeypatch, successes)
    samples, batch = two_sample_batch()
    # index 1 的概率最高 -> max_success=1 时应该选它
    row = log_prob_row(batch, {0: -5.0, 1: -0.1, 2: -3.0})
    _, metrics = trajectory_set_loss(
        row, batch, enabled_config(max_success=1)
    )
    assert metrics["num_success"] == pytest.approx(2.0)      # GT + 1
    # 被选中的那条 nlcs = 0（假候选没有真实节点），GT 的 nlcs = 1
    assert metrics["mean_success_nlcs"] == pytest.approx(0.5)


def test_success_candidates_are_ordered_by_mean_log_prob(monkeypatch):
    """挑候选与最终 S(P) 必须用**同一个**分数：长度归一化的 mean log p。

    早先筛选用累计 log p，于是"按累计挑进来、按平均打分"，被选中的恰好是平均分
    更低的那批（累计口径偏好短轨迹）。这里用一组"短但单步概率低 / 长但单步概率高"
    的候选把这个偏置钉死。
    """
    zero = _FakeCandidate([90, 100], "goal", indices=[0])
    one = _FakeCandidate([90, 101], "goal", indices=[1])
    two = _FakeCandidate([90, 102], "goal", indices=[2])
    patch_decoder(monkeypatch, [zero, one, two])
    sample = make_manual_sample()
    probs = [0.5] * sample.num_candidates
    probs[0] = 0.4
    probs[1] = 0.6
    probs[2] = 0.9
    kept, _, _ = mine_success_and_failure(sample, probs, enabled_config(max_success=1))
    assert [item.nodes for item in kept] == [[90, 102]]


def test_candidate_selection_is_not_biased_towards_short_trajectories(monkeypatch):
    """长度归一化：长而单步概率高的轨迹必须赢过短而单步概率低的。

    具体数字（手工样本只有 9 个候选，所以用 3 步 vs 6 步）：
        短  3 步 × p=0.75 -> 累计 -0.863，均值 -0.288   <- 累计口径选它
        长  6 步 × p=0.85 -> 累计 -0.975，均值 -0.163   <- 平均口径选它
    两个口径的**排序正好相反**，所以 max_success=1 时留下的那条就能分辨实现。
    """
    short = _FakeCandidate([90, 100], "goal", indices=[0, 1, 2])
    long = _FakeCandidate([90, 101], "goal", indices=[3, 4, 5, 6, 7, 8])
    patch_decoder(monkeypatch, [short, long])
    sample = make_manual_sample()
    assert sample.num_candidates >= 9, "手工样本的候选数不够构造这个反例"
    # 反例成立的前提：累计口径下短的那条赢
    assert 3 * math.log(0.75) > 6 * math.log(0.85)
    assert math.log(0.75) < math.log(0.85)
    probs = [0.5] * sample.num_candidates
    for index in (0, 1, 2):
        probs[index] = 0.75
    for index in (3, 4, 5, 6, 7, 8):
        probs[index] = 0.85

    kept, _, _ = mine_success_and_failure(sample, probs, enabled_config(max_success=1))
    assert [item.nodes for item in kept] == [[90, 101]], (
        "累计口径会选短轨迹；长度归一化后应当选模型更相信的长轨迹"
    )


# ---------------------------------------------------------------------------
# 空 trace（模型零 decision）不能进候选池
# ---------------------------------------------------------------------------
def test_forced_failure_without_any_decision_is_excluded(monkeypatch):
    """forced walk 在第一个 decision 之前就断了 -> 模型没有任何选择权。

    这种轨迹的 ``S(P)`` 是空和（记 0），而 0 是所有候选里的**最大值**：放进去
    softmax 会把质量送给一条根本没法优化的轨迹。必须剔除并计数。
    """
    forced = _FakeCandidate(
        [80, 200], "broken", "dead end before any decision", indices=[]
    )
    real = _FakeCandidate([80, 201], "loop", indices=[0])
    patch_decoder(monkeypatch, [forced, real])
    sample = make_manual_sample()

    success, failure, stats = mine_success_and_failure(
        sample, [0.5] * sample.num_candidates, enabled_config()
    )
    assert success == []
    assert [item.nodes for item in failure] == [[80, 201]]
    assert stats["raw_no_decision"] == 1
    # raw_finished 仍然记"截断前一共产出多少条"（含被剔除的那条）
    assert stats["raw_finished"] == 2
    assert stats["raw_loop"] == 1


def test_forced_goal_without_any_decision_is_excluded(monkeypatch):
    """forced walk 直接走到 goal：也是一条模型零 decision 的路径。"""
    forced_goal = _FakeCandidate([90, 100], "goal", indices=[])
    patch_decoder(monkeypatch, [forced_goal])
    samples, batch = two_sample_batch()
    _, metrics = trajectory_set_loss(
        uniform_log_prob(batch), batch, enabled_config()
    )
    # 池子里只剩 GT：success 集合不会被这条"白捡的成功"灌水
    assert metrics["num_candidates"] == pytest.approx(1.0)
    assert metrics["num_success"] == pytest.approx(1.0)
    assert metrics["success_mass"] == pytest.approx(1.0)
    assert metrics["raw_no_decision"] == pytest.approx(1.0)


def test_empty_trace_does_not_steal_softmax_mass(monkeypatch):
    """剔除空 trace 之后，其余候选的分值与 loss 必须与"它压根没出现过"完全一致。

    这就是这条 guard 的全部意义：空 trace 若以 S=0 入池，会稀释真正该学的候选。
    """
    real = _FakeCandidate([80, 200], "loop", indices=[0])
    samples, batch = two_sample_batch()
    config = enabled_config()

    patch_decoder(monkeypatch, [real])
    without, metrics_without = trajectory_set_loss(
        log_prob_row(batch), batch, config
    )
    patch_decoder(
        monkeypatch,
        [_FakeCandidate([80, 201], "broken", "ambiguous forced step", indices=[]), real],
    )
    with_empty, metrics_with = trajectory_set_loss(
        log_prob_row(batch), batch, config
    )

    assert value(with_empty) == pytest.approx(value(without))
    assert metrics_with["failure_mass"] == pytest.approx(metrics_without["failure_mass"])
    assert metrics_with["num_candidates"] == pytest.approx(
        metrics_without["num_candidates"]
    )
    # 唯一的变化是"被剔除了几条"这件事本身被记下来了
    assert metrics_with["raw_no_decision"] == pytest.approx(1.0)
    assert metrics_without["raw_no_decision"] == pytest.approx(0.0)


def test_gt_is_kept_even_when_it_has_no_decisions(monkeypatch):
    """GT 永远进池 —— 它是监督目标，不走 miner 的剔除逻辑。

    GT 一个 active decision 都没有时 ``S(GT)=0`` 在语义上是对的（forced 路径的
    概率恒为 1），此时这个样本对 L_traj 贡献恰好 0，不会把 loss 带偏。
    """
    patch_decoder(monkeypatch, [])
    samples, batch = two_sample_batch()
    # 把 GT 的 target 全设成 NULL -> build_gt_trajectory 得到空 indices
    batch.target_candidate.fill_(0)
    batch.candidate_is_null.fill_(True)
    loss, metrics = trajectory_set_loss(
        uniform_log_prob(batch), batch, enabled_config()
    )
    assert value(loss) == pytest.approx(0.0, **ZERO)
    assert metrics["num_candidates"] == pytest.approx(1.0)
    assert metrics["num_success"] == pytest.approx(1.0)


def test_raw_no_decision_is_a_declared_log_field():
    """新指标键必须同时存在于 METRIC_KEYS 与 RecurrentLossOutput 里。"""
    assert "raw_no_decision" in METRIC_KEYS
    import dataclasses

    fields = {field.name for field in dataclasses.fields(RecurrentLossOutput)}
    assert "traj_raw_no_decision" in fields


# ---------------------------------------------------------------------------
# 真实 miner（不打桩）：确认它和 decoder 的接口没漂
# ---------------------------------------------------------------------------
def test_real_miner_sees_failures_on_the_manual_graph():
    """手工图 J1 有通向 dead-end 的分支、J2 有回到 J1 的环，miner 必须能挖到。"""
    sample = make_manual_sample()
    probs = [1.0 / sample.num_candidates] * sample.num_candidates
    success, failure, stats = mine_success_and_failure(sample, probs, enabled_config())
    classified = sum(
        stats[key]
        for key in ("raw_success", "raw_null", "raw_loop", "raw_dead_end", "raw_broken")
    )
    # raw_finished 是截断前的 finished 总数（含被跳过的 GT 重复项）
    assert stats["raw_finished"] >= classified
    assert len(success) <= TrajectoryLossConfig().max_success
    assert len(failure) <= TrajectoryLossConfig().max_failure
    # 每条 trajectory 的 candidate_indices 都必须是**样本局部**下标
    for item in list(success) + list(failure):
        assert item.candidate_indices
        assert 0 <= min(item.candidate_indices)
        assert max(item.candidate_indices) < sample.num_candidates
    assert all(item.status == "goal" for item in success)


def test_real_miner_keeps_strict_off():
    """训练 miner 必须是历史 decoder：strict=False、filter_dead_branches=False。

    这是**不能改**的一条：strict 会在 top-k 之前把 NULL / loop / dead-end 全 mask
    掉，miner 就再也看不到失败轨迹，L_fail 恒为 0、整个多轨迹项失去意义。
    """
    config = TrajectoryLossConfig()
    assert config.strict is False
    assert config.filter_dead_branches is False
    assert config.null_policy == "stop"


def test_multi_sample_batch_uses_local_candidate_offsets(monkeypatch):
    """回归：``batch.target_candidate`` 是全局下标，GT 必须减掉样本起点。

    修复前第二个样本的 GT 下标（全局）会越出它自己的候选切片，直接 IndexError。
    """
    samples, batch = two_sample_batch()
    first, second = samples
    patch_miner(
        monkeypatch,
        {
            id(first): ([], [mined([8, 8], [0, 1], "loop")]),
            id(second): ([], [mined([7, 7], [0, 1], "loop")]),
        },
    )
    loss, metrics = trajectory_set_loss(log_prob_row(batch), batch, enabled_config())
    assert torch.isfinite(loss)
    # 每个样本的池子 = GT + 1 条失败 = 2，跨样本平均仍是 2
    assert metrics["num_candidates"] == pytest.approx(2.0)
    assert metrics["num_failure"] == pytest.approx(1.0)
    assert metrics["num_success"] == pytest.approx(1.0)


def test_gt_indices_stay_inside_the_sample_slice_without_patching():
    """不打桩的真 miner + 多样本 batch：整条路径都不许越界。"""
    samples, batch = two_sample_batch()
    loss, metrics = trajectory_set_loss(log_prob_row(batch), batch, enabled_config())
    assert torch.isfinite(loss)
    assert metrics["num_candidates"] >= 1.0
    # 每个样本的 GT 都真的被算进了 success 集合
    assert metrics["num_success"] >= 1.0


# ---------------------------------------------------------------------------
# 饱和式 NULL
# ---------------------------------------------------------------------------
def test_saturating_null_zero_above_target():
    log_prob = torch.log(torch.tensor([0.60, 0.80, 0.99]))
    assert value(saturating_null_loss(log_prob, 0.60)) == pytest.approx(0.0)


def test_saturating_null_positive_below_target():
    log_prob = torch.log(torch.tensor([0.10, 0.30]))
    expected = (-torch.log(torch.tensor([0.10, 0.30])) + math.log(0.60)).mean()
    assert value(saturating_null_loss(log_prob, 0.60)) == pytest.approx(value(expected))


def test_saturating_null_is_capped_by_the_threshold():
    """再低的 p(NULL) 也只按 log(rho/p) 罚，不会像 NLL 那样上不封顶。"""
    log_prob = torch.log(torch.tensor([1e-6]))
    assert value(saturating_null_loss(log_prob, 0.60)) == pytest.approx(
        math.log(0.60 / 1e-6), rel=1e-6
    )


def test_saturating_null_has_zero_gradient_above_target():
    # log_prob 本身要是 leaf，否则 .grad 不会被填充
    log_prob = torch.tensor([math.log(0.90)], requires_grad=True)
    saturating_null_loss(log_prob, 0.60).backward()
    assert value(log_prob.grad.abs().sum()) == pytest.approx(0.0)


def test_saturating_null_gradient_is_nll_below_target():
    """阈值以下它退化成普通 NLL：d/dlogp (log rho - log p) = -1。"""
    log_prob = torch.tensor([math.log(0.20)], requires_grad=True)
    saturating_null_loss(log_prob, 0.60).backward()
    assert value(log_prob.grad) == pytest.approx(-1.0)


def test_saturating_null_empty_is_zero():
    empty = torch.zeros(0)
    assert value(saturating_null_loss(empty, 0.6)) == 0.0


# ---------------------------------------------------------------------------
# LossWeights 接线
# ---------------------------------------------------------------------------
def test_loss_weights_default_null_type_is_backward_compatible():
    """默认构造 = 旧 CE baseline；sampled_null 的 NULL 仍是普通 NLL。"""
    weights = LossWeights()
    assert weights.loss_type == "ce"
    assert weights.null_loss_type == "nll"
    assert weights.is_saturating_null is False
    assert weights.trajectory.enabled is False
    assert weights.describe() == "loss=ce(null_w=1.0, active_w=1.0)"

    sampled = LossWeights(loss_type="path_nll_sampled_null")
    assert sampled.is_saturating_null is False
    assert "sampled_null" in sampled.describe()
    assert "sat" not in sampled.describe()


def test_loss_weights_parses_saturating_null_and_trajectory():
    config = FakeConfig(
        {
            "loss": {
                "type": "path_nll_sampled_null",
                "null_loss_type": SATURATING_NLL,
                "null_target_prob": 0.6,
                "null_loss_weight": 0.10,
                "trajectory": {"enabled": True, "weight": 0.5, "timestep": 1},
            }
        }
    )
    weights = LossWeights.from_config(config)
    assert weights.is_saturating_null is True
    assert weights.null_target_prob == pytest.approx(0.6)
    assert weights.null_loss_weight == pytest.approx(0.10)
    assert weights.trajectory.enabled is True
    assert weights.trajectory.weight == pytest.approx(0.5)
    assert weights.trajectory.timestep == 1
    assert "sat(rho=0.6)" in weights.describe()
    weights.validate()


def test_loss_weights_rejects_bad_null_settings():
    def build(extra):
        return LossWeights.from_config(
            FakeConfig({"loss": {"type": "path_nll_sampled_null", **extra}})
        )

    with pytest.raises(ValueError, match="null_loss_type"):
        build({"null_loss_type": "banana"}).validate()
    with pytest.raises(ValueError, match="null_target_prob"):
        build({"null_target_prob": 0.0}).validate()
    with pytest.raises(ValueError, match="null_target_prob"):
        build({"null_target_prob": 1.0}).validate()
    with pytest.raises(ValueError, match="null_target_prob"):
        build({"null_target_prob": 1.5}).validate()


# ---------------------------------------------------------------------------
# 端到端：真的过一遍 reverse chain
# ---------------------------------------------------------------------------
def _recurrent_loss(weights, steps=4):
    from src.training.losses import recurrent_reverse_loss
    from src.training.setup import build_diffusion, build_model
    from src.utils.config import load_config

    config = load_config(PROJECT_ROOT / "configs" / "controlled_unweighted.yaml")
    torch.manual_seed(0)
    model = build_model(config)
    diffusion = build_diffusion(config)
    batch = collate_samples([make_manual_sample()], device="cpu")
    return recurrent_reverse_loss(
        model, diffusion, batch, weights=weights, max_steps=steps, record=True
    )


def test_recurrent_loss_reports_trajectory_metrics():
    weights = LossWeights(
        loss_type="path_nll_sampled_null",
        null_loss_type=SATURATING_NLL,
        null_target_prob=0.6,
        null_loss_weight=0.10,
        trajectory=enabled_config(weight=0.5),
    )
    out = _recurrent_loss(weights)
    assert torch.isfinite(out.loss)
    assert out.trajectory_loss == out.trajectory_loss            # not NaN
    assert out.traj_num_candidates >= 1.0
    assert out.traj_success_mass > 0.0
    assert out.traj_success_mass + out.traj_failure_mass == pytest.approx(1.0)
    assert out.null_saturation_rate == out.null_saturation_rate  # not NaN
    assert 0.0 <= out.null_saturation_rate <= 1.0
    out.loss.backward()


def test_recurrent_loss_trajectory_is_off_by_default():
    """两个对照 config 走的就是这条路径：traj_* 必须全是 NaN，不是 0。"""
    out = _recurrent_loss(LossWeights(loss_type="path_nll_sampled_null"))
    assert out.trajectory_loss != out.trajectory_loss            # NaN
    assert out.traj_success_mass != out.traj_success_mass


def test_recurrent_loss_honours_configured_timestep():
    """``trajectory.timestep`` 不是死配置：换成别的 reverse step 仍然算得出来。"""
    weights = LossWeights(
        loss_type="path_nll_sampled_null",
        trajectory=enabled_config(weight=0.5, timestep=3),
    )
    out = _recurrent_loss(weights, steps=4)
    assert torch.isfinite(out.loss)
    assert out.trajectory_loss == out.trajectory_loss


def test_recurrent_loss_timestep_outside_the_chain_never_fires():
    """``timestep`` 超出 reverse 链时这一项整段不参与 —— 而且不报错。"""
    weights = LossWeights(
        loss_type="path_nll_sampled_null",
        trajectory=enabled_config(weight=0.5, timestep=99),
    )
    out = _recurrent_loss(weights, steps=4)
    assert torch.isfinite(out.loss)
    assert out.trajectory_loss != out.trajectory_loss            # NaN = 从没算过


def test_recurrent_loss_goal_reach_zero_with_record_does_not_crash():
    """回归：``goal_reach_weight=0`` 且 ``record=True`` 时 soft goal 整段被跳过，
    以前 ``step_goal_loss`` 未赋值会抛 UnboundLocalError（DiDi 配置正是 0）。"""
    weights = LossWeights(
        loss_type="path_nll_sampled_null",
        goal_reach_weight=0.0,
    )
    out = _recurrent_loss(weights)
    assert len(out.per_step_goal_loss) == 4
    assert all(item != item for item in out.per_step_goal_loss)   # 全 NaN
    assert all(item != item for item in out.per_step_soft_goal)
    assert all(item == item for item in out.per_step_loss)
    assert all(item == item for item in out.per_step_accuracy)


def test_recurrent_loss_soft_goal_still_recorded_when_enabled():
    """对照组：``goal_reach_weight > 0`` 时记录的是真值，不是 NaN。"""
    weights = LossWeights(
        loss_type="path_nll_sampled_null",
        goal_reach_weight=0.1,
    )
    out = _recurrent_loss(weights)
    assert all(item == item for item in out.per_step_goal_loss)
    assert all(item == item for item in out.per_step_soft_goal)
