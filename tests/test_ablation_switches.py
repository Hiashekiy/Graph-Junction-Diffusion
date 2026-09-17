"""四个结构消融开关的单元测试（对应 docs/ABLATION_GUIDE.md）。

这四个开关**默认值必须等于改造前的 Full model**，所以本文件的第一组测试就是
回归：默认构造出来的模型必须落在 Full 行为上，且 ``branch_readout`` 的默认
表示与 ``branch_mean_pool`` 逐位一致。

另外三组分别验证开关真的生效 —— 注意最容易出错的地方：

* ``persistent_state`` 的开关加在**循环里**（losses.py / sampler.py），
  ``model.step`` 本身不知道这件事。所以测试必须跑完整条链、检查**每次 step
  收到的 H_t**，只测 step 会"测试通过但功能没生效"。
* ``use_edge_state_conditioning`` 的开关加在 ``step`` 内部取边特征的地方，
  所以要测 step 的输出对 z_t 是否敏感，而不是只测 ``expand_to_edge_state``。
* ``branch_readout`` 只改表示怎么算，候选集合 / z_t 语义 / decoder 都不许动。
"""

from __future__ import annotations

import inspect

import pytest
import torch

from src.diffusion.categorical import CategoricalDiffusion
from src.diffusion.sampler import sample_reverse_chain
from src.diffusion.schedule import NoiseSchedule
from src.models.branch_scorer import BranchScorer, branch_mean_pool
from src.models.denoiser import GraphFlowDenoiser
from src.models.edge_state import static_edge_state_ids
from src.training.losses import direct_prediction_loss, recurrent_reverse_loss


def make_diffusion(T: int = 4) -> CategoricalDiffusion:
    return CategoricalDiffusion(
        NoiseSchedule(T=T, schedule="linear", beta_start=0.05, beta_end=0.5)
    )


def tiny_model(**kwargs) -> GraphFlowDenoiser:
    base = dict(d_model=16, ffn_hidden=32, use_edge_cost=True)
    base.update(kwargs)
    return GraphFlowDenoiser(**base)


def spy_on_step(model):
    """记录每次 ``model.step`` 收到的 H_t（返回 (calls, restore)）。"""
    seen = []
    original = model.step

    def wrapper(batch, H_t, z_t, t, flow_steps=None):
        seen.append(H_t.detach().clone())
        return original(batch, H_t, z_t, t, flow_steps=flow_steps)

    model.step = wrapper
    return seen


def another_z(batch) -> torch.Tensor:
    """构造一个与 target 不同的 z（用于检验输出是否依赖 z_t）。"""
    return (batch.target_candidate + 1) % batch.num_candidates


# ---------------------------------------------------------------------------
# 0) 回归：默认值 == 改造前的 Full model
# ---------------------------------------------------------------------------
def test_default_switches_are_full_model():
    model = tiny_model()
    assert model.persistent_state is True
    assert model.use_edge_state_conditioning is True
    assert model.branch_scorer.readout == "mean_pool"
    assert model.generation_mode == "diffusion"


def test_default_readout_is_bitwise_mean_pool(manual_batch):
    """默认 readout 必须与直接调 branch_mean_pool 逐位相同。"""
    model = tiny_model()
    H = torch.randn(manual_batch.num_nodes, 16)
    is_branch = ~manual_batch.candidate_is_null
    got = model.branch_scorer.branch_representation(H, manual_batch, is_branch)
    want = branch_mean_pool(
        H,
        manual_batch.branch_node_ids,
        manual_batch.branch_node_lengths,
        manual_batch.num_candidates,
    )[is_branch]
    assert torch.equal(got, want)


def test_default_edge_features_depend_on_z(manual_batch):
    """Full model 的边特征必须随 z_t 变化（否则说明动态反馈被弄丢了）。"""
    model = tiny_model()
    a = model.edge_state_encoder.state_ids(manual_batch, manual_batch.target_candidate)
    b = model.edge_state_encoder.state_ids(manual_batch, another_z(manual_batch))
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# 1) persistent_state
# ---------------------------------------------------------------------------
def test_persistent_true_feeds_previous_H(manual_batch):
    model = tiny_model(persistent_state=True)
    seen = spy_on_step(model)
    recurrent_reverse_loss(model, make_diffusion(T=3), manual_batch, max_steps=3)

    assert len(seen) == 3
    H_init = model.init_nodes(manual_batch)
    assert torch.equal(seen[0], H_init)              # 第一步一定是 init
    # 之后每一步的输入都是上一步的输出，所以不该再等于 H_init
    for H in seen[1:]:
        assert not torch.equal(H, H_init)


def test_persistent_false_resets_to_init_every_step(manual_batch):
    model = tiny_model(persistent_state=False)
    seen = spy_on_step(model)
    recurrent_reverse_loss(model, make_diffusion(T=3), manual_batch, max_steps=3)

    assert len(seen) == 3
    H_init = model.init_nodes(manual_batch)
    for H in seen:
        assert torch.equal(H, H_init), "reset-H 的每一步输入都必须是 init_nodes()"


def test_reset_is_not_detach(manual_batch):
    """reset 与 detach 是两件事：reset 的输入与 init 逐位相等。"""
    model = tiny_model(persistent_state=False)
    seen = spy_on_step(model)
    out = recurrent_reverse_loss(model, make_diffusion(T=2), manual_batch, max_steps=2)
    H_init = model.init_nodes(manual_batch)
    assert all(torch.equal(H, H_init) for H in seen)
    # detach 会保留上一轮数值，所以"最后一步的输入 == init"就排除了 detach
    assert out.loss.requires_grad


def test_sampler_honours_persistent_flag(manual_batch):
    """推理侧也要生效（训练与推理必须一致）。"""
    for flag, should_reset in ((True, False), (False, True)):
        model = tiny_model(persistent_state=flag)
        seen = spy_on_step(model)
        sample_reverse_chain(make_diffusion(T=3), model, manual_batch, stochastic=False)
        H_init = model.init_nodes(manual_batch)
        assert len(seen) == 3
        if should_reset:
            assert all(torch.equal(H, H_init) for H in seen)
        else:
            assert not torch.equal(seen[1], H_init)


# ---------------------------------------------------------------------------
# 2) use_edge_state_conditioning
# ---------------------------------------------------------------------------
def test_no_edge_state_conditioning_ignores_z(manual_batch):
    """关掉之后，同一个 H / 同一个 t、只换 z_t，输出必须逐位相同。"""
    model = tiny_model(use_edge_state_conditioning=False)
    H = model.init_nodes(manual_batch)
    out_a = model.step(manual_batch, H, manual_batch.target_candidate, 3)
    out_b = model.step(manual_batch, H, another_z(manual_batch), 3)
    assert torch.equal(out_a.candidate_log_prob, out_b.candidate_log_prob)
    assert torch.equal(out_a.H_next, out_b.H_next)


def test_edge_state_conditioning_on_still_depends_on_z(manual_batch):
    model = tiny_model(use_edge_state_conditioning=True)
    H = model.init_nodes(manual_batch)
    out_a = model.step(manual_batch, H, manual_batch.target_candidate, 3)
    out_b = model.step(manual_batch, H, another_z(manual_batch), 3)
    assert not torch.equal(out_a.candidate_log_prob, out_b.candidate_log_prob)


def test_static_edge_state_keeps_source_forced_edges(manual_batch):
    """静态边状态只允许 source-forced 边是 selected，其余一律 unselected。"""
    state = static_edge_state_ids(manual_batch)
    forced = getattr(manual_batch, "source_forced_edge_ids", None)
    if forced is None or not forced.numel():
        assert int(state.sum()) == 0
    else:
        physical = torch.zeros(manual_batch.num_physical_edges, dtype=torch.long)
        physical[forced] = 1
        assert torch.equal(state, physical[manual_batch.msg_to_phys_edge])


def test_static_features_does_not_accept_z():
    """``static_features`` 的签名里不能有 z —— 从结构上杜绝 label leakage。"""
    model = tiny_model()
    params = list(inspect.signature(model.edge_state_encoder.static_features).parameters)
    assert params == ["batch"], f"unexpected signature: {params}"


# ---------------------------------------------------------------------------
# 3) branch_readout
# ---------------------------------------------------------------------------
def _interior_node_ids(batch):
    """branch 上**非第一个**、且**不是任何 branch 的第一个**的节点。

    后半个条件不能少：手工图里节点 X 既在 J1 的 GT 分支 ``[J1,A,B,X,J2]`` 内部，
    又是 ``J2 -> X`` 那条回头分支的**第一个**节点。只按"列号 >= 1"筛，
    改 X 就会同时改到 J2 那条分支的 first_node 表示，测试会误判。
    """
    ids = batch.branch_node_ids
    lengths = batch.branch_node_lengths
    width = ids.shape[1]
    positions = torch.arange(width)[None, :]
    valid = positions < lengths[:, None]
    interior = ids[(positions >= 1) & valid]
    firsts = ids[:, 0][lengths >= 1]
    keep = [int(n) for n in torch.unique(interior) if int(n) not in set(firsts.tolist())]
    return torch.tensor(keep, dtype=torch.long)


def test_first_node_readout_ignores_interior_nodes(manual_batch):
    """spec §8 的核心测试：B=(J,a,b,c)，只改 h_b,h_c，first_node 的分数不变。"""
    H = torch.randn(manual_batch.num_nodes, 8)
    interior = _interior_node_ids(manual_batch)
    assert interior.numel() > 0, "手工图上必须有 branch 内部节点，否则测试没意义"

    H2 = H.clone()
    H2[interior] += 3.0

    is_branch = ~manual_batch.candidate_is_null
    first = BranchScorer(d_model=8, readout="first_node")
    mean = BranchScorer(d_model=8, readout="mean_pool")

    rep_first_a = first.branch_representation(H, manual_batch, is_branch)
    rep_first_b = first.branch_representation(H2, manual_batch, is_branch)
    rep_mean_a = mean.branch_representation(H, manual_batch, is_branch)
    rep_mean_b = mean.branch_representation(H2, manual_batch, is_branch)

    assert torch.equal(rep_first_a, rep_first_b), "first_node 不该看内部节点"
    assert not torch.allclose(rep_mean_a, rep_mean_b), "mean_pool 必须被内部节点影响"


def test_first_node_keeps_same_shapes_and_parameters(manual_batch):
    """消融不许改输入维度 / 参数量，否则就变成容量对照了。"""
    mean = BranchScorer(d_model=16, readout="mean_pool")
    first = BranchScorer(d_model=16, readout="first_node")
    assert sum(p.numel() for p in mean.parameters()) == sum(
        p.numel() for p in first.parameters()
    )
    H = torch.randn(manual_batch.num_nodes, 16)
    is_branch = ~manual_batch.candidate_is_null
    assert (
        mean.branch_representation(H, manual_batch, is_branch).shape
        == first.branch_representation(H, manual_batch, is_branch).shape
    )


def test_branch_readout_rejects_unknown_value():
    with pytest.raises(ValueError):
        BranchScorer(d_model=8, readout="max_pool")


def test_all_variants_keep_parameter_count(manual_batch):
    """四个消融的参数量都必须与 Full 完全一致。"""
    full = tiny_model()
    n = sum(p.numel() for p in full.parameters())
    for kwargs in (
        dict(persistent_state=False),
        dict(use_edge_state_conditioning=False),
        dict(branch_readout="first_node"),
        dict(generation_mode="direct"),
    ):
        other = tiny_model(**kwargs)
        assert sum(p.numel() for p in other.parameters()) == n, kwargs


# ---------------------------------------------------------------------------
# 4) generation_mode = direct
# ---------------------------------------------------------------------------
def test_direct_logits_signature_has_no_z():
    params = list(inspect.signature(GraphFlowDenoiser.direct_logits).parameters)
    assert params == ["self", "batch"], "direct_logits 不能接受 z（label leakage）"


def test_direct_chain_never_calls_step(manual_batch):
    model = tiny_model(generation_mode="direct")
    calls = []
    original = model.step
    model.step = lambda *a, **k: calls.append(1) or original(*a, **k)

    chain = sample_reverse_chain(
        make_diffusion(T=4), model, manual_batch, stochastic=False, record=True
    )
    assert calls == [], "direct 不许走 reverse step"
    assert chain["trace"] is not None and len(chain["trace"].H_path) == 1


def test_direct_probabilities_are_group_normalised(manual_batch):
    model = tiny_model(generation_mode="direct")
    out = model.direct_logits(manual_batch)
    sums = torch.zeros(manual_batch.num_decisions).index_add_(
        0, manual_batch.candidate_owner, out.candidate_prob
    )
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_direct_loss_runs_and_backprops(manual_batch):
    model = tiny_model(generation_mode="direct")
    out = direct_prediction_loss(model, make_diffusion(T=4), manual_batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


def test_direct_ignores_diffusion_object(manual_batch):
    """direct 损失不许碰 diffusion 的任何采样接口。"""
    model = tiny_model(generation_mode="direct")
    diffusion = make_diffusion(T=4)
    touched = []

    def boom(*args, **kwargs):
        touched.append(1)
        raise AssertionError("direct_prediction_loss must not use diffusion")

    for name in ("sample_forward_trajectory", "sample_xt_at_time"):
        setattr(diffusion, name, boom)
    direct_prediction_loss(model, diffusion, manual_batch)
    assert touched == []
