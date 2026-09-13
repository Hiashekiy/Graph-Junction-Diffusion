"""Forward categorical diffusion tests (guide section 19).

Test 1: every row of Q_t sums to 1
Test 2: prod_t Q_t equals the closed form Qbar_t
Test 4: Monte-Carlo forward sampling matches the theoretical distribution
Test 5: t = 1 boundary
"""

from __future__ import annotations

import pytest
import torch

from src.diffusion.categorical import CategoricalDiffusion
from src.diffusion.schedule import NoiseSchedule


def _toy_diffusion(T: int = 5) -> CategoricalDiffusion:
    schedule = NoiseSchedule(T=T, schedule="linear", beta_start=0.05, beta_end=0.5)
    return CategoricalDiffusion(schedule)


def test_schedule_monotone_alpha_bar():
    schedule = NoiseSchedule(T=50, schedule="linear", beta_start=0.02, beta_end=0.20)
    assert schedule.alpha_bar[0].item() == 1.0
    alpha_bar = schedule.alpha_bar.tolist()
    assert all(alpha_bar[i] > alpha_bar[i + 1] for i in range(len(alpha_bar) - 1))
    schedule.validate(atol=0.02)
    assert schedule.terminal_alpha_bar() < 0.02


def test_qt_row_sum():
    diffusion = _toy_diffusion()
    for t in range(1, diffusion.T + 1):
        for c in (2, 3, 5):
            q = diffusion.transition_matrix(c, t)
            assert torch.allclose(q.sum(dim=1), torch.ones(c, dtype=q.dtype), atol=1e-10)
            assert (q >= 0).all()


def test_qbar_closed_form():
    """prod_{tau<=t} Q_tau == alpha_bar_t I + (1 - alpha_bar_t) 1 pi^T."""
    diffusion = _toy_diffusion(T=5)
    for c in (3, 4):
        for t in range(1, diffusion.T + 1):
            product = diffusion.cumulative_transition_matrix_product(c, t)
            closed = diffusion.cumulative_transition_matrix(c, t)
            assert torch.allclose(product, closed, atol=1e-12), (
                f"mismatch for C={c}, t={t}: {product - closed}"
            )


def test_qbar_row_sum_and_large_t():
    diffusion = _toy_diffusion()
    q = diffusion.cumulative_transition_matrix(4, diffusion.T)
    assert torch.allclose(q.sum(dim=1), torch.ones(4, dtype=q.dtype), atol=1e-10)
    # at t == T the state is (nearly) uniform
    assert torch.allclose(q, torch.full((4, 4), 0.25, dtype=q.dtype), atol=0.2)


def test_forward_sampling_frequency():
    """Empirical q(z_t | z_0) must match alpha_bar_t [c==b] + (1-alpha_bar_t)/C."""
    torch.manual_seed(0)
    diffusion = _toy_diffusion(T=5)
    num_categories = 3
    num_decisions = 20000
    owner = torch.arange(num_decisions).repeat_interleave(num_categories)
    starts = torch.arange(num_decisions) * num_categories
    target = starts.clone()  # every decision starts in its own first category

    t = torch.tensor(3)
    alpha_bar = float(diffusion.schedule.alpha_bar_at(t))
    totals = torch.zeros(num_categories)
    for _ in range(10):
        z = diffusion.sample_xt(
            target, owner, num_decisions, torch.full((num_decisions,), alpha_bar)
        )
        totals += torch.bincount(z - starts, minlength=num_categories).float()
    empirical = totals / totals.sum()

    expected = torch.full((num_categories,), (1 - alpha_bar) / num_categories)
    expected[0] += alpha_bar
    assert torch.allclose(empirical, expected, atol=0.01), (empirical, expected)


def test_forward_sampling_keeps_original_category_at_least_alpha_bar():
    """The resampled category may coincide with z_0, so P(keep) >= alpha_bar."""
    torch.manual_seed(1)
    diffusion = _toy_diffusion(T=5)
    num_categories = 2
    batch = 20000
    owner = torch.arange(batch).repeat_interleave(num_categories)
    starts = torch.arange(batch) * num_categories
    t = torch.tensor(1)
    alpha_bar = float(diffusion.schedule.alpha_bar_at(t))
    z = diffusion.sample_xt(
        starts.clone(),
        owner,
        batch,
        torch.full((batch,), alpha_bar),
    )
    local = z - starts
    keep = (local == 0).float().mean().item()
    expected = alpha_bar + (1 - alpha_bar) / num_categories
    assert abs(keep - expected) < 0.02
    assert keep > alpha_bar


def test_variable_category_number():
    """Groups with different cardinalities must be sampled independently."""
    torch.manual_seed(2)
    diffusion = _toy_diffusion(T=5)
    sizes = torch.tensor([2, 5, 3])
    owner = torch.repeat_interleave(torch.arange(3), sizes)
    num_decisions = 3
    starts = diffusion.group_starts(sizes)
    z = diffusion.sample_xt(
        starts.clone(), owner, num_decisions, torch.full((num_decisions,), 0.5)
    )
    for m in range(num_decisions):
        assert starts[m] <= z[m] < starts[m] + sizes[m]


def test_t1_boundary_reverse_posterior_is_clean_state():
    """At t = 1 the exact posterior must place all mass on z_0."""
    diffusion = _toy_diffusion(T=5)
    sizes = torch.tensor([3, 4])
    owner = torch.repeat_interleave(torch.arange(2), sizes)
    starts = diffusion.group_starts(sizes)
    z0 = starts + torch.tensor([2, 0])
    zt = starts + torch.tensor([1, 3])
    posterior, mask = diffusion.true_posterior(z0, zt, owner, 2, torch.tensor(1))
    for m in range(2):
        local = int(z0[m].item() - starts[m].item())
        assert posterior[m, local].item() > 1 - 1e-6
        assert mask[m].sum().item() == sizes[m]


# ---------------------------------------------------------------------------
# ISSUES.md 3.4: sampling must tolerate a generator on another device
# ---------------------------------------------------------------------------
def test_sampling_with_cpu_generator_on_cpu():
    diffusion = _toy_diffusion(T=5)
    sizes = torch.tensor([3, 4])
    owner = torch.repeat_interleave(torch.arange(2), sizes)
    starts = diffusion.group_starts(sizes)
    z = diffusion.sample_xt(
        starts.clone(), owner, 2, torch.full((2,), 0.5),
        generator=torch.Generator().manual_seed(0),
    )
    for m in range(2):
        assert starts[m] <= z[m] < starts[m] + sizes[m]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sampling_with_cpu_generator_on_cuda_tensors():
    """A CPU generator + CUDA tensors used to raise
    `Expected a 'cuda' device type for generator but found 'cpu'`."""
    diffusion = _toy_diffusion(T=5)
    sizes = torch.tensor([3, 4], device="cuda")
    owner = torch.repeat_interleave(torch.arange(2, device="cuda"), sizes)
    starts = diffusion.group_starts(sizes)
    generator = torch.Generator().manual_seed(0)  # deliberately CPU
    z = diffusion.sample_xt(
        starts.clone(), owner, 2, torch.full((2,), 0.5, device="cuda"),
        generator=generator,
    )
    assert z.device.type == "cuda"
    for m in range(2):
        assert starts[m] <= z[m] < starts[m] + sizes[m]

    mask = torch.arange(4, device="cuda")[None, :] < sizes[:, None]
    prob = (torch.rand(2, 4, device="cuda") * mask)
    prob = prob / prob.sum(dim=1, keepdim=True)
    prev = diffusion.sample_prev(prob, mask, owner, 2, generator=generator)
    assert prev.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sampling_with_cuda_generator_on_cuda_tensors():
    diffusion = _toy_diffusion(T=5)
    sizes = torch.tensor([2, 2], device="cuda")
    owner = torch.repeat_interleave(torch.arange(2, device="cuda"), sizes)
    starts = diffusion.group_starts(sizes)
    z = diffusion.sample_xt(
        starts.clone(), owner, 2, torch.full((2,), 0.5, device="cuda"),
        generator=torch.Generator(device="cuda").manual_seed(0),
    )
    assert z.device.type == "cuda"
