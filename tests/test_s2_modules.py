import numpy as np
import pytest
import torch

from sctrange.fields import covariance, exponential_spectrum, gaussian_random_field
from sctrange.hlut import HLUT
from sctrange.losses import fz_upper_tail_loss
from sctrange.posterior import GAS, MixturePosterior
from sctrange.scoring import crps_ensemble, expectile, joint_mix
from sctrange.spectral import (
    effective_length,
    gradient_transfer,
    predicted_range_rms,
    segment_energy_spectral,
    segment_transfer,
    sliding_segment_sums,
)


def test_random_field_matches_its_exact_covariance(rng: np.random.Generator) -> None:
    fields = gaussian_random_field((96, 48), 5.0, 600, rng)
    exact = covariance(exponential_spectrum((96, 48), 5.0))
    assert exact[0, 0] == pytest.approx(1.0)
    for lag in (1, 3, 8):
        empirical = np.mean(fields[:, lag:, :] * fields[:, :-lag, :])
        assert empirical == pytest.approx(exact[lag, 0], abs=0.03)


def test_white_limit() -> None:
    assert np.allclose(exponential_spectrum((16, 16), 0.0), 1.0)


def test_parseval_identity_is_exact(rng: np.random.Generator) -> None:
    field = rng.normal(size=(64, 20))
    for length in (1, 7, 64):
        sums = sliding_segment_sums(field, length, axis=0, step=0.5)
        assert np.sum(sums**2) == pytest.approx(segment_energy_spectral(field, length, axis=0, step=0.5), rel=1e-10)


def test_sliding_sums_match_direct(rng: np.random.Generator) -> None:
    field = rng.normal(size=(12, 3))
    direct = np.stack([sum(np.roll(field, -j, axis=0)[p] for j in range(5)) for p in range(12)])
    np.testing.assert_allclose(sliding_segment_sums(field, 5, axis=0), direct)


def test_transfer_functions() -> None:
    assert segment_transfer(np.array([0.0]), 30)[0] == pytest.approx(1.0)
    assert segment_transfer(np.array([1 / 30]), 30)[0] == pytest.approx(0.0, abs=1e-20)
    assert gradient_transfer(np.array([0.5]))[0] == pytest.approx(1.0)


def test_effective_length_limits() -> None:
    white = np.r_[1.0, np.zeros(99)]
    assert effective_length(white, 100) == pytest.approx(1.0)
    constant = np.ones(100)
    assert effective_length(constant, 100) == pytest.approx(100.0)
    assert predicted_range_rms(0.0, 1.0, 100, constant) == pytest.approx(100.0)


def test_mixture_marginals(hlut: HLUT, rng: np.random.Generator) -> None:
    posterior = MixturePosterior(np.array([0.0, 0.15, 0.4, 0.7]))
    median = posterior.median_hu()
    np.testing.assert_allclose(posterior.cdf(median), 0.5, atol=1e-9)
    samples = posterior.sample(40_000, correlation_length=0.0, rng=rng)
    np.testing.assert_allclose(samples.mean(0), posterior.mean_hu(), atol=6.0)
    np.testing.assert_allclose(hlut.numpy(samples).mean(0), posterior.mean_spr(hlut), atol=4e-3)
    np.testing.assert_allclose((samples < -400).mean(0), posterior.probability, atol=8e-3)


def test_copula_changes_joint_law_not_marginals(rng: np.random.Generator) -> None:
    posterior = MixturePosterior(np.full((64, 8), 0.3))
    independent = posterior.sample(3000, 0.0, rng)
    correlated = posterior.sample(3000, 12.0, rng)
    assert (independent < -400).mean() == pytest.approx((correlated < -400).mean(), abs=0.01)
    assert independent.sum(axis=1).std() < 0.6 * correlated.sum(axis=1).std()


def test_spr_mean_estimate_reproduces_expected_spr(hlut: HLUT) -> None:
    posterior = MixturePosterior(np.linspace(0.0, 0.9, 10), alternative=GAS)
    np.testing.assert_allclose(hlut.numpy(posterior.spr_mean_estimate(hlut)), posterior.mean_spr(hlut), atol=1e-9)


def test_hlut_inverse_round_trip(hlut: HLUT) -> None:
    hu = np.linspace(-990, 2900, 500)
    np.testing.assert_allclose(hlut.inverse(hlut.numpy(hu)), hu, atol=1e-6)


def test_joint_mix_has_constant_sum_and_identical_marginals() -> None:
    values = np.array([0.0, 0.0, 3.0, 10.0, 2.0])
    support = joint_mix(values)
    np.testing.assert_allclose(support.sum(axis=1), values.sum())
    for column in support.T:
        np.testing.assert_array_equal(np.sort(column), np.sort(values))


def test_expectile_half_is_mean(rng: np.random.Generator) -> None:
    samples = rng.exponential(size=(5000, 3))
    np.testing.assert_allclose(expectile(samples, 0.5), samples.mean(0), atol=1e-8)
    assert np.all(expectile(samples, 0.8) > samples.mean(0))


def test_crps_ensemble_matches_gaussian_closed_form(rng: np.random.Generator) -> None:
    ensemble = rng.normal(size=(4000, 1))
    closed_form = 2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)
    assert crps_ensemble(ensemble, np.zeros(1))[0] == pytest.approx(closed_form, abs=0.01)


def test_fz_loss_minimised_at_exponential_var_and_es(rng: np.random.Generator) -> None:
    outcome = torch.from_numpy(rng.exponential(size=200_000))
    var_grid = torch.linspace(1.9, 2.7, 81)
    es_grid = torch.linspace(2.9, 3.7, 81)
    scores = torch.stack([torch.stack([fz_upper_tail_loss(v, e, outcome) for e in es_grid]) for v in var_grid])
    index = int(scores.argmin())
    assert var_grid[index // 81].item() == pytest.approx(np.log(10), abs=0.05)
    assert es_grid[index % 81].item() == pytest.approx(np.log(10) + 1, abs=0.05)
