import numpy as np
import pytest

from sctrange.benchmark import (
    angle_energy_experiment,
    autocorrelation_experiment,
    estimator_experiment,
    fz_check,
    projected_crps_experiment,
    projection_consistency_table,
    rearrangement_experiment,
    summarize_estimators,
    tail_experiment,
    transfer_experiment,
)
from sctrange.hlut import HLUT


def test_autocorrelation_law(rng: np.random.Generator) -> None:
    frame = autocorrelation_experiment([0.0, 4.0], rng, shape=(128, 32), segment=50, n_fields=60)
    np.testing.assert_allclose(frame["rms_observed_mm"], frame["rms_predicted_mm"], rtol=0.1)
    np.testing.assert_allclose(frame["rms_permuted_mm"], frame["rms_predicted_mm"].iloc[0], rtol=0.05)
    assert frame["mae_spr"].std() < 0.02 * frame["mae_spr"].mean()


def test_transfer_is_exact() -> None:
    frame = transfer_experiment((8, 32), n=128)
    assert np.max(np.abs(frame["measured"] - frame["theory"])) < 1e-10


def test_angle_slope(rng: np.random.Generator) -> None:
    _, slope = angle_energy_experiment(np.geomspace(0.03, 0.12, 4), rng, size=97, radius=40.0,
                                       n_orientations=2, angle_step=3.0)
    assert -1.3 < slope < -0.8


def test_estimator_ranking_and_bias_signs(hlut: HLUT, rng: np.random.Generator) -> None:
    frames = {s: estimator_experiment(hlut, s, rng, depths=(60,), n_samples=400, shape=(60, 32))
              for s in ("gas", "bone")}
    for scenario, frame in frames.items():
        summary = summarize_estimators(frame).loc[scenario]
        assert abs(summary.loc[("SPR mean (Bayes act)", 60), "mean_bias_mm"]) < 1e-9
        sign = 1 if scenario == "gas" else -1
        assert sign * summary.loc[("HU median (L1)", 60), "mean_bias_mm"] > 0
        assert np.allclose(summary["rms_mc_mm"], summary["rms_predicted_mm"], rtol=0.05)


def test_projection_consistency_only_for_the_mean() -> None:
    table = projection_consistency_table([-850.0, 40.0, 40.0, 700.0]).set_index("functional")
    for column in ("value on sum, cyclic coupling", "value on sum, independent coupling"):
        assert table.loc["mean", column] == pytest.approx(table.loc["mean", "sum of voxel values"])
    assert table.loc["0.9-quantile", "value on sum, cyclic coupling"] != pytest.approx(
        table.loc["0.9-quantile", "sum of voxel values"])


def test_tails_depend_on_dependence(hlut: HLUT, rng: np.random.Generator) -> None:
    summary, errors = tail_experiment(hlut, (0.0, 12.0), rng, shape=(60, 16), n_samples=800)
    assert abs(summary["alternative_fraction"].iloc[0] - summary["alternative_fraction"].iloc[1]) < 0.02
    assert summary["rms_mm"].iloc[1] > 2 * summary["rms_mm"].iloc[0]
    check = fz_check(errors[12.0])
    assert abs(check["FZ argmin VaR (mm)"] - check["empirical VaR (mm)"]) <= 1.5 * check["grid step VaR (mm)"]
    assert abs(check["FZ argmin ES (mm)"] - check["empirical ES (mm)"]) <= 1.5 * check["grid step ES (mm)"]


def test_projected_crps_runs(hlut: HLUT, rng: np.random.Generator) -> None:
    frame = projected_crps_experiment(hlut, rng, shape=(40, 8), n_observations=6, n_members=8)
    assert frame.shape == (6, 4) and np.isfinite(frame.to_numpy()).all()


def test_rearrangement_extremes_and_comonotone_maximisers(rng: np.random.Generator) -> None:
    frame = rearrangement_experiment(rng, n_voxels=6, n_on_ray=2, n_trials=5)
    np.testing.assert_allclose(frame["exhaustive_max"], frame["bound_max"], atol=1e-12)
    np.testing.assert_allclose(frame["exhaustive_min"], frame["bound_min"], atol=1e-12)
    assert frame["all_maximisers_comonotone"].all()
    assert (frame["maximisers"] == 2 * 24).all()
