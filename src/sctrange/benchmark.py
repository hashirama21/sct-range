"""Synthetic benchmark with a known posterior: numerical checks of the statements in docs/theory.md."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

import numpy as np
import pandas as pd
import torch

from .fields import covariance, exponential_spectrum, gaussian_random_field
from .hlut import HLUT
from .losses import fz_upper_tail_loss
from .metrics import rearrangement_bounds
from .posterior import BONE, GAS, SOFT, MixturePosterior, blob_probability
from .projection import ParallelBeamProjector
from .scoring import crps_ensemble, expectile, joint_mix
from .spectral import (
    effective_length,
    gradient_transfer,
    predicted_range_rms,
    segment_transfer,
    sliding_segment_sums,
)

BLOB_CENTRES = [(22, 12), (48, 42), (80, 20), (64, 54), (100, 36), (32, 30)]
SCENARIOS = {"gas": GAS, "bone": BONE}


def rearrangement_experiment(rng: np.random.Generator, n_voxels: int = 7, n_on_ray: int = 3,
                             n_trials: int = 20) -> pd.DataFrame:
    """Exhaustive search over all rearrangements of errors against a segment weighting (Propositions 1 and 1').

    Errors are continuous with both signs; weights are a segment indicator (ties). For every trial, records the
    exhaustive extremes, the sorted bounds, and whether every maximiser is comonotone with the weights.
    """
    weights = np.r_[np.ones(n_on_ray), np.zeros(n_voxels - n_on_ray)]
    orders = np.array(list(permutations(range(n_voxels))))
    rows = []
    for trial in range(n_trials):
        errors = rng.normal(size=n_voxels)
        values = errors[orders] @ weights
        low, high = rearrangement_bounds(torch.from_numpy(errors), torch.from_numpy(weights))
        maximisers = orders[np.isclose(values, values.max())]
        arranged = errors[maximisers]
        comonotone = np.all((weights[:, None] - weights[None, :])[None]
                            * (arranged[:, :, None] - arranged[:, None, :]) >= -1e-12, axis=(1, 2))
        rows.append({"trial": trial, "exhaustive_min": values.min(), "bound_min": low,
                     "exhaustive_max": values.max(), "bound_max": high,
                     "maximisers": len(maximisers), "all_maximisers_comonotone": bool(comonotone.all()),
                     "gap_to_second_value": float(np.sort(np.unique(np.round(values, 12)))[-1]
                                                  - np.sort(np.unique(np.round(values, 12)))[-2])})
    return pd.DataFrame(rows)


def autocorrelation_experiment(correlation_lengths: Sequence[float], rng: np.random.Generator,
                               shape: tuple[int, int] = (256, 64), segment: int = 100, std: float = 0.03,
                               mean: float = 0.0, n_fields: int = 200) -> pd.DataFrame:
    """RMS segment error of stationary SPR errors at fixed marginal law (Proposition 2).

    The control shuffles all error values across fields and voxels: same marginal law and MAE, white layout.
    """
    rows = []
    for length in correlation_lengths:
        rho = covariance(exponential_spectrum(shape, length))[:, 0]
        errors = mean + std * gaussian_random_field(shape, length, n_fields, rng)
        sums = sliding_segment_sums(errors, segment, axis=1)
        shuffled = rng.permutation(errors.ravel()).reshape(errors.shape)
        rows.append({
            "correlation_length_px": length,
            "effective_length_mm": effective_length(rho, segment),
            "mae_spr": float(np.abs(errors).mean()),
            "rms_observed_mm": float(np.sqrt(np.mean(sums**2))),
            "rms_predicted_mm": predicted_range_rms(mean, std, segment, rho),
            "rms_permuted_mm": float(np.sqrt(np.mean(sliding_segment_sums(shuffled, segment, axis=1) ** 2))),
        })
    return pd.DataFrame(rows)


def transfer_experiment(segment_lengths: Sequence[int] = (10, 60), n: int = 512) -> pd.DataFrame:
    """Measured vs theoretical power transfer of plane-wave errors (Proposition 3)."""
    samples = np.arange(n)
    rows = []
    for index in range(1, n // 2 + 1):
        frequency = index / n
        along = np.cos(2 * np.pi * frequency * samples)[:, None]
        energy = np.sum(along**2)
        for length in segment_lengths:
            sums = sliding_segment_sums(along, length, axis=0)
            rows.append((frequency, f"range, L = {length} mm", np.sum(sums**2) / (length**2 * energy),
                         float(segment_transfer(frequency, length))))
        across = np.tile(np.cos(2 * np.pi * frequency * samples)[None, :], (64, 1))
        sums = sliding_segment_sums(across, segment_lengths[-1], axis=0)
        rows.append((frequency, "range, across beam", np.sum(sums**2) / (segment_lengths[-1] ** 2 * np.sum(across**2)), 1.0))
        difference = np.roll(along, -1, axis=0) - along
        rows.append((frequency, "gradient loss", np.sum(difference**2) / (4 * energy), float(gradient_transfer(frequency))))
        rows.append((frequency, "voxel L2", 1.0, 1.0))
    return pd.DataFrame(rows, columns=["frequency", "functional", "measured", "theory"])


def angle_energy_experiment(frequencies: Sequence[float], rng: np.random.Generator, size: int = 129,
                            radius: float = 56.0, taper: float = 10.0, n_orientations: int = 4,
                            angle_step: float = 1.0) -> tuple[pd.DataFrame, float]:
    """Angle-summed full-traversal energy of windowed plane waves; returns the log-log slope (Proposition 3.1)."""
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    distance = np.hypot(y, x)
    ramp = np.clip((radius - distance) / taper, 0.0, 1.0)
    window = ramp**2 * (3 - 2 * ramp)
    projector = ParallelBeamProjector((size, size), np.arange(0.0, 180.0, angle_step), spacing_mm=1.0)
    rows = []
    for frequency in frequencies:
        direction = rng.uniform(0, np.pi, n_orientations)[:, None, None]
        phase = rng.uniform(0, 2 * np.pi, n_orientations)[:, None, None]
        waves = window * np.cos(2 * np.pi * frequency * (x * np.cos(direction) + y * np.sin(direction)) + phase)
        integrals = projector.integrate(torch.from_numpy(waves), torch.ones(1, dtype=torch.bool))
        projected = float(integrals.square().sum(dim=(0, 2)).mean()) * angle_step
        voxel = float(np.mean(np.sum(waves**2, axis=(1, 2))))
        rows.append({"frequency": frequency, "projected_energy": projected, "voxel_energy": voxel,
                     "ratio": projected / voxel})
    frame = pd.DataFrame(rows)
    slope = float(np.polyfit(np.log(frame["frequency"]), np.log(frame["ratio"]), 1)[0])
    return frame, slope


def scenario_posterior(scenario: str, shape: tuple[int, int] = (120, 64), peak: float = 0.45,
                       radius: float = 9.0) -> MixturePosterior:
    centres = [(r * shape[0] / 120, c * shape[1] / 64) for r, c in BLOB_CENTRES]
    return MixturePosterior(blob_probability(shape, centres, radius, peak), SOFT, SCENARIOS[scenario])


def point_estimates(posterior: MixturePosterior, hlut: HLUT) -> dict[str, np.ndarray]:
    return {
        "HU mean (L2 in HU)": posterior.mean_hu(),
        "HU median (L1)": posterior.median_hu(),
        "SPR mean (Bayes act)": posterior.spr_mean_estimate(hlut),
    }


def estimator_experiment(hlut: HLUT, scenario: str, rng: np.random.Generator, depths: Sequence[int] = (40, 80, 120),
                         correlation_length: float = 8.0, n_samples: int = 3000,
                         shape: tuple[int, int] = (120, 64)) -> pd.DataFrame:
    """Per-ray bias and Monte Carlo risk of three point estimates against the known posterior (Props 4-5).

    Segments run along rows from depth 0 to each depth in ``depths``; one ray per column; 1 mm steps.
    """
    posterior = scenario_posterior(scenario, shape)
    reference = np.cumsum(hlut.numpy(posterior.sample(n_samples, correlation_length, rng)), axis=1)
    expected = np.cumsum(posterior.mean_spr(hlut), axis=0)
    probability = np.cumsum(posterior.probability, axis=0)
    variance = reference.var(axis=0)

    rows = []
    for name, estimate in point_estimates(posterior, hlut).items():
        predicted = np.cumsum(hlut.numpy(estimate), axis=0)
        bias = predicted - expected
        risk = np.mean((predicted[None] - reference) ** 2, axis=0)
        for depth in depths:
            for ray in range(shape[1]):
                rows.append({"scenario": scenario, "estimator": name, "depth_mm": depth, "ray": ray,
                             "probability_sum": probability[depth - 1, ray], "bias_mm": bias[depth - 1, ray],
                             "risk_mc_mm2": risk[depth - 1, ray],
                             "risk_predicted_mm2": bias[depth - 1, ray] ** 2 + variance[depth - 1, ray]})
    return pd.DataFrame(rows)


def summarize_estimators(detail: pd.DataFrame) -> pd.DataFrame:
    bayes = detail[detail["estimator"] == "SPR mean (Bayes act)"].set_index(["scenario", "depth_mm", "ray"])["risk_mc_mm2"]
    detail = detail.join(bayes.rename("bayes_risk_mm2"), on=["scenario", "depth_mm", "ray"])
    grouped = detail.groupby(["scenario", "estimator", "depth_mm"])
    return pd.DataFrame({
        "mean_bias_mm": grouped["bias_mm"].mean(),
        "rms_mc_mm": np.sqrt(grouped["risk_mc_mm2"].mean()),
        "rms_predicted_mm": np.sqrt(grouped["risk_predicted_mm2"].mean()),
        "excess_risk_mc_mm2": grouped["risk_mc_mm2"].mean() - grouped["bayes_risk_mm2"].mean(),
        "excess_risk_analytic_mm2": grouped["bias_mm"].apply(lambda b: float(np.mean(b**2))),
    })


def projection_consistency_table(values: Sequence[float]) -> pd.DataFrame:
    """Functional of the sum vs sum of functionals, exactly, under cyclic and independent couplings (Theorem 6).

    ``m`` copies of the uniform law on ``values``; the independent law of the sum is enumerated over ``m**m`` outcomes.
    """
    values = np.asarray(values, dtype=np.float64)
    m = values.size
    if m > 8:
        raise ValueError("exact enumeration is limited to 8 support points")
    cyclic_sum = joint_mix(values).sum(axis=1)
    independent_sum = values[np.indices((m,) * m).reshape(m, -1)].sum(axis=0)
    functionals = {
        "mean": lambda s: float(np.mean(s)),
        "median": lambda s: float(np.quantile(s, 0.5, method="inverted_cdf")),
        "0.9-quantile": lambda s: float(np.quantile(s, 0.9, method="inverted_cdf")),
        "0.8-expectile": lambda s: float(expectile(np.asarray(s), 0.8)),
        "0.2-expectile": lambda s: float(expectile(np.asarray(s), 0.2)),
    }
    rows = []
    for name, functional in functionals.items():
        rows.append({"functional": name, "sum of voxel values": m * functional(values),
                     "value on sum, cyclic coupling": functional(cyclic_sum),
                     "value on sum, independent coupling": functional(independent_sum)})
    return pd.DataFrame(rows)


def tail_experiment(hlut: HLUT, correlation_lengths: Sequence[float], rng: np.random.Generator,
                    shape: tuple[int, int] = (100, 32), n_samples: int = 4000,
                    level: float = 0.9) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Same marginals and Bayes act, different dependence: law of the full-depth range error (Proposition 7)."""
    posterior = scenario_posterior("gas", shape, peak=0.35)
    estimate = posterior.spr_mean_estimate(hlut)
    predicted = hlut.numpy(estimate).sum(axis=0)
    rows, errors = [], {}
    for length in correlation_lengths:
        samples = posterior.sample(n_samples, length, rng)
        delta = predicted[None] - hlut.numpy(samples).sum(axis=1)
        magnitude = np.abs(delta).ravel()
        threshold = np.quantile(magnitude, level)
        errors[length] = magnitude
        rows.append({"correlation_length_px": length,
                     "alternative_fraction": float((samples < -400).mean()),
                     "mean_signed_mm": float(delta.mean()),
                     "rms_mm": float(np.sqrt(np.mean(delta**2))),
                     f"VaR{int(level * 100)}_mm": float(threshold),
                     f"ES{int(level * 100)}_mm": float(magnitude[magnitude >= threshold].mean())})
    return pd.DataFrame(rows), errors


def fz_check(magnitude: np.ndarray, level: float = 0.9, grid: int = 61, max_samples: int = 20_000,
             rng: np.random.Generator | None = None) -> dict[str, float]:
    """Grid minimiser of the FZ0 upper-tail score against empirical VaR and ES.

    The grid is offset so that it does not contain the empirical values; agreement within one step is expected.
    """
    rng = rng or np.random.default_rng(0)
    if magnitude.size > max_samples:
        magnitude = rng.choice(magnitude, max_samples, replace=False)
    var = float(np.quantile(magnitude, level))
    es = float(magnitude[magnitude >= var].mean())
    outcome = torch.from_numpy(magnitude)
    var_grid = torch.linspace(0.703 * var, 1.303 * var, grid, dtype=torch.float64)
    es_grid = torch.linspace(0.703 * es, 1.303 * es, grid, dtype=torch.float64)
    scores = torch.stack([torch.stack([fz_upper_tail_loss(v, e, outcome, level) for e in es_grid]) for v in var_grid])
    index = int(scores.argmin())
    return {"empirical VaR (mm)": var, "FZ argmin VaR (mm)": var_grid[index // grid].item(),
            "empirical ES (mm)": es, "FZ argmin ES (mm)": es_grid[index % grid].item(),
            "grid step VaR (mm)": float(var_grid[1] - var_grid[0]), "grid step ES (mm)": float(es_grid[1] - es_grid[0])}


def projected_crps_experiment(hlut: HLUT, rng: np.random.Generator, true_length: float = 16.0,
                              wrong_length: float = 0.0, shape: tuple[int, int] = (100, 32),
                              n_observations: int = 150, n_members: int = 48) -> pd.DataFrame:
    """Voxel-wise vs projected CRPS of a correct and a wrong-dependence forecast with equal marginals (Prop 8)."""
    posterior = scenario_posterior("gas", shape, peak=0.35)
    observations = hlut.numpy(posterior.sample(n_observations, true_length, rng))
    rows = []
    for observation in observations:
        row = {}
        for label, length in (("correct", true_length), ("wrong dependence", wrong_length)):
            ensemble = hlut.numpy(posterior.sample(n_members, length, rng))
            row[f"voxel CRPS, {label}"] = float(crps_ensemble(ensemble, observation).mean())
            row[f"projected CRPS, {label}"] = float(crps_ensemble(ensemble.sum(axis=1), observation.sum(axis=0)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(frame: pd.DataFrame, better: str, worse: str, rng: np.random.Generator,
                      n_bootstrap: int = 5000) -> dict[str, float]:
    """Mean of ``worse - better`` with a percentile bootstrap 95% interval."""
    difference = (frame[worse] - frame[better]).to_numpy()
    resampled = rng.choice(difference, size=(n_bootstrap, difference.size)).mean(axis=1)
    low, high = np.quantile(resampled, [0.025, 0.975])
    return {"mean difference": float(difference.mean()), "ci_low": float(low), "ci_high": float(high),
            "relative to score": float(difference.mean() / frame[better].mean())}
