from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from sctrange.cache import VolumeCache
from sctrange.data import discover_cases
from sctrange.evaluation import DiagnosticConfig
from sctrange.generative import (GenerativeConfig, TimeUNet2d, base_image, crossfit_generative, ensemble_metrics,
                                 evaluate_generative, interpolate, sample_case, sample_slices, select_eta,
                                 train_generative)
from sctrange.hlut import HLUT
from sctrange.normalization import encode_ct
from sctrange.phantoms import write_fixture_dataset

TINY = dict(context=1, patch=48, batch_size=4, steps=40, base_channels=8, depth=3, warmup_steps=5, log_every=10,
            device="cpu", samples_per_case=500, time_dim=16)


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("s7")
    write_fixture_dataset(root / "data", regions=("AB", "HN"), centers=("A",), cases_per_stratum=2,
                          shape_yx=(64, 80), n_slices=3, seed=5)
    return root


def test_interpolant_endpoints() -> None:
    x0, x1, z = torch.zeros(2, 1, 4, 4), torch.ones(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
    start, _ = interpolate(x0, x1, torch.zeros(2), z, 0.2)
    end, _ = interpolate(x0, x1, torch.ones(2), z, 0.2)
    assert torch.allclose(start, x0) and torch.allclose(end, x1)
    _, drift = interpolate(x0, x1, torch.full((2,), 0.3), z, 0.0)
    assert torch.allclose(drift, x1 - x0)


def test_time_unet_starts_at_zero_drift() -> None:
    model = TimeUNet2d(5, base_channels=8, depth=3, out_channels=2, time_dim=16)
    out = model(torch.randn(3, 5, 24, 40), torch.rand(3))
    assert out.shape == (3, 2, 24, 40) and torch.count_nonzero(out) == 0


class OracleDrift(nn.Module):
    """Drift ``x_1 - x_0`` for a known target; the denoiser output is zero."""

    def __init__(self, x1: torch.Tensor) -> None:
        super().__init__()
        self.x1 = x1
        self.head = nn.Conv2d(1, 2, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        masks = x[:, -2:-1]
        return torch.cat([self.x1 - base_image(masks), torch.zeros_like(masks)], dim=1)


def test_deterministic_sampler_reaches_target_and_stochastic_one_spreads() -> None:
    masks = torch.zeros(2, 1, 16, 16)
    masks[..., 4:12, 4:12] = 1
    x1 = torch.where(masks > 0, encode_ct(torch.full_like(masks, 300.0)), base_image(masks))
    inputs = masks.clone()
    model = OracleDrift(x1)
    for steps in (1, 4):
        assert torch.allclose(sample_slices(model, inputs, masks, steps, eta=0.0, diffusion=1.0), x1, atol=1e-6)
    generator = torch.Generator().manual_seed(0)
    noisy = sample_slices(model, inputs, masks, 8, eta=0.2, diffusion=1.0, generator=generator)
    assert noisy[masks > 0].std() > 0 and torch.equal(noisy[masks == 0], x1[masks == 0])


def test_ensemble_metrics_calibration(rng: np.random.Generator) -> None:
    rays, k = 4000, 99
    centre = rng.normal(size=rays)
    truth = centre + rng.normal(size=rays)
    members = centre[None] + rng.normal(size=(k, rays)) - truth[None]
    point = centre - truth
    row, ranks = ensemble_metrics(members, point)
    assert abs(row["coverage90"] - 0.9) < 0.03 and row["pcrps_mm"] < row["pcrps_point_mm"]
    counts = np.bincount(ranks // 10, minlength=10)
    assert counts.min() > 0.6 * counts.mean()
    collapsed, _ = ensemble_metrics(np.repeat(point[None], 8, axis=0), point)
    assert collapsed["spread_mm"] < 1e-12 and np.isclose(collapsed["pcrps_mm"], collapsed["pcrps_point_mm"])


def test_select_eta_excludes_deterministic_and_breaks_ties_towards_smaller() -> None:
    sweep = pd.DataFrame({"eta": [0.0, 0.0, 0.1, 0.1, 0.2, 0.2], "pcrps_mm": [1, 1, 2, 2, 2, 2],
                          "spread_mm": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]})
    assert select_eta(sweep) == 0.1
    from sctrange.generative import collapse_ratio

    assert collapse_ratio(sweep) == 0.0


def test_generative_training_sampling_and_evaluation(fixture_root: Path, hlut: HLUT) -> None:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    cache.build(records)
    config = GenerativeConfig(eta=0.1, **TINY)
    model, history = train_generative([cache.load(r.case_id) for r in records], replace_steps(config, 80))
    assert history["loss"].iloc[-1] < history["loss"].iloc[0]
    case = cache.load(records[0].case_id)
    samples = sample_case(model, case, config, n_samples=3, n_steps=4, device="cpu")
    assert samples.shape == (3,) + case.mask.shape and np.all(samples[:, np.asarray(case.mask) == 0] == -1000.0)

    run = fixture_root / "gen" / "eta0p1"
    crossfit_generative(records, cache, config, 2, run)
    diagnostic = DiagnosticConfig(angles_deg=(0.0, 90.0), target_radius_mm=10.0, min_path_mm=5.0)
    rows, pits = evaluate_generative(records, cache, run, config, hlut, diagnostic, 3, 2, "cpu", write_mean=True)
    assert len(rows) == len(records) and pits.shape[1] == 4
    assert {"pcrps_mm", "coverage90", "risk_fz", "target_cvar_mm", "target_global_ratio"} <= set(rows.columns)
    assert (run / "predictions" / f"{records[0].case_id}.mha").exists()
    stamp = (run / "eval" / "K3_N2" / f"{records[0].case_id}.json").stat().st_mtime_ns
    again, _ = evaluate_generative(records, cache, run, config, hlut, diagnostic, 3, 2, "cpu")
    assert (run / "eval" / "K3_N2" / f"{records[0].case_id}.json").stat().st_mtime_ns == stamp
    pd.testing.assert_frame_equal(rows, again)
    with pytest.raises(ValueError):
        crossfit_generative(records, cache, GenerativeConfig(eta=0.2, **TINY), 2, run)


def replace_steps(config: GenerativeConfig, steps: int) -> GenerativeConfig:
    from dataclasses import replace

    return replace(config, steps=steps)


def test_generative_decision_and_comparison_frame() -> None:
    import json

    from sctrange.generative import comparison_frame, decide_generative

    prereg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "s7_preregistration.json").read_text())
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"case_id": [f"c{i}" for i in range(25)], "region": "AB"})
    rms = rng.uniform(1, 3, 25)
    gen = base.assign(mae_hu=60.0, target_rms_mm=rms, target_mean_signed_mm=0.0, target_mean_abs_mm=0.8 * rms,
                      target_cvar_mm=2 * rms, pcrps_mm=0.5 * rms, risk_fz=1.0, risk_coverage=0.88,
                      risk_spearman=0.3, unc_err_spearman=0.4)
    ref = pd.concat([base.assign(source=arm, mae_hu=60.0, target_rms_mm=rms + 0.01 * rng.normal(size=25),
                                 target_mean_signed_mm=0.0, target_mean_abs_mm=0.8 * rms, target_cvar_mm=2 * rms)
                     for arm in ("R3", "R6")], ignore_index=True)
    risk = base.assign(source="R6", risk_fz=1.0 + rng.uniform(0.1, 0.2, 25), risk_coverage=0.7, risk_spearman=0.2)
    frame = comparison_frame(gen, ref, risk)
    sweep = pd.DataFrame({"eta": [0.0, 0.1, 0.2] * 4, "pcrps_mm": [2, 1, 1.5] * 4, "spread_mm": [0.0, 1.0, 2.0] * 4})
    decision = decide_generative(frame, prereg, sweep)
    assert decision["selected_eta"] == 0.1 and decision["eta0_collapses"]
    assert decision["tests"]["proper_score_better_than_R3"]["significant"]
    assert not decision["tests"]["mean_better_than_R3"]["significant"]
    assert decision["tests"]["tail_score_vs_R6"]["significant"]
    assert decision["theory_checks"] == {"mean_better_than_R3": True, "proper_score_better_than_R3": True}
