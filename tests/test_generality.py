import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sctrange.cache import VolumeCache
from sctrange.data import discover_cases, read_volume, write_volume
from sctrange.generality import (ablation_grid, bootstrap_median, compare_family, prediction_shim,
                                 registration_matrix, registration_overlay)
from sctrange.phantoms import write_fixture_dataset
from sctrange.tomography import (SparseViewScanner, TomographyConfig, attenuation_slice, crossfit_tomography,
                                 evaluate_tomography, tomography_loss)
from sctrange.training import TrainConfig, truncate_segments

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("generality")
    write_fixture_dataset(root / "data", regions=("AB", "HN"), centers=("A",), cases_per_stratum=2,
                          shape_yx=(64, 80), n_slices=3, seed=9)
    return root


def test_task2_fixture_has_cbct_source(tmp_path: Path) -> None:
    ids = write_fixture_dataset(tmp_path, regions=("TH",), centers=("A",), cases_per_stratum=1, shape_yx=(32, 40),
                                n_slices=2, task="2")
    record = discover_cases(tmp_path)[0]
    assert ids == ["2THA001"] and record.task == "2" and record.path("cbct").exists()


def test_truncated_segments_keep_depth_from_entry() -> None:
    weights = torch.zeros(1, 1, 12, 2, dtype=torch.bool)
    weights[..., 3:11, 0] = True
    weights[..., 6:8, 1] = True
    kept = truncate_segments(weights, 4)[0, 0]
    assert kept[:, 0].nonzero().flatten().tolist() == [3, 4, 5, 6] and kept[:, 1].sum() == 2
    with pytest.raises(ValueError):
        TrainConfig(segment_mm=-1)


def test_registration_overlay_and_shim(fixture_root: Path, tmp_path: Path) -> None:
    records = discover_cases(fixture_root / "data")
    alternative = tmp_path / "impact"
    for index, record in enumerate(records[:3]):
        ct = read_volume(record.path("ct"))
        array = ct.array + 10.0 if index < 2 else ct.array[:, :-1]
        write_volume(array, alternative / record.directory.relative_to(fixture_root / "data") / "ct.mha", like=ct
                     if index < 2 else None)
    report = registration_overlay(fixture_root / "data", alternative, tmp_path / "overlay",
                                  [r.case_id for r in records])
    assert report["kept"].sum() == 2
    assert set(report.loc[~report["kept"], "reason"]) == {"different grid", "no alternative CT"}
    overlaid = discover_cases(tmp_path / "overlay")
    assert len(overlaid) == 2
    first = overlaid[0]
    original = next(r for r in records if r.case_id == first.case_id)
    assert np.allclose(read_volume(first.path("ct")).array, read_volume(original.path("ct")).array + 10.0)
    assert np.array_equal(read_volume(first.path("mr")).array, read_volume(original.path("mr")).array)
    shim = prediction_shim(tmp_path / "overlay", tmp_path / "shim", "G")
    assert (shim / "seed0" / "G" / "predictions").resolve() == (tmp_path / "overlay").resolve()


def test_ablation_grid_matches_preregistration() -> None:
    raw = json.loads((REPO / "configs" / "s8_s10_preregistration.json").read_text())["s10_ablations"]
    grid = ablation_grid(raw)
    assert sum(a.center for a in grid) == 4
    for ablation in grid:
        TrainConfig(**{"loss": "l2_spr", "range_weight": 1e-4, **ablation.overrides})
    names = {a.name for a in grid}
    for family in raw["families"].values():
        assert all(set(pair) <= names for pair in family["comparisons"])


def test_compare_family_and_bootstrap(rng: np.random.Generator) -> None:
    rows = []
    for index in range(30):
        base = rng.normal()
        rows += [{"case_id": f"c{index}", "source": "a", "m": base + 1.0},
                 {"case_id": f"c{index}", "source": "b", "m": base}]
    table = compare_family(pd.DataFrame(rows), [("a", "b")], ["m"], n_boot=200)
    assert table.loc[0, "p_holm"] < 1e-4 and table.loc[0, "ci_low"] <= 1.0 <= table.loc[0, "ci_high"]
    assert np.isnan(bootstrap_median(np.array([np.nan]))[0])
    matrix = registration_matrix(pd.DataFrame({"train": ["E", "E", "I", "I"], "test": ["E", "I", "E", "I"],
                                               "mae_hu": [1, 2, 3, 4]}), "mae_hu")
    assert matrix.loc["I", "E"] == 3


def test_scanner_adjoint_and_fbp_accuracy() -> None:
    n = 96
    scanner = SparseViewScanner(n, 2.0, [k * 1.0 for k in range(180)], 0.0)
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(2, n, n, generator=generator)
    y = torch.rand(180, 2, scanner.projector.side, generator=generator)
    lhs = (scanner.project(x) * y).sum()
    rhs = (x * scanner.backproject(y) * scanner.pixel_mm).sum()
    assert torch.isclose(lhs, rhs, rtol=1e-4)
    yy, xx = np.indices((n, n))
    disk = torch.from_numpy(((yy - 47.5) ** 2 + (xx - 47.5) ** 2 <= 35**2).astype(np.float32))[None]
    reconstruction = scanner.fbp(scanner.project(disk))[0]
    assert float((reconstruction - disk[0]).abs()[disk[0] > 0].mean()) < 0.04


def test_attenuation_slice_and_losses() -> None:
    ct = np.full((40, 60), -1000.0, dtype=np.float32)
    mask = np.zeros_like(ct)
    ct[10:30, 10:50], mask[10:30, 10:50] = 0.0, 1.0
    u, body = attenuation_slice(ct, mask, 1.0, 2.0, 32)
    assert u.shape == (32, 32) and np.isclose(u.max(), 1.0) and body.sum() == 200
    config = TomographyConfig(size=32, depth=3, loss="l2_proj", min_path_mm=5.0)
    truth, region = torch.from_numpy(u)[None], torch.from_numpy(body)[None]
    loss, parts = tomography_loss(truth + 0.1, truth, region, config, (0.0, 45.0))
    assert "projection" in parts and loss > parts["pixel"]
    l1, _ = tomography_loss(truth + 0.1, truth, region, replace(config, loss="l1"))
    assert torch.isclose(l1, torch.tensor(0.1), atol=1e-5)


def test_tomography_training_and_evaluation(fixture_root: Path) -> None:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    cache.build(records)
    runs = {}
    for loss in ("l2", "l2_proj"):
        config = TomographyConfig(loss=loss, size=64, depth=3, base_channels=8, steps=6, batch_size=2,
                                  warmup_steps=2, log_every=3, device="cpu", sparse_views=10, eval_views=30,
                                  min_path_mm=10.0)
        crossfit_tomography(records, cache, config, 2, fixture_root / "tomo" / loss)
        runs[loss] = fixture_root / "tomo" / loss
    frame = evaluate_tomography(records, cache, runs, config, n_slices=2, device="cpu")
    assert set(frame["source"]) == {"FBP", "l2", "l2_proj"} and len(frame) == 3 * len(records)
    assert frame[["mae_hu", "li_cvar_mm", "li_global_ratio"]].notna().all().all()
    with pytest.raises(ValueError):
        crossfit_tomography(records, cache, replace(config, loss="l1"), 2, runs["l2"])


def test_artifacts_redirected_and_late_fields_neutral(tmp_path: Path) -> None:
    from sctrange import runs

    persistent, scratch = tmp_path / "outputs" / "s9", tmp_path / "scratch"
    runs.redirect_artifacts(persistent, scratch)
    try:
        assert runs.artifact_dir(persistent / "task2" / "R1") == scratch.resolve() / "task2" / "R1" / "predictions"
        assert runs.artifact_dir(tmp_path / "elsewhere", "risk") == tmp_path / "elsewhere" / "risk"
    finally:
        runs._ARTIFACT_ROOTS.clear()
    assert "segment_mm" not in TrainConfig().comparable()
    assert TrainConfig(segment_mm=40.0).comparable()["segment_mm"] == 40.0


def test_tomography_fold_resumes_from_progress(fixture_root: Path) -> None:
    from sctrange.tomography import train_reconstructor

    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    cache.build(records)
    cases = [cache.load(r.case_id) for r in records]
    config = TomographyConfig(size=64, depth=3, base_channels=8, steps=6, batch_size=2, warmup_steps=2, log_every=1,
                              device="cpu", sparse_views=10, eval_views=30, min_path_mm=10.0, loss="l2_proj")
    straight, _ = train_reconstructor(cases, config)
    partial = fixture_root / "tomo_partial.pt"
    train_reconstructor(cases, config, resume_path=partial, checkpoint_every=4)  # saves at step 4 of 6
    assert torch.load(partial, weights_only=True)["step"] == 4
    resumed, _ = train_reconstructor(cases, config, resume_path=partial, checkpoint_every=4)
    for a, b in zip(straight.state_dict().values(), resumed.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_redirected_run_rebuilds_predictions_without_training(fixture_root: Path, hlut, tmp_path: Path) -> None:
    import shutil

    from sctrange import runs
    from sctrange.correction import crossfit_bias_correction
    from sctrange.training import crossfit

    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    cache.build(records)
    outputs, scratch = tmp_path / "outputs", tmp_path / "scratch"
    runs.redirect_artifacts(outputs, scratch)
    try:
        config = TrainConfig(loss="l2_spr", context=1, patch=48, batch_size=2, steps=4, base_channels=8, depth=3,
                             warmup_steps=1, log_every=2, device="cpu", samples_per_case=200)
        run = outputs / "task2" / "R3"
        crossfit(records, cache, config, hlut, 2, run)
        ids = [r.case_id for r in records]
        assert (run / "config.json").exists() and not (run / "predictions").exists()
        assert runs.run_complete(run, ids) and len(list(runs.artifact_dir(run).glob("*.mha"))) == len(ids)
        crossfit_bias_correction(records, run, outputs / "task2" / "R4", hlut)
        assert runs.run_complete(outputs / "task2" / "R4", ids)
        stamps = {p.name: p.stat().st_mtime_ns for p in run.glob("fold*.pt")}
        shutil.rmtree(scratch)  # a new session: the session disk is empty
        assert not runs.run_complete(run, ids)
        crossfit(records, cache, config, hlut, 2, run)
        assert runs.run_complete(run, ids)
        assert stamps == {p.name: p.stat().st_mtime_ns for p in run.glob("fold*.pt")}  # predicted, not retrained
    finally:
        runs._ARTIFACT_ROOTS.clear()
