import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.ndimage import gaussian_filter1d

from sctrange.cache import VolumeCache
from sctrange.data import discover_cases, read_volume
from sctrange.diagnostics import Preregistration, beam_spectra, decide, paired_test
from sctrange.evaluation import DiagnosticConfig, prepare_case
from sctrange.hlut import HLUT
from sctrange.models import UNet2d
from sctrange.normalization import decode_ct, encode_ct, normalize_source
from sctrange.phantoms import disk, write_fixture_dataset
from sctrange.training import TrainConfig, assign_folds, crossfit, point_loss, predict_case, train_model

REPO = Path(__file__).resolve().parents[1]
TINY = dict(context=1, patch=48, batch_size=4, steps=30, base_channels=8, depth=3, warmup_steps=5,
            log_every=10, device="cpu", samples_per_case=500)


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("s3")
    write_fixture_dataset(root / "data", regions=("AB", "HN"), centers=("A", "B"), cases_per_stratum=2,
                          shape_yx=(64, 80), n_slices=3, seed=3)
    return root


def test_ct_encoding_round_trip() -> None:
    hu = np.array([-1024.0, -500.0, 0.0, 1500.0, 3071.0])
    np.testing.assert_allclose(decode_ct(encode_ct(hu)), hu)
    assert encode_ct(np.array([-5000.0]))[0] == -1.0


@pytest.mark.parametrize("modality", ["mr", "cbct"])
def test_source_normalisation(modality: str, rng: np.random.Generator) -> None:
    mask = np.zeros((2, 20, 20), dtype=np.float32)
    mask[:, 5:15, 5:15] = 1
    array = rng.normal(300, 80, size=mask.shape)
    normalized = normalize_source(array, mask, modality)
    assert np.all(normalized[mask == 0] == 0)
    inside = normalized[mask > 0]
    if modality == "mr":
        assert inside.mean() == pytest.approx(0, abs=1e-5) and inside.std() == pytest.approx(1, abs=1e-4)
    else:
        assert inside.min() == pytest.approx(-1) and inside.max() == pytest.approx(1)


def test_unet_shapes() -> None:
    model = UNet2d(in_channels=4, base_channels=8, depth=3)
    assert model(torch.zeros(2, 4, 40, 56)).shape == (2, 1, 40, 56)
    assert model.stride == 4


def test_point_losses(hlut: HLUT) -> None:
    ct = torch.tensor([[40.0, -900.0]])
    pred = torch.tensor([[50.0, -850.0]], requires_grad=True)
    mask = torch.ones_like(ct)
    assert point_loss(pred, ct, mask, "l1_hu", hlut).item() == pytest.approx(30.0)
    assert point_loss(pred, ct, mask, "l2_hu", hlut).item() == pytest.approx(1300.0)
    flat = torch.tensor([[-1010.0]], requires_grad=True)
    point_loss(flat, torch.tensor([[40.0]]), torch.ones(1, 1), "l2_spr", hlut).backward()
    assert flat.grad.abs().item() > 0
    with pytest.raises(ValueError):
        TrainConfig(loss="huber")


def test_folds_are_stratified_and_complete(fixture_root: Path) -> None:
    records = discover_cases(fixture_root / "data")
    folds = assign_folds(records, 2, seed=0)
    assert set(folds) == {r.case_id for r in records}
    for stratum in {r.stratum for r in records}:
        assigned = [folds[r.case_id] for r in records if r.stratum == stratum]
        assert sorted(assigned) == [0, 1]


def test_training_reduces_loss_and_predicts(fixture_root: Path, hlut: HLUT) -> None:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    assert cache.build(records) == len(records)
    assert cache.build(records) == 0
    cases = [cache.load(r.case_id) for r in records]
    model, history = train_model(cases, TrainConfig(loss="l1_hu", **{**TINY, "steps": 60}), hlut)
    assert history["loss"].iloc[-1] < history["loss"].iloc[0]
    prediction = predict_case(model, cases[0], context=1, device="cpu")
    assert prediction.shape == cases[0].mask.shape
    assert np.all(prediction[np.asarray(cases[0].mask) == 0] == -1000.0)


def test_crossfit_writes_out_of_fold_predictions_and_resumes(fixture_root: Path, hlut: HLUT) -> None:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    cache.build(records)
    out = fixture_root / "runs" / "l2_spr"
    folds = crossfit(records, cache, TrainConfig(loss="l2_spr", **TINY), hlut, 2, out)
    assert len(folds) == len(records)
    for record in records:
        prediction = read_volume(out / "predictions" / f"{record.case_id}.mha")
        reference = read_volume(record.path("ct"))
        assert prediction.array.shape == reference.array.shape
        assert prediction.spacing == reference.spacing and prediction.origin == reference.origin
    stamp = (out / "fold0.pt").stat().st_mtime_ns
    crossfit(records, cache, TrainConfig(loss="l2_spr", **TINY), hlut, 2, out)
    assert (out / "fold0.pt").stat().st_mtime_ns == stamp


def _spectra_case(error: np.ndarray, angle: float) -> object:
    shape = error.shape
    body = np.repeat((disk(shape[1:], (shape[1] / 2, shape[2] / 2), 0.45 * shape[1]) > 0.5)[None], shape[0], 0)
    ct = np.where(body, 40.0, -1000.0).astype(np.float32)
    sct = (ct + error * body).astype(np.float32)
    return prepare_case(ct, body.astype(np.float32), sct, (1.0, 1.0, 3.0), DiagnosticConfig(angles_deg=(angle,)))


def test_beam_spectra_white_and_anisotropic(hlut: HLUT, rng: np.random.Generator) -> None:
    white = beam_spectra(_spectra_case(30 * rng.normal(size=(4, 160, 160)), 0.0), hlut, 64,
                         torch.Generator().manual_seed(0))
    assert white.lf_excess("along", 0.05) == pytest.approx(1.0, abs=0.1)
    streaks = gaussian_filter1d(rng.normal(size=(4, 160, 160)), 8, axis=1)
    streaks *= 30 / streaks.std()
    along_y = beam_spectra(_spectra_case(streaks, 0.0), hlut, 64, torch.Generator().manual_seed(0))
    along_x = beam_spectra(_spectra_case(streaks, 90.0), hlut, 64, torch.Generator().manual_seed(0))
    assert along_y.lf_excess("along", 0.05) > 4 and along_y.lf_excess("across", 0.05) < 1.5
    assert along_x.lf_excess("along", 0.05) < 1.5 and along_x.lf_excess("across", 0.05) > 4


def test_preregistration_and_decision(tmp_path: Path) -> None:
    prereg = Preregistration.load(REPO / "configs" / "s3_preregistration.json")
    assert prereg.primary_source == "l1_hu" and len(prereg.fingerprint) == 64
    base = {"source": "l1_hu", "lf_excess_across": 1.0}
    go = pd.DataFrame([{**base, "lf_excess_along": 2.0, "target_global_ratio": 1.4}] * 3)
    weak = pd.DataFrame([{**base, "lf_excess_along": 1.2, "target_global_ratio": 1.0}] * 3)
    nogo = pd.DataFrame([{**base, "lf_excess_along": 1.0, "target_global_ratio": 1.0}] * 3)
    assert decide(go, prereg)["verdict"] == "GO"
    assert decide(weak, prereg)["verdict"] == "WEAK"
    assert decide(nogo, prereg)["verdict"] == "NO-GO"
    assert decide(go.assign(source="other"), prereg)["verdict"] == "NOT EVALUATED"
    json.loads((REPO / "configs" / "s3_preregistration.json").read_text())


def test_paired_test_detects_shift(rng: np.random.Generator) -> None:
    rows = []
    for index in range(20):
        base = rng.normal()
        rows += [{"case_id": f"c{index}", "region": "AB", "source": "a", "value": base + 1.0},
                 {"case_id": f"c{index}", "region": "AB", "source": "b", "value": base}]
    result = paired_test(pd.DataFrame(rows), "value", "a", "b", "AB", "greater")
    assert result["cases"] == 20 and result["p_value"] < 1e-3


def test_holm_adjustment() -> None:
    from sctrange.diagnostics import holm

    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adjusted["a"] == pytest.approx(0.03) and adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06) and np.isnan(adjusted["d"])


def _arm_frame(rng: np.random.Generator, r5_gain: float, r4_gain: float, n: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(n):
        base, mae = rng.normal(10, 1), rng.normal(60, 3)
        for arm, cvar in (("R3", base), ("R4", base - r4_gain), ("R5", base - r5_gain), ("R6", base)):
            rows.append({"case_id": f"c{index}", "region": "AB", "source": arm,
                         "target_cvar_mm": cvar + rng.normal(0, 0.05), "mae_hu": mae + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def test_arm_decision_rule(rng: np.random.Generator) -> None:
    from sctrange.diagnostics import decide_arms

    prereg = Preregistration.load(REPO / "configs" / "s4_preregistration.json")
    assert decide_arms(_arm_frame(rng, r5_gain=1.0, r4_gain=0.0), prereg)["verdict"] == "GAIN"
    assert decide_arms(_arm_frame(rng, r5_gain=1.0, r4_gain=1.5), prereg)["verdict"] == "BIAS CORRECTION SUFFICES"
    assert decide_arms(_arm_frame(rng, r5_gain=0.0, r4_gain=0.0), prereg)["verdict"] == "NO GAIN"
    assert decide_arms(_arm_frame(rng, 1.0, 0.0).query("source != 'R5' and source != 'R6'"), prereg)["verdict"] == "NOT EVALUATED"


def test_spectrum_angles_separate_along_from_across(hlut: HLUT, rng: np.random.Generator) -> None:
    from sctrange.diagnostics import diagnose_case

    prereg = Preregistration.load(REPO / "configs" / "s3_preregistration.json")
    shape = (3, 160, 160)
    body = np.repeat((disk(shape[1:], (80, 80), 72) > 0.5)[None], 3, 0).astype(np.float32)
    ct = np.where(body > 0, 40.0, -1000.0).astype(np.float32)
    streaks = gaussian_filter1d(rng.normal(size=shape), 8, axis=1)
    sct = (ct + 30 * streaks / streaks.std() * body).astype(np.float32)
    row, _ = diagnose_case(ct, body, sct, (1.0, 1.0, 3.0), hlut, DiagnosticConfig(angles_deg=(0.0, 90.0)), prereg)
    assert abs(row["lf_excess_along"] - row["lf_excess_across"]) > 0.3
    with pytest.raises(ValueError):
        Preregistration({**prereg.raw, "spectrum_angles_deg": [0.0, 90.0]}, "x").spectrum_angles
