from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sctrange.cache import VolumeCache
from sctrange.correction import apply_bias, bias_sums, crossfit_bias_correction, tissue_classes
from sctrange.data import discover_cases, load_case, read_volume, write_volume
from sctrange.evaluation import DiagnosticConfig, evaluate_risk
from sctrange.hlut import HLUT
from sctrange.losses import ray_risk
from sctrange.metrics import TISSUE_EDGES_HU
from sctrange.normalization import encode_ct
from sctrange.phantoms import disk, write_fixture_dataset
from sctrange.projection import ParallelBeamProjector
from sctrange.runs import mark_complete, run_complete
from sctrange.training import (
    ARMS,
    TrainConfig,
    arm_config,
    beam_angles,
    build_model,
    crossfit,
    main,
    training_objective,
)

TINY = dict(context=1, patch=48, batch_size=4, steps=20, base_channels=8, depth=3, warmup_steps=5,
            log_every=10, device="cpu", samples_per_case=300, min_path_mm=8.0)


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("arms")
    write_fixture_dataset(root / "data", regions=("AB",), centers=("A", "B"), cases_per_stratum=2,
                          shape_yx=(56, 64), n_slices=3, seed=5)
    return root


def _batch(size: int = 40) -> tuple[torch.Tensor, torch.Tensor]:
    """Water-like body at 40 HU: +-30 HU stays on one HLUT segment, so |SPR error| is the same for both signs."""
    body = torch.from_numpy(disk((size, size), (size / 2, size / 2), 0.4 * size) > 0.5).float()[None, None]
    ct = torch.where(body > 0, torch.tensor(40.0), torch.tensor(-1000.0)).repeat(2, 1, 1, 1)
    return ct, body.repeat(2, 1, 1, 1)


def test_arm_table_matches_plan() -> None:
    assert arm_config("R1").loss == "l1_hu" and arm_config("R3").loss == "l2_spr"
    assert arm_config("R5").range_weight > 0 and arm_config("R5").out_channels == 1
    assert arm_config("R6").risk_weight > 0 and arm_config("R6").out_channels == 3
    assert set(ARMS) == {"R1", "R2", "R3", "R5", "R6"}
    with pytest.raises(ValueError):
        arm_config("R4")
    with pytest.raises(ValueError):
        TrainConfig(range_weight=-1.0)


def test_beam_angles_keep_plan_and_add_random() -> None:
    config = arm_config("R5", plan_angles=(0.0, 90.0), random_angles=2)
    angles = beam_angles(config, np.random.default_rng(0))
    assert angles[:2] == (0.0, 90.0) and len(angles) == 4
    assert beam_angles(config, np.random.default_rng(1))[2:] != angles[2:]


def test_range_term_penalises_coherent_bias_more_than_white_noise(hlut: HLUT, rng: np.random.Generator) -> None:
    ct, body = _batch()
    config = arm_config("R5", **TINY)
    angles = (0.0, 45.0, 90.0)
    white = ct + torch.from_numpy(rng.choice([-30.0, 30.0], ct.shape)).float() * body
    coherent = ct + 30.0 * body
    losses = {}
    for name, sct in (("white", white), ("coherent", coherent)):
        output = encode_ct(sct).requires_grad_(True)
        loss, parts = training_objective(output, ct, body, config, hlut, angles)
        loss.backward()
        assert output.grad.abs().sum() > 0
        losses[name] = parts
    # identical voxel L2 (same |error| everywhere), very different range error
    assert losses["white"]["point"].item() == pytest.approx(losses["coherent"]["point"].item(), rel=1e-3)
    assert losses["coherent"]["range"] > 10 * losses["white"]["range"]


def test_risk_head_gets_gradients_but_does_not_move_the_sct(hlut: HLUT) -> None:
    ct, body = _batch()
    config = replace(arm_config("R6", **TINY), range_weight=0.0)
    sct_channel = encode_ct(ct + 20.0 * body)
    risk = torch.full_like(sct_channel, -4.0).repeat(1, 2, 1, 1)
    output = torch.cat([sct_channel, risk], dim=1).requires_grad_(True)
    loss, parts = training_objective(output, ct, body, config, hlut, (0.0, 90.0))
    loss.backward()
    assert "risk" in parts and torch.isfinite(parts["risk"])
    assert output.grad[:, 1:].abs().sum() > 0
    reference = output.detach().requires_grad_(True)
    training_objective(reference, ct, body, replace(config, risk_weight=0.0), hlut)[0].backward()
    torch.testing.assert_close(output.grad[:, :1], reference.grad[:, :1])


def test_ray_risk_orders_var_below_es() -> None:
    projector = ParallelBeamProjector((20, 20), (0.0, 60.0), 1.0)
    body = torch.from_numpy(disk((20, 20), (10, 10), 8) > 0.5).float()[None]
    weights, _ = projector.segment_weights(body)
    var, es = ray_risk(projector, 0.02 * body, 0.01 * body, weights)
    assert (es >= var).all() and (var > 0).all()


def test_r6_crossfit_writes_risk_maps_and_risk_is_evaluable(fixture_root: Path, hlut: HLUT) -> None:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    cache.build(records)
    out = fixture_root / "runs" / "R6"
    crossfit(records, cache, arm_config("R6", **TINY), hlut, 2, out)
    history = pd.read_csv(out / "history.csv")
    assert {"point", "range", "risk", "fold"} <= set(history.columns) and set(history["fold"]) == {0, 1}
    record = records[0]
    volumes = load_case(record, ("ct", "mask"))
    sct = read_volume(out / "predictions" / f"{record.case_id}.mha").array
    var = read_volume(out / "risk" / f"{record.case_id}_var.mha").array
    excess = read_volume(out / "risk" / f"{record.case_id}_excess.mha").array
    assert var.shape == sct.shape and (var >= 0).all() and (var[volumes["mask"].array == 0] == 0).all()
    row = evaluate_risk(volumes["ct"].array, volumes["mask"].array, sct, var, excess, volumes["ct"].spacing, hlut,
                        DiagnosticConfig(angles_deg=(0.0, 90.0), target_radius_mm=10.0, min_path_mm=5.0))
    assert row["risk_rays"] > 0 and 0.0 <= row["risk_coverage"] <= 1.0 and np.isfinite(row["risk_fz"])


def test_checkpoint_from_another_configuration_is_refused(fixture_root: Path, hlut: HLUT) -> None:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    cache.build(records)
    out = fixture_root / "runs" / "mixed"
    crossfit(records, cache, arm_config("R3", **TINY), hlut, 2, out)
    crossfit(records, cache, arm_config("R3", **{**TINY, "device": "cpu", "log_every": 5}), hlut, 2, out)
    with pytest.raises(ValueError, match="different configuration"):
        crossfit(records, cache, arm_config("R5", **TINY), hlut, 2, out)


def test_tissue_classes_follow_permutation_bins() -> None:
    hu = np.array([-1000.0, -500.0, -100.1, -100.0, 149.9, 150.0, 3000.0])
    np.testing.assert_array_equal(tissue_classes(hu, TISSUE_EDGES_HU), [0, 1, 1, 2, 2, 3, 3])


def test_bias_correction_removes_a_tissue_offset(hlut: HLUT, rng: np.random.Generator) -> None:
    mask = np.ones((2, 30, 30), dtype=np.float32)
    ct = np.where(rng.random(mask.shape) < 0.5, 40.0, 600.0)
    sct = ct + np.where(ct < 150, 25.0, -80.0)
    sums, counts = bias_sums(sct, ct, mask, hlut)
    offsets = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    corrected = apply_bias(sct, mask, offsets, hlut)
    before = np.abs(hlut.numpy(sct) - hlut.numpy(ct)).mean()
    after = np.abs(hlut.numpy(corrected) - hlut.numpy(ct)).mean()
    assert after < 0.05 * before


def test_crossfit_bias_correction_is_leak_free(fixture_root: Path, hlut: HLUT, tmp_path: Path) -> None:
    records = discover_cases(fixture_root / "data")
    source = tmp_path / "R3"
    folds = {r.case_id: i % 2 for i, r in enumerate(records)}
    source.mkdir()
    pd.DataFrame(sorted(folds.items()), columns=["case_id", "fold"]).to_csv(source / "folds.csv", index=False)
    for record in records:
        ct = load_case(record, ("ct",))["ct"]
        write_volume(ct.array + (30.0 if folds[record.case_id] == 0 else -30.0), source / "predictions" /
                     f"{record.case_id}.mha", like=ct)
    with pytest.raises(FileNotFoundError, match="has not finished"):
        crossfit_bias_correction(records, source, tmp_path / "R4", hlut)
    mark_complete(source, folds)
    offsets = crossfit_bias_correction(records, source, tmp_path / "R4", hlut)
    # the offsets for fold 0 come from fold 1 (sCT too low there), so they are positive in soft tissue
    soft = "[-100, 150) HU"
    assert offsets.loc[0, soft] > 0 > offsets.loc[1, soft]
    assert len(list((tmp_path / "R4" / "predictions").glob("*.mha"))) == len(records)
    assert run_complete(tmp_path / "R4", folds)
    stamp = (tmp_path / "R4" / "offsets.csv").stat().st_mtime_ns
    pd.testing.assert_frame_equal(crossfit_bias_correction(records, source, tmp_path / "R4", hlut), offsets,
                                  check_dtype=False)
    assert (tmp_path / "R4" / "offsets.csv").stat().st_mtime_ns == stamp  # unchanged source: not recomputed


def test_r4_cli_requires_r3(fixture_root: Path, tmp_path: Path) -> None:
    from sctrange.split import SplitSpec, stratified_split, write_split

    records = discover_cases(fixture_root / "data")
    write_split(stratified_split(records, SplitSpec(test_fraction=0.25)), tmp_path / "split.json")
    with pytest.raises(SystemExit):
        main(["--data-root", str(fixture_root / "data"), "--split", str(tmp_path / "split.json"),
              "--cache-dir", str(tmp_path / "cache"), "--out", str(tmp_path / "runs"), "--arm", "R4"])


def test_autocast_dtype_falls_back_to_float16_without_native_bfloat16(monkeypatch: pytest.MonkeyPatch) -> None:
    from sctrange.training import autocast_dtype

    assert autocast_dtype(torch.device("cpu")) is None
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))  # T4
    assert autocast_dtype(torch.device("cuda:1")) is torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))  # A100
    assert autocast_dtype(torch.device("cuda:0")) is torch.bfloat16


def test_parallel_folds_match_the_sequential_run(fixture_root: Path, hlut: HLUT) -> None:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    cache.build(records)
    config = arm_config("R3", **{**TINY, "steps": 6, "log_every": 3})
    sequential = fixture_root / "runs" / "sequential"
    parallel = fixture_root / "runs" / "parallel"
    crossfit(records, cache, config, hlut, 2, sequential)
    crossfit(records, cache, config, hlut, 2, parallel, devices=["cpu", "cpu"])
    history = pd.read_csv(parallel / "history.csv")
    assert set(history["fold"]) == {0, 1} and set(history["device"]) == {"cpu"}
    for record in records:
        a = read_volume(sequential / "predictions" / f"{record.case_id}.mha").array
        b = read_volume(parallel / "predictions" / f"{record.case_id}.mha").array
        np.testing.assert_array_equal(a, b)
    import SimpleITK as sitk

    stored = sitk.ReadImage(str(parallel / "predictions" / f"{records[0].case_id}.mha"))
    assert stored.GetPixelID() == sitk.sitkInt16


def test_build_model_initial_outputs() -> None:
    model = build_model(arm_config("R6", **TINY))
    with torch.no_grad():
        output = model(torch.zeros(1, 4, 16, 16))
    assert output.shape[1] == 3
    assert float(torch.nn.functional.softplus(output[0, 1]).mean()) == pytest.approx(0.01, rel=0.5)


def test_arm_improving_mae_and_tail_is_a_gain() -> None:
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from sctrange.diagnostics import Preregistration, decide_arms

    prereg = Preregistration.load(Path(__file__).resolve().parents[1] / "configs" / "s4_preregistration.json")
    rng = np.random.default_rng(0)
    rows = []
    for index in range(30):
        base = rng.normal(5, 1)
        for arm, tail, mae in (("R3", base, 60.0), ("R4", base - 0.1, 60.0), ("R5", base - 1.0, 50.0),
                               ("R6", base + 0.5, 70.0)):
            rows.append({"case_id": f"c{index}", "region": "AB", "source": arm, "mae_hu": mae,
                         "target_cvar_mm": tail + rng.normal(0, 0.05)})
    decision = decide_arms(pd.DataFrame(rows), prereg)
    assert decision["verdict"] == "GAIN"
    assert decision["arms"]["R5"]["mae_within_tolerance"] and not decision["arms"]["R6"]["mae_within_tolerance"]
