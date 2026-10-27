import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sctrange.data import CaseRecord
from sctrange.dosimetry import (DoseConfig, average_seeds, decide_go_no_go_2, distal_positions, dose_case,
                                gantry_angle, load_selection, partial_spearman, proxy_validation, select_cases,
                                write_selection)
from sctrange.gamma import gamma_pass_rate, search_offsets_mm
from sctrange.phantoms import synthetic_ct

REPO = Path(__file__).resolve().parents[1]
HAS_PYRADPLAN = importlib.util.find_spec("pyRadPlan") is not None


def _blob(shape=(12, 60, 60), spacing=(3.0, 1.0, 1.0), shift_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
    z, y, x = (np.indices(shape).astype(float) * np.array(spacing)[:, None, None, None])
    centre = np.array(shape) * np.array(spacing) / 2 + np.array(shift_mm)
    return 50 * np.exp(-((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2) / (2 * 12.0**2))


def test_gamma_offsets_and_identity() -> None:
    offsets = search_offsets_mm(2.0, 0.5)
    assert np.allclose(offsets[0], 0) and np.all(np.linalg.norm(offsets, axis=1) <= 2.0 + 1e-9)
    dose = _blob()
    assert gamma_pass_rate(dose, dose, (3.0, 1.0, 1.0)).pass_rate == 1.0


def test_gamma_dose_and_distance_criteria() -> None:
    dose, spacing = _blob(), (3.0, 1.0, 1.0)
    assert gamma_pass_rate(dose, 1.015 * dose, spacing).pass_rate == 1.0
    assert gamma_pass_rate(dose, _blob(shift_mm=(0, 0, 1.5)), spacing).pass_rate == 1.0
    assert gamma_pass_rate(dose, _blob(shift_mm=(0, 0, 6.0)), spacing).pass_rate < 0.9
    shifted_z = _blob(shift_mm=(3.0, 0, 0))
    strict = gamma_pass_rate(dose, shifted_z, spacing, 1.0, 2.0).pass_rate
    loose = gamma_pass_rate(dose, shifted_z, spacing, 1.0, 3.5).pass_rate
    assert strict < 0.9 < 0.95 < loose


def test_distal_positions_on_analytic_profile() -> None:
    depth = np.arange(100.0)
    profile = np.where(depth < 60, 1.0, np.clip(1.0 - (depth - 60) / 10, 0, 1))
    canvas = np.tile(profile[None, :, None], (2, 1, 3))
    positions = distal_positions(canvas, 0.8, 0.5)
    assert np.allclose(positions, 62.0)
    canvas[:, :, 2] = 0.0
    assert np.isnan(distal_positions(canvas, 0.8, 0.5)[:, 2]).all()


def test_gantry_convention() -> None:
    assert [gantry_angle(a) for a in (0.0, 90.0, 120.0, 240.0)] == [0.0, 270.0, 240.0, 120.0]


def test_selection_is_stratified_deterministic_and_frozen(tmp_path: Path) -> None:
    records = [CaseRecord(f"1{region}A{i:03d}", "1", region, "A", tmp_path) for region in ("AB", "HN", "TH")
               for i in range(10 if region == "AB" else 5)]
    first = select_cases(records, 8, seed=1)
    assert first == select_cases(records, 8, seed=1) and len(first) == 8
    assert sum("AB" in i for i in first) == 4
    path = tmp_path / "cases.json"
    write_selection(first, path, {"seed": 1})
    assert load_selection(path)["case_ids"] == sorted(first)
    with pytest.raises(ValueError):
        write_selection(select_cases(records, 8, seed=2), path, {"seed": 2})


def test_partial_spearman_removes_confounder(rng: np.random.Generator) -> None:
    confounder = rng.normal(size=400)
    x = confounder + 0.1 * rng.normal(size=400)
    y = confounder + 0.1 * rng.normal(size=400)
    assert abs(partial_spearman(x, y, confounder)) < 0.2
    assert partial_spearman(x + y, y, confounder) > 0.5


def _decision_frame(gain: float) -> pd.DataFrame:
    rows = []
    for index in range(20):
        base = 95 + index * 0.1
        for arm, value in (("R3", base), ("R4", base + 0.05), ("R5", base + gain), ("R6", base)):
            rows.append({"case_id": f"c{index}", "source": arm, "proton_gamma_2_2": value + 0.01 * (index % 3)})
    return pd.DataFrame(rows)


def test_go_no_go_2_rules() -> None:
    prereg = json.loads((REPO / "configs" / "s6_preregistration.json").read_text())
    s4 = lambda reference, control: {"decision": {"verdict": "GAIN", "arms": {  # noqa: E731
        "R5": {"p_vs_reference_holm": reference, "p_vs_control_holm": control, "mae_within_tolerance": True},
        "R6": {"p_vs_reference_holm": 0.5, "p_vs_control_holm": 0.5, "mae_within_tolerance": True}}}}
    assert decide_go_no_go_2(_decision_frame(1.0), s4(0.001, 0.001), prereg)["verdict"] == "GO"
    assert decide_go_no_go_2(_decision_frame(0.0), s4(0.001, 0.001), prereg)["verdict"] == \
        "RANGE GAIN WITHOUT DOSE CONFIRMATION"
    assert decide_go_no_go_2(_decision_frame(1.0), s4(0.001, 0.5), prereg)["verdict"] == "RIGHT ESTIMATOR SUFFICES"
    assert decide_go_no_go_2(_decision_frame(1.0), s4(0.5, 0.5), prereg)["verdict"] == "NO-GO"
    assert decide_go_no_go_2(_decision_frame(1.0), None, prereg)["verdict"] == "NOT EVALUATED"


def test_average_seeds() -> None:
    frame = pd.DataFrame({"case_id": ["a"] * 3, "source": ["R3@0", "R3@1", "bulk density"], "m": [1.0, 3.0, 5.0]})
    averaged = average_seeds(frame).set_index("source")["m"]
    assert averaged["R3"] == 2.0 and averaged["bulk density"] == 5.0


@pytest.mark.skipif(not HAS_PYRADPLAN, reason="pyRadPlan not installed")
def test_dose_case_end_to_end(rng: np.random.Generator) -> None:
    from sctrange.hlut import HLUT

    ct, mask = synthetic_ct("AB", (72, 88), 10, rng)
    half = np.zeros_like(mask, dtype=bool)
    half[..., : mask.shape[-1] // 2] = True
    sources = {"exact": ct.copy(), "plus": np.where((mask > 0) & half, ct + 80, ct)}
    config = DoseConfig(target_radius_mm=12.0, dose_grid_mm=4.0, bixel_width_mm=6.0,
                        photon_angles_deg=(0.0, 120.0, 240.0))
    result = dose_case(ct, mask, (1.0, 1.0, 3.0), sources, HLUT.pyradplan_default(), config)
    rows = pd.DataFrame(result.rows).set_index("source")
    assert rows.loc["exact", "proton_gamma_2_2"] == 100.0 and rows.loc["exact", "proxy_rms_mm"] == 0.0
    assert rows.loc["plus", "proton_gamma_1_1"] < rows.loc["plus", "photon_gamma_1_1"]
    assert rows.loc["plus", "proxy_mean_signed_mm"] > 0
    plus = result.rays[result.rays["source"] == "plus"]
    assert len(plus) > 20 and plus["delta_r80_mm"].median() < 0 < plus["delta_wet_mm"].median()
    validation = proxy_validation(
        pd.DataFrame({"case_id": ["a", "a", "b", "b", "c", "c"], "proxy_rms_mm": [0, 1, 0, 2, 0, 3],
                      "proton_gamma_fail_2_2": [0, 1, 0, 2, 0, 4], "mae_hu": [0, 5, 1, 6, 2, 7]}),
        plus, json.loads((REPO / "configs" / "s6_preregistration.json").read_text()), n_boot=50)
    assert validation["ray_spearman"] > 0.5 and validation["case_spearman"] > 0.5


@pytest.mark.skipif(not HAS_PYRADPLAN, reason="pyRadPlan not installed")
def test_engine_checks_beam_direction_and_identity() -> None:
    from sctrange.dosimetry import engine_checks

    checks = engine_checks(DoseConfig(proton_angles_deg=(90.0,), dose_grid_mm=4.0), shape_yx=(64, 64), n_slices=8,
                           radius_mm=8.0)
    assert checks["direction_ok"].all() and (checks["identity_gamma_2_2"] == 100.0).all()
