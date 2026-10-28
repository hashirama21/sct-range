"""S6 dosimetry: proton and photon plans optimised on the reference CT, recomputed on synthetic CTs.

Protocol, per case (SynthRAD2025 releases no PTV, so the target is the 25 mm spherical surrogate of the range
harness):

1. crop to the body (plus a margin) and set every voxel outside the body mask to air in the reference CT and in each
   sCT, so that only errors inside the patient reach the dose;
2. optimise an IMPT plan (3 beams) and an IMRT plan (7 beams) on the reference CT with pyRadPlan (the dose engine of
   the SynthRAD2025 evaluation and its default HU -> SPR table);
3. recompute both plans with the same weights on every sCT (``calc_dose_forward``) and on the reference CT;
4. compare: global gamma pass rates, target DVH points, and for protons the distal R80 shift of each beam along every
   ray, matched with the WET error of the same ray to the distal target edge (projector of ``sctrange.projection``).

Beam angles are given in the projector convention (angle 0 travels along +y, 90 along +x). pyRadPlan's gantry 0
also travels along +y but gantry 90 travels along -x, so ``gantry = (360 - angle) mod 360``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy.stats import spearmanr, theilslopes

from .data import CaseRecord, bounding_box, discover_cases, load_case, read_prediction, spherical_target
from .diagnostics import holm, paired_test
from .download import usable_records
from .evaluation import DiagnosticConfig, evaluate_case
from .gamma import gamma_pass_rate
from .hlut import HLUT
from .metrics import range_error
from .projection import ParallelBeamProjector
from .split import load_selection, select_cases, write_selection  # noqa: F401  (part of this module's interface)

AIR_HU = -1000.0
MODALITIES = ("proton", "photon")


@dataclass(frozen=True)
class DoseConfig:
    proton_angles_deg: tuple[float, ...] = (0.0, 120.0, 240.0)
    photon_angles_deg: tuple[float, ...] = tuple(round(k * 360.0 / 7.0, 4) for k in range(7))
    bixel_width_mm: float = 5.0
    dose_grid_mm: float = 3.0
    prescription_gy: float = 60.0
    fractions: int = 30
    target_radius_mm: float = 25.0
    body_limit_fraction: float = 0.5
    margin_voxels: int = 8
    gamma_criteria: tuple[tuple[float, float], ...] = ((2.0, 2.0), (1.0, 1.0))
    gamma_cutoff_percent: float = 10.0
    machine: str = "Generic"
    min_path_mm: float = 20.0
    distal_level: float = 0.8
    ray_dose_fraction: float = 0.5

    def angles(self, modality: str) -> tuple[float, ...]:
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}")
        return self.proton_angles_deg if modality == "proton" else self.photon_angles_deg

    @classmethod
    def from_preregistration(cls, raw: Mapping) -> DoseConfig:
        plans = raw["plans"]
        return cls(proton_angles_deg=tuple(plans["proton_angles_deg"]), photon_angles_deg=tuple(plans["photon_angles_deg"]),
                   bixel_width_mm=plans["bixel_width_mm"], dose_grid_mm=plans["dose_grid_mm"],
                   prescription_gy=plans["prescription_gy"], target_radius_mm=plans["target_radius_mm"],
                   gamma_criteria=tuple(tuple(c) for c in raw["gamma"]["criteria"]),
                   gamma_cutoff_percent=raw["gamma"]["cutoff_percent"])


def gantry_angle(angle_deg: float) -> float:
    """pyRadPlan gantry angle of a beam given in the projector convention."""
    return float((360.0 - angle_deg) % 360.0)


def _pyradplan():
    os.environ.setdefault("TQDM_DISABLE", "1")
    warnings.filterwarnings("ignore", module=r"pyRadPlan.*")
    warnings.filterwarnings("ignore", message="IProgress not found")
    import pyRadPlan
    from pyRadPlan.cst import create_cst
    from pyRadPlan.cst._voi import ExternalVOI, Target
    from pyRadPlan.ct import create_ct

    try:  # pyRadPlan >= 0.5 renders console progress bars per beam and ray
        from pyRadPlan.core._progress import ProgressReporter

        ProgressReporter.console_progress = False
    except ImportError:
        pass
    return pyRadPlan, create_ct, create_cst, Target, ExternalVOI


def _array(image) -> np.ndarray:
    return sitk.GetArrayFromImage(image) if isinstance(image, sitk.Image) else np.asarray(image)


def _image(array: np.ndarray, spacing_xyz: Sequence[float], dtype=np.float32) -> sitk.Image:
    image = sitk.GetImageFromArray(np.ascontiguousarray(array, dtype=dtype))
    image.SetSpacing(tuple(float(s) for s in spacing_xyz))
    return image


@dataclass
class CasePlan:
    """A plan optimised on the reference CT, with everything needed to recompute it on another CT."""

    modality: str
    box: tuple[slice, ...]
    spacing_xyz: tuple[float, float, float]
    body: np.ndarray
    target: np.ndarray
    reference_ct: np.ndarray
    weights: np.ndarray
    reference_dose: np.ndarray
    reference_beams: list[np.ndarray]
    handles: dict = field(repr=False)


def _masked_crop(hu: np.ndarray, plan_box: tuple[slice, ...], body: np.ndarray) -> np.ndarray:
    return np.where(body, np.asarray(hu, dtype=np.float32)[plan_box], AIR_HU).astype(np.float32)


def plan_case(ct_hu: np.ndarray, mask: np.ndarray, target: np.ndarray, spacing_xyz: Sequence[float], modality: str,
              config: DoseConfig = DoseConfig()) -> CasePlan:
    """Optimise one plan on the reference CT and compute its reference dose with the forward engine."""
    pyRadPlan, create_ct, create_cst, Target, ExternalVOI = _pyradplan()
    box = bounding_box(mask, margin=config.margin_voxels)
    body = np.asarray(mask[box]) > 0.5
    target_crop = (np.asarray(target[box]) > 0.5) & body
    if not target_crop.any():
        raise ValueError("empty target")
    reference = _masked_crop(ct_hu, box, body)
    image = _image(reference, spacing_xyz)
    ct = create_ct(cube_hu=image)

    def voi_mask(array: np.ndarray) -> sitk.Image:
        voi = _image(array, spacing_xyz, np.uint8)
        voi.CopyInformation(image)
        return voi

    prescription = config.prescription_gy
    cst = create_cst(vois=[
        Target(name="target", mask=voi_mask(target_crop), ct_image=ct,
               objectives=[{"name": "Squared Deviation", "d_ref": prescription, "priority": 800.0}]),
        ExternalVOI(name="body", mask=voi_mask(body), ct_image=ct,
                    objectives=[{"name": "Squared Overdosing", "d_max": config.body_limit_fraction * prescription,
                                 "priority": 10.0}]),
    ], ct=ct)
    if modality == "proton":
        pln = pyRadPlan.IonPlan(radiation_mode="protons", machine=config.machine)
    elif modality == "photon":
        pln = pyRadPlan.PhotonPlan(machine=config.machine)
    else:
        raise ValueError(f"modality must be one of {MODALITIES}")
    pln.num_of_fractions = config.fractions
    pln.prescribed_dose = prescription
    angles = [gantry_angle(a) for a in config.angles(modality)]
    pln.prop_stf = {"gantry_angles": angles, "couch_angles": [0.0] * len(angles), "bixel_width": config.bixel_width_mm}
    grid = config.dose_grid_mm
    pln.prop_dose_calc = {"dose_grid": {"resolution": {"x": grid, "y": grid, "z": grid}}}
    pln.prop_opt = {"solver": "scipy"}
    with np.errstate(invalid="ignore", divide="ignore"):  # pyRadPlan's Siddon tracer differences infinite bounds
        stf = pyRadPlan.generate_stf(ct, cst, pln)
        dij = pyRadPlan.calc_dose_influence(ct, cst, stf, pln)
        weights = np.asarray(pyRadPlan.fluence_optimization(ct, cst, stf, dij, pln), dtype=np.float64)
    handles = {"cst": cst, "pln": pln, "stf": stf, "forward": pyRadPlan.calc_dose_forward, "create_ct": create_ct}
    plan = CasePlan(modality, box, tuple(float(s) for s in spacing_xyz), body, target_crop, reference, weights,
                    np.empty(0), [], handles)
    plan.reference_dose, plan.reference_beams = _forward(plan, reference)
    return plan


def _forward(plan: CasePlan, cropped_hu: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h = plan.handles
    ct = h["create_ct"](cube_hu=_image(cropped_hu, plan.spacing_xyz))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = h["forward"](ct, h["cst"], h["stf"], h["pln"], plan.weights)
    beams = [_array(b).astype(np.float32) for b in result.get("physical_dose_beam", [])]
    return _array(result["physical_dose"]).astype(np.float32), beams


def recompute(plan: CasePlan, sct_hu: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Dose of the plan's fixed weights on ``sct_hu`` (full grid); returns dose, per-beam doses and the cropped sCT."""
    cropped = _masked_crop(sct_hu, plan.box, plan.body)
    dose, beams = _forward(plan, cropped)
    return dose, beams, cropped


def dvh_point(dose: np.ndarray, mask: np.ndarray, volume_percent: float) -> float:
    """``D_v``: the dose received by at least ``volume_percent`` % of the structure."""
    values = dose[mask]
    return float(np.percentile(values, 100.0 - volume_percent)) if values.size else float("nan")


def dose_metrics(plan: CasePlan, dose: np.ndarray, config: DoseConfig = DoseConfig()) -> dict[str, float]:
    """Gamma pass rates (%) and target DVH differences (% of prescription) against the reference recomputation."""
    reference = plan.reference_dose
    spacing_zyx = plan.spacing_xyz[::-1]
    prescription = config.prescription_gy
    row: dict[str, float] = {}
    for dose_percent, dta in config.gamma_criteria:
        tag = f"{dose_percent:g}_{dta:g}".replace(".", "p")
        result = gamma_pass_rate(reference, dose, spacing_zyx, dose_percent, dta, config.gamma_cutoff_percent)
        row[f"gamma_{tag}"] = 100.0 * result.pass_rate
        row[f"gamma_fail_{tag}"] = 100.0 * (1.0 - result.pass_rate)
    target = plan.target
    for name, value in (("mean", None), ("d95", 95.0), ("d2", 2.0)):
        ref = float(reference[target].mean()) if value is None else dvh_point(reference, target, value)
        new = float(dose[target].mean()) if value is None else dvh_point(dose, target, value)
        row[f"target_{name}_diff_pct"] = 100.0 * (new - ref) / prescription
    high = reference >= config.gamma_cutoff_percent / 100.0 * reference.max()
    row["dose_mae_pct"] = 100.0 * float(np.abs(dose - reference)[high].mean()) / prescription
    return row


def distal_positions(canvas: np.ndarray, level: float, min_fraction: float) -> np.ndarray:
    """Depth (pixels, sub-pixel) of the distal ``level`` crossing of every ray of a beam's-eye dose ``(Z, S, S)``.

    Depth runs along axis 1. Rays whose maximum is below ``min_fraction`` of the beam maximum, or that never fall
    below the level after their maximum, are NaN.
    """
    peak = canvas.max(axis=1)
    threshold = level * peak
    above = canvas >= threshold[:, None, :]
    depth = canvas.shape[1]
    last = depth - 1 - np.argmax(above[:, ::-1, :], axis=1)
    inside = last < depth - 1
    nxt = np.minimum(last + 1, depth - 1)
    z, col = np.indices(last.shape)
    high_value, low_value = canvas[z, last, col], canvas[z, nxt, col]
    drop = high_value - low_value
    with np.errstate(invalid="ignore", divide="ignore"):
        position = last + np.where(drop > 0, (high_value - threshold) / drop, 0.0)
    valid = inside & (peak >= min_fraction * canvas.max()) & (peak > 0)
    return np.where(valid, position, np.nan)


def distal_shifts(plan: CasePlan, beams: list[np.ndarray], sct_cropped: np.ndarray, hlut: HLUT,
                  config: DoseConfig = DoseConfig()) -> pd.DataFrame:
    """Per proton beam and ray: distal R80 shift of the sCT dose (mm, positive = deeper) and WET error (mm of water)
    from entry to the distal target edge along the same ray."""
    if plan.modality != "proton":
        raise ValueError("distal shifts are defined for proton plans")
    angles = config.proton_angles_deg
    projector = ParallelBeamProjector(plan.body.shape[-2:], angles, plan.spacing_xyz[0])
    as_tensor = lambda array: torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32)  # noqa: E731
    body, target = as_tensor(plan.body), as_tensor(plan.target)
    wet = range_error(as_tensor(sct_cropped), as_tensor(plan.reference_ct), body, projector, hlut, target, slab=16)
    selected = wet.selected(config.min_path_mm).numpy()
    frames = []
    for index, angle in enumerate(angles):
        reference = projector.rotate(as_tensor(plan.reference_beams[index]))[index].numpy()
        evaluated = projector.rotate(as_tensor(beams[index]))[index].numpy()
        shift = (distal_positions(evaluated, config.distal_level, config.ray_dose_fraction)
                 - distal_positions(reference, config.distal_level, config.ray_dose_fraction)) * plan.spacing_xyz[0]
        keep = selected[index] & np.isfinite(shift)
        z, column = np.nonzero(keep)
        frames.append(pd.DataFrame({"beam_deg": angle, "z": z, "ray": column, "delta_r80_mm": shift[keep],
                                    "delta_wet_mm": wet.delta_mm[index].numpy()[keep],
                                    "path_mm": wet.path_mm[index].numpy()[keep]}))
    return pd.concat(frames, ignore_index=True)


@dataclass
class CaseDose:
    rows: list[dict]
    rays: pd.DataFrame
    slices: dict[str, np.ndarray]


def dose_case(ct_hu: np.ndarray, mask: np.ndarray, spacing_xyz: Sequence[float], sources: Mapping[str, np.ndarray],
              hlut: HLUT, config: DoseConfig = DoseConfig(), target: np.ndarray | None = None,
              modalities: Sequence[str] = MODALITIES) -> CaseDose:
    """Plan on the reference CT, recompute on every source, return metric rows, proton ray shifts and central slices.

    Each row also carries the range metrics of the source on the proton beams (``proxy_*``, entry to distal target
    edge), so that the range proxy and the dose can be compared case by case.
    """
    if target is None:
        target = spherical_target(mask, spacing_xyz, config.target_radius_mm)
    proxy_config = DiagnosticConfig(angles_deg=config.proton_angles_deg, target_radius_mm=config.target_radius_mm,
                                    min_path_mm=config.min_path_mm)
    proxies = {}
    for name, sct in sources.items():
        metrics = evaluate_case(ct_hu, mask, sct, spacing_xyz, hlut, proxy_config, target)
        proxies[name] = {"mae_hu": metrics["mae_hu"], **{f"proxy_{k[7:]}": v for k, v in metrics.items()
                                                         if k.startswith("target_") and "ratio" not in k}}
    rows = {name: {"source": name, **proxy} for name, proxy in proxies.items()}
    rays, slices = [], {}
    for modality in modalities:
        plan = plan_case(ct_hu, mask, target, spacing_xyz, modality, config)
        centre = int(np.argmax(plan.target.sum(axis=(1, 2))))
        slices[f"{modality}/reference"] = plan.reference_dose[centre]
        slices.setdefault("ct", plan.reference_ct[centre])
        slices.setdefault("target", plan.target[centre].astype(np.uint8))
        for name, sct in sources.items():
            dose, beams, cropped = recompute(plan, sct)
            rows[name].update({f"{modality}_{k}": v for k, v in dose_metrics(plan, dose, config).items()})
            slices[f"{modality}/{name}"] = dose[centre]
            if modality == "proton":
                rays.append(distal_shifts(plan, beams, cropped, hlut, config).assign(source=name))
    ray_frame = pd.concat(rays, ignore_index=True) if rays else pd.DataFrame()
    return CaseDose(list(rows.values()), ray_frame, slices)


def engine_checks(config: DoseConfig = DoseConfig(), shape_yx: tuple[int, int] = (96, 96), n_slices: int = 12,
                  radius_mm: float = 12.0) -> pd.DataFrame:
    """Two checks of the dose pipeline on a water cylinder (1 x 1 x 3 mm):

    - beam direction: a single-beam proton plan per angle deposits dose upstream of the target along the projector
      direction and almost none downstream, which validates ``gantry_angle``;
    - identity: recomputing the optimised weights on the planning CT reproduces the reference dose (gamma 100 %).
    """
    spacing = (1.0, 1.0, 3.0)
    z, y, x = np.indices((n_slices,) + tuple(shape_yx)).astype(float)
    cy, cx, cz = (shape_yx[0] - 1) / 2, (shape_yx[1] - 1) / 2, (n_slices - 1) / 2
    mask = ((y - cy) ** 2 + (x - cx) ** 2 <= (0.45 * min(shape_yx)) ** 2).astype(np.float32)
    ct = np.where(mask > 0, 0.0, AIR_HU).astype(np.float32)
    target = ((y - cy) ** 2 + (x - cx) ** 2 + ((z - cz) * spacing[2]) ** 2 <= radius_mm**2).astype(np.float32)
    rows = []
    for angle in config.proton_angles_deg:
        single = DoseConfig(**{**asdict(config), "proton_angles_deg": (angle,)})
        plan = plan_case(ct, mask, target, spacing, "proton", single)
        dose = plan.reference_dose[plan.reference_dose.shape[0] // 2]
        direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        centre = np.array([(dose.shape[0] - 1) / 2, (dose.shape[1] - 1) / 2])
        distance = 0.3 * min(shape_yx)
        upstream, downstream = (tuple(np.rint(centre + sign * distance * direction).astype(int)) for sign in (-1, 1))
        recomputed, _, _ = recompute(plan, ct)
        rows.append({"angle_deg": angle, "gantry_deg": gantry_angle(angle),
                     "upstream_gy": float(dose[upstream]), "downstream_gy": float(dose[downstream]),
                     "direction_ok": bool(dose[upstream] > 5 * max(dose[downstream], 1e-3 * dose.max())),
                     "identity_gamma_2_2": 100.0 * gamma_pass_rate(plan.reference_dose, recomputed,
                                                                   plan.spacing_xyz[::-1]).pass_rate})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- statistics


def _ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank().to_numpy()


def partial_spearman(x: np.ndarray, y: np.ndarray, given: np.ndarray) -> float:
    """Spearman correlation of ``x`` and ``y`` after removing the linear effect of ``rank(given)`` from both ranks."""
    rx, ry, rz = _ranks(x), _ranks(y), _ranks(given)
    design = np.column_stack([np.ones_like(rz), rz])
    residual = lambda r: r - design @ np.linalg.lstsq(design, r, rcond=None)[0]  # noqa: E731
    ex, ey = residual(rx), residual(ry)
    denominator = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / denominator) if denominator > 0 else float("nan")


def cluster_bootstrap(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float], cluster: str = "case_id",
                      n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Point estimate and percentile interval of ``statistic``, resampling whole clusters (cases) with replacement."""
    estimate = statistic(frame)
    groups = [g for _, g in frame.groupby(cluster)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = pd.concat([groups[i] for i in rng.integers(len(groups), size=len(groups))], ignore_index=True)
        value = statistic(sample)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return estimate, float("nan"), float("nan")
    low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    return estimate, float(low), float(high)


def proxy_validation(cases: pd.DataFrame, rays: pd.DataFrame, prereg: Mapping, n_boot: int = 2000,
                     seed: int = 0, max_slope_rays: int = 3000) -> dict[str, object]:
    """Does the range proxy track proton dose? Case level (with and without MAE held fixed) and ray level.

    The ray Spearman uses every ray; the Theil-Sen slope a fixed random subset of ``max_slope_rays`` rays."""
    rule = prereg["proxy"]
    x, y, z = rule["case_metric"], rule["dose_metric"], rule["partial_given"]
    data = cases[["case_id", x, y, z]].dropna()
    result: dict[str, object] = {"pairs": int(len(data)), "cases": int(data["case_id"].nunique())}
    if len(data) >= 4:
        spear = lambda f: float(spearmanr(f[x], f[y]).statistic)  # noqa: E731
        partial = lambda f: partial_spearman(f[x].to_numpy(), f[y].to_numpy(), f[z].to_numpy())  # noqa: E731
        mae_only = lambda f: float(spearmanr(f[z], f[y]).statistic)  # noqa: E731
        for name, function in (("case_spearman", spear), ("case_partial_spearman", partial),
                               ("mae_spearman", mae_only)):
            estimate, low, high = cluster_bootstrap(data, function, n_boot=n_boot, seed=seed)
            result.update({name: estimate, f"{name}_low": low, f"{name}_high": high})
        # secondary, not part of the verdict: the bulk-density baseline is far from every arm and can carry the
        # pooled correlation on its own, so the same statistic is reported on the trained arms only
        arms_only = cases.loc[cases["source"] != "bulk density", ["case_id", x, y]].dropna() if "source" in cases \
            else pd.DataFrame()
        if len(arms_only) >= 4:
            estimate, low, high = cluster_bootstrap(arms_only, spear, n_boot=n_boot, seed=seed)
            result.update({"arms_only_case_spearman": estimate, "arms_only_case_spearman_low": low,
                           "arms_only_case_spearman_high": high})
    if not rays.empty and len(rays) >= 4:
        ray_rho = float(spearmanr(rays["delta_wet_mm"], -rays["delta_r80_mm"]).statistic)
        sample = rays.sample(n=min(len(rays), max_slope_rays), random_state=seed)  # Theil-Sen is O(n^2) in memory
        slope, intercept, low, high = theilslopes(sample["delta_r80_mm"], sample["delta_wet_mm"])
        result.update({"rays": int(len(rays)), "ray_spearman": ray_rho, "ray_slope": float(slope),
                       "ray_slope_low": float(low), "ray_slope_high": float(high), "ray_intercept_mm": float(intercept)})
    case_ok = (result.get("case_spearman", np.nan) >= rule["min_spearman"]
               and result.get("case_spearman_low", np.nan) > 0 and result.get("case_partial_spearman_low", np.nan) > 0)
    ray_ok = result.get("ray_spearman", np.nan) >= rule["ray_min_spearman"]
    if "case_spearman" not in result:
        verdict = "NOT EVALUATED"
    elif case_ok and ray_ok:
        verdict = "VALIDATED"
    elif case_ok or ray_ok:
        verdict = "PARTIAL"
    else:
        verdict = "NOT VALIDATED"
    result["verdict"] = verdict
    return result


def decide_go_no_go_2(cases: pd.DataFrame, s4_report: Mapping | None, prereg: Mapping) -> dict[str, object]:
    """Go/no-go 2: does a range-aware arm lower the range tail (S4) and improve proton gamma (S6), beyond R4?"""
    metric, reference, control = prereg["primary_metric"], prereg["reference_arm"], prereg["control_arm"]
    alpha, arms = float(prereg["alpha"]), prereg["method_arms"]
    raw = {}
    for arm in arms:
        for comparator in (reference, control):
            raw[f"{arm} > {comparator}"] = paired_test(cases, metric, arm, comparator, alternative="greater")["p_value"]
    adjusted = holm(raw)
    details = {arm: {"gamma_p_vs_reference_holm": adjusted[f"{arm} > {reference}"],
                     "gamma_p_vs_control_holm": adjusted[f"{arm} > {control}"]} for arm in arms}
    if s4_report is None or "decision" not in s4_report or s4_report["decision"].get("verdict") == "NOT EVALUATED":
        return {"verdict": "NOT EVALUATED", "reason": "no S4 verdict on real data", "arms": details,
                "raw_p_values": raw}
    s4_arms = s4_report["decision"]["arms"]
    for arm in arms:
        s4 = s4_arms.get(arm, {})
        range_gain = s4.get("p_vs_reference_holm", 1.0) < alpha and bool(s4.get("mae_within_tolerance", False))
        details[arm].update({"range_gain": range_gain,
                             "range_beats_control": range_gain and s4.get("p_vs_control_holm", 1.0) < alpha,
                             "dose_gain": details[arm]["gamma_p_vs_reference_holm"] < alpha})
    if not any(np.isfinite(p) for p in raw.values()):
        verdict = "NOT EVALUATED"
    elif any(d["range_beats_control"] and d["dose_gain"] for d in details.values()):
        verdict = "GO"
    elif any(d["range_beats_control"] for d in details.values()):
        verdict = "RANGE GAIN WITHOUT DOSE CONFIRMATION"
    elif any(d["range_gain"] for d in details.values()):
        verdict = "RIGHT ESTIMATOR SUFFICES"
    else:
        verdict = "NO-GO"
    return {"verdict": verdict, "s4_verdict": s4_report["decision"]["verdict"], "metric": metric, "arms": details,
            "raw_p_values": raw}


# ------------------------------------------------------------------------------------------------------- CLI


def source_paths(record: CaseRecord, runs_root: Path, arms: Sequence[str], seeds: Sequence[int]) -> dict[str, Path]:
    """Existing out-of-fold predictions ``<runs_root>/seed<k>/<arm>/predictions/<case>.mha``, keyed ``"<arm>@<seed>"``."""
    paths = {}
    for seed in seeds:
        for arm in arms:
            path = Path(runs_root) / f"seed{seed}" / arm / "predictions" / f"{record.case_id}.mha"
            if path.exists():
                paths[f"{arm}@{seed}"] = path
    return paths


def _signature(config: DoseConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True, default=list).encode()).hexdigest()[:16]


def _case_done(record: CaseRecord, output_dir: Path, expected: set[str], signature: str) -> bool:
    path = output_dir / "cases" / f"{record.case_id}.csv"
    if not path.exists():
        return False
    done = pd.read_csv(path)
    return set(done["source"]) == expected and bool((done["config"] == signature).all())


def dose_record(record: CaseRecord, runs_root: Path, output_dir: Path, config: DoseConfig, arms: Sequence[str],
                seeds: Sequence[int], include_bulk: bool = True) -> str:
    """Dose one case and write ``cases/<id>.csv``, ``rays/<id>.csv`` and ``slices/<id>.npz``. Top-level and
    picklable so that cases can run in parallel processes."""
    from .data import bulk_density_sct

    volumes = load_case(record, ("ct", "mask"))
    ct, mask, spacing = volumes["ct"].array, volumes["mask"].array, volumes["ct"].spacing
    sources = {name: read_prediction(path, ct.shape)
               for name, path in source_paths(record, runs_root, arms, seeds).items()}
    if include_bulk:
        sources = {"bulk density": bulk_density_sct(mask), **sources}
    result = dose_case(ct, mask, spacing, sources, HLUT.pyradplan_default(), config)
    keys = dict(case_id=record.case_id, region=record.region, center=record.center)
    result.rays.assign(**keys).to_csv(output_dir / "rays" / f"{record.case_id}.csv", index=False)
    np.savez_compressed(output_dir / "slices" / f"{record.case_id}.npz", **result.slices)
    pd.DataFrame(result.rows).assign(**keys, config=_signature(config)).to_csv(
        output_dir / "cases" / f"{record.case_id}.csv", index=False)  # written last: marks the case as done
    return record.case_id


def run_dosimetry(records: Sequence[CaseRecord], runs_root: str | Path, output_dir: str | Path, config: DoseConfig,
                  arms: Sequence[str], seeds: Sequence[int], include_bulk: bool = True, workers: int = 1,
                  log: Callable[[str], None] = print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dose every record that is not already done with the same sources and configuration; return all results.

    With ``workers > 1`` cases run in parallel processes (pyRadPlan is single-threaded for most of the work).
    """
    output_dir, runs_root = Path(output_dir), Path(runs_root)
    for sub in ("cases", "rays", "slices"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    signature = _signature(config)
    pending = []
    for record in records:
        expected = set(source_paths(record, runs_root, arms, seeds)) | ({"bulk density"} if include_bulk else set())
        if not _case_done(record, output_dir, expected, signature):
            pending.append(record)
    log(f"{len(records) - len(pending)} cases reused, {len(pending)} to dose")
    job = (runs_root, output_dir, config, tuple(arms), tuple(seeds), include_bulk)
    if workers > 1 and len(pending) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(dose_record, record, *job) for record in pending]
            for future in as_completed(futures):
                log(f"dosed {future.result()}")
    else:
        for record in pending:
            log(f"dosed {dose_record(record, *job)}")
    cases = pd.concat([pd.read_csv(output_dir / "cases" / f"{r.case_id}.csv") for r in records], ignore_index=True)
    ray_frames = []
    for record in records:
        try:
            ray_frames.append(pd.read_csv(output_dir / "rays" / f"{record.case_id}.csv"))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
    ray_frames = [frame for frame in ray_frames if not frame.empty]
    rays = pd.concat(ray_frames, ignore_index=True) if ray_frames else pd.DataFrame()
    return cases, rays


def split_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Split ``"<arm>@<seed>"`` into ``arm`` and ``seed`` columns (``seed`` NaN for seedless sources)."""
    parts = frame["source"].str.split("@", n=1, expand=True)
    return frame.assign(arm=parts[0], seed=pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan)


def average_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """Per case and arm, the mean of every numeric metric over seeds (``source`` becomes the arm)."""
    frame = split_source(frame)
    numeric = [c for c in frame.select_dtypes("number").columns if c != "seed"]
    keys = [k for k in ("case_id", "region", "center", "arm") if k in frame]
    return frame.groupby(keys, as_index=False)[numeric].mean().rename(columns={"arm": "source"})


def main(argv: list[str] | None = None) -> None:
    from .split import load_split

    parser = argparse.ArgumentParser(description="S6 dosimetry: plans on the reference CT, recomputed on sCTs.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path, help="S4 runs root containing seed<k>/<arm>/")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--preregistration", type=Path, default=Path("configs/s6_preregistration.json"))
    parser.add_argument("--task", choices=("1", "2"), default=None)
    parser.add_argument("--workers", type=int, default=1, help="cases dosed in parallel processes")
    args = parser.parse_args(argv)

    prereg = json.loads(args.preregistration.read_text())
    config = DoseConfig.from_preregistration(prereg)
    selection_rule = prereg["cases"]
    task = args.task or selection_rule["task"]
    split = load_split(args.split)
    development = set(split["development"])
    records = [r for r in discover_cases(args.data_root) if r.case_id in development and r.task == task]
    records, excluded = usable_records(args.data_root, records)
    for case_id, problems in excluded.items():
        print(f"excluded {case_id}: {problems}")
    reference_dir = lambda r: args.runs_root / "seed0" / prereg["reference_arm"] / "predictions" / f"{r.case_id}.mha"  # noqa: E731
    selection_path = args.out / "dosimetry_cases.json"
    ids = select_cases(records, selection_rule["n_cases"], selection_rule["seed"], lambda r: reference_dir(r).exists())
    write_selection(ids, selection_path, {"task": task, "seed": selection_rule["seed"], "rule": "region-stratified"})
    chosen = [r for r in records if r.case_id in set(load_selection(selection_path)["case_ids"])]
    cases, rays = run_dosimetry(chosen, args.runs_root, args.out, config, prereg["arms"], prereg["seeds"],
                                workers=args.workers)
    cases.to_csv(args.out / "s6_dose_per_seed.csv", index=False)
    rays.to_csv(args.out / "s6_rays.csv", index=False)
    print(f"{cases['case_id'].nunique()} cases dosed; results in {args.out}")


if __name__ == "__main__":
    main()
