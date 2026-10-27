"""S3 diagnostics on real sCTs: beam-aligned error spectra and the pre-registered go/no-go rule."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from .evaluation import DiagnosticConfig, PreparedCase, evaluate_prepared, prepare_case
from .hlut import HLUT
from .metrics import GLOBAL_EDGES_HU, permute_errors, spr_error
from .projection import ParallelBeamProjector


@dataclass(frozen=True)
class Preregistration:
    """Decision rule fixed before any real prediction is inspected; see ``configs/s3_preregistration.json``."""

    raw: dict
    fingerprint: str

    @classmethod
    def load(cls, path: str | Path) -> Preregistration:
        data = Path(path).read_bytes()
        return cls(json.loads(data), hashlib.sha256(data).hexdigest())

    @property
    def window_mm(self) -> float:
        return float(self.raw["window_mm"])

    @property
    def cutoff(self) -> float:
        return float(self.raw["cutoff_cycles_per_mm"])

    @property
    def primary_source(self) -> str:
        return self.raw["primary_source"]

    @property
    def spectrum_angles(self) -> tuple[float, ...]:
        """Beam angles for spectra. A set closed under +90 degrees would make along and across identical."""
        angles = tuple(float(a) for a in self.raw["spectrum_angles_deg"])
        along = {round(a % 180.0, 6) for a in angles}
        if along & {round((a + 90.0) % 180.0, 6) for a in angles}:
            raise ValueError("spectrum angles must not contain two directions 90 degrees apart")
        return angles


@dataclass
class BeamSpectra:
    """Mean tapered periodograms of the SPR error along and across beams, real and shuffled.

    Windows of ``len(frequency)`` complex bins come from beam's-eye canvases; only windows lying entirely
    inside the body are used. The shuffled control permutes the same errors over the whole body.
    """

    frequency: np.ndarray
    along_real: np.ndarray
    along_control: np.ndarray
    across_real: np.ndarray
    across_control: np.ndarray
    n_along: int
    n_across: int

    @staticmethod
    def lf_fraction(power: np.ndarray, frequency: np.ndarray, cutoff: float) -> float:
        total = power.sum()
        return float(power[frequency < cutoff].sum() / total) if total > 0 else float("nan")

    def lf_excess(self, direction: str, cutoff: float) -> float:
        real, control = (self.along_real, self.along_control) if direction == "along" else (
            self.across_real, self.across_control)
        return self.lf_fraction(real, self.frequency, cutoff) / self.lf_fraction(control, self.frequency, cutoff)


def _windows(canvas: torch.Tensor, inside: torch.Tensor, window: int, along: bool) -> torch.Tensor:
    """Half-overlapping windows ``(N, window)`` fully inside the body, along rows (beam) or columns."""
    if along:
        canvas, inside = canvas.transpose(-1, -2), inside.transpose(-1, -2)
    segments = canvas.unfold(-1, window, window // 2)
    keep = inside.unfold(-1, window, window // 2).all(dim=-1)
    return segments[keep]


def beam_spectra(case: PreparedCase, hlut: HLUT, window_px: int, generator: torch.Generator,
                 slab: int = 8) -> BeamSpectra:
    error = spr_error(case.sct, case.ct, case.mask, hlut).double()
    shuffled = permute_errors(case.sct, case.ct, case.mask, GLOBAL_EDGES_HU, generator)
    control = spr_error(shuffled, case.ct, case.mask, hlut).double()
    taper = torch.hann_window(window_px, periodic=False, dtype=torch.float64, device=error.device)
    norm = taper.square().sum()
    sums = {key: torch.zeros(window_px // 2 + 1, dtype=torch.float64, device=error.device)
            for key in ("along_real", "along_control", "across_real", "across_control")}
    counts = {"along": 0, "across": 0}

    for start in range(0, error.shape[0], slab):
        rows = slice(start, start + slab)
        inside = case.projector.rotate(case.mask[rows].double()) > 0.999
        canvases = {"real": case.projector.rotate(error[rows]), "control": case.projector.rotate(control[rows])}
        for direction, along in (("along", True), ("across", False)):
            for name, canvas in canvases.items():
                segments = _windows(canvas, inside, window_px, along)
                if name == "real":
                    counts[direction] += segments.shape[0]
                if segments.shape[0]:
                    power = torch.fft.rfft(segments * taper, dim=-1).abs().square() / norm
                    sums[f"{direction}_{name}"] += power.sum(dim=0)

    frequency = np.fft.rfftfreq(window_px, d=case.projector.spacing_mm)
    mean = {key: (value / max(counts[key.split("_")[0]], 1)).cpu().numpy() for key, value in sums.items()}
    return BeamSpectra(frequency, mean["along_real"], mean["along_control"], mean["across_real"],
                       mean["across_control"], counts["along"], counts["across"])


def diagnose_case(ct: np.ndarray, mask: np.ndarray, sct: np.ndarray, spacing_xyz: tuple[float, float, float],
                  hlut: HLUT, config: DiagnosticConfig, prereg: Preregistration,
                  target: np.ndarray | None = None) -> tuple[dict[str, float], BeamSpectra]:
    """Range metrics with permutation controls, plus beam spectra and their low-frequency excess."""
    case = prepare_case(ct, mask, sct, spacing_xyz, config, target)
    row = evaluate_prepared(case, hlut, config)
    window_px = int(round(prereg.window_mm / spacing_xyz[0]))
    generator = torch.Generator(device=case.ct.device).manual_seed(config.seed + 1)
    spectral_projector = ParallelBeamProjector(case.projector.shape_yx, prereg.spectrum_angles,
                                               case.projector.spacing_mm, device=case.ct.device)
    spectra = beam_spectra(replace(case, projector=spectral_projector), hlut, window_px, generator)
    row.update({
        "lf_excess_along": spectra.lf_excess("along", prereg.cutoff),
        "lf_excess_across": spectra.lf_excess("across", prereg.cutoff),
        "spectral_windows_along": spectra.n_along,
    })
    return row, spectra


def decide(frame: pd.DataFrame, prereg: Preregistration) -> dict[str, object]:
    """Apply the pre-registered rule to the primary source."""
    go, weak = prereg.raw["go"], prereg.raw["weak"]
    primary = frame[frame["source"] == prereg.primary_source]
    if primary.empty:
        return {"verdict": "NOT EVALUATED", "reason": f"no predictions for {prereg.primary_source!r}"}
    lf = float(primary["lf_excess_along"].median())
    ratio = float(primary["target_global_ratio"].median())
    if lf >= go["median_lf_excess_along_min"] and ratio >= go["median_target_global_ratio_min"]:
        verdict = "GO"
    elif lf >= weak["median_lf_excess_along_min"]:
        verdict = "WEAK"
    else:
        verdict = "NO-GO"
    return {"verdict": verdict, "source": prereg.primary_source, "cases": int(len(primary)),
            "median_lf_excess_along": lf, "median_lf_excess_across": float(primary["lf_excess_across"].median()),
            "median_target_global_ratio": ratio, "preregistration_sha256": prereg.fingerprint}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (NaN entries are left out of the family and returned as NaN)."""
    finite = sorted((p, name) for name, p in p_values.items() if np.isfinite(p))
    adjusted, running = {name: float("nan") for name in p_values}, 0.0
    for rank, (p, name) in enumerate(finite):
        running = max(running, min(1.0, (len(finite) - rank) * p))
        adjusted[name] = running
    return adjusted


def decide_arms(frame: pd.DataFrame, prereg: Preregistration) -> dict[str, object]:
    """Pre-registered S4-S5 rule (``configs/s4_preregistration.json``) on per-case metrics, one row per case and arm.

    Each method arm is compared with the reference (R3) and with the bias-correction control (R4) by one-sided paired
    Wilcoxon tests on the primary metric, Holm-adjusted over the whole family. The MAE condition is one-sided: the median
    relative MAE change against the reference must not exceed the tolerance, so an arm that also improves MAE qualifies
    (version 1 of the rule used ``|change| <= tolerance``, which rejected such an arm; amended before any real result).
    """
    rule = prereg.raw
    metric, reference, control = rule["primary_metric"], rule["reference_arm"], rule["control_arm"]
    tolerance, alpha = float(rule["mae_relative_tolerance"]), float(rule["alpha"])
    raw, details = {}, {}
    for arm in rule["method_arms"]:
        for comparator in (reference, control):
            key = f"{arm} < {comparator}"
            raw[key] = paired_test(frame, metric, arm, comparator, alternative="less")["p_value"]
    adjusted = holm(raw)
    wide_mae = frame.pivot_table(index="case_id", columns="source", values="mae_hu")
    for arm in rule["method_arms"]:
        both = wide_mae[[arm, reference]].dropna() if {arm, reference} <= set(wide_mae) else pd.DataFrame()
        relative = float(np.median(both[arm] / both[reference] - 1.0)) if len(both) else float("nan")
        details[arm] = {
            "p_vs_reference_holm": adjusted[f"{arm} < {reference}"],
            "p_vs_control_holm": adjusted[f"{arm} < {control}"],
            "median_relative_mae_change": relative,
            "mae_within_tolerance": bool(relative <= tolerance) if np.isfinite(relative) else False,
        }
    beats_reference = [a for a, d in details.items() if d["p_vs_reference_holm"] < alpha and d["mae_within_tolerance"]]
    beats_control = [a for a in beats_reference if details[a]["p_vs_control_holm"] < alpha]
    if not any(np.isfinite(p) for p in raw.values()):
        verdict = "NOT EVALUATED"
    elif beats_control:
        verdict = "GAIN"
    elif beats_reference:
        verdict = "BIAS CORRECTION SUFFICES"
    else:
        verdict = "NO GAIN"
    return {"verdict": verdict, "metric": metric, "arms": details, "raw_p_values": raw,
            "preregistration_sha256": prereg.fingerprint}


def paired_test(frame: pd.DataFrame, column: str, first: str, second: str, region: str | None = None,
                alternative: str = "two-sided") -> dict[str, float]:
    """Wilcoxon signed-rank test of ``column`` for ``first - second`` over cases evaluated with both sources."""
    subset = frame if region is None else frame[frame["region"] == region]
    wide = subset.pivot_table(index="case_id", columns="source", values=column)
    if first not in wide or second not in wide:
        return {"cases": 0, "median_difference": float("nan"), "p_value": float("nan")}
    wide = wide.dropna(subset=[first, second])
    difference = (wide[first] - wide[second]).to_numpy()
    if len(difference) < 2 or np.allclose(difference, 0):
        return {"cases": int(len(difference)), "median_difference": float(np.median(difference)) if len(difference) else float("nan"),
                "p_value": float("nan")}
    result = wilcoxon(difference, alternative=alternative)
    return {"cases": int(len(difference)), "median_difference": float(np.median(difference)),
            "p_value": float(result.pvalue)}
