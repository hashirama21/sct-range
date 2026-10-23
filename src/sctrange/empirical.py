"""Statements of docs/theory.md that hold for every error field, measured on real SynthRAD2025 cases.

The synthetic benchmark checks them where the posterior is known. Three of them are identities, true for any error
field, so they can be checked exactly on the SPR error of real synthetic CTs, and the quantities they involve can be
measured on patients:

- Proposition 3 (Parseval): the energy of segment sums equals the Dirichlet-weighted error spectrum;
- Proposition 2: with the measured mean, variance and autocorrelation along the beam, ``L²μ² + Lσ²l`` is the mean
  squared segment error, so the effective correlation length ``l`` of real errors says how coherent they are;
- Propositions 1 and 1′: the range error of every ray lies between the sum of its ``n`` smallest and ``n`` largest
  voxel errors of the slice (the sorted rearrangement bounds); where it sits between them measures how close the real
  layout is to the worst case.

Beams run along the two in-plane axes (0° along rows, 90° along columns) with circular segments, so that the
identities are exact; ray geometry at other angles is covered by the projector checks of S1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .data import bounding_box, central_slab
from .hlut import HLUT
from .spectral import effective_length, segment_energy_spectral, sliding_segment_sums

BEAM_AXES = {"0°": 1, "90°": 2}


def spr_error_slab(ct: np.ndarray, mask: np.ndarray, sct: np.ndarray, hlut: HLUT,
                   n_slices: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """SPR error inside the body (float64, zero outside) on ``n_slices`` slices around the largest body section,
    cropped to the body; returns ``(error, body)`` of shape ``(Z, Y, X)``."""
    slab = central_slab(mask, n_slices)
    box = (slab, *bounding_box(mask[slab])[1:])
    body = mask[box] > 0.5
    error = (hlut.numpy(sct[box].astype(np.float64)) - hlut.numpy(ct[box].astype(np.float64))) * body
    return error, body


def segment_decomposition(error: np.ndarray, length: int, axis: int, step: float = 1.0) -> dict[str, float]:
    """Mean squared circular segment sum of ``length`` samples along ``axis``, measured and predicted twice:
    by Parseval (Proposition 3) and by ``L²μ² + Lσ²l`` with the measured moments (Proposition 2)."""
    sums = sliding_segment_sums(error, length, axis=axis, step=step)
    energy = float(np.sum(sums**2))
    moments = np.array([np.mean(error * np.roll(error, -lag, axis=axis)) for lag in range(length)])
    mean = float(error.mean())
    variance = float(moments[0] - mean**2)
    correlation = (moments - mean**2) / variance if variance > 0 else np.zeros(length)
    path = length * step
    l_eff = effective_length(correlation, length, step)
    predicted = path**2 * mean**2 + path * variance * l_eff
    observed = energy / sums.size
    return {
        "mean_spr": mean,
        "std_spr": float(np.sqrt(max(variance, 0.0))),
        "effective_length_mm": l_eff,
        "rms_mm": float(np.sqrt(observed)),
        "white_rms_mm": float(np.sqrt(path**2 * mean**2 + path * variance * step)),
        "parseval_gap": abs(segment_energy_spectral(error, length, axis=axis, step=step) - energy) / energy
        if energy > 0 else 0.0,
        "decomposition_gap": abs(predicted - observed) / observed if observed > 0 else 0.0,
    }


def ray_bounds(error: np.ndarray, body: np.ndarray, axis: int, step: float = 1.0) -> dict[str, float]:
    """Position of every ray's range error between its sorted rearrangement bounds (Propositions 1 and 1′).

    A ray is a line of the 2-D slice along ``axis`` (0 or 1 for a ``(Y, X)`` slice); its ``n`` body voxels are
    compared with all body voxels of the slice. ``position`` is 0 at the lower bound and 1 at the upper bound.
    """
    values = np.sort(error[body])
    lower = np.concatenate([[0.0], np.cumsum(values)])
    upper = np.concatenate([[0.0], np.cumsum(values[::-1])])
    n = body.sum(axis=axis)
    keep = n > 0
    delta = (error * body).sum(axis=axis)[keep]
    low, high = lower[n[keep]], upper[n[keep]]
    tolerance = 1e-9 * (np.abs(low) + np.abs(high) + 1.0)
    width = high - low
    position = np.where(width > 0, (delta - low) / np.where(width > 0, width, 1.0), 0.5)
    return {
        "rays": int(keep.sum()),
        "violations": int(((delta < low - tolerance) | (delta > high + tolerance)).sum()),
        "median_position": float(np.median(position)) if keep.any() else float("nan"),
        "max_position": float(position.max()) if keep.any() else float("nan"),
        "worst_case_mm": float(step * np.max(np.maximum(np.abs(low), np.abs(high)))) if keep.any() else float("nan"),
        "max_abs_mm": float(step * np.abs(delta).max()) if keep.any() else float("nan"),
    }


def case_identities(ct: np.ndarray, mask: np.ndarray, sct: np.ndarray, spacing_mm: float, hlut: HLUT,
                    segments_mm: Sequence[float] = (10.0, 60.0), n_slices: int = 16) -> list[dict[str, float]]:
    """One row per beam axis and segment length (Propositions 2, 3) plus one row per beam axis (Propositions 1, 1′)."""
    error, body = spr_error_slab(ct, mask, sct, hlut, n_slices)
    rows = []
    for beam, axis in BEAM_AXES.items():
        for segment in segments_mm:
            length = int(round(segment / spacing_mm))
            if 1 <= length <= error.shape[axis]:
                rows.append({"check": "segments", "beam": beam, "segment_mm": length * spacing_mm,
                             **segment_decomposition(error, length, axis, spacing_mm)})
        largest = int((body.sum(axis=(1, 2))).argmax())
        rows.append({"check": "rays", "beam": beam,
                     **ray_bounds(error[largest], body[largest], axis - 1, spacing_mm)})
    return rows


def summarize_identities(frame: pd.DataFrame) -> dict[str, float]:
    """Worst gaps over all cases and sources, the evidence for the pass/fail checks."""
    segments, rays = frame[frame["check"] == "segments"], frame[frame["check"] == "rays"]
    return {
        "cases": int(frame["case_id"].nunique()),
        "max_parseval_gap": float(segments["parseval_gap"].max()),
        "max_decomposition_gap": float(segments["decomposition_gap"].max()),
        "rays": int(rays["rays"].sum()),
        "violations": int(rays["violations"].sum()),
    }
