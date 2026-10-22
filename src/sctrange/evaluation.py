"""Per-case range diagnostics: chord and target-segment WET errors and permutation controls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from .data import bounding_box, spherical_target
from .hlut import HLUT
from .losses import fz_upper_tail_loss, ray_risk
from .metrics import GLOBAL_EDGES_HU, TISSUE_EDGES_HU, mae_hu, permute_errors, range_error, spr_error, summarize
from .projection import ParallelBeamProjector


@dataclass(frozen=True)
class DiagnosticConfig:
    angles_deg: Sequence[float] = field(default_factory=lambda: tuple(np.arange(0.0, 360.0, 45.0)))
    target_radius_mm: float = 25.0
    alpha: float = 0.9
    min_path_mm: float = 20.0
    seed: int = 0
    device: str = "cpu"
    slab_slices: int = 16


@dataclass
class PreparedCase:
    ct: torch.Tensor
    sct: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    projector: ParallelBeamProjector


def prepare_case(ct: np.ndarray, mask: np.ndarray, sct: np.ndarray, spacing_xyz: tuple[float, float, float],
                 config: DiagnosticConfig, target: np.ndarray | None = None) -> PreparedCase:
    """Crop to the body bounding box, move to the device and build the projector."""
    if target is None:
        target = spherical_target(mask, spacing_xyz, config.target_radius_mm)
    box = bounding_box(mask)
    device = torch.device(config.device)

    def as_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(array[box]), dtype=torch.float32, device=device)

    ct_t, sct_t, mask_t, target_t = map(as_tensor, (ct, sct, mask, target))
    projector = ParallelBeamProjector(tuple(ct_t.shape[-2:]), config.angles_deg, spacing_xyz[0], device=device)
    return PreparedCase(ct_t, sct_t, mask_t, target_t, projector)


def evaluate_prepared(case: PreparedCase, hlut: HLUT, config: DiagnosticConfig) -> dict[str, float]:
    """Image and range metrics for one prepared case.

    Permutation controls keep MAE identical and change only the spatial layout of errors;
    ``*_ratio`` > 1 means the real layout accumulates more range error than a shuffled one.
    """
    generator = torch.Generator(device=case.ct.device).manual_seed(config.seed)
    controls = {
        "intra": permute_errors(case.sct, case.ct, case.mask, TISSUE_EDGES_HU, generator),
        "global": permute_errors(case.sct, case.ct, case.mask, GLOBAL_EDGES_HU, generator),
    }

    def stats(sct: torch.Tensor, target: torch.Tensor | None):
        result = range_error(sct, case.ct, case.mask, case.projector, hlut, target, config.slab_slices)
        return summarize(result, config.alpha, config.min_path_mm)

    row: dict[str, float] = {"mae_hu": mae_hu(case.sct, case.ct, case.mask)}
    for mode, target_mask in (("chord", None), ("target", case.target)):
        real = stats(case.sct, target_mask)
        row.update(real.as_dict(f"{mode}_"))
        for name, permuted in controls.items():
            control = stats(permuted, target_mask)
            row[f"{mode}_{name}_ratio"] = real.rms_mm / control.rms_mm if control.rms_mm > 0 else float("nan")
    return row


def evaluate_risk(
    ct: np.ndarray,
    mask: np.ndarray,
    sct: np.ndarray,
    var_density: np.ndarray,
    excess_density: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    hlut: HLUT,
    config: DiagnosticConfig = DiagnosticConfig(),
    level: float = 0.9,
    target: np.ndarray | None = None,
) -> dict[str, float]:
    """Calibration of a per-ray (VaR, ES) head (arm R6) on entry-to-distal-target segments.

    ``risk_coverage`` should be close to ``level``; ``risk_realised_es_mm`` (mean |ΔWET| beyond the predicted VaR)
    should be close to ``risk_mean_es_mm``; ``risk_fz`` is the FZ0 score, lower is better and comparable across
    models on the same cases.
    """
    case = prepare_case(ct, mask, sct, spacing_xyz, config, target)
    box = bounding_box(mask)
    device = case.ct.device
    densities = [torch.as_tensor(np.ascontiguousarray(d[box]), dtype=torch.float32, device=device)
                 for d in (var_density, excess_density)]
    error = spr_error(case.sct, case.ct, case.mask, hlut)
    step = config.slab_slices or error.shape[0]
    magnitude, var, es = [], [], []
    for start in range(0, error.shape[0], step):
        rows = slice(start, start + step)
        weights, valid = case.projector.segment_weights(case.mask[rows], case.target[rows])
        valid &= case.projector.path_length(weights) >= config.min_path_mm
        ray_var, ray_es = ray_risk(case.projector, densities[0][rows], densities[1][rows], weights)
        magnitude.append(case.projector.integrate(error[rows], weights).abs()[valid])
        var.append(ray_var[valid])
        es.append(ray_es[valid])
    z, v, e = (torch.cat(parts).double() for parts in (magnitude, var, es))
    if z.numel() == 0:
        return {"risk_rays": 0}
    exceed = z >= v
    rank = lambda t: t.argsort().argsort().double()  # noqa: E731
    return {
        "risk_rays": int(z.numel()),
        "risk_coverage": float((z < v).double().mean()),
        "risk_mean_var_mm": float(v.mean()),
        "risk_mean_es_mm": float(e[exceed].mean()) if exceed.any() else float("nan"),
        "risk_realised_es_mm": float(z[exceed].mean()) if exceed.any() else float("nan"),
        "risk_fz": float(fz_upper_tail_loss(v, e, z, level)),
        "risk_spearman": float(torch.corrcoef(torch.stack([rank(v), rank(z)]))[0, 1]),
    }


def evaluate_case(
    ct: np.ndarray,
    mask: np.ndarray,
    sct: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    hlut: HLUT,
    config: DiagnosticConfig = DiagnosticConfig(),
    target: np.ndarray | None = None,
) -> dict[str, float]:
    """``evaluate_prepared`` on a case given as arrays in its original grid."""
    return evaluate_prepared(prepare_case(ct, mask, sct, spacing_xyz, config, target), hlut, config)
