"""Global 3-D gamma index (Low et al., 1998) for dose comparisons on anisotropic grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GammaResult:
    pass_rate: float
    n_points: int
    dose_percent: float
    dta_mm: float


def search_offsets_mm(dta_mm: float, step_mm: float) -> np.ndarray:
    """Offsets ``(N, 3)`` in mm, ``(z, y, x)`` order, on a cubic lattice inside the ball of radius ``dta_mm``,
    sorted by distance so that the zero offset comes first."""
    n = int(np.floor(dta_mm / step_mm + 1e-9))
    axis = np.arange(-n, n + 1) * step_mm
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    distance = np.linalg.norm(grid, axis=1)
    keep = distance <= dta_mm + 1e-9
    return grid[keep][np.argsort(distance[keep], kind="stable")]


def gamma_pass_rate(reference: np.ndarray, evaluated: np.ndarray, spacing_zyx: tuple[float, float, float],
                    dose_percent: float = 2.0, dta_mm: float = 2.0, cutoff_percent: float = 10.0,
                    normalization: float | None = None, step_fraction: float = 0.1,
                    offset_batch: int = 64, max_samples: int = 4_000_000) -> GammaResult:
    """Share of reference points above the cutoff with gamma <= 1.

    Global normalisation (``dose_percent`` of ``normalization``, by default the reference maximum). The evaluated dose
    is interpolated trilinearly at offsets on a lattice of pitch ``step_fraction * dta_mm`` inside the ball of radius
    ``dta_mm``. Restricting the search to that ball is exact for the pass rate: any offset beyond it already gives
    gamma > 1. Offsets are scanned by increasing distance, at most ``offset_batch``
    at a time and ``max_samples`` interpolations per batch; points are dropped from the
    search as soon as one offset passes them.
    """
    if reference.shape != evaluated.shape:
        raise ValueError("reference and evaluated doses must share a grid")
    norm = float(reference.max()) if normalization is None else float(normalization)
    if norm <= 0:
        return GammaResult(float("nan"), 0, dose_percent, dta_mm)
    dose_tolerance = dose_percent / 100.0 * norm
    points = np.argwhere(reference >= cutoff_percent / 100.0 * norm)
    if points.size == 0:
        return GammaResult(float("nan"), 0, dose_percent, dta_mm)
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    shape = np.asarray(reference.shape, dtype=np.float64)
    volume = torch.as_tensor(np.ascontiguousarray(evaluated), dtype=torch.float32)[None, None]
    points_t = torch.as_tensor(points, dtype=torch.float32)
    reference_values = torch.as_tensor(reference[tuple(points.T)], dtype=torch.float32)
    open_points = torch.ones(len(points), dtype=torch.bool)
    offsets = search_offsets_mm(dta_mm, step_fraction * dta_mm)
    scale = torch.as_tensor(2.0 / np.maximum(shape - 1.0, 1.0), dtype=torch.float32)
    start = 0
    while start < len(offsets):
        index = torch.nonzero(open_points).flatten()
        if index.numel() == 0:
            break
        size = int(np.clip(max_samples // index.numel(), 1, offset_batch))
        batch = offsets[start:start + size]
        start += size
        shift = torch.as_tensor(batch / spacing, dtype=torch.float32)
        coordinates = points_t[index][None].float() + shift[:, None].float()  # (B, N, 3) in (z, y, x) voxel units
        grid = (coordinates * scale - 1.0).flip(-1)[None, :, :, None]  # grid_sample wants (x, y, z)
        values = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, :, :, 0]
        distance = torch.as_tensor((batch**2).sum(axis=1) / dta_mm**2, dtype=torch.float32)[:, None]
        gamma_sq = distance + ((values - reference_values[index][None]) / dose_tolerance) ** 2
        open_points[index[(gamma_sq <= 1.0).any(dim=0)]] = False
    return GammaResult(float(1.0 - open_points.double().mean()), int(len(points)), dose_percent, dta_mm)
