"""Verification of the range projector against analytic geometry and pyRadPlan's Siddon tracer.

Analytic checks need an exact reference and run on disks; the Siddon comparison, the gradient check and the timing
also run on real patient anatomy (``*_case`` functions, ``ct``/``mask`` arguments).
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import torch

from .data import bounding_box, central_slab
from .hlut import HLUT
from .losses import range_mse
from .phantoms import chord_length, disk
from .projection import ParallelBeamProjector

_ALL = torch.ones(1, dtype=torch.bool)


def _centre(shape_yx: tuple[int, int]) -> tuple[float, float]:
    return (shape_yx[0] - 1) / 2, (shape_yx[1] - 1) / 2


def disk_chord_errors(angles_deg: Sequence[float], shape_yx: tuple[int, int] = (121, 141),
                      radius: float = 45.0, margin: float = 3.0) -> pd.DataFrame:
    """Integral of a unit disk versus its exact chord length, per beam angle (mm)."""
    centre = _centre(shape_yx)
    field = torch.from_numpy(disk(shape_yx, centre, radius)[None].astype(np.float64))
    projector = ParallelBeamProjector(shape_yx, angles_deg, spacing_mm=1.0)
    integrals = projector.integrate(field, _ALL)
    rows = []
    for index, angle in enumerate(projector.angles_deg):
        offsets = projector.ray_offsets(index, centre)
        interior = np.abs(offsets) < radius - margin
        error = np.abs(integrals[index, 0].numpy()[interior] - chord_length(radius, offsets[interior]))
        rows.append({"angle_deg": angle, "rays": int(interior.sum()),
                     "median_abs_mm": float(np.median(error)), "max_abs_mm": float(error.max())})
    return pd.DataFrame(rows)


def target_path_errors(angles_deg: Sequence[float], shape_yx: tuple[int, int] = (121, 141),
                       body_radius: float = 45.0, target_radius: float = 15.0,
                       margin: float = 2.0) -> pd.DataFrame:
    """Entry-to-distal-edge path length versus ``sqrt(R^2 - u^2) + sqrt(r^2 - u^2)`` (mm)."""
    centre = _centre(shape_yx)
    body = torch.from_numpy(disk(shape_yx, centre, body_radius)[None])
    target = torch.from_numpy(disk(shape_yx, centre, target_radius)[None])
    projector = ParallelBeamProjector(shape_yx, angles_deg, spacing_mm=1.0)
    weights, valid = projector.segment_weights(body, target)
    lengths = projector.path_length(weights)
    rows = []
    for index, angle in enumerate(projector.angles_deg):
        offsets = projector.ray_offsets(index, centre)
        interior = np.abs(offsets) < target_radius - margin
        expected = (chord_length(body_radius, offsets) + chord_length(target_radius, offsets)) / 2
        error = np.abs(lengths[index, 0].numpy()[interior] - expected[interior])
        rows.append({"angle_deg": angle, "rays": int(interior.sum()),
                     "median_abs_mm": float(np.median(error)), "max_abs_mm": float(error.max()),
                     "valid_inside": bool(valid[index, 0].numpy()[interior].all()),
                     "valid_outside": bool(valid[index, 0].numpy()[np.abs(offsets) > target_radius + margin].any())})
    return pd.DataFrame(rows)


def heterogeneous_phantom(shape_yx: tuple[int, int] = (101, 121)) -> tuple[np.ndarray, np.ndarray]:
    """Water disk with a lung insert and a bone insert; returns ``(hu, body)`` in 2-D."""
    centre = _centre(shape_yx)
    body = disk(shape_yx, centre, 45.0)
    hu = -1000.0 + 1040.0 * body
    hu -= 840.0 * disk(shape_yx, (centre[0] - 5, centre[1] - 18), 14.0)
    hu += 1060.0 * disk(shape_yx, (centre[0] + 25, centre[1] + 6), 8.0)
    return hu, (body > 0.5).astype(np.float64)


def _siddon_compare(spr: np.ndarray, angles_deg: Sequence[float], spacing_mm: float,
                    rays: Callable[[int, np.ndarray, np.ndarray], np.ndarray]) -> pd.DataFrame:
    """Projector integrals of a 2-D SPR map versus pyRadPlan's Siddon tracer, on the rays chosen by
    ``rays(angle_index, offsets_px, projector_mm)``."""
    import SimpleITK as sitk
    from pyRadPlan.raytracer import RayTracerSiddon

    shape_yx = spr.shape
    centre = _centre(shape_yx)
    volume = np.repeat(spr[None].astype(np.float64), 3, axis=0)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((spacing_mm,) * 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tracer = RayTracerSiddon([image])
    projector = ParallelBeamProjector(shape_yx, angles_deg, spacing_mm=spacing_mm)
    integrals = projector.integrate(torch.from_numpy(volume), _ALL)
    isocentre = spacing_mm * np.array([centre[1], centre[0], 1.0])

    rows = []
    for index, angle in enumerate(projector.angles_deg):
        geometry = projector.ray_geometry(index)
        offsets = projector.ray_offsets(index, centre)
        projected = integrals[index, 1].numpy()
        for column in rays(index, offsets, projected):
            start = geometry.origins_xy[column] - 10 * geometry.direction_xy
            stop = geometry.origins_xy[column] + (geometry.n_samples + 10) * geometry.direction_xy
            source = spacing_mm * np.array([*start, 1.0]) - isocentre
            target = spacing_mm * np.array([*stop, 1.0]) - isocentre
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                _, lengths, rho, _, _ = tracer.trace_ray(isocentre, source, target)
            reference = float(np.nansum(lengths * rho[0]))
            rows.append({"angle_deg": angle, "offset_px": float(offsets[column]), "siddon_mm": reference,
                         "projector_mm": float(projected[column]),
                         "relative_error": abs(float(projected[column]) - reference) / reference})
    return pd.DataFrame(rows)


def siddon_relative_errors(hlut: HLUT, angles_deg: Sequence[float] = (0.0, 30.0, 75.0, 140.0),
                           shape_yx: tuple[int, int] = (101, 121), stride: int = 4,
                           max_offset: float = 38.0) -> pd.DataFrame:
    """Relative WET difference to pyRadPlan's Siddon tracer on a voxelised heterogeneous phantom."""
    hu, body = heterogeneous_phantom(shape_yx)
    return _siddon_compare(hlut.numpy(hu) * body, angles_deg, 1.0,
                           lambda _, offsets, __: np.flatnonzero(np.abs(offsets) < max_offset)[::stride])


def siddon_relative_errors_case(hlut: HLUT, ct: np.ndarray, mask: np.ndarray, spacing_mm: float,
                                angles_deg: Sequence[float] = (0.0, 60.0, 135.0), stride: int = 8,
                                min_wet_mm: float = 20.0) -> pd.DataFrame:
    """Relative WET difference to pyRadPlan's Siddon tracer on the real anatomy of one patient.

    The axial slice with the largest body area is cropped to the body; every ``stride``-th ray crossing at least
    ``min_wet_mm`` of water-equivalent tissue is compared (shorter rays only graze the skin).
    """
    slab = central_slab(mask, 1)
    box = bounding_box(mask[slab])[1:]
    hu, body = ct[slab][0][box], (mask[slab][0][box] > 0.5)
    return _siddon_compare(hlut.numpy(hu) * body, angles_deg, spacing_mm,
                           lambda _, __, projected: np.flatnonzero(projected >= min_wet_mm)[::stride])


def _away_from_nodes(hu: torch.Tensor, nodes: np.ndarray, distance: float = 0.05) -> torch.Tensor:
    """Shift values lying within ``distance`` HU of an HLUT node, where the table has no derivative."""
    near = (hu.unsqueeze(-1) - torch.as_tensor(nodes, dtype=hu.dtype)).abs().amin(dim=-1) < distance
    return hu + 2 * distance * near.to(hu.dtype)


def gradcheck_range_loss(hlut: HLUT, seed: int = 0, ct: np.ndarray | None = None,
                         mask: np.ndarray | None = None) -> bool:
    """Finite-difference check of the range loss gradient in float64.

    With ``ct`` and ``mask`` (a real case), the check runs on a 17 × 19 patch of that patient's anatomy at the
    body centroid of its largest slice; otherwise on a uniform soft-tissue disk.
    """
    rng = np.random.default_rng(seed)
    shape = (17, 19)
    if ct is None or mask is None:
        body = torch.from_numpy(disk(shape, (8.0, 9.0), 7.0)[None] > 0.5).double()
        reference = torch.from_numpy(rng.uniform(-80, 80, size=(1, *shape)))
    else:
        slab = central_slab(mask, 1)
        centre = np.argwhere(mask[slab][0] > 0.5).mean(0).round().astype(int)
        y0 = int(np.clip(centre[0] - shape[0] // 2, 0, mask.shape[1] - shape[0]))
        x0 = int(np.clip(centre[1] - shape[1] // 2, 0, mask.shape[2] - shape[1]))
        patch = (slab, slice(y0, y0 + shape[0]), slice(x0, x0 + shape[1]))
        body = torch.from_numpy(mask[patch] > 0.5).double()
        reference = torch.from_numpy(ct[patch].astype(np.float64))
    noise = torch.from_numpy(rng.uniform(5, 30, size=(1, *shape)))
    sct = _away_from_nodes(reference + noise, hlut.hu).requires_grad_(True)
    projector = ParallelBeamProjector(shape, (0.0, 37.0, 101.0), spacing_mm=1.0)
    weights, valid = projector.segment_weights(body)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return range_mse(x, reference, body, projector, hlut, weights, valid)

    return bool(torch.autograd.gradcheck(loss, (sct,), eps=1e-3, atol=1e-6))


def benchmark_range_loss(hlut: HLUT, ct: np.ndarray, mask: np.ndarray, sct: np.ndarray, spacing_mm: float,
                         angles_deg: Sequence[float], n_slices: int = 16, device: str = "cpu",
                         repeats: int = 3) -> dict[str, float]:
    """Wall-clock time of one range-loss forward and backward pass on a real case (ms, best of ``repeats``).

    The ``n_slices`` slices around the largest body section are cropped to the body, as in training.
    """
    slab = central_slab(mask, n_slices)
    box = (slab, *bounding_box(mask[slab])[1:])

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(array[box]), dtype=torch.float32, device=device)

    ct, body = tensor(ct), tensor(mask)
    sct = tensor(sct).requires_grad_(True)
    z, y, x = ct.shape
    projector = ParallelBeamProjector((y, x), angles_deg, spacing_mm=spacing_mm, device=device)
    weights, valid = projector.segment_weights(body)

    def synchronise() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    forward, backward = [], []
    for _ in range(repeats):
        synchronise()
        start = time.perf_counter()
        loss = range_mse(sct, ct, body, projector, hlut, weights, valid)
        synchronise()
        middle = time.perf_counter()
        loss.backward()
        synchronise()
        forward.append(middle - start)
        backward.append(time.perf_counter() - middle)
        sct.grad = None
    return {"slices": z, "height": y, "width": x, "angles": len(angles_deg),
            "forward_ms": 1e3 * min(forward), "backward_ms": 1e3 * min(backward)}
