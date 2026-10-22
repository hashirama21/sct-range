"""Verification of the range projector against analytic geometry and pyRadPlan's Siddon tracer."""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

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


def siddon_relative_errors(hlut: HLUT, angles_deg: Sequence[float] = (0.0, 30.0, 75.0, 140.0),
                           shape_yx: tuple[int, int] = (101, 121), stride: int = 4,
                           max_offset: float = 38.0) -> pd.DataFrame:
    """Relative WET difference to pyRadPlan's Siddon tracer on a voxelised heterogeneous phantom."""
    import SimpleITK as sitk
    from pyRadPlan.raytracer import RayTracerSiddon

    centre = _centre(shape_yx)
    hu, body = heterogeneous_phantom(shape_yx)
    volume = np.repeat((hlut.numpy(hu) * body)[None], 3, axis=0)
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((1.0, 1.0, 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tracer = RayTracerSiddon([image])
    projector = ParallelBeamProjector(shape_yx, angles_deg, spacing_mm=1.0)
    integrals = projector.integrate(torch.from_numpy(volume), _ALL)
    isocentre = np.array([centre[1], centre[0], 1.0])

    rows = []
    for index, angle in enumerate(projector.angles_deg):
        geometry = projector.ray_geometry(index)
        offsets = projector.ray_offsets(index, centre)
        for column in np.flatnonzero(np.abs(offsets) < max_offset)[::stride]:
            start = geometry.origins_xy[column] - 10 * geometry.direction_xy
            stop = geometry.origins_xy[column] + (geometry.n_samples + 10) * geometry.direction_xy
            source = np.array([*start, 1.0]) - isocentre
            target = np.array([*stop, 1.0]) - isocentre
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                _, lengths, rho, _, _ = tracer.trace_ray(isocentre, source, target)
            reference = float(np.nansum(lengths * rho[0]))
            projected = integrals[index, 1, column].item()
            rows.append({"angle_deg": angle, "offset_px": float(offsets[column]), "siddon_mm": reference,
                         "projector_mm": projected, "relative_error": abs(projected - reference) / reference})
    return pd.DataFrame(rows)


def gradcheck_range_loss(hlut: HLUT, seed: int = 0) -> bool:
    """Finite-difference check of the range loss gradient in float64."""
    rng = np.random.default_rng(seed)
    shape = (17, 19)
    body = torch.from_numpy(disk(shape, (8.0, 9.0), 7.0)[None] > 0.5).double()
    ct = torch.from_numpy(rng.uniform(-80, 80, size=(1, *shape)))
    sct = (ct + torch.from_numpy(rng.uniform(5, 30, size=(1, *shape)))).requires_grad_(True)
    projector = ParallelBeamProjector(shape, (0.0, 37.0, 101.0), spacing_mm=1.0)
    weights, valid = projector.segment_weights(body)

    def loss(x: torch.Tensor) -> torch.Tensor:
        return range_mse(x, ct, body, projector, hlut, weights, valid)

    return bool(torch.autograd.gradcheck(loss, (sct,), eps=1e-3, atol=1e-6))


def benchmark_range_loss(hlut: HLUT, shape_zyx: tuple[int, int, int], angles_deg: Sequence[float],
                         device: str = "cpu", repeats: int = 3) -> dict[str, float]:
    """Wall-clock time of one range-loss forward and backward pass (ms, best of ``repeats``)."""
    z, y, x = shape_zyx
    body = torch.from_numpy(disk((y, x), _centre((y, x)), 0.45 * min(y, x))[None] > 0.5).float()
    body = body.expand(z, -1, -1).contiguous().to(device)
    ct = (torch.randn(z, y, x, device=device) * 100.0) * body
    sct = (ct + 20.0 * torch.randn_like(ct)).requires_grad_(True)
    projector = ParallelBeamProjector((y, x), angles_deg, spacing_mm=1.0, device=device)
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
