"""Analytic phantoms and a synthetic SynthRAD2025-shaped fixture for pipeline testing.

Fixture images are geometric caricatures with plausible HU values. They exercise the code path
end to end and carry no clinical or scientific meaning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .data import write_volume

SPACING_XYZ = (1.0, 1.0, 3.0)


def ellipse(shape_yx: tuple[int, int], center_yx: tuple[float, float], radii_yx: tuple[float, float],
            supersample: int = 4) -> np.ndarray:
    """Anti-aliased ellipse: fractional pixel coverage in [0, 1]."""
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    y = np.arange(shape_yx[0])[:, None, None, None] + offsets[None, None, :, None]
    x = np.arange(shape_yx[1])[None, :, None, None] + offsets[None, None, None, :]
    inside = ((y - center_yx[0]) / radii_yx[0]) ** 2 + ((x - center_yx[1]) / radii_yx[1]) ** 2 <= 1.0
    return inside.mean(axis=(2, 3))


def disk(shape_yx: tuple[int, int], center_yx: tuple[float, float], radius: float,
         supersample: int = 4) -> np.ndarray:
    return ellipse(shape_yx, center_yx, (radius, radius), supersample)


def chord_length(radius: float, offset: np.ndarray) -> np.ndarray:
    """Length of a line at perpendicular distance ``offset`` from the centre of a disk."""
    return 2.0 * np.sqrt(np.clip(radius**2 - np.asarray(offset) ** 2, 0.0, None))


def _paint(image: np.ndarray, coverage: np.ndarray, value: float) -> None:
    image[:] = image * (1.0 - coverage) + value * coverage


def synthetic_ct(region: str, shape_yx: tuple[int, int], n_slices: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ct_hu, body_mask)`` arrays of shape ``(Z, Y, X)``."""
    height, width = shape_yx
    cy, cx = height / 2, width / 2
    radii = {"HN": (0.33, 0.30), "TH": (0.42, 0.46), "AB": (0.40, 0.43)}[region]
    body_r = (radii[0] * height * rng.uniform(0.93, 1.05), radii[1] * width * rng.uniform(0.93, 1.05))

    body = ellipse(shape_yx, (cy, cx), body_r)
    slice_hu = np.full(shape_yx, -1000.0)
    _paint(slice_hu, body, -100.0)
    _paint(slice_hu, ellipse(shape_yx, (cy, cx), (0.9 * body_r[0], 0.9 * body_r[1])), 40.0)
    _paint(slice_hu, disk(shape_yx, (cy + 0.7 * body_r[0], cx), 0.12 * body_r[0]), 1100.0)

    if region == "HN":
        outer = ellipse(shape_yx, (cy - 0.05 * body_r[0], cx), (0.78 * body_r[0], 0.8 * body_r[1]))
        inner = ellipse(shape_yx, (cy - 0.05 * body_r[0], cx), (0.68 * body_r[0], 0.7 * body_r[1]))
        _paint(slice_hu, np.clip(outer - inner, 0, 1), 1000.0)
        _paint(slice_hu, ellipse(shape_yx, (cy - 0.55 * body_r[0], cx), (0.12 * body_r[0], 0.3 * body_r[1])), -950.0)
    elif region == "TH":
        for side in (-1, 1):
            _paint(slice_hu, ellipse(shape_yx, (cy - 0.05 * body_r[0], cx + side * 0.42 * body_r[1]),
                                     (0.6 * body_r[0], 0.33 * body_r[1])), -800.0)
        ring = ellipse(shape_yx, (cy, cx), (0.86 * body_r[0], 0.86 * body_r[1])) - ellipse(
            shape_yx, (cy, cx), (0.82 * body_r[0], 0.82 * body_r[1]))
        _paint(slice_hu, np.clip(ring, 0, 1), 650.0)
    elif region == "AB":
        _paint(slice_hu, ellipse(shape_yx, (cy - 0.1 * body_r[0], cx - 0.35 * body_r[1]),
                                 (0.45 * body_r[0], 0.35 * body_r[1])), 60.0)
        for _ in range(rng.integers(2, 5)):
            center = (cy + rng.uniform(-0.3, 0.3) * body_r[0], cx + rng.uniform(0.0, 0.5) * body_r[1])
            _paint(slice_hu, disk(shape_yx, center, rng.uniform(4.0, 10.0)), -900.0)
    else:
        raise ValueError(f"unknown region {region!r}")

    ct = np.repeat(slice_hu[None], n_slices, axis=0)
    mask = np.repeat((body > 0.5)[None], n_slices, axis=0)
    ct = ct + rng.normal(0.0, 10.0, ct.shape) * mask
    return ct.astype(np.float32), mask.astype(np.float32)


def synthetic_mr(ct: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Non-physical surrogate intensity image that fills the dataset layout."""
    signal = np.interp(ct, [-1000, -100, 40, 300, 1500], [0, 900, 450, 250, 80])
    return (signal * mask + rng.normal(0.0, 15.0, ct.shape) * mask).astype(np.float32)


def synthetic_prediction(ct: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                         blur_px: float = 1.5) -> np.ndarray:
    """Imperfect sCT: tissue-level biases, interface blur and spatially correlated noise."""
    offsets = {(-np.inf, -500.0): 60.0, (-500.0, -100.0): 25.0, (-100.0, 150.0): 20.0, (150.0, np.inf): 120.0}
    biased = ct.copy()
    for (low, high), scale in offsets.items():
        biased[(ct >= low) & (ct < high)] += rng.normal(0.0, scale)
    blurred = gaussian_filter(biased, sigma=(0, blur_px, blur_px))
    noise = gaussian_filter(rng.normal(0.0, 1.0, ct.shape), sigma=(0, 6, 6))
    noise *= 25.0 / max(noise.std(), 1e-6)
    return np.where(mask > 0.5, blurred + noise, -1000.0).astype(np.float32)


def synthetic_cbct(ct: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """CBCT-like source: the CT with a smooth shading artefact, noise and a global intensity offset."""
    z, y, x = np.indices(ct.shape).astype(np.float32)
    shading = 60.0 * np.cos(2 * np.pi * (y / ct.shape[1] + rng.uniform())) * np.cos(np.pi * x / ct.shape[2])
    noisy = ct + shading + rng.normal(0, 25, ct.shape) + rng.uniform(-80, 80)
    return np.where(mask > 0.5, noisy, -1000.0).astype(np.float32)


def write_fixture_dataset(root: str | Path, prediction_root: str | Path | None = None,
                          regions: tuple[str, ...] = ("AB", "HN", "TH"), centers: tuple[str, ...] = ("A", "B"),
                          cases_per_stratum: int = 5, shape_yx: tuple[int, int] = (192, 256),
                          n_slices: int = 6, seed: int = 0, task: str = "1") -> list[str]:
    """Write a Task 1 (MR) or Task 2 (CBCT) fixture in the SynthRAD2025 layout and optional synthetic predictions."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    case_ids = []
    for region in regions:
        for center in centers:
            for number in range(1, cases_per_stratum + 1):
                case_id = f"{task}{region}{center}{number:03d}"
                directory = root / f"Task{task}" / region / case_id
                ct, mask = synthetic_ct(region, shape_yx, n_slices, rng)
                write_volume(ct, directory / "ct.mha", spacing=SPACING_XYZ)
                write_volume(mask.astype(np.uint8), directory / "mask.mha", spacing=SPACING_XYZ)
                if task == "1":
                    write_volume(synthetic_mr(ct, mask, rng), directory / "mr.mha", spacing=SPACING_XYZ)
                else:
                    write_volume(synthetic_cbct(ct, mask, rng), directory / "cbct.mha", spacing=SPACING_XYZ)
                if prediction_root is not None:
                    prediction = synthetic_prediction(ct, mask, rng)
                    write_volume(prediction, Path(prediction_root) / f"{case_id}.mha", spacing=SPACING_XYZ)
                case_ids.append(case_id)
    return case_ids
