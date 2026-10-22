"""SynthRAD2025 case discovery, volume I/O and preprocessing helpers.

Expected layout: ``<root>/Task<k>/<region>/<case_id>/{ct,mask,mr|cbct}.mha`` with case IDs such as
``1ABA005`` = task 1, region AB, centre A, case 005. Discovery only requires one directory per case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

CASE_ID_PATTERN = re.compile(r"^(?P<task>[12])(?P<region>AB|HN|TH)(?P<center>[A-Z])(?P<number>\d+)$")
SOURCE_BY_TASK = {"1": "mr", "2": "cbct"}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    task: str
    region: str
    center: str
    directory: Path

    @property
    def stratum(self) -> str:
        return f"task{self.task}/{self.region}/{self.center}"

    @property
    def source_name(self) -> str:
        return SOURCE_BY_TASK[self.task]

    def path(self, name: str) -> Path:
        return self.directory / f"{name}.mha"


def parse_case_id(case_id: str) -> dict[str, str]:
    match = CASE_ID_PATTERN.match(case_id)
    if match is None:
        raise ValueError(f"unrecognised SynthRAD2025 case id: {case_id!r}")
    return match.groupdict()


def discover_cases(root: str | Path) -> list[CaseRecord]:
    """Every directory holding ``ct.mha`` and ``mask.mha`` under ``root``, sorted by case id."""
    records = []
    for ct_path in sorted(Path(root).rglob("ct.mha")):
        directory = ct_path.parent
        if not (directory / "mask.mha").exists():
            continue
        fields = parse_case_id(directory.name)
        records.append(
            CaseRecord(directory.name, fields["task"], fields["region"], fields["center"], directory)
        )
    ids = [record.case_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids found under root")
    return sorted(records, key=lambda record: record.case_id)


@dataclass(frozen=True)
class Volume:
    """Array in ``(Z, Y, X)`` order with SimpleITK geometry, spacing in ``(x, y, z)`` mm."""

    array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]

    @property
    def inplane_spacing(self) -> float:
        sx, sy, _ = self.spacing
        if not np.isclose(sx, sy, rtol=1e-4):
            raise ValueError(f"anisotropic in-plane spacing {sx} x {sy} mm is not supported")
        return float(sx)


def read_volume(path: str | Path, dtype: type = np.float32) -> Volume:
    image = sitk.ReadImage(str(path))
    return Volume(
        sitk.GetArrayFromImage(image).astype(dtype),
        tuple(image.GetSpacing()),
        tuple(image.GetOrigin()),
        tuple(image.GetDirection()),
    )


def write_volume(array: np.ndarray, path: str | Path, like: Volume | None = None,
                 spacing: tuple[float, float, float] = (1.0, 1.0, 3.0),
                 origin: tuple[float, float, float] | None = None,
                 direction: tuple[float, ...] | None = None) -> None:
    """Write ``(Z, Y, X)`` with the geometry of ``like`` if given, else the explicit geometry."""
    image = sitk.GetImageFromArray(array)
    if like is not None:
        spacing, origin, direction = like.spacing, like.origin, like.direction
    image.SetSpacing(spacing)
    if origin is not None:
        image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=True)


def load_case(record: CaseRecord, names: tuple[str, ...] = ("ct", "mask")) -> dict[str, Volume]:
    volumes = {name: read_volume(record.path(name)) for name in names}
    reference = volumes[names[0]]
    for name, volume in volumes.items():
        if volume.array.shape != reference.array.shape or not np.allclose(volume.spacing, reference.spacing):
            raise ValueError(f"{record.case_id}: {name} geometry differs from {names[0]}")
    return volumes


def bounding_box(mask: np.ndarray, margin: int = 4) -> tuple[slice, ...]:
    index = np.nonzero(mask > 0.5)
    if not index[0].size:
        raise ValueError("empty mask")
    return tuple(
        slice(max(int(axis.min()) - margin, 0), min(int(axis.max()) + margin + 1, size))
        for axis, size in zip(index, mask.shape)
    )


def bulk_density_sct(mask: np.ndarray, inside_hu: float = 0.0, outside_hu: float = -1000.0) -> np.ndarray:
    """SynthRAD2025 baseline: water inside the body contour, air outside."""
    return np.where(mask > 0.5, inside_hu, outside_hu).astype(np.float32)


def spherical_target(mask: np.ndarray, spacing: tuple[float, float, float], radius_mm: float,
                     center_zyx: tuple[float, float, float] | None = None) -> np.ndarray:
    """Surrogate target: a sphere clipped to the body, centred on the body centroid by default.

    The public SynthRAD2025 release contains no PTV; replace with a delineated target when available.
    """
    sx, sy, sz = spacing
    center = np.array(center_zyx if center_zyx is not None else np.argwhere(mask > 0.5).mean(0))
    z, y, x = np.ogrid[: mask.shape[0], : mask.shape[1], : mask.shape[2]]
    distance = ((z - center[0]) * sz) ** 2 + ((y - center[1]) * sy) ** 2 + ((x - center[2]) * sx) ** 2
    return ((distance <= radius_mm**2) & (mask > 0.5)).astype(np.float32)
