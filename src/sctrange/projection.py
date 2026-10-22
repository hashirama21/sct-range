"""Differentiable parallel-beam projector for coplanar beams in the axial plane."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RayGeometry:
    """Rays of one beam in original index coordinates ``(x, y)``, one ray per canvas column.

    Sample ``r`` of ray ``c`` lies at ``origins[c] + r * direction`` (unit step = one pixel).
    """

    origins_xy: np.ndarray
    direction_xy: np.ndarray
    n_samples: int


class ParallelBeamProjector:
    """Integrates fields along parallel rays for a set of gantry angles.

    Each axial slice is padded to a square canvas and rotated so that the beam propagates
    along increasing row index. Angle 0 travels along +row (``+y``), angle 90 along +column
    (``+x``). Sample spacing along and across rays equals the in-plane pixel spacing.
    """

    def __init__(
        self,
        shape_yx: tuple[int, int],
        angles_deg: Sequence[float],
        spacing_mm: float,
        device: torch.device | str | None = None,
    ) -> None:
        height, width = shape_yx
        self.shape_yx = (int(height), int(width))
        self.angles_deg = tuple(float(a) for a in angles_deg)
        self.spacing_mm = float(spacing_mm)
        self.side = int(math.ceil(math.hypot(height, width))) + 2
        self.pad_y = (self.side - height) // 2
        self.pad_x = (self.side - width) // 2
        self._padding = (
            self.pad_x,
            self.side - width - self.pad_x,
            self.pad_y,
            self.side - height - self.pad_y,
        )
        canvas_center = (self.side - 1) / 2
        self.center_xy = np.array([canvas_center - self.pad_x, canvas_center - self.pad_y])

        radians = torch.deg2rad(torch.tensor(self.angles_deg, dtype=torch.float64))
        cos, sin = torch.cos(radians), torch.sin(radians)
        rotation = torch.stack([torch.stack([cos, sin], -1), torch.stack([-sin, cos], -1)], -2)
        theta = torch.cat([rotation, torch.zeros(len(self.angles_deg), 2, 1, dtype=torch.float64)], -1)
        self._rotation = rotation.numpy()
        self._grid = F.affine_grid(
            theta, [len(self.angles_deg), 1, self.side, self.side], align_corners=False
        ).to(device)
        self._rows = torch.arange(self.side, device=device).view(self.side, 1)

    @property
    def n_angles(self) -> int:
        return len(self.angles_deg)

    def rotate(self, volume: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        """Map ``(Z, Y, X)`` to beam's-eye canvases ``(A, Z, S, S)``, depth along dim -2."""
        if volume.shape[-2:] != self.shape_yx:
            raise ValueError(f"expected in-plane shape {self.shape_yx}, got {tuple(volume.shape[-2:])}")
        padded = F.pad(volume, self._padding)
        stacked = padded.unsqueeze(0).expand(self.n_angles, -1, -1, -1)
        grid = self._grid.to(device=volume.device, dtype=volume.dtype)
        return F.grid_sample(stacked, grid, mode=mode, padding_mode="zeros", align_corners=False)

    def segment_weights(
        self, body: torch.Tensor, target: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integration masks from beam entry to the body exit or to the distal target edge.

        Returns ``weights`` of shape ``(A, Z, S, S)`` and ``valid`` rays of shape ``(A, Z, S)``.
        """
        rows = self._rows.to(body.device)
        inside = self.rotate(body.float()) > 0.5
        stop_mask = inside if target is None else self.rotate(target.float()) > 0.5
        end = torch.where(stop_mask, rows, -1).amax(dim=-2)
        weights = (rows <= end.unsqueeze(-2)) & inside
        return weights, end >= 0

    def integrate(self, field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Line integrals ``(A, Z, S)`` in field units times millimetres."""
        return self.spacing_mm * (self.rotate(field) * weights.to(field.dtype)).sum(dim=-2)

    def path_length(self, weights: torch.Tensor) -> torch.Tensor:
        return self.spacing_mm * weights.sum(dim=-2)

    def ray_geometry(self, angle_index: int) -> RayGeometry:
        rotation = self._rotation[angle_index]
        across, along = rotation[:, 0], rotation[:, 1]
        offsets = np.arange(self.side) - (self.side - 1) / 2
        start = -(self.side - 1) / 2
        origins = self.center_xy + offsets[:, None] * across + start * along
        return RayGeometry(origins_xy=origins, direction_xy=along.copy(), n_samples=self.side)

    def ray_offsets(self, angle_index: int, point_yx: tuple[float, float]) -> np.ndarray:
        """Signed perpendicular distance (pixels) from each ray of one beam to a point."""
        geometry = self.ray_geometry(angle_index)
        across = np.array([geometry.direction_xy[1], -geometry.direction_xy[0]])
        return (geometry.origins_xy - np.array([point_yx[1], point_yx[0]])) @ across
