"""Hounsfield-unit to stopping-power-ratio conversion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import torch

_PYRADPLAN_DEFAULT_HU = (-1024.0, -999.0, -90.0, -45.0, 0.0, 100.0, 350.0, 3000.0)
_PYRADPLAN_DEFAULT_SPR = (0.001, 0.001, 0.95, 0.99, 1.0, 1.095, 1.199, 2.505)


@dataclass(frozen=True, eq=False)
class HLUT:
    """Piecewise-linear lookup table with constant extrapolation, as in ``np.interp``."""

    hu: np.ndarray
    spr: np.ndarray
    source: str = field(default="custom")

    def __post_init__(self) -> None:
        hu = np.asarray(self.hu, dtype=np.float64)
        spr = np.asarray(self.spr, dtype=np.float64)
        if hu.ndim != 1 or hu.shape != spr.shape or hu.size < 2:
            raise ValueError("hu and spr must be 1-D arrays of equal length >= 2")
        if np.any(np.diff(hu) <= 0):
            raise ValueError("hu nodes must be strictly increasing")
        if np.any(np.diff(spr) < 0):
            raise ValueError("spr values must be non-decreasing")
        object.__setattr__(self, "hu", hu)
        object.__setattr__(self, "spr", spr)

    @classmethod
    def pyradplan_default(cls) -> HLUT:
        """Default table of pyRadPlan 0.2.8 to 0.5.0, used by the SynthRAD2025 dose evaluation."""
        return cls(np.array(_PYRADPLAN_DEFAULT_HU), np.array(_PYRADPLAN_DEFAULT_SPR), "pyRadPlan default")

    @classmethod
    def from_pyradplan(cls) -> HLUT:
        """Read the table from the installed pyRadPlan package."""
        from pyRadPlan.ct import default_hlut

        table = np.asarray(default_hlut(), dtype=np.float64)
        order = np.argsort(table[:, 0])
        return cls(table[order, 0], table[order, 1], "pyRadPlan (installed)")

    def __call__(self, hu: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(hu, torch.Tensor):
            return self.torch(hu)
        return self.numpy(hu)

    def numpy(self, hu: np.ndarray) -> np.ndarray:
        return np.interp(hu, self.hu, self.spr)

    def torch(self, hu: torch.Tensor) -> torch.Tensor:
        """Differentiable conversion; the gradient is the local slope and zero outside the table."""
        xp = torch.as_tensor(self.hu, dtype=hu.dtype, device=hu.device)
        fp = torch.as_tensor(self.spr, dtype=hu.dtype, device=hu.device)
        clamped = hu.clamp(xp[0], xp[-1])
        index = torch.bucketize(clamped.detach(), xp[1:-1])
        x0, x1 = xp[index], xp[index + 1]
        f0, f1 = fp[index], fp[index + 1]
        return f0 + (clamped - x0) * (f1 - f0) / (x1 - x0)

    def inverse(self, spr: np.ndarray) -> np.ndarray:
        """HU with the given SPR on the strictly increasing part of the table (upper end of flat runs)."""
        keep = np.r_[np.diff(self.spr) > 0, True]
        return np.interp(spr, self.spr[keep], self.hu[keep])

    def slopes(self) -> np.ndarray:
        """SPR per HU on each segment."""
        return np.diff(self.spr) / np.diff(self.hu)

    def fingerprint(self) -> str:
        payload = json.dumps({"hu": self.hu.tolist(), "spr": self.spr.tolist()}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
