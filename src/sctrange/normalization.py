"""Intensity normalisation of network inputs and targets."""

from __future__ import annotations

import numpy as np
import torch

CT_RANGE_HU = (-1024.0, 3071.0)
CBCT_PERCENTILES = (0.5, 99.5)
OUTSIDE_HU = -1000.0


def encode_ct(hu: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Clip to the CT range and map linearly to ``[-1, 1]``."""
    low, high = CT_RANGE_HU
    clipped = hu.clip(low, high)
    return 2.0 * (clipped - low) / (high - low) - 1.0


def decode_ct(value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Inverse of ``encode_ct`` without clipping, so that gradients reach out-of-range outputs."""
    low, high = CT_RANGE_HU
    return (value + 1.0) * (high - low) / 2.0 + low


def normalize_source(array: np.ndarray, mask: np.ndarray, modality: str) -> np.ndarray:
    """Per-case normalisation inside the body mask; zero outside.

    MR: z-score inside the mask. CBCT: clip to the 0.5-99.5 percentiles inside the mask, map to ``[-1, 1]``.
    """
    inside = mask > 0.5
    if not inside.any():
        raise ValueError("empty body mask")
    values = array[inside].astype(np.float64)
    if modality == "mr":
        scale = values.std()
        normalized = (array - values.mean()) / (scale if scale > 0 else 1.0)
    elif modality == "cbct":
        low, high = np.percentile(values, CBCT_PERCENTILES)
        normalized = 2.0 * (np.clip(array, low, high) - low) / max(high - low, 1e-6) - 1.0
    else:
        raise ValueError(f"unknown modality {modality!r}")
    return np.where(inside, normalized, 0.0).astype(np.float32)
