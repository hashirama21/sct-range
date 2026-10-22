"""Spectral weighting of voxel-wise, gradient and segment-integral (range) error functionals."""

from __future__ import annotations

import numpy as np


def segment_transfer(frequency: np.ndarray, length: int) -> np.ndarray:
    """Normalised power transfer ``|B(f)|^2 / L^2`` of a length-``L`` segment sum (Dirichlet kernel).

    ``frequency`` is in cycles per sample along the beam. Equals ``sinc^2(fL)`` up to aliasing.
    """
    frequency = np.asarray(frequency, dtype=np.float64)
    numerator = np.sin(np.pi * frequency * length)
    denominator = length * np.sin(np.pi * frequency)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.isclose(np.sin(np.pi * frequency), 0.0), 1.0, numerator / denominator)
    return ratio**2


def gradient_transfer(frequency: np.ndarray) -> np.ndarray:
    """Power transfer of a forward-difference gradient loss, normalised to 1 at Nyquist."""
    return np.sin(np.pi * np.asarray(frequency, dtype=np.float64)) ** 2


def sliding_segment_sums(field: np.ndarray, length: int, axis: int = 0, step: float = 1.0) -> np.ndarray:
    """Circular sums of ``length`` consecutive samples along ``axis``, times ``step``."""
    if not 1 <= length <= field.shape[axis]:
        raise ValueError("length must be between 1 and the field size along axis")
    cumulative = np.cumsum(np.concatenate([field, np.take(field, range(length), axis=axis)], axis=axis), axis=axis)
    n = field.shape[axis]
    upper = np.take(cumulative, range(length - 1, length - 1 + n), axis=axis)
    lower = np.concatenate([np.zeros_like(np.take(cumulative, [0], axis=axis)),
                            np.take(cumulative, range(n - 1), axis=axis)], axis=axis)
    return step * (upper - lower)


def segment_energy_spectral(field: np.ndarray, length: int, axis: int = 0, step: float = 1.0) -> float:
    """``sum_p W_p^2`` predicted by Parseval from the field spectrum (exact for circular segments)."""
    spectrum = np.abs(np.fft.fftn(field)) ** 2
    shape = [1] * field.ndim
    shape[axis] = field.shape[axis]
    frequency = np.fft.fftfreq(field.shape[axis]).reshape(shape)
    weight = length**2 * segment_transfer(frequency, length)
    return float(step**2 * (spectrum * weight).sum() / field.size)


def effective_length(correlation: np.ndarray, n: int, step: float = 1.0) -> float:
    """``l = step * (1 + 2 sum_{k=1}^{n-1} (1 - k/n) rho(k))`` from a correlation sequence along the ray."""
    lags = np.arange(1, n)
    return float(step * (1.0 + 2.0 * np.sum((1.0 - lags / n) * correlation[lags])))


def predicted_range_rms(mean: float, std: float, n: int, correlation: np.ndarray, step: float = 1.0) -> float:
    """``sqrt(L^2 mu^2 + L sigma^2 l)`` for a stationary error along a segment of ``n`` samples."""
    path = n * step
    return float(np.sqrt(path**2 * mean**2 + path * std**2 * effective_length(correlation, n, step)))
