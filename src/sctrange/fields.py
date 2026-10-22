"""Stationary Gaussian random fields on periodic grids, with exactly known covariance."""

from __future__ import annotations

import numpy as np


def exponential_spectrum(shape: tuple[int, ...], length: float) -> np.ndarray:
    """Spectral density of the exponential (Matern 1/2) covariance, normalised to unit variance.

    ``length`` is the correlation length in pixels; ``length -> 0`` tends to white noise.
    """
    if length <= 0:
        return np.ones(shape)
    frequencies = np.meshgrid(*(np.fft.fftfreq(n) for n in shape), indexing="ij")
    squared = sum(f**2 for f in frequencies)
    exponent = (len(shape) + 1) / 2
    density = (1.0 / length**2 + 4.0 * np.pi**2 * squared) ** (-exponent)
    return density / density.mean()


def covariance(spectrum: np.ndarray) -> np.ndarray:
    """Exact circular covariance ``C[tau]`` of fields drawn with ``spectrum``; ``C[0] = 1``."""
    return np.real(np.fft.ifftn(spectrum))


def gaussian_random_field(shape: tuple[int, ...], length: float, n: int,
                          rng: np.random.Generator) -> np.ndarray:
    """``n`` independent zero-mean, unit-variance fields of shape ``shape``."""
    amplitude = np.sqrt(exponential_spectrum(shape, length))
    axes = tuple(range(1, len(shape) + 1))
    white = rng.standard_normal((n, *shape))
    return np.real(np.fft.ifftn(amplitude * np.fft.fftn(white, axes=axes), axes=axes))
