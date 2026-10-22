"""Voxel-wise two-component Gaussian mixture posterior with a Gaussian-copula dependence.

Every voxel ``i`` is the alternative tissue with probability ``p_i`` and the base tissue otherwise,
``Y_i | class ~ N(mu_class, sigma_class^2)``. Class indicators are coupled across voxels by thresholding
a unit-variance Gaussian random field ``Z`` of correlation length ``lambda``: alternative iff
``Phi(Z_i) < p_i``. Marginals are therefore exact and independent of ``lambda``; only the joint law changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .fields import gaussian_random_field
from .hlut import HLUT


@dataclass(frozen=True)
class Tissue:
    mean_hu: float
    std_hu: float


SOFT = Tissue(40.0, 20.0)
GAS = Tissue(-850.0, 60.0)
BONE = Tissue(700.0, 120.0)


@dataclass(frozen=True, eq=False)
class MixturePosterior:
    probability: np.ndarray
    base: Tissue = SOFT
    alternative: Tissue = GAS

    def mean_hu(self) -> np.ndarray:
        p = self.probability
        return p * self.alternative.mean_hu + (1 - p) * self.base.mean_hu

    def cdf(self, hu: np.ndarray) -> np.ndarray:
        p = self.probability
        return (p * ndtr((hu - self.alternative.mean_hu) / self.alternative.std_hu)
                + (1 - p) * ndtr((hu - self.base.mean_hu) / self.base.std_hu))

    def quantile_hu(self, level: float, iterations: int = 80) -> np.ndarray:
        """Voxel-wise quantile by bisection on the mixture CDF."""
        components = (self.base, self.alternative)
        low = np.full(self.probability.shape, min(c.mean_hu - 10 * c.std_hu for c in components))
        high = np.full(self.probability.shape, max(c.mean_hu + 10 * c.std_hu for c in components))
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            below = self.cdf(middle) < level
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        return 0.5 * (low + high)

    def median_hu(self) -> np.ndarray:
        return self.quantile_hu(0.5)

    def mean_spr(self, hlut: HLUT, nodes: int = 4001) -> np.ndarray:
        """``E[g(Y_i)]`` by quadrature of the piecewise-linear table against each Gaussian component."""
        def component_mean(tissue: Tissue) -> float:
            grid = np.linspace(tissue.mean_hu - 9 * tissue.std_hu, tissue.mean_hu + 9 * tissue.std_hu, nodes)
            density = np.exp(-0.5 * ((grid - tissue.mean_hu) / tissue.std_hu) ** 2)
            return float(np.trapezoid(hlut.numpy(grid) * density, grid) / np.trapezoid(density, grid))

        p = self.probability
        return p * component_mean(self.alternative) + (1 - p) * component_mean(self.base)

    def spr_mean_estimate(self, hlut: HLUT) -> np.ndarray:
        """Bayes act for squared range loss: ``g^{-1}(E[g(Y_i)])``."""
        return hlut.inverse(self.mean_spr(hlut))

    def sample(self, n: int, correlation_length: float, rng: np.random.Generator) -> np.ndarray:
        """``n`` joint draws of shape ``(n, *probability.shape)`` in HU."""
        latent = gaussian_random_field(self.probability.shape, correlation_length, n, rng)
        alternative = ndtr(latent) < self.probability
        noise = rng.standard_normal(latent.shape)
        return np.where(alternative,
                        self.alternative.mean_hu + self.alternative.std_hu * noise,
                        self.base.mean_hu + self.base.std_hu * noise)


def blob_probability(shape: tuple[int, int], centers: list[tuple[float, float]], radius: float,
                     peak: float) -> np.ndarray:
    """Smooth probability map: sum of Gaussian bumps, clipped to ``[0, peak]``."""
    y, x = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    field = sum(np.exp(-0.5 * ((y - cy) ** 2 + (x - cx) ** 2) / radius**2) for cy, cx in centers)
    return np.clip(peak * field, 0.0, peak)
