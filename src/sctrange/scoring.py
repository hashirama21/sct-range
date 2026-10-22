"""Point functionals and proper scoring rules evaluated from samples."""

from __future__ import annotations

import numpy as np


def expectile(samples: np.ndarray, tau: float, axis: int = 0, iterations: int = 100) -> np.ndarray:
    """Root of ``tau E[(Y-e)_+] = (1-tau) E[(e-Y)_+]`` by bisection; ``tau = 0.5`` is the mean."""
    low = samples.min(axis=axis)
    high = samples.max(axis=axis)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        gap = np.expand_dims(middle, axis) - samples
        balance = tau * np.mean(np.maximum(-gap, 0), axis=axis) - (1 - tau) * np.mean(np.maximum(gap, 0), axis=axis)
        low = np.where(balance > 0, middle, low)
        high = np.where(balance > 0, high, middle)
    return 0.5 * (low + high)


def joint_mix(values: np.ndarray) -> np.ndarray:
    """Support ``(m, m)`` of ``m`` copies of the uniform law on ``values`` whose sum is constant.

    Row ``u`` is one equally likely outcome; copy ``k`` takes ``values[(u + k) mod m]``. Every column is a
    permutation of ``values`` and every row sums to ``sum(values)``.
    """
    values = np.asarray(values, dtype=np.float64)
    m = values.size
    return values[(np.arange(m)[:, None] + np.arange(m)[None, :]) % m]


def crps_ensemble(ensemble: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Fair ensemble CRPS ``E|X - y| - E|X - X'| / 2``, members along axis 0.

    The spread term uses the unbiased ``1 / (m (m - 1))`` normalisation.
    """
    members = np.sort(ensemble, axis=0)
    m = members.shape[0]
    accuracy = np.mean(np.abs(members - observation[None]), axis=0)
    weights = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * (members.ndim - 1))
    spread = np.sum(weights * members, axis=0) / (m * (m - 1))
    return accuracy - spread
