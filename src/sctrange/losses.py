"""Training losses built on the range projector."""

from __future__ import annotations

import torch

from .hlut import HLUT
from .projection import ParallelBeamProjector


def range_mse(
    sct: torch.Tensor,
    ct: torch.Tensor,
    body: torch.Tensor,
    projector: ParallelBeamProjector,
    hlut: HLUT,
    weights: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean squared WET error over valid rays, in mm^2.

    ``weights`` and ``valid`` come from ``projector.segment_weights`` on the reference and are
    computed once per case, outside the autograd graph.
    """
    error = (hlut.torch(sct) - hlut.torch(ct)) * (body > 0.5).to(sct.dtype)
    delta = projector.integrate(error, weights)
    return delta[valid].square().mean()


def ray_risk(projector: ParallelBeamProjector, var_density: torch.Tensor, excess_density: torch.Tensor,
             weights: torch.Tensor, floor_mm: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-ray ``(VaR, ES)`` of ``|ΔWET|`` from two non-negative densities (mm of water per mm of path).

    Integrating densities along the same segments as the range error makes the risk head plan-independent, like the
    regressor: ``VaR_r = Δs Σ v_i`` and ``ES_r = VaR_r + Δs Σ e_i``, so ``ES >= VaR > 0`` holds by construction, as the
    FZ0 score requires. ``floor_mm`` keeps both away from zero on very short segments.
    """
    value_at_risk = projector.integrate(var_density, weights) + floor_mm
    return value_at_risk, value_at_risk + projector.integrate(excess_density, weights)


def fz_upper_tail_loss(value_at_risk: torch.Tensor, expected_shortfall: torch.Tensor,
                       outcome: torch.Tensor, level: float = 0.9) -> torch.Tensor:
    """FZ0 score (Patton, Ziegel and Chen, 2019) for the upper tail of a non-negative outcome.

    Strictly consistent for ``(VaR_level, ES_level)`` jointly, with ``ES = E[Z | Z >= VaR]``.
    Requires ``expected_shortfall > 0``. Obtained from the lower-tail FZ0 score applied to ``-Z``.
    """
    tail = 1.0 - level
    exceedance = (outcome >= value_at_risk).to(outcome.dtype) * (outcome - value_at_risk)
    score = (exceedance / (tail * expected_shortfall) + value_at_risk / expected_shortfall
             + torch.log(expected_shortfall) - 1.0)
    return score.mean()
