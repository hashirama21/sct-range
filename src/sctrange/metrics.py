"""Image and range-error metrics for synthetic CT."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from .hlut import HLUT
from .projection import ParallelBeamProjector

TISSUE_EDGES_HU: tuple[float, ...] = (-math.inf, -500.0, -100.0, 150.0, math.inf)
GLOBAL_EDGES_HU: tuple[float, ...] = (-math.inf, math.inf)


def mae_hu(sct: torch.Tensor, ct: torch.Tensor, body: torch.Tensor) -> float:
    mask = body > 0.5
    return (sct - ct)[mask].abs().mean().item()


def spr_error(sct: torch.Tensor, ct: torch.Tensor, body: torch.Tensor, hlut: HLUT) -> torch.Tensor:
    return (hlut.torch(sct) - hlut.torch(ct)) * (body > 0.5).to(sct.dtype)


@dataclass
class RangeError:
    """Per-ray water-equivalent range error, shape ``(A, Z, S)`` for every field."""

    delta_mm: torch.Tensor
    absolute_mm: torch.Tensor
    path_mm: torch.Tensor
    valid: torch.Tensor

    def selected(self, min_path_mm: float = 0.0) -> torch.Tensor:
        return self.valid & (self.path_mm >= min_path_mm)


def range_error(
    sct: torch.Tensor,
    ct: torch.Tensor,
    body: torch.Tensor,
    projector: ParallelBeamProjector,
    hlut: HLUT,
    target: torch.Tensor | None = None,
    slab: int | None = None,
) -> RangeError:
    """Signed WET error ``dx * sum(SPR(sCT) - SPR(CT))`` from entry to body exit or distal target edge.

    ``slab`` processes that many slices at a time to bound memory; results are identical.
    """
    error = spr_error(sct, ct, body, hlut)
    step = slab or error.shape[0]
    parts = []
    for start in range(0, error.shape[0], step):
        rows = slice(start, start + step)
        weights, valid = projector.segment_weights(body[rows], None if target is None else target[rows])
        parts.append((projector.integrate(error[rows], weights), projector.integrate(error[rows].abs(), weights),
                      projector.path_length(weights), valid))
    delta, absolute, path, valid = (torch.cat(tensors, dim=1) for tensors in zip(*parts))
    return RangeError(delta_mm=delta, absolute_mm=absolute, path_mm=path, valid=valid)


@dataclass(frozen=True)
class RangeSummary:
    n_rays: int
    mean_path_mm: float
    mean_signed_mm: float
    rms_mm: float
    mean_abs_mm: float
    p95_mm: float
    cvar_mm: float
    coherence: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{key}": value for key, value in asdict(self).items()}


def cvar(values: torch.Tensor, alpha: float = 0.9) -> float:
    """Mean of the upper ``1 - alpha`` tail."""
    ordered = values.flatten().sort(descending=True).values
    k = max(1, math.ceil((1.0 - alpha) * ordered.numel()))
    return ordered[:k].mean().item()


def summarize(result: RangeError, alpha: float = 0.9, min_path_mm: float = 20.0) -> RangeSummary:
    mask = result.selected(min_path_mm)
    delta = result.delta_mm[mask].double()
    if delta.numel() == 0:
        nan = float("nan")
        return RangeSummary(0, nan, nan, nan, nan, nan, nan, nan)
    magnitude = delta.abs()
    absolute = result.absolute_mm[mask].double()
    coherence = magnitude[absolute > 0] / absolute[absolute > 0]
    return RangeSummary(
        n_rays=int(delta.numel()),
        mean_path_mm=result.path_mm[mask].double().mean().item(),
        mean_signed_mm=delta.mean().item(),
        rms_mm=delta.square().mean().sqrt().item(),
        mean_abs_mm=magnitude.mean().item(),
        p95_mm=torch.quantile(magnitude, 0.95).item(),
        cvar_mm=cvar(magnitude, alpha),
        coherence=coherence.median().item() if coherence.numel() else float("nan"),
    )


def permute_errors(
    sct: torch.Tensor,
    ct: torch.Tensor,
    body: torch.Tensor,
    edges_hu: Sequence[float] = TISSUE_EDGES_HU,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Shuffle HU errors among body voxels of the same reference tissue class.

    The multiset of errors, hence MAE, is preserved exactly; only their spatial layout changes.
    """
    error = sct - ct
    shuffled = error.clone()
    inside = body > 0.5
    for low, high in zip(edges_hu[:-1], edges_hu[1:]):
        index = torch.nonzero(inside & (ct >= low) & (ct < high), as_tuple=True)
        values = error[index]
        order = torch.randperm(values.numel(), generator=generator, device=values.device)
        shuffled[index] = values[order]
    return ct + shuffled


def rearrangement_bounds(errors: torch.Tensor, weights: torch.Tensor) -> tuple[float, float]:
    """Extremes of ``sum_i w_i e_pi(i)`` over all rearrangements ``pi`` of ``errors`` (Hardy-Littlewood).

    The maximum pairs both sequences in the same order (comonotone), the minimum in opposite orders
    (antimonotone). When neither sequence is constant, only comonotone (resp. antimonotone) arrangements
    attain them (discrete form of the sharp Frechet-Hoeffding bound, Bellini et al. 2022, Lemma A.2). ``errors`` and
    ``weights`` are flattened and must have the same number of elements; zero-padded weights model voxels
    off the ray.
    """
    e = errors.flatten().double().sort().values
    w = weights.flatten().double().sort().values
    if e.numel() != w.numel():
        raise ValueError("errors and weights must have the same number of elements")
    return (e * w.flip(0)).sum().item(), (e * w).sum().item()
