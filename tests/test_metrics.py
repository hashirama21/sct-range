import numpy as np
import pytest
import torch

from sctrange.hlut import HLUT
from sctrange.metrics import GLOBAL_EDGES_HU, cvar, mae_hu, permute_errors, range_error, summarize
from sctrange.phantoms import disk
from sctrange.projection import ParallelBeamProjector

SHAPE = (64, 72)


@pytest.fixture
def case(rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    body = torch.from_numpy((disk(SHAPE, (32, 36), 28) > 0.5)[None].repeat(2, 0)).float()
    ct = torch.where(body > 0, torch.from_numpy(rng.normal(30, 200, size=(2, *SHAPE))).float(), torch.tensor(-1000.0))
    sct = ct + torch.from_numpy(rng.normal(0, 40, size=(2, *SHAPE))).float() * body
    return ct, sct, body


def test_permutation_preserves_mae_and_error_multiset(case) -> None:
    ct, sct, body = case
    permuted = permute_errors(sct, ct, body, generator=torch.Generator().manual_seed(1))
    assert mae_hu(permuted, ct, body) == pytest.approx(mae_hu(sct, ct, body), rel=1e-6)
    inside = body > 0.5
    np.testing.assert_allclose(np.sort((permuted - ct)[inside].numpy()), np.sort((sct - ct)[inside].numpy()), atol=1e-4)
    assert torch.equal(permuted[~inside], ct[~inside])


def test_uniform_bias_is_fully_coherent(case, hlut: HLUT) -> None:
    ct, _, body = case
    ct = torch.where(body > 0, torch.tensor(40.0), torch.tensor(-1000.0))
    sct = ct + 20.0 * body
    projector = ParallelBeamProjector(SHAPE, (0.0, 45.0, 90.0), spacing_mm=1.0)
    stats = summarize(range_error(sct, ct, body, projector, hlut))
    assert stats.coherence == pytest.approx(1.0, abs=1e-5)


def test_coherence_is_bounded_and_drops_for_white_errors(case, hlut: HLUT) -> None:
    ct, sct, body = case
    projector = ParallelBeamProjector(SHAPE, (0.0, 60.0, 120.0), spacing_mm=1.0)
    result = range_error(sct, ct, body, projector, hlut)
    mask = result.selected(20.0)
    ratio = result.delta_mm[mask].abs() / result.absolute_mm[mask]
    assert (ratio <= 1 + 1e-5).all()
    assert summarize(result).coherence < 0.3


def test_global_permutation_breaks_tissue_bias(hlut: HLUT) -> None:
    body = torch.from_numpy((disk(SHAPE, (32, 36), 28) > 0.5)[None]).float()
    ct = torch.where(body > 0, torch.tensor(40.0), torch.tensor(-1000.0))
    ct[:, :, :36] = torch.where(body[:, :, :36] > 0, torch.tensor(-800.0), ct[:, :, :36])
    sct = ct + torch.where(ct < -500, torch.tensor(50.0), torch.tensor(-50.0)) * body
    projector = ParallelBeamProjector(SHAPE, (0.0,), spacing_mm=1.0)
    real = summarize(range_error(sct, ct, body, projector, hlut))
    shuffled = permute_errors(sct, ct, body, GLOBAL_EDGES_HU, torch.Generator().manual_seed(0))
    control = summarize(range_error(shuffled, ct, body, projector, hlut))
    assert real.rms_mm > 2 * control.rms_mm


def test_cvar_upper_tail() -> None:
    values = torch.arange(1.0, 101.0)
    assert cvar(values, 0.9) == pytest.approx(95.5)
    assert cvar(values, 0.0) == pytest.approx(values.mean().item())


def test_rearrangement_bounds_match_exhaustive_search() -> None:
    from itertools import permutations

    from sctrange.metrics import rearrangement_bounds

    errors = torch.tensor([-3.0, -0.5, 0.0, 1.0, 2.5, 4.0])
    weights = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    values = [float(torch.dot(errors[list(order)], weights)) for order in permutations(range(6))]
    low, high = rearrangement_bounds(errors, weights)
    assert low == pytest.approx(min(values)) == pytest.approx(-3.5)
    assert high == pytest.approx(max(values)) == pytest.approx(7.5)
