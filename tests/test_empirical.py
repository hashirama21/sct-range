import numpy as np
import pandas as pd
import pytest

from sctrange.data import bulk_density_sct, central_slab
from sctrange.empirical import case_identities, ray_bounds, segment_decomposition, summarize_identities
from sctrange.hlut import HLUT
from sctrange.phantoms import synthetic_ct
from sctrange.validation import benchmark_range_loss, gradcheck_range_loss, siddon_relative_errors_case


@pytest.fixture
def case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ct, mask = synthetic_ct("AB", (72, 88), 6, rng)
    return ct + rng.normal(0.0, 15.0, size=ct.shape), mask


def test_identities_are_exact_on_any_field(rng: np.random.Generator) -> None:
    error = 0.01 + rng.normal(size=(3, 40, 50)).cumsum(axis=1) * 0.01
    for axis in (1, 2):
        row = segment_decomposition(error, 12, axis, step=0.8)
        assert row["parseval_gap"] < 1e-10 and row["decomposition_gap"] < 1e-10
    # a random walk along axis 1 is coherent there: effective length well above one sample
    assert segment_decomposition(error, 12, 1, step=0.8)["effective_length_mm"] > 3 * 0.8


def test_ray_bounds_hold_and_are_reached(rng: np.random.Generator) -> None:
    body = np.zeros((6, 5), dtype=bool)
    body[1:5, 1:4] = True
    error = rng.normal(size=body.shape) * body
    assert ray_bounds(error, body, axis=0)["violations"] == 0
    comonotone = np.zeros_like(error)
    comonotone[1:5, 1] = [4.0, 3.0, 2.0, 1.0]  # the largest errors all on one ray
    assert ray_bounds(comonotone, body, axis=0)["max_position"] == pytest.approx(1.0)


def test_case_identities_on_a_volume(case: tuple[np.ndarray, np.ndarray], hlut: HLUT) -> None:
    ct, mask = case
    frame = case_identities(ct, mask, bulk_density_sct(mask), 1.0, hlut, segments_mm=(10.0, 30.0), n_slices=4)
    summary = summarize_identities(pd.DataFrame(frame).assign(case_id="x"))
    assert summary["violations"] == 0 and summary["rays"] > 0
    assert summary["max_parseval_gap"] < 1e-10 and summary["max_decomposition_gap"] < 1e-10


def test_central_slab_is_clipped_to_the_volume() -> None:
    mask = np.zeros((10, 4, 4))
    mask[9, 1:3, 1:3] = 1
    assert central_slab(mask, 4) == slice(6, 10)
    assert central_slab(mask, 20) == slice(0, 20)


def test_real_case_projector_checks(case: tuple[np.ndarray, np.ndarray], hlut: HLUT) -> None:
    ct, mask = case
    assert gradcheck_range_loss(hlut, ct=ct, mask=mask)
    timing = benchmark_range_loss(hlut, ct, mask, bulk_density_sct(mask), 1.0, (0.0, 120.0), n_slices=4, repeats=1)
    assert timing["slices"] == 4 and timing["forward_ms"] > 0
    pytest.importorskip("pyRadPlan")
    errors = siddon_relative_errors_case(hlut, ct, mask, 1.0, stride=4)
    assert len(errors) > 10
    assert errors["relative_error"].median() < 0.01
