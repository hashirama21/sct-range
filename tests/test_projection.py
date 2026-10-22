import numpy as np
import pytest
import torch
from scipy.ndimage import map_coordinates

from sctrange.hlut import HLUT
from sctrange.phantoms import disk
from sctrange.projection import ParallelBeamProjector
from sctrange.validation import (
    disk_chord_errors,
    gradcheck_range_loss,
    siddon_relative_errors,
    target_path_errors,
)

ANGLES = (0.0, 17.0, 45.0, 90.0, 133.0, 210.0, 301.0)
SHAPE = (121, 141)
ALL = torch.ones(1, dtype=torch.bool)


def test_uniform_disk_chords_match_analytic() -> None:
    errors = disk_chord_errors(ANGLES)
    assert (errors["median_abs_mm"] < 0.1).all()
    assert (errors["max_abs_mm"] < 0.5).all()


def test_target_segment_stops_at_distal_edge() -> None:
    errors = target_path_errors(ANGLES)
    assert errors["valid_inside"].all() and not errors["valid_outside"].any()
    assert (errors["max_abs_mm"] < 1.5).all()


def test_agrees_with_pyradplan_siddon(hlut: HLUT) -> None:
    pytest.importorskip("pyRadPlan")
    errors = siddon_relative_errors(hlut)
    assert errors["relative_error"].median() < 0.01
    assert errors["relative_error"].max() < 0.04


def test_range_loss_gradcheck(hlut: HLUT) -> None:
    assert gradcheck_range_loss(hlut)


def test_ray_geometry_matches_independent_sampling(rng: np.random.Generator) -> None:
    field = rng.normal(size=(1, *SHAPE))
    projector = ParallelBeamProjector(SHAPE, ANGLES, spacing_mm=1.0)
    integrals = projector.integrate(torch.from_numpy(field), ALL)
    for index in range(len(ANGLES)):
        geometry = projector.ray_geometry(index)
        steps = np.arange(geometry.n_samples)
        for column in rng.choice(projector.side, size=12, replace=False):
            points = geometry.origins_xy[column] + steps[:, None] * geometry.direction_xy
            samples = map_coordinates(field[0], [points[:, 1], points[:, 0]], order=1, mode="grid-constant")
            assert integrals[index, 0, column].item() == pytest.approx(samples.sum(), abs=1e-8)


def test_spacing_scales_integrals() -> None:
    field = torch.from_numpy(disk(SHAPE, (60.0, 70.0), 45.0)[None])
    unit = ParallelBeamProjector(SHAPE, ANGLES, 1.0).integrate(field, ALL)
    scaled = ParallelBeamProjector(SHAPE, ANGLES, 0.8).integrate(field, ALL)
    torch.testing.assert_close(scaled, 0.8 * unit)


def test_linearity(rng: np.random.Generator) -> None:
    projector = ParallelBeamProjector((40, 50), (0.0, 63.0), spacing_mm=1.0)
    a, b = (torch.from_numpy(rng.normal(size=(2, 40, 50))) for _ in range(2))
    torch.testing.assert_close(projector.integrate(2 * a - 3 * b, ALL),
                               2 * projector.integrate(a, ALL) - 3 * projector.integrate(b, ALL))


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        ParallelBeamProjector((10, 12), (0.0,), 1.0).rotate(torch.zeros(1, 12, 10))
