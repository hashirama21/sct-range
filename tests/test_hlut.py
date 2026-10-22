import numpy as np
import pytest
import torch

from sctrange.hlut import HLUT


def test_bundled_table_matches_installed_pyradplan(hlut: HLUT) -> None:
    pytest.importorskip("pyRadPlan")
    installed = HLUT.from_pyradplan()
    np.testing.assert_array_equal(installed.hu, hlut.hu)
    np.testing.assert_array_equal(installed.spr, hlut.spr)
    assert installed.fingerprint() == hlut.fingerprint()


def test_matches_pyradplan_compute_wet(hlut: HLUT, rng: np.random.Generator) -> None:
    pytest.importorskip("pyRadPlan")
    import SimpleITK as sitk
    from pyRadPlan.ct import CT

    hu = rng.uniform(-1500, 3500, size=(4, 16, 16)).astype(np.float32)
    image = sitk.GetImageFromArray(hu)
    image.SetSpacing((1.0, 1.0, 3.0))
    reference = sitk.GetArrayFromImage(CT(cube_hu=image).compute_wet(np.stack([hlut.hu, hlut.spr], 1)))
    np.testing.assert_allclose(hlut.numpy(hu), reference, rtol=0, atol=1e-12)


def test_torch_matches_numpy(hlut: HLUT, rng: np.random.Generator) -> None:
    hu = rng.uniform(-1500, 3500, size=10_000)
    np.testing.assert_allclose(hlut.torch(torch.from_numpy(hu)).numpy(), hlut.numpy(hu), atol=1e-12)


def test_anchor_points_and_monotonicity(hlut: HLUT) -> None:
    assert hlut(np.array(0.0)) == pytest.approx(1.0)
    assert hlut(np.array(-1024.0)) == pytest.approx(0.001)
    grid = np.linspace(-2000, 4000, 5001)
    assert np.all(np.diff(hlut(grid)) >= 0)
    assert hlut(np.array(-5000.0)) == hlut(np.array(-1024.0))
    assert hlut(np.array(9000.0)) == hlut(np.array(3000.0))


def test_gradient_is_local_slope(hlut: HLUT) -> None:
    centers = (hlut.hu[:-1] + hlut.hu[1:]) / 2
    hu = torch.tensor(np.concatenate([centers, [-3000.0, 5000.0]]), requires_grad=True)
    hlut.torch(hu).sum().backward()
    expected = np.concatenate([hlut.slopes(), [0.0, 0.0]])
    np.testing.assert_allclose(hu.grad.numpy(), expected, rtol=1e-12)


def test_rejects_invalid_tables() -> None:
    with pytest.raises(ValueError):
        HLUT(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        HLUT(np.array([0.0, 1.0]), np.array([1.0, 0.5]))
