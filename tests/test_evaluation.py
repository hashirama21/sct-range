import numpy as np
import pytest

from sctrange.data import bulk_density_sct
from sctrange.evaluation import DiagnosticConfig, evaluate_case
from sctrange.hlut import HLUT
from sctrange.phantoms import synthetic_ct, synthetic_prediction


@pytest.mark.parametrize("region", ["AB", "HN", "TH"])
def test_evaluate_case_on_fixture(region: str, hlut: HLUT, rng: np.random.Generator) -> None:
    ct, mask = synthetic_ct(region, (96, 128), 3, rng)
    config = DiagnosticConfig(angles_deg=(0.0, 90.0, 180.0), target_radius_mm=15.0)
    for sct in (bulk_density_sct(mask), synthetic_prediction(ct, mask, rng)):
        row = evaluate_case(ct, mask, sct, (1.0, 1.0, 3.0), hlut, config)
        assert row["target_n_rays"] > 0 and row["chord_n_rays"] > 0
        assert 0.0 <= row["target_coherence"] <= 1.0
        assert row["target_mean_path_mm"] < row["chord_mean_path_mm"]
        assert np.isfinite(row["target_intra_ratio"])
