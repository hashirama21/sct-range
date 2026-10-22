from pathlib import Path

import numpy as np
import pytest

from sctrange.data import bounding_box, discover_cases, load_case, parse_case_id, spherical_target
from sctrange.phantoms import write_fixture_dataset


def test_parse_case_id() -> None:
    assert parse_case_id("1ABA005") == {"task": "1", "region": "AB", "center": "A", "number": "005"}
    assert parse_case_id("2THE120")["task"] == "2"
    with pytest.raises(ValueError):
        parse_case_id("ABA005")


def test_discover_and_load_fixture(tmp_path: Path) -> None:
    ids = write_fixture_dataset(tmp_path, regions=("HN",), centers=("A", "C"), cases_per_stratum=2,
                                shape_yx=(40, 48), n_slices=2)
    records = discover_cases(tmp_path)
    assert [r.case_id for r in records] == sorted(ids)
    assert {r.stratum for r in records} == {"task1/HN/A", "task1/HN/C"}
    volumes = load_case(records[0], ("ct", "mask", "mr"))
    assert volumes["ct"].array.shape == (2, 40, 48)
    assert volumes["ct"].inplane_spacing == 1.0


def test_bounding_box_and_target() -> None:
    mask = np.zeros((5, 40, 40), dtype=np.float32)
    mask[1:4, 10:30, 12:28] = 1
    box = bounding_box(mask, margin=2)
    assert box == (slice(0, 5), slice(8, 32), slice(10, 30))
    target = spherical_target(mask, (1.0, 1.0, 3.0), radius_mm=5.0)
    assert target.sum() > 0 and np.all(target <= mask)
