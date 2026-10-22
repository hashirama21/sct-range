import json
from pathlib import Path

import pytest

from sctrange.data import CaseRecord
from sctrange.split import SplitSpec, load_split, main, stratified_split, write_split


def _records(regions=("AB", "HN", "TH"), centers=("A", "B", "C"), n=10) -> list[CaseRecord]:
    return [
        CaseRecord(f"1{r}{c}{i:03d}", "1", r, c, Path("."))
        for r in regions for c in centers for i in range(1, n + 1)
    ]


def test_partition_is_complete_and_disjoint() -> None:
    records = _records()
    split = stratified_split(records)
    assert not set(split["test"]) & set(split["development"])
    assert set(split["test"]) | set(split["development"]) == {r.case_id for r in records}


def test_every_stratum_contributes_its_fraction() -> None:
    split = stratified_split(_records(n=10), SplitSpec(test_fraction=0.2))
    assert all(counts["test"] == 2 for counts in split["strata"].values())


def test_deterministic_and_seed_sensitive() -> None:
    records = _records()
    assert stratified_split(records) == stratified_split(list(reversed(records)))
    assert stratified_split(records)["test"] != stratified_split(records, SplitSpec(seed=1))["test"]


def test_adding_a_stratum_leaves_others_unchanged() -> None:
    base = stratified_split(_records(centers=("A", "B")))
    extended = stratified_split(_records(centers=("A", "B", "C")))
    assert set(base["test"]) <= set(extended["test"])


def test_small_strata_stay_in_development() -> None:
    split = stratified_split(_records(n=1))
    assert split["test"] == []


def test_hash_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    write_split(stratified_split(_records()), path)
    assert load_split(path)["version"] == 1
    content = json.loads(path.read_text())
    content["test"].append("1ABZ999")
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_split(path)


def test_cli_refuses_to_overwrite(tmp_path: Path) -> None:
    from sctrange.phantoms import write_fixture_dataset

    write_fixture_dataset(tmp_path / "data", cases_per_stratum=3, shape_yx=(32, 32), n_slices=1)
    out = tmp_path / "split.json"
    main(["--data-root", str(tmp_path / "data"), "--out", str(out)])
    with pytest.raises(SystemExit):
        main(["--data-root", str(tmp_path / "data"), "--out", str(out)])
