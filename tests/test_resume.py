"""Resumption after interruptions, and the guards that keep resumed work consistent."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sctrange.cache import VolumeCache
from sctrange.data import discover_cases, read_prediction, write_volume
from sctrange.download import usable_records
from sctrange.generative import GenerativeConfig, train_generative
from sctrange.hlut import HLUT
from sctrange.phantoms import write_fixture_dataset
from sctrange.pipeline import StageTracker
from sctrange.plotting import plot_training_curves
from sctrange.runs import cached_result, mark_complete, run_complete
from sctrange.training import TrainConfig, crossfit, fold_seed, train_model

TINY = dict(context=1, patch=48, batch_size=4, steps=20, base_channels=8, depth=3, warmup_steps=5,
            log_every=5, device="cpu", samples_per_case=300)


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("resume")
    write_fixture_dataset(root / "data", regions=("AB",), centers=("A", "B"), cases_per_stratum=2,
                          shape_yx=(56, 64), n_slices=3, seed=7)
    return root


@pytest.fixture(scope="module")
def cases(fixture_root: Path) -> list:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    cache.build(records)
    return [cache.load(r.case_id) for r in records]


def _same_weights(a: torch.nn.Module, b: torch.nn.Module) -> None:
    for (name, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
        torch.testing.assert_close(x, y, rtol=0, atol=0, msg=name)


def test_interrupted_training_resumes_to_the_same_model(cases: list, hlut: HLUT, tmp_path: Path) -> None:
    config = TrainConfig(loss="l2_spr", **TINY)
    reference, history = train_model(cases, config, hlut)
    partial = tmp_path / "fold0.partial.pt"
    train_model(cases, config, hlut, resume_path=partial, checkpoint_every=10)  # saves at step 10 of 20
    assert torch.load(partial, weights_only=True)["step"] == 10
    resumed, resumed_history = train_model(cases, config, hlut, resume_path=partial, checkpoint_every=10)
    _same_weights(reference, resumed)
    assert resumed_history["step"].tolist() == history["step"].tolist()
    other = TrainConfig(loss="l1_hu", **TINY)  # another experiment never resumes from this file
    fresh, _ = train_model(cases, other, hlut, resume_path=partial)
    _same_weights(fresh, train_model(cases, other, hlut)[0])


def test_interrupted_generator_resumes_to_the_same_model(cases: list, tmp_path: Path) -> None:
    config = GenerativeConfig(eta=0.1, **{**TINY, "log_every": 5})
    reference, _ = train_generative(cases, config)
    partial = tmp_path / "gen.partial.pt"
    train_generative(cases, config, resume_path=partial, checkpoint_every=10)
    resumed, _ = train_generative(cases, config, resume_path=partial, checkpoint_every=10)
    _same_weights(reference, resumed)


def test_fold_seeds_never_collide_across_runs() -> None:
    seeds = {fold_seed(seed, fold) for seed in range(3) for fold in range(5)}
    assert len(seeds) == 15


def test_checkpoint_trained_on_other_cases_is_refused(fixture_root: Path, hlut: HLUT) -> None:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(fixture_root / "cache")
    out = fixture_root / "runs" / "R3"
    config = TrainConfig(loss="l2_spr", **{**TINY, "steps": 6})
    crossfit(records, cache, config, hlut, 2, out)
    assert run_complete(out, [r.case_id for r in records])
    with pytest.raises(ValueError, match="other cases"):
        crossfit(records[:-1], cache, config, hlut, 2, out)  # another pool: the folds change
    assert not run_complete(out, [r.case_id for r in records])  # a refused rerun is not left marked complete


def test_cache_is_tied_to_the_source_files(fixture_root: Path, tmp_path: Path) -> None:
    records = discover_cases(fixture_root / "data")
    cache = VolumeCache(tmp_path / "cache")
    assert cache.build(records[:1]) == 1 and cache.build(records[:1]) == 0
    other = tmp_path / "other"
    write_fixture_dataset(other, regions=("AB",), centers=("A",), cases_per_stratum=1, shape_yx=(40, 48),
                          n_slices=2, seed=11)
    foreign = discover_cases(other)[0]
    assert foreign.case_id == records[0].case_id
    assert not cache.is_current(foreign.case_id, foreign) and cache.build([foreign]) == 1


def test_prediction_on_another_grid_is_refused(tmp_path: Path) -> None:
    write_volume(np.zeros((2, 4, 4), dtype=np.int16), tmp_path / "p.mha")
    assert read_prediction(tmp_path / "p.mha", (2, 4, 4)).shape == (2, 4, 4)
    with pytest.raises(ValueError, match="stale"):
        read_prediction(tmp_path / "p.mha", (3, 4, 4))


def test_invalid_cases_are_left_out(fixture_root: Path, tmp_path: Path) -> None:
    root = fixture_root / "data"
    records = discover_cases(root)
    manifest = pd.DataFrame({"case_id": [r.case_id for r in records], "task": "1",
                             "ok": [i != 0 for i in range(len(records))], "problems": ["empty body mask"] + [""] * 3})
    copy = tmp_path / "data"
    copy.mkdir()
    manifest.to_csv(copy / "manifest.csv", index=False)
    usable, excluded = usable_records(copy, records)
    assert excluded == {records[0].case_id: "empty body mask"} and len(usable) == len(records) - 1
    usable, excluded = usable_records(root, records)  # no manifest: every case is checked here
    assert not excluded and len(usable) == len(records)


def test_cached_result_reuses_until_the_key_changes(tmp_path: Path) -> None:
    calls = []

    def compute():
        calls.append(1)
        return {"value": len(calls)}

    path = tmp_path / "eval" / "case.pkl"
    assert cached_result(path, {"k": 1}, compute) == {"value": 1}
    assert cached_result(path, {"k": 1}, compute) == {"value": 1} and len(calls) == 1
    assert cached_result(path, {"k": 2}, compute) == {"value": 2}
    path.write_bytes(b"truncated")
    assert cached_result(path, {"k": 2}, compute) == {"value": 3}


def test_run_complete_needs_the_marker_and_every_prediction(tmp_path: Path) -> None:
    (tmp_path / "predictions").mkdir()
    assert not run_complete(tmp_path, ["a"])
    (tmp_path / "predictions" / "a.mha").write_bytes(b"x")
    assert not run_complete(tmp_path, ["a"])
    mark_complete(tmp_path, ["a", "b"])
    assert run_complete(tmp_path, ["a"]) and not run_complete(tmp_path, ["a", "b"])


def test_stage_tracker_skips_finished_stages_only(tmp_path: Path) -> None:
    report = tmp_path / "s3" / "s3_report.json"
    tracker = StageTracker(tmp_path)
    assert not tracker.begin("S3", {"steps": 10}, report)
    assert tracker.transform(["x = 1\n"]) == ["x = 1\n"]
    tracker.finish("S3", complete=False)  # stopped early: runs again
    assert not StageTracker(tmp_path).begin("S3", {"steps": 10}, report)
    report.parent.mkdir()
    report.write_text("{}")
    tracker.finish("S3", complete=True)
    again = StageTracker(tmp_path)
    assert again.begin("S3", {"steps": 10}, report)
    assert again.transform(["x = 1\n"]) == ["pass\n"]
    assert again.transform(['PIPELINE.begin("S4", {}, "r")\n']) != ["pass\n"]
    assert not StageTracker(tmp_path).begin("S3", {"steps": 20}, report)  # other parameters
    tracker = StageTracker(tmp_path)
    tracker.begin("S3", {"steps": 10}, report)
    tracker.finish("S3", complete=True)
    assert not StageTracker(tmp_path, force=["s3"]).begin("S3", {"steps": 10}, report)
    downstream = StageTracker(tmp_path)
    downstream.begin("S6", {}, report, after=("S3",))
    downstream.finish("S6", complete=True)
    assert StageTracker(tmp_path).begin("S6", {}, report, after=("S3",))
    rerun = StageTracker(tmp_path, force=["S3"])
    rerun.begin("S3", {"steps": 10}, report)
    rerun.finish("S3", complete=True)  # S3 finished again: S6 depends on it
    assert not StageTracker(tmp_path).begin("S6", {}, report, after=("S3",))
    report.unlink()
    assert not StageTracker(tmp_path).begin("S3", {"steps": 10}, report)  # report gone
    assert json.loads((tmp_path / "pipeline_state.json").read_text())["S3"]["status"] == "running"


def test_training_curves_without_history() -> None:
    assert plot_training_curves({}) is not None
