import sys
import types
from pathlib import Path

import pytest

from sctrange import runtime
from sctrange.runtime import SPLIT_NAME, frozen_split_path, resolve_paths, save_split_copy
from sctrange.split import load_split, write_split


@pytest.fixture(autouse=True)
def _clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SYNTHRAD2025_ROOT", "SCT_OUTPUT_ROOT", "SCT_CACHE_ROOT", "KAGGLE_KERNEL_RUN_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setattr(runtime, "KAGGLE_WORKING", tmp_path / "no-kaggle")


def test_local_defaults_live_in_the_repository(tmp_path: Path) -> None:
    paths = resolve_paths(tmp_path)
    assert paths.platform == "local" and not paths.ephemeral
    assert paths.data_root == tmp_path / "data" / "synthrad2025"
    assert paths.output_root == paths.cache_root == tmp_path / "outputs"
    assert paths.describe(tmp_path)["outputs"] == "outputs"


def test_environment_variables_win(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTHRAD2025_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("SCT_OUTPUT_ROOT", str(tmp_path / "o"))
    monkeypatch.setenv("SCT_CACHE_ROOT", str(tmp_path / "c"))
    paths = resolve_paths(tmp_path)
    assert (paths.data_root, paths.output_root, paths.cache_root) == (tmp_path / "d", tmp_path / "o", tmp_path / "c")


def test_colab_uses_drive_for_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    mounted = []
    monkeypatch.setattr(runtime, "mount_drive", lambda: mounted.append(True))
    monkeypatch.setattr(runtime, "DRIVE_OUTPUTS", tmp_path / "drive")
    monkeypatch.setattr(runtime, "COLAB_CACHE", tmp_path / "cache")
    paths = resolve_paths(tmp_path)
    assert paths.platform == "colab" and paths.ephemeral and mounted and paths.output_root == tmp_path / "drive"
    assert paths.data_root == runtime.COLAB_DATA and paths.cache_root == tmp_path / "cache"
    without_drive = resolve_paths(tmp_path, use_drive=False)
    assert without_drive.output_root == tmp_path / "outputs" and len(mounted) == 1


def test_kaggle_keeps_data_in_tmp_and_outputs_in_working(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setattr(runtime, "KAGGLE_TMP", tmp_path / "kaggle" / "tmp")
    monkeypatch.setattr(runtime, "KAGGLE_WORKING", tmp_path / "kaggle" / "working")
    paths = resolve_paths(tmp_path)
    assert paths.platform == "kaggle" and paths.ephemeral
    assert paths.data_root == tmp_path / "kaggle" / "tmp" / "data" / "synthrad2025"
    assert paths.cache_root == tmp_path / "kaggle" / "tmp" / "cache"
    assert paths.output_root == tmp_path / "kaggle" / "working" / "outputs"


def test_kaggle_resumes_outputs_from_an_attached_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setattr(runtime, "KAGGLE_TMP", tmp_path / "tmp")
    monkeypatch.setattr(runtime, "KAGGLE_WORKING", tmp_path / "working")
    monkeypatch.setattr(runtime, "KAGGLE_INPUT", tmp_path / "input")
    previous = tmp_path / "input" / "sct-range-s4-v3" / "outputs" / "s4" / "runs" / "seed0" / "R3"
    previous.mkdir(parents=True)
    (previous / "fold0.pt").write_bytes(b"checkpoint")
    paths = resolve_paths(tmp_path)
    assert (paths.output_root / "s4" / "runs" / "seed0" / "R3" / "fold0.pt").read_bytes() == b"checkpoint"
    (previous / "fold0.pt").write_bytes(b"older")
    resolve_paths(tmp_path)  # outputs no longer empty: nothing is overwritten
    assert (paths.output_root / "s4" / "runs" / "seed0" / "R3" / "fold0.pt").read_bytes() == b"checkpoint"


def test_split_copy_round_trip(tmp_path: Path) -> None:
    paths = resolve_paths(tmp_path)
    committed = tmp_path / "splits" / SPLIT_NAME
    assert frozen_split_path(tmp_path, paths) == committed
    write_split({"version": 1, "test": ["a"], "development": ["b"]}, committed)
    copy = save_split_copy(committed, paths)
    assert copy == paths.output_root / "splits" / SPLIT_NAME and load_split(copy)["test"] == ["a"]
    committed.unlink()
    committed.with_suffix(".json.sha256").unlink()
    assert frozen_split_path(tmp_path, paths) == copy
    assert save_split_copy(copy, paths) is None
