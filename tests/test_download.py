import fnmatch
import shutil
from pathlib import Path

import pandas as pd
import pytest

from sctrange import download
from sctrange.data import discover_cases
from sctrange.download import ensure_dataset, fetch, validate_dataset
from sctrange.phantoms import write_fixture_dataset


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """A fake Hugging Face dataset: ``snapshot_download`` copies the files matching ``allow_patterns``."""
    import huggingface_hub

    published = tmp_path / "published"
    write_fixture_dataset(published, regions=("AB", "HN"), centers=("A",), cases_per_stratum=1, shape_yx=(16, 16),
                          n_slices=1)
    validate_dataset(published, ("1",))
    calls: list[dict] = []

    def snapshot_download(repo_id, *, repo_type, revision, local_dir, allow_patterns, max_workers):
        calls.append({"repo_id": repo_id, "revision": revision, "patterns": list(allow_patterns)})
        for path in published.rglob("*"):
            relative = path.relative_to(published).as_posix()
            if path.is_file() and any(fnmatch.fnmatch(relative, p) for p in allow_patterns):
                (Path(local_dir) / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, Path(local_dir) / relative)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.delenv("SCT_HF_DATASET", raising=False)
    monkeypatch.delenv("SCT_HF_REVISION", raising=False)
    return calls


def test_fetch_pins_the_revision_and_filters_regions(hub: list[dict], tmp_path: Path) -> None:
    root = fetch(tmp_path / "hn", ("1",), regions=("HN",))
    assert [r.case_id for r in discover_cases(root)] == ["1HNA001"] and (root / "manifest.csv").exists()
    assert hub[-1] == {"repo_id": download.HF_DATASET, "revision": download.HF_REVISION,
                       "patterns": ["manifest.csv", "Task1/HN/**"]}
    assert not list(root.glob("Task1/.fetched-*"))  # a region subset is not a complete task


def test_ensure_dataset_fetches_once_then_works_offline(hub: list[dict], tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    root = ensure_dataset(tmp_path / "data", ("1",))
    assert sorted(r.case_id for r in discover_cases(root)) == ["1ABA001", "1HNA001"]
    assert ensure_dataset(root, ("1",)) == root and len(hub) == 1
    monkeypatch.setenv("SCT_HF_REVISION", "v3")  # another revision is fetched again
    ensure_dataset(root, ("1",))
    assert len(hub) == 2 and hub[-1]["revision"] == "v3"
    with pytest.raises(FileNotFoundError, match="task"):
        ensure_dataset(root, ("1", "2"))  # the mirror has no task 2 here


def test_ensure_dataset_without_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no SynthRAD2025 case"):
        ensure_dataset(tmp_path / "empty", download=False)
    write_fixture_dataset(tmp_path / "local", regions=("AB",), centers=("A",), cases_per_stratum=1, shape_yx=(16, 16),
                          n_slices=1)
    assert ensure_dataset(tmp_path / "local", ("1",), download=False) == tmp_path / "local"


def test_validate_dataset_merges_tasks_and_reports_problems(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_fixture_dataset(root, regions=("AB",), centers=("A",), cases_per_stratum=2, shape_yx=(24, 32), n_slices=2)
    manifest = validate_dataset(root, ("1",))
    assert manifest["ok"].all() and (root / "manifest.csv").exists()
    pd.DataFrame([{"case_id": "2HNA001", "task": "2", "ok": True}]).pipe(
        lambda other: pd.concat([pd.read_csv(root / "manifest.csv", dtype=str), other])).to_csv(root / "manifest.csv",
                                                                                             index=False)
    (root / "Task1" / "AB" / "1ABA002" / "mr.mha").unlink()
    assert validate_dataset(root, ("1",)).set_index("case_id").loc["1ABA002", "problems"] == "missing mr"
    assert "2HNA001" in set(download.read_manifest(root / "manifest.csv")["case_id"])


def test_cli_uses_the_published_manifest(hub: list[dict], tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    download.main(["--root", str(tmp_path / "cli"), "--task", "1", "--region", "AB"])
    output = capsys.readouterr().out
    assert "1/1 cases valid" in output and "SYNTHRAD2025_ROOT" in output
