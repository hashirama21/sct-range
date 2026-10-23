import hashlib
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath

import pandas as pd
import pytest

from sctrange import download
from sctrange.data import discover_cases
from sctrange.download import RemoteFile, download_file, extract_archive, member_destination, validate_dataset
from sctrange.phantoms import write_fixture_dataset


class RangeHandler(SimpleHTTPRequestHandler):
    """Static server with single-range support; ``fail_after`` truncates the first response to simulate a drop."""

    fail_after: int | None = None

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        path = Path(self.translate_path(self.path))
        data = path.read_bytes()
        start = 0
        header = self.headers.get("Range")
        if header:
            start = int(header.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        if RangeHandler.fail_after is not None:
            body, RangeHandler.fail_after = body[: RangeHandler.fail_after], None
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path: Path):
    served = tmp_path / "served"
    served.mkdir()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(served)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield served, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _remote(served: Path, base: str, name: str, payload: bytes, md5: str | None = "auto") -> RemoteFile:
    (served / name).write_bytes(payload)
    digest = hashlib.md5(payload).hexdigest() if md5 == "auto" else md5
    return RemoteFile(name, f"{base}/{name}", len(payload), digest)


def test_download_resumes_after_interruption(server, tmp_path: Path) -> None:
    served, base = server
    payload = bytes(range(256)) * 4096
    remote = _remote(served, base, "a.zip", payload)
    RangeHandler.fail_after = 300_000
    destination = download_file(remote, tmp_path / "out" / "a.zip", progress=lambda *_: None)
    assert destination.read_bytes() == payload
    assert not destination.with_name("a.zip.part").exists()
    stamp = destination.stat().st_mtime_ns
    download_file(remote, destination, progress=lambda *_: None)
    assert destination.stat().st_mtime_ns == stamp


def test_download_rejects_bad_checksum(server, tmp_path: Path) -> None:
    served, base = server
    remote = _remote(served, base, "b.zip", b"x" * 1000, md5="0" * 32)
    with pytest.raises(RuntimeError, match="MD5"):
        download_file(remote, tmp_path / "b.zip", progress=lambda *_: None)
    assert not (tmp_path / "b.zip.part").exists()


def test_member_destination_normalises_layout() -> None:
    assert member_destination("synthRAD2025_Task1_Train/Task1/AB/1ABA005/ct.mha") == PurePosixPath("Task1/AB/1ABA005/ct.mha")
    assert member_destination("Train/1HNC012/mr.mha") == PurePosixPath("Task1/HN/1HNC012/mr.mha")
    assert member_destination("x/Task2/TH/overview/2_TH_parameters.xlsx") == PurePosixPath("Task2/TH/overview/2_TH_parameters.xlsx")
    assert member_destination("__MACOSX/Task1/AB/._ct.mha") is None
    assert member_destination("../evil.mha") is None
    assert member_destination("readme.txt") is None
    assert member_destination("top/readme.txt", default_task="2") == PurePosixPath("Task2/readme.txt")


def test_extract_and_validate_nested_archive(tmp_path: Path) -> None:
    source = tmp_path / "fixture"
    write_fixture_dataset(source, regions=("AB",), centers=("A",), cases_per_stratum=2, shape_yx=(24, 32), n_slices=2)
    archive = tmp_path / "synthRAD2025_Task1_Train.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in source.rglob("*.mha"):
            bundle.write(path, f"synthRAD2025_Task1_Train/{path.relative_to(source).as_posix()}")
    root = tmp_path / "dataset"
    assert extract_archive(archive, root, "1") == 6
    assert extract_archive(archive, root, "1") == 0
    assert [r.case_id for r in discover_cases(root)] == ["1ABA001", "1ABA002"]
    manifest = validate_dataset(root, ("1",))
    assert manifest["ok"].all() and (root / "manifest.csv").exists()
    (root / "Task1" / "AB" / "1ABA002" / "mr.mha").unlink()
    assert validate_dataset(root, ("1",)).set_index("case_id").loc["1ABA002", "problems"] == "missing mr"


def test_ensure_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sctrange.download import ensure_dataset

    with pytest.raises(FileNotFoundError, match="no SynthRAD2025 case"):
        ensure_dataset(tmp_path / "empty", download=False)
    calls = []
    monkeypatch.setattr(download, "prepare", lambda root, tasks, **_: calls.append(tuple(tasks)) or write_fixture_dataset(
        root, regions=("AB",), centers=("A",), cases_per_stratum=1, shape_yx=(16, 16), n_slices=1))
    root = ensure_dataset(tmp_path / "real", tasks=("1",))
    assert calls == [("1",)] and ensure_dataset(root, tasks=("1",)) == root and len(calls) == 1
    with pytest.raises(FileNotFoundError, match="task"):
        ensure_dataset(root, tasks=("1", "2"))


def test_prepare_processes_archives_one_by_one_and_never_refetches(server, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    served, base = server
    remotes = []
    for task, region in (("1", "AB"), ("2", "HN")):
        source = tmp_path / f"source{task}"
        write_fixture_dataset(source, regions=(region,), centers=("A",), cases_per_stratum=1, shape_yx=(16, 16),
                              n_slices=1)
        archive = tmp_path / f"synthRAD2025_Task{task}_Train.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in source.rglob("*.mha"):
                relative = path.relative_to(source).as_posix().replace("Task1/", f"Task{task}/")
                relative = relative.replace(f"/1{region}", f"/{task}{region}").replace("mr.mha", "mr.mha" if task == "1"
                                                                                       else "cbct.mha")
                bundle.write(path, f"top/{relative}")
        remotes.append(_remote(served, base, archive.name, archive.read_bytes()))
    monkeypatch.setattr(download, "list_record_files", lambda record: remotes)
    root = tmp_path / "dataset"

    manifest = download.prepare(root, ("1", "2"), keep_archives=False)
    assert sorted(manifest["case_id"]) == ["1ABA001", "2HNA001"] and manifest["ok"].all()
    assert not list((root / "archives").glob("*.zip"))
    monkeypatch.setattr(download, "download_file", lambda *a, **k: pytest.fail("extracted archive downloaded again"))
    download.prepare(root, ("1", "2"), keep_archives=False)
    assert sorted(pd.read_csv(root / "manifest.csv")["case_id"]) == ["1ABA001", "2HNA001"]


def test_extract_refuses_path_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("Task1/../../escape.mha", b"x")
    monkeypatch.setattr(download, "member_destination", lambda member, task=None: PurePosixPath(member))
    with pytest.raises(ValueError, match="outside"):
        extract_archive(archive, tmp_path / "root")
