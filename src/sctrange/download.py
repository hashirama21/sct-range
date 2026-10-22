"""Download, verify, extract and validate the SynthRAD2025 training data from Zenodo.

Record 15373853 (v2, 11 May 2025, CT and CBCT clipped to [-1024, 3071] HU) holds
``synthRAD2025_Task1_Train.zip`` (MR -> CT, 14.8 GB) and ``synthRAD2025_Task2_Train.zip`` (CBCT -> CT, 12.0 GB).
Licence CC BY-NC 4.0; cite Thummerer et al. (2025), doi:10.5281/zenodo.15373853.

Pipeline (each step is idempotent, so an interrupted run resumes where it stopped):

1. query the Zenodo REST API for the file list, sizes and MD5 checksums;
2. stream every archive to ``<root>/archives`` with HTTP range resumption and verify its MD5;
3. extract into ``<root>/Task<k>/<region>/<case_id>/`` whatever the folder nesting inside the zip,
   refusing members that would escape the target directory;
4. check every case (files present, identical geometry, isotropic in-plane spacing, CT range) and write
   ``<root>/manifest.csv``.

The resulting ``<root>`` is what ``SYNTHRAD2025_ROOT`` and ``sctrange.data.discover_cases`` expect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .data import CASE_ID_PATTERN, SOURCE_BY_TASK, CaseRecord, discover_cases

ZENODO_API = "https://zenodo.org/api/records/{record}"
TRAINING_RECORD = "15373853"
CHUNK_BYTES = 8 << 20
CT_CLIP_HU = (-1024.0, 3071.0)
_TASK_DIR = {"1": "Task1", "2": "Task2"}


@dataclass(frozen=True)
class RemoteFile:
    key: str
    url: str
    size: int
    md5: str | None

    @property
    def task(self) -> str | None:
        """Task number read from the archive name (``..._Task1_...``), if any."""
        for task, name in _TASK_DIR.items():
            if name.lower() in self.key.lower():
                return task
        return None


def _open(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0):
    request = urllib.request.Request(url, headers={"User-Agent": "sctrange", **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)


def list_record_files(record: str = TRAINING_RECORD, api: str = ZENODO_API) -> list[RemoteFile]:
    """Files of a public Zenodo record, from its REST API entry."""
    with _open(api.format(record=record)) as response:
        payload = json.load(response)
    files = []
    for entry in payload.get("files", []):
        checksum = entry.get("checksum", "")
        algorithm, _, digest = checksum.partition(":")
        files.append(RemoteFile(
            key=entry["key"],
            url=entry["links"].get("content") or entry["links"]["self"],
            size=int(entry["size"]),
            md5=digest if algorithm == "md5" and digest else None,
        ))
    if not files:
        raise RuntimeError(f"Zenodo record {record} lists no files")
    return files


def md5sum(path: str | Path, chunk: int = CHUNK_BYTES) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _progress(key: str) -> Callable[[int, int], None]:
    state = {"last": 0.0}

    def report(done: int, total: int) -> None:
        now = time.monotonic()
        if now - state["last"] >= 2.0 or done >= total:
            state["last"] = now
            share = done / total if total else 1.0
            print(f"\r{key}: {done / 2**30:6.2f} / {total / 2**30:.2f} GiB ({share:6.1%})", end="", file=sys.stderr)
            if done >= total:
                print(file=sys.stderr)
    return report


def download_file(remote: RemoteFile, destination: str | Path, retries: int = 8,
                  progress: Callable[[int, int], None] | None = None) -> Path:
    """Stream ``remote`` to ``destination``, resuming a ``.part`` file; verify size and MD5.

    An existing complete file with the right checksum is kept as is.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == remote.size:
        if remote.md5 is None or md5sum(destination) == remote.md5:
            return destination
        destination.unlink()
    partial = destination.with_name(destination.name + ".part")
    progress = progress or _progress(remote.key)

    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > remote.size:
            partial.unlink()
            offset = 0
        if offset == remote.size:
            break
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with _open(remote.url, headers, timeout=120.0) as response:
                if offset and response.status != 206:
                    offset = 0  # server ignored the range: restart from scratch
                with open(partial, "ab" if offset else "wb") as handle:
                    done = offset
                    while block := response.read(CHUNK_BYTES):
                        handle.write(block)
                        done += len(block)
                        progress(done, remote.size)
            if partial.stat().st_size == remote.size:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if attempt == retries:
                raise RuntimeError(f"download of {remote.key} failed after {retries + 1} attempts") from error
            wait = min(2**attempt, 60)
            print(f"\n{remote.key}: {error!s}; retrying in {wait} s", file=sys.stderr)
            time.sleep(wait)
    else:
        raise RuntimeError(f"download of {remote.key} stopped at {partial.stat().st_size} of {remote.size} bytes")

    if remote.md5 is not None and (actual := md5sum(partial)) != remote.md5:
        partial.unlink()
        raise RuntimeError(f"{remote.key}: MD5 {actual} does not match Zenodo's {remote.md5}; partial file removed")
    partial.replace(destination)
    return destination


def member_destination(member: str, default_task: str | None = None) -> PurePosixPath | None:
    """Canonical relative path of a zip member, or ``None`` to skip it.

    Case files go to ``Task<k>/<region>/<case_id>/<file>`` wherever the case folder sits in the archive; the
    per-region ``overview`` folders keep their path from the ``Task<k>`` component on. Other members are
    placed under ``Task<k>/`` when the task is known, else skipped.
    """
    parts = PurePosixPath(member.replace("\\", "/")).parts
    if not parts or member.endswith("/") or any(p in ("..", "") for p in parts) or parts[0].startswith("/"):
        return None
    if parts[0].startswith("__MACOSX") or parts[-1].startswith("._"):
        return None
    for index, part in enumerate(parts[:-1]):
        match = CASE_ID_PATTERN.match(part)
        if match:
            fields = match.groupdict()
            return PurePosixPath(_TASK_DIR[fields["task"]], fields["region"], part, *parts[index + 1:])
    for index, part in enumerate(parts):
        if part in _TASK_DIR.values():
            return PurePosixPath(*parts[index:])
    if default_task is not None:
        return PurePosixPath(_TASK_DIR[default_task], *parts[1:] if len(parts) > 1 else parts)
    return None


def extract_archive(archive: str | Path, root: str | Path, default_task: str | None = None,
                    overwrite: bool = False) -> int:
    """Extract ``archive`` into the canonical layout under ``root``; return the number of files written.

    Files already present with the expected size are skipped, so extraction resumes after an interruption.
    """
    root = Path(root).resolve()
    written = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            relative = member_destination(info.filename, default_task)
            if relative is None:
                continue
            target = (root / relative).resolve()
            if root not in target.parents:
                raise ValueError(f"refusing to extract {info.filename!r} outside {root}")
            if target.exists() and target.stat().st_size == info.file_size and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            with bundle.open(info) as source, open(temporary, "wb") as sink:
                shutil.copyfileobj(source, sink, CHUNK_BYTES)
            temporary.replace(target)
            written += 1
    return written


@dataclass(frozen=True)
class CaseCheck:
    case_id: str
    task: str
    region: str
    center: str
    source: str
    shape_zyx: str
    spacing_xyz: str
    body_fraction: float
    ct_min_hu: float
    ct_max_hu: float
    problems: str

    @property
    def ok(self) -> bool:
        return not self.problems


def check_case(record: CaseRecord) -> CaseCheck:
    """Light-weight integrity check of one case; reads headers and the CT and mask arrays."""
    import SimpleITK as sitk

    problems: list[str] = []
    names = ("ct", "mask", record.source_name)
    missing = [name for name in names if not record.path(name).exists()]
    shape, spacing, body, low, high = (), (), float("nan"), float("nan"), float("nan")
    if missing:
        problems.append(f"missing {', '.join(missing)}")
    else:
        readers = {}
        for name in names:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(record.path(name)))
            reader.ReadImageInformation()
            readers[name] = reader
        reference = readers["ct"]
        shape = tuple(reversed(reference.GetSize()))
        spacing = tuple(round(s, 6) for s in reference.GetSpacing())
        for name, reader in readers.items():
            if reader.GetSize() != reference.GetSize():
                problems.append(f"{name} size differs from ct")
            elif not np.allclose(reader.GetSpacing(), reference.GetSpacing(), rtol=1e-5):
                problems.append(f"{name} spacing differs from ct")
            elif not np.allclose(reader.GetOrigin(), reference.GetOrigin(), atol=1e-3):
                problems.append(f"{name} origin differs from ct")
        if not np.isclose(spacing[0], spacing[1], rtol=1e-4):
            problems.append(f"anisotropic in-plane spacing {spacing[0]} x {spacing[1]}")
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(record.path("ct"))))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(record.path("mask")))) > 0
        low, high = float(ct.min()), float(ct.max())
        body = float(mask.mean())
        if body == 0.0:
            problems.append("empty body mask")
        if low < CT_CLIP_HU[0] - 0.5 or high > CT_CLIP_HU[1] + 0.5:
            problems.append(f"CT outside [{CT_CLIP_HU[0]:g}, {CT_CLIP_HU[1]:g}] HU")
    return CaseCheck(record.case_id, record.task, record.region, record.center, record.source_name,
                     "x".join(map(str, shape)), "x".join(map(str, spacing)), body, low, high, "; ".join(problems))


def validate_dataset(root: str | Path, tasks: Iterable[str] = ("1", "2")) -> pd.DataFrame:
    """Check every case of the requested tasks and write ``<root>/manifest.csv``."""
    tasks = set(tasks)
    records = [r for r in discover_cases(root) if r.task in tasks]
    frame = pd.DataFrame([{**asdict(c), "ok": c.ok} for c in map(check_case, records)])
    if not frame.empty:
        frame.to_csv(Path(root) / "manifest.csv", index=False)
    return frame


def prepare(root: str | Path, tasks: Iterable[str] = ("1", "2"), record: str = TRAINING_RECORD,
            keep_archives: bool = True, skip_download: bool = False, skip_validation: bool = False) -> pd.DataFrame:
    """Full pipeline for the requested tasks; returns the manifest (empty if validation is skipped)."""
    root = Path(root)
    tasks = tuple(sorted(set(tasks)))
    archives = root / "archives"
    if skip_download:
        selected = [(path, next((t for t, name in _TASK_DIR.items() if name.lower() in path.name.lower()), None))
                    for path in sorted(archives.glob("*.zip"))]
        selected = [(path, task) for path, task in selected if task in tasks]
    else:
        remote = [f for f in list_record_files(record) if f.key.endswith(".zip") and f.task in tasks]
        if not remote:
            raise RuntimeError(f"no archive for tasks {tasks} in record {record}")
        pending = sum(f.size for f in remote if not (archives / f.key).exists())
        archives.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(archives).free
        # archives plus extracted content (stored .mha are already compressed, so roughly the same size)
        if free < 2 * pending + sum(f.size for f in remote if (archives / f.key).exists()):
            print(f"warning: {free / 2**30:.1f} GiB free; about {2 * pending / 2**30:.1f} GiB needed", file=sys.stderr)
        selected = [(download_file(f, archives / f.key), f.task) for f in remote]

    for path, task in selected:
        print(f"extracting {path.name}: {extract_archive(path, root, task)} files written", file=sys.stderr)
        if not keep_archives:
            path.unlink()

    if skip_validation:
        return pd.DataFrame()
    manifest = validate_dataset(root, tasks)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the SynthRAD2025 training data (Zenodo).")
    parser.add_argument("--root", type=Path, help="dataset root, later used as SYNTHRAD2025_ROOT")
    parser.add_argument("--task", choices=("1", "2", "all"), default="all")
    parser.add_argument("--record", default=TRAINING_RECORD, help="Zenodo record id (default: training v2)")
    parser.add_argument("--delete-archives", action="store_true", help="remove each zip after extraction")
    parser.add_argument("--skip-download", action="store_true", help="only extract archives already in <root>/archives")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the record's files and exit")
    args = parser.parse_args(argv)

    if args.list:
        for remote in list_record_files(args.record):
            print(f"{remote.key}\t{remote.size / 2**30:.2f} GiB\tmd5:{remote.md5}")
        return
    if args.root is None:
        parser.error("--root is required")
    tasks = ("1", "2") if args.task == "all" else (args.task,)
    manifest = prepare(args.root, tasks, args.record, keep_archives=not args.delete_archives,
                       skip_download=args.skip_download, skip_validation=args.skip_validation)
    if manifest.empty:
        if not args.skip_validation:
            parser.error(f"no cases found under {args.root}")
        return
    summary = manifest.groupby(["task", "region", "center"])["ok"].agg(cases="size", valid="sum")
    print(summary.to_string())
    failures = manifest[~manifest["ok"]]
    for row in failures.itertuples():
        print(f"{row.case_id}: {row.problems}", file=sys.stderr)
    print(f"{int(manifest['ok'].sum())}/{len(manifest)} cases valid; manifest at {Path(args.root) / 'manifest.csv'}")
    print(f"export SYNTHRAD2025_ROOT={Path(args.root).resolve()}")
    if len(failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
