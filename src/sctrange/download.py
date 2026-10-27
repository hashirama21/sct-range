"""Fetch and validate the SynthRAD2025 training data from its Hugging Face mirror.

The training set (Zenodo record 15373853, v2, CT and CBCT clipped to [-1024, 3071] HU; licence CC BY-NC 4.0; cite
Thummerer et al. (2025), doi:10.5281/zenodo.15373853) is mirrored, already extracted, as a public Hugging Face
dataset (default ``developper669/synthrad2025``, tag ``v2``)::

    Task<k>/<region>/<case_id>/{ct,mask,mr|cbct}.mha
    Task<k>/<region>/overview/...
    manifest.csv                  integrity check of every case (``check_case``)

Every notebook and command reads the data from there only. ``fetch`` downloads the requested tasks (optionally some
regions) into ``<root>`` with ``huggingface_hub.snapshot_download``: parallel, resumable, files already present are
skipped, and the revision is pinned so that every run sees the same files. The resulting ``<root>`` is what
``SYNTHRAD2025_ROOT`` and ``sctrange.data.discover_cases`` expect.

``SCT_HF_DATASET`` and ``SCT_HF_REVISION`` override the repository and the revision.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CaseRecord, discover_cases

HF_DATASET = "developper669/synthrad2025"
HF_REVISION = "v2"
CT_CLIP_HU = (-1024.0, 3071.0)
TASK_DIRS = {"1": "Task1", "2": "Task2"}


def dataset_repo() -> str:
    return os.environ.get("SCT_HF_DATASET", HF_DATASET)


def dataset_revision() -> str:
    return os.environ.get("SCT_HF_REVISION", HF_REVISION)


def _fetched_marker(root: Path, task: str, repo_id: str, revision: str) -> Path:
    return root / TASK_DIRS[task] / f".fetched-{repo_id.replace('/', '--')}-{revision}"


def fetch(root: str | Path, tasks: Iterable[str] = ("1", "2"), regions: Iterable[str] | None = None,
          repo_id: str | None = None, revision: str | None = None, max_workers: int = 16) -> Path:
    """Download the cases of ``tasks`` (all regions, or only ``regions``) and the manifest into ``root``.

    Interrupted downloads resume; files already in ``root`` are not downloaded again. A complete task (no region
    filter) leaves a marker, so that later calls need no network.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    root = Path(root)
    tasks = tuple(sorted(set(tasks)))
    regions = tuple(sorted(set(regions))) if regions else None
    repo_id, revision = repo_id or dataset_repo(), revision or dataset_revision()
    folders = [f"{TASK_DIRS[t]}/{r}" for t in tasks for r in regions] if regions else [TASK_DIRS[t] for t in tasks]
    patterns = ["manifest.csv", *(f"{folder}/**" for folder in folders)]
    print(f"fetching task(s) {', '.join(tasks)}{' (' + ', '.join(regions) + ')' if regions else ''} "
          f"from huggingface.co/datasets/{repo_id}@{revision} into {root}", file=sys.stderr)
    try:
        snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=root,
                          allow_patterns=patterns, max_workers=max_workers)
    except (RepositoryNotFoundError, RevisionNotFoundError) as error:
        raise FileNotFoundError(f"Hugging Face dataset {repo_id}@{revision} not found: has the Zenodo mirror been "
                                "published (notebooks/s0_zenodo_to_huggingface.ipynb)? SCT_HF_DATASET and "
                                "SCT_HF_REVISION select another repository or revision") from error
    present = {record.task for record in discover_cases(root)}
    if missing := [t for t in tasks if t not in present]:
        raise FileNotFoundError(f"{repo_id}@{revision} holds no case of task(s) {', '.join(missing)}")
    if regions is None:
        for task in tasks:
            _fetched_marker(root, task, repo_id, revision).write_text(f"{repo_id}@{revision}\n")
    return root


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


def read_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"task": str, "case_id": str}, keep_default_na=False)
    frame["ok"] = frame["ok"].astype(str).str.lower() == "true"
    return frame


def validate_dataset(root: str | Path, tasks: Iterable[str] = ("1", "2")) -> pd.DataFrame:
    """Check every case of the requested tasks; update ``<root>/manifest.csv``, keeping rows of other tasks."""
    tasks = set(tasks)
    records = [r for r in discover_cases(root) if r.task in tasks]
    frame = pd.DataFrame([{**asdict(c), "ok": c.ok} for c in map(check_case, records)])
    path = Path(root) / "manifest.csv"
    if not frame.empty:
        if path.exists():
            previous = read_manifest(path)
            frame = pd.concat([previous[~previous["task"].isin(tasks)], frame], ignore_index=True)
        frame.sort_values("case_id").to_csv(path, index=False)
    return frame[frame["task"].isin(tasks)] if not frame.empty else frame


def ensure_dataset(root: str | Path, tasks: Iterable[str] = ("1", "2"), download: bool = True,
                   repo_id: str | None = None, revision: str | None = None) -> Path:
    """Return ``root`` once it holds the SynthRAD2025 cases of ``tasks``, fetching the missing tasks from Hugging Face.

    A task fetched before (marker present) needs no network. With ``download=False`` the cases already in ``root``
    are used as they are and a task without any case raises.
    """
    root = Path(root)
    tasks = tuple(sorted(set(tasks)))
    repo_id, revision = repo_id or dataset_repo(), revision or dataset_revision()
    if download:
        if pending := [t for t in tasks if not _fetched_marker(root, t, repo_id, revision).exists()]:
            fetch(root, pending, repo_id=repo_id, revision=revision)
    present = {record.task for record in discover_cases(root)} if root.exists() else set()
    if missing := [task for task in tasks if task not in present]:
        raise FileNotFoundError(f"no SynthRAD2025 case of task(s) {', '.join(missing)} under {root}; run "
                                f"`sctrange-download --root {root}` or allow the download")
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch the SynthRAD2025 training data from its Hugging Face mirror.")
    parser.add_argument("--root", type=Path, required=True, help="dataset root, later used as SYNTHRAD2025_ROOT")
    parser.add_argument("--task", choices=("1", "2", "all"), default="all")
    parser.add_argument("--region", action="append", choices=("AB", "HN", "TH"),
                        help="only this region (repeatable; default: all regions)")
    parser.add_argument("--repo", default=None, help=f"Hugging Face dataset (default {HF_DATASET} or SCT_HF_DATASET)")
    parser.add_argument("--revision", default=None, help=f"tag, branch or commit (default {HF_REVISION})")
    parser.add_argument("--validate", action="store_true",
                        help="re-check every case locally instead of trusting the published manifest")
    args = parser.parse_args(argv)

    tasks = ("1", "2") if args.task == "all" else (args.task,)
    fetch(args.root, tasks, args.region, args.repo, args.revision)
    if args.validate or not (args.root / "manifest.csv").exists():
        manifest = validate_dataset(args.root, tasks)
    else:
        manifest = read_manifest(args.root / "manifest.csv")
        manifest = manifest[manifest["task"].isin(tasks)]
        present = {record.case_id for record in discover_cases(args.root)}
        manifest = manifest[manifest["case_id"].isin(present)]
    summary = manifest.groupby(["task", "region", "center"])["ok"].agg(cases="size", valid="sum")
    print(summary.to_string())
    failures = manifest[~manifest["ok"]]
    for row in failures.itertuples():
        print(f"{row.case_id}: {row.problems}", file=sys.stderr)
    print(f"{int(manifest['ok'].sum())}/{len(manifest)} cases valid")
    print(f"export SYNTHRAD2025_ROOT={args.root.resolve()}")
    if len(failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
