"""Where the notebooks keep data, outputs and caches: on a local machine, on Google Colab or on Kaggle.

| Platform | Dataset and volume cache | Outputs (checkpoints, predictions, reports, frozen split) |
|---|---|---|
| local  | ``<repo>/data/synthrad2025``, ``<repo>/outputs`` | ``<repo>/outputs`` |
| Colab  | ``/content/data/synthrad2025``, ``/content/cache`` | Google Drive ``MyDrive/sct-range/outputs`` (or local) |
| Kaggle | ``/kaggle/tmp/data/synthrad2025``, ``/kaggle/tmp/cache`` | ``/kaggle/working/outputs`` |

On Colab and Kaggle the session disk is temporary: outputs go where they outlive the session (Drive; the saved
``/kaggle/working`` of a notebook version), the 27 GB dataset and the cache stay on the scratch disk. Environment
variables always win: ``SYNTHRAD2025_ROOT``, ``SCT_OUTPUT_ROOT``, ``SCT_CACHE_ROOT``.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

COLAB_DATA = Path("/content/data/synthrad2025")
COLAB_CACHE = Path("/content/cache")
DRIVE_MOUNT = Path("/content/drive")
DRIVE_OUTPUTS = DRIVE_MOUNT / "MyDrive" / "sct-range" / "outputs"
KAGGLE_TMP = Path("/kaggle/tmp")
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")
SPLIT_NAME = "synthrad2025_local_test.json"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def in_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or KAGGLE_WORKING.is_dir()


def platform_name() -> str:
    return "colab" if in_colab() else "kaggle" if in_kaggle() else "local"


@dataclass(frozen=True)
class RuntimePaths:
    data_root: Path
    output_root: Path
    cache_root: Path
    platform: str

    @property
    def ephemeral(self) -> bool:
        """Session disk wiped after the run (Colab, Kaggle): archives are deleted once extracted to save space."""
        return self.platform in ("colab", "kaggle")

    def describe(self, repo_root: Path) -> dict[str, str]:
        def shown(path: Path) -> str:
            try:
                return path.resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                return str(path)

        return {"platform": self.platform, "data": shown(self.data_root), "outputs": shown(self.output_root),
                "cache": shown(self.cache_root)}


def mount_drive() -> None:
    from google.colab import drive  # type: ignore[import-not-found]

    if not (DRIVE_MOUNT / "MyDrive").exists():
        drive.mount(str(DRIVE_MOUNT))


def resolve_paths(repo_root: str | Path, use_drive: bool = True) -> RuntimePaths:
    """Data, output and cache roots for this platform (environment variables first, then platform defaults)."""
    repo_root = Path(repo_root)
    platform = platform_name()
    if platform == "kaggle":
        data, output, cache = KAGGLE_TMP / "data" / "synthrad2025", KAGGLE_WORKING / "outputs", KAGGLE_TMP / "cache"
    elif platform == "colab":
        data, cache = COLAB_DATA, COLAB_CACHE
        if use_drive and "SCT_OUTPUT_ROOT" not in os.environ:
            mount_drive()
            output = DRIVE_OUTPUTS
        else:
            output = repo_root / "outputs"
    else:
        data, output = repo_root / "data" / "synthrad2025", repo_root / "outputs"
        cache = output
    data = Path(os.environ.get("SYNTHRAD2025_ROOT", data))
    output = Path(os.environ.get("SCT_OUTPUT_ROOT", output))
    cache = Path(os.environ.get("SCT_CACHE_ROOT", cache))
    for path in (output, cache):
        path.mkdir(parents=True, exist_ok=True)
    if platform == "kaggle":
        restore_kaggle_outputs(output)
    return RuntimePaths(data, output, cache, platform)


def restore_kaggle_outputs(output_root: Path) -> Path | None:
    """Resume on Kaggle: copy the ``outputs/`` of an attached input (a previous version's output) into an empty
    ``output_root``, since inputs are read-only and ``/kaggle/working`` starts empty in every session."""
    if any(output_root.iterdir()) or not KAGGLE_INPUT.is_dir():
        return None
    candidates = sorted(KAGGLE_INPUT.glob("*/outputs")) + sorted(KAGGLE_INPUT.glob("*/*/outputs"))
    source = next((c for c in candidates if c.is_dir() and any(c.iterdir())), None)
    if source is not None:
        shutil.copytree(source, output_root, dirs_exist_ok=True)
        print(f"resumed outputs from {source}")
    return source


def frozen_split_path(repo_root: str | Path, paths: RuntimePaths) -> Path:
    """The committed split if present, else a copy kept with the outputs, else where S1 creates it (the repository)."""
    committed = Path(repo_root) / "splits" / SPLIT_NAME
    copy = paths.output_root / "splits" / SPLIT_NAME
    for candidate in (committed, copy):
        if candidate.exists() and candidate.with_suffix(candidate.suffix + ".sha256").exists():
            return candidate
    return committed


def save_split_copy(split_path: str | Path, paths: RuntimePaths) -> Path | None:
    """Copy the split and its hash next to the outputs (Drive, /kaggle/working) so that it outlives the session."""
    split_path = Path(split_path)
    target = paths.output_root / "splits"
    if split_path.parent.resolve() == target.resolve():
        return None
    target.mkdir(parents=True, exist_ok=True)
    for path in (split_path, split_path.with_suffix(split_path.suffix + ".sha256")):
        shutil.copy2(path, target / path.name)
    return target / split_path.name
