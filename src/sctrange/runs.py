"""Bookkeeping shared by training, correction and the notebooks: completed runs and cached per-case results.

A run directory (``crossfit``, R4 correction) is *complete* only once every out-of-fold prediction is written; the
``complete.json`` marker records for which cases. Directories, ``folds.csv`` and ``predictions/`` exist from the start
of a run, so they never mean that the run finished.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

COMPLETE = "complete.json"
_ARTIFACT_ROOTS: list[tuple[Path, Path]] = []


def redirect_artifacts(persistent_root: str | Path, scratch_root: str | Path) -> None:
    """Keep the bulky outputs (``predictions/``, ``risk/``) of every run under ``persistent_root`` on ``scratch_root``.

    On Colab and Kaggle the persistent outputs (Drive, ``/kaggle/working``, 20 GB) hold checkpoints, in-fold progress,
    histories and evaluations, which a new session needs; out-of-fold predictions only feed evaluations cached with
    the outputs, and are rebuilt from the checkpoints when a later session needs them again (a run whose predictions
    are missing is not complete, so ``crossfit_runs`` predicts it again without training). Without a call, every
    artefact stays inside its run directory.
    """
    entry = (Path(persistent_root).resolve(), Path(scratch_root).resolve())
    if entry not in _ARTIFACT_ROOTS:
        _ARTIFACT_ROOTS.append(entry)


def artifact_dir(run_dir: str | Path, name: str = "predictions") -> Path:
    """Where the ``name`` folder of ``run_dir`` lives (``<run_dir>/<name>`` unless redirected)."""
    run_dir = Path(run_dir)
    resolved = run_dir.resolve()
    for persistent, scratch in _ARTIFACT_ROOTS:
        if resolved == persistent or persistent in resolved.parents:
            return scratch / resolved.relative_to(persistent) / name
    return run_dir / name


def ids_hash(case_ids: Iterable[str]) -> str:
    """Order-independent fingerprint of a set of case ids."""
    return hashlib.sha256("\n".join(sorted(set(case_ids))).encode()).hexdigest()[:16]


def files_signature(paths: Iterable[str | Path], mtime: bool = True) -> list[tuple[str, int, int]]:
    """``(name, size, mtime_ns)`` of every path (``-1`` when missing): changes whenever a file is rewritten.

    ``mtime=False`` keeps the size only, for the dataset: Colab and Kaggle download it again in every session, which
    changes the modification time of identical files.
    """
    rows = []
    for path in map(Path, paths):
        stat = path.stat() if path.exists() else None
        rows.append((path.name, stat.st_size if stat else -1, stat.st_mtime_ns if stat and mtime else -1))
    return rows


def config_key(config: Any, drop: Iterable[str] = ("device",)) -> str:
    """A dataclass configuration as a stable string, without the fields that do not change results (the device)."""
    from dataclasses import asdict

    drop = set(drop)
    return json.dumps({k: v for k, v in asdict(config).items() if k not in drop}, sort_keys=True, default=str)


def mark_incomplete(run_dir: str | Path) -> None:
    (Path(run_dir) / COMPLETE).unlink(missing_ok=True)


def mark_complete(run_dir: str | Path, case_ids: Iterable[str], **extra: Any) -> None:
    payload = {"cases": sorted(set(case_ids)), **extra}
    (Path(run_dir) / COMPLETE).write_text(json.dumps(payload, indent=2, default=str))


def completion(run_dir: str | Path) -> dict[str, Any]:
    """The ``complete.json`` marker of a finished run (cases and the extra fields of ``mark_complete``), else ``{}``."""
    path = Path(run_dir) / COMPLETE
    return json.loads(path.read_text()) if path.exists() else {}


def completed_cases(run_dir: str | Path) -> set[str]:
    return set(completion(run_dir).get("cases", ()))


def run_complete(run_dir: str | Path, case_ids: Iterable[str]) -> bool:
    """True when the run finished for every case of ``case_ids`` and each prediction is on disk."""
    run_dir, case_ids = Path(run_dir), set(case_ids)
    return (bool(case_ids) and case_ids <= completed_cases(run_dir)
            and all((artifact_dir(run_dir) / f"{case_id}.mha").exists() for case_id in case_ids))


def cached_result(path: str | Path, key: Any, compute: Callable[[], Any]) -> Any:
    """``compute()`` stored at ``path`` with its ``key``; reused while the key is unchanged (resumable evaluations).

    Pickle files are written by this code only, under the run's own output folder.
    """
    path = Path(path)
    if path.exists():
        try:
            saved = pickle.loads(path.read_bytes())
            if saved["key"] == key:
                return saved["value"]
        except Exception:  # truncated by an interrupted session, or written by another version
            pass
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps({"key": key, "value": value}))
    temporary.replace(path)
    return value
