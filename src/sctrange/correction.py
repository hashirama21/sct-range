"""Arm R4: post-hoc per-tissue SPR bias correction of an existing arm (R3 by default).

The correction is the cheap control of the plan: if subtracting one SPR offset per tissue class recovers the gain of
the range loss (R5), the gain was a tissue-level bias, not a property of the loss. Classes are read from the
**prediction** (the reference CT is unavailable at test time), and offsets are cross-fitted: the offsets applied to the
cases of fold ``k`` are estimated on the out-of-fold predictions of the other folds only.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CaseRecord, load_case, read_volume, write_volume
from .hlut import HLUT
from .metrics import TISSUE_EDGES_HU


def tissue_classes(hu: np.ndarray, edges_hu: Sequence[float] = TISSUE_EDGES_HU) -> np.ndarray:
    """Class index in ``[0, len(edges) - 2]`` of every voxel, with the same half-open bins as ``permute_errors``."""
    return np.clip(np.searchsorted(np.asarray(edges_hu), hu, side="right") - 1, 0, len(edges_hu) - 2)


def bias_sums(sct: np.ndarray, ct: np.ndarray, mask: np.ndarray, hlut: HLUT,
              edges_hu: Sequence[float] = TISSUE_EDGES_HU) -> tuple[np.ndarray, np.ndarray]:
    """Per predicted class, the sum of ``g(CT) - g(sCT)`` over body voxels and the voxel count."""
    inside = mask > 0.5
    classes = tissue_classes(sct[inside], edges_hu)
    residual = hlut.numpy(ct[inside]) - hlut.numpy(sct[inside])
    n = len(edges_hu) - 1
    return np.bincount(classes, residual, minlength=n), np.bincount(classes, minlength=n).astype(np.float64)


def apply_bias(sct: np.ndarray, mask: np.ndarray, offsets: np.ndarray, hlut: HLUT,
               edges_hu: Sequence[float] = TISSUE_EDGES_HU) -> np.ndarray:
    """Add the class offset in SPR and map back to HU inside the body; voxels outside are unchanged."""
    inside = mask > 0.5
    spr = hlut.numpy(sct[inside]) + np.asarray(offsets)[tissue_classes(sct[inside], edges_hu)]
    corrected = sct.astype(np.float32, copy=True)
    corrected[inside] = hlut.inverse(np.clip(spr, hlut.spr[0], hlut.spr[-1]))
    return corrected


def crossfit_bias_correction(records: list[CaseRecord], source_dir: str | Path, output_dir: str | Path, hlut: HLUT,
                             edges_hu: Sequence[float] = TISSUE_EDGES_HU) -> pd.DataFrame:
    """Write cross-fitted corrected predictions of ``source_dir`` to ``output_dir``; return the offsets per fold.

    ``source_dir`` is a ``crossfit`` output (``folds.csv`` and ``predictions/``). Each prediction is read twice: once to
    accumulate per-case sums, once to apply its fold's offsets, so memory stays at one case.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    folds = pd.read_csv(source_dir / "folds.csv", dtype={"case_id": str}).set_index("case_id")["fold"].to_dict()
    records = [r for r in records if r.case_id in folds]
    if not records:
        raise ValueError(f"no record of {source_dir / 'folds.csv'} is available")

    def prediction(record: CaseRecord) -> np.ndarray:
        return read_volume(source_dir / "predictions" / f"{record.case_id}.mha").array

    sums, counts = {}, {}
    for record in records:
        volumes = load_case(record, ("ct", "mask"))
        sums[record.case_id], counts[record.case_id] = bias_sums(prediction(record), volumes["ct"].array,
                                                                 volumes["mask"].array, hlut, edges_hu)
    labels = [f"[{low:g}, {high:g}) HU" for low, high in zip(edges_hu[:-1], edges_hu[1:])]
    offsets = {}
    for fold in sorted(set(folds[r.case_id] for r in records)):
        others = [r.case_id for r in records if folds[r.case_id] != fold]
        total = np.sum([sums[i] for i in others], axis=0) if others else np.zeros(len(labels))
        count = np.sum([counts[i] for i in others], axis=0) if others else np.zeros(len(labels))
        offsets[fold] = np.divide(total, count, out=np.zeros_like(total), where=count > 0)

    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    for record in records:
        volume = read_volume(source_dir / "predictions" / f"{record.case_id}.mha")
        mask = load_case(record, ("mask",))["mask"].array
        corrected = apply_bias(volume.array, mask, offsets[folds[record.case_id]], hlut, edges_hu)
        write_volume(corrected, output_dir / "predictions" / f"{record.case_id}.mha", like=volume)

    frame = pd.DataFrame.from_dict(offsets, orient="index", columns=labels).rename_axis("fold")
    pd.read_csv(source_dir / "folds.csv").to_csv(output_dir / "folds.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps({"arm": "R4", "source": str(source_dir),
                                                         "edges_hu": [float(e) for e in edges_hu],
                                                         "hlut": hlut.fingerprint()}, indent=2))
    frame.to_csv(output_dir / "offsets.csv")
    return frame
