"""On-disk cache of normalised, body-cropped volumes, memory-mapped during training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import CaseRecord, bounding_box, load_case
from .normalization import CT_RANGE_HU, normalize_source

CACHE_VERSION = 2


def source_signature(record: CaseRecord) -> dict[str, int]:
    """Size in bytes of the source files of a case: re-downloading the same release keeps it, other data change it."""
    return {name: record.path(name).stat().st_size for name in ("ct", "mask", record.source_name)}


@dataclass(frozen=True)
class CachedCase:
    """Arrays cropped to the body bounding box; ``box`` locates them in the original ``(Z, Y, X)`` grid."""

    case_id: str
    modality: str
    source: np.ndarray
    mask: np.ndarray
    ct: np.ndarray
    box: tuple[slice, slice, slice]
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]


class VolumeCache:
    """``<root>/<case_id>/{source,mask,ct}.npy`` plus ``meta.json``; source float16, mask uint8, CT int16 HU."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _meta_path(self, case_id: str) -> Path:
        return self.root / case_id / "meta.json"

    def is_current(self, case_id: str, record: CaseRecord | None = None) -> bool:
        """Cached with this cache version and, when ``record`` is given, from these very source files (a case id is not
        enough: another dataset copy or a fixture can reuse it)."""
        path = self._meta_path(case_id)
        if not path.exists():
            return False
        meta = json.loads(path.read_text())
        if meta.get("cache_version") != CACHE_VERSION:
            return False
        return record is None or meta.get("source_files") == source_signature(record)

    def build(self, records: list[CaseRecord], overwrite: bool = False) -> int:
        """Cache every record that is missing or stale; return the number of cases written."""
        written = 0
        for record in records:
            if self.is_current(record.case_id, record) and not overwrite:
                continue
            volumes = load_case(record, ("ct", "mask", record.source_name))
            ct, mask = volumes["ct"], volumes["mask"].array
            box = bounding_box(mask)
            source = normalize_source(volumes[record.source_name].array, mask, record.source_name)
            directory = self.root / record.case_id
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "source.npy", source[box].astype(np.float16))
            np.save(directory / "mask.npy", (mask[box] > 0.5).astype(np.uint8))
            np.save(directory / "ct.npy", np.rint(np.clip(ct.array[box], *CT_RANGE_HU)).astype(np.int16))
            meta = {
                "cache_version": CACHE_VERSION,
                "case_id": record.case_id,
                "modality": record.source_name,
                "box": [[s.start, s.stop] for s in box],
                "shape": list(mask.shape),
                "spacing": list(ct.spacing),
                "origin": list(ct.origin),
                "direction": list(ct.direction),
                "source_files": source_signature(record),
            }
            self._meta_path(record.case_id).write_text(json.dumps(meta, indent=2))
            written += 1
        return written

    def load(self, case_id: str) -> CachedCase:
        if not self.is_current(case_id):
            raise FileNotFoundError(f"{case_id} is not cached under {self.root} (or the cache is stale)")
        meta = json.loads(self._meta_path(case_id).read_text())
        directory = self.root / case_id
        return CachedCase(
            case_id=case_id,
            modality=meta["modality"],
            source=np.load(directory / "source.npy", mmap_mode="r"),
            mask=np.load(directory / "mask.npy", mmap_mode="r"),
            ct=np.load(directory / "ct.npy", mmap_mode="r"),
            box=tuple(slice(start, stop) for start, stop in meta["box"]),
            shape=tuple(meta["shape"]),
            spacing=tuple(meta["spacing"]),
            origin=tuple(meta["origin"]),
            direction=tuple(meta["direction"]),
        )
