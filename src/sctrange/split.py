"""Frozen, stratified, hash-verified local test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import CaseRecord, discover_cases

SPLIT_VERSION = 1


@dataclass(frozen=True)
class SplitSpec:
    test_fraction: float = 0.2
    seed: int = 20261013
    min_stratum_size: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1)")


def _stratum_rng(seed: int, stratum: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{stratum}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def stratified_split(records: Iterable[CaseRecord], spec: SplitSpec = SplitSpec()) -> dict:
    """Hold out ``test_fraction`` of every task/region/centre stratum.

    Each stratum draws from its own seed, so adding cases to one stratum never moves cases in
    another. Strata smaller than ``min_stratum_size`` stay entirely in the development pool.
    """
    strata: dict[str, list[str]] = defaultdict(list)
    for record in records:
        strata[record.stratum].append(record.case_id)

    test, development, counts = [], [], {}
    for stratum in sorted(strata):
        ids = sorted(strata[stratum])
        n_test = 0
        if len(ids) >= spec.min_stratum_size:
            n_test = min(len(ids) - 1, max(1, round(spec.test_fraction * len(ids))))
        order = _stratum_rng(spec.seed, stratum).permutation(len(ids))
        held_out = {ids[i] for i in order[:n_test]}
        test.extend(i for i in ids if i in held_out)
        development.extend(i for i in ids if i not in held_out)
        counts[stratum] = {"total": len(ids), "test": n_test}

    return {
        "version": SPLIT_VERSION,
        "spec": asdict(spec),
        "strata": counts,
        "test": sorted(test),
        "development": sorted(development),
    }


def select_cases(records: Sequence[CaseRecord], n_cases: int, seed: int,
                 available: Callable[[CaseRecord], bool] = lambda record: True) -> list[str]:
    """Deterministic region-stratified subset: largest-remainder allocation by region, hashed order within region.

    Used wherever a notebook works on part of the development pool, so that a subset is never just the first case ids
    (which all come from one region and centre).
    """
    pool = [r for r in records if available(r)]
    if not pool:
        raise ValueError("no case available")
    n_cases = min(n_cases, len(pool))
    regions = sorted({r.region for r in pool})
    counts = {g: sum(r.region == g for r in pool) for g in regions}
    quotas = {g: n_cases * counts[g] / len(pool) for g in regions}
    allocation = {g: int(np.floor(q)) for g, q in quotas.items()}
    for g in sorted(regions, key=lambda g: (-(quotas[g] - allocation[g]), g))[: n_cases - sum(allocation.values())]:
        allocation[g] += 1
    chosen = []
    for g in regions:
        ids = sorted(r.case_id for r in pool if r.region == g)
        ranked = sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:dose:{i}".encode()).hexdigest())
        chosen += ranked[: allocation[g]]
    return sorted(chosen)


def write_selection(case_ids: Sequence[str], path: str | Path, meta: dict) -> str:
    """Write a case subset and its SHA-256 (``<path>.sha256``); refuse to change an existing selection."""
    path = Path(path)
    payload = json.dumps({**meta, "case_ids": sorted(case_ids)}, indent=2, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{path} exists with a different selection; delete it explicitly to reselect")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode())  # bytes, so that the hash holds on every platform (no newline translation)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def load_selection(path: str | Path) -> dict:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = path.with_suffix(path.suffix + ".sha256").read_text().strip()
    if digest != expected:
        raise ValueError(f"{path} does not match its recorded hash")
    return json.loads(path.read_text())


def canonical_bytes(split: dict) -> bytes:
    return json.dumps(split, sort_keys=True, separators=(",", ":")).encode()


def split_hash(split: dict) -> str:
    return hashlib.sha256(canonical_bytes(split)).hexdigest()


def write_split(split: dict, path: str | Path) -> str:
    """Write ``<path>`` and ``<path>.sha256``; return the hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = split_hash(split)
    path.write_text(json.dumps(split, sort_keys=True, indent=2) + "\n")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def load_split(path: str | Path) -> dict:
    """Load a split and fail if it does not match its recorded hash."""
    path = Path(path)
    split = json.loads(path.read_text())
    recorded = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    if split_hash(split) != recorded:
        raise ValueError(f"{path} does not match its recorded SHA-256; the frozen split was modified")
    return split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Freeze a stratified local test split.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--test-fraction", type=float, default=SplitSpec.test_fraction)
    parser.add_argument("--seed", type=int, default=SplitSpec.seed)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        parser.error(f"{args.out} exists; a frozen split must not be regenerated (use --force)")
    records = discover_cases(args.data_root)
    if not records:
        parser.error(f"no cases found under {args.data_root}")
    split = stratified_split(records, SplitSpec(args.test_fraction, args.seed))
    digest = write_split(split, args.out)
    print(f"{len(split['test'])} test / {len(split['development'])} development cases")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
