"""Weeks 8-10 helpers: registration overlays, the S10 ablation grid, paired comparisons against a centre run."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CaseRecord, discover_cases, read_volume
from .diagnostics import holm, paired_test


# ------------------------------------------------------------------------------------------- registration overlay


def registration_overlay(original_root: str | Path, alternative_root: str | Path, overlay_root: str | Path,
                         case_ids: Sequence[str]) -> pd.DataFrame:
    """A dataset view whose ``ct.mha`` comes from ``alternative_root`` and every other file from ``original_root``.

    Both roots use the SynthRAD2025 layout (``Task<k>/<region>/<case_id>/``). Files are symbolic links, so the overlay
    costs no space. A case is kept only when the alternative CT exists and shares the grid of the original one (the
    source image and mask are reused unchanged); the returned table says which cases were kept and why the others
    were not.
    """
    original_root, alternative_root, overlay_root = Path(original_root), Path(alternative_root), Path(overlay_root)
    by_id = {r.case_id: r for r in discover_cases(original_root)}
    rows = []
    for case_id in sorted(case_ids):
        record = by_id.get(case_id)
        if record is None:
            rows.append({"case_id": case_id, "kept": False, "reason": "not in the original dataset"})
            continue
        relative = record.directory.relative_to(original_root)
        alternative = alternative_root / relative / "ct.mha"
        if not alternative.exists():
            rows.append({"case_id": case_id, "kept": False, "reason": "no alternative CT"})
            continue
        a, b = read_volume(record.path("ct")), read_volume(alternative)
        if a.array.shape != b.array.shape or not np.allclose(a.spacing, b.spacing, atol=1e-4):
            rows.append({"case_id": case_id, "kept": False, "reason": "different grid"})
            continue
        target = overlay_root / relative
        target.mkdir(parents=True, exist_ok=True)
        for source in record.directory.glob("*.mha"):
            link = target / source.name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink((alternative if source.name == "ct.mha" else source).resolve(), link)
        rows.append({"case_id": case_id, "kept": True, "reason": ""})
    return pd.DataFrame(rows)


def prediction_shim(predictions_dir: str | Path, shim_root: str | Path, arm: str, seed: int = 0) -> Path:
    """Expose a folder of ``<case_id>.mha`` predictions as ``<shim_root>/seed<k>/<arm>/predictions`` (a link), the
    layout that ``sctrange.dosimetry`` reads."""
    target = Path(shim_root) / f"seed{seed}" / arm
    target.mkdir(parents=True, exist_ok=True)
    link = target / "predictions"
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == Path(predictions_dir).resolve():
            return Path(shim_root)
        link.unlink()
    os.symlink(Path(predictions_dir).resolve(), link)
    return Path(shim_root)


# -------------------------------------------------------------------------------------------------- ablations


@dataclass(frozen=True)
class Ablation:
    name: str
    family: str
    arm: str
    overrides: Mapping = field(default_factory=dict)
    center: bool = False


def ablation_grid(raw: Mapping) -> list[Ablation]:
    """The S10 variants of ``configs/s8_s10_preregistration.json`` (centre runs flagged)."""
    grid = []
    for family, spec in raw["families"].items():
        for name, variant in spec["variants"].items():
            overrides = {k: (tuple(v) if isinstance(v, list) else v) for k, v in variant.get("overrides", {}).items()}
            grid.append(Ablation(name, family, variant["arm"], overrides, bool(variant.get("center", False))))
    return grid


def bootstrap_median(values: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
                     ) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.median(values[rng.integers(values.size, size=(n_boot, values.size))], axis=1)
    low, high = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return float(np.median(values)), float(low), float(high)


def compare_family(frame: pd.DataFrame, comparisons: Sequence[Sequence[str]], metrics: Sequence[str],
                   n_boot: int = 2000) -> pd.DataFrame:
    """Paired two-sided Wilcoxon of each ``(variant, reference)`` pair on every metric, Holm within the family, with
    the median paired difference and its bootstrap interval. ``frame`` has one row per case and ``source``."""
    rows = []
    for variant, reference in comparisons:
        for metric in metrics:
            result = paired_test(frame, metric, variant, reference)
            wide = frame.pivot_table(index="case_id", columns="source", values=metric)
            difference = (wide[variant] - wide[reference]).dropna().to_numpy() \
                if {variant, reference} <= set(wide.columns) else np.array([])
            median, low, high = bootstrap_median(difference, n_boot)
            rows.append({"variant": variant, "reference": reference, "metric": metric, "cases": result["cases"],
                         "median_difference": median, "ci_low": low, "ci_high": high, "p_value": result["p_value"]})
    table = pd.DataFrame(rows)
    if not table.empty:
        adjusted = holm({i: p for i, p in table["p_value"].items()})
        table["p_holm"] = pd.Series(adjusted)
    return table


def registration_matrix(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Median ``metric`` by training registration (rows) and test registration (columns)."""
    return frame.pivot_table(index="train", columns="test", values=metric, aggfunc="median")


# ------------------------------------------------------------------------------------------- cached evaluation


def evaluate_predictions(records: Sequence[CaseRecord], sources: Mapping[str, Path], hlut, config,
                         cache_path: str | Path, risk_dirs: Mapping[str, Path] | None = None,
                         log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S4 metrics (``evaluate_case``) of every ``<predictions_dir>/<case_id>.mha`` in ``sources``, and the risk-head
    calibration (``evaluate_risk``) where ``risk_dirs`` has ``<case_id>_{var,excess}.mha``.

    Results are appended to ``cache_path`` (and ``<cache_path stem>_risk.csv``) case by case and reused on rerun, so
    an interrupted evaluation resumes. Missing predictions are skipped.
    """
    from .data import load_case
    from .evaluation import evaluate_case, evaluate_risk

    cache_path = Path(cache_path)
    risk_path = cache_path.with_name(cache_path.stem + "_risk.csv")
    done = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame(columns=["case_id", "source"])
    risk_done = pd.read_csv(risk_path) if risk_path.exists() else pd.DataFrame(columns=["case_id", "source"])
    seen = set(zip(done["case_id"].astype(str), done["source"].astype(str)))
    risk_dirs = dict(risk_dirs or {})
    for record in records:
        pending = [name for name, directory in sources.items()
                   if (record.case_id, name) not in seen and (Path(directory) / f"{record.case_id}.mha").exists()]
        if not pending:
            continue
        volumes = load_case(record, ("ct", "mask"))
        ct, mask, spacing = volumes["ct"].array, volumes["mask"].array, volumes["ct"].spacing
        keys = {"case_id": record.case_id, "region": record.region, "center": record.center}
        rows, risk_rows = [], []
        for name in pending:
            sct = read_volume(Path(sources[name]) / f"{record.case_id}.mha").array
            rows.append({**keys, "source": name, **evaluate_case(ct, mask, sct, spacing, hlut, config)})
            risk = risk_dirs.get(name)
            if risk is not None and (Path(risk) / f"{record.case_id}_var.mha").exists():
                var = read_volume(Path(risk) / f"{record.case_id}_var.mha").array
                excess = read_volume(Path(risk) / f"{record.case_id}_excess.mha").array
                risk_rows.append({**keys, "source": name,
                                  **evaluate_risk(ct, mask, sct, var, excess, spacing, hlut, config)})
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(cache_path, index=False)
        if risk_rows:
            risk_done = pd.concat([risk_done, pd.DataFrame(risk_rows)], ignore_index=True)
            risk_done.to_csv(risk_path, index=False)
        seen.update((record.case_id, name) for name in pending)
        log(f"{record.case_id}: {', '.join(pending)}")
    wanted = set(sources)
    frame = done[done["source"].isin(wanted) & done["case_id"].isin({r.case_id for r in records})]
    risk_frame = risk_done[risk_done["source"].isin(wanted) & risk_done["case_id"].isin({r.case_id for r in records})]
    return frame.reset_index(drop=True), risk_frame.reset_index(drop=True)


def evaluated_sources(cache_path: str | Path, sources: Sequence[str], case_ids: Sequence[str]) -> set[str]:
    """The ``sources`` whose cached evaluation (``evaluate_predictions``) already covers every case of ``case_ids``;
    their runs need no prediction in this session."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return set()
    done = pd.read_csv(cache_path, dtype={"case_id": str})
    wanted = set(case_ids)
    return {s for s in sources if wanted and wanted <= set(done.loc[done["source"] == s, "case_id"])}
