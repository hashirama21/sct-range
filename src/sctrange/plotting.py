"""Figures for the notebooks (validated three-slot categorical palette, one y-scale per axis)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .hlut import HLUT

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
LINE_SERIES = SERIES + ("#eda100",)
MARKERS = ("o", "s", "^")
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 9.5, "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "figure.dpi": 110,
    })


def plot_hlut(hlut: HLUT) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (low, high) in zip(axes, ((-1200, 3200), (-200, 450))):
        grid = np.linspace(low, high, 2000)
        ax.plot(grid, hlut.numpy(grid), color=SERIES[0])
        nodes = (hlut.hu >= low) & (hlut.hu <= high)
        ax.plot(hlut.hu[nodes], hlut.spr[nodes], "o", color=SERIES[0], markeredgecolor=SURFACE)
        ax.set_xlabel("HU")
        ax.set_ylabel("SPR (relative to water)")
    axes[0].set_title(f"{hlut.source} lookup table")
    axes[1].set_title("Soft-tissue range")
    axes[1].axhline(1.0, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    axes[1].annotate("water", (0, 1.0), xytext=(-190, 1.008), color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_projector_checks(chord: pd.DataFrame, target: pd.DataFrame, siddon: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, frame, title in ((axes[0], chord, "Disk chords vs analytic"),
                             (axes[1], target, "Entry-to-distal-edge path vs analytic")):
        positions = np.arange(len(frame))
        ax.bar(positions - 0.18, frame["median_abs_mm"], 0.34, color=SERIES[0], label="median")
        ax.bar(positions + 0.18, frame["max_abs_mm"], 0.34, color=SERIES[1], label="max")
        ax.set_xticks(positions, [f"{a:g}°" for a in frame["angle_deg"]])
        ax.set_ylabel("absolute error (mm)")
        ax.set_title(title)
        ax.legend()
    ax = axes[2]
    groups = list(siddon.groupby("angle_deg"))
    if len(groups) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} beams can be distinguished in one panel")
    for index, (angle, group) in enumerate(groups):
        ax.plot(group["siddon_mm"], group["projector_mm"], MARKERS[index], color=SERIES[index],
                linestyle="none", label=f"{angle:g}°", markeredgecolor=SURFACE)
    limits = [siddon[["siddon_mm", "projector_mm"]].min().min(), siddon[["siddon_mm", "projector_mm"]].max().max()]
    ax.plot(limits, limits, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.set_xlabel("pyRadPlan Siddon WET (mm)")
    ax.set_ylabel("projector WET (mm)")
    ax.set_title("Agreement with dose-engine ray tracer")
    ax.legend(title="beam")
    fig.tight_layout()
    return fig


def plot_case(ct: np.ndarray, sct: np.ndarray, mask: np.ndarray, delta_mm: np.ndarray, valid: np.ndarray,
              angles_deg: Sequence[float], slice_index: int, title: str) -> plt.Figure:
    """Reference slice, signed HU error and per-ray WET error for up to three beams."""
    fig = plt.figure(figsize=(12, 3.6))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.4))
    ax_ct, ax_err, ax_wet = (fig.add_subplot(grid[0, i]) for i in range(3))

    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    ax_ct.imshow(ct[slice_index], cmap="gray", vmin=-1000, vmax=1200)
    ax_ct.set_title("Reference CT")
    error = np.where(mask[slice_index] > 0.5, sct[slice_index] - ct[slice_index], np.nan)
    limit = np.nanpercentile(np.abs(error), 99)
    image = ax_err.imshow(error, cmap=DIVERGING, vmin=-limit, vmax=limit)
    ax_err.set_title("sCT − CT (HU)")
    fig.colorbar(image, ax=ax_err, fraction=0.046, pad=0.02)
    for ax in (ax_ct, ax_err):
        ax.axis("off")

    for index, angle in enumerate(list(angles_deg)[:3]):
        rays = np.flatnonzero(valid[index, slice_index])
        if rays.size:
            ax_wet.plot(rays - rays.mean(), delta_mm[index, slice_index, rays], color=SERIES[index],
                        label=f"{angle:g}°")
    ax_wet.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax_wet.set_xlabel("lateral ray position (px, centred)")
    ax_wet.set_ylabel("ΔWET to distal target edge (mm)")
    ax_wet.set_title("Signed range error per ray")
    ax_wet.legend(title="beam")
    fig.tight_layout()
    return fig


def plot_mae_vs_range(frame: pd.DataFrame, y: str = "target_rms_mm") -> plt.Figure:
    groups = list(frame.groupby("source", sort=True))
    if len(groups) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} sources can be distinguished in one panel")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for index, (source, group) in enumerate(groups):
        style = dict(color=SERIES[index], marker=MARKERS[index], linestyle="none",
                     markeredgecolor=SURFACE, label=source)
        axes[0].plot(group["mae_hu"], group[y], **style)
        axes[1].plot(group["target_coherence"], group[y], **style)
    axes[0].set_xlabel("MAE in body (HU)")
    axes[1].set_xlabel("median coherence |Σε| / Σ|ε|")
    for ax in axes:
        ax.set_ylabel("RMS ΔWET to distal edge (mm)")
        ax.legend()
    axes[0].set_title("Image error vs range error")
    axes[1].set_title("Error coherence vs range error")
    fig.tight_layout()
    return fig


def _check_series(count: int, palette: tuple[str, ...]) -> None:
    if count > len(palette):
        raise ValueError(f"at most {len(palette)} series can be distinguished in one panel")


def plot_autocorrelation_law(frame: pd.DataFrame) -> plt.Figure:
    """RMS range error and MAE against effective correlation length at a fixed marginal law."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": (1.6, 1)})
    ell = frame["effective_length_mm"]
    ax = axes[0]
    ax.plot(ell, frame["rms_predicted_mm"], color=SERIES[0], label="Proposition 2")
    ax.plot(ell, frame["rms_observed_mm"], "o", color=SERIES[1], markeredgecolor=SURFACE, label="observed")
    ax.plot(ell, frame["rms_permuted_mm"], "s-", color=SERIES[2], markeredgecolor=SURFACE,
            label="same errors, shuffled")
    ax.set_xscale("log")
    ax.set_xlabel("effective correlation length ℓ along the beam (mm)")
    ax.set_ylabel("RMS ΔWET over 100 mm (mm)")
    ax.set_title("Range error grows with error coherence")
    ax.legend()
    ax = axes[1]
    ax.plot(ell, frame["mae_spr"] * 1e3, "o-", color=SERIES[0], markeredgecolor=SURFACE)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.3 * frame["mae_spr"].max() * 1e3)
    ax.set_xlabel("effective correlation length ℓ (mm)")
    ax.set_ylabel("MAE (10⁻³ SPR)")
    ax.set_title("…at an unchanged MAE")
    fig.tight_layout()
    return fig


def plot_spectral_weights(frame: pd.DataFrame, labels: dict[str, tuple[float, float]],
                          floor: float = 1e-4) -> plt.Figure:
    """Normalised power weight of each error functional against frequency along the beam.

    ``labels`` maps each functional to ``(frequency, vertical offset in points)`` of its direct label.
    """
    order = [name for name in frame["functional"].unique() if name != "range, across beam"]
    _check_series(len(order), LINE_SERIES)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for color, name in zip(LINE_SERIES, order):
        group = frame[frame["functional"] == name]
        theory = np.clip(group["theory"].to_numpy(), floor, None)
        ax.plot(group["frequency"], theory, color=color, linewidth=1.6)
        sample = group.iloc[::12]
        ax.plot(sample["frequency"], np.clip(sample["measured"], floor, None), "o", color=color,
                markersize=4, markeredgecolor=SURFACE)
        frequency, offset = labels[name]
        anchor = int(np.argmin(np.abs(group["frequency"].to_numpy() - frequency)))
        ax.annotate(name, (group["frequency"].iloc[anchor], theory[anchor]), xytext=(0, offset),
                    textcoords="offset points", ha="center", va="center", color=INK, fontsize=8.5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5, "alpha": 0.9})
    ax.set_yscale("log")
    ax.set_ylim(floor, 3.0)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("frequency along the beam (cycles / mm)")
    ax.set_ylabel("normalised power weight")
    ax.set_title("What each error functional sees (lines: theory, dots: measured)")
    fig.tight_layout()
    return fig


def plot_angle_energy(frame: pd.DataFrame, slope: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    f = frame["frequency"].to_numpy()
    ratio = frame["ratio"].to_numpy()
    coefficients = np.polyfit(np.log(f), np.log(ratio), 1)
    ax.plot(f, ratio, "o", color=SERIES[0], markeredgecolor=SURFACE, label="measured")
    ax.plot(f, np.exp(np.polyval(coefficients, np.log(f))), color=SERIES[1], label=f"fit, slope {slope:.2f}")
    ax.plot(f, ratio[0] * (f / f[0]) ** -1.0, color=INK_SECONDARY, linestyle="--", linewidth=1,
            label="slope −1 (Prop. 3.1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("plane-wave frequency (cycles / mm)")
    ax.set_ylabel("angle-summed range energy / voxel energy")
    ax.set_title("Full traversal over all angles weights errors by 1/|k|")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_estimator_bias(detail: pd.DataFrame) -> plt.Figure:
    scenarios = list(detail["scenario"].unique())
    estimators = list(detail["estimator"].unique())
    _check_series(len(estimators), SERIES)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.6 * len(scenarios), 3.8), sharey=False)
    for ax, scenario in zip(np.atleast_1d(axes), scenarios):
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, zorder=1)
        for index, estimator in enumerate(estimators):
            group = detail[(detail["scenario"] == scenario) & (detail["estimator"] == estimator)]
            ax.plot(group["probability_sum"], group["bias_mm"], MARKERS[index], color=SERIES[index],
                    markersize=5.5, markeredgecolor=SURFACE, linestyle="none", label=estimator, zorder=3)
        ax.set_xlabel(f"expected {scenario} voxels along the segment (Σ pᵢ)")
        ax.set_ylabel("range bias (mm)")
        ax.set_title(f"Uncertain {scenario}")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    return fig


def plot_tail_ecdf(errors: dict[float, np.ndarray], level: float = 0.9) -> plt.Figure:
    _check_series(len(errors), LINE_SERIES)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for color, (length, magnitude) in zip(LINE_SERIES, errors.items()):
        ordered = np.sort(magnitude)
        ax.plot(ordered, np.arange(1, ordered.size + 1) / ordered.size, color=color,
                label=f"λ = {length:g} mm")
        threshold = np.quantile(magnitude, level)
        ax.plot([threshold], [level], "o", color=color, markeredgecolor=SURFACE)
    ax.axhline(level, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("|ΔWET| of the Bayes act over 100 mm (mm)")
    ax.set_ylabel("cumulative fraction of rays")
    ax.set_title("Same marginals, same point estimate, different tails")
    ax.legend(title="copula correlation")
    fig.tight_layout()
    return fig


def plot_training_curves(histories: dict[str, pd.DataFrame]) -> plt.Figure:
    """One panel per loss (units differ); one line per fold. Without any history, a single panel says so."""
    if not histories:
        fig, ax = plt.subplots(figsize=(5.2, 1.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "no training history yet", ha="center", va="center", color=INK_SECONDARY)
        return fig
    fig, axes = plt.subplots(1, len(histories), figsize=(5.2 * len(histories), 3.4), squeeze=False)
    for ax, (loss, history) in zip(axes[0], histories.items()):
        for _, fold in history.groupby("fold"):
            ax.plot(fold["step"], fold["loss"], color=SERIES[0], linewidth=1.4, alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel({"l1_hu": "L1 (HU)", "l2_hu": "L2 (HU²)", "l2_spr": "L2 (SPR²)"}.get(loss, "loss"))
        ax.set_title(f"{loss} · one line per fold")
    fig.tight_layout()
    return fig


def plot_beam_spectra(spectra: dict[str, list], cutoff: float) -> plt.Figure:
    """Mean error power relative to shuffled errors, along and across the beam, one line per source."""
    _check_series(len(spectra), SERIES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    for ax, direction in zip(axes, ("along", "across")):
        for color, (source, items) in zip(SERIES, spectra.items()):
            frequency = items[0].frequency
            real = np.mean([getattr(s, f"{direction}_real") / getattr(s, f"{direction}_real").sum() for s in items], axis=0)
            control = np.mean([getattr(s, f"{direction}_control") / getattr(s, f"{direction}_control").sum() for s in items], axis=0)
            ax.plot(frequency, real / control, color=color, label=source)
        ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.8)
        ax.axvspan(0, cutoff, color=GRID, alpha=0.6, linewidth=0)
        ax.set_xscale("symlog", linthresh=frequency[1])
        ax.set_yscale("log")
        ax.set_xlabel("frequency (cycles / mm)")
        ax.set_title(f"{direction.capitalize()} the beam")
    axes[0].set_ylabel("power share relative to shuffled errors")
    axes[0].legend()
    fig.tight_layout()
    return fig


ARM_ROLES = {"R1": "point loss in HU", "R2": "point loss in HU", "R3": "SPR Bayes act", "R4": "SPR Bayes act",
             "R5": "range-aware", "R6": "range-aware"}


def plot_arm_pareto(summary: pd.DataFrame, x: str = "mae_hu", y: str = "target_cvar_mm") -> plt.Figure:
    """Median image error against median range-error tail, one point per arm (plan figure 4).

    ``summary`` is indexed by arm with ``x`` and ``y`` columns and optional ``{x,y}_q25`` / ``{x,y}_q75`` for the
    interquartile whiskers. Colour encodes the role of the arm (three validated slots), the marker separates the two
    arms of a role, and every point carries a direct label, so identity never rests on colour alone.
    """
    roles = list(dict.fromkeys(ARM_ROLES[arm] for arm in summary.index))
    _check_series(len(roles), SERIES)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    seen: dict[str, int] = {}
    for arm, row in summary.iterrows():
        role = ARM_ROLES[arm]
        color = SERIES[roles.index(role)]
        marker = ("o", "s")[seen.get(role, 0) % 2]
        seen[role] = seen.get(role, 0) + 1
        if f"{x}_q25" in summary:
            ax.plot([row[f"{x}_q25"], row[f"{x}_q75"]], [row[y], row[y]], color=color, linewidth=1.2, alpha=0.6)
            ax.plot([row[x], row[x]], [row[f"{y}_q25"], row[f"{y}_q75"]], color=color, linewidth=1.2, alpha=0.6)
        ax.plot(row[x], row[y], marker, color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2,
                linestyle="none", label=role if seen[role] == 1 else None, zorder=3)
        ax.annotate(arm, (row[x], row[y]), xytext=(7, 6), textcoords="offset points", color=INK, fontsize=9,
                    fontweight="bold")
    ax.set_xlabel("median MAE in body (HU)")
    ax.set_ylabel("median CVaR90 |ΔWET| to distal edge (mm)")
    ax.set_title("Image error vs range-error tail per arm (lower left is better)")
    ax.legend(title="arm role")
    fig.tight_layout()
    return fig


def plot_signed_bias(frame: pd.DataFrame, column: str = "target_mean_signed_mm") -> plt.Figure:
    """Mean signed range error per region (bars) and per case (dots), one colour per source."""
    sources = list(frame["source"].unique())
    _check_series(len(sources), SERIES)
    regions = sorted(frame["region"].unique())
    width = 0.8 / len(sources)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for index, (color, source) in enumerate(zip(SERIES, sources)):
        subset = frame[frame["source"] == source]
        positions = np.arange(len(regions)) + (index - (len(sources) - 1) / 2) * width
        means = [subset[subset["region"] == region][column].mean() for region in regions]
        ax.bar(positions, means, width * 0.92, color=color, label=source)
        for position, region in zip(positions, regions):
            values = subset[subset["region"] == region][column]
            ax.plot(np.full(len(values), position), values, "o", color=INK, markersize=3, alpha=0.6)
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_xticks(np.arange(len(regions)), regions)
    ax.set_ylabel("mean signed ΔWET to distal edge (mm)")
    ax.set_title("Signed range bias by region (bars: mean, dots: cases)")
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------- S6 dosimetry figures

DOSE_MAP = LinearSegmentedColormap.from_list("dose", ["#f0efec", "#9ec5f0", "#2a78d6", "#0d3a73"])


def plot_dose_slices(slices: dict[str, np.ndarray], sources: Sequence[str], modality: str = "proton",
                     prescription: float = 60.0) -> plt.Figure:
    """Reference dose on the reference CT, then the dose difference of each source (central target slice)."""
    reference = slices[f"{modality}/reference"]
    sources = [s for s in sources if f"{modality}/{s}" in slices]
    fig, axes = plt.subplots(1, 1 + len(sources), figsize=(3.0 * (1 + len(sources)), 3.2))
    axes = np.atleast_1d(axes)
    axes[0].imshow(slices["ct"], cmap="gray", vmin=-1000, vmax=1200)
    shown = np.ma.masked_less(reference, 0.1 * prescription)
    image = axes[0].imshow(shown, cmap=DOSE_MAP, vmin=0, vmax=1.05 * prescription, alpha=0.85)
    axes[0].contour(slices["target"], levels=[0.5], colors=[INK], linewidths=0.8)
    axes[0].set_title(f"{modality} dose (Gy)")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.02)
    limit = 0.1 * prescription
    for ax, source in zip(axes[1:], sources):
        difference = slices[f"{modality}/{source}"] - reference
        diff_image = ax.imshow(difference, cmap=DIVERGING, vmin=-limit, vmax=limit)
        ax.contour(slices["target"], levels=[0.5], colors=[INK], linewidths=0.8)
        ax.set_title(f"{source} − reference")
    if sources:
        fig.colorbar(diff_image, ax=axes[-1], fraction=0.046, pad=0.02, label="Gy")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_gamma_by_source(frame: pd.DataFrame, order: Sequence[str], criteria: str = "2_2") -> plt.Figure:
    """Gamma pass rate per source: one panel per modality, cases as dots, the median as a bar end."""
    modalities = [m for m in ("proton", "photon") if f"{m}_gamma_{criteria}" in frame]
    fig, axes = plt.subplots(1, len(modalities), figsize=(5.2 * len(modalities), 3.6), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, modality, color in zip(axes[0], modalities, SERIES):
        column = f"{modality}_gamma_{criteria}"
        for position, source in enumerate(order):
            values = frame.loc[frame["source"] == source, column].dropna()
            if values.empty:
                continue
            ax.plot(position + rng.uniform(-0.15, 0.15, len(values)), values, "o", color=color, alpha=0.55,
                    markersize=4, markeredgewidth=0)
            ax.plot([position - 0.3, position + 0.3], [values.median()] * 2, color=INK, linewidth=2)
        ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
        ax.set_ylabel(f"γ {criteria.replace('_', ' %/')} mm pass rate (%)")
        ax.set_title(f"{modality.capitalize()} plans recomputed on each sCT")
    fig.tight_layout()
    return fig


def plot_proxy_validation(cases: pd.DataFrame, rays: pd.DataFrame, x: str = "proxy_rms_mm",
                          y: str = "proton_gamma_fail_2_2") -> plt.Figure:
    """Range proxy against proton dose: case level (left) and ray level (right, density of rays)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    data = cases[[x, y]].dropna()
    axes[0].plot(data[x], data[y], "o", color=SERIES[0], alpha=0.7, markeredgecolor=SURFACE)
    axes[0].set_xlabel("RMS ΔWET to distal edge, plan beams (mm)")
    axes[0].set_ylabel("proton γ 2 %/2 mm failure rate (%)")
    axes[0].set_title("Case level: one dot per case and source")
    if not rays.empty:
        limit = float(np.nanpercentile(np.abs(np.r_[rays["delta_wet_mm"], rays["delta_r80_mm"]]), 99))
        limit = max(limit, 1.0)
        image = axes[1].hexbin(rays["delta_wet_mm"], rays["delta_r80_mm"], gridsize=40, mincnt=1, bins="log",
                               cmap=DOSE_MAP, extent=(-limit, limit, -limit, limit))
        line = np.array([-limit, limit])
        axes[1].plot(line, -line, color=INK_SECONDARY, linewidth=1, linestyle="--")
        axes[1].annotate("ΔR80 = −ΔWET", (0.55 * limit, -0.55 * limit), xytext=(6, 4), textcoords="offset points",
                         color=INK_SECONDARY)
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.02, label="rays (log)")
    axes[1].set_xlabel("ΔWET to distal target edge (mm of water)")
    axes[1].set_ylabel("distal R80 shift (mm)")
    axes[1].set_title("Ray level: proton beams of every case")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------- S7 generative figures


def plot_eta_sweep(sweep: pd.DataFrame, selected: float | None = None) -> plt.Figure:
    """Median projected CRPS, WET-interval coverage and ensemble spread against eta (one panel each)."""
    panels = (("pcrps_mm", "projected CRPS (mm)"), ("coverage90", "90 % interval coverage"),
              ("spread_mm", "ensemble spread of ΔWET (mm)"))
    medians = sweep.groupby("eta")[[p for p, _ in panels]].median()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (column, label) in zip(axes, panels):
        ax.plot(medians.index, medians[column], "o-", color=SERIES[0], markeredgecolor=SURFACE)
        if column == "coverage90":
            ax.axhline(0.9, color=INK_SECONDARY, linestyle=":", linewidth=1)
            ax.annotate("nominal", (medians.index.max(), 0.9), xytext=(-48, 4), textcoords="offset points",
                        color=INK_SECONDARY)
        if selected is not None and selected in medians.index:
            ax.plot(selected, medians.loc[selected, column], "o", color=SERIES[1], markersize=10,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.set_xlabel("η (interpolant noise)")
        ax.set_ylabel(label)
    axes[0].set_title("η sweep (orange: selected)")
    fig.tight_layout()
    return fig


def plot_steps_curve(curve: pd.DataFrame) -> plt.Figure:
    """Quality against the number of sampling steps: image error, range error of the mean, projected CRPS."""
    panels = (("mae_hu", "MAE of the SPR mean (HU)"), ("target_rms_mm", "RMS ΔWET of the mean (mm)"),
              ("pcrps_mm", "projected CRPS (mm)"))
    medians = curve.groupby("n_steps")[[p for p, _ in panels]].median()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (column, label) in zip(axes, panels):
        ax.plot(medians.index, medians[column], "o-", color=SERIES[0], markeredgecolor=SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_xticks(medians.index, [str(int(n)) for n in medians.index])
        ax.set_xlabel("sampling steps")
        ax.set_ylabel(label)
    axes[0].set_title("Quality vs number of steps")
    fig.tight_layout()
    return fig


def plot_pit(counts: pd.DataFrame) -> plt.Figure:
    """Pooled PIT histogram of the true WET among the samples (flat when calibrated)."""
    pooled = counts.sum(axis=0).to_numpy(dtype=float)
    share = pooled / pooled.sum()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(np.arange(len(share)), share, 0.9, color=SERIES[0])
    ax.axhline(1.0 / len(share), color=INK_SECONDARY, linestyle=":", linewidth=1)
    ax.annotate("uniform", (len(share) - 1, 1.0 / len(share)), xytext=(-40, 5), textcoords="offset points",
                color=INK_SECONDARY)
    ax.set_xlabel("rank of the truth among the samples")
    ax.set_ylabel("share of rays")
    ax.set_title("Probability integral transform of ray WET")
    fig.tight_layout()
    return fig


def plot_paired_differences(frame: pd.DataFrame, comparisons: Sequence[tuple[str, str, str, str]]) -> plt.Figure:
    """Per-case differences ``first - second`` of a column, one panel per comparison; below zero favours ``first``.

    ``comparisons``: ``(column, first, second, label)``.
    """
    fig, axes = plt.subplots(1, len(comparisons), figsize=(3.6 * len(comparisons), 3.4), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (column, first, second, label) in zip(axes[0], comparisons):
        wide = frame.pivot_table(index="case_id", columns="source", values=column)
        if first not in wide or second not in wide:
            ax.set_title(f"{label}: no data")
            continue
        difference = (wide[first] - wide[second]).dropna()
        ax.plot(rng.uniform(-0.2, 0.2, len(difference)), difference, "o", color=SERIES[0], alpha=0.6,
                markersize=4, markeredgewidth=0)
        ax.plot([-0.35, 0.35], [difference.median()] * 2, color=INK, linewidth=2)
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        ax.set_xticks([])
        ax.set_ylabel(f"{first} − {second}")
        ax.set_title(label)
    fig.tight_layout()
    return fig


def plot_sample_gallery(ct: np.ndarray, samples: np.ndarray, point: np.ndarray, mask: np.ndarray,
                        slice_index: int) -> plt.Figure:
    """Reference, SPR mean, two samples and the voxel-wise standard deviation of the samples on one slice."""
    inside = mask[slice_index] > 0.5
    panels = [("reference CT", ct[slice_index]), ("SPR mean of samples", point[slice_index]),
              ("sample 1", samples[0, slice_index]), ("sample 2", samples[min(1, len(samples) - 1), slice_index])]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=-1000, vmax=1200)
        ax.set_title(title)
    spread = np.where(inside, samples[:, slice_index].std(axis=0), np.nan)
    image = axes[4].imshow(spread, cmap=DOSE_MAP, vmin=0, vmax=np.nanpercentile(spread, 99) or 1)
    axes[4].set_title("sample std (HU)")
    fig.colorbar(image, ax=axes[4], fraction=0.046, pad=0.02)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
