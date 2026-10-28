"""Generative arm (S7-S8): conditional flow matching source -> CT with a stochastic interpolant.

Interpolant, in encoded CT units (``sctrange.normalization.encode_ct``), per 2.5D slice:

    x_t = (1 - t) x_0 + t x_1 + gamma(t) z,     gamma(t) = 2 eta t (1 - t),     z ~ N(0, I) inside the body,

with ``x_1`` the reference CT and ``x_0`` the bulk-density image of the body (water inside, air outside), a fixed
function of the input. The network sees the source slices, the body mask, ``x_t`` and ``t``, and predicts the drift
``b = E[x_1 - x_0 + gamma'(t) z | x_t]`` and, when ``eta > 0``, the denoiser ``eta_z = E[z | x_t]``. Sampling
integrates

    dX = [b - (eps(t) / gamma(t)) eta_z] dt + sqrt(2 eps(t)) dW,     eps(t) = kappa * gamma(t),

from ``X_0 = x_0`` with Euler-Maruyama (Albergo, Boffi and Vanden-Eijnden, 2023); the drift correction reduces to
``-kappa * eta_z`` and stays bounded. The injected noise has total variance ``2 kappa eta / 3`` (encoded units);
with the default ``kappa = 0.1`` its standard deviation is about 160 HU at ``eta = 0.1``, the scale of the interpolant
noise itself, so the learned denoiser only has to undo noise of the size it was trained on. With ``eta = 0`` the interpolant and the sampler are deterministic: every sample
is the same image, so the ensemble collapses to a point estimate. With a single step, the sample is
``x_0 + b(0, x_0) = E[x_1 | source]``, the mean in encoded HU (the Bayes act of L2 in HU, not of the range loss).

Slices are sampled independently. Coplanar axial beams integrate within a slice, so the joint law that matters for
in-plane range errors is the in-slice one; samples are not coherent across slices.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from .cache import CachedCase, VolumeCache
from .data import CaseRecord, bounding_box, write_volume
from .download import usable_records
from .evaluation import DiagnosticConfig, evaluate_prepared, prepare_case
from .hlut import HLUT
from .losses import fz_upper_tail_loss
from .metrics import mae_hu, range_error, summarize
from .models import _groups
from .normalization import CT_RANGE_HU, OUTSIDE_HU, decode_ct, encode_ct
from .scoring import crps_ensemble
from .runs import ids_hash
from .training import (PatchSampler, _schedule, assign_folds, autocast_dtype, check_checkpoint, fold_seed,
                       load_progress, progress_key, save_progress, source_stack, uncrop)

_RUNTIME_FIELDS = ("device", "amp", "log_every", "diffusion")  # diffusion only affects sampling


@dataclass(frozen=True)
class GenerativeConfig:
    eta: float = 0.1
    diffusion: float = 0.1
    context: int = 2
    patch: int = 160
    batch_size: int = 8
    steps: int = 30_000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    base_channels: int = 32
    depth: int = 4
    time_dim: int = 128
    grad_clip: float = 1.0
    log_every: int = 100
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    samples_per_case: int = 20_000

    def __post_init__(self) -> None:
        if self.eta < 0 or self.diffusion < 0:
            raise ValueError("eta and diffusion must be non-negative")

    @property
    def in_channels(self) -> int:
        return 2 * self.context + 3

    @property
    def out_channels(self) -> int:
        return 2 if self.eta > 0 else 1

    @property
    def uses_projector(self) -> bool:  # PatchSampler interface
        return False

    def comparable(self) -> dict:
        return json.loads(json.dumps({k: v for k, v in asdict(self).items() if k not in _RUNTIME_FIELDS}))


def gamma(t: torch.Tensor, eta: float) -> torch.Tensor:
    return 2.0 * eta * t * (1.0 - t)


def gamma_dot(t: torch.Tensor, eta: float) -> torch.Tensor:
    return 2.0 * eta * (1.0 - 2.0 * t)


def base_image(mask: torch.Tensor) -> torch.Tensor:
    """``x_0``: water inside the body, air outside, encoded."""
    return encode_ct(torch.where(mask > 0.5, torch.zeros_like(mask), torch.full_like(mask, OUTSIDE_HU)))


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, z: torch.Tensor, eta: float
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x_t, drift target)`` for per-sample times ``t`` of shape ``(B,)``."""
    tt = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1 - tt) * x0 + tt * x1 + gamma(tt, eta) * z, x1 - x0 + gamma_dot(tt, eta) * z


# ------------------------------------------------------------------------------------------------------ model


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angles = 1000.0 * t.float()[:, None] * frequencies[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class FiLMBlock(nn.Module):
    """The ``ConvBlock`` of ``sctrange.models`` with a time-dependent scale and shift after the second norm."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.film = nn.Linear(time_dim, 2 * out_channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.film(embedding)[:, :, None, None].chunk(2, dim=1)
        return F.silu(self.norm2(self.conv2(x)) * (1 + scale) + shift)


class TimeUNet2d(nn.Module):
    """The 2.5D U-Net of the regression arms (same widths and depth), conditioned on ``t`` by FiLM."""

    def __init__(self, in_channels: int, base_channels: int = 32, depth: int = 4, out_channels: int = 1,
                 time_dim: int = 128) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        widths = [base_channels * 2**level for level in range(depth)]
        self.depth, self.time_dim = depth, time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.encoders = nn.ModuleList([FiLMBlock(in_channels, widths[0], time_dim)])
        self.encoders.extend(FiLMBlock(widths[i - 1], widths[i], time_dim) for i in range(1, depth))
        self.decoders = nn.ModuleList(FiLMBlock(widths[i] + widths[i - 1], widths[i - 1], time_dim)
                                      for i in range(depth - 1, 0, -1))
        self.head = nn.Conv2d(widths[0], out_channels, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def stride(self) -> int:
        return 2 ** (self.depth - 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embedding = self.time_mlp(time_embedding(t, self.time_dim))
        skips = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x if index == 0 else F.max_pool2d(x, 2), embedding)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1), embedding)
        return self.head(x)


def build_generator(config: GenerativeConfig) -> TimeUNet2d:
    return TimeUNet2d(config.in_channels, config.base_channels, config.depth, config.out_channels, config.time_dim)


# --------------------------------------------------------------------------------------------------- training


def flow_matching_loss(model: TimeUNet2d, inputs: torch.Tensor, targets_hu: torch.Tensor, masks: torch.Tensor,
                       config: GenerativeConfig, generator: torch.Generator | None = None
                       ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Drift regression (plus denoiser regression when ``eta > 0``), averaged over body pixels."""
    batch = inputs.shape[0]
    x1 = encode_ct(targets_hu)
    x0 = base_image(masks)
    t = torch.rand(batch, device=inputs.device, generator=generator)
    z = torch.randn(x1.shape, device=inputs.device, generator=generator) * masks
    xt, drift = interpolate(x0, x1, t, z, config.eta)
    output = model(torch.cat([inputs, xt], dim=1), t).float()
    weight = masks / masks.sum().clamp_min(1.0)
    drift_loss = ((output[:, :1] - drift) ** 2 * weight).sum()
    parts = {"drift": drift_loss.detach()}
    loss = drift_loss
    if config.eta > 0:
        denoise_loss = ((output[:, 1:2] - z) ** 2 * weight).sum()
        parts["denoiser"] = denoise_loss.detach()
        loss = loss + denoise_loss
    return loss, parts


def train_generative(cases: list[CachedCase], config: GenerativeConfig, resume_path: str | Path | None = None,
                     checkpoint_every: int = 1000) -> tuple[TimeUNet2d, pd.DataFrame]:
    """Train one generator; with ``resume_path`` the training state is saved every ``checkpoint_every`` steps and an
    interrupted job continues from its last save (``sctrange.training.train_model``)."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(config.seed + 7)
    model = build_generator(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(step, config))
    sampler = PatchSampler(cases, config, rng)
    amp_dtype = autocast_dtype(device) if config.amp else None
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    resume_path = Path(resume_path) if resume_path is not None else None
    key = progress_key(config, [case.case_id for case in cases])
    modules = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler}
    first_step, history = load_progress(resume_path, key, modules, {"sampler": rng}, {"noise": generator})
    running = defaultdict(list)
    start = time.perf_counter() - (history[-1]["seconds"] if history else 0.0)
    model.train()
    for step in range(first_step, config.steps):
        inputs, targets, masks = (tensor.to(device, non_blocking=True) for tensor in sampler.batch())
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            loss, parts = flow_matching_loss(model, inputs, targets, masks, config, generator)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running["loss"].append(loss.detach())
        for name, value in parts.items():
            running[name].append(value)
        if (step + 1) % config.log_every == 0 or step + 1 == config.steps:
            row = {name: float(torch.stack(values).float().mean()) for name, values in running.items()}
            history.append({"step": step + 1, **row, "learning_rate": scheduler.get_last_lr()[0],
                            "seconds": time.perf_counter() - start})
            running = defaultdict(list)
        if resume_path is not None and not running and (step + 1) % checkpoint_every == 0 and step + 1 < config.steps:
            save_progress(resume_path, step + 1, key, modules, history, {"sampler": rng}, {"noise": generator})
    return model.eval(), pd.DataFrame(history)


def _load_generator(path: Path, config: GenerativeConfig, training_ids: Sequence[str] | None = None) -> TimeUNet2d:
    """Generator from a fold checkpoint; refuses another configuration or another training set."""
    state = torch.load(path, map_location=config.device, weights_only=True)
    check_checkpoint(state, path, {k: v for k, v in config.comparable().items() if k != "seed"}, training_ids)
    model = build_generator(config).to(config.device)
    model.load_state_dict(state["state_dict"])
    return model.eval()


def train_generative_fold(cache: VolumeCache, case_ids: list[str], config: GenerativeConfig, fold: int,
                          checkpoint: str | Path, device: str) -> pd.DataFrame:
    checkpoint = Path(checkpoint)
    partial = checkpoint.with_suffix(".partial.pt")
    cases = [cache.load(case_id) for case_id in case_ids]
    model, history = train_generative(cases, replace(config, seed=fold_seed(config.seed, fold), device=device),
                                      resume_path=partial)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"state_dict": model.state_dict(), "config": asdict(config), "fold": fold,
                "training_cases": ids_hash(case_ids)}, temporary)
    temporary.replace(checkpoint)
    partial.unlink(missing_ok=True)
    return history.assign(fold=fold, device=device)


def run_name(config: GenerativeConfig) -> str:
    return f"eta{config.eta:g}".replace(".", "p")


def crossfit_generative(records: list[CaseRecord], cache: VolumeCache, config: GenerativeConfig, n_folds: int,
                        output_dir: str | Path, devices: Sequence[str] | None = None) -> pd.DataFrame:
    """Train one generator per fold on the other folds (checkpoints reused and checked); returns the folds.

    Layout: ``<output_dir>/fold<k>.pt``, ``folds.csv``, ``history.csv``, ``config.json``. Samples are drawn at
    evaluation time (``evaluate_generative``), where the number of samples and of steps are chosen.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = list(devices or [config.device])
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    folds = assign_folds(records, n_folds, config.seed)
    frame = pd.DataFrame(sorted(folds.items()), columns=["case_id", "fold"])
    frame.to_csv(output_dir / "folds.csv", index=False)
    history_path = output_dir / "history.csv"

    def training_ids(fold: int) -> list[str]:
        return [r.case_id for r in records if folds[r.case_id] != fold]

    def job(fold: int, device: str) -> tuple:
        return cache, training_ids(fold), config, fold, output_dir / f"fold{fold}.pt", device

    def append(history: pd.DataFrame) -> None:
        previous = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        pd.concat([previous, history], ignore_index=True).to_csv(history_path, index=False)

    pending = []
    for fold in range(n_folds):
        checkpoint = output_dir / f"fold{fold}.pt"
        if checkpoint.exists():
            _load_generator(checkpoint, replace(config, device="cpu"), training_ids(fold))
        else:
            pending.append(fold)
    if len(devices) > 1 and len(pending) > 1:
        context = multiprocessing.get_context("spawn")
        for start in range(0, len(pending), len(devices)):
            wave = pending[start:start + len(devices)]
            with ProcessPoolExecutor(max_workers=len(wave), mp_context=context) as pool:
                futures = [pool.submit(train_generative_fold, *job(f, d)) for f, d in zip(wave, devices)]
                for future in futures:
                    append(future.result())
    else:
        for fold in pending:
            append(train_generative_fold(*job(fold, devices[0])))
    return frame


# --------------------------------------------------------------------------------------------------- sampling


@torch.no_grad()
def sample_slices(model: TimeUNet2d, inputs: torch.Tensor, masks: torch.Tensor, n_steps: int, eta: float,
                  diffusion: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Euler(-Maruyama) samples in encoded units, ``(N, 1, H, W)``; air outside the body."""
    x0 = base_image(masks)
    x = x0.clone()
    dt = 1.0 / n_steps
    stochastic = eta > 0 and diffusion > 0 and model.head.out_channels > 1
    for k in range(n_steps):
        t = torch.full((x.shape[0],), k * dt, device=x.device)
        output = model(torch.cat([inputs, x], dim=1), t).float()
        drift = output[:, :1]
        g = float(gamma(torch.tensor(k * dt), eta))
        if stochastic and g > 0:
            drift = drift - diffusion * output[:, 1:2]
        x = x + drift * dt
        if stochastic and g > 0 and k < n_steps - 1:
            noise = torch.randn(x.shape, device=x.device, generator=generator)
            x = x + math.sqrt(2.0 * diffusion * g * dt) * noise * masks
    return torch.where(masks > 0.5, x, x0)


@torch.no_grad()
def sample_case(model: TimeUNet2d, case: CachedCase, config: GenerativeConfig, n_samples: int, n_steps: int,
                device: str, seed: int = 0, slice_batch: int = 4) -> np.ndarray:
    """``(K, Z, Y, X)`` samples in HU on the cropped grid of ``case``; ``OUTSIDE_HU`` outside the body."""
    stride = model.stride
    depth, height, width = case.mask.shape
    pad_h, pad_w = (-height) % stride, (-width) % stride
    generator = torch.Generator(device=device).manual_seed(seed)
    out = np.empty((n_samples, depth, height, width), dtype=np.float32)
    amp_dtype = autocast_dtype(torch.device(device)) if config.amp else None
    for start in range(0, depth, slice_batch):
        zs = range(start, min(start + slice_batch, depth))
        stacks = torch.from_numpy(np.stack([source_stack(case.source, case.mask, z, config.context) for z in zs]))
        stacks = F.pad(stacks, (0, pad_w, 0, pad_h)).to(device)
        masks = stacks[:, -1:].repeat(n_samples, 1, 1, 1)
        inputs = stacks.repeat(n_samples, 1, 1, 1)
        with torch.autocast(torch.device(device).type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            samples = sample_slices(model, inputs, masks, n_steps, config.eta, config.diffusion, generator)
        hu = decode_ct(samples.float()).clamp(*CT_RANGE_HU)[:, 0, :height, :width]
        out[:, start:start + len(zs)] = hu.view(n_samples, len(zs), height, width).cpu().numpy()
    inside = np.asarray(case.mask) > 0
    return np.where(inside[None], out, OUTSIDE_HU).astype(np.float32)


def spr_mean(samples_hu: np.ndarray, hlut: HLUT) -> np.ndarray:
    """``g^-1(mean_k g(sample_k))``: the Bayes act of the squared range loss estimated from samples (Proposition 4)."""
    return hlut.inverse(np.mean(hlut.numpy(samples_hu), axis=0)).astype(np.float32)


# ------------------------------------------------------------------------------------------------- evaluation


def _spearman(x: np.ndarray, y: np.ndarray, tolerance: float = 1e-9) -> float:
    """Spearman correlation, NaN when either input is constant (e.g. a collapsed ensemble)."""
    if np.ptp(x) <= tolerance or np.ptp(y) <= tolerance:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def ensemble_metrics(members: np.ndarray, point: np.ndarray, level: float = 0.9, coverage: float = 0.9,
                     floor_mm: float = 1e-2, rng: np.random.Generator | None = None) -> tuple[dict[str, float], np.ndarray]:
    """Projected scores of an ensemble of ray WET errors.

    ``members`` ``(K, R)`` are the WET errors ``W(sample_k) - W(CT)`` of the rays, ``point`` ``(R,)`` the WET error of
    the point prediction (the SPR mean of the samples). The truth is 0 in these coordinates. Returns per-case scores and
    the randomised PIT rank of the truth on every ray (integers in ``[0, K]``).
    """
    rng = rng or np.random.default_rng(0)
    k = members.shape[0]
    crps = crps_ensemble(members, np.zeros(members.shape[1]))
    low, high = np.quantile(members, [(1 - coverage) / 2, (1 + coverage) / 2], axis=0)
    spread = members.std(axis=0, ddof=1) if k > 1 else np.zeros(members.shape[1])
    below, ties = (members < 0).sum(axis=0), (members == 0).sum(axis=0)
    ranks = below + np.floor(rng.random(members.shape[1]) * (ties + 1)).astype(int)
    outcome = np.abs(point)
    predicted = np.abs(point[None] - members)
    var_raw = np.quantile(predicted, level, axis=0)
    in_tail = predicted >= var_raw[None]
    es_raw = np.where(in_tail, predicted, 0.0).sum(axis=0) / in_tail.sum(axis=0)
    var, es = var_raw + floor_mm, np.maximum(es_raw, var_raw) + floor_mm
    exceed = outcome >= var
    as_tensor = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    row = {
        "ens_rays": int(members.shape[1]),
        "pcrps_mm": float(crps.mean()),
        "pcrps_point_mm": float(outcome.mean()),
        "coverage90": float(((low <= 0) & (0 <= high)).mean()),
        "spread_mm": float(spread.mean()),
        "unc_err_spearman": _spearman(spread, outcome),
        "risk_rays": int(members.shape[1]),
        "risk_coverage": float((outcome < var).mean()),
        "risk_mean_var_mm": float(var.mean()),
        "risk_mean_es_mm": float(es[exceed].mean()) if exceed.any() else float("nan"),
        "risk_realised_es_mm": float(outcome[exceed].mean()) if exceed.any() else float("nan"),
        "risk_fz": float(fz_upper_tail_loss(as_tensor(var), as_tensor(es), as_tensor(outcome), level)),
        "risk_spearman": _spearman(var, outcome),
    }
    return row, ranks


def evaluate_ensemble(ct: np.ndarray, mask: np.ndarray, samples: np.ndarray, spacing_xyz: Sequence[float],
                      hlut: HLUT, config: DiagnosticConfig = DiagnosticConfig(), level: float = 0.9,
                      target: np.ndarray | None = None, full: bool = True) -> tuple[dict[str, float], np.ndarray]:
    """Point metrics of the SPR mean of the samples (the S4 columns), plus projected ensemble scores on the
    entry-to-distal-target segments. ``full=False`` skips the permutation controls of the point metrics."""
    point = spr_mean(samples, hlut)
    case = prepare_case(ct, mask, point, spacing_xyz, config, target)
    point_result = range_error(case.sct, case.ct, case.mask, case.projector, hlut, case.target, config.slab_slices)
    if full:
        row = evaluate_prepared(case, hlut, config)
    else:
        row = {"mae_hu": mae_hu(case.sct, case.ct, case.mask),
               **summarize(point_result, config.alpha, config.min_path_mm).as_dict("target_")}
    inside = mask > 0.5
    row["median_mae_hu"] = float(np.abs(np.median(samples, axis=0) - ct)[inside].mean())
    row["sample_mae_hu"] = float(np.mean([np.abs(s - ct)[inside].mean() for s in samples]))
    selected = point_result.selected(config.min_path_mm)
    box = bounding_box(mask)
    members = []
    for sample in samples:
        sct = torch.as_tensor(np.ascontiguousarray(sample[box]), dtype=torch.float32, device=case.ct.device)
        result = range_error(sct, case.ct, case.mask, case.projector, hlut, case.target, config.slab_slices)
        members.append(result.delta_mm[selected].double().cpu().numpy())
    scores, ranks = ensemble_metrics(np.stack(members), point_result.delta_mm[selected].double().cpu().numpy(), level)
    row.update(scores)
    return row, ranks


def evaluate_generative(records: Sequence[CaseRecord], cache: VolumeCache, run_dir: str | Path,
                        config: GenerativeConfig, hlut: HLUT, diagnostic: DiagnosticConfig, n_samples: int,
                        n_steps: int, device: str, seed: int = 0, write_mean: bool = False, full: bool = True,
                        log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample every record with the model of its held-out fold and evaluate; results are cached per case under
    ``<run_dir>/eval/K<k>_N<n>/`` and reused on rerun. Returns per-case rows and PIT rank counts."""
    run_dir = Path(run_dir)
    tag = f"K{n_samples}_N{n_steps}_kappa{config.diffusion:g}_seed{seed}{'' if full else '_fast'}"
    eval_dir = run_dir / "eval" / tag
    eval_dir.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(run_dir / "folds.csv", dtype={"case_id": str}).set_index("case_id")["fold"].to_dict()
    models: dict[int, TimeUNet2d] = {}
    rows, pits = [], []
    for record in records:
        path = eval_dir / f"{record.case_id}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            rows.append(saved["row"])
            pits.append(pd.Series(saved["pit"], name=record.case_id))
            continue
        fold = folds[record.case_id]
        if fold not in models:
            training = [case_id for case_id, k in folds.items() if k != fold]
            models[fold] = _load_generator(run_dir / f"fold{fold}.pt", replace(config, device=device), training)
        case = cache.load(record.case_id)
        case_seed = seed + int.from_bytes(record.case_id.encode()[-4:], "little")
        samples = sample_case(models[fold], case, config, n_samples, n_steps, device, case_seed)
        ct, mask = np.asarray(case.ct, dtype=np.float32), np.asarray(case.mask, dtype=np.float32)
        row, ranks = evaluate_ensemble(ct, mask, samples, case.spacing, hlut, diagnostic, full=full)
        row = {"case_id": record.case_id, "region": record.region, "center": record.center, "eta": config.eta,
               "n_samples": n_samples, "n_steps": n_steps, **row}
        pit = np.bincount(ranks, minlength=n_samples + 1).tolist()
        path.write_text(json.dumps({"row": row, "pit": pit}, default=float))
        if write_mean:
            (run_dir / "predictions").mkdir(exist_ok=True)
            mean = uncrop(case, spr_mean(samples, hlut))
            hu = np.clip(np.rint(mean), *CT_RANGE_HU).astype(np.int16)
            write_volume(hu, run_dir / "predictions" / f"{record.case_id}.mha", spacing=case.spacing,
                         origin=case.origin, direction=case.direction)
        rows.append(row)
        pits.append(pd.Series(pit, name=record.case_id))
        log(f"{record.case_id}: eta {config.eta:g}, K {n_samples}, N {n_steps}")
    return pd.DataFrame(rows), pd.DataFrame(pits)


def select_eta(sweep: pd.DataFrame, metric: str = "pcrps_mm") -> float:
    """Pre-registered choice among the stochastic generators (eta > 0): the lowest median ``metric``, ties to the
    smaller eta. ``eta = 0`` is deterministic, so it is the collapse control, never a candidate."""
    medians = sweep[sweep["eta"] > 0].groupby("eta")[metric].median()
    if medians.empty:
        raise ValueError("the sweep has no generator with eta > 0")
    best = medians.min()
    return float(min(eta for eta, value in medians.items() if np.isclose(value, best)))


def collapse_ratio(sweep: pd.DataFrame) -> float:
    """Median ensemble spread at ``eta = 0`` over the smallest median spread among ``eta > 0`` (NaN if undefined)."""
    spread = sweep.groupby("eta")["spread_mm"].median()
    stochastic = spread[spread.index > 0]
    if 0.0 not in spread.index or stochastic.empty or stochastic.min() <= 0:
        return float("nan")
    return float(spread[0.0] / stochastic.min())


# ------------------------------------------------------------------------------------------- decision and CLI


def comparison_frame(generative: pd.DataFrame, reference: pd.DataFrame, risk: pd.DataFrame | None = None,
                     generative_name: str = "G") -> pd.DataFrame:
    """One row per case and source (``G``, and the S4 arms in ``reference``) with the columns the S7 tests use.

    A point prediction is a degenerate ensemble, so its projected CRPS is its mean absolute WET error.
    """
    columns = ["case_id", "region", "source", "mae_hu", "target_rms_mm", "target_mean_signed_mm", "target_mean_abs_mm",
               "target_cvar_mm", "pcrps_mm", "risk_fz", "risk_coverage", "risk_spearman"]
    gen = generative.assign(source=generative_name)
    ref = reference.assign(pcrps_mm=reference["target_mean_abs_mm"])
    if risk is not None and not risk.empty:
        keep = ["case_id", "source", "risk_fz", "risk_coverage", "risk_spearman"]
        ref = ref.drop(columns=[c for c in keep[2:] if c in ref]).merge(risk[keep], on=["case_id", "source"], how="left")
    frame = pd.concat([gen, ref], ignore_index=True)
    frame = frame.reindex(columns=[c for c in columns if c in frame.columns])
    frame["risk_miscalibration"] = (frame["risk_coverage"] - 0.9).abs() if "risk_coverage" in frame else np.nan
    return frame


def decide_generative(frame: pd.DataFrame, prereg: dict, sweep: pd.DataFrame | None = None) -> dict[str, object]:
    """Pre-registered S7 tests (``configs/s7_preregistration.json``), Holm-adjusted over the family."""
    from .diagnostics import holm, paired_test

    alpha = float(prereg["alpha"])
    tests, details = {}, {}
    for name, spec in prereg["tests"].items():
        result = paired_test(frame, spec["column"], spec["first"], spec["second"], alternative=spec["alternative"])
        tests[name] = result["p_value"]
        details[name] = {**spec, **result}
    adjusted = holm(tests)
    for name in details:
        details[name]["p_holm"] = adjusted[name]
        details[name]["significant"] = bool(np.isfinite(adjusted[name]) and adjusted[name] < alpha)
    summary: dict[str, object] = {"tests": details}
    if sweep is not None and not sweep.empty:
        summary["selected_eta"] = select_eta(sweep, prereg["sweep"]["selection_metric"])
        ratio = collapse_ratio(sweep)
        summary["collapse_ratio"] = ratio
        summary["eta0_collapses"] = bool(np.isfinite(ratio) and ratio <= prereg["collapse"]["max_spread_ratio"])
    gen = frame[frame["source"] == prereg["generative_name"]]
    if not gen.empty and "unc_err_spearman" in gen:
        summary["median_unc_err_spearman"] = float(gen["unc_err_spearman"].median())
    expectations = prereg["expectations"]
    summary["theory_checks"] = {
        name: (details[name]["significant"] == expected) if np.isfinite(details[name]["p_value"]) else None
        for name, expected in expectations.items() if name in details
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .data import discover_cases
    from .split import load_split
    from .training import visible_devices

    parser = argparse.ArgumentParser(description="S7: cross-fitted stochastic-interpolant generators (one per eta).")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="writes <out>/eta<value>/")
    parser.add_argument("--eta", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--task", choices=("1", "2"), default="1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=GenerativeConfig.steps)
    parser.add_argument("--batch-size", type=int, default=GenerativeConfig.batch_size)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--devices", help="comma-separated devices (default: every visible GPU)")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)
    split = load_split(args.split)
    development = set(split["development"])
    records = [r for r in discover_cases(args.data_root) if r.case_id in development and r.task == args.task]
    records, excluded = usable_records(args.data_root, records)
    for case_id, problems in excluded.items():
        print(f"excluded {case_id}: {problems}")
    if not records:
        parser.error("no development cases found for this task")
    devices = args.devices.split(",") if args.devices else visible_devices()
    cache = VolumeCache(args.cache_dir)
    print(f"caching {cache.build(records)} new cases · training on {', '.join(devices)}")
    for eta in args.eta:
        config = GenerativeConfig(eta=eta, steps=args.steps, batch_size=args.batch_size, seed=args.seed,
                                  device=devices[0], amp=not args.no_amp)
        crossfit_generative(records, cache, config, args.folds, args.out / run_name(config), devices)
        print(f"eta {eta:g}: checkpoints in {args.out / run_name(config)}")


if __name__ == "__main__":
    main()
