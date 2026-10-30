"""Second domain (S9): sparse-view parallel-beam CT, where the quantity of interest is again a line integral.

Images are relative linear attenuation ``u = mu / mu_water`` (water 1, air 0; ``u = 1 + HU / 1000`` in the body),
built from the reference CTs of SynthRAD2025 resampled to ``pixel_mm`` and centred on a square field of view. A
parallel-beam scanner measures ``p_theta = int u dl`` in millimetres of water at ``sparse_views`` angles over 180
degrees, with Poisson noise of ``photons`` incident quanta per detector bin. Filtered back-projection (ramp filter,
exact adjoint of the projector) gives a streaky image that a post-processing U-Net (FBPConvNet-style residual) cleans.

Three objectives at a matched budget:

- ``l1``      |u_hat - u| over the body                            Bayes act: voxel-wise median
- ``l2``      (u_hat - u)^2 over the body                          Bayes act: voxel-wise mean
- ``l2_proj`` ``l2`` + w * mean (P u_hat - P u)^2 at random angles  same Bayes act as ``l2``

Here ``g`` is the identity, so Proposition 4 makes the voxel-wise mean the Bayes act for squared line-integral loss
at every angle: unlike protons, any gain of ``l2_proj`` cannot come from the estimator and must come from the
spectral weighting of Result A (finite capacity spends itself where the loss looks). Evaluation uses line integrals at
``eval_views`` angles that the scanner never measured, in millimetres of water, the unit of a proton range error.
"""

from __future__ import annotations

import hashlib
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
from torch.nn import functional as F

from .cache import CachedCase, VolumeCache
from .data import CaseRecord
from .models import UNet2d
from .projection import ParallelBeamProjector
from .runs import ids_hash
from .training import (_schedule, assign_folds, autocast_dtype, check_checkpoint, fold_seed, load_progress,
                       progress_key, save_progress)

LOSSES = ("l1", "l2", "l2_proj")
_RUNTIME_FIELDS = ("device", "amp", "log_every")


@dataclass(frozen=True)
class TomographyConfig:
    loss: str = "l2"
    pixel_mm: float = 2.0
    size: int = 256
    sparse_views: int = 30
    eval_views: int = 180
    photons: float = 1e5
    projection_weight: float = 1e-4
    projection_angles: int = 8
    min_path_mm: float = 20.0
    batch_size: int = 8
    steps: int = 20_000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    base_channels: int = 32
    depth: int = 4
    grad_clip: float = 1.0
    log_every: int = 100
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if self.size % 2 ** (self.depth - 1):
            raise ValueError("size must be divisible by the U-Net stride")

    @property
    def sparse_angles(self) -> tuple[float, ...]:
        return tuple(k * 180.0 / self.sparse_views for k in range(self.sparse_views))

    @property
    def eval_angles(self) -> tuple[float, ...]:
        """Interleaved with no measured angle when ``eval_views`` is a multiple of ``sparse_views``."""
        return tuple((k + 0.5) * 180.0 / self.eval_views for k in range(self.eval_views))

    def comparable(self) -> dict:
        return json.loads(json.dumps({k: v for k, v in asdict(self).items() if k not in _RUNTIME_FIELDS}))


# ----------------------------------------------------------------------------------------------------- images


def attenuation_slice(ct_hu: np.ndarray, mask: np.ndarray, spacing_mm: float, pixel_mm: float, size: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """``(u, body)`` of one axial slice: block-averaged to ``pixel_mm``, centred on a ``size`` x ``size`` field."""
    factor = max(1, int(round(pixel_mm / spacing_mm)))
    u = np.where(mask > 0.5, np.clip(1.0 + ct_hu / 1000.0, 0.0, None), 0.0).astype(np.float32)
    body = (mask > 0.5).astype(np.float32)
    height, width = u.shape
    pad_h, pad_w = (-height) % factor, (-width) % factor
    u, body = (np.pad(a, ((0, pad_h), (0, pad_w))) for a in (u, body))
    shape = (u.shape[0] // factor, factor, u.shape[1] // factor, factor)
    u, body = u.reshape(shape).mean(axis=(1, 3)), body.reshape(shape).mean(axis=(1, 3)) > 0.5

    def centre(image: np.ndarray) -> np.ndarray:
        out = np.zeros((size, size), dtype=np.float32)
        h, w = image.shape
        src_y, src_x = max(0, (h - size) // 2), max(0, (w - size) // 2)
        dst_y, dst_x = max(0, (size - h) // 2), max(0, (size - w) // 2)
        ch, cw = min(h, size), min(w, size)
        out[dst_y:dst_y + ch, dst_x:dst_x + cw] = image[src_y:src_y + ch, src_x:src_x + cw]
        return out

    return centre(u), centre(body.astype(np.float32))


def case_slices(case: CachedCase, config: TomographyConfig, indices: Sequence[int] | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(u, body)`` stacks ``(Z, size, size)`` of a cached case (all slices, or ``indices``)."""
    indices = range(case.mask.shape[0]) if indices is None else indices
    pairs = [attenuation_slice(np.asarray(case.ct[z], dtype=np.float32), np.asarray(case.mask[z]),
                               case.spacing[0], config.pixel_mm, config.size) for z in indices]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


def central_indices(case: CachedCase, n_slices: int) -> list[int]:
    areas = np.asarray(case.mask).reshape(case.mask.shape[0], -1).sum(axis=1)
    centre = int(np.argmax(areas))
    start = int(np.clip(centre - n_slices // 2, 0, max(case.mask.shape[0] - n_slices, 0)))
    return list(range(start, min(start + n_slices, case.mask.shape[0])))


# ---------------------------------------------------------------------------------------------------- scanner


class SparseViewScanner:
    """Parallel-beam projection, Poisson noise and filtered back-projection on a ``size`` x ``size`` field."""

    MU_WATER_PER_MM = 0.0192  # about 70 keV

    def __init__(self, size: int, pixel_mm: float, angles_deg: Sequence[float], photons: float,
                 device: torch.device | str = "cpu") -> None:
        self.size, self.pixel_mm, self.photons = size, pixel_mm, photons
        self.angles = tuple(angles_deg)
        self.projector = ParallelBeamProjector((size, size), self.angles, pixel_mm, device=device)
        n = self.projector.side
        self.padded = 2 ** math.ceil(math.log2(2 * n))
        # Ram-Lak filter from its sampled spatial kernel (Kak and Slaney, eq. 3.61): no DC offset, unlike |f|
        offsets = np.fft.fftfreq(self.padded, d=1.0 / self.padded).astype(np.int64)
        kernel = np.zeros(self.padded)
        kernel[offsets == 0] = 1.0 / (4.0 * pixel_mm**2)
        odd = offsets % 2 == 1
        kernel[odd] = -1.0 / (np.pi * offsets[odd] * pixel_mm) ** 2
        self.ramp = torch.as_tensor(np.fft.rfft(kernel).real * pixel_mm, dtype=torch.float32, device=device)

    def project(self, u: torch.Tensor) -> torch.Tensor:
        """``(B, H, W)`` -> line integrals ``(A, B, S)`` in mm of water."""
        return self.pixel_mm * self.projector.rotate(u).sum(dim=-2)

    def measure(self, u: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Noisy line integrals from Poisson transmission counts (``photons`` per bin, at least one count)."""
        clean = self.project(u)
        if not self.photons:
            return clean
        expected = self.photons * torch.exp(-self.MU_WATER_PER_MM * clean)
        counts = torch.poisson(expected, generator=generator).clamp_min(1.0)
        return -torch.log(counts / self.photons) / self.MU_WATER_PER_MM

    def backproject(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Exact adjoint of ``project`` divided by the pixel size: ``(A, B, S)`` -> ``(B, H, W)``."""
        with torch.enable_grad():
            probe = torch.zeros(sinogram.shape[1], self.size, self.size, device=sinogram.device,
                                dtype=sinogram.dtype, requires_grad=True)
            (image,) = torch.autograd.grad(self.project(probe), probe, grad_outputs=sinogram)
        return image / self.pixel_mm

    def fbp(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Ramp-filtered back-projection over angles spanning 180 degrees."""
        n = sinogram.shape[-1]
        spectrum = torch.fft.rfft(sinogram, n=self.padded, dim=-1) * self.ramp
        filtered = torch.fft.irfft(spectrum, n=self.padded, dim=-1)[..., :n]
        return math.pi / len(self.angles) * self.backproject(filtered)


# --------------------------------------------------------------------------------------------------- training


def build_reconstructor(config: TomographyConfig) -> UNet2d:
    model = UNet2d(1, config.base_channels, config.depth, 1)
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)
    return model


def reconstruct(model: UNet2d, fbp: torch.Tensor) -> torch.Tensor:
    """Residual post-processing: ``u_hat = fbp + net(fbp)``; ``(B, H, W)`` in and out."""
    return fbp + model(fbp[:, None]).float()[:, 0]


def tomography_loss(u_hat: torch.Tensor, u: torch.Tensor, body: torch.Tensor, config: TomographyConfig,
                    angles: Sequence[float] = ()) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = body / body.sum().clamp_min(1.0)
    error = u_hat - u
    pixel = (error.abs() if config.loss == "l1" else error.square()) * weight
    loss = pixel.sum()
    parts = {"pixel": loss.detach()}
    if config.loss == "l2_proj" and angles:
        projector = ParallelBeamProjector((config.size, config.size), angles, config.pixel_mm, device=u.device)
        with torch.no_grad():
            chords = config.pixel_mm * projector.rotate(body).sum(dim=-2)
        delta = config.pixel_mm * projector.rotate(error).sum(dim=-2)
        valid = chords >= config.min_path_mm
        if bool(valid.any()):
            term = delta[valid].square().mean()
            loss = loss + config.projection_weight * term
            parts["projection"] = term.detach()
    return loss, parts


class SliceSampler:
    """Random body slices of cached cases, converted to ``(u, body)`` on the fly, with left-right flips."""

    def __init__(self, cases: list[CachedCase], config: TomographyConfig, rng: np.random.Generator) -> None:
        if not cases:
            raise ValueError("no training cases")
        self.cases, self.config, self.rng = cases, config, rng
        self.slices = [np.flatnonzero(np.asarray(c.mask).reshape(c.mask.shape[0], -1).sum(axis=1) > 0) for c in cases]

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        us, bodies = [], []
        for _ in range(self.config.batch_size):
            index = int(self.rng.integers(len(self.cases)))
            case = self.cases[index]
            z = int(self.rng.choice(self.slices[index]))
            u, body = attenuation_slice(np.asarray(case.ct[z], dtype=np.float32), np.asarray(case.mask[z]),
                                        case.spacing[0], self.config.pixel_mm, self.config.size)
            if self.rng.random() < 0.5:
                u, body = u[:, ::-1], body[:, ::-1]
            us.append(u)
            bodies.append(body)
        return (torch.from_numpy(np.ascontiguousarray(np.stack(us))),
                torch.from_numpy(np.ascontiguousarray(np.stack(bodies))))


def train_reconstructor(cases: list[CachedCase], config: TomographyConfig, resume_path: str | Path | None = None,
                        checkpoint_every: int = 1000) -> tuple[UNet2d, pd.DataFrame]:
    """Train one reconstructor; with ``resume_path`` the training state is saved every ``checkpoint_every`` steps and
    an interrupted job continues from its last save with the same result (``sctrange.training.train_model``)."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    angle_rng = np.random.default_rng([config.seed, 3])
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(config.seed + 11)
    scanner = SparseViewScanner(config.size, config.pixel_mm, config.sparse_angles, config.photons, device)
    model = build_reconstructor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(step, config))
    sampler = SliceSampler(cases, config, rng)
    amp_dtype = autocast_dtype(device) if config.amp else None
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    resume_path = Path(resume_path) if resume_path is not None else None
    key = progress_key(config, [case.case_id for case in cases])
    modules = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler}
    rngs = {"sampler": rng, "angles": angle_rng}
    first_step, history = load_progress(resume_path, key, modules, rngs, {"noise": generator})
    running, start = defaultdict(list), time.perf_counter() - (history[-1]["seconds"] if history else 0.0)
    model.train()
    for step in range(first_step, config.steps):
        u, body = (t.to(device) for t in sampler.batch())
        with torch.no_grad():
            fbp = scanner.fbp(scanner.measure(u, generator))
        angles = (tuple(angle_rng.uniform(0.0, 180.0, config.projection_angles).tolist())
                  if config.loss == "l2_proj" else ())
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            u_hat = reconstruct(model, fbp)
        loss, parts = tomography_loss(u_hat.float(), u, body, config, angles)
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
            save_progress(resume_path, step + 1, key, modules, history, rngs, {"noise": generator})
    return model.eval(), pd.DataFrame(history)


def _load_reconstructor(path: Path, config: TomographyConfig, training_ids: Sequence[str] | None = None) -> UNet2d:
    """Reconstructor from a fold checkpoint; refuses another configuration or another training set."""
    state = torch.load(path, map_location=config.device, weights_only=True)
    check_checkpoint(state, path, {k: v for k, v in config.comparable().items() if k != "seed"}, training_ids)
    model = build_reconstructor(config).to(config.device)
    model.load_state_dict(state["state_dict"])
    return model.eval()


def train_reconstructor_fold(cache: VolumeCache, case_ids: list[str], config: TomographyConfig, fold: int,
                             checkpoint: str | Path, device: str) -> pd.DataFrame:
    """One fold, with in-fold progress in ``fold<k>.partial.pt`` (resumed after an interruption)."""
    checkpoint = Path(checkpoint)
    partial = checkpoint.with_suffix(".partial.pt")
    cases = [cache.load(case_id) for case_id in case_ids]
    model, history = train_reconstructor(cases, replace(config, seed=fold_seed(config.seed, fold), device=device),
                                         resume_path=partial)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"state_dict": model.state_dict(), "config": asdict(config), "fold": fold,
                "training_cases": ids_hash(case_ids)}, temporary)
    temporary.replace(checkpoint)
    partial.unlink(missing_ok=True)
    return history.assign(fold=fold, device=device)


_WORKER_DEVICE: str | None = None


def _claim_device(devices) -> None:
    global _WORKER_DEVICE
    _WORKER_DEVICE = devices.get()


def _fold_on_worker(*args) -> pd.DataFrame:
    return train_reconstructor_fold(*args, _WORKER_DEVICE)


def crossfit_tomography_runs(records: list[CaseRecord], cache: VolumeCache,
                             runs: Sequence[tuple[TomographyConfig, str | Path]], n_folds: int,
                             devices: Sequence[str] | None = None) -> None:
    """Every missing fold of several reconstructors in one queue; with several devices, one process per device takes
    the next fold as soon as it is free. Checkpoints (``fold<k>.pt``) are reused after a configuration check."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    devices = list(devices or [runs[0][0].device])
    jobs = []
    for config, output_dir in runs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
        folds = assign_folds(records, n_folds, config.seed)
        pd.DataFrame(sorted(folds.items()), columns=["case_id", "fold"]).to_csv(output_dir / "folds.csv", index=False)
        for fold in range(n_folds):
            training = [r.case_id for r in records if folds[r.case_id] != fold]
            checkpoint = output_dir / f"fold{fold}.pt"
            if checkpoint.exists():
                _load_reconstructor(checkpoint, replace(config, device="cpu"), training)
            else:
                jobs.append((output_dir, (cache, training, config, fold, checkpoint)))

    def append(output_dir: Path, history: pd.DataFrame) -> None:
        path = output_dir / "history.csv"
        previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([previous, history], ignore_index=True).to_csv(path, index=False)

    if len(devices) > 1 and len(jobs) > 1:
        context = multiprocessing.get_context("spawn")
        with context.Manager() as manager:
            free = manager.Queue()
            for device in devices:
                free.put(device)
            with ProcessPoolExecutor(max_workers=len(devices), mp_context=context, initializer=_claim_device,
                                     initargs=(free,)) as pool:
                futures = {pool.submit(_fold_on_worker, *arguments): output_dir for output_dir, arguments in jobs}
                for future in as_completed(futures):
                    append(futures[future], future.result())
    else:
        for output_dir, arguments in jobs:
            append(output_dir, train_reconstructor_fold(*arguments, devices[0]))


def crossfit_tomography(records: list[CaseRecord], cache: VolumeCache, config: TomographyConfig, n_folds: int,
                        output_dir: str | Path, devices: Sequence[str] | None = None) -> pd.DataFrame:
    """One reconstructor per fold (``crossfit_tomography_runs`` with one run); returns the folds."""
    crossfit_tomography_runs(records, cache, [(config, output_dir)], n_folds, devices)
    return pd.read_csv(Path(output_dir) / "folds.csv", dtype={"case_id": str})


# ------------------------------------------------------------------------------------------------- evaluation


def cvar(values: np.ndarray, alpha: float = 0.9) -> float:
    ordered = np.sort(values)[::-1]
    return float(ordered[: max(1, math.ceil((1 - alpha) * ordered.size))].mean()) if ordered.size else float("nan")


def line_integral_metrics(u_hat: torch.Tensor, u: torch.Tensor, body: torch.Tensor, scanner: SparseViewScanner,
                          min_path_mm: float, generator: torch.Generator, alpha: float = 0.9) -> dict[str, float]:
    """Image error and line-integral error (mm of water) at the scanner's angles, with a global-shuffle control."""
    inside = body > 0.5
    error = (u_hat - u) * inside
    chords = scanner.project(body.float())
    valid = chords >= min_path_mm
    delta = scanner.project(error)[valid].double().cpu().numpy()
    absolute = scanner.project(error.abs())[valid].double().cpu().numpy()
    values = error[inside]
    shuffled = torch.zeros_like(error)
    shuffled[inside] = values[torch.randperm(values.numel(), generator=generator, device=values.device)]
    control = scanner.project(shuffled)[valid].double().cpu().numpy()
    rms = float(np.sqrt(np.mean(delta**2))) if delta.size else float("nan")
    control_rms = float(np.sqrt(np.mean(control**2))) if control.size else float("nan")
    return {
        "mae_hu": 1000.0 * float(values.abs().mean()),
        "li_rays": int(delta.size),
        "li_mean_signed_mm": float(delta.mean()) if delta.size else float("nan"),
        "li_rms_mm": rms,
        "li_mean_abs_mm": float(np.abs(delta).mean()) if delta.size else float("nan"),
        "li_cvar_mm": cvar(np.abs(delta), alpha),
        "li_coherence": float(np.median(np.abs(delta)[absolute > 0] / absolute[absolute > 0])) if delta.size
        else float("nan"),
        "li_global_ratio": rms / control_rms if control_rms > 0 else float("nan"),
    }


@torch.no_grad()
def evaluate_tomography(records: Sequence[CaseRecord], cache: VolumeCache, run_dirs: dict[str, Path],
                        config: TomographyConfig, n_slices: int = 16, device: str = "cpu", seed: int = 0,
                        cache_path: str | Path | None = None, log=print) -> pd.DataFrame:
    """Per case and source (``"FBP"`` and every run in ``run_dirs``): metrics on the central ``n_slices``, at the
    held-out ``eval_angles``. The noisy sparse sinogram of a case is drawn once and shared by every source.

    With ``cache_path`` (CSV), the rows of every finished case are appended there and reused on rerun."""
    cache_path = Path(cache_path) if cache_path is not None else None
    done = pd.read_csv(cache_path, dtype={"case_id": str}) if cache_path is not None and cache_path.exists() \
        else pd.DataFrame(columns=["case_id", "source"])
    wanted = {"FBP", *run_dirs}
    measured = SparseViewScanner(config.size, config.pixel_mm, config.sparse_angles, config.photons, device)
    held_out = SparseViewScanner(config.size, config.pixel_mm, config.eval_angles, 0.0, device)
    folds = {name: pd.read_csv(d / "folds.csv", dtype={"case_id": str}).set_index("case_id")["fold"].to_dict()
             for name, d in run_dirs.items()}
    models: dict[tuple[str, int], UNet2d] = {}
    for record in records:
        if wanted <= set(done.loc[done["case_id"] == record.case_id, "source"]):
            continue
        case = cache.load(record.case_id)
        u_np, body_np = case_slices(case, config, central_indices(case, n_slices))
        u, body = torch.from_numpy(u_np).to(device), torch.from_numpy(body_np).to(device)
        digest = int.from_bytes(hashlib.sha256(f"{seed}:{record.case_id}".encode()).digest()[:4], "little")
        noise = torch.Generator(device=device).manual_seed(digest)
        fbp = measured.fbp(measured.measure(u, noise))
        sources = {"FBP": fbp}
        for name, run_dir in run_dirs.items():
            key = (name, folds[name][record.case_id])
            if key not in models:
                models[key] = _load_reconstructor(run_dir / f"fold{key[1]}.pt", run_config(run_dir, device))
            sources[name] = reconstruct(models[key], fbp)
        rows = []
        for name, u_hat in sources.items():
            shuffle = torch.Generator(device=device).manual_seed(digest + 1)
            rows.append({"case_id": record.case_id, "region": record.region, "center": record.center, "source": name,
                         **line_integral_metrics(u_hat, u, body, held_out, config.min_path_mm, shuffle)})
        done = pd.concat([done[done["case_id"] != record.case_id], pd.DataFrame(rows)], ignore_index=True)
        if cache_path is not None:
            done.to_csv(cache_path, index=False)
        log(f"{record.case_id}: {len(sources)} sources")
    ids = {r.case_id for r in records}
    return done[done["case_id"].isin(ids) & done["source"].isin(wanted)].reset_index(drop=True)


def run_config(run_dir: str | Path, device: str) -> TomographyConfig:
    """The configuration a run was trained with, on ``device``."""
    saved = json.loads((Path(run_dir) / "config.json").read_text())
    return TomographyConfig(**{**saved, "device": device})
