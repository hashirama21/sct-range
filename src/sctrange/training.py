"""Cross-fitted training of 2.5D U-Nets for the regression arms R1-R6, and out-of-fold prediction.

Point losses (all averaged over body voxels):
  ``l1_hu``   |sCT - CT| in HU                      Bayes act: voxel-wise median
  ``l2_hu``   (sCT - CT)^2 in HU                    Bayes act: voxel-wise mean in HU
  ``l2_spr``  (g~(sCT) - g~(CT))^2, g~ = g + eps*HU Bayes act: mean in SPR (docs/theory.md, Proposition 4)

The small slope ``eps`` keeps gradients alive on the flat parts of the lookup table (below -999 HU and outside
the table); with the default 1e-5 per HU it shifts the SPR scale by about 1 % of the soft-tissue slope.

Arms of the S4-S5 plan, at a strictly matched budget (same backbone, data, steps and seeds):
  R1 ``l1_hu`` · R2 ``l2_hu`` · R3 ``l2_spr`` · R4 R3 + post-hoc tissue bias correction (``sctrange.correction``)
  R5 R3 + ``range_weight`` x mean squared ΔWET over plan and random directions (on patch chords)
  R6 R5 + ``risk_weight`` x FZ0 score of a per-ray (VaR, ES) head for |ΔWET| (two extra output channels)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .cache import CachedCase, VolumeCache
from .data import CaseRecord, discover_cases, write_volume
from .hlut import HLUT
from .losses import fz_upper_tail_loss, ray_risk
from .models import UNet2d
from .normalization import CT_RANGE_HU, OUTSIDE_HU, decode_ct, encode_ct
from .projection import ParallelBeamProjector
from .split import load_split

LOSSES = ("l1_hu", "l2_hu", "l2_spr")
ARMS: dict[str, dict] = {
    "R1": {"loss": "l1_hu"},
    "R2": {"loss": "l2_hu"},
    "R3": {"loss": "l2_spr"},
    "R5": {"loss": "l2_spr", "range_weight": 1e-4},
    "R6": {"loss": "l2_spr", "range_weight": 1e-4, "risk_weight": 1e-2},
}
POSTHOC_ARMS = {"R4": "R3"}
_RUNTIME_FIELDS = ("device", "amp", "log_every")
_RISK_DENSITY_INIT = 0.01  # mm of water per mm of path, roughly a 1 mm tail over 10 cm


@dataclass(frozen=True)
class TrainConfig:
    loss: str = "l1_hu"
    context: int = 2
    patch: int = 160
    batch_size: int = 8
    steps: int = 30_000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    base_channels: int = 32
    depth: int = 4
    spr_leak: float = 1e-5
    grad_clip: float = 1.0
    log_every: int = 100
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    samples_per_case: int = 20_000
    range_weight: float = 0.0
    risk_weight: float = 0.0
    risk_level: float = 0.9
    plan_angles: tuple[float, ...] = field(default=(0.0, 120.0, 240.0))
    random_angles: int = 3
    min_path_mm: float = 20.0

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if self.range_weight < 0 or self.risk_weight < 0:
            raise ValueError("range_weight and risk_weight must be non-negative")
        if not 0.0 < self.risk_level < 1.0:
            raise ValueError("risk_level must be in (0, 1)")
        object.__setattr__(self, "plan_angles", tuple(float(a) for a in self.plan_angles))
        if self.uses_projector and not (self.plan_angles or self.random_angles > 0):
            raise ValueError("the range and risk terms need at least one beam direction")

    @property
    def in_channels(self) -> int:
        return 2 * self.context + 2

    @property
    def out_channels(self) -> int:
        return 3 if self.risk_weight > 0 else 1

    @property
    def uses_projector(self) -> bool:
        return self.range_weight > 0 or self.risk_weight > 0

    def comparable(self) -> dict:
        """Fields that define the experiment (runtime-only fields removed), JSON-normalised."""
        values = {k: v for k, v in asdict(self).items() if k not in _RUNTIME_FIELDS}
        return json.loads(json.dumps(values))


def arm_config(arm: str, **overrides) -> TrainConfig:
    """``TrainConfig`` of a trained arm (R1-R3, R5, R6) with optional overrides."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)} (R4 is post-hoc, see sctrange.correction)")
    return TrainConfig(**{**ARMS[arm], **overrides})


def point_loss(pred_hu: torch.Tensor, ct_hu: torch.Tensor, mask: torch.Tensor, loss: str, hlut: HLUT,
               spr_leak: float = 1e-5) -> torch.Tensor:
    """Masked mean of the chosen point loss."""
    if loss == "l1_hu":
        values = (pred_hu - ct_hu).abs()
    elif loss == "l2_hu":
        values = (pred_hu - ct_hu).square()
    elif loss == "l2_spr":
        values = (hlut.torch(pred_hu) - hlut.torch(ct_hu) + spr_leak * (pred_hu - ct_hu)).square()
    else:
        raise ValueError(f"unknown loss {loss!r}")
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def beam_angles(config: TrainConfig, rng: np.random.Generator) -> tuple[float, ...]:
    """Plan directions plus ``random_angles`` uniform directions, redrawn at every step (plan independence)."""
    return config.plan_angles + tuple(rng.uniform(0.0, 360.0, config.random_angles).tolist())


def training_objective(output: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor, config: TrainConfig,
                       hlut: HLUT, angles: tuple[float, ...] = (), spacing_mm: float = 1.0
                       ) -> tuple[torch.Tensor, dict[str, float]]:
    """Point loss, plus the projected range term and the risk-head score when enabled.

    ``output`` is ``(B, C, H, W)``: channel 0 the encoded CT, channels 1-2 the VaR and ES-excess densities.
    Segments are the chords of the body inside each patch; rays shorter than ``min_path_mm`` are ignored.
    The risk head is scored on the detached range error, so it never pushes the sCT itself.
    """
    pred_hu = decode_ct(output[:, :1].float())
    loss = point_loss(pred_hu, targets, masks, config.loss, hlut, config.spr_leak)
    parts = {"point": loss.detach()}
    if not config.uses_projector:
        return loss, parts

    body, ct, pred = masks[:, 0], targets[:, 0], pred_hu[:, 0]
    projector = ParallelBeamProjector(tuple(body.shape[-2:]), angles, spacing_mm, device=body.device)
    with torch.no_grad():
        weights, valid = projector.segment_weights(body)
        valid &= projector.path_length(weights) >= config.min_path_mm
    if not bool(valid.any()):
        return loss, parts
    error = (hlut.torch(pred) - hlut.torch(ct)) * body
    delta = projector.integrate(error, weights)
    if config.range_weight > 0:
        range_term = delta[valid].square().mean()
        loss = loss + config.range_weight * range_term
        parts["range"] = range_term.detach()
    if config.risk_weight > 0:
        densities = F.softplus(output[:, 1:3].float()) * body.unsqueeze(1)
        var, es = ray_risk(projector, densities[:, 0], densities[:, 1], weights)
        risk_term = fz_upper_tail_loss(var[valid], es[valid], delta.detach().abs()[valid], config.risk_level)
        loss = loss + config.risk_weight * risk_term
        parts["risk"] = risk_term.detach()
    return loss, parts


def _crop(stack: np.ndarray, y0: int, x0: int, size: int, fill: float) -> np.ndarray:
    """Crop ``(..., H, W)`` to ``size`` x ``size`` at ``(y0, x0)``, padding with ``fill`` where out of bounds."""
    height, width = stack.shape[-2:]
    out = np.full(stack.shape[:-2] + (size, size), fill, dtype=np.float32)
    ys, xs = slice(max(y0, 0), min(y0 + size, height)), slice(max(x0, 0), min(x0 + size, width))
    out[..., ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0] = stack[..., ys, xs]
    return out


def source_stack(source: np.ndarray, mask: np.ndarray, z: int, context: int) -> np.ndarray:
    """``(2 * context + 2, H, W)``: neighbouring source slices (edge-clamped) and the body mask."""
    indices = np.clip(np.arange(z - context, z + context + 1), 0, source.shape[0] - 1)
    return np.concatenate([np.asarray(source[indices], dtype=np.float32),
                           np.asarray(mask[z : z + 1], dtype=np.float32)], axis=0)


class PatchSampler:
    """Random 2.5D patches centred on body voxels, with left-right flips."""

    def __init__(self, cases: list[CachedCase], config: TrainConfig, rng: np.random.Generator) -> None:
        if not cases:
            raise ValueError("no training cases")
        spacings = {round(case.spacing[0], 4) for case in cases} | {round(case.spacing[1], 4) for case in cases}
        if config.uses_projector and len(spacings) > 1:
            raise ValueError(f"range terms need one in-plane spacing across cases, got {sorted(spacings)} mm")
        self.spacing_mm = float(cases[0].spacing[0])
        self.cases = cases
        self.config = config
        self.rng = rng
        self.centres = []
        for case in cases:
            body = np.flatnonzero(np.asarray(case.mask).ravel())
            keep = rng.choice(body.size, size=min(body.size, config.samples_per_case), replace=False)
            self.centres.append(np.stack(np.unravel_index(body[keep], case.mask.shape), axis=1))

    def batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.config.patch
        inputs, targets, masks = [], [], []
        for _ in range(self.config.batch_size):
            index = int(self.rng.integers(len(self.cases)))
            case = self.cases[index]
            z, y, x = self.centres[index][self.rng.integers(len(self.centres[index]))]
            y0, x0 = int(y) - size // 2, int(x) - size // 2
            stack = _crop(source_stack(case.source, case.mask, int(z), self.config.context), y0, x0, size, 0.0)
            target = _crop(np.asarray(case.ct[z : z + 1], dtype=np.float32), y0, x0, size, OUTSIDE_HU)
            body = _crop(np.asarray(case.mask[z : z + 1], dtype=np.float32), y0, x0, size, 0.0)
            if self.rng.random() < 0.5:
                stack, target, body = stack[..., ::-1], target[..., ::-1], body[..., ::-1]
            inputs.append(stack)
            targets.append(target)
            masks.append(body)
        return _stacked(inputs), _stacked(targets), _stacked(masks)


def _stacked(arrays: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.stack(arrays)))


def _schedule(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_model(config: TrainConfig) -> UNet2d:
    """U-Net whose untrained output decodes to water (0 HU), with small initial risk densities."""
    model = UNet2d(config.in_channels, config.base_channels, config.depth, config.out_channels)
    with torch.no_grad():
        model.head.bias[0].fill_(float(encode_ct(torch.tensor(0.0))))
        if config.out_channels > 1:
            model.head.bias[1:].fill_(math.log(math.expm1(_RISK_DENSITY_INIT)))
    return model


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Mixed-precision dtype for ``device``: bfloat16 where the GPU supports it natively (compute capability 8.0+),
    float16 otherwise (e.g. T4), ``None`` on CPU. float16 needs loss scaling, bfloat16 does not."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device) >= (8, 0) else torch.float16


def train_model(cases: list[CachedCase], config: TrainConfig, hlut: HLUT) -> tuple[UNet2d, pd.DataFrame]:
    """Train one model; returns it in eval mode with the logged history."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    angle_rng = np.random.default_rng([config.seed, 1])
    device = torch.device(config.device)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(step, config))
    sampler = PatchSampler(cases, config, rng)
    amp_dtype = autocast_dtype(device) if config.amp else None
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    history, running, start = [], defaultdict(list), time.perf_counter()
    model.train()
    for step in range(config.steps):
        inputs, targets, masks = (tensor.to(device, non_blocking=True) for tensor in sampler.batch())
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            output = model(inputs)
        angles = beam_angles(config, angle_rng) if config.uses_projector else ()
        loss, parts = training_objective(output, targets, masks, config, hlut, angles, sampler.spacing_mm)
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
            # one host synchronisation per logging interval rather than per step
            row = {name: float(torch.stack(values).float().mean()) for name, values in running.items()}
            history.append({"step": step + 1, **row, "learning_rate": scheduler.get_last_lr()[0],
                            "seconds": time.perf_counter() - start})
            running = defaultdict(list)
    return model.eval(), pd.DataFrame(history)


@torch.no_grad()
def predict_case(model: UNet2d, case: CachedCase, context: int, device: str, batch_size: int = 8,
                 with_risk: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """sCT in HU on the cropped grid of ``case``; ``OUTSIDE_HU`` outside the body.

    With ``with_risk``, also returns the risk densities ``(2, Z, Y, X)`` (VaR, ES excess; zero outside the body).
    """
    stride = model.stride
    depth, height, width = case.mask.shape
    pad_h, pad_w = (-height) % stride, (-width) % stride
    channels = model.head.out_channels
    if with_risk and channels < 3:
        raise ValueError("this model has no risk head")
    output = np.empty((channels, depth, height, width), dtype=np.float32)
    for start in range(0, depth, batch_size):
        stacks = np.stack([source_stack(case.source, case.mask, z, context) for z in range(start, min(start + batch_size, depth))])
        inputs = F.pad(torch.from_numpy(stacks), (0, pad_w, 0, pad_h)).to(device)
        raw = model(inputs).float()[..., :height, :width]
        values = torch.cat([decode_ct(raw[:, :1]), F.softplus(raw[:, 1:])], dim=1)
        output[:, start : start + len(stacks)] = values.transpose(0, 1).cpu().numpy()
    inside = np.asarray(case.mask) > 0
    sct = np.where(inside, output[0], OUTSIDE_HU).astype(np.float32)
    if not with_risk:
        return sct
    return sct, np.where(inside[None], output[1:3], 0.0).astype(np.float32)


def uncrop(case: CachedCase, cropped: np.ndarray, fill: float = OUTSIDE_HU) -> np.ndarray:
    full = np.full(cropped.shape[:-3] + case.shape, fill, dtype=np.float32)
    full[(...,) + case.box] = cropped
    return full


def assign_folds(records: list[CaseRecord], n_folds: int, seed: int) -> dict[str, int]:
    """Stratified folds: each task/region/centre stratum is shuffled with its own seed and dealt round-robin."""
    strata: dict[str, list[str]] = defaultdict(list)
    for record in records:
        strata[record.stratum].append(record.case_id)
    folds, offset = {}, 0
    for stratum in sorted(strata):
        ids = sorted(strata[stratum])
        digest = hashlib.sha256(f"{seed}:folds:{stratum}".encode()).digest()
        order = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(len(ids))
        for position, index in enumerate(order):
            folds[ids[index]] = (offset + position) % n_folds
        offset += len(ids)
    return folds


def _load_checkpoint(path: Path, config: TrainConfig) -> UNet2d:
    """Model from a fold checkpoint; refuses a checkpoint trained with a different experiment configuration."""
    state = torch.load(path, map_location=config.device, weights_only=True)
    saved = {k: v for k, v in json.loads(json.dumps(state["config"])).items() if k not in _RUNTIME_FIELDS and k != "seed"}
    current = {k: v for k, v in config.comparable().items() if k != "seed"}
    defaults = TrainConfig(loss=config.loss).comparable()
    saved = {**{k: defaults[k] for k in current if k not in saved}, **saved}  # checkpoints from before new fields
    if saved != current:
        changed = sorted(k for k in current if saved.get(k) != current[k])
        raise ValueError(f"{path} was trained with a different configuration ({', '.join(changed)}); "
                         "use another output directory")
    model = build_model(config).to(config.device)
    model.load_state_dict(state["state_dict"])
    return model.eval()


def visible_devices(default: str = "cpu") -> list[str]:
    """Every visible GPU (``cuda:0``, ``cuda:1`` on a 2 x T4 machine), else ``[default]``."""
    return [f"cuda:{index}" for index in range(torch.cuda.device_count())] or [default]


def train_fold(cache: VolumeCache, case_ids: list[str], config: TrainConfig, hlut: HLUT, fold: int,
               checkpoint: str | Path, device: str) -> pd.DataFrame:
    """Train the model of one fold on ``device`` and save its checkpoint; returns the history.

    Top-level and argument-picklable so that folds can run in parallel processes, one per GPU.
    ``config`` is the run configuration; the fold's model uses seed ``config.seed + fold``.
    """
    cases = [cache.load(case_id) for case_id in case_ids]
    model, history = train_model(cases, replace(config, seed=config.seed + fold, device=device), hlut)
    temporary = Path(checkpoint).with_suffix(".pt.tmp")
    torch.save({"state_dict": model.state_dict(), "config": asdict(config), "fold": fold}, temporary)
    temporary.replace(checkpoint)
    return history.assign(fold=fold, device=device)


def _append_history(path: Path, history: pd.DataFrame) -> None:
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([previous, history], ignore_index=True).to_csv(path, index=False)


def _write_prediction(sct: np.ndarray, case: CachedCase, path: Path) -> None:
    """Out-of-fold sCT as int16 HU (half the size of float32; the reference CT is integer HU in the same range)."""
    hu = np.clip(np.rint(uncrop(case, sct)), *CT_RANGE_HU).astype(np.int16)
    write_volume(hu, path, spacing=case.spacing, origin=case.origin, direction=case.direction)


def crossfit(records: list[CaseRecord], cache: VolumeCache, config: TrainConfig, hlut: HLUT, n_folds: int,
             output_dir: str | Path, devices: Sequence[str] | None = None) -> pd.DataFrame:
    """Train one model per fold on the other folds; write out-of-fold predictions and checkpoints.

    Layout: ``<output_dir>/predictions/<case_id>.mha`` (int16 HU), ``fold<k>.pt``, ``history.csv``, ``folds.csv``,
    ``config.json``, and ``risk/<case_id>_{var,excess}.mha`` when the model has a risk head. Folds whose checkpoint
    exists are reused (after checking that it was trained with the same configuration), so interrupted runs resume;
    the history is appended after every fold. With several ``devices`` (e.g. ``["cuda:0", "cuda:1"]`` on two T4),
    missing folds are trained in parallel, one process per device; prediction then runs on the first device.
    """
    output_dir = Path(output_dir)
    devices = list(devices or [config.device])
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    folds = assign_folds(records, n_folds, config.seed)
    frame = pd.DataFrame(sorted(folds.items()), columns=["case_id", "fold"])
    frame.to_csv(output_dir / "folds.csv", index=False)
    history_path = output_dir / "history.csv"
    with_risk = config.out_channels > 1

    def job(fold: int, device: str) -> tuple:
        training = [r.case_id for r in records if folds[r.case_id] != fold]
        return cache, training, config, hlut, fold, output_dir / f"fold{fold}.pt", device

    pending = []
    for fold in range(n_folds):
        checkpoint = output_dir / f"fold{fold}.pt"
        if checkpoint.exists():
            _load_checkpoint(checkpoint, config)  # refuses a checkpoint trained with another configuration
        else:
            pending.append(fold)
    if len(devices) > 1 and len(pending) > 1:
        context = multiprocessing.get_context("spawn")  # CUDA cannot be shared with forked processes
        for start in range(0, len(pending), len(devices)):
            wave = pending[start:start + len(devices)]
            with ProcessPoolExecutor(max_workers=len(wave), mp_context=context) as pool:
                futures = [pool.submit(train_fold, *job(fold, device)) for fold, device in zip(wave, devices)]
                for future in futures:
                    _append_history(history_path, future.result())
    else:
        for fold in pending:
            _append_history(history_path, train_fold(*job(fold, devices[0])))

    device = devices[0]
    for fold in range(n_folds):
        model = _load_checkpoint(output_dir / f"fold{fold}.pt", replace(config, device=device))
        for case_id in (r.case_id for r in records if folds[r.case_id] == fold):
            case = cache.load(case_id)
            if with_risk:
                sct, risk = predict_case(model, case, config.context, device, with_risk=True)
                geometry = dict(spacing=case.spacing, origin=case.origin, direction=case.direction)
                for name, density in zip(("var", "excess"), uncrop(case, risk, fill=0.0)):
                    write_volume(density, output_dir / "risk" / f"{case_id}_{name}.mha", **geometry)
            else:
                sct = predict_case(model, case, config.context, device)
            _write_prediction(sct, case, output_dir / "predictions" / f"{case_id}.mha")
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted training on the development cases of a frozen split.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--loss", choices=LOSSES, help="point loss only; writes <out>/<loss>")
    which.add_argument("--arm", choices=sorted({*ARMS, *POSTHOC_ARMS}), help="plan arm R1-R6; writes <out>/<arm>")
    parser.add_argument("--task", choices=("1", "2"), default="1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--patch", type=int, default=TrainConfig.patch)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--base-channels", type=int, default=TrainConfig.base_channels)
    parser.add_argument("--context", type=int, default=TrainConfig.context)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--range-weight", type=float, help="override λ of R5/R6")
    parser.add_argument("--risk-weight", type=float, help="override β of R6")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--devices", help="comma-separated devices to train folds in parallel, e.g. cuda:0,cuda:1 "
                                          "(default: every visible GPU)")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)

    split = load_split(args.split)
    development = set(split["development"])
    records = [r for r in discover_cases(args.data_root) if r.case_id in development and r.task == args.task]
    if not records:
        parser.error("no development cases found for this task")
    hlut = HLUT.pyradplan_default()

    if args.arm in POSTHOC_ARMS:
        from .correction import crossfit_bias_correction

        source = args.out / POSTHOC_ARMS[args.arm]
        if not (source / "folds.csv").exists():
            parser.error(f"{args.arm} corrects {POSTHOC_ARMS[args.arm]}: train it first ({source} not found)")
        offsets = crossfit_bias_correction(records, source, args.out / args.arm, hlut)
        print(offsets.round(4).to_string())
        print(f"out-of-fold predictions in {args.out / args.arm / 'predictions'}")
        return

    overrides = dict(steps=args.steps, patch=args.patch, batch_size=args.batch_size, base_channels=args.base_channels,
                     context=args.context, learning_rate=args.learning_rate, seed=args.seed, device=args.device,
                     amp=not args.no_amp)
    if args.range_weight is not None:
        overrides["range_weight"] = args.range_weight
    if args.risk_weight is not None:
        overrides["risk_weight"] = args.risk_weight
    if args.arm:
        config, name = arm_config(args.arm, **overrides), args.arm
    else:
        config, name = TrainConfig(loss=args.loss, **overrides), args.loss
    devices = args.devices.split(",") if args.devices else visible_devices(args.device)
    cache = VolumeCache(args.cache_dir)
    print(f"caching {cache.build(records)} new cases · training on {', '.join(devices)}")
    crossfit(records, cache, config, hlut, args.folds, args.out / name, devices)
    print(f"out-of-fold predictions in {args.out / name / 'predictions'}")


if __name__ == "__main__":
    main()
