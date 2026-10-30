"""Build ``notebooks/s8_s10_generality.ipynb``: weeks 8 to 10 of the plan in one notebook with automatic resumption.

The notebook is generated, never edited by hand: change this script, then run

    python tools/build_s8_s10_notebook.py

``tests/test_notebooks.py`` fails while the committed notebook differs from what this script produces. The setup cell
is the one of ``notebooks/s7_generative.ipynb``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
NAME = "s8_s10_generality.ipynb"
BASE = "github/hashirama21/sct-range/blob/main/notebooks"
BADGES = (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
          f"(https://colab.research.google.com/{BASE}/{NAME}) "
          f"[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]"
          f"(https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/{NAME})")

INTRO = f"""# S8–S10 · Generative arm in dose and on CBCT, second domain, ablations

{BADGES}

Weeks 8 to 10 of the plan in one notebook. It starts where `s1_s7_pipeline` (or the S1–S7 notebooks) stopped: it reads
the frozen split of S1, the runs and report of S4, the dosimetry subset of S6 and the selected generator of S7, and
writes to `outputs/s8_s10/`.

| Stage | Week | Plan item | Rule (`configs/`) |
|---|---|---|---|
| `S8.dose` | S8 | the S7 generator's SPR mean recomputed in dose with the S6 plans, against R3 | `s8_s10_preregistration.json` · `s8` |
| `S8.task2` | S8 | the generative arm on task 2 (CBCT → CT) with the η chosen on task 1: projected CRPS, coverage, PIT | `s8` |
| `S9.task2` | S9 | second domain, mandatory: R1–R6 on task 2 with the S4 rule unchanged; replication of the task-1 verdict | `s9_task2` |
| `S9.spectra` | S9 | the S3 go/no-go rule on the task-2 L1 models (beam spectra, shuffle controls) | `s9_task2` |
| `S9.tomography` | S9 | second domain, if possible: sparse-view CT, where the target is again a line integral but g is linear | `s9_tomography` |
| `S9.rainfall` | S9 | basin rainfall: pre-registered as not run (gated on S6 timeliness, external data) | `s9_precipitation` |
| `S10.ablations` | S10 | number and choice of directions, segment length, λ, β, against the S4 centre runs | `s10_ablations` |
| `S10.registration` | S10 | registration cross-matrix: train Elastix / IMPACT × test Elastix / IMPACT | `s10_registration` |

**Automatic resumption.** After a crash, an error or an expired session, reopen the notebook and *Run all* again with
the same parameters; nothing has to be cleaned up.

- The state of every stage is kept with the outputs (`outputs/pipeline_state.json`, shared with `s1_s7_pipeline`; on
  Kaggle, attach the previous version's output as an input). A finished stage is skipped (its cells run as `pass`);
  it runs again when its parameters change, when an upstream stage (S3, S4, S6, S7, or an S8–S10 stage it reads)
  finished again, when its report is deleted, or when it is listed in `FORCE_STAGES`.
- A stage that stopped continues where it stopped: fold checkpoints are reused and a fold cut mid-training resumes
  from its last save (every 1000 steps, the same model as an uninterrupted run), for the regression arms, the
  generator and the tomography reconstructors alike; per-case evaluations, generator samples, dosimetry cases and
  spectra are cached; a run whose evaluation is complete is not predicted again.
- **Storage.** On Colab and Kaggle the persistent outputs (Drive, the 20 GB of `/kaggle/working`) keep checkpoints,
  in-fold progress, histories, evaluations and reports. The out-of-fold predictions of the runs trained here go to
  the session disk (`/kaggle/tmp`, `/content`), and a later session that needs them again rebuilds them from the
  checkpoints, without training.

**Compute at the pre-registered budget.** One task-2 generator and five task-2 arms, eight ablation variants and one
IMPACT model (5 folds × 30 000 steps each), six small tomography models (2 folds × 20 000 steps), sampling, and the
dosimetry of 20 cases on the CPU. With two T4 GPUs the folds of all runs of a stage share one queue. The `RUN_*`
switches skip stages; a skipped stage is not marked complete and runs on a later *Run all* with its switch on.

The notebook runs on the **real SynthRAD2025 training data** (Hugging Face mirror `developper669/synthrad2025`, tag
`v2`), fetched when missing. `S10.registration` also needs the IMPACT-registered CTs in the SynthRAD2025 layout
(`Task1/<region>/<case_id>/ct.mha`) under `IMPACT_ROOT`; without them the stage reports that it did not run.
"""

PARAMS_MD = """### Parameters

Set them here and run all cells (*Runtime → Run all* on Colab, where this cell is a form). Environment variables of the
same meaning, given in brackets, override them for unattended runs (`jupyter nbconvert --execute`). Pre-registered
verdicts assume the defaults; smaller values give a quick pass only, and every report says so.

| Platform | Dataset, cache, predictions of this notebook | Outputs (checkpoints, evaluations, reports) |
|---|---|---|
| local | `data/synthrad2025`, `outputs/` | `outputs/` |
| Colab | `/content/data`, `/content/cache` | Google Drive `MyDrive/sct-range/outputs` (`USE_DRIVE`) |
| Kaggle | `/kaggle/tmp/data`, `/kaggle/tmp/cache` | `/kaggle/working/outputs` |
"""

PARAMS = """# @title Parameters { display-mode: "form" }
DOWNLOAD = True  # @param {type:"boolean"}  fetch missing data from Hugging Face [SCT_DOWNLOAD]
USE_DRIVE = True  # @param {type:"boolean"}  Colab only
FORCE_STAGES = ""  # @param {type:"string"}  stages to run again even if complete, e.g. "S9.task2" [SCT_FORCE_STAGES]
TRAIN = True  # @param {type:"boolean"}  train missing folds here (needs a GPU) [SCT_TRAIN_IN_NOTEBOOK]
STEPS = 30000  # @param {type:"integer"}  steps per fold of the sCT models [SCT_STEPS]
FOLDS = 5  # @param {type:"integer"}  cross-fitting folds of the sCT models [SCT_FOLDS]
# S8
RUN_S8 = True  # @param {type:"boolean"}  [SCT_RUN_S8]
DOSE_WORKERS = 1  # @param {type:"integer"}  cases dosed in parallel (CPU) [SCT_WORKERS]
TASK2_GEN_CASES = 100  # @param {type:"integer"}  [SCT_TASK2_GEN_CASES]
# S9
RUN_S9 = True  # @param {type:"boolean"}  [SCT_RUN_S9]
TOMO_STEPS = 20000  # @param {type:"integer"}  [SCT_TOMO_STEPS]
TOMO_CASES = 0  # @param {type:"integer"}  tomography evaluation cases, 0 = every task-1 development case [SCT_TOMO_CASES]
# S10
RUN_S10 = True  # @param {type:"boolean"}  [SCT_RUN_S10]
ABLATION_CASES = 0  # @param {type:"integer"}  0 = every task-1 development case [SCT_ABLATION_CASES]
IMPACT_ROOT = ""  # @param {type:"string"}  IMPACT-registered CTs, SynthRAD2025 layout [SCT_IMPACT_ROOT]
"""

RESUME_MD = """### Resumption

The state of every stage is read from the outputs; the table below shows what the previous runs left (S1–S7 included
when they ran through `s1_s7_pipeline`). Stages marked `complete` with unchanged inputs are skipped below.
"""

RESUME = """from sctrange.pipeline import StageTracker
from sctrange.runtime import resolve_paths

PIPELINE_OUTPUTS = resolve_paths(REPO_ROOT, USE_DRIVE).output_root
PIPELINE = StageTracker(PIPELINE_OUTPUTS, os.environ.get("SCT_FORCE_STAGES", FORCE_STAGES).split(","))
PIPELINE.install()
PIPELINE.show()
"""

IMPORTS = """import hashlib
import importlib.util
import json
import os
import shutil
import warnings
from dataclasses import asdict, replace

warnings.filterwarnings("ignore", message=r"`torch\\.jit\\.script` is deprecated")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

from sctrange.cache import VolumeCache
from sctrange.correction import crossfit_bias_correction
from sctrange.data import bulk_density_sct, discover_cases, load_case, read_prediction
from sctrange.diagnostics import Preregistration, decide, decide_arms, diagnose_case, holm, paired_test
from sctrange.dosimetry import DoseConfig, average_seeds, run_dosimetry
from sctrange.download import ensure_dataset, usable_records
from sctrange.evaluation import DiagnosticConfig
from sctrange.generality import (ablation_grid, compare_family, evaluate_predictions, evaluated_sources,
                                 prediction_shim, registration_matrix, registration_overlay)
from sctrange.generative import GenerativeConfig, crossfit_generative, evaluate_generative, run_name
from sctrange.hlut import HLUT
from sctrange.plotting import (apply_style, plot_ablation_forest, plot_arm_pareto, plot_beam_spectra,
                               plot_gamma_by_source, plot_pit, plot_registration_matrix, plot_tomography_example,
                               plot_training_curves)
from sctrange.runs import artifact_dir, cached_result, files_signature, redirect_artifacts, run_complete
from sctrange.runtime import frozen_split_path
from sctrange.split import load_selection, load_split, select_cases, split_hash
from sctrange.tomography import (SparseViewScanner, TomographyConfig, _load_reconstructor, case_slices,
                                 central_indices, crossfit_tomography_runs, evaluate_tomography, reconstruct,
                                 run_config)
from sctrange.training import ARMS, arm_config, crossfit_runs, visible_devices
"""

CONTEXT_MD = """### Common context

Paths, pre-registrations, the frozen split, the development cases of both tasks and the volume cache. This cell runs
in every session (stages below may be skipped, so each stage reads what it needs from disk).
"""

CONTEXT = """PATHS = resolve_paths(REPO_ROOT, USE_DRIVE)
flag = lambda name, value: os.environ.get(name, "1" if value else "0") == "1"
DOWNLOAD = os.environ.get("SCT_DOWNLOAD", "1" if DOWNLOAD else "0") != "0"
TRAIN = flag("SCT_TRAIN_IN_NOTEBOOK", TRAIN)
STEPS = int(os.environ.get("SCT_STEPS", STEPS))
N_FOLDS = int(os.environ.get("SCT_FOLDS", FOLDS))
RUN_S8, RUN_S9, RUN_S10 = flag("SCT_RUN_S8", RUN_S8), flag("SCT_RUN_S9", RUN_S9), flag("SCT_RUN_S10", RUN_S10)
DOSE_WORKERS = int(os.environ.get("SCT_WORKERS", DOSE_WORKERS))
TASK2_GEN_CASES = int(os.environ.get("SCT_TASK2_GEN_CASES", TASK2_GEN_CASES))
TOMO_STEPS = int(os.environ.get("SCT_TOMO_STEPS", TOMO_STEPS))
TOMO_CASES = int(os.environ.get("SCT_TOMO_CASES", TOMO_CASES)) or None
ABLATION_CASES = int(os.environ.get("SCT_ABLATION_CASES", ABLATION_CASES)) or None
IMPACT_ROOT = os.environ.get("SCT_IMPACT_ROOT", IMPACT_ROOT)
IMPACT_ROOT = Path(IMPACT_ROOT) if IMPACT_ROOT else None

OUTPUT_DIR = PATHS.output_root / "s8_s10"
FIGURE_DIR = OUTPUT_DIR / "figures"
S4_RUNS = Path(os.environ.get("SCT_S4_RUNS_ROOT", PATHS.output_root / "s4" / "runs"))
S7_DIR = Path(os.environ.get("SCT_S7_DIR", PATHS.output_root / "s7"))
S7_RUNS = Path(os.environ.get("SCT_GEN_RUNS_ROOT", S7_DIR / "runs"))
S6_DIR = PATHS.output_root / "s6"
CACHE_DIR = Path(os.environ.get("SCT_CACHE_DIR", PATHS.cache_root / "cache"))
FROZEN_SPLIT_PATH = frozen_split_path(REPO_ROOT, PATHS)
if PATHS.ephemeral:  # predictions on the session disk, rebuilt from checkpoints when needed (see the introduction)
    redirect_artifacts(OUTPUT_DIR, PATHS.cache_root / "s8_s10_artifacts")


def load_prereg(name):
    data = (REPO_ROOT / "configs" / name).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


PREREG, PREREG_SHA256 = load_prereg("s8_s10_preregistration.json")
PREREG_S6, _ = load_prereg("s6_preregistration.json")
PREREG_S7, _ = load_prereg("s7_preregistration.json")
S3_RULE = Preregistration.load(REPO_ROOT / "configs" / "s3_preregistration.json")
S4_RULE = Preregistration.load(REPO_ROOT / "configs" / "s4_preregistration.json")

SEED = 20261013
DEVICES = visible_devices()
DEVICE = DEVICES[0]
DIAGNOSTIC = DiagnosticConfig(angles_deg=tuple(range(0, 360, 45)), target_radius_mm=25.0, seed=SEED, device=DEVICE)
hlut = HLUT.pyradplan_default()
HAS_PYRADPLAN = importlib.util.find_spec("pyRadPlan") is not None
TRAIN_IN_NOTEBOOK = TRAIN and (DEVICE.startswith("cuda") or os.environ.get("SCT_TRAIN_ON_CPU") == "1")
if TRAIN and not TRAIN_IN_NOTEBOOK:
    print("no GPU: training skipped, existing checkpoints only (SCT_TRAIN_ON_CPU=1 forces CPU training)")

DATA_ROOT = ensure_dataset(PATHS.data_root, ("1", "2"), download=DOWNLOAD)
if not FROZEN_SPLIT_PATH.exists():
    raise FileNotFoundError(f"{FROZEN_SPLIT_PATH} is missing: run S1 (s1_s7_pipeline) once on the real data first")
split = load_split(FROZEN_SPLIT_PATH)
development_ids = set(split["development"])
assert not development_ids & set(split["test"])
records = discover_cases(DATA_ROOT)
dev = {}
for task in ("1", "2"):
    dev[task], excluded = usable_records(DATA_ROOT, [r for r in records if r.case_id in development_ids and r.task == task])
    if excluded:
        print(f"task {task}: {len(excluded)} development cases left out by the integrity check")
cache = VolumeCache(CACHE_DIR)
print(f"{cache.build(dev['1'] + dev['2'])} new cases cached")

s4_prediction = lambda arm, r: artifact_dir(S4_RUNS / "seed0" / arm) / f"{r.case_id}.mha"
HAVE_S4 = all(run_complete(S4_RUNS / "seed0" / arm, [r.case_id for r in dev["1"]]) for arm in ("R3", "R5", "R6"))
s4_report_path = PATHS.output_root / "s4" / "s4_report.json"
s7_report_path = S7_DIR / "s7_report.json"
S7_REPORT = json.loads(s7_report_path.read_text()) if s7_report_path.exists() else {}
SELECTED_ETA = S7_REPORT.get("selected_eta")
SELECTED_CONFIG = (GenerativeConfig(**{**S7_REPORT["config"], "eta": SELECTED_ETA, "device": DEVICE})
                   if SELECTED_ETA is not None else None)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
apply_style()
pd.set_option("display.width", 170)
QUICK = (STEPS, N_FOLDS) != (PREREG["s9_task2"]["steps"], PREREG["s9_task2"]["folds"])


def report_path(stage: str) -> Path:
    return OUTPUT_DIR / f"{stage.lower().replace('.', '_')}_report.json"


def write_report(stage: str, payload: dict) -> None:
    payload = {"stage": stage, "quick_pass": QUICK, "preregistration_sha256": PREREG_SHA256,
               "split_sha256": split_hash(split), **payload}
    report_path(stage).write_text(json.dumps(payload, indent=2, default=float))


display(pd.Series({**PATHS.describe(REPO_ROOT), "task 1 development cases": len(dev["1"]),
                   "task 2 development cases": len(dev["2"]), "S4 seed 0 (R3, R5, R6) complete": HAVE_S4,
                   "S7 selected eta": SELECTED_ETA, "S6 dosimetry subset": (S6_DIR / "dosimetry_cases.json").exists(),
                   "pyRadPlan": HAS_PYRADPLAN, "IMPACT CTs": str(IMPACT_ROOT) if IMPACT_ROOT else "not set",
                   "devices": ", ".join(DEVICES), "quick pass": QUICK}, name="value").to_frame())
"""

S8_DOSE_MD = """# S8 · The generative arm in dose and on CBCT

## S8.dose · The generator's SPR mean recomputed in dose

The S7 generator (selected η, K and N of the S7 main evaluation) writes its SPR mean for the frozen S6 dosimetry cases;
the S6 plans (same configuration, same engine) are recomputed on it. Its proton γ 2 %/2 mm is compared with R3 seed 0
from S6 (paired two-sided Wilcoxon); the pre-registered expectation is no difference, since the generator's mean equals
the Bayes act at best.
"""

S8_DOSE_BEGIN = """PIPELINE.begin("S8.dose", {"workers": DOSE_WORKERS, "run": RUN_S8}, report_path("S8.dose"), after=("S6", "S7"))
"""

S8_DOSE = """reasons = [m for m, bad in (("RUN_S8 off", not RUN_S8), ("no S7 report with a selected eta", SELECTED_ETA is None),
                               ("no S6 dosimetry subset", not (S6_DIR / "dosimetry_cases.json").exists()),
                               ("no S6 dose table", not (S6_DIR / "s6_dose_per_seed.csv").exists()),
                               ("pyRadPlan missing", not HAS_PYRADPLAN)) if bad]
S8_DOSE_DONE = False
if reasons:
    print("S8.dose not run: " + "; ".join(reasons))
else:
    dose_cases = [r for r in dev["1"] if r.case_id in set(load_selection(S6_DIR / "dosimetry_cases.json")["case_ids"])]
    selected_run = S7_RUNS / run_name(SELECTED_CONFIG)
    main_spec = PREREG_S7["main"]
    evaluate_generative(dose_cases, cache, selected_run, SELECTED_CONFIG, hlut, DIAGNOSTIC, main_spec["n_samples"],
                        main_spec["n_steps"], DEVICE, write_mean=True, full=True)
    shim = prediction_shim(artifact_dir(selected_run), PATHS.cache_root / "s8_dose_shim", "G")
    dose_config = DoseConfig.from_preregistration(PREREG_S6)
    gen_dose, _ = run_dosimetry(dose_cases, shim, OUTPUT_DIR / "s8_dose", dose_config, ["G"], [0], include_bulk=False,
                                workers=DOSE_WORKERS)
    s6_dose = pd.read_csv(S6_DIR / "s6_dose_per_seed.csv")
    dose = average_seeds(pd.concat([gen_dose, s6_dose[s6_dose["source"] == "R3@0"]], ignore_index=True))
    dose.to_csv(OUTPUT_DIR / "s8_dose.csv", index=False)
    columns = ["mae_hu", "proxy_rms_mm", "proton_gamma_2_2", "proton_gamma_1_1", "photon_gamma_2_2",
               "proton_target_d95_diff_pct"]
    display(dose.groupby("source")[columns].median().round(3))
    test = paired_test(dose, "proton_gamma_2_2", "G", "R3")
    display(pd.Series(test, name="proton γ 2 %/2 mm, G − R3").to_frame())
    plot_gamma_by_source(dose, ["R3", "G"]).savefig(FIGURE_DIR / "s8_dose_gamma.png", bbox_inches="tight")
    plt.show()
    S8_DOSE_DONE = set(dose.loc[dose["source"] == "G", "case_id"]) == {r.case_id for r in dose_cases}
    write_report("S8.dose", {"selected_eta": SELECTED_ETA, "cases": [r.case_id for r in dose_cases], "test": test,
                             "medians": dose.groupby("source")[columns].median().to_dict(orient="index")})
"""

S8_DOSE_END = """PIPELINE.finish("S8.dose", complete=S8_DOSE_DONE, note="; ".join(reasons))
"""

S8_T2_MD = """## S8.task2 · The generative arm on CBCT → CT

A generator with the η selected on task 1 (no new selection) is cross-fitted on the task-2 development cases and
evaluated on a region-stratified subset of them with the S7 scores. Calibration on task 2 is set beside task 1.
"""

S8_T2_BEGIN = """PIPELINE.begin("S8.task2", {"steps": STEPS, "folds": N_FOLDS, "cases": TASK2_GEN_CASES, "train": TRAIN,
                           "run": RUN_S8}, report_path("S8.task2"), after=("S7",))
"""

S8_T2 = """reasons = [m for m, bad in (("RUN_S8 off", not RUN_S8), ("no S7 report with a selected eta", SELECTED_ETA is None),
                               ("no task-2 development case", not dev["2"])) if bad]
S8_T2_DONE = False
if reasons:
    print("S8.task2 not run: " + "; ".join(reasons))
else:
    spec = PREREG["s8"]["task2_generative"]
    t2_config = replace(SELECTED_CONFIG, steps=STEPS)
    t2_run = OUTPUT_DIR / "s8_task2" / run_name(t2_config)
    if TRAIN_IN_NOTEBOOK:
        crossfit_generative(dev["2"], cache, t2_config, N_FOLDS, t2_run, DEVICES)
    trained = all((t2_run / f"fold{k}.pt").exists() for k in range(N_FOLDS))
    if not trained:
        reasons.append("task-2 generator not trained (GPU needed)")
        print("S8.task2: " + reasons[-1])
    else:
        t2_ids = select_cases(dev["2"], TASK2_GEN_CASES, spec["subset_seed"])
        t2_cases = [r for r in dev["2"] if r.case_id in set(t2_ids)]
        t2_rows, t2_pits = evaluate_generative(t2_cases, cache, t2_run, t2_config, hlut, DIAGNOSTIC,
                                               spec["n_samples"], spec["n_steps"], DEVICE, full=True)
        t2_rows.to_csv(OUTPUT_DIR / "s8_task2_generative.csv", index=False)
        columns = spec["report"]
        summary = {"task 2": t2_rows[columns].median()}
        if (S7_DIR / "s7_main.csv").exists():
            summary["task 1 (S7)"] = pd.read_csv(S7_DIR / "s7_main.csv")[columns].median()
        summary = pd.concat(summary, axis=1)
        display(summary.round(3))
        display(t2_rows.groupby("region")[columns].median().round(3))
        if (t2_run / "history.csv").exists():
            plot_training_curves({"task-2 generator": pd.read_csv(t2_run / "history.csv")}).savefig(
                FIGURE_DIR / "s8_task2_training.png", bbox_inches="tight")
            plt.show()
        plot_pit(t2_pits).savefig(FIGURE_DIR / "s8_task2_pit.png", bbox_inches="tight")
        plt.show()
        S8_T2_DONE = len(t2_rows) == len(t2_cases)
        write_report("S8.task2", {"eta": SELECTED_ETA, "cases": t2_ids, "summary": summary.to_dict()})
"""

S8_T2_END = """PIPELINE.finish("S8.task2", complete=S8_T2_DONE, note="; ".join(reasons))
"""

S9_T2_MD = """# S9 · Second domain

## S9.task2 · CBCT → CT: the regression arms and the S4 rule

R1, R2, R3, R5 and R6 are cross-fitted on the task-2 development cases (seed 0, the S4 budget per fold; one fold queue
for all arms), R4 is derived from R3, every out-of-fold prediction is evaluated with the S4 metrics, and
`configs/s4_preregistration.json` is applied unchanged (one seed). Task 2 replicates task 1 when both verdicts agree.
"""

S9_T2_BEGIN = """PIPELINE.begin("S9.task2", {"steps": STEPS, "folds": N_FOLDS, "train": TRAIN, "run": RUN_S9}, report_path("S9.task2"),
               after=("S4",))
"""

S9_T2 = """ARM_ORDER = ("R1", "R2", "R3", "R4", "R5", "R6")
T2_RUNS = OUTPUT_DIR / "s9_task2" / "seed0"
T2_EVAL = OUTPUT_DIR / "s9_task2_arms.csv"
reasons = [m for m, bad in (("RUN_S9 off", not RUN_S9), ("no task-2 development case", not dev["2"])) if bad]
S9_T2_DONE = False
t2_ids = [r.case_id for r in dev["2"]]
if reasons:
    print("S9.task2 not run: " + "; ".join(reasons))
else:
    done = evaluated_sources(T2_EVAL, ARM_ORDER, t2_ids)
    needed = [arm for arm in sorted(ARMS) if arm not in done or (arm == "R3" and "R4" not in done)]
    if needed and TRAIN_IN_NOTEBOOK:
        crossfit_runs(dev["2"], cache, [(arm_config(arm, seed=0, device=DEVICE, steps=STEPS), T2_RUNS / arm)
                                        for arm in needed], hlut, N_FOLDS, DEVICES)
    elif needed:  # no GPU: predictions only for runs whose folds are all trained
        trained = [arm for arm in needed if all((T2_RUNS / arm / f"fold{k}.pt").exists() for k in range(N_FOLDS))]
        if trained:
            crossfit_runs(dev["2"], cache, [(arm_config(arm, seed=0, device=DEVICE, steps=STEPS), T2_RUNS / arm)
                                            for arm in trained], hlut, N_FOLDS, DEVICES)
    if "R4" not in done and run_complete(T2_RUNS / "R3", t2_ids):
        crossfit_bias_correction(dev["2"], T2_RUNS / "R3", T2_RUNS / "R4", hlut)
    sources = {arm: artifact_dir(T2_RUNS / arm) for arm in ARM_ORDER
               if arm in done or run_complete(T2_RUNS / arm, t2_ids)}
    t2_frame, t2_risk = evaluate_predictions(dev["2"], sources, hlut, DIAGNOSTIC, T2_EVAL,
                                             {"R6": artifact_dir(T2_RUNS / "R6", "risk")})
    complete_sources = evaluated_sources(T2_EVAL, ARM_ORDER, t2_ids)
    order = [a for a in ARM_ORDER if a in set(t2_frame["source"])]
    if not order:
        reasons.append("no task-2 run evaluated yet (GPU needed)")
        print("S9.task2: " + reasons[-1])
    else:
        decision = decide_arms(t2_frame, S4_RULE)
        display(t2_frame.groupby("source")[["mae_hu", "target_mean_signed_mm", "target_rms_mm", "target_cvar_mm",
                                            "target_global_ratio"]].median().reindex(order).round(3))
        display(pd.Series({k: v for k, v in decision.items() if k not in ("arms", "raw_p_values")},
                          name="value").to_frame())
        display(pd.DataFrame(decision["arms"]).T.round(4))
        grouped = t2_frame.groupby("source")
        pareto = pd.concat({f"{m}{s}": grouped[m].quantile(q) for m in ("mae_hu", "target_cvar_mm")
                            for s, q in (("", 0.5), ("_q25", 0.25), ("_q75", 0.75))}, axis=1).reindex(order)
        plot_arm_pareto(pareto).savefig(FIGURE_DIR / "s9_task2_pareto.png", bbox_inches="tight")
        plt.show()
        task1 = json.loads(s4_report_path.read_text()) if s4_report_path.exists() else {}
        task1_verdict = task1.get("decision", {}).get("verdict")
        replicates = task1_verdict is not None and task1_verdict == decision["verdict"]
        print(f"task 2: {decision['verdict']} · task 1 (S4): {task1_verdict}"
              f"{' (preliminary)' if task1.get('preliminary') else ''} · replicates: {replicates}")
        secondary = {"signed bias AB, R1 − R3 > 0": paired_test(t2_frame, "target_mean_signed_mm", "R1", "R3", "AB",
                                                               "greater")}
        display(pd.DataFrame(secondary).T.round(4))
        S9_T2_DONE = complete_sources == set(ARM_ORDER) and decision["verdict"] != "NOT EVALUATED"
        write_report("S9.task2", {"decision": decision, "task1_verdict": task1_verdict, "replicates": replicates,
                                  "secondary": secondary, "complete_sources": sorted(complete_sources),
                                  "risk_calibration": t2_risk.median(numeric_only=True).to_dict()
                                  if not t2_risk.empty else {}})
"""

S9_T2_END = """PIPELINE.finish("S9.task2", complete=S9_T2_DONE, note="; ".join(reasons))
"""

S9_SP_MD = """## S9.spectra · The S3 go/no-go rule on task 2

The S3 diagnostic (low-frequency excess of the SPR error along and across the beam, relative to the same errors
shuffled; RMS range error against shuffled errors at equal MAE) on the task-2 R1 models, which are the L1-in-HU models
of the S3 rule, with the bulk-density baseline. Each case is cached.
"""

S9_SP_BEGIN = """PIPELINE.begin("S9.spectra", {"run": RUN_S9}, report_path("S9.spectra"), after=("S3", "S9.task2"))
"""

S9_SP = """T2_RUNS = OUTPUT_DIR / "s9_task2" / "seed0"
SPECTRA_CACHE = OUTPUT_DIR / "s9_spectra_cache"
reasons = [m for m, bad in (("RUN_S9 off", not RUN_S9),
                            ("task-2 R1 not trained (S9.task2)", not all((T2_RUNS / "R1" / f"fold{k}.pt").exists()
                                                                       for k in range(N_FOLDS)))) if bad]
S9_SP_DONE = False
if reasons:
    print("S9.spectra not run: " + "; ".join(reasons))
else:
    t2_ids = [r.case_id for r in dev["2"]]
    cached = all((SPECTRA_CACHE / f"{i}.pkl").exists() for i in t2_ids)
    if not cached and not run_complete(T2_RUNS / "R1", t2_ids):  # predictions of an earlier session are gone
        crossfit_runs(dev["2"], cache, [(arm_config("R1", seed=0, device=DEVICE, steps=STEPS), T2_RUNS / "R1")],
                      hlut, N_FOLDS, DEVICES)
    rows, spectra = [], {}
    for record in dev["2"]:
        path = artifact_dir(T2_RUNS / "R1") / f"{record.case_id}.mha"

        def compute(record=record, path=path):
            volumes = load_case(record, ("ct", "mask"))
            ct, mask, spacing = volumes["ct"].array, volumes["mask"].array, volumes["ct"].spacing
            out = []
            for source, sct in (("bulk density", bulk_density_sct(mask)), ("l1_hu", read_prediction(path, ct.shape))):
                metrics, case_spectra = diagnose_case(ct, mask, sct, spacing, hlut, DIAGNOSTIC, S3_RULE)
                out.append(({"case_id": record.case_id, "region": record.region, "source": source, **metrics},
                            case_spectra))
            return out

        key = {"prereg": S3_RULE.fingerprint, "config": asdict(DIAGNOSTIC), "case": record.case_id}
        for row, case_spectra in cached_result(SPECTRA_CACHE / f"{record.case_id}.pkl", key, compute):
            rows.append(row)
            spectra.setdefault(row["source"], []).append(case_spectra)
    t2_spectra = pd.DataFrame(rows)
    t2_spectra.to_csv(OUTPUT_DIR / "s9_task2_spectra.csv", index=False)
    decision = decide(t2_spectra, S3_RULE)
    display(pd.Series(decision, name="value").to_frame())
    display(t2_spectra.groupby(["source", "region"])[["mae_hu", "target_rms_mm", "target_global_ratio",
                                                      "lf_excess_along", "lf_excess_across"]].median().round(3))
    plot_beam_spectra(spectra, S3_RULE.cutoff).savefig(FIGURE_DIR / "s9_task2_spectra.png", bbox_inches="tight")
    plt.show()
    s3_path = PATHS.output_root / "s3" / "s3_report.json"
    task1 = json.loads(s3_path.read_text())["decision"]["verdict"] if s3_path.exists() else None
    print(f"task 2: {decision['verdict']} · task 1 (S3): {task1}")
    S9_SP_DONE = decision["verdict"] != "NOT EVALUATED"
    write_report("S9.spectra", {"decision": decision, "task1_verdict": task1,
                                "replicates": task1 is not None and task1 == decision["verdict"]})
"""

S9_SP_END = """PIPELINE.finish("S9.spectra", complete=S9_SP_DONE, note="; ".join(reasons))
"""

S9_TOMO_MD = """## S9.tomography · Sparse-view CT: another line integral

`sctrange.tomography`: relative attenuation u = μ/μ_water from the task-1 reference CTs (2 mm pixels, 256 × 256),
measured at 30 parallel-beam angles with Poisson noise (10⁵ photons per bin), reconstructed by filtered
back-projection (Ram-Lak filter, exact adjoint of the projector) and a residual U-Net trained with L1, L2, or L2 plus
the squared error of line integrals at 8 random angles. Evaluation: 180 angles the scanner never measured, in mm of
water, on the central 16 slices of every case. g is the identity here, so L2 is already the Bayes act for squared
line-integral loss: a gain of `l2_proj` isolates Result A (where a finite model spends its capacity). Tests (Holm over
three): `l2_proj` lowers the line-integral CVaR90 (one-sided), its MAE cost and the L1–L2 signed bias (two-sided).
"""

S9_TOMO_BEGIN = """PIPELINE.begin("S9.tomography", {"steps": TOMO_STEPS, "cases": TOMO_CASES, "train": TRAIN, "run": RUN_S9},
               report_path("S9.tomography"), after=("S1",))
"""

S9_TOMO = """reasons = ["RUN_S9 off"] if not RUN_S9 else []
S9_TOMO_DONE = False
if reasons:
    print("S9.tomography not run: " + "; ".join(reasons))
else:
    spec = PREREG["s9_tomography"]
    base_tomo = TomographyConfig(**{**spec["config"], "steps": TOMO_STEPS}, seed=spec["seed"], device=DEVICE)
    tomo_dirs = {loss: OUTPUT_DIR / "s9_tomography" / loss for loss in spec["losses"]}
    if TRAIN_IN_NOTEBOOK:
        crossfit_tomography_runs(dev["1"], cache, [(replace(base_tomo, loss=loss), d) for loss, d in tomo_dirs.items()],
                                 spec["folds"], DEVICES)
    tomo_runs = {loss: d for loss, d in tomo_dirs.items() if all((d / f"fold{k}.pt").exists() for k in range(spec["folds"]))}
    if len(tomo_runs) < len(tomo_dirs):
        reasons.append(f"reconstructors trained: {sorted(tomo_runs)} of {spec['losses']} (GPU needed)")
        print("S9.tomography: " + reasons[-1])
    if tomo_runs:
        tomo_ids = select_cases(dev["1"], TOMO_CASES or len(dev["1"]), PREREG_S7["subsets"]["seed"])
        tomo_cases = [r for r in dev["1"] if r.case_id in set(tomo_ids)]
        tomo = evaluate_tomography(tomo_cases, cache, tomo_runs, base_tomo, n_slices=16, device=DEVICE,
                                   cache_path=OUTPUT_DIR / "s9_tomography.csv")
        columns = ["mae_hu", "li_mean_signed_mm", "li_rms_mm", "li_cvar_mm", "li_coherence", "li_global_ratio"]
        order = [s for s in ("FBP", *spec["losses"]) if s in set(tomo["source"])]
        display(tomo.groupby("source")[columns].median().reindex(order).round(3))
        raw = {name: paired_test(tomo, t["column"], t["first"], t["second"], alternative=t["alternative"])
               for name, t in spec["tests"].items()}
        adjusted = holm({name: r["p_value"] for name, r in raw.items()})
        tests = pd.DataFrame(raw).T.assign(p_holm=pd.Series(adjusted))
        display(tests.round(4))
        histories = {loss: pd.read_csv(d / "history.csv") for loss, d in tomo_runs.items() if (d / "history.csv").exists()}
        plot_training_curves(histories).savefig(FIGURE_DIR / "s9_tomography_training.png", bbox_inches="tight")
        plt.show()
        case = cache.load(tomo_cases[0].case_id)
        u_np, body_np = case_slices(case, base_tomo, central_indices(case, 1))
        scanner = SparseViewScanner(base_tomo.size, base_tomo.pixel_mm, base_tomo.sparse_angles, base_tomo.photons,
                                    DEVICE)
        folds = pd.read_csv(next(iter(tomo_runs.values())) / "folds.csv", dtype={"case_id": str}).set_index("case_id")
        with torch.no_grad():
            u_t = torch.from_numpy(u_np).to(DEVICE)
            fbp = scanner.fbp(scanner.measure(u_t, torch.Generator(device=DEVICE).manual_seed(0)))
            shown = {"FBP": fbp[0].cpu().numpy()}
            for loss, run_dir in tomo_runs.items():
                fold = int(folds.loc[tomo_cases[0].case_id, "fold"])
                model = _load_reconstructor(run_dir / f"fold{fold}.pt", run_config(run_dir, DEVICE))
                shown[loss] = reconstruct(model, fbp)[0].cpu().numpy()
        plot_tomography_example(u_np[0], shown, body_np[0]).savefig(FIGURE_DIR / "s9_tomography.png",
                                                                    bbox_inches="tight")
        plt.show()
        S9_TOMO_DONE = len(tomo_runs) == len(tomo_dirs) and set(tomo["case_id"]) == set(tomo_ids)
        write_report("S9.tomography", {"tests": tests.to_dict(orient="index"), "cases": len(tomo_ids),
                                       "medians": tomo.groupby("source")[columns].median().to_dict(orient="index")})
"""

S9_TOMO_END = """PIPELINE.finish("S9.tomography", complete=S9_TOMO_DONE, note="; ".join(reasons))
"""

S9_RAIN_MD = """## S9.rainfall · Basin rainfall (not run)

The plan schedules the third domain, rainfall accumulated over a basin (a line integral of radar rain rates in time and
space), only if S6 finished without delay, and it needs an external gauge–radar dataset that this repository does not
provide. It is pre-registered as not run (`s9_precipitation`) rather than replaced by synthetic data; the two domains
above carry the generality claim (the same physics with another modality, and another physics with g linear).
"""

S9_RAIN = """PIPELINE.begin("S9.rainfall", {}, report_path("S9.rainfall"), after=())
"""

S9_RAIN_END = """write_report("S9.rainfall", PREREG["s9_precipitation"])
print(PREREG["s9_precipitation"]["reason"])
PIPELINE.finish("S9.rainfall", complete=True, note="not run by design")
"""

S10_AB_MD = """# S10 · Ablations

## S10.ablations · Directions, segment length, λ and β

Every variant changes one setting of R5 (or R6 for β) and keeps the rest: data, folds (seed 0, the S4 partition),
steps, backbone. The centre runs are S4's seed-0 R5 and R6, only read here. All variants share one fold queue. Each
listed comparison is a paired two-sided Wilcoxon over the development cases, Holm within its family, with the median
paired difference and its bootstrap interval.

| Family | Variants |
|---|---|
| directions | plan beams only (3) · 3 random · 6 random · centre: 3 plan + 3 random |
| segment | first 40 mm · first 80 mm from the entry · centre: whole chord in the patch |
| λ (range weight) | 1e-5 · 1e-3 · centre 1e-4 |
| β (risk weight) | 1e-3 · 1e-1 · centre 1e-2 |
"""

S10_AB_BEGIN = """PIPELINE.begin("S10.ablations", {"steps": STEPS, "folds": N_FOLDS, "cases": ABLATION_CASES, "train": TRAIN,
                                "run": RUN_S10}, report_path("S10.ablations"), after=("S4",))
"""

S10_AB = """ABLATION_RUNS = OUTPUT_DIR / "s10_ablations"
reasons = [m for m, bad in (("RUN_S10 off", not RUN_S10), ("S4 seed 0 R3/R5/R6 not complete", not HAVE_S4)) if bad]
S10_AB_DONE = False
if reasons:
    print("S10.ablations not run: " + "; ".join(reasons))
else:
    spec = PREREG["s10_ablations"]
    grid = ablation_grid(spec)
    ablation_ids = select_cases(dev["1"], ABLATION_CASES or len(dev["1"]), PREREG_S7["subsets"]["seed"],
                                lambda r: s4_prediction("R5", r).exists() and s4_prediction("R6", r).exists())
    ablation_cases = [r for r in dev["1"] if r.case_id in set(ablation_ids)]
    run_dirs, configs = {}, {}
    for ablation in grid:
        key = (ablation.family, ablation.name)
        run_dirs[key] = (S4_RUNS / "seed0" / ablation.arm if ablation.center
                         else ABLATION_RUNS / ablation.family / ablation.name.replace(" ", "_"))
        configs[key] = arm_config(ablation.arm, seed=spec["seed"], device=DEVICE, steps=STEPS, **ablation.overrides)
    pending = []
    for (family, name), run_dir in run_dirs.items():
        ablation = next(a for a in grid if (a.family, a.name) == (family, name))
        evaluated = name in evaluated_sources(OUTPUT_DIR / f"s10_{family}.csv", [name], ablation_ids)
        if not ablation.center and not evaluated:
            pending.append(((family, name), run_dir))
    if pending and TRAIN_IN_NOTEBOOK:
        crossfit_runs(dev["1"], cache, [(configs[key], run_dir) for key, run_dir in pending], hlut, N_FOLDS, DEVICES)
    tables = []
    for family, family_spec in spec["families"].items():
        names = [name for (f, name) in run_dirs if f == family]
        cache_csv = OUTPUT_DIR / f"s10_{family}.csv"
        evaluated = evaluated_sources(cache_csv, names, ablation_ids)
        sources = {name: artifact_dir(run_dirs[(family, name)]) for name in names
                   if name in evaluated or run_complete(run_dirs[(family, name)], [r.case_id for r in dev["1"]])}
        if len(sources) < 2:
            continue
        frame, _ = evaluate_predictions(ablation_cases, sources, hlut, DIAGNOSTIC, cache_csv)
        comparisons = [pair for pair in family_spec["comparisons"] if set(pair) <= set(sources)]
        tables.append(compare_family(frame, comparisons, spec["metrics"]).assign(family=family))
        display(frame.groupby("source")[spec["metrics"]].median().round(3).rename_axis(columns=family))
    expected = sum(len(f["comparisons"]) for f in spec["families"].values()) * len(spec["metrics"])
    if tables:
        ablations = pd.concat(tables, ignore_index=True)
        ablations.to_csv(OUTPUT_DIR / "s10_ablations.csv", index=False)
        display(ablations[["family", "variant", "reference", "metric", "cases", "median_difference", "ci_low",
                           "ci_high", "p_value", "p_holm"]].round(4))
        labels = {"target_cvar_mm": "CVaR90 |ΔWET| (mm)", "target_rms_mm": "RMS ΔWET (mm)", "mae_hu": "MAE (HU)"}
        plot_ablation_forest(ablations, spec["metrics"], labels).savefig(FIGURE_DIR / "s10_ablations.png",
                                                                         bbox_inches="tight")
        plt.show()
        S10_AB_DONE = len(ablations) == expected and (ablations["cases"] == len(ablation_ids)).all()
        if not S10_AB_DONE:
            reasons.append(f"{len(ablations)} of {expected} comparisons available")
        write_report("S10.ablations", {"cases": len(ablation_ids), "comparisons": ablations.to_dict(orient="records")})
    else:
        reasons.append("no ablation variant trained yet (GPU needed)")
        print("S10.ablations: " + reasons[-1])
"""

S10_AB_END = """PIPELINE.finish("S10.ablations", complete=S10_AB_DONE, note="; ".join(reasons))
"""

S10_REG_MD = """## S10.registration · Registration cross-matrix (Elastix / IMPACT)

SynthRAD2025's reference CTs are registered to the source with Elastix; IMPACT (Boussot et al., 2025) realigns them. A
dataset overlay on the session disk takes `ct.mha` from `IMPACT_ROOT` and every other file from the original data, for
the task-1 development cases whose IMPACT CT shares the grid. R3 is cross-fitted on the overlay (seed 0), and both R3
models (Elastix: S4 seed 0; IMPACT: here) are scored against both reference CTs. The off-diagonal cells measure how
much of a model's apparent error, and of its range error, is registration.
"""

S10_REG_BEGIN = """PIPELINE.begin("S10.registration", {"steps": STEPS, "folds": N_FOLDS, "impact": str(IMPACT_ROOT), "train": TRAIN,
                                   "run": RUN_S10}, report_path("S10.registration"), after=("S4",))
"""

S10_REG = """reasons = [m for m, bad in (("RUN_S10 off", not RUN_S10), ("IMPACT_ROOT not set", IMPACT_ROOT is None),
                            ("S4 seed 0 R3 not complete", not HAVE_S4)) if bad]
S10_REG_DONE = False
if reasons:
    print("S10.registration not run: " + "; ".join(reasons))
else:
    spec = PREREG["s10_registration"]
    overlay_root = PATHS.cache_root / "impact_overlay"
    overlay = registration_overlay(DATA_ROOT, IMPACT_ROOT, overlay_root, [r.case_id for r in dev["1"]])
    overlay.to_csv(OUTPUT_DIR / "s10_registration_overlay.csv", index=False)
    display(overlay["reason"].replace("", "kept").value_counts().rename("cases").to_frame())
    impact_records = [r for r in discover_cases(overlay_root) if r.case_id in development_ids and r.task == "1"]
    impact_ids = [r.case_id for r in impact_records]
    impact_cache = VolumeCache(PATHS.cache_root / "cache_impact")
    impact_cache.build(impact_records)
    impact_run = OUTPUT_DIR / "s10_registration" / "R3_impact"
    impact_config = arm_config(spec["arm"], seed=spec["seed"], device=DEVICE, steps=STEPS)
    evaluated = all(m in evaluated_sources(OUTPUT_DIR / f"s10_registration_test_{t}.csv", [m], impact_ids)
                    for t in ("Elastix", "IMPACT") for m in ("Elastix", "IMPACT"))
    if not evaluated and (TRAIN_IN_NOTEBOOK or all((impact_run / f"fold{k}.pt").exists() for k in range(N_FOLDS))):
        crossfit_runs(impact_records, impact_cache, [(impact_config, impact_run)], hlut, N_FOLDS, DEVICES)
    models = {"Elastix": artifact_dir(S4_RUNS / "seed0" / spec["arm"]), "IMPACT": artifact_dir(impact_run)}
    if not evaluated and not run_complete(impact_run, impact_ids):
        reasons.append("R3 on IMPACT not trained (GPU needed)")
        print("S10.registration: " + reasons[-1])
    else:
        references = {"Elastix": {r.case_id: r for r in dev["1"]}, "IMPACT": {r.case_id: r for r in impact_records}}
        parts = []
        for test, by_id in references.items():
            frame, _ = evaluate_predictions([by_id[i] for i in impact_ids], models, hlut, DIAGNOSTIC,
                                            OUTPUT_DIR / f"s10_registration_test_{test}.csv")
            parts.append(frame.rename(columns={"source": "train"}).assign(test=test))
        matrix_frame = pd.concat(parts, ignore_index=True)
        matrix_frame["source"] = "trained on " + matrix_frame["train"] + ", " + matrix_frame["test"] + " test"
        matrix_frame.to_csv(OUTPUT_DIR / "s10_registration.csv", index=False)
        labels = {"mae_hu": "MAE (HU)", "target_rms_mm": "RMS ΔWET (mm)", "target_cvar_mm": "CVaR90 |ΔWET| (mm)",
                  "target_mean_signed_mm": "signed ΔWET (mm)"}
        matrices = {label: registration_matrix(matrix_frame, metric) for metric, label in labels.items()}
        for label, matrix in matrices.items():
            display(matrix.round(3).rename_axis(index=f"{label} · train", columns="test"))
        plot_registration_matrix({k: v for k, v in matrices.items() if "signed" not in k}).savefig(
            FIGURE_DIR / "s10_registration.png", bbox_inches="tight")
        plt.show()
        tests = compare_family(matrix_frame, spec["tests"], spec["metrics"])
        display(tests.round(4))
        S10_REG_DONE = matrix_frame.groupby("source")["case_id"].nunique().eq(len(impact_ids)).all() and \\
            matrix_frame["source"].nunique() == 4
        write_report("S10.registration", {"cases": len(impact_ids), "overlay_excluded": int((~overlay["kept"]).sum()),
                                          "matrices": {k: v.to_dict() for k, v in matrices.items()},
                                          "tests": tests.to_dict(orient="records")})
"""

S10_REG_END = """PIPELINE.finish("S10.registration", complete=S10_REG_DONE, note="; ".join(reasons))
"""

STATE_MD = """## State and summary

Every stage, its state (S1–S7 included when they ran through `s1_s7_pipeline`), the summary of the S8–S10 reports on
disk, and the space used by the outputs. This cell always runs.
"""

STATE = """PIPELINE.show()
stages = ["S8.dose", "S8.task2", "S9.task2", "S9.spectra", "S9.tomography", "S9.rainfall", "S10.ablations",
          "S10.registration"]
state = {entry["stage"]: entry for entry in PIPELINE.summary()}


def headline(stage: str, report: dict) -> str:
    if stage == "S8.dose" and "test" in report:
        return f"γ 2/2 G − R3: median {report['test']['median_difference']:.2f} %, p = {report['test']['p_value']:.3g}"
    if stage == "S8.task2" and "summary" in report:
        task2 = report["summary"].get("task 2", {})
        return f"coverage90 {task2.get('coverage90', float('nan')):.2f}, projected CRPS {task2.get('pcrps_mm', float('nan')):.2f} mm"
    if stage in ("S9.task2", "S9.spectra") and "decision" in report:
        return f"{report['decision']['verdict']} (task 1: {report.get('task1_verdict')})"
    if stage == "S9.tomography" and "tests" in report:
        return ", ".join(f"{name}: p_holm {row['p_holm']:.3g}" for name, row in report["tests"].items())
    if stage == "S10.ablations" and "comparisons" in report:
        significant = sum(1 for row in report["comparisons"] if row.get("p_holm", 1) < 0.05)
        return f"{len(report['comparisons'])} comparisons, {significant} with Holm p < 0.05"
    if stage == "S10.registration" and "cases" in report:
        return f"{report['cases']} cases"
    return report.get("status", "")


rows = []
for stage in stages:
    path = report_path(stage)
    report = json.loads(path.read_text()) if path.exists() else {}
    rows.append({"stage": stage, "status": state.get(stage, {}).get("status", "not started"),
                 "note": state.get(stage, {}).get("note", ""), "result": headline(stage, report),
                 "quick pass": report.get("quick_pass")})
summary = pd.DataFrame(rows)
display(summary)
(OUTPUT_DIR / "s8_s10_report.json").write_text(json.dumps(
    {"stages": summary.to_dict(orient="records"),
     "reports": {s: json.loads(report_path(s).read_text()) for s in stages if report_path(s).exists()}},
    indent=2, default=float))


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9 if path.exists() else 0.0


usage = {p.name: size_gb(p) for p in sorted(PATHS.output_root.iterdir()) if p.is_dir()}
display(pd.Series(usage, name="GB").round(2).to_frame().T)
free = shutil.disk_usage(PATHS.output_root).free / 1e9
print(f"outputs: {sum(usage.values()):.1f} GB used · {free:.1f} GB free on that disk")
"""

READING_MD = """## Reading the results

| Result | Meaning for the paper |
|---|---|
| S8.dose: no γ difference between G and R3 | Consistent with S7 (the generator's mean equals the Bayes act at best), now in dose. |
| S8.task2: calibration close to task 1 | The generative arm's distributional claim holds for CBCT → CT. |
| S9.task2 / S9.spectra: verdicts equal to task 1 | The range results replicate on a second modality; a different verdict is reported as a domain effect. |
| S9.tomography: `l2_proj` lowers the line-integral tail at similar MAE | Result A holds where g is linear: the gain is spectral, not an estimator effect. |
| S10.ablations | Which settings matter, and how robust the S4 verdict is to them; fixes the configuration opened on the test set. |
| S10.registration | How much of the apparent sCT error and of the range error is registration; read every result against it. |

Week 11 opens the frozen test set once, with the arms, settings and generator fixed by S4–S10.
"""


def cell(kind: str, text: str) -> dict:
    new = {"cell_type": kind, "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        new.update(execution_count=None, outputs=[])
    return new


def build() -> dict:
    template = json.loads((NOTEBOOKS / "s7_generative.ipynb").read_text(encoding="utf-8"))
    setup_md, setup_code = ("".join(template["cells"][i]["source"]) for i in (1, 2))
    stages = [
        (S8_DOSE_MD, S8_DOSE_BEGIN, S8_DOSE, S8_DOSE_END),
        (S8_T2_MD, S8_T2_BEGIN, S8_T2, S8_T2_END),
        (S9_T2_MD, S9_T2_BEGIN, S9_T2, S9_T2_END),
        (S9_SP_MD, S9_SP_BEGIN, S9_SP, S9_SP_END),
        (S9_TOMO_MD, S9_TOMO_BEGIN, S9_TOMO, S9_TOMO_END),
        (S9_RAIN_MD, S9_RAIN, S9_RAIN_END),
        (S10_AB_MD, S10_AB_BEGIN, S10_AB, S10_AB_END),
        (S10_REG_MD, S10_REG_BEGIN, S10_REG, S10_REG_END),
    ]
    cells = [cell("markdown", INTRO), cell("markdown", setup_md), cell("code", setup_code),
             cell("markdown", PARAMS_MD), cell("code", PARAMS), cell("markdown", RESUME_MD), cell("code", RESUME),
             cell("markdown", CONTEXT_MD), cell("code", IMPORTS), cell("code", CONTEXT)]
    for parts in stages:
        cells.append(cell("markdown", parts[0]))
        cells.extend(cell("code", text) for text in parts[1:])
    cells += [cell("markdown", STATE_MD), cell("code", STATE), cell("markdown", READING_MD)]
    for index, item in enumerate(cells):
        item["id"] = f"s8s10-{index:03d}"
    return {"cells": cells, "metadata": template["metadata"], "nbformat": 4, "nbformat_minor": 5}


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    (NOTEBOOKS / NAME).write_text(render(), encoding="utf-8")
    print(f"wrote notebooks/{NAME}", file=sys.stderr)
