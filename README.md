# sctrange

Range-aware evaluation and training utilities for synthetic CT (sCT) on SynthRAD2025.

The central object is a differentiable projector that turns an sCT error into a proton
water-equivalent range error, ray by ray, using the same HU → SPR conversion as the dose
engine of the SynthRAD2025 evaluation (pyRadPlan).


## Notebooks

Click a badge to open a notebook directly on Colab or Kaggle, then run all cells: the first cell installs `sctrange`
from this repository, the parameters cell (a form on Colab) sets the run, and every later step — download of the real
SynthRAD2025 data, frozen split, training, evaluation, verdict — runs from the notebook.

| Notebook | Plan | What it does | Hardware | Open |
|---|---|---|---|---|
| `s1_harness` | S1 | HLUT and projector checks, data integrity, frozen split, range diagnostics on the real data | CPU is enough | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s1_harness.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s1_harness.ipynb) |
| `s2_synthetic_benchmark` | S2 | numerical checks of `docs/theory.md` with a known posterior (no dataset) | CPU, ~30 s | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s2_synthetic_benchmark.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s2_synthetic_benchmark.ipynb) |
| `s3_real_diagnostic` | S3 | cross-fitted L1-HU and L2-SPR models, beam spectra, pre-registered go/no-go 1 | GPU (T4 or 2 × T4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s3_real_diagnostic.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s3_real_diagnostic.ipynb) |
| `s4_regression_arms` | S4–S5 | arms R1–R6 over three seeds, Pareto front, R6 risk calibration, pre-registered verdict | GPU (T4 or 2 × T4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s4_regression_arms.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s4_regression_arms.ipynb) |

Run S1 first (with both tasks): it freezes the local test split that S3 and S4 verify.

| Platform | Dataset and cache | Outputs | Settings |
|---|---|---|---|
| Colab | `/content/data`, `/content/cache` | Google Drive `MyDrive/sct-range/outputs` (`USE_DRIVE`) | *Runtime → Change runtime type → T4 GPU* |
| Kaggle | `/kaggle/tmp/data`, `/kaggle/tmp/cache` | `/kaggle/working/outputs` only | *Accelerator → GPU T4 × 2*, *Internet on* |
| local | `data/synthrad2025`, `outputs/` | `outputs/` | every visible GPU |

- **Two GPUs** (Kaggle T4 × 2): missing folds train in parallel, one process per GPU; prediction runs on the first.
- **T4**: no native bfloat16, so training uses float16 autocast with loss scaling (bfloat16 on Ampere and newer).
- **Sessions end**: every fold is checkpointed in the outputs; rerunning the notebook resumes where it stopped. On
  Colab the outputs stay in Drive. On Kaggle, *Save Version* keeps `/kaggle/working`; to continue, add that version's
  output as an input (*Add Input → Your Work*): when `/kaggle/working/outputs` is empty, the notebook copies the
  attached `outputs/` back before starting. Archives are deleted after extraction on Colab and Kaggle; predictions are
  stored as int16 HU to fit the 20 GB of `/kaggle/working`.
- The data download takes about 27 GB from Zenodo; it resumes after an interruption.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dose,dev]"
pytest
```

`pyradplan` is optional for the package and required for the HLUT equivalence and Siddon checks (the S1 notebook
reports them as not run without it).

Every notebook starts with a setup cell that installs `sctrange` itself, so no manual step is needed:

- inside a clone, it runs `pip install -e ".[dose]"` on that clone (only when `sctrange` is missing or installed from
  elsewhere), so the notebook always executes the repository's code, local edits included;
- opened on its own (Colab, another machine), it downloads the repository from
  [GitHub](https://github.com/hashirama21/sct-range) as an archive (no git needed), then installs it the same way.
  `SCTRANGE_REF` selects a branch, tag or commit (default `main`), `SCTRANGE_HOME` the destination, and
  `SCTRANGE_REINSTALL=1` forces a reinstall.

The cell then checks that the imported `sctrange` is the repository's copy.
SynthRAD2025 dosimetry used pyRadPlan 0.2.8; its default HLUT is identical to that of 0.5.0.

## Data

The training set is Zenodo record [15373853](https://doi.org/10.5281/zenodo.15373853) (v2, CT/CBCT clipped to
[-1024, 3071] HU; CC BY-NC 4.0): `synthRAD2025_Task1_Train.zip` (MR → CT, 513 cases, 14.8 GB) and
`synthRAD2025_Task2_Train.zip` (CBCT → CT, 774 cases, 12.0 GB).

```bash
sctrange-download --list                                   # files, sizes and MD5 from the Zenodo API
sctrange-download --root /data/synthrad2025 --task all     # ~27 GB of archives, about as much once extracted
export SYNTHRAD2025_ROOT=/data/synthrad2025
```

The command streams each archive to `<root>/archives` with HTTP range resumption (rerun it after an interruption),
checks the MD5 published by Zenodo, extracts to `<root>/Task<k>/<region>/<case_id>/{ct,mask,mr|cbct}.mha` (overview
images and parameter sheets go to `<root>/Task<k>/<region>/overviews/`), then checks every case (files present,
identical geometry, isotropic in-plane spacing, CT within [-1024, 3071] HU) and writes `<root>/manifest.csv`. It exits
with a non-zero status if any case fails. `--delete-archives` frees the zips after extraction; `--skip-download`
re-extracts zips already present.

## Week 1 (S1)

1. Point the harness at the data and run it:
   ```bash
   export SYNTHRAD2025_ROOT=/path/to/synthrad2025      # contains Task1/, Task2/
   export SCT_PREDICTION_ROOT=/path/to/predictions     # optional, <case_id>.mha
   jupyter nbconvert --to notebook --execute --inplace notebooks/s1_harness.ipynb
   ```
   The notebooks S1, S3 and S4 run on the real data only. When `SYNTHRAD2025_ROOT` (default `data/synthrad2025`)
   holds no case, they download and prepare it first; `SCT_DOWNLOAD=0` makes them stop with an error instead.
   `SCT_TASKS` (S1, default `1,2`) and `SCT_TASK` (S3, S4) select the tasks; `SCT_MAX_CASES` limits the S1 diagnostics.
   S2 alone stays synthetic: its checks need the exact posterior, which no patient dataset provides. The unit tests
   use small analytic phantoms (`sctrange.phantoms`) so that they run without the 27 GB dataset.
2. Freeze the local test split once, on the real data, and commit it (the first S1 run on the real data creates it):
   ```bash
   sctrange-split --data-root "$SYNTHRAD2025_ROOT" --out splits/synthrad2025_local_test.json
   git add splits/ && git commit -m "Freeze SynthRAD2025 local test split"
   ```
   The notebook verifies this file on every run and refuses a modified copy.

## Week 2 (S2)

`notebooks/s2_synthetic_benchmark.ipynb` checks every statement of `docs/theory.md` against an explicit criterion on
synthetic data with a known posterior, and writes the verdicts to `outputs/s2/s2_report.json`. It needs no dataset and
runs in about 30 s on a laptop CPU:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/s2_synthetic_benchmark.ipynb
```

## Week 3 (S3)

Train cross-fitted models on the development cases (GPU recommended), then run the diagnostic notebook:

```bash
for loss in l1_hu l2_spr; do
  sctrange-train --data-root "$SYNTHRAD2025_ROOT" --split splits/synthrad2025_local_test.json \
    --cache-dir outputs/s3/cache --out outputs/s3/runs --loss $loss --task 1 --folds 5
done
jupyter nbconvert --to notebook --execute --inplace notebooks/s3_real_diagnostic.ipynb
```

The verdict follows `configs/s3_preregistration.json`; do not edit that file once real predictions exist. Test cases are
never loaded. The cache takes roughly 60 MB per case.

## Weeks 4–5 (S4–S5): regression arms at a matched budget

| Arm | Objective | Command |
|---|---|---|
| R1 | L1 in HU | `--arm R1` |
| R2 | L2 in HU | `--arm R2` |
| R3 | L2 in SPR (Bayes act, Proposition 4) | `--arm R3` |
| R4 | R3 + per-tissue SPR offset, cross-fitted on out-of-fold predictions | `--arm R4` (after R3, no training) |
| R5 | R3 + λ · mean ΔWET² over plan beams (0°, 120°, 240°) and 3 random directions per step | `--arm R5` |
| R6 | R5 + β · FZ0 score of a per-ray (VaR₀.₉, ES₀.₉) head for \|ΔWET\| | `--arm R6` |

```bash
for seed in 0 1 2; do
  for arm in R1 R2 R3 R5 R6; do
    sctrange-train --data-root "$SYNTHRAD2025_ROOT" --split splits/synthrad2025_local_test.json \
      --cache-dir outputs/s4/cache --out outputs/s4/runs/seed$seed --arm $arm --task 1 --folds 5 --seed $seed
  done
  sctrange-train --data-root "$SYNTHRAD2025_ROOT" --split splits/synthrad2025_local_test.json \
    --cache-dir outputs/s4/cache --out outputs/s4/runs/seed$seed --arm R4 --task 1
done
jupyter nbconvert --to notebook --execute --inplace notebooks/s4_regression_arms.ipynb
```

`notebooks/s4_regression_arms.ipynb` evaluates every arm and seed, draws the MAE × CVaR Pareto front (plan figure 4),
checks the calibration of the R6 risk head and applies the rule fixed in `configs/s4_preregistration.json` (one-sided
paired Wilcoxon of target CVaR90 against R3 and R4, Holm-adjusted, MAE within 5 %). It stops with a clear message
until the arms have been trained.

Every arm shares the backbone, data, folds, number of steps and seed; the range and risk terms only add projector
work per step. λ = 1e-4 mm⁻² and β = 1e-2 are starting values that put the extra terms at the scale of the SPR loss;
the S10 ablations sweep them (`--range-weight`, `--risk-weight`). Range terms use body chords inside each training patch
(rays shorter than 20 mm ignored); evaluation keeps the entry-to-distal-target segments of `sctrange.evaluation`.
R6 also writes `risk/<case_id>_{var,excess}.mha` densities; `sctrange.evaluation.evaluate_risk` integrates them along
the evaluation segments and reports coverage (target 0.9), realised versus predicted ES and the FZ0 score.
A fold checkpoint trained with a different configuration is refused rather than silently reused.

## Conventions

- Arrays are `(Z, Y, X)`; spacing is `(x, y, z)` in mm, as returned by SimpleITK. In-plane spacing must be isotropic.
- Beams are coplanar in the axial plane and parallel. Angle 0 travels along +row (+y), angle 90 along +column (+x).
- `ΔWET = Σ (SPR(sCT) − SPR(CT)) · Δx` along each ray, from beam entry to the body exit (`chord`) or to the
  distal edge of the target (`target`), in mm of water.

## Known limitations

- SynthRAD2025 releases no PTV; the harness uses a spherical surrogate target at the body centroid.
- The HLUT is pyRadPlan's generic default, chosen for consistency with the challenge dose engine, not a
  scanner-specific stoichiometric calibration.
- Non-coplanar beams, beam divergence and lateral scattering are not modelled; the projector is a range proxy,
  not a dose engine.
