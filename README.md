# sctrange

Range-aware evaluation and training utilities for synthetic CT (sCT) on SynthRAD2025.

The central object is a differentiable projector that turns an sCT error into a proton
water-equivalent range error, ray by ray, using the same HU → SPR conversion as the dose
engine of the SynthRAD2025 evaluation (pyRadPlan).


## Notebooks

Click a badge to open a notebook directly on Colab or Kaggle, then run all cells: the first cell installs `sctrange`
from this repository, the parameters cell (a form on Colab) sets the run, and every later step — fetching the real
SynthRAD2025 data from Hugging Face, frozen split, training, evaluation, verdict — runs from the notebook.

| Notebook | Plan | What it does | Hardware | Open |
|---|---|---|---|---|
| `s1_harness` | S1 | data integrity, frozen split, HLUT and projector checks on a real patient, range diagnostics on the real data | CPU is enough | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s1_harness.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s1_harness.ipynb) |
| `s2_synthetic_benchmark` | S2 | checks of `docs/theory.md` with a known posterior, then its identities on real patients | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s2_synthetic_benchmark.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s2_synthetic_benchmark.ipynb) |
| `s3_real_diagnostic` | S3 | cross-fitted L1-HU and L2-SPR models, beam spectra, pre-registered go/no-go 1 | GPU (T4 or 2 × T4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s3_real_diagnostic.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s3_real_diagnostic.ipynb) |
| `s4_regression_arms` | S4–S5 | arms R1–R6 over three seeds, Pareto front, R6 risk calibration, pre-registered verdict | GPU (T4 or 2 × T4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s4_regression_arms.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s4_regression_arms.ipynb) |
| `s6_dosimetry` | S6 | proton and photon plans on 20 cases recomputed on R1–R6, proxy validation, pre-registered go/no-go 2 | CPU (several hours, resumable) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s6_dosimetry.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s6_dosimetry.ipynb) |
| `s7_generative` | S7–S8 | stochastic-interpolant flow matching, η sweep, step curve, projected CRPS and calibration, comparison with R3 and R6 | GPU (T4 or 2 × T4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s7_generative.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s7_generative.ipynb) |

| `s1_s7_pipeline` | S1–S8 | the six notebooks above in order, one setup and one parameters cell, same outputs | GPU for S3, S4, S7 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/s1_s7_pipeline.ipynb) [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/s1_s7_pipeline.ipynb) |

Run S1 first (with both tasks): it freezes the local test split that S2–S7 verify. S6 and S7 use the S4 runs.
`s1_s7_pipeline` runs every stage in one go; it writes to the same `outputs/s<k>/` folders as the stage notebooks, so
a stage can be resumed in either.

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
  attached `outputs/` back before starting. Predictions are
  stored as int16 HU to fit the 20 GB of `/kaggle/working`.
- The data come from the Hugging Face mirror (about 14 GB per task, fetched once per session, resumable).

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
[-1024, 3071] HU; CC BY-NC 4.0): task 1 (MR → CT, 513 cases) and task 2 (CBCT → CT, 774 cases), about 27 GB.
The code reads it only from its public Hugging Face mirror,
[developper669/synthrad2025](https://huggingface.co/datasets/developper669/synthrad2025) (tag `v2`), which holds the
extracted cases in the layout the code expects and the integrity check of every case:

```
Task<k>/<region>/<case_id>/{ct,mask,mr|cbct}.mha    Task<k>/<region>/overview/...    manifest.csv
```

```bash
sctrange-download --root /data/synthrad2025 --task all          # both tasks
sctrange-download --root /data/synthrad2025 --task 1 --region HN  # a subset
export SYNTHRAD2025_ROOT=/data/synthrad2025
```

Downloads are parallel and resumable, files already present are skipped, and the revision is pinned, so every run
sees the same files. In the notebooks `ensure_dataset` fetches the missing task on first use, then works offline.
`SCT_HF_DATASET` and `SCT_HF_REVISION` select another repository or revision; `--validate` re-checks every case
locally instead of trusting the published `manifest.csv`. The mirror was built once from Zenodo (archives checked
against the published MD5, cases checked, every uploaded file checked against its size).
## Week 1 (S1)

1. Point the harness at the data and run it:
   ```bash
   export SYNTHRAD2025_ROOT=/path/to/synthrad2025      # contains Task1/, Task2/
   export SCT_PREDICTION_ROOT=/path/to/predictions     # optional, <case_id>.mha
   jupyter nbconvert --to notebook --execute --inplace notebooks/s1_harness.ipynb
   ```
   Every notebook runs on the real data. When `SYNTHRAD2025_ROOT` (default `data/synthrad2025`)
   lacks a task, they fetch it from Hugging Face first; `SCT_DOWNLOAD=0` makes them stop with an error instead.
   `SCT_TASKS` (S1, default `1,2`) and `SCT_TASK` (S2, S3, S4) select the tasks; `SCT_MAX_CASES` limits the S1 diagnostics
   and the S2 real-data section (`N_CASES`). Only two kinds of check use constructed inputs, because they need an exact answer: the analytic geometry checks
   of S1 (disk chords, target segments; the projector is then compared with the Siddon tracer on a patient) and
   sections 1–6 of S2 (a known posterior, which no patient dataset provides). The unit tests use small analytic
   phantoms (`sctrange.phantoms`) so that they run without the 27 GB dataset.
2. Freeze the local test split once, on the real data, and commit it (the first S1 run on the real data creates it):
   ```bash
   sctrange-split --data-root "$SYNTHRAD2025_ROOT" --out splits/synthrad2025_local_test.json
   git add splits/ && git commit -m "Freeze SynthRAD2025 local test split"
   ```
   The notebook verifies this file on every run and refuses a modified copy.

## Week 2 (S2)

`notebooks/s2_synthetic_benchmark.ipynb` checks every statement of `docs/theory.md` against an explicit criterion on
synthetic data with a known posterior (sections 1–6, about 30 s on a laptop CPU), then checks the statements that are
identities (Propositions 1, 1′, 2, 3) on the SPR errors of real patients and measures the coherence of real errors
(section 7: development cases of the frozen split, bulk-density baseline and any S3/S4 prediction). The verdicts go to
`outputs/s2/s2_report.json`, the real-data table to `outputs/s2/real_identities.csv`:

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
paired Wilcoxon of target CVaR90 against R3 and R4, Holm-adjusted, median MAE at most 5 % above R3; version 2 of the
rule, amended before any real result: version 1 also rejected arms that improve MAE by more than 5 %). It stops with a clear message
until the arms have been trained.

Every arm shares the backbone, data, folds, number of steps and seed; the range and risk terms only add projector
work per step. λ = 1e-4 mm⁻² and β = 1e-2 are starting values that put the extra terms at the scale of the SPR loss;
the S10 ablations sweep them (`--range-weight`, `--risk-weight`). Range terms use body chords inside each training patch
(rays shorter than 20 mm ignored); evaluation keeps the entry-to-distal-target segments of `sctrange.evaluation`.
R6 also writes `risk/<case_id>_{var,excess}.mha` densities; `sctrange.evaluation.evaluate_risk` integrates them along
the evaluation segments and reports coverage (target 0.9), realised versus predicted ES and the FZ0 score.
A fold checkpoint trained with a different configuration is refused rather than silently reused.

## Week 6 (S6): first real dosimetry and go/no-go 2

```bash
sctrange-dose --data-root "$SYNTHRAD2025_ROOT" --split splits/synthrad2025_local_test.json \
  --runs-root outputs/s4/runs --out outputs/s6 --workers 4
jupyter nbconvert --to notebook --execute --inplace notebooks/s6_dosimetry.ipynb
```

`sctrange.dosimetry` optimises an IMPT plan (3 beams, the plan angles of R5/R6) and an IMRT plan (7 beams) on the
reference CT of each case with pyRadPlan, then recomputes the same weights on the bulk-density baseline and on every
S4 arm and seed. The 20 cases are drawn once (region-stratified) and frozen with their hash in
`outputs/s6/dosimetry_cases.json`. Per source: γ 2 %/2 mm and 1 %/1 mm (`sctrange.gamma`, global, 10 % cutoff),
target DVH differences, and for protons the distal R80 shift of every beam along every ray, matched with the WET error
of the same ray. The notebook first checks the dose engine on a water cylinder (beam direction, identity
recomputation), then validates the range proxy against dose and applies `configs/s6_preregistration.json`, which
combines the S4 range verdict with a proton γ test (GO, RANGE GAIN WITHOUT DOSE CONFIRMATION, RIGHT ESTIMATOR
SUFFICES, NO-GO). Beam angles follow the projector convention; pyRadPlan's gantry angle is `(360 − angle) mod 360`.

## Weeks 7–8 (S7): generative arm

```bash
sctrange-generative --data-root "$SYNTHRAD2025_ROOT" --split splits/synthrad2025_local_test.json \
  --cache-dir outputs/s4/cache --out outputs/s7/runs --eta 0 0.05 0.1 0.2 --task 1 --folds 5
jupyter nbconvert --to notebook --execute --inplace notebooks/s7_generative.ipynb
```

`sctrange.generative` trains conditional flow-matching generators with a stochastic interpolant
`x_t = (1 − t) x₀ + t x₁ + 2η t(1 − t) z` from the bulk-density image `x₀` to the CT `x₁`, on the U-Net of R1–R6 with
FiLM time conditioning, cross-fitted like the regression arms. The notebook sweeps η, selects it by the pre-registered
rule (lowest median projected CRPS), draws the step curve (1–32 steps), evaluates the SPR mean of the samples with the
S4 metrics and the ensembles with projected CRPS, 90 % interval coverage, PIT and spread–error correlation, then tests
the generator against R3 (mean, proper score) and R6 (tail score, tail calibration) as fixed in
`configs/s7_preregistration.json`.

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
