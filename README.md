# sctrange

Range-aware evaluation and training utilities for synthetic CT (sCT) on SynthRAD2025.

The central object is a differentiable projector that turns an sCT error into a proton
water-equivalent range error, ray by ray, using the same HU → SPR conversion as the dose
engine of the SynthRAD2025 evaluation (pyRadPlan).


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
   Without `SYNTHRAD2025_ROOT`, the notebook generates a synthetic fixture so every cell runs;
   fixture numbers validate the pipeline only.
2. Freeze the local test split once, on the real data, and commit it:
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
paired Wilcoxon of target CVaR90 against R3 and R4, Holm-adjusted, MAE within 5 %). Without the dataset it trains
tiny models on the fixture in about five minutes on a CPU.

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
