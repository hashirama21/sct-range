"""Build ``notebooks/s1_s7_pipeline.ipynb`` from the stage notebooks S1-S4, S6 and S7.

The merged notebook is generated, never edited by hand: change a stage notebook, then run

    python tools/build_pipeline_notebook.py

``tests/test_notebooks.py`` fails while the committed notebook differs from what this script produces.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
NAME = "s1_s7_pipeline.ipynb"
BADGES = (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
          f"(https://colab.research.google.com/github/hashirama21/sct-range/blob/main/notebooks/{NAME}) "
          f"[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]"
          f"(https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/sct-range/blob/main/notebooks/{NAME})")


class Stage:
    def __init__(self, name: str, notebook: str, renames: dict[str, str], params: str, after: tuple[str, ...],
                 complete: str, note: str = '""') -> None:
        self.name, self.notebook, self.renames = name, notebook, renames
        self.params, self.after, self.complete, self.note = params, after, complete, note


SEEDS_RENAME = {r'"SCT_SEEDS", SEEDS\)': '"SCT_SEEDS", SEED_LIST)'}
STAGES = [
    Stage("S1", "s1_harness.ipynb", {r"\bMAX_CASES\b": "S1_MAX_CASES"},
          "{'tasks': TASKS, 'max_cases': S1_MAX_CASES}", (), "True"),
    Stage("S2", "s2_synthetic_benchmark.ipynb", {r"\bN_CASES\b": "S2_CASES"},
          "{'task': TASK, 'cases': S2_CASES}", ("S1",), "True"),
    Stage("S3", "s3_real_diagnostic.ipynb", {},
          "{'task': TASK, 'steps': STEPS, 'folds': FOLDS, 'train': TRAIN}", ("S1",),
          'decision["verdict"] != "NOT EVALUATED" and all(loss in spectra for loss in LOSSES)', 'decision["verdict"]'),
    Stage("S4", "s4_regression_arms.ipynb", SEEDS_RENAME,
          "{'task': TASK, 'steps': STEPS, 'folds': FOLDS, 'seeds': SEED_LIST, 'train': TRAIN}", ("S1",),
          'not missing and decision["verdict"] != "NOT EVALUATED"', 'decision["verdict"]'),
    Stage("S6", "s6_dosimetry.ipynb", {r"\bN_CASES\b": "DOSE_CASES", **SEEDS_RENAME},
          "{'task': TASK, 'cases': DOSE_CASES, 'seeds': SEED_LIST, 'workers': WORKERS}", ("S4",),
          'decision["verdict"] != "NOT EVALUATED" and all(f"{a}@{s}" in set(per_seed["source"]) for a in ARMS for s in SEEDS)',
          'decision["verdict"]'),
    Stage("S7", "s7_generative.ipynb", {},
          "{'task': TASK, 'steps': STEPS, 'folds': FOLDS, 'etas': ETAS, 'sweep': SWEEP_CASES, "
          "'step_curve': STEP_CURVE_CASES, 'main': MAIN_CASES, 'train': TRAIN}", ("S4",),
          "HAVE_S4 and not missing", 'f"eta {SELECTED_ETA:g}"'),
]

INTRO = f"""# S1–S7 · Full pipeline in one notebook

{BADGES}

The week-by-week notebooks, merged in order so that one *Run all* goes from the raw data to every verdict. Each stage
is the corresponding notebook unchanged, below its own title; they share the setup cell and the parameters cell.

| Stage | Notebook it reproduces | Needs |
|---|---|---|
| S1 | `s1_harness` · data integrity, frozen split, HLUT, projector, range diagnostics | CPU |
| S2 | `s2_synthetic_benchmark` · theory checks, identities on real patients | CPU |
| S3 | `s3_real_diagnostic` · L1-HU and L2-SPR models, beam spectra, go/no-go 1 | GPU |
| S4–S5 | `s4_regression_arms` · arms R1–R6 over three seeds, pre-registered verdict | GPU |
| S6 | `s6_dosimetry` · proton and photon plans, proxy validation, go/no-go 2 | CPU (hours), pyRadPlan |
| S7–S8 | `s7_generative` · stochastic-interpolant generator, projected CRPS, comparison with R3 and R6 | GPU |

Every stage runs on the real SynthRAD2025 data (Hugging Face mirror `developper669/synthrad2025`, tag `v2`) and writes
to its own `outputs/s<k>/` folder, exactly as the separate notebooks do, so the two can be mixed: a stage interrupted
here can be finished in its own notebook and vice versa. `s0_zenodo_to_huggingface` (one-time mirror) is not part of
the pipeline.

**Automatic resumption.** After a crash, an error or an expired session (Colab, Kaggle), reopen the notebook and
*Run all* again with the same parameters; nothing has to be cleaned up.

- Stages that finished are skipped: their cells run as `pass` and print where their report is
  (`outputs/pipeline_state.json`, kept on Drive on Colab and in `/kaggle/working` on Kaggle; on Kaggle, attach the
  previous version's output as an input). A stage runs again when its parameters change, when a stage it depends on
  finished again, when its report was deleted, or when it is listed in `FORCE_STAGES`.
- A stage that stopped early (an error, no GPU, upstream runs missing) runs again and continues where it stopped:
  fold checkpoints are reused, a fold cut mid-training resumes from its last save (every 1000 steps, exactly the
  same model as an uninterrupted run), per-case evaluations, dosimetry cases and generator samples are cached.
- The dataset is fetched again on a new Colab or Kaggle session (resumable, with retries on network errors); the
  volume cache is rebuilt from it.
"""


def source(cell: dict) -> str:
    return "".join(cell["source"])


def cell(kind: str, text: str) -> dict:
    new = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        new.update(execution_count=None, outputs=[])
    return new


def demote(text: str) -> str:
    """Stage sections one level below the stage title."""
    return re.sub(r"^(#{2,5}) ", lambda m: "#" + m.group(1) + " ", text, flags=re.M)


def build() -> dict:
    first = json.loads((NOTEBOOKS / STAGES[0].notebook).read_text(encoding="utf-8"))
    cells = [
        cell("markdown", INTRO),
        cell("markdown", source(first["cells"][1])),
        cell("code", source(first["cells"][2])),
        cell("markdown", """### Parameters

Set them here and run all cells (*Runtime → Run all* on Colab, where this cell is a form). Environment variables of
the same meaning, given in brackets, override them for unattended runs (`jupyter nbconvert --execute`). Pre-registered
verdicts assume the defaults (30 000 steps, 5 folds, seeds 0, 1, 2, 20 dosimetry cases, the S7 budget); smaller values
give a quick pass only. S1 freezes the local test split the first time it runs on both tasks.

| Platform | Dataset, cache | Outputs | GPU |
|---|---|---|---|
| local | `data/synthrad2025`, `outputs/` | `outputs/` | every visible GPU |
| Colab | `/content/data`, `/content/cache` | Google Drive `MyDrive/sct-range/outputs` (`USE_DRIVE`) | *Runtime → Change runtime type → T4 GPU* |
| Kaggle | `/kaggle/tmp/data`, `/kaggle/tmp/cache` | `/kaggle/working/outputs` | *Settings → Accelerator → GPU T4 × 2*, *Internet on* |
"""),
        cell("code", """# @title Parameters { display-mode: "form" }
# shared
TASK = "1"  # @param ["1", "2"]  S2-S7: 1 = MRI, 2 = CBCT [SCT_TASK]
DOWNLOAD = True  # @param {type:"boolean"}  fetch missing data from Hugging Face [SCT_DOWNLOAD]
USE_DRIVE = True  # @param {type:"boolean"}  Colab only
FORCE_STAGES = ""  # @param {type:"string"}  stages to run again even if complete, e.g. "S3,S4" [SCT_FORCE_STAGES]
# S1
TASKS = "1,2"  # @param ["1,2", "1", "2"]  tasks checked by S1 [SCT_TASKS]
S1_MAX_CASES = 0  # @param {type:"integer"}  S1 diagnostics, 0 = every development case [SCT_MAX_CASES]
# S2
S2_CASES = 20  # @param {type:"integer"}  S2 section 7 cases, 0 = all [SCT_MAX_CASES]
# S3, S4, S7 training
TRAIN = True  # @param {type:"boolean"}  train missing folds here (needs a GPU) [SCT_TRAIN_IN_NOTEBOOK]
STEPS = 30000  # @param {type:"integer"}  optimisation steps per fold [SCT_STEPS]
FOLDS = 5  # @param {type:"integer"}  cross-fitting folds [SCT_FOLDS]
SEED_LIST = "0,1,2"  # @param {type:"string"}  S4 seeds, also recomputed by S6 [SCT_SEEDS]
# S6
DOSE_CASES = 20  # @param {type:"integer"}  dosimetry cases [SCT_DOSE_CASES]
WORKERS = 1  # @param {type:"integer"}  cases dosed in parallel [SCT_WORKERS]
# S7
ETAS = "0,0.05,0.1,0.2"  # @param {type:"string"}  [SCT_ETAS]
SWEEP_CASES = 30  # @param {type:"integer"}  [SCT_SWEEP_CASES]
STEP_CURVE_CASES = 10  # @param {type:"integer"}  [SCT_STEP_CASES]
MAIN_CASES = 100  # @param {type:"integer"}  [SCT_MAIN_CASES]
"""),
        cell("markdown", """### Resumption

The state of every stage is read from the outputs; the table below shows what the previous runs left. Stages marked
`complete` with unchanged inputs are skipped below.
"""),
        cell("code", """from sctrange.pipeline import StageTracker
from sctrange.runtime import resolve_paths

PIPELINE_OUTPUTS = resolve_paths(REPO_ROOT, USE_DRIVE).output_root
PIPELINE = StageTracker(PIPELINE_OUTPUTS, os.environ.get("SCT_FORCE_STAGES", FORCE_STAGES).split(","))
PIPELINE.install()
PIPELINE.show()
"""),
    ]
    for stage in STAGES:
        stage_cells = json.loads((NOTEBOOKS / stage.notebook).read_text(encoding="utf-8"))["cells"]
        title = "\n".join(line for line in source(stage_cells[0]).splitlines() if "colab-badge.svg" not in line)
        title = re.sub(r"\n{3,}", "\n\n", title)
        cells.append(cell("markdown", title + f"\n\n*Stage notebook: `notebooks/{stage.notebook}`.*\n"))
        report = f'PIPELINE_OUTPUTS / "{stage.name.lower()}" / "{stage.name.lower()}_report.json"'
        cells.append(cell("code", f'PIPELINE.begin("{stage.name}", {stage.params}, {report}, after={stage.after!r})\n'))
        for original in stage_cells[5:]:  # 0 title, 1-2 setup, 3-4 parameters: shared above
            text = source(original)
            for pattern, replacement in stage.renames.items():
                text = re.sub(pattern, replacement, text)
            cells.append(cell(original["cell_type"], demote(text) if original["cell_type"] == "markdown" else text))
        cells.append(cell("code", f'PIPELINE.finish("{stage.name}", complete=bool({stage.complete}), '
                                  f'note=str({stage.note}))\n'))
    cells.append(cell("markdown", "## Pipeline state\n"))
    cells.append(cell("code", "PIPELINE.show()\n"))
    for index, item in enumerate(cells):
        item["id"] = f"pipeline-{index:03d}"
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    path = NOTEBOOKS / NAME
    path.write_text(render(), encoding="utf-8")
    print(f"{path}: {len(build()['cells'])} cells", file=sys.stderr)
