"""Stage tracking for ``notebooks/s1_s7_pipeline.ipynb``: a rerun after a crash or an expired session skips the stages
that already finished and resumes inside the one that stopped.

The state lives with the outputs (``<outputs>/pipeline_state.json``: Google Drive on Colab, ``/kaggle/working`` on
Kaggle), so it survives the session. A stage counts as finished only when it reached its complete result (every run
trained, a verdict on real data); a stage that ended early (no GPU, missing upstream runs) runs again next time. A
finished stage is redone when its parameters change, when a stage it depends on finished again, when its report
file disappeared, or when it is listed in ``FORCE_STAGES``.

Skipping works at cell level through an IPython input transformer: after ``tracker.begin(...)`` decided to skip, the
cells of that stage execute as ``pass`` until the next ``begin``. Inside a stage, resumption comes from the stages
themselves: fold checkpoints and in-fold progress files (``sctrange.training``), cached per-case evaluations
(``sctrange.runs.cached_result``), per-case dosimetry and generative results.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from . import __version__

STATE_NAME = "pipeline_state.json"
_MARKERS = ("PIPELINE.begin(", "PIPELINE.show(")  # cells that always run


class StageTracker:
    def __init__(self, output_root: str | Path, force: Iterable[str] = ()) -> None:
        self.path = Path(output_root) / STATE_NAME
        self.force = {s.strip().upper() for s in force if s.strip()}
        self.skipping = False
        self.current: str | None = None
        self._fingerprint = ""
        self._report: Path | None = None

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, default=str))
        temporary.replace(self.path)

    def begin(self, stage: str, params: dict[str, Any], report: str | Path, after: Sequence[str] = ()) -> bool:
        """Start ``stage``; returns True (and skips its cells) when it already finished with the same inputs."""
        state = self._load()
        upstream = {name: state.get(name, {}).get("finish_id") for name in after}
        environment = {k: v for k, v in os.environ.items()
                       if k.startswith(("SCT_", "SYNTHRAD2025")) and k != "SCT_FORCE_STAGES"}
        payload = {"stage": stage, "params": params, "environment": environment, "upstream": upstream,
                   "sctrange": __version__}
        self._fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        self._report = Path(report)
        self.current = stage
        previous = state.get(stage, {})
        done = (previous.get("status") == "complete" and previous.get("fingerprint") == self._fingerprint
                and self._report.exists() and stage.upper() not in self.force)
        if done:
            print(f"{stage} already complete ({previous['finished_at']}, report {self._report}); its cells are skipped. "
                  f"Add {stage} to FORCE_STAGES to run it again.")
        else:
            reason = ("forced" if stage.upper() in self.force else "first run" if not previous
                      else "inputs changed" if previous.get("fingerprint") != self._fingerprint
                      else "report missing" if not self._report.exists() else f"previous run {previous.get('status')}")
            print(f"{stage}: running ({reason})")
            state[stage] = {**previous, "status": "running", "started_at": _now(), "fingerprint": self._fingerprint}
            self._save(state)
        self.skipping = done
        return done

    def finish(self, stage: str, complete: bool, note: str = "") -> None:
        """End ``stage``: ``complete`` marks it finished (skipped on the next run), otherwise it runs again."""
        state = self._load()
        entry = {"status": "complete" if complete else "partial", "fingerprint": self._fingerprint, "note": note,
                 "report": str(self._report) if self._report else None}
        if complete:
            entry.update(finished_at=_now(), finish_id=uuid.uuid4().hex)  # new id: dependent stages run again
        state[stage] = {**state.get(stage, {}), **entry}
        self._save(state)
        print(f"{stage}: {'complete' if complete else 'partial, will run again on the next Run all'}"
              f"{f' ({note})' if note else ''}")
        self.skipping = False

    def summary(self) -> list[dict]:
        return [{"stage": name, **{k: v for k, v in entry.items() if k not in ("fingerprint", "finish_id")}}
                for name, entry in self._load().items()]

    def show(self) -> None:
        """Print the state of every stage; ends any skipping (the cell calling it always runs)."""
        self.skipping = False
        for entry in self.summary() or [{"stage": "none", "status": "first run"}]:
            print(" · ".join(f"{key} {value}" for key, value in entry.items() if value))

    def transform(self, lines: list[str]) -> list[str]:
        """IPython input transformer: the cells of a skipped stage run as ``pass``."""
        if self.skipping and not any(marker in line for line in lines for marker in _MARKERS):
            return ["pass\n"]
        return lines

    def install(self) -> None:
        """Register ``transform`` with the running IPython kernel (once)."""
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return
        shell.input_transformers_post[:] = [t for t in shell.input_transformers_post
                                            if getattr(t, "__name__", "") != "transform"
                                            or not isinstance(getattr(t, "__self__", None), StageTracker)]
        shell.input_transformers_post.append(self.transform)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
