"""The notebooks are valid, their code parses, and the merged pipeline notebook is the one the builder produces."""

import ast
import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"))


def _code(path: Path) -> list[str]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


def _builder():
    spec = importlib.util.spec_from_file_location("build_pipeline_notebook", REPO / "tools" / "build_pipeline_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_parses(path: Path) -> None:
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(path, as_version=4))
    for code in _code(path):
        ast.parse(code)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_stored_without_outputs(path: Path) -> None:
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None


def test_pipeline_notebook_is_up_to_date() -> None:
    committed = (REPO / "notebooks" / "s1_s7_pipeline.ipynb").read_text(encoding="utf-8")
    assert committed == _builder().render(), "run `python tools/build_pipeline_notebook.py`"


def test_pipeline_reads_no_name_before_defining_it() -> None:
    defined = set(dir(builtins)) | {"display", "get_ipython"}
    for index, code in enumerate(_code(REPO / "notebooks" / "s1_s7_pipeline.ipynb")):
        tree = ast.parse(code)
        local = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                local |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                local.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local.add(node.id)
            elif isinstance(node, ast.arg):
                local.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local.add(node.name)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assert not used - defined - local, f"code cell {index}: {sorted(used - defined - local)}"
        defined |= local


def test_skipped_stage_cells_do_not_run(tmp_path: Path) -> None:
    pytest.importorskip("IPython")
    from IPython.core.interactiveshell import InteractiveShell

    from sctrange.pipeline import StageTracker

    shell = InteractiveShell.instance()
    try:
        tracker = StageTracker(tmp_path)
        report = tmp_path / "s1_report.json"
        report.write_text("{}")
        tracker.begin("S1", {}, report)
        tracker.finish("S1", complete=True)
        shell.user_ns.update(PIPELINE=StageTracker(tmp_path), report=report)
        shell.user_ns["PIPELINE"].install()
        shell.run_cell('PIPELINE.begin("S1", {}, report)')
        shell.run_cell("ran = True")
        assert "ran" not in shell.user_ns  # skipped: executed as pass
        shell.run_cell("PIPELINE.show()")
        shell.run_cell("ran = True")
        assert shell.user_ns["ran"] is True
    finally:
        shell.input_transformers_post.clear()
        InteractiveShell.clear_instance()
        sys.displayhook = sys.__displayhook__
