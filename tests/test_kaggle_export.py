import ast, json
from pathlib import Path


def test_kaggle_export_size_is_property():
    text = Path("kaggle/alpha_sweep_dual_t4_runner.py").read_text(encoding="utf8")
    ast.parse(text)
    assert "archive.stat().st_size()" not in text
    assert "archive.stat().st_size" in text


def test_repaired_notebook_has_no_stale_outputs_and_compiles():
    notebook = json.loads(
        Path("kaggle/sp_full_alpha_sweep_fixed.ipynb").read_text(encoding="utf8")
    )
    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        assert ".stat().st_size()" not in text
        if cell["cell_type"] == "code":
            ast.parse(text)
            assert cell["outputs"] == [] and cell["execution_count"] is None
