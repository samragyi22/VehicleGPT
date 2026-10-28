"""
tests/test_entry_point.py - run_pipeline must be the ONLY pipeline entry point.

llm.router.route() was retired on freeze-prep: eval scripts migrated to
engine.pipeline.run_pipeline, and no module, eval script or test may call a
second pipeline implementation.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_router_has_no_route_function():
    import llm.router

    assert not hasattr(llm.router, "route"), "llm.router.route must not exist"
    assert not hasattr(llm.router, "route_batch"), "llm.router.route_batch must not exist"
    assert not hasattr(llm.router, "RouterResult"), "RouterResult must not exist"


def test_no_module_imports_route():
    """No repo module may import llm.router.route or llm.router.route_batch."""
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if ".venv" in rel.parts or "__pycache__" in rel.parts:
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module == "llm.router":
                names = [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names if a.name == "llm.router"]
            if any(n in ("route", "route_batch") for n in names):
                offenders.append(f"{rel}: line {node.lineno}")
    assert not offenders, f"route imports remain: {offenders}"


def test_no_source_uses_stray_route_calls():
    """No eval/ or engine/ source calls route( outside comments/docstrings."""
    offenders = []
    for path in (REPO_ROOT / "eval").glob("*.py"):
        src = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(src.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "route(" in line and "run_pipeline(" not in line:
                offenders.append(f"eval/{path.name}:{lineno}: {stripped}")
    assert not offenders, f"stray route() calls: {offenders}"
