#!/usr/bin/env python3
"""Run property-sentence test suite and enforce >=90% line coverage on src/groundedness_referee."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src" / "groundedness_referee"
TESTS_DIR = ROOT / "tests"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _executable_lines(py_file: Path) -> set[int]:
    """Identify executable statement line numbers in a Python source file using AST."""
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    lines: set[int] = set()

    skip_types = (
        ast.Module,
        ast.Import,
        ast.ImportFrom,
        ast.Pass,
    )

    for node in ast.walk(tree):
        if isinstance(node, skip_types):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.stmt) and hasattr(node, "lineno"):
            lines.add(node.lineno)
    return lines


def run_suite_with_coverage(min_coverage: float = 0.90) -> int:
    target_files = {
        str(p.resolve()): p.resolve()
        for p in sorted(SRC_DIR.glob("*.py"))
    }
    executed: dict[str, set[int]] = {k: set() for k in target_files}

    def tracer(frame, event, arg):
        if event == "line":
            filename = frame.f_code.co_filename
            if filename in executed:
                executed[filename].add(frame.f_lineno)
        return tracer

    sys.settrace(tracer)
    try:
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=str(TESTS_DIR), pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    finally:
        sys.settrace(None)

    if not result.wasSuccessful():
        return 1

    total_exec = 0
    total_hit = 0
    per_file_stats: dict[str, dict[str, object]] = {}

    print("\n=== LINE COVERAGE REPORT (src/groundedness_referee) ===")
    print(f"{'Module':<34} | {'Statements':>10} | {'Covered':>8} | {'Coverage':>8}")
    print("-" * 70)

    for abs_str, path_obj in target_files.items():
        stmts = _executable_lines(path_obj)
        hits = stmts.intersection(executed[abs_str])
        n_stmts = len(stmts)
        n_hits = len(hits)
        cov = (n_hits / n_stmts) if n_stmts > 0 else 1.0
        total_exec += n_stmts
        total_hit += n_hits
        rel_name = path_obj.name
        per_file_stats[rel_name] = {
            "statements": n_stmts,
            "covered": n_hits,
            "coverage": round(cov, 4),
        }
        print(
            f"{rel_name:<34} | {n_stmts:>10d} | {n_hits:>8d} | {cov*100:>7.1f}%"
        )

    overall_cov = total_hit / total_exec if total_exec > 0 else 1.0
    print("-" * 70)
    print(
        f"{'TOTAL':<34} | {total_exec:>10d} | {total_hit:>8d} | {overall_cov*100:>7.1f}%"
    )

    cov_path = ROOT / "docs" / "coverage.json"
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    cov_path.write_text(
        json.dumps(
            {
                "overall_coverage": round(overall_cov, 4),
                "total_statements": total_exec,
                "total_covered": total_hit,
                "minimum_required": min_coverage,
                "files": per_file_stats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if overall_cov < min_coverage:
        print(
            f"ERROR: Overall coverage {overall_cov*100:.1f}% is below threshold {min_coverage*100:.1f}%",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: Coverage {overall_cov*100:.1f}% >= required {min_coverage*100:.1f}%."
    )
    return 0


if __name__ == "__main__":
    sys.exit(run_suite_with_coverage())
