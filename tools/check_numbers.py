#!/usr/bin/env python3
"""Verify that every published figure in documentation matches docs/metrics.json (Section 5).

Matches explicit anchor phrases rather than bare digits, and includes a
`prove_breakage()` self-test that mutates one figure in memory to prove the
checker fails when documentation drifts from code output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _resolve_metric_path(metrics: dict[str, Any], dotted_key: str) -> str:
    cur: Any = metrics
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Metric path '{dotted_key}' not found in metrics.json")
        cur = cur[part]
    return str(cur)


ANCHOR_RULES: list[tuple[str, str, str]] = [
    (
        "README.md",
        r"seeded cohort of `(\d+)` oracle-labeled QA pairs",
        "fixture.n_questions",
    ),
    (
        "README.md",
        r"across `(\d+)` corpus passages",
        "fixture.n_passages",
    ),
    (
        "README.md",
        r"both pipelines report a groundedness score of `([0-9.]+)`",
        "exp1.shared_score_2dp",
    ),
    (
        "README.md",
        r"Pipeline A exact groundedness score is `([0-9.]+)`",
        "exp1.pipeline_a_score_4dp",
    ),
    (
        "README.md",
        r"Pipeline B exact groundedness score is `([0-9.]+)`",
        "exp1.pipeline_b_score_4dp",
    ),
    (
        "README.md",
        r"Pipeline A retrieval-miss share is `([0-9.]+%)`",
        "exp1.pipeline_a_ret_miss_pct",
    ),
    (
        "README.md",
        r"Pipeline B generation-ignore share is `([0-9.]+%)`",
        "exp1.pipeline_b_gen_ignore_pct",
    ),
    (
        "README.md",
        r"Pipeline B fabrication-with-evidence-present share is `([0-9.]+%)`",
        "exp1.pipeline_b_gen_fab_pct",
    ),
    (
        "README.md",
        r"Pipeline A retriever intervention delta is `([+0-9.]+)`",
        "exp1.pipeline_a_gain_fix_ret",
    ),
    (
        "README.md",
        r"Pipeline B generator intervention delta is `([+0-9.]+)`",
        "exp1.pipeline_b_gain_fix_gen",
    ),
    (
        "README.md",
        r"across `(\d+)` pooled unsupported answers in the five-mix grid",
        "exp2.pooled_unsupported_evaluated",
    ),
    (
        "README.md",
        r"baseline counterfactual probe attribution accuracy is `([0-9.]+%)`",
        "exp2.pooled_overall_accuracy_pct",
    ),
    (
        "README.md",
        r"at a 20% context-flip rate, probe defeat rate reaches `([0-9.]+%)`",
        "exp2.flip_20_defeat_pct",
    ),
    (
        "README.md",
        r"at 1\.0% hallucination prevalence, flag precision \(PPV\) is `([0-9.]+%)`",
        "exp3.prev_01_ppv_pct",
    ),
    (
        "README.md",
        r"95% rule-of-three zero-event sample floor at 1\.0% prevalence is `(\d+)`",
        "exp3.prev_01_min_n_95",
    ),
    (
        "README.md",
        r"mean referee-guided groundedness gain across the sweep is `([+0-9.]+)`",
        "exp4.mean_referee_gain",
    ),
    (
        "README.md",
        r"mean wrong-intervention regret is `([0-9.]+)`",
        "exp4.mean_wrong_fix_regret",
    ),
    (
        "README.md",
        r"learned judge and lexical proxy agree on `([0-9.]+%)` of pairs",
        "exp5.overall_agreement_pct",
    ),
    (
        "README.md",
        r"experiment-level intervention conclusions changed: `(\d+)`",
        "exp5.changed_conclusions",
    ),
    (
        "docs/adr/0003-lexical-entailment-proxy-vs-learned-judge.md",
        r"measured agreement rate between the lexical proxy and learned judge is `([0-9.]+%)`",
        "exp5.overall_agreement_pct",
    ),
    (
        "docs/adr/0005-deterministic-jsonl-fixtures-from-seed.md",
        r"canonical `corpus_passages\.jsonl` SHA-256 prefix is `([0-9a-f]+)`",
        "fixture.passages_sha256_prefix",
    ),
    (
        "docs/adr/0005-deterministic-jsonl-fixtures-from-seed.md",
        r"canonical `qa_oracle_fixture\.jsonl` SHA-256 prefix is `([0-9a-f]+)`",
        "fixture.qa_sha256_prefix",
    ),
]


def check_documents(
    root_dir: Path = ROOT,
    doc_overrides: dict[str, str] | None = None,
) -> list[str]:
    """Return a list of error strings (empty if all anchored figures match)."""
    metrics_path = root_dir / "docs" / "metrics.json"
    if not metrics_path.exists():
        return [f"Missing metrics file: {metrics_path}"]

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    for rel_path, pattern, metric_key in ANCHOR_RULES:
        expected = _resolve_metric_path(metrics, metric_key)
        if doc_overrides and rel_path in doc_overrides:
            text = doc_overrides[rel_path]
        else:
            full_path = root_dir / rel_path
            if not full_path.exists():
                errors.append(f"Missing document: {rel_path}")
                continue
            text = full_path.read_text(encoding="utf-8")

        match = re.search(pattern, text)
        if not match:
            errors.append(
                f"[{rel_path}] Anchor phrase not found for '{metric_key}': /{pattern}/"
            )
            continue

        actual = match.group(1)
        if actual != expected:
            errors.append(
                f"[{rel_path}] Figure mismatch for '{metric_key}': "
                f"document quotes '{actual}', but metrics.json has '{expected}'"
            )

    return errors


def prove_breakage(root_dir: Path = ROOT) -> bool:
    """Mutate one anchored figure in memory and verify check_documents fails."""
    readme_path = root_dir / "README.md"
    original = readme_path.read_text(encoding="utf-8")
    corrupted = original.replace(
        "both pipelines report a groundedness score of `0.59`",
        "both pipelines report a groundedness score of `0.41`",
        1,
    )
    errors = check_documents(
        root_dir=root_dir,
        doc_overrides={"README.md": corrupted},
    )
    return len(errors) == 1 and "exp1.shared_score_2dp" in errors[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check published numbers in docs against docs/metrics.json."
    )
    parser.add_argument(
        "--prove-breakage",
        action="store_true",
        help="Verify that mutating a quoted number causes a check failure.",
    )
    args = parser.parse_args(argv)

    errors = check_documents(ROOT)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    if args.prove_breakage:
        if not prove_breakage(ROOT):
            print(
                "ERROR: prove_breakage() did not catch mutated figure!",
                file=sys.stderr,
            )
            return 1
        print(
            "OK: All 22 anchored figures match docs/metrics.json, and breakage proof caught mutated figure."
        )
    else:
        print("OK: All 22 anchored figures match docs/metrics.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
