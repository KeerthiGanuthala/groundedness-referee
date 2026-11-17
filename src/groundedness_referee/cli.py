"""Command-line entry point for groundedness-referee."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from groundedness_referee.corpus import (
    DEFAULT_N_QUESTIONS,
    DEFAULT_SEED,
    write_fixture_files,
)
from groundedness_referee.experiments import (
    format_experiment_tables,
    run_experiment_1_twin_pipelines,
    write_all_experiments,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="groundedness-referee",
        description="Counterfactual failure attribution referee for RAG groundedness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fix = sub.add_parser("fixtures", help="Generate deterministic JSONL fixtures.")
    p_fix.add_argument("--output-dir", type=Path, default=Path("data"))
    p_fix.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_fix.add_argument("--n-questions", type=int, default=DEFAULT_N_QUESTIONS)

    p_twins = sub.add_parser(
        "twin-demo",
        help="Reproduce Experiment 1 twin pipelines (identical scores, opposite fixes).",
    )
    p_twins.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p_exp = sub.add_parser(
        "experiments",
        help="Run all experiments (1-5), write JSON files, and print tables.",
    )
    p_exp.add_argument(
        "--output-dir", type=Path, default=Path("docs/experiments")
    )
    p_exp.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args(argv)

    if args.command == "fixtures":
        meta = write_fixture_files(
            output_dir=args.output_dir,
            seed=args.seed,
            n_questions=args.n_questions,
        )
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0

    if args.command == "twin-demo":
        exp1 = run_experiment_1_twin_pipelines(seed=args.seed)
        pa = exp1["pipeline_a"]
        pb = exp1["pipeline_b"]
        print("=== TWIN PIPELINE SYMMETRY BREAK (SEED = 20261117) ===")
        print(
            f"Shared Groundedness Score (2 d.p.): {exp1['shared_groundedness_score_2dp']:.2f}"
        )
        for p in (pa, pb):
            sh = p["referee_shares"]
            iv = p["interventions"]
            print(
                f"\n[{p['pipeline_name']}]\n"
                f"  Groundedness Score : {p['groundedness_score']:.4f} "
                f"({p['grounded_count']}/{p['answered_questions']})\n"
                f"  Referee Blame      : retrieval-miss={sh['retrieval-miss']*100:.1f}%, "
                f"generation-ignore={sh['generation-ignore']*100:.1f}%, "
                f"fabrication-present={sh['fabrication-with-evidence-present']*100:.1f}%, "
                f"unattributable={sh['unattributable']*100:.1f}%\n"
                f"  Referee Verdict    : {p['recommended_fix']}\n"
                f"  Intervention Delta : fix_retriever={iv['fix_retriever_gain']:+.4f} vs "
                f"fix_generator={iv['fix_generator_gain']:+.4f}"
            )
        return 0

    if args.command == "experiments":
        results = write_all_experiments(output_dir=args.output_dir, seed=args.seed)
        print(format_experiment_tables(results))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
