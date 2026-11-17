#!/usr/bin/env python3
"""Re-measure every published figure and write docs/metrics.json (Section 5)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from groundedness_referee.corpus import (  # noqa: E402
    DEFAULT_N_QUESTIONS,
    DEFAULT_SEED,
    generate_oracle_fixture,
    write_fixture_files,
)
from groundedness_referee.experiments import (  # noqa: E402
    format_experiment_tables,
    write_all_experiments,
)
from groundedness_referee.schema import (  # noqa: E402
    CASE_DISTRACTOR_TRAP,
    CASE_SINGLE_PASSAGE,
    CASE_SPLIT_EVIDENCE,
    CASE_UNANSWERABLE,
)


def collect_all_metrics(root_dir: Path = ROOT) -> dict[str, object]:
    data_dir = root_dir / "data"
    exp_dir = root_dir / "docs" / "experiments"
    metrics_path = root_dir / "docs" / "metrics.json"

    fixture_meta = write_fixture_files(
        output_dir=data_dir,
        seed=DEFAULT_SEED,
        n_questions=DEFAULT_N_QUESTIONS,
    )
    passages, qa_pairs = generate_oracle_fixture(
        seed=DEFAULT_SEED,
        n_questions=DEFAULT_N_QUESTIONS,
    )
    case_counts = {
        CASE_SINGLE_PASSAGE: sum(
            1 for q in qa_pairs if q.case_type == CASE_SINGLE_PASSAGE
        ),
        CASE_SPLIT_EVIDENCE: sum(
            1 for q in qa_pairs if q.case_type == CASE_SPLIT_EVIDENCE
        ),
        CASE_DISTRACTOR_TRAP: sum(
            1 for q in qa_pairs if q.case_type == CASE_DISTRACTOR_TRAP
        ),
        CASE_UNANSWERABLE: sum(
            1 for q in qa_pairs if q.case_type == CASE_UNANSWERABLE
        ),
    }

    exps = write_all_experiments(
        output_dir=exp_dir,
        seed=DEFAULT_SEED,
        n_questions=DEFAULT_N_QUESTIONS,
    )

    exp1 = exps["exp1"]
    pa = exp1["pipeline_a"]
    pb = exp1["pipeline_b"]

    exp2 = exps["exp2"]
    flip_map = {
        f"flip_{int(row['context_flip_rate']*100):02d}": row
        for row in exp2["context_flip_sweep"]
    }

    exp3 = exps["exp3"]
    prev_map = {
        f"prev_{int(row['hallucination_prevalence']*100):02d}": row
        for row in exp3["prevalence_rows"]
    }

    exp4 = exps["exp4"]
    exp5 = exps["exp5"]
    phen_map = {row["phenomenon"]: row for row in exp5["phenomenon_breakdown"]}

    metrics_payload: dict[str, object] = {
        "seed": DEFAULT_SEED,
        "fixture": {
            "n_questions": len(qa_pairs),
            "n_passages": len(passages),
            "single_passage_count": case_counts[CASE_SINGLE_PASSAGE],
            "split_evidence_count": case_counts[CASE_SPLIT_EVIDENCE],
            "distractor_trap_count": case_counts[CASE_DISTRACTOR_TRAP],
            "unanswerable_count": case_counts[CASE_UNANSWERABLE],
            "passages_sha256_prefix": fixture_meta["passages_sha256"][:12],
            "qa_sha256_prefix": fixture_meta["qa_sha256"][:12],
        },
        "exp1": {
            "shared_score_2dp": f"{exp1['shared_groundedness_score_2dp']:.2f}",
            "pipeline_a_score_4dp": f"{pa['groundedness_score']:.4f}",
            "pipeline_b_score_4dp": f"{pb['groundedness_score']:.4f}",
            "pipeline_a_grounded_count": pa["grounded_count"],
            "pipeline_a_answered_count": pa["answered_questions"],
            "pipeline_b_grounded_count": pb["grounded_count"],
            "pipeline_b_answered_count": pb["answered_questions"],
            "pipeline_a_ret_miss_pct": f"{pa['referee_shares']['retrieval-miss']*100:.1f}%",
            "pipeline_a_gen_ignore_pct": f"{pa['referee_shares']['generation-ignore']*100:.1f}%",
            "pipeline_a_gen_fab_pct": f"{pa['referee_shares']['fabrication-with-evidence-present']*100:.1f}%",
            "pipeline_b_ret_miss_pct": f"{pb['referee_shares']['retrieval-miss']*100:.1f}%",
            "pipeline_b_gen_ignore_pct": f"{pb['referee_shares']['generation-ignore']*100:.1f}%",
            "pipeline_b_gen_fab_pct": f"{pb['referee_shares']['fabrication-with-evidence-present']*100:.1f}%",
            "pipeline_b_unattributable_pct": f"{pb['referee_shares']['unattributable']*100:.1f}%",
            "pipeline_a_gain_fix_ret": f"+{pa['interventions']['fix_retriever_gain']:.4f}",
            "pipeline_a_gain_fix_gen": f"+{pa['interventions']['fix_generator_gain']:.4f}",
            "pipeline_b_gain_fix_ret": f"+{pb['interventions']['fix_retriever_gain']:.4f}",
            "pipeline_b_gain_fix_gen": f"+{pb['interventions']['fix_generator_gain']:.4f}",
        },
        "exp2": {
            "pooled_unsupported_evaluated": exp2["pooled_unsupported_evaluated"],
            "pooled_overall_accuracy_pct": f"{exp2['pooled_overall_accuracy']*100:.1f}%",
            "flip_00_accuracy_pct": f"{flip_map['flip_00']['attribution_accuracy']*100:.1f}%",
            "flip_10_accuracy_pct": f"{flip_map['flip_10']['attribution_accuracy']*100:.1f}%",
            "flip_10_defeat_pct": f"{flip_map['flip_10']['probe_defeat_rate']*100:.1f}%",
            "flip_20_accuracy_pct": f"{flip_map['flip_20']['attribution_accuracy']*100:.1f}%",
            "flip_20_defeat_pct": f"{flip_map['flip_20']['probe_defeat_rate']*100:.1f}%",
            "flip_35_accuracy_pct": f"{flip_map['flip_35']['attribution_accuracy']*100:.1f}%",
            "flip_35_defeat_pct": f"{flip_map['flip_35']['probe_defeat_rate']*100:.1f}%",
        },
        "exp3": {
            "scorer_tpr_pct": f"{exp3['measured_scorer_tpr']*100:.1f}%",
            "scorer_fpr_pct": f"{exp3['measured_scorer_fpr']*100:.1f}%",
            "prev_01_ppv_pct": f"{prev_map['prev_01']['flag_precision_ppv']*100:.1f}%",
            "prev_01_min_n_95": prev_map["prev_01"]["min_n_95pct_zero_bound"],
            "prev_05_ppv_pct": f"{prev_map['prev_05']['flag_precision_ppv']*100:.1f}%",
            "prev_05_min_n_95": prev_map["prev_05"]["min_n_95pct_zero_bound"],
            "prev_10_ppv_pct": f"{prev_map['prev_10']['flag_precision_ppv']*100:.1f}%",
            "prev_20_ppv_pct": f"{prev_map['prev_20']['flag_precision_ppv']*100:.1f}%",
        },
        "exp4": {
            "mean_referee_gain": f"+{exp4['mean_referee_gain']:.4f}",
            "mean_wrong_fix_gain": f"+{exp4['mean_wrong_fix_gain']:.4f}",
            "mean_wrong_fix_regret": f"{exp4['mean_wrong_fix_regret']:.4f}",
            "mean_score_only_regret": f"{exp4['mean_score_only_regret']:.4f}",
            "p1_regret": f"{exp4['sweep_rows'][0]['wrong_fix_regret']:.4f}",
            "p7_regret": f"{exp4['sweep_rows'][-1]['wrong_fix_regret']:.4f}",
        },
        "exp5": {
            "total_eval_pairs": exp5["total_eval_pairs"],
            "overall_agreement_pct": f"{exp5['overall_agreement_rate']*100:.1f}%",
            "lexical_overall_acc_pct": f"{exp5['lexical_overall_accuracy']*100:.1f}%",
            "learned_overall_acc_pct": f"{exp5['learned_overall_accuracy']*100:.1f}%",
            "negation_lexical_acc_pct": f"{phen_map['negation_polarity']['lexical_accuracy']*100:.1f}%",
            "negation_learned_acc_pct": f"{phen_map['negation_polarity']['learned_judge_accuracy']*100:.1f}%",
            "paraphrase_lexical_acc_pct": f"{phen_map['paraphrase_entailed']['lexical_accuracy']*100:.1f}%",
            "paraphrase_learned_acc_pct": f"{phen_map['paraphrase_entailed']['learned_judge_accuracy']*100:.1f}%",
            "changed_conclusions": exp5["changed_experiment_conclusions"],
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(format_experiment_tables(exps))
    print(f"\nWrote {metrics_path}")
    return metrics_payload


if __name__ == "__main__":
    collect_all_metrics()
