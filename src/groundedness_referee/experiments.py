"""Experiments 1 through 5 for groundedness-referee (M5 + Stretch)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from groundedness_referee.components import (
    GeneratorKnobs,
    RetrieverKnobs,
    ScriptedGenerator,
    ScriptedRetriever,
)
from groundedness_referee.corpus import (
    DEFAULT_N_QUESTIONS,
    DEFAULT_SEED,
    generate_oracle_fixture,
)
from groundedness_referee.judge import (
    build_entailment_dataset,
    train_default_learned_judge,
)
from groundedness_referee.referee import GroundednessReferee
from groundedness_referee.schema import (
    CAUSE_FABRICATION_PRESENT,
    CAUSE_GENERATION_IGNORE,
    CAUSE_RETRIEVAL_MISS,
    CAUSE_UNATTRIBUTABLE,
    FAILURE_CAUSES,
    Passage,
    QAPair,
)
from groundedness_referee.scorer import LexicalGroundednessScorer


def _fixture_bundle(
    seed: int = DEFAULT_SEED,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> tuple[dict[str, Passage], list[QAPair]]:
    passages_list, qa_pairs = generate_oracle_fixture(
        seed=seed, n_questions=n_questions
    )
    return {p.passage_id: p for p in passages_list}, qa_pairs


def run_experiment_1_twin_pipelines(
    seed: int = DEFAULT_SEED,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> dict[str, Any]:
    """Experiment 1: Same score, opposite fixes.

    Constructs two pipelines whose groundedness scores match to the second decimal
    place on the seeded fixture:
      - Pipeline A (Retriever-Bottlenecked): weak retriever, faithful generator.
      - Pipeline B (Generator-Bottlenecked): strong retriever, unfaithful generator.
    """
    passages, qa_pairs = _fixture_bundle(seed=seed, n_questions=n_questions)
    referee = GroundednessReferee()

    ret_a = ScriptedRetriever(
        RetrieverKnobs(recall=0.60, distractor_rate=0.45, ranking_noise=0.20),
        seed=seed,
    )
    gen_a = ScriptedGenerator(
        GeneratorKnobs(
            faithful_rate=0.95,
            ignore_rate=0.03,
            fabricate_rate=0.02,
            abstain_rate=0.00,
        ),
        seed=seed,
    )
    report_a = referee.evaluate_pipeline(
        "pipeline_a_retriever_bottleneck", qa_pairs, passages, ret_a, gen_a
    )

    target_score = report_a.groundedness_score
    best_b = None
    best_knobs_b = None
    best_diff = 10**9

    for faithful_bp in range(620, 650):
        f_rate = faithful_bp / 1000.0
        rem = round(1.0 - f_rate, 6)
        ig_rate = round(rem * 0.58, 6)
        fab_rate = round(rem - ig_rate, 6)

        ret_b = ScriptedRetriever(
            RetrieverKnobs(recall=0.95, distractor_rate=0.15, ranking_noise=0.05),
            seed=seed + 11,
        )
        knobs_b = GeneratorKnobs(
            faithful_rate=f_rate,
            ignore_rate=ig_rate,
            fabricate_rate=fab_rate,
            abstain_rate=0.00,
        )
        gen_b = ScriptedGenerator(knobs_b, seed=seed + 11)
        cand = referee.evaluate_pipeline(
            "pipeline_b_generator_bottleneck", qa_pairs, passages, ret_b, gen_b
        )
        diff = abs(cand.groundedness_score - target_score)
        if diff < best_diff:
            best_diff = diff
            best_b = cand
            best_knobs_b = knobs_b

    assert best_b is not None and best_knobs_b is not None
    assert f"{report_a.groundedness_score:.2f}" == f"{best_b.groundedness_score:.2f}"

    ret_a_fixed = ScriptedRetriever(
        RetrieverKnobs(recall=0.85, distractor_rate=0.45, ranking_noise=0.20),
        seed=seed,
    )
    gen_a_fixed = ScriptedGenerator(
        GeneratorKnobs(
            faithful_rate=0.99,
            ignore_rate=0.005,
            fabricate_rate=0.005,
            abstain_rate=0.00,
        ),
        seed=seed,
    )
    a_after_ret_fix = referee.evaluate_pipeline(
        "a_after_retriever_fix", qa_pairs, passages, ret_a_fixed, gen_a
    )
    a_after_gen_fix = referee.evaluate_pipeline(
        "a_after_generator_fix", qa_pairs, passages, ret_a, gen_a_fixed
    )

    ret_b_orig = ScriptedRetriever(
        RetrieverKnobs(recall=0.95, distractor_rate=0.15, ranking_noise=0.05),
        seed=seed + 11,
    )
    ret_b_fixed = ScriptedRetriever(
        RetrieverKnobs(recall=0.99, distractor_rate=0.15, ranking_noise=0.05),
        seed=seed + 11,
    )
    b_new_f = min(0.99, round(best_knobs_b.faithful_rate + 0.25, 4))
    b_rem = round(1.0 - b_new_f, 4)
    gen_b_fixed = ScriptedGenerator(
        GeneratorKnobs(
            faithful_rate=b_new_f,
            ignore_rate=round(b_rem * 0.58, 6),
            fabricate_rate=round(b_rem - round(b_rem * 0.58, 6), 6),
            abstain_rate=0.00,
        ),
        seed=seed + 11,
    )
    gen_b_orig = ScriptedGenerator(best_knobs_b, seed=seed + 11)
    b_after_ret_fix = referee.evaluate_pipeline(
        "b_after_retriever_fix", qa_pairs, passages, ret_b_fixed, gen_b_orig
    )
    b_after_gen_fix = referee.evaluate_pipeline(
        "b_after_generator_fix", qa_pairs, passages, ret_b_orig, gen_b_fixed
    )

    return {
        "experiment": "exp1_twin_pipelines",
        "seed": seed,
        "n_questions": n_questions,
        "score_match_2dp": (
            f"{report_a.groundedness_score:.2f}"
            == f"{best_b.groundedness_score:.2f}"
        ),
        "shared_groundedness_score_2dp": round(report_a.groundedness_score, 2),
        "pipeline_a": {
            **report_a.to_summary_dict(),
            "knobs": {
                "retriever_recall": 0.60,
                "generator_faithful_rate": 0.95,
            },
            "interventions": {
                "fix_retriever_groundedness": round(
                    a_after_ret_fix.groundedness_score, 4
                ),
                "fix_retriever_gain": round(
                    a_after_ret_fix.groundedness_score - report_a.groundedness_score,
                    4,
                ),
                "fix_generator_groundedness": round(
                    a_after_gen_fix.groundedness_score, 4
                ),
                "fix_generator_gain": round(
                    a_after_gen_fix.groundedness_score - report_a.groundedness_score,
                    4,
                ),
            },
        },
        "pipeline_b": {
            **best_b.to_summary_dict(),
            "knobs": {
                "retriever_recall": 0.95,
                "generator_faithful_rate": round(best_knobs_b.faithful_rate, 4),
            },
            "interventions": {
                "fix_retriever_groundedness": round(
                    b_after_ret_fix.groundedness_score, 4
                ),
                "fix_retriever_gain": round(
                    b_after_ret_fix.groundedness_score - best_b.groundedness_score,
                    4,
                ),
                "fix_generator_groundedness": round(
                    b_after_gen_fix.groundedness_score, 4
                ),
                "fix_generator_gain": round(
                    b_after_gen_fix.groundedness_score - best_b.groundedness_score,
                    4,
                ),
            },
        },
    }


def run_experiment_2_probe_validation(
    seed: int = DEFAULT_SEED,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> dict[str, Any]:
    """Experiment 2: Probe validation across failure mixes and context-flip rates."""
    passages, qa_pairs = _fixture_bundle(seed=seed, n_questions=n_questions)
    referee = GroundednessReferee()

    mixes = [
        (
            "mix_retrieval_heavy",
            RetrieverKnobs(recall=0.55, distractor_rate=0.40, ranking_noise=0.15),
            (0.92, 0.05, 0.03, 0.00),
        ),
        (
            "mix_balanced",
            RetrieverKnobs(recall=0.75, distractor_rate=0.35, ranking_noise=0.15),
            (0.75, 0.14, 0.11, 0.00),
        ),
        (
            "mix_ignore_heavy",
            RetrieverKnobs(recall=0.88, distractor_rate=0.25, ranking_noise=0.10),
            (0.62, 0.28, 0.10, 0.00),
        ),
        (
            "mix_fabrication_heavy",
            RetrieverKnobs(recall=0.88, distractor_rate=0.25, ranking_noise=0.10),
            (0.62, 0.10, 0.28, 0.00),
        ),
        (
            "mix_high_stress",
            RetrieverKnobs(recall=0.65, distractor_rate=0.50, ranking_noise=0.25),
            (0.68, 0.16, 0.14, 0.02),
        ),
    ]

    grid_rows: list[dict[str, Any]] = []
    pooled_confusion: dict[str, dict[str, int]] = {
        t: {p: 0 for p in FAILURE_CAUSES} for t in FAILURE_CAUSES
    }
    pooled_oracle_totals = {c: 0 for c in FAILURE_CAUSES}
    pooled_correct = 0
    pooled_unsupported = 0

    for idx, (mix_name, ret_knobs, (f_r, ig_r, fab_r, abs_r)) in enumerate(mixes):
        ret = ScriptedRetriever(ret_knobs, seed=seed + idx * 17)
        gen = ScriptedGenerator(
            GeneratorKnobs(
                faithful_rate=f_r,
                ignore_rate=ig_r,
                fabricate_rate=fab_r,
                abstain_rate=abs_r,
                context_flip_rate=0.0,
            ),
            seed=seed + idx * 17,
        )
        rep = referee.evaluate_pipeline(mix_name, qa_pairs, passages, ret, gen)
        grid_rows.append(
            {
                "mix_name": mix_name,
                "retriever_recall": ret_knobs.recall,
                "generator_faithful_rate": f_r,
                "unsupported_count": rep.unsupported_count,
                "attribution_accuracy": round(rep.attribution_accuracy, 4),
                "per_cause_accuracy": {
                    k: round(v, 4) for k, v in rep.per_cause_accuracy.items()
                },
                "recommended_fix": rep.recommended_fix,
            }
        )
        pooled_unsupported += rep.unsupported_count
        for t in FAILURE_CAUSES:
            pooled_oracle_totals[t] += rep.oracle_counts[t]
            for p in FAILURE_CAUSES:
                cnt = rep.confusion_matrix[t][p]
                pooled_confusion[t][p] += cnt
                if t == p:
                    pooled_correct += cnt

    pooled_per_cause_accuracy = {
        c: round(
            pooled_confusion[c][c] / pooled_oracle_totals[c]
            if pooled_oracle_totals[c] > 0
            else 1.0,
            4,
        )
        for c in FAILURE_CAUSES
    }
    pooled_overall_accuracy = round(
        pooled_correct / pooled_unsupported if pooled_unsupported > 0 else 1.0, 4
    )

    flip_rates = [0.00, 0.05, 0.10, 0.20, 0.35]
    flip_sweep: list[dict[str, Any]] = []
    for flip_rate in flip_rates:
        ret = ScriptedRetriever(
            RetrieverKnobs(recall=0.75, distractor_rate=0.35, ranking_noise=0.15),
            seed=seed + 99,
        )
        gen = ScriptedGenerator(
            GeneratorKnobs(
                faithful_rate=0.72,
                ignore_rate=0.16,
                fabricate_rate=0.12,
                abstain_rate=0.00,
                context_flip_rate=flip_rate,
            ),
            seed=seed + 99,
        )
        rep = referee.evaluate_pipeline(
            f"flip_{int(flip_rate * 100):02d}", qa_pairs, passages, ret, gen
        )
        gen_fault_total = (
            rep.oracle_counts[CAUSE_GENERATION_IGNORE]
            + rep.oracle_counts[CAUSE_FABRICATION_PRESENT]
        )
        spurious_ret_blame = (
            rep.confusion_matrix[CAUSE_GENERATION_IGNORE][CAUSE_RETRIEVAL_MISS]
            + rep.confusion_matrix[CAUSE_FABRICATION_PRESENT][CAUSE_RETRIEVAL_MISS]
        )
        probe_defeat_rate = round(1.0 - rep.attribution_accuracy, 4)
        flip_sweep.append(
            {
                "context_flip_rate": flip_rate,
                "unsupported_count": rep.unsupported_count,
                "attribution_accuracy": round(rep.attribution_accuracy, 4),
                "probe_defeat_rate": probe_defeat_rate,
                "spurious_retrieval_blame_rate": round(
                    spurious_ret_blame / gen_fault_total
                    if gen_fault_total > 0
                    else 0.0,
                    4,
                ),
            }
        )

    return {
        "experiment": "exp2_probe_validation",
        "seed": seed,
        "n_questions": n_questions,
        "pooled_unsupported_evaluated": pooled_unsupported,
        "pooled_overall_accuracy": pooled_overall_accuracy,
        "pooled_per_cause_accuracy": pooled_per_cause_accuracy,
        "pooled_confusion_matrix": pooled_confusion,
        "grid_rows": grid_rows,
        "context_flip_sweep": flip_sweep,
    }


def run_experiment_3_precision_at_prevalence(
    seed: int = DEFAULT_SEED,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> dict[str, Any]:
    """Experiment 3: Flag precision at operating prevalence and sample-size floors."""
    passages, qa_pairs = _fixture_bundle(seed=seed, n_questions=n_questions)
    scorer = LexicalGroundednessScorer()

    true_unsupported_flagged = 0
    true_unsupported_total = 0
    true_supported_flagged = 0
    true_supported_total = 0

    for idx, qa in enumerate(qa_pairs):
        if not qa.gold_passage_ids:
            continue
        gold_ps = [passages[pid] for pid in qa.gold_passage_ids]
        use_paraphrase = (idx % 16) == 0
        sup_ans = qa.paraphrased_answer if use_paraphrase else qa.expected_answer
        s_eval = scorer.score(sup_ans, gold_ps)
        true_supported_total += 1
        if not s_eval.is_grounded:
            true_supported_flagged += 1

        for unsup_ans in (qa.prior_myth_answer, qa.fabricated_answer):
            u_eval = scorer.score(unsup_ans, gold_ps)
            true_unsupported_total += 1
            if not u_eval.is_grounded:
                true_unsupported_flagged += 1

    tpr = true_unsupported_flagged / true_unsupported_total
    fpr = true_supported_flagged / true_supported_total

    prevalences = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35]
    prevalence_table: list[dict[str, Any]] = []
    for prev in prevalences:
        denom = tpr * prev + fpr * (1.0 - prev)
        ppv = (tpr * prev) / denom if denom > 0 else 0.0

        single_event_n = int(math.ceil(1.0 / prev))
        rule_of_three_95_n = int(math.ceil(3.0 / prev))
        blame_share_ci_n = int(math.ceil(96.0 / prev))

        prevalence_table.append(
            {
                "hallucination_prevalence": prev,
                "flag_precision_ppv": round(ppv, 4),
                "false_alarm_share": round(1.0 - ppv, 4),
                "min_n_single_event": single_event_n,
                "min_n_95pct_zero_bound": rule_of_three_95_n,
                "min_n_for_96_flagged_events": blame_share_ci_n,
            }
        )

    return {
        "experiment": "exp3_precision_at_prevalence",
        "seed": seed,
        "measured_scorer_tpr": round(tpr, 4),
        "measured_scorer_fpr": round(fpr, 4),
        "supported_evaluated": true_supported_total,
        "unsupported_evaluated": true_unsupported_total,
        "prevalence_rows": prevalence_table,
    }


def run_experiment_4_wrong_fix_cost(
    seed: int = DEFAULT_SEED,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> dict[str, Any]:
    """Experiment 4: The cost of fixing the wrong thing across a pipeline sweep."""
    passages, qa_pairs = _fixture_bundle(seed=seed, n_questions=n_questions)
    referee = GroundednessReferee()

    sweep_configs = [
        ("p1_severe_retrieval_gap", 0.50, 0.94),
        ("p2_moderate_retrieval_gap", 0.62, 0.90),
        ("p3_mild_retrieval_tilt", 0.72, 0.84),
        ("p4_near_parity_retrieval", 0.76, 0.80),
        ("p5_mild_generator_tilt", 0.86, 0.70),
        ("p6_moderate_generator_gap", 0.92, 0.60),
        ("p7_severe_generator_gap", 0.96, 0.50),
    ]

    delta_knob = 0.20
    rows: list[dict[str, Any]] = []

    for idx, (name, base_recall, base_faithful) in enumerate(sweep_configs):
        run_seed = seed + idx * 31
        rem = round(1.0 - base_faithful, 6)
        ig = round(rem * 0.55, 6)
        fab = round(rem - ig, 6)

        ret_base = ScriptedRetriever(
            RetrieverKnobs(recall=base_recall, distractor_rate=0.35, ranking_noise=0.15),
            seed=run_seed,
        )
        gen_base = ScriptedGenerator(
            GeneratorKnobs(
                faithful_rate=base_faithful,
                ignore_rate=ig,
                fabricate_rate=fab,
                abstain_rate=0.0,
            ),
            seed=run_seed,
        )
        base_rep = referee.evaluate_pipeline(
            name, qa_pairs, passages, ret_base, gen_base
        )

        boosted_recall = min(0.99, round(base_recall + delta_knob, 4))
        ret_boost = ScriptedRetriever(
            RetrieverKnobs(
                recall=boosted_recall, distractor_rate=0.35, ranking_noise=0.15
            ),
            seed=run_seed,
        )
        rep_r = referee.evaluate_pipeline(
            f"{name}_fix_ret", qa_pairs, passages, ret_boost, gen_base
        )

        boosted_faithful = min(0.99, round(base_faithful + delta_knob, 4))
        rem_g = round(1.0 - boosted_faithful, 6)
        ig_g = round(rem_g * 0.55, 6)
        fab_g = round(rem_g - ig_g, 6)
        gen_boost = ScriptedGenerator(
            GeneratorKnobs(
                faithful_rate=boosted_faithful,
                ignore_rate=ig_g,
                fabricate_rate=fab_g,
                abstain_rate=0.0,
            ),
            seed=run_seed,
        )
        rep_g = referee.evaluate_pipeline(
            f"{name}_fix_gen", qa_pairs, passages, ret_base, gen_boost
        )

        gain_r = round(rep_r.groundedness_score - base_rep.groundedness_score, 4)
        gain_g = round(rep_g.groundedness_score - base_rep.groundedness_score, 4)

        ref_choice = base_rep.recommended_fix
        ref_gain = gain_r if ref_choice == "fix_retriever" else gain_g
        wrong_gain = gain_g if ref_choice == "fix_retriever" else gain_r
        score_only_expected_gain = round(0.5 * (gain_r + gain_g), 4)
        wrong_fix_regret = round(max(gain_r, gain_g) - wrong_gain, 4)
        score_only_regret = round(ref_gain - score_only_expected_gain, 4)

        rows.append(
            {
                "pipeline_name": name,
                "base_recall": base_recall,
                "base_faithful": base_faithful,
                "base_groundedness": round(base_rep.groundedness_score, 4),
                "referee_recommended_fix": ref_choice,
                "gain_fix_retriever": gain_r,
                "gain_fix_generator": gain_g,
                "referee_chosen_gain": ref_gain,
                "wrong_fix_gain": wrong_gain,
                "wrong_fix_regret": wrong_fix_regret,
                "score_only_expected_gain": score_only_expected_gain,
                "score_only_regret": score_only_regret,
            }
        )

    mean_referee_gain = round(
        sum(r["referee_chosen_gain"] for r in rows) / len(rows), 4
    )
    mean_wrong_fix_gain = round(
        sum(r["wrong_fix_gain"] for r in rows) / len(rows), 4
    )
    mean_wrong_fix_regret = round(
        sum(r["wrong_fix_regret"] for r in rows) / len(rows), 4
    )
    mean_score_only_regret = round(
        sum(r["score_only_regret"] for r in rows) / len(rows), 4
    )

    return {
        "experiment": "exp4_wrong_fix_cost",
        "seed": seed,
        "intervention_step": delta_knob,
        "mean_referee_gain": mean_referee_gain,
        "mean_wrong_fix_gain": mean_wrong_fix_gain,
        "mean_wrong_fix_regret": mean_wrong_fix_regret,
        "mean_score_only_regret": mean_score_only_regret,
        "sweep_rows": rows,
    }


def run_experiment_5_learned_judge(
    seed: int = DEFAULT_SEED,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> dict[str, Any]:
    """Experiment 5 (Stretch): Learned neural judge vs. lexical proxy."""
    passages, qa_pairs = _fixture_bundle(seed=seed, n_questions=n_questions)
    lexical_scorer = LexicalGroundednessScorer()
    learned_judge, train_stats = train_default_learned_judge(
        train_seed=seed + 999,
        n_train_questions=80,
    )

    eval_examples = build_entailment_dataset(passages, qa_pairs)
    by_phenomenon: dict[str, dict[str, int]] = {}
    total_agree = 0
    lex_correct_total = 0
    judge_correct_total = 0

    for ex in eval_examples:
        lex_pred = 1 if lexical_scorer.score(ex.answer, ex.passages).is_grounded else 0
        judge_pred = 1 if learned_judge.score(ex.answer, ex.passages).is_grounded else 0

        bucket = by_phenomenon.setdefault(
            ex.phenomenon,
            {"count": 0, "agree": 0, "lex_correct": 0, "judge_correct": 0},
        )
        bucket["count"] += 1
        if lex_pred == judge_pred:
            bucket["agree"] += 1
            total_agree += 1
        if lex_pred == ex.label:
            bucket["lex_correct"] += 1
            lex_correct_total += 1
        if judge_pred == ex.label:
            bucket["judge_correct"] += 1
            judge_correct_total += 1

    phenomenon_rows: list[dict[str, Any]] = []
    for phen, stats in sorted(by_phenomenon.items()):
        cnt = stats["count"]
        phenomenon_rows.append(
            {
                "phenomenon": phen,
                "count": cnt,
                "agreement_rate": round(stats["agree"] / cnt, 4),
                "lexical_accuracy": round(stats["lex_correct"] / cnt, 4),
                "learned_judge_accuracy": round(stats["judge_correct"] / cnt, 4),
            }
        )

    ref_lex = GroundednessReferee(scorer=lexical_scorer)
    ref_judge = GroundednessReferee(scorer=learned_judge)  # type: ignore[arg-type]

    decision_checks = [
        ("exp1_pipeline_a", 0.60, 0.95),
        ("exp1_pipeline_b", 0.95, 0.635),
        ("exp4_p1", 0.50, 0.94),
        ("exp4_p7", 0.96, 0.50),
    ]
    changed_conclusions = 0
    pipeline_comparisons: list[dict[str, Any]] = []
    for idx, (label, r_knob, f_knob) in enumerate(decision_checks):
        ret = ScriptedRetriever(
            RetrieverKnobs(recall=r_knob, distractor_rate=0.35, ranking_noise=0.15),
            seed=seed + idx * 19,
        )
        rem = round(1.0 - f_knob, 6)
        ig = round(rem * 0.58, 6)
        fab = round(rem - ig, 6)
        gen = ScriptedGenerator(
            GeneratorKnobs(
                faithful_rate=f_knob,
                ignore_rate=ig,
                fabricate_rate=fab,
                abstain_rate=0.0,
            ),
            seed=seed + idx * 19,
        )
        rep_l = ref_lex.evaluate_pipeline(label, qa_pairs, passages, ret, gen)
        rep_j = ref_judge.evaluate_pipeline(label, qa_pairs, passages, ret, gen)
        same_fix = rep_l.recommended_fix == rep_j.recommended_fix
        if not same_fix:
            changed_conclusions += 1
        pipeline_comparisons.append(
            {
                "pipeline": label,
                "lexical_recommended_fix": rep_l.recommended_fix,
                "learned_recommended_fix": rep_j.recommended_fix,
                "conclusion_unchanged": same_fix,
            }
        )

    n_total = len(eval_examples)
    return {
        "experiment": "exp5_learned_judge",
        "seed": seed,
        "backend": learned_judge.backend,
        "train_stats": train_stats,
        "total_eval_pairs": n_total,
        "overall_agreement_rate": round(total_agree / n_total, 4),
        "lexical_overall_accuracy": round(lex_correct_total / n_total, 4),
        "learned_overall_accuracy": round(judge_correct_total / n_total, 4),
        "changed_experiment_conclusions": changed_conclusions,
        "phenomenon_breakdown": phenomenon_rows,
        "pipeline_decision_comparisons": pipeline_comparisons,
    }


def format_experiment_tables(results: dict[str, dict[str, Any]]) -> str:
    """Format human-readable ASCII summary tables for all experiments."""
    lines: list[str] = []

    exp1 = results["exp1"]
    pa = exp1["pipeline_a"]
    pb = exp1["pipeline_b"]
    lines.append("=== EXPERIMENT 1: TWIN PIPELINES (SAME SCORE, OPPOSITE FIXES) ===")
    lines.append(
        f"{'Pipeline':<34} | {'Score':>6} | {'Ret-Miss':>8} | {'Gen-Ign':>8} | {'Gen-Fab':>8} | {'Correct Fix':<14} | {'Gain(Ret)':>9} | {'Gain(Gen)':>9}"
    )
    lines.append("-" * 116)
    for p in (pa, pb):
        sh = p["referee_shares"]
        iv = p["interventions"]
        lines.append(
            f"{p['pipeline_name']:<34} | {p['groundedness_score']:>6.2f} | "
            f"{sh['retrieval-miss']*100:>7.1f}% | {sh['generation-ignore']*100:>7.1f}% | "
            f"{sh['fabrication-with-evidence-present']*100:>7.1f}% | "
            f"{p['recommended_fix']:<14} | {iv['fix_retriever_gain']:>+9.4f} | {iv['fix_generator_gain']:>+9.4f}"
        )
    lines.append("")

    exp2 = results["exp2"]
    lines.append("=== EXPERIMENT 2: PROBE VALIDATION & CONTEXT-FLIP DEFEAT RATE ===")
    lines.append(
        f"{'Failure Mix':<24} | {'Unsupported':>11} | {'Overall Acc':>11} | {'Ret-Miss Acc':>12} | {'Gen-Ign Acc':>11} | {'Gen-Fab Acc':>11}"
    )
    lines.append("-" * 94)
    for row in exp2["grid_rows"]:
        pca = row["per_cause_accuracy"]
        lines.append(
            f"{row['mix_name']:<24} | {row['unsupported_count']:>11d} | "
            f"{row['attribution_accuracy']*100:>10.1f}% | "
            f"{pca['retrieval-miss']*100:>11.1f}% | "
            f"{pca['generation-ignore']*100:>10.1f}% | "
            f"{pca['fabrication-with-evidence-present']*100:>10.1f}%"
        )
    lines.append("")

    exp3 = results["exp3"]
    lines.append("=== EXPERIMENT 3: FLAG PRECISION AT OPERATING PREVALENCE ===")
    lines.append(
        f"{'Prevalence':>10} | {'Flag Precision (PPV)':>20} | {'False Alarm Share':>17} | {'Min N (1/n Floor)':>17} | {'Min N (95% Zero Bound)':>22}"
    )
    lines.append("-" * 98)
    for row in exp3["prevalence_rows"]:
        lines.append(
            f"{row['hallucination_prevalence']*100:>9.1f}% | "
            f"{row['flag_precision_ppv']*100:>19.1f}% | "
            f"{row['false_alarm_share']*100:>16.1f}% | "
            f"{row['min_n_single_event']:>17d} | "
            f"{row['min_n_95pct_zero_bound']:>22d}"
        )
    lines.append("")

    exp4 = results["exp4"]
    lines.append("=== EXPERIMENT 4: THE COST OF FIXING THE WRONG THING ===")
    lines.append(
        f"{'Pipeline':<28} | {'Base':>6} | {'Referee Fix':<14} | {'Gain(Referee)':>13} | {'Gain(Wrong)':>11} | {'Regret(Wrong)':>13}"
    )
    lines.append("-" * 98)
    for row in exp4["sweep_rows"]:
        lines.append(
            f"{row['pipeline_name']:<28} | {row['base_groundedness']:>6.2f} | "
            f"{row['referee_recommended_fix']:<14} | "
            f"{row['referee_chosen_gain']:>+13.4f} | "
            f"{row['wrong_fix_gain']:>+11.4f} | "
            f"{row['wrong_fix_regret']:>13.4f}"
        )
    lines.append("")

    exp5 = results["exp5"]
    lines.append("=== EXPERIMENT 5 (STRETCH): LEARNED JUDGE VS LEXICAL PROXY ===")
    lines.append(
        f"{'Phenomenon':<26} | {'Pairs':>6} | {'Agreement':>9} | {'Lexical Acc':>11} | {'Learned Acc':>11}"
    )
    lines.append("-" * 74)
    for row in exp5["phenomenon_breakdown"]:
        lines.append(
            f"{row['phenomenon']:<26} | {row['count']:>6d} | "
            f"{row['agreement_rate']*100:>8.1f}% | "
            f"{row['lexical_accuracy']*100:>10.1f}% | "
            f"{row['learned_judge_accuracy']*100:>10.1f}%"
        )
    return "\n".join(lines)


def write_all_experiments(
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> dict[str, dict[str, Any]]:
    """Run Experiments 1 through 5 and write JSON artifacts to `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exp1 = run_experiment_1_twin_pipelines(seed=seed, n_questions=n_questions)
    exp2 = run_experiment_2_probe_validation(seed=seed, n_questions=n_questions)
    exp3 = run_experiment_3_precision_at_prevalence(seed=seed, n_questions=n_questions)
    exp4 = run_experiment_4_wrong_fix_cost(seed=seed, n_questions=n_questions)
    exp5 = run_experiment_5_learned_judge(seed=seed, n_questions=n_questions)

    mapping = {
        "exp1_twin_pipelines.json": exp1,
        "exp2_probe_validation.json": exp2,
        "exp3_precision_at_prevalence.json": exp3,
        "exp4_wrong_fix_cost.json": exp4,
        "exp5_learned_judge.json": exp5,
    }
    for filename, payload in mapping.items():
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4": exp4, "exp5": exp5}
