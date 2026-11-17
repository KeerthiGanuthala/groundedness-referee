"""Property-sentence test suite for groundedness-referee (M1 through M6)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from groundedness_referee.cli import main as cli_main
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
    load_fixture_files,
    serialize_jsonl,
    write_fixture_files,
)
from groundedness_referee.experiments import (
    run_experiment_1_twin_pipelines,
    run_experiment_2_probe_validation,
    run_experiment_3_precision_at_prevalence,
    run_experiment_4_wrong_fix_cost,
    run_experiment_5_learned_judge,
)
from groundedness_referee.judge import (
    LearnedEntailmentJudge,
    extract_pair_features,
)
from groundedness_referee.referee import GroundednessReferee
from groundedness_referee.schema import (
    CASE_DISTRACTOR_TRAP,
    CASE_SINGLE_PASSAGE,
    CASE_SPLIT_EVIDENCE,
    CASE_UNANSWERABLE,
    CAUSE_RETRIEVAL_MISS,
    Passage,
)
from groundedness_referee.scorer import LexicalGroundednessScorer


class TestGroundednessRefereeProperties(unittest.TestCase):
    """All test names state the invariant or property being verified."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.passages_list, cls.qa_pairs = generate_oracle_fixture(
            seed=DEFAULT_SEED,
            n_questions=DEFAULT_N_QUESTIONS,
        )
        cls.passages_map = {p.passage_id: p for p in cls.passages_list}

    def test_m1_regenerating_fixture_from_same_seed_is_byte_identical(self) -> None:
        p1, q1 = generate_oracle_fixture(seed=DEFAULT_SEED, n_questions=80)
        p2, q2 = generate_oracle_fixture(seed=DEFAULT_SEED, n_questions=80)

        h_p1 = hashlib.sha256(serialize_jsonl(p1).encode("utf-8")).hexdigest()
        h_p2 = hashlib.sha256(serialize_jsonl(p2).encode("utf-8")).hexdigest()
        h_q1 = hashlib.sha256(serialize_jsonl(q1).encode("utf-8")).hexdigest()
        h_q2 = hashlib.sha256(serialize_jsonl(q2).encode("utf-8")).hexdigest()

        self.assertEqual(h_p1, h_p2)
        self.assertEqual(h_q1, h_q2)
        with self.assertRaises(ValueError):
            generate_oracle_fixture(seed=DEFAULT_SEED, n_questions=10)

    def test_m1_fixture_populates_all_four_structural_case_types_including_awkward_cases(
        self,
    ) -> None:
        counts = {
            case_type: sum(1 for q in self.qa_pairs if q.case_type == case_type)
            for case_type in (
                CASE_SINGLE_PASSAGE,
                CASE_SPLIT_EVIDENCE,
                CASE_DISTRACTOR_TRAP,
                CASE_UNANSWERABLE,
            )
        }
        self.assertEqual(counts[CASE_SINGLE_PASSAGE], 120)
        self.assertEqual(counts[CASE_SPLIT_EVIDENCE], 48)
        self.assertEqual(counts[CASE_DISTRACTOR_TRAP], 48)
        self.assertEqual(counts[CASE_UNANSWERABLE], 24)

        for q in self.qa_pairs:
            if q.case_type == CASE_SPLIT_EVIDENCE:
                self.assertEqual(len(q.gold_passage_ids), 2)
            elif q.case_type == CASE_UNANSWERABLE:
                self.assertEqual(len(q.gold_passage_ids), 0)
            else:
                self.assertEqual(len(q.gold_passage_ids), 1)

    def test_m1_fixture_roundtrips_through_jsonl_files_without_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meta = write_fixture_files(tmp_path, seed=DEFAULT_SEED, n_questions=40)
            self.assertEqual(meta["n_questions"], "40")
            loaded_p, loaded_q = load_fixture_files(tmp_path)
            self.assertEqual(len(loaded_q), 40)
            self.assertGreater(len(loaded_p), 80)
            self.assertEqual(loaded_q[0].to_dict(), loaded_q[0].to_dict())

    def test_m2_retriever_and_generator_empirical_rates_match_declared_knobs(
        self,
    ) -> None:
        _, large_qas = generate_oracle_fixture(seed=42, n_questions=400)
        ret = ScriptedRetriever(
            RetrieverKnobs(recall=0.75, distractor_rate=0.40, ranking_noise=0.20),
            seed=42,
        )
        answerable_qas = [q for q in large_qas if q.gold_passage_ids]
        hits = sum(
            1
            for q in answerable_qas
            if ret.retrieve(q, self.passages_map).gold_complete
        )
        empirical_recall = hits / len(answerable_qas)
        self.assertAlmostEqual(empirical_recall, 0.75, delta=0.05)

        gen = ScriptedGenerator(
            GeneratorKnobs(
                faithful_rate=0.60,
                ignore_rate=0.20,
                fabricate_rate=0.15,
                abstain_rate=0.05,
            ),
            seed=42,
        )
        mode_counts: dict[str, int] = {}
        for q in large_qas:
            res = gen.generate(q, q.gold_passage_ids)
            mode_counts[res.drawn_mode] = mode_counts.get(res.drawn_mode, 0) + 1

        self.assertAlmostEqual(
            mode_counts.get("faithful", 0) / len(large_qas), 0.60, delta=0.05
        )
        self.assertAlmostEqual(
            mode_counts.get("ignore_evidence", 0) / len(large_qas), 0.20, delta=0.05
        )
        self.assertAlmostEqual(
            mode_counts.get("fabricate_with_evidence", 0) / len(large_qas),
            0.15,
            delta=0.05,
        )

    def test_m2_invalid_knob_ranges_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            RetrieverKnobs(recall=1.2)
        with self.assertRaises(ValueError):
            RetrieverKnobs(top_k=0)
        with self.assertRaises(ValueError):
            GeneratorKnobs(
                faithful_rate=0.5,
                ignore_rate=0.5,
                fabricate_rate=0.5,
                abstain_rate=0.0,
            )
        with self.assertRaises(ValueError):
            GeneratorKnobs(
                faithful_rate=1.2,
                ignore_rate=0.0,
                fabricate_rate=0.0,
                abstain_rate=0.0,
            )

    def test_m3_scorer_blind_spot_1_negation_polarity_is_invisible_to_lexical_overlap(
        self,
    ) -> None:
        scorer = LexicalGroundednessScorer(threshold=0.65)
        qa = next(
            q for q in self.qa_pairs if q.case_type == CASE_SINGLE_PASSAGE
        )
        gold_p = [self.passages_map[qa.gold_passage_ids[0]]]
        neg_eval = scorer.score(qa.negated_answer, gold_p)
        self.assertTrue(neg_eval.is_grounded)
        self.assertGreaterEqual(neg_eval.score, 0.65)

    def test_m3_scorer_blind_spot_2_cross_passage_value_swap_passes_pooled_context_check(
        self,
    ) -> None:
        scorer = LexicalGroundednessScorer(threshold=0.65)
        p_a = Passage(
            passage_id="p_a",
            doc_id="d1",
            system_name="Atlas-Gateway-v11",
            attribute="failover_timeout",
            value="150 milliseconds",
            text="Production specification for Atlas-Gateway-v11: the leader failover timeout is enforced at 150 milliseconds.",
        )
        p_b = Passage(
            passage_id="p_b",
            doc_id="d1",
            system_name="Borealis-Router-v22",
            attribute="failover_timeout",
            value="450 milliseconds",
            text="Production specification for Borealis-Router-v22: the leader failover timeout is enforced at 450 milliseconds.",
        )
        swapped_claim = (
            "For Atlas-Gateway-v11, the leader failover timeout is enforced at 450 milliseconds."
        )
        eval_res = scorer.score(swapped_claim, [p_a, p_b])
        self.assertTrue(eval_res.is_grounded)

    def test_m3_scorer_blind_spot_3_faithful_synonym_paraphrase_is_penalized_below_threshold(
        self,
    ) -> None:
        scorer = LexicalGroundednessScorer(threshold=0.65)
        qa = next(
            q for q in self.qa_pairs if q.case_type == CASE_SINGLE_PASSAGE
        )
        gold_p = [self.passages_map[qa.gold_passage_ids[0]]]
        para_eval = scorer.score(qa.paraphrased_answer, gold_p)
        self.assertFalse(para_eval.is_grounded)
        self.assertLess(para_eval.score, 0.65)

        with self.assertRaises(ValueError):
            LexicalGroundednessScorer(threshold=1.5)
        empty_eval = scorer.score("", gold_p)
        self.assertEqual(empty_eval.score, 0.0)
        empty_ctx_eval = scorer.score("Atlas timeout", [""])
        self.assertEqual(empty_ctx_eval.score, 0.0)
        single_tok_eval = scorer.score("Atlas", ["Atlas"])
        self.assertGreater(single_tok_eval.score, 0.0)
        self.assertEqual(extract_pair_features("", []), [0.0] * 8)
        judge = LearnedEntailmentJudge()
        self.assertEqual(judge.predict_proba("ABSTAIN: insufficient evidence", gold_p), 0.0)
        self.assertTrue(judge.score("ABSTAIN: insufficient evidence", gold_p).is_abstention)

    def test_m3_pinned_bug_numeric_token_stripping_inflated_fabrication_scores(
        self,
    ) -> None:
        """Pinned bug story:
        In our initial prototype of `tokenize_content()`, we used `re.findall(r'[a-z]+', text.lower())`,
        which stripped digits (`325` -> dropped) and split hyphenated identifiers.
        Because `fabricate_with_evidence` answers preserve the passage's lexical template
        and only corrupt the numeric literal, the buggy tokenizer scored fabricated answers
        at >0.85 groundedness! Preserving alphanumeric tokens and checking `numeric_value_match`
        drops the fabricated answer score below threshold (<0.55).
        """
        qa = next(
            q for q in self.qa_pairs if q.case_type == CASE_SINGLE_PASSAGE
        )
        gold_p = [self.passages_map[qa.gold_passage_ids[0]]]

        buggy_scorer = LexicalGroundednessScorer(
            threshold=0.65, preserve_numbers=False
        )
        fixed_scorer = LexicalGroundednessScorer(
            threshold=0.65, preserve_numbers=True
        )

        buggy_eval = buggy_scorer.score(qa.fabricated_answer, gold_p)
        fixed_eval = fixed_scorer.score(qa.fabricated_answer, gold_p)

        self.assertTrue(buggy_eval.is_grounded)
        self.assertGreater(buggy_eval.score, 0.80)
        self.assertFalse(fixed_eval.is_grounded)
        self.assertLess(fixed_eval.score, 0.55)

    def test_m4_counterfactual_referee_separates_retrieval_miss_from_generation_faults(
        self,
    ) -> None:
        referee = GroundednessReferee()
        ret_weak = ScriptedRetriever(
            RetrieverKnobs(recall=0.50, distractor_rate=0.40, ranking_noise=0.10),
            seed=DEFAULT_SEED,
        )
        gen_faithful = ScriptedGenerator(
            GeneratorKnobs(
                faithful_rate=1.0,
                ignore_rate=0.0,
                fabricate_rate=0.0,
                abstain_rate=0.0,
            ),
            seed=DEFAULT_SEED,
        )
        rep = referee.evaluate_pipeline(
            "pure_retrieval_faults",
            self.qa_pairs,
            self.passages_map,
            ret_weak,
            gen_faithful,
        )
        self.assertEqual(rep.attribution_accuracy, 1.0)
        self.assertEqual(rep.recommended_fix, "fix_retriever")
        self.assertEqual(
            rep.referee_counts[CAUSE_RETRIEVAL_MISS], rep.unsupported_count
        )

    def test_m5_twin_pipelines_have_equal_groundedness_scores_and_opposite_optimal_fixes(
        self,
    ) -> None:
        exp1 = run_experiment_1_twin_pipelines(
            seed=DEFAULT_SEED, n_questions=DEFAULT_N_QUESTIONS
        )
        self.assertTrue(exp1["score_match_2dp"])
        pa = exp1["pipeline_a"]
        pb = exp1["pipeline_b"]
        self.assertEqual(
            f"{pa['groundedness_score']:.2f}",
            f"{pb['groundedness_score']:.2f}",
        )
        self.assertEqual(pa["recommended_fix"], "fix_retriever")
        self.assertEqual(pb["recommended_fix"], "fix_generator")
        self.assertGreater(
            pa["interventions"]["fix_retriever_gain"],
            pa["interventions"]["fix_generator_gain"] * 5.0,
        )
        self.assertGreater(
            pb["interventions"]["fix_generator_gain"],
            pb["interventions"]["fix_retriever_gain"] * 5.0,
        )

    def test_m5_experiments_2_3_4_5_and_cli_produce_consistent_metrics(self) -> None:
        exp2 = run_experiment_2_probe_validation(seed=DEFAULT_SEED, n_questions=40)
        self.assertEqual(exp2["pooled_overall_accuracy"], 1.0)
        self.assertGreater(
            exp2["context_flip_sweep"][-1]["probe_defeat_rate"],
            exp2["context_flip_sweep"][0]["probe_defeat_rate"],
        )

        exp3 = run_experiment_3_precision_at_prevalence(
            seed=DEFAULT_SEED, n_questions=DEFAULT_N_QUESTIONS
        )
        self.assertLess(
            exp3["prevalence_rows"][0]["flag_precision_ppv"],
            exp3["prevalence_rows"][-1]["flag_precision_ppv"],
        )

        exp4 = run_experiment_4_wrong_fix_cost(seed=DEFAULT_SEED, n_questions=40)
        self.assertGreater(exp4["mean_referee_gain"], exp4["mean_wrong_fix_gain"])

        exp5 = run_experiment_5_learned_judge(seed=DEFAULT_SEED, n_questions=40)
        self.assertGreater(
            exp5["learned_overall_accuracy"], exp5["lexical_overall_accuracy"]
        )

        with tempfile.TemporaryDirectory() as tmp, patch("sys.stdout", new=io.StringIO()):
            rc1 = cli_main(["fixtures", "--output-dir", tmp, "--n-questions", "40"])
            self.assertEqual(rc1, 0)
            rc2 = cli_main(["twin-demo"])
            self.assertEqual(rc2, 0)
            rc3 = cli_main(["experiments", "--output-dir", tmp])
            self.assertEqual(rc3, 0)


if __name__ == "__main__":
    unittest.main()
