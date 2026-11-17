"""The referee and counterfactual injection probe (M4)."""

from __future__ import annotations

from typing import Sequence

from groundedness_referee.components import ScriptedGenerator, ScriptedRetriever
from groundedness_referee.schema import (
    CAUSE_ABSTAINED,
    CAUSE_FABRICATION_PRESENT,
    CAUSE_GENERATION_IGNORE,
    CAUSE_RETRIEVAL_MISS,
    CAUSE_SUPPORTED,
    CAUSE_UNATTRIBUTABLE,
    FAILURE_CAUSES,
    Passage,
    PipelineReport,
    QAPair,
    RefereeVerdict,
)
from groundedness_referee.scorer import LexicalGroundednessScorer


class GroundednessReferee:
    """Counterfactual failure attribution referee for RAG pipelines.

    Protocol:
      1. Score the initial pipeline answer against the initially retrieved passages.
      2. If the answer is flagged unsupported (`initial_score < threshold` and not
         an abstention), check whether gold evidence passages exist in the corpus.
         - If no gold passage exists (`len(qa.gold_passage_ids) == 0`), classify as
           `unattributable`.
      3. Otherwise, run the counterfactual probe: force-inject the gold passage(s)
         into the context window and re-ask the generator (`is_probe=True`).
      4. Score the probed answer against the gold-injected context (`probed_score`):
         - If `probed_score >= threshold`: the generator produced a grounded answer
           once given the gold evidence -> `retrieval-miss`.
         - If `probed_score < threshold`: the generator stayed ungrounded even with
           gold evidence present -> generation fault:
             * If `probed_span_overlap >= fabrication_overlap_floor`: the generator
               attended to the passage's entities and framing but fabricated the
               claim/value -> `fabrication-with-evidence-present`.
             * If `probed_span_overlap < fabrication_overlap_floor`: the generator
               ignored the injected passages and answered from its prior ->
               `generation-ignore`.
    """

    def __init__(
        self,
        scorer: LexicalGroundednessScorer | None = None,
        fabrication_overlap_floor: float = 0.45,
    ) -> None:
        self.scorer = scorer or LexicalGroundednessScorer()
        self.fabrication_overlap_floor = fabrication_overlap_floor

    def evaluate_item(
        self,
        qa: QAPair,
        passages: dict[str, Passage],
        retriever: ScriptedRetriever,
        generator: ScriptedGenerator,
        fallback_pool: list[str] | None = None,
    ) -> RefereeVerdict:
        ret = retriever.retrieve(qa, passages, fallback_pool=fallback_pool)
        gen = generator.generate(qa, ret.retrieved_passage_ids, is_probe=False)

        retrieved_passage_objs = [
            passages[pid] for pid in ret.retrieved_passage_ids if pid in passages
        ]
        init_eval = self.scorer.score(gen.answer, retrieved_passage_objs)

        if init_eval.is_abstention:
            return RefereeVerdict(
                question_id=qa.question_id,
                case_type=qa.case_type,
                initial_answer=gen.answer,
                initial_score=init_eval.score,
                flagged_unsupported=False,
                abstained=True,
                probe_executed=False,
                probed_answer=gen.answer,
                probed_score=init_eval.score,
                probed_span_overlap=init_eval.span_overlap,
                score_delta=0.0,
                attributed_cause=CAUSE_ABSTAINED,
                oracle_cause=gen.oracle_true_cause,
                correct_attribution=(gen.oracle_true_cause == CAUSE_ABSTAINED),
            )

        if init_eval.is_grounded:
            return RefereeVerdict(
                question_id=qa.question_id,
                case_type=qa.case_type,
                initial_answer=gen.answer,
                initial_score=init_eval.score,
                flagged_unsupported=False,
                abstained=False,
                probe_executed=False,
                probed_answer=gen.answer,
                probed_score=init_eval.score,
                probed_span_overlap=init_eval.span_overlap,
                score_delta=0.0,
                attributed_cause=CAUSE_SUPPORTED,
                oracle_cause=gen.oracle_true_cause,
                correct_attribution=(gen.oracle_true_cause == CAUSE_SUPPORTED),
            )

        # Answer is flagged unsupported (`init_eval.score < threshold`)
        if not qa.gold_passage_ids:
            return RefereeVerdict(
                question_id=qa.question_id,
                case_type=qa.case_type,
                initial_answer=gen.answer,
                initial_score=init_eval.score,
                flagged_unsupported=True,
                abstained=False,
                probe_executed=False,
                probed_answer=gen.answer,
                probed_score=init_eval.score,
                probed_span_overlap=init_eval.span_overlap,
                score_delta=0.0,
                attributed_cause=CAUSE_UNATTRIBUTABLE,
                oracle_cause=gen.oracle_true_cause,
                correct_attribution=(gen.oracle_true_cause == CAUSE_UNATTRIBUTABLE),
            )

        # Counterfactual probe: force-inject gold passage(s) at top of context
        probed_context_ids = list(qa.gold_passage_ids)
        for pid in ret.retrieved_passage_ids:
            if pid not in probed_context_ids:
                probed_context_ids.append(pid)

        probed_gen = generator.generate(qa, probed_context_ids, is_probe=True)
        gold_passage_objs = [
            passages[pid] for pid in qa.gold_passage_ids if pid in passages
        ]
        probed_eval = self.scorer.score(probed_gen.answer, gold_passage_objs)
        score_delta = round(probed_eval.score - init_eval.score, 6)

        if probed_eval.is_abstention:
            attributed_cause = CAUSE_UNATTRIBUTABLE
        elif probed_eval.is_grounded:
            attributed_cause = CAUSE_RETRIEVAL_MISS
        else:
            if probed_eval.span_overlap >= self.fabrication_overlap_floor:
                attributed_cause = CAUSE_FABRICATION_PRESENT
            else:
                attributed_cause = CAUSE_GENERATION_IGNORE

        return RefereeVerdict(
            question_id=qa.question_id,
            case_type=qa.case_type,
            initial_answer=gen.answer,
            initial_score=init_eval.score,
            flagged_unsupported=True,
            abstained=False,
            probe_executed=True,
            probed_answer=probed_gen.answer,
            probed_score=probed_eval.score,
            probed_span_overlap=probed_eval.span_overlap,
            score_delta=score_delta,
            attributed_cause=attributed_cause,
            oracle_cause=gen.oracle_true_cause,
            correct_attribution=(attributed_cause == gen.oracle_true_cause),
        )

    def evaluate_pipeline(
        self,
        pipeline_name: str,
        qa_pairs: Sequence[QAPair],
        passages: dict[str, Passage],
        retriever: ScriptedRetriever,
        generator: ScriptedGenerator,
    ) -> PipelineReport:
        fallback_pool = sorted(passages.keys())
        verdicts: list[RefereeVerdict] = []

        for qa in qa_pairs:
            verdict = self.evaluate_item(
                qa=qa,
                passages=passages,
                retriever=retriever,
                generator=generator,
                fallback_pool=fallback_pool,
            )
            verdicts.append(verdict)

        total_q = len(verdicts)
        abstained_q = sum(1 for v in verdicts if v.abstained)
        answered_q = total_q - abstained_q
        grounded_count = sum(
            1 for v in verdicts if not v.abstained and not v.flagged_unsupported
        )
        unsupported_count = sum(1 for v in verdicts if v.flagged_unsupported)

        groundedness_score = (
            grounded_count / answered_q if answered_q > 0 else 0.0
        )
        overall_grounded_rate = grounded_count / total_q if total_q > 0 else 0.0
        unsupported_rate = (
            unsupported_count / answered_q if answered_q > 0 else 0.0
        )

        referee_counts = {c: 0 for c in FAILURE_CAUSES}
        oracle_counts = {c: 0 for c in FAILURE_CAUSES}
        confusion: dict[str, dict[str, int]] = {
            true_c: {pred_c: 0 for pred_c in FAILURE_CAUSES}
            for true_c in FAILURE_CAUSES
        }

        unsupported_verdicts = [v for v in verdicts if v.flagged_unsupported]
        for v in unsupported_verdicts:
            if v.attributed_cause in referee_counts:
                referee_counts[v.attributed_cause] += 1
            if v.oracle_cause in oracle_counts:
                oracle_counts[v.oracle_cause] += 1
            if v.oracle_cause in confusion and v.attributed_cause in FAILURE_CAUSES:
                confusion[v.oracle_cause][v.attributed_cause] += 1

        referee_shares = {
            c: (referee_counts[c] / unsupported_count if unsupported_count > 0 else 0.0)
            for c in FAILURE_CAUSES
        }
        oracle_shares = {
            c: (oracle_counts[c] / unsupported_count if unsupported_count > 0 else 0.0)
            for c in FAILURE_CAUSES
        }

        correct_unsupported = sum(
            1 for v in unsupported_verdicts if v.correct_attribution
        )
        attribution_accuracy = (
            correct_unsupported / unsupported_count
            if unsupported_count > 0
            else 1.0
        )

        per_cause_accuracy: dict[str, float] = {}
        for c in FAILURE_CAUSES:
            denom = oracle_counts[c]
            if denom > 0:
                per_cause_accuracy[c] = confusion[c][c] / denom
            else:
                per_cause_accuracy[c] = 1.0

        ret_blame = referee_counts[CAUSE_RETRIEVAL_MISS]
        gen_blame = (
            referee_counts[CAUSE_GENERATION_IGNORE]
            + referee_counts[CAUSE_FABRICATION_PRESENT]
        )
        recommended_fix = (
            "fix_retriever" if ret_blame >= gen_blame else "fix_generator"
        )

        return PipelineReport(
            pipeline_name=pipeline_name,
            seed=retriever.seed,
            total_questions=total_q,
            answered_questions=answered_q,
            abstained_questions=abstained_q,
            grounded_count=grounded_count,
            unsupported_count=unsupported_count,
            groundedness_score=groundedness_score,
            overall_grounded_rate=overall_grounded_rate,
            unsupported_rate=unsupported_rate,
            referee_counts=referee_counts,
            referee_shares=referee_shares,
            oracle_counts=oracle_counts,
            oracle_shares=oracle_shares,
            attribution_accuracy=attribution_accuracy,
            per_cause_accuracy=per_cause_accuracy,
            confusion_matrix=confusion,
            recommended_fix=recommended_fix,
            verdicts=verdicts,
        )
