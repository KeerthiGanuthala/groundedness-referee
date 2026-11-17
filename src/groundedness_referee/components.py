"""Scripted retriever and scripted generator with declared behavior knobs (M2)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

from groundedness_referee.schema import (
    CASE_DISTRACTOR_TRAP,
    CASE_SPLIT_EVIDENCE,
    CASE_UNANSWERABLE,
    CAUSE_ABSTAINED,
    CAUSE_FABRICATION_PRESENT,
    CAUSE_GENERATION_IGNORE,
    CAUSE_RETRIEVAL_MISS,
    CAUSE_SUPPORTED,
    CAUSE_UNATTRIBUTABLE,
    MODE_ABSTAIN,
    MODE_FABRICATE_WITH_EVIDENCE,
    MODE_FAITHFUL,
    MODE_IGNORE_EVIDENCE,
    GenerationResult,
    Passage,
    QAPair,
    RetrievalResult,
)


def _item_rng(seed: int, component: str, question_id: str) -> random.Random:
    digest = hashlib.sha256(
        f"{seed}::{component}::{question_id}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class RetrieverKnobs:
    """Declared behavior knobs for the scripted retriever."""

    recall: float = 0.80
    distractor_rate: float = 0.35
    ranking_noise: float = 0.15
    top_k: int = 3

    def __post_init__(self) -> None:
        for name, val in (
            ("recall", self.recall),
            ("distractor_rate", self.distractor_rate),
            ("ranking_noise", self.ranking_noise),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


@dataclass(frozen=True)
class GeneratorKnobs:
    """Declared behavior knobs for the scripted generator."""

    faithful_rate: float = 0.80
    ignore_rate: float = 0.10
    fabricate_rate: float = 0.08
    abstain_rate: float = 0.02
    context_flip_rate: float = 0.00

    def __post_init__(self) -> None:
        for name, val in (
            ("faithful_rate", self.faithful_rate),
            ("ignore_rate", self.ignore_rate),
            ("fabricate_rate", self.fabricate_rate),
            ("abstain_rate", self.abstain_rate),
            ("context_flip_rate", self.context_flip_rate),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")
        total = (
            self.faithful_rate
            + self.ignore_rate
            + self.fabricate_rate
            + self.abstain_rate
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Generator mode probabilities must sum to 1.0, got {total:.6f}"
            )


class ScriptedRetriever:
    """Stand-in retriever controlled by seed and declared knobs."""

    def __init__(self, knobs: RetrieverKnobs, seed: int = 20261117) -> None:
        self.knobs = knobs
        self.seed = seed

    def retrieve(
        self,
        qa: QAPair,
        all_passages: dict[str, Passage],
        fallback_pool: list[str] | None = None,
    ) -> RetrievalResult:
        rng = _item_rng(self.seed, "retriever", qa.question_id)
        selected_ids: list[str] = []

        has_gold_target = len(qa.gold_passage_ids) > 0
        gold_hit_draw = rng.random() < self.knobs.recall

        if has_gold_target:
            if gold_hit_draw:
                selected_ids.extend(qa.gold_passage_ids)
                gold_complete = True
                gold_partial = False
            else:
                gold_complete = False
                if (
                    qa.case_type == CASE_SPLIT_EVIDENCE
                    and len(qa.gold_passage_ids) >= 2
                    and rng.random() < 0.50
                ):
                    # Partial retrieval miss on split-evidence question: only first passage fetched
                    selected_ids.append(qa.gold_passage_ids[0])
                    gold_partial = True
                else:
                    gold_partial = False
        else:
            gold_complete = False
            gold_partial = False

        distractor_present = False
        for d_id in qa.distractor_passage_ids:
            if len(selected_ids) >= self.knobs.top_k:
                break
            # If gold missed, always include at least one distractor so context is non-empty
            if rng.random() < self.knobs.distractor_rate or not selected_ids:
                selected_ids.append(d_id)
                distractor_present = True

        if not selected_ids and fallback_pool:
            pick = fallback_pool[rng.randint(0, len(fallback_pool) - 1)]
            selected_ids.append(pick)
            distractor_present = True

        rank_perturbed = False
        if len(selected_ids) > 1 and rng.random() < self.knobs.ranking_noise:
            rng.shuffle(selected_ids)
            rank_perturbed = True

        selected_ids = selected_ids[: self.knobs.top_k]
        actual_gold_present = (
            all(gid in selected_ids for gid in qa.gold_passage_ids)
            if has_gold_target
            else False
        )
        oracle_retrieval_fault = has_gold_target and (not actual_gold_present)

        return RetrievalResult(
            question_id=qa.question_id,
            retrieved_passage_ids=selected_ids,
            gold_complete=actual_gold_present,
            gold_partial=gold_partial and not actual_gold_present,
            distractor_present=distractor_present,
            rank_perturbed=rank_perturbed,
            oracle_retrieval_fault=oracle_retrieval_fault,
        )


class ScriptedGenerator:
    """Stand-in generator with declared behavior modes and oracle ground-truth labels."""

    def __init__(self, knobs: GeneratorKnobs, seed: int = 20261117) -> None:
        self.knobs = knobs
        self.seed = seed

    def _draw_mode(self, rng: random.Random) -> str:
        u = rng.random()
        c1 = self.knobs.faithful_rate
        c2 = c1 + self.knobs.ignore_rate
        c3 = c2 + self.knobs.fabricate_rate
        if u < c1:
            return MODE_FAITHFUL
        if u < c2:
            return MODE_IGNORE_EVIDENCE
        if u < c3:
            return MODE_FABRICATE_WITH_EVIDENCE
        return MODE_ABSTAIN

    def generate(
        self,
        qa: QAPair,
        retrieved_passage_ids: list[str],
        is_probe: bool = False,
    ) -> GenerationResult:
        """Generate an answer given question and retrieved context.

        The base mode draw (`faithful`, `ignore_evidence`, `fabricate_with_evidence`, `abstain`)
        is keyed strictly on `(seed, "generator_mode", question_id)` so the generator's
        underlying behavioral disposition for a question is fixed between the initial
        run and the counterfactual probe, unless `context_flip_rate` triggers during
        the probe.
        """
        mode_rng = _item_rng(self.seed, "generator_mode", qa.question_id)
        drawn_mode = self._draw_mode(mode_rng)

        context_flipped = False
        if is_probe and self.knobs.context_flip_rate > 0.0:
            flip_rng = _item_rng(self.seed, "generator_probe_flip", qa.question_id)
            if flip_rng.random() < self.knobs.context_flip_rate:
                context_flipped = True

        has_gold_target = len(qa.gold_passage_ids) > 0
        gold_complete = has_gold_target and all(
            gid in retrieved_passage_ids for gid in qa.gold_passage_ids
        )
        gold_partial = (
            has_gold_target
            and not gold_complete
            and any(gid in retrieved_passage_ids for gid in qa.gold_passage_ids)
        )

        # Case 1: Unanswerable questions (no gold evidence exists in the universe)
        if qa.case_type == CASE_UNANSWERABLE:
            if drawn_mode in (MODE_FAITHFUL, MODE_ABSTAIN):
                return GenerationResult(
                    question_id=qa.question_id,
                    answer="ABSTAIN: insufficient evidence in context.",
                    drawn_mode=drawn_mode,
                    used_passage_ids=[],
                    context_flipped=context_flipped,
                    oracle_true_cause=CAUSE_ABSTAINED,
                )
            # When an unfaithful mode hallucinates on an unanswerable item, no gold
            # passage exists in the corpus to rescue or attribute it via injection.
            hallucinated = (
                qa.prior_myth_answer
                if drawn_mode == MODE_IGNORE_EVIDENCE
                else qa.fabricated_answer
            )
            return GenerationResult(
                question_id=qa.question_id,
                answer=hallucinated,
                drawn_mode=drawn_mode,
                used_passage_ids=list(retrieved_passage_ids),
                context_flipped=context_flipped,
                oracle_true_cause=CAUSE_UNATTRIBUTABLE,
            )

        # Case 2: Explicit abstention mode on answerable question
        if drawn_mode == MODE_ABSTAIN:
            return GenerationResult(
                question_id=qa.question_id,
                answer="ABSTAIN: insufficient evidence in context.",
                drawn_mode=drawn_mode,
                used_passage_ids=[],
                context_flipped=context_flipped,
                oracle_true_cause=CAUSE_ABSTAINED,
            )

        # Case 3: Faithful generator mode
        if drawn_mode == MODE_FAITHFUL:
            if gold_complete:
                if context_flipped:
                    # Context-sensitive instability: flips away from grounded answer on probe
                    return GenerationResult(
                        question_id=qa.question_id,
                        answer=qa.prior_myth_answer,
                        drawn_mode=drawn_mode,
                        used_passage_ids=list(retrieved_passage_ids),
                        context_flipped=True,
                        oracle_true_cause=CAUSE_RETRIEVAL_MISS,
                    )
                return GenerationResult(
                    question_id=qa.question_id,
                    answer=qa.expected_answer,
                    drawn_mode=drawn_mode,
                    used_passage_ids=list(qa.gold_passage_ids),
                    context_flipped=False,
                    oracle_true_cause=CAUSE_SUPPORTED,
                )
            if gold_partial:
                return GenerationResult(
                    question_id=qa.question_id,
                    answer=qa.partial_answer,
                    drawn_mode=drawn_mode,
                    used_passage_ids=list(retrieved_passage_ids),
                    context_flipped=context_flipped,
                    oracle_true_cause=CAUSE_RETRIEVAL_MISS,
                )
            # Gold completely missing from retrieved context: faithful generator falls back
            # to distractor-misled claim (on distractor traps) or parametric fallback
            fallback_answer = (
                qa.distractor_misled_answer
                if qa.case_type == CASE_DISTRACTOR_TRAP
                else qa.prior_myth_answer
            )
            return GenerationResult(
                question_id=qa.question_id,
                answer=fallback_answer,
                drawn_mode=drawn_mode,
                used_passage_ids=list(retrieved_passage_ids),
                context_flipped=context_flipped,
                oracle_true_cause=CAUSE_RETRIEVAL_MISS,
            )

        # Case 4: Ignore-evidence mode (answers from parametric prior)
        if drawn_mode == MODE_IGNORE_EVIDENCE:
            if context_flipped and gold_complete:
                # Chameleon generator: spuriously flips to grounded answer when context changes
                return GenerationResult(
                    question_id=qa.question_id,
                    answer=qa.expected_answer,
                    drawn_mode=drawn_mode,
                    used_passage_ids=list(retrieved_passage_ids),
                    context_flipped=True,
                    oracle_true_cause=CAUSE_GENERATION_IGNORE,
                )
            return GenerationResult(
                question_id=qa.question_id,
                answer=qa.prior_myth_answer,
                drawn_mode=drawn_mode,
                used_passage_ids=list(retrieved_passage_ids),
                context_flipped=False,
                oracle_true_cause=CAUSE_GENERATION_IGNORE,
            )

        # Case 5: Fabricate-with-evidence mode (uses passage framing, fabricates value)
        if drawn_mode == MODE_FABRICATE_WITH_EVIDENCE:
            if context_flipped and gold_complete:
                return GenerationResult(
                    question_id=qa.question_id,
                    answer=qa.expected_answer,
                    drawn_mode=drawn_mode,
                    used_passage_ids=list(retrieved_passage_ids),
                    context_flipped=True,
                    oracle_true_cause=CAUSE_FABRICATION_PRESENT,
                )
            return GenerationResult(
                question_id=qa.question_id,
                answer=qa.fabricated_answer,
                drawn_mode=drawn_mode,
                used_passage_ids=list(retrieved_passage_ids),
                context_flipped=False,
                oracle_true_cause=CAUSE_FABRICATION_PRESENT,
            )

        raise ValueError(f"Unhandled generator mode: {drawn_mode}")
