"""Core data structures for groundedness-referee."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CASE_SINGLE_PASSAGE = "single_passage"
CASE_SPLIT_EVIDENCE = "split_evidence"
CASE_DISTRACTOR_TRAP = "distractor_trap"
CASE_UNANSWERABLE = "unanswerable"

ALL_CASE_TYPES = (
    CASE_SINGLE_PASSAGE,
    CASE_SPLIT_EVIDENCE,
    CASE_DISTRACTOR_TRAP,
    CASE_UNANSWERABLE,
)

MODE_FAITHFUL = "faithful"
MODE_IGNORE_EVIDENCE = "ignore_evidence"
MODE_FABRICATE_WITH_EVIDENCE = "fabricate_with_evidence"
MODE_ABSTAIN = "abstain"

CAUSE_SUPPORTED = "supported"
CAUSE_ABSTAINED = "abstained"
CAUSE_RETRIEVAL_MISS = "retrieval-miss"
CAUSE_GENERATION_IGNORE = "generation-ignore"
CAUSE_FABRICATION_PRESENT = "fabrication-with-evidence-present"
CAUSE_UNATTRIBUTABLE = "unattributable"

FAILURE_CAUSES = (
    CAUSE_RETRIEVAL_MISS,
    CAUSE_GENERATION_IGNORE,
    CAUSE_FABRICATION_PRESENT,
    CAUSE_UNATTRIBUTABLE,
)


@dataclass(frozen=True)
class Passage:
    """A single corpus passage with planted entity-attribute-value metadata."""

    passage_id: str
    doc_id: str
    system_name: str
    attribute: str
    value: str
    text: str
    is_distractor: bool = False
    distractor_kind: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Passage:
        return cls(**data)


@dataclass(frozen=True)
class QAPair:
    """An oracle-labeled question-answer fixture item."""

    question_id: str
    question: str
    case_type: str
    system_name: str
    attribute: str
    gold_passage_ids: list[str]
    distractor_passage_ids: list[str]
    gold_values: list[str]
    expected_answer: str
    partial_answer: str
    prior_myth_answer: str
    fabricated_answer: str
    distractor_misled_answer: str
    paraphrased_answer: str
    negated_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QAPair:
        return cls(**data)


@dataclass(frozen=True)
class RetrievalResult:
    """Output of the scripted retriever for one question."""

    question_id: str
    retrieved_passage_ids: list[str]
    gold_complete: bool
    gold_partial: bool
    distractor_present: bool
    rank_perturbed: bool
    oracle_retrieval_fault: bool


@dataclass(frozen=True)
class GenerationResult:
    """Output of the scripted generator for one question."""

    question_id: str
    answer: str
    drawn_mode: str
    used_passage_ids: list[str]
    context_flipped: bool
    oracle_true_cause: str


@dataclass(frozen=True)
class GroundednessScore:
    """Deterministic lexical entailment proxy score."""

    score: float
    span_overlap: float
    claim_alignment: float
    numeric_value_match: float
    is_grounded: bool
    is_abstention: bool
    threshold: float


@dataclass(frozen=True)
class RefereeVerdict:
    """Single-item referee evaluation and counterfactual probe result."""

    question_id: str
    case_type: str
    initial_answer: str
    initial_score: float
    flagged_unsupported: bool
    abstained: bool
    probe_executed: bool
    probed_answer: str
    probed_score: float
    probed_span_overlap: float
    score_delta: float
    attributed_cause: str
    oracle_cause: str
    correct_attribution: bool


@dataclass
class PipelineReport:
    """Aggregated pipeline evaluation and blame decomposition."""

    pipeline_name: str
    seed: int
    total_questions: int
    answered_questions: int
    abstained_questions: int
    grounded_count: int
    unsupported_count: int
    groundedness_score: float
    overall_grounded_rate: float
    unsupported_rate: float
    referee_counts: dict[str, int] = field(default_factory=dict)
    referee_shares: dict[str, float] = field(default_factory=dict)
    oracle_counts: dict[str, int] = field(default_factory=dict)
    oracle_shares: dict[str, float] = field(default_factory=dict)
    attribution_accuracy: float = 0.0
    per_cause_accuracy: dict[str, float] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    recommended_fix: str = ""
    verdicts: list[RefereeVerdict] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "seed": self.seed,
            "total_questions": self.total_questions,
            "answered_questions": self.answered_questions,
            "abstained_questions": self.abstained_questions,
            "grounded_count": self.grounded_count,
            "unsupported_count": self.unsupported_count,
            "groundedness_score": round(self.groundedness_score, 4),
            "overall_grounded_rate": round(self.overall_grounded_rate, 4),
            "unsupported_rate": round(self.unsupported_rate, 4),
            "referee_counts": self.referee_counts,
            "referee_shares": {k: round(v, 4) for k, v in self.referee_shares.items()},
            "oracle_counts": self.oracle_counts,
            "oracle_shares": {k: round(v, 4) for k, v in self.oracle_shares.items()},
            "attribution_accuracy": round(self.attribution_accuracy, 4),
            "per_cause_accuracy": {
                k: round(v, 4) for k, v in self.per_cause_accuracy.items()
            },
            "confusion_matrix": self.confusion_matrix,
            "recommended_fix": self.recommended_fix,
        }
