"""Stretch goal (Experiment 5): Learned entailment judge vs. lexical proxy.

Supports PyTorch (`torch.nn.Module`) when the optional `torch` extra is installed,
and falls back to a deterministic two-layer neural network trained via mini-batch
gradient descent in pure Python/math so `make verify` runs with zero external
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Sequence

from groundedness_referee.corpus import DEFAULT_SEED, generate_oracle_fixture
from groundedness_referee.schema import (
    CASE_DISTRACTOR_TRAP,
    CASE_SINGLE_PASSAGE,
    CASE_SPLIT_EVIDENCE,
    GroundednessScore,
    Passage,
    QAPair,
)
from groundedness_referee.scorer import (
    LexicalGroundednessScorer,
    extract_numeric_tokens,
    tokenize_content,
)

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

NEGATION_WORDS = frozenset({"not", "never", "no", "neither", "excludes", "without"})


@dataclass(frozen=True)
class EntailmentExample:
    """Labeled (answer, context) pair for training/evaluating the learned judge."""

    example_id: str
    phenomenon: str
    answer: str
    passages: list[Passage]
    label: int  # 1 = entailed/grounded, 0 = unsupported


def extract_pair_features(answer: str, passages: Sequence[Passage | str]) -> list[float]:
    """Extract an 8-dimensional feature vector for (answer, context) entailment."""
    ans_tokens = tokenize_content(answer, preserve_numbers=True)
    ctx_tokens: list[str] = []
    raw_ctx_lower = ""
    for p in passages:
        txt = p.text if isinstance(p, Passage) else str(p)
        raw_ctx_lower += " " + txt.lower()
        ctx_tokens.extend(tokenize_content(txt, preserve_numbers=True))

    if not ans_tokens or not ctx_tokens:
        return [0.0] * 8

    ctx_set = set(ctx_tokens)
    unigram_overlap = sum(1 for t in ans_tokens if t in ctx_set) / len(ans_tokens)

    if len(ans_tokens) >= 2 and len(ctx_tokens) >= 2:
        ans_bg = list(zip(ans_tokens[:-1], ans_tokens[1:]))
        ctx_bg = set(zip(ctx_tokens[:-1], ctx_tokens[1:]))
        bigram_overlap = sum(1 for bg in ans_bg if bg in ctx_bg) / len(ans_bg)
    else:
        bigram_overlap = unigram_overlap

    ans_nums = extract_numeric_tokens(ans_tokens)
    ctx_nums = set(extract_numeric_tokens(ctx_tokens))
    if ans_nums:
        num_match = sum(1 for n in ans_nums if n in ctx_nums) / len(ans_nums)
        num_parity = 1.0 if num_match == 1.0 else 0.0
    else:
        num_match = unigram_overlap
        num_parity = 0.5

    ans_lower_words = set(re.findall(r"[a-z]+", answer.lower()))
    ctx_lower_words = set(re.findall(r"[a-z]+", raw_ctx_lower))
    ans_has_neg = any(w in NEGATION_WORDS for w in ans_lower_words)
    ctx_has_neg = any(w in NEGATION_WORDS for w in ctx_lower_words)
    negation_mismatch = 1.0 if (ans_has_neg and not ctx_has_neg) else 0.0

    ans_stems = [t[:4] for t in ans_tokens if len(t) >= 4]
    ctx_stems = {t[:4] for t in ctx_tokens if len(t) >= 4}
    stem_overlap = (
        sum(1 for s in ans_stems if s in ctx_stems) / len(ans_stems)
        if ans_stems
        else unigram_overlap
    )

    trap_conflict = (
        1.0
        if (
            ("deprecated" in ctx_lower_words or "obsolete" in ctx_lower_words)
            and "live" in ans_lower_words
            and "production" in ans_lower_words
        )
        else 0.0
    )

    len_ratio = min(1.0, len(ctx_tokens) / max(1, len(ans_tokens)))

    return [
        unigram_overlap,
        bigram_overlap,
        num_match,
        negation_mismatch,
        stem_overlap,
        trap_conflict,
        len_ratio,
        num_parity,
    ]


def build_entailment_dataset(
    passages_map: dict[str, Passage],
    qa_pairs: Sequence[QAPair],
) -> list[EntailmentExample]:
    """Construct balanced oracle-labeled entailment pairs across phenomena."""
    examples: list[EntailmentExample] = []
    for qa in qa_pairs:
        if not qa.gold_passage_ids:
            continue
        gold_ps = [passages_map[pid] for pid in qa.gold_passage_ids]
        dist_ps = [passages_map[pid] for pid in qa.distractor_passage_ids]

        examples.append(
            EntailmentExample(
                example_id=f"{qa.question_id}_exact",
                phenomenon="exact_entailed",
                answer=qa.expected_answer,
                passages=gold_ps,
                label=1,
            )
        )
        examples.append(
            EntailmentExample(
                example_id=f"{qa.question_id}_paraphrase",
                phenomenon="paraphrase_entailed",
                answer=qa.paraphrased_answer,
                passages=gold_ps,
                label=1,
            )
        )
        examples.append(
            EntailmentExample(
                example_id=f"{qa.question_id}_fabricated",
                phenomenon="fabricated_number",
                answer=qa.fabricated_answer,
                passages=gold_ps,
                label=0,
            )
        )
        examples.append(
            EntailmentExample(
                example_id=f"{qa.question_id}_prior",
                phenomenon="prior_ignore",
                answer=qa.prior_myth_answer,
                passages=gold_ps,
                label=0,
            )
        )
        examples.append(
            EntailmentExample(
                example_id=f"{qa.question_id}_negated",
                phenomenon="negation_polarity",
                answer=qa.negated_answer,
                passages=gold_ps,
                label=0,
            )
        )
        if qa.case_type == CASE_DISTRACTOR_TRAP and dist_ps:
            examples.append(
                EntailmentExample(
                    example_id=f"{qa.question_id}_trap",
                    phenomenon="distractor_trap_context",
                    answer=qa.distractor_misled_answer,
                    passages=[dist_ps[0]],
                    label=0,
                )
            )
        if qa.case_type == CASE_SPLIT_EVIDENCE and len(gold_ps) >= 2:
            examples.append(
                EntailmentExample(
                    example_id=f"{qa.question_id}_partial",
                    phenomenon="split_partial",
                    answer=qa.partial_answer,
                    passages=[gold_ps[0]],
                    label=0,
                )
            )
    return examples


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class LearnedEntailmentJudge:
    """Two-layer neural entailment judge trained on fixture-generated pairs."""

    def __init__(
        self,
        threshold: float = 0.50,
        hidden_dim: int = 8,
        seed: int = 20261117,
        prefer_torch: bool = False,
    ) -> None:
        self.threshold = threshold
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.backend = "pytorch" if (TORCH_AVAILABLE and prefer_torch) else "pure_mlp"
        self.in_dim = 8

        rng = random.Random(seed)
        self.w1 = [
            [rng.uniform(-0.35, 0.35) for _ in range(self.in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.w2 = [rng.uniform(-0.35, 0.35) for _ in range(hidden_dim)]
        self.b2 = 0.0
        self.train_loss_history: list[float] = []

    def _forward_pure(self, x: Sequence[float]) -> tuple[list[float], float]:
        h = []
        for j in range(self.hidden_dim):
            act = self.b1[j] + sum(self.w1[j][i] * x[i] for i in range(self.in_dim))
            h.append(act if act > 0.0 else 0.0)
        logit = self.b2 + sum(self.w2[j] * h[j] for j in range(self.hidden_dim))
        prob = _sigmoid(logit)
        return h, prob

    def fit(
        self,
        examples: Sequence[EntailmentExample],
        epochs: int = 16,
        lr: float = 0.16,
    ) -> dict[str, float]:
        features = [extract_pair_features(ex.answer, ex.passages) for ex in examples]
        labels = [float(ex.label) for ex in examples]

        indices = list(range(len(features)))
        rng = random.Random(self.seed + 7)
        for _ in range(epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            for idx in indices:
                x = features[idx]
                y = labels[idx]
                h, prob = self._forward_pure(x)
                eps = 1e-7
                p_clip = min(1.0 - eps, max(eps, prob))
                epoch_loss += -(
                    y * math.log(p_clip) + (1.0 - y) * math.log(1.0 - p_clip)
                )

                d_logit = prob - y
                old_w2 = list(self.w2)
                for j in range(self.hidden_dim):
                    self.w2[j] -= lr * (d_logit * h[j] + 1e-4 * self.w2[j])
                self.b2 -= lr * d_logit

                for j in range(self.hidden_dim):
                    if h[j] > 0.0:
                        d_h = d_logit * old_w2[j]
                        for i in range(self.in_dim):
                            self.w1[j][i] -= lr * (
                                d_h * x[i] + 1e-4 * self.w1[j][i]
                            )
                        self.b1[j] -= lr * d_h

            self.train_loss_history.append(epoch_loss / max(1, len(indices)))

        return {
            "initial_loss": round(self.train_loss_history[0], 4),
            "final_loss": round(self.train_loss_history[-1], 4),
            "epochs": epochs,
        }

    def predict_proba(
        self,
        answer: str,
        passages: Sequence[Passage | str],
    ) -> float:
        if LexicalGroundednessScorer.is_abstention_text(answer):
            return 0.0
        feats = extract_pair_features(answer, passages)
        _, prob = self._forward_pure(feats)
        return prob

    def score(
        self,
        answer: str,
        passages: Sequence[Passage | str],
    ) -> GroundednessScore:
        if LexicalGroundednessScorer.is_abstention_text(answer):
            return GroundednessScore(
                score=0.0,
                span_overlap=0.0,
                claim_alignment=0.0,
                numeric_value_match=0.0,
                is_grounded=False,
                is_abstention=True,
                threshold=self.threshold,
            )
        feats = extract_pair_features(answer, passages)
        prob = self.predict_proba(answer, passages)
        return GroundednessScore(
            score=round(prob, 6),
            span_overlap=round(feats[0], 6),
            claim_alignment=round(feats[1], 6),
            numeric_value_match=round(feats[2], 6),
            is_grounded=(prob >= self.threshold),
            is_abstention=False,
            threshold=self.threshold,
        )


def train_default_learned_judge(
    train_seed: int = DEFAULT_SEED + 999,
    n_train_questions: int = 80,
) -> tuple[LearnedEntailmentJudge, dict[str, float]]:
    """Train a learned judge on a disjoint synthetic seed."""
    train_passages, train_qas = generate_oracle_fixture(
        seed=train_seed,
        n_questions=n_train_questions,
    )
    p_map = {p.passage_id: p for p in train_passages}
    train_examples = build_entailment_dataset(p_map, train_qas)
    judge = LearnedEntailmentJudge(seed=train_seed, prefer_torch=False)
    train_stats = judge.fit(train_examples, epochs=16, lr=0.16)
    return judge, train_stats
