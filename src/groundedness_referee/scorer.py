"""Deterministic lexical groundedness scorer and entailment proxy (M3)."""

from __future__ import annotations

import re
from typing import Sequence

from groundedness_referee.schema import GroundednessScore, Passage

DEFAULT_THRESHOLD = 0.65

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "to",
        "via",
        "with",
        "while",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_BUGGY_ALPHA_ONLY_RE = re.compile(r"[a-z]+")
_NUMERIC_TOKEN_RE = re.compile(r"^\d+[a-z]*$")


def tokenize_content(text: str, preserve_numbers: bool = True) -> list[str]:
    """Extract normalized content tokens from text.

    Note on `preserve_numbers`:
    An early implementation used `_BUGGY_ALPHA_ONLY_RE` (`[a-z]+`), which silently
    dropped numeric literals (`450`, `6x`) and hyphenated system identifiers.
    That allowed `fabricate_with_evidence` answers (which copy the passage's lexical
    framing while altering only the numeric target) to score above 0.85. Setting
    `preserve_numbers=True` is the production fix; `preserve_numbers=False` is
    retained solely for the pinned regression test.
    """
    lower = text.lower()
    raw_tokens = (
        _TOKEN_RE.findall(lower)
        if preserve_numbers
        else _BUGGY_ALPHA_ONLY_RE.findall(lower)
    )
    return [tok for tok in raw_tokens if tok not in STOPWORDS]


def extract_numeric_tokens(tokens: Sequence[str]) -> list[str]:
    """Return tokens containing digits (e.g., '325', '18x', 'v21')."""
    return [
        tok
        for tok in tokens
        if _NUMERIC_TOKEN_RE.match(tok)
        or (any(ch.isdigit() for ch in tok) and not tok.startswith("v"))
    ]


class LexicalGroundednessScorer:
    """Deterministic lexical entailment proxy combining span, bigram, and value alignment.

    Known and documented blind spots (pinned in `tests/test_scorer.py`):
      1. Negation blindness: inserting 'not' or 'never' preserves nearly all content
         tokens, bigrams, and numbers, so a negated claim still scores above threshold.
      2. Cross-passage entity/attribute conflation: if two passages in context mention
         different attributes or numbers, an answer that swaps them still matches all
         unigrams and numeric tokens in the pooled context.
      3. Paraphrase penalty: a faithful semantic paraphrase that replaces domain nouns
         and phrasing with synonyms loses bigram and span overlap and falls below
         threshold despite being entailed.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        preserve_numbers: bool = True,
    ) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"threshold must be in (0.0, 1.0), got {threshold}")
        self.threshold = threshold
        self.preserve_numbers = preserve_numbers

    @staticmethod
    def is_abstention_text(answer: str) -> bool:
        normalized = answer.strip().upper()
        return normalized.startswith("ABSTAIN:") or "INSUFFICIENT EVIDENCE" in normalized

    def score(
        self,
        answer: str,
        passages: Sequence[Passage | str],
    ) -> GroundednessScore:
        """Compute normalized groundedness score of `answer` against `passages`."""
        if self.is_abstention_text(answer):
            return GroundednessScore(
                score=0.0,
                span_overlap=0.0,
                claim_alignment=0.0,
                numeric_value_match=0.0,
                is_grounded=False,
                is_abstention=True,
                threshold=self.threshold,
            )

        ans_tokens = tokenize_content(answer, preserve_numbers=self.preserve_numbers)
        if not ans_tokens or not passages:
            return GroundednessScore(
                score=0.0,
                span_overlap=0.0,
                claim_alignment=0.0,
                numeric_value_match=0.0,
                is_grounded=False,
                is_abstention=False,
                threshold=self.threshold,
            )

        ctx_tokens: list[str] = []
        for p in passages:
            p_text = p.text if isinstance(p, Passage) else str(p)
            ctx_tokens.extend(
                tokenize_content(p_text, preserve_numbers=self.preserve_numbers)
            )

        if not ctx_tokens:
            return GroundednessScore(
                score=0.0,
                span_overlap=0.0,
                claim_alignment=0.0,
                numeric_value_match=0.0,
                is_grounded=False,
                is_abstention=False,
                threshold=self.threshold,
            )

        ctx_token_set = set(ctx_tokens)
        matched_unigrams = sum(1 for tok in ans_tokens if tok in ctx_token_set)
        span_overlap = matched_unigrams / len(ans_tokens)

        if len(ans_tokens) >= 2 and len(ctx_tokens) >= 2:
            ans_bigrams = list(zip(ans_tokens[:-1], ans_tokens[1:]))
            ctx_bigrams = set(zip(ctx_tokens[:-1], ctx_tokens[1:]))
            matched_bigrams = sum(1 for bg in ans_bigrams if bg in ctx_bigrams)
            claim_alignment = matched_bigrams / len(ans_bigrams)
        else:
            claim_alignment = span_overlap

        if self.preserve_numbers:
            ans_nums = extract_numeric_tokens(ans_tokens)
            ctx_nums = set(extract_numeric_tokens(ctx_tokens))
            if ans_nums:
                matched_nums = sum(1 for num in ans_nums if num in ctx_nums)
                numeric_value_match = matched_nums / len(ans_nums)
            else:
                numeric_value_match = span_overlap
            combined = (
                0.40 * span_overlap
                + 0.35 * claim_alignment
                + 0.25 * numeric_value_match
            )
            # If the answer makes explicit numeric claims and not all numbers are
            # supported by the retrieved context, cap the score below threshold
            # (or scale by numeric_value_match) so fabricated or partial numbers fail.
            if ans_nums and numeric_value_match < 1.0:
                combined = min(combined, 0.45 * span_overlap + 0.20 * numeric_value_match)
        else:
            numeric_value_match = span_overlap
            combined = 0.55 * span_overlap + 0.45 * claim_alignment

        combined = max(0.0, min(1.0, round(combined, 6)))
        return GroundednessScore(
            score=combined,
            span_overlap=round(span_overlap, 6),
            claim_alignment=round(claim_alignment, 6),
            numeric_value_match=round(numeric_value_match, 6),
            is_grounded=combined >= self.threshold,
            is_abstention=False,
            threshold=self.threshold,
        )
