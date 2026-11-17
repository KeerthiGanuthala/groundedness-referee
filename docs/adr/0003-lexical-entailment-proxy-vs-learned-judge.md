# ADR 0003: Lexical Entailment Proxy as Default with Learned Judge Behind an Optional Extra

- **Status**: Accepted
- **Date**: 2026-11-17

## Context

The referee needs a groundedness scorer to decide whether an initial answer is supported by retrieved passages and whether a probed answer becomes supported after gold evidence is injected. A neural natural-language inference (NLI) model captures semantic paraphrase and negation better than token overlap, but requiring `torch` in the default install path bloats dependencies and slows CI verification.

## Decision

We make `LexicalGroundednessScorer` (combining content-token span overlap, bigram claim alignment, and numeric literal verification at threshold `0.65`) the default offline scorer with zero external dependencies. We place the neural classifier (`LearnedEntailmentJudge`) in `src/groundedness_referee/judge.py` with `torch` isolated under `[project.optional-dependencies]` in `pyproject.toml` and a deterministic pure-Python two-layer MLP fallback for dependency-free training and evaluation.

## Measured Comparison (Experiment 5)

Across `1176` oracle-labeled `(answer, passage)` evaluation pairs spanning seven linguistic phenomena:

- The measured agreement rate between the lexical proxy and learned judge is `67.3%`.
- They agree `100.0%` of the time on exact entailed claims, fabricated numeric claims, parametric prior myths, distractor-trap contexts, and split-evidence partial claims.
- They disagree on two phenomena that correspond to the lexical proxy's pinned blind spots:
  1. **Negation polarity (`negation_polarity`)**: The lexical proxy scores `0.0%` accuracy because inserting `not` or `never` preserves content unigrams, bigrams, and numbers; the learned judge uses the explicit negation-mismatch feature to reach `100.0%` accuracy.
  2. **Semantic paraphrase (`paraphrase_entailed`)**: The lexical proxy scores `22.2%` accuracy because synonym substitution drops bigram overlap below `0.65`; the learned judge leverages stem overlap and numeric parity to reach `100.0%` accuracy.
- Across all pipeline-level intervention checks in Experiment 5, swapping the lexical proxy for the learned judge changed `0` experiment-level intervention conclusions (`fix_retriever` vs. `fix_generator`).
