# ADR 0004: Oracle Labels Planted at Generation Time Versus Annotated After the Fact

- **Status**: Accepted
- **Date**: 2026-11-17

## Context

To validate whether an attribution referee is accurate, we need ground-truth blame labels (`retrieval-miss`, `generation-ignore`, `fabrication-with-evidence-present`, `unattributable`) for every unsupported answer. In typical RAG benchmarks, human annotators or an LLM-as-a-judge inspect failure traces after the fact and guess why the pipeline failed.

## Decision

We plant the oracle labels at generation time:

1. The corpus generator constructs each question alongside its exact `gold_passage_ids`, `distractor_passage_ids`, and structural case type (`single_passage`, `split_evidence`, `distractor_trap`, `unanswerable`).
2. During pipeline execution, the scripted retriever and generator record the exact causal draw (`oracle_retrieval_fault` and `oracle_true_cause`) before the scorer or referee ever inspects the output.

## Rationale

Post-hoc annotation is circular when validating an attribution protocol:

- If a human or LLM annotator labels an error by reading the retrieved passages and the generated answer, the annotator is trapped in the exact same observational symmetry as the static groundedness score whenever evidence is split or a generator fabricates a plausible-sounding claim on a missing passage.
- If the annotator instead runs a counterfactual re-prompt to decide the label, the annotation process *is* the counterfactual probe, which merely demonstrates self-consistency rather than validating the probe against independent truth.

Planting the causal mechanism at generation time is the only way to measure true attribution precision, recall, and confusion matrices against a known data-generating process.
