# ADR 0001: Blame by Counterfactual Injection Versus Decomposing the Groundedness Score

- **Status**: Accepted
- **Date**: 2026-11-17

## Context

When a RAG pipeline produces an ungrounded answer, engineering teams need to know whether to spend the next sprint fixing retrieval (chunking, embedding recall, reranking) or fixing generation (prompt adherence, decoding temperature, context grounding). A standard groundedness evaluator takes `(answer, retrieved_passages)` and emits a scalar entailment score $S(a, C_{\text{ret}}) \in [0, 1]$.

Teams often attempt to diagnose failures by slicing or decomposing $S(a, C_{\text{ret}})$ into sub-metrics such as unigram overlap, bigram alignment, or numeric match.

## Decision

We do not infer component blame from static sub-scores of $S(a, C_{\text{ret}})$. Instead, for every answer flagged unsupported ($S(a, C_{\text{ret}}) < \tau$), the referee executes a counterfactual probe: it force-injects the gold evidence passages $C_{\text{gold}}$ into the context window, re-invokes the generator on the identical question, and scores the new output $S(a_{\text{probe}}, C_{\text{gold}})$.

- If $S(a_{\text{probe}}, C_{\text{gold}}) \ge \tau$, the fault is attributed to **`retrieval-miss`**.
- If $S(a_{\text{probe}}, C_{\text{gold}}) < \tau$, the fault is attributed to the generator (**`generation-ignore`** when $C_{\text{gold}}$ span overlap is below `0.45`, or **`fabrication-with-evidence-present`** when the generator attends to the passage framing but alters the factual value).

## Rationale

A static score $S(a, C_{\text{ret}})$ is structurally symmetric across two opposite failure modes:

1. **Retrieval miss + faithful generator**: The retriever fetches distractor passages $C_{\text{ret}}$ that omit the target fact. The generator falls back to a prior guess or a distractor value that does not align with the production claim, yielding $S(a, C_{\text{ret}}) < \tau$.
2. **Retrieval hit + unfaithful generator**: The retriever fetches the exact gold passage $C_{\text{gold}}$, but the generator ignores the context or fabricates a number, yielding the exact same $S(a, C_{\text{ret}}) < \tau$.

No algebraic decomposition of $S(a, C_{\text{ret}})$ can separate these two cases without knowing whether supplying the missing evidence changes the generator's behavior. Counterfactual gold injection directly tests the causal counterfactual: *would this generator have grounded its answer if the retriever had done its job?*

## Consequences

- Counterfactual probing requires an offline evaluation dataset where gold supporting passages $C_{\text{gold}}$ are known.
- Each flagged answer incurs one additional generator call during offline evaluation.
- The probe assumes the generator's failure mode is stationary across context edits; a generator that randomly flips behavior whenever prompt length changes (`context_flip_rate > 0`) degrades attribution accuracy, which we measure explicitly in Experiment 2.
