# ADR 0002: Scripted Components with Declared Behavior Modes Versus a Live LLM

- **Status**: Accepted
- **Date**: 2026-11-17

## Context

Validating a failure-attribution referee requires comparing its predicted blame labels against the true underlying cause of every hallucination across dozens of failure mixes. If we run the benchmark against a live hosted LLM and a dense vector index, two problems arise:

1. **No ground-truth internal state**: When a live LLM emits an ungrounded claim despite a partially relevant passage in context, we do not know with certainty whether the retriever omitted a necessary second hop or the model ignored sufficient evidence.
2. **Non-reproducibility and cost**: API drift, quantization changes, and network requirements prevent deterministic `make verify` runs on a fresh clone.

## Decision

We implement `ScriptedRetriever` and `ScriptedGenerator` as deterministic stand-ins governed by declared probability knobs (`recall`, `distractor_rate`, `ranking_noise` on the retriever; `faithful_rate`, `ignore_rate`, `fabricate_rate`, `abstain_rate`, `context_flip_rate` on the generator) keyed via SHA-256 from `(seed, component, question_id)`.

## Boundary: What Scripted Components Establish vs. What They Cannot

### What this architecture establishes:
- **Protocol arithmetic and identifiability**: It proves that two pipelines can share an identical groundedness score to two decimal places (`0.59`) while requiring opposite interventions, and that counterfactual injection recovers the true component blame across controlled failure mixes.
- **Exact failure boundaries**: By varying `context_flip_rate` from `0.00` to `0.35`, we measure the exact slope at which context-sensitive generator instability defeats counterfactual attribution.
- **Sample-size and prevalence mathematics**: Resolution floors (`1/n` and `3/n`) and Bayesian flag precision (PPV) depend on scorer sensitivity/specificity and operating prevalence, which hold regardless of whether the text came from a scripted stand-in or a 70B transformer.

### What this architecture cannot establish:
- **Any empirical claim about a specific commercial or open-weight LLM**: Scripted rates are inputs, not discoveries about GPT-4, Claude, or Llama.
- **Real-world evidence distributions**: Real corpora contain noisy OCR, ambiguous coreference, and soft entailment gradients that are rougher than planted synthetic templates.
