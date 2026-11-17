# Build Journal: `groundedness-referee`

A chronological engineering log recorded while building `groundedness-referee` from M1 through M6 and the stretch goals.

## Milestone 1 (M1): Oracle-Labeled Corpus & Awkward Cases
- Designed `Passage` and `QAPair` schemas in `src/groundedness_referee/schema.py` and the seeded generator in `src/groundedness_referee/corpus.py` (`DEFAULT_SEED = 20261117`, `240` questions, `696` passages).
- Planted four explicit structural classes:
  1. `single_passage` (`120` items): one gold passage states the live production specification; two distractors cover adjacent attributes.
  2. `split_evidence` (`48` items): answering requires joining two distinct gold passages (`gold_a` for baseline quota and `gold_b` for the multi-region multiplier). Fetching only one passage leaves the second clause unsupported.
  3. `distractor_trap` (`48` items): includes a high-overlap staging proposal passage (`lexical_lookalike_trap`) that shares the target attribute phrasing but states a rejected numeric value.
  4. `unanswerable` (`24` items): queries an unplanted attribute (`gold_passage_ids = []`) where abstention is the only valid behavior.
- Verified byte-identical JSONL serialization (`passages_sha256` prefix `a94a7309491e`, `qa_sha256` prefix `210376de6247`).

## Milestone 2 (M2): Scripted Retriever & Generator with Declared Knobs
- Implemented `ScriptedRetriever` (`recall`, `distractor_rate`, `ranking_noise`, `top_k`) and `ScriptedGenerator` (`faithful_rate`, `ignore_rate`, `fabricate_rate`, `abstain_rate`, `context_flip_rate`) in `src/groundedness_referee/components.py`.
- Keyed per-item PRNG streams with `hashlib.sha256(f"{seed}::{component}::{question_id}")` so every question's retrieval outcome and generator mode are invariant to batch ordering and preserved between the initial pass and the counterfactual probe.
- Verified on a 400-question cohort that empirical recall and generator mode frequencies land within `< 0.05` of declared knob settings.

## Milestone 3 (M3): Deterministic Groundedness Scorer & Pinned Bug Story
- Built `LexicalGroundednessScorer` in `src/groundedness_referee/scorer.py` with threshold `0.65`.
- **Pinned Bug Story (`test_m3_pinned_bug_numeric_token_stripping_inflated_fabrication_scores`)**:
  - *What broke*: Our initial tokenizer used `re.findall(r"[a-z]+", text.lower())`, which stripped numeric literals (`325`, `18x`) and split hyphenated system names. Because `fabricate_with_evidence` answers copy the gold passage's sentence structure and only corrupt the numeric literal, the alpha-only tokenizer scored fabricated answers above `0.80` (`is_grounded = True`), letting every numeric fabrication slip past the filter.
  - *The fix*: Switched to `r"[a-z0-9]+(?:-[a-z0-9]+)*"`, added `extract_numeric_tokens()` to compute `numeric_value_match`, and capped the combined score whenever numeric literals in the answer fail to match the context. Fabricated answers now score `< 0.55` (`is_grounded = False`).
- Pinned all three structural blind spots of lexical overlap with dedicated tests:
  1. Negation polarity blindness (`not` / `never` preserves unigrams, bigrams, and numbers).
  2. Cross-passage value swapping when multiple entities appear in pooled context.
  3. Synonym paraphrase penalty dropping entailed answers below `0.65`.

## Milestone 4 (M4): Counterfactual Referee Protocol
- Implemented `GroundednessReferee` in `src/groundedness_referee/referee.py`.
- For every non-abstained answer scoring `< 0.65`, the referee checks whether gold passages exist (`unattributable` if empty), force-injects `gold_passage_ids` at the head of the context, re-asks the generator with `is_probe=True`, and classifies the outcome into `retrieval-miss`, `generation-ignore`, `fabrication-with-evidence-present`, or `unattributable`.

## Milestone 5 (M5): Experiments 1 through 4 & Stretch Experiment 5
- **Calibration note on Experiment 1**: On the first run, matching raw `grounded_count` (`128`) between Pipeline A and Pipeline B left a `0.02` gap in `groundedness_score` (`0.5926` vs. `0.5664`) because Pipeline B's unfaithful generator hallucinated on `10` of the `24` `unanswerable` questions instead of abstaining, raising its denominator from `216` to `226` answered items. Updating the calibration target to match `groundedness_score` directly aligned Pipeline A (`128/216 = 0.5926`) and Pipeline B (`134/226 = 0.5929`) at `0.59` to two decimal places.
- Built `LearnedEntailmentJudge` in `src/groundedness_referee/judge.py` for Stretch Experiment 5, training on a disjoint seed (`20262116`) and evaluating on the `1176` benchmark entailment pairs.

## Milestone 6 (M6): Measurement Discipline, Coverage & Verification
- Built `tools/collect_metrics.py` to write `docs/metrics.json` and `docs/experiments/*.json`.
- Built `tools/check_numbers.py` with 22 regex anchor-phrase rules across `README.md` and ADRs, plus `--prove-breakage` which mutates a figure in memory and proves the checker rejects stale documentation.
- Built `tools/run_tests.py` using `ast` statement discovery and `sys.settrace` (enabled prior to module import so schema and dataclass definitions are traced), achieving `98.4%` line coverage across `src/groundedness_referee/`.
