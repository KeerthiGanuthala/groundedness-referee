# groundedness-referee

A groundedness score says an answer is unsupported, and cannot say whether the retriever failed to fetch the evidence or the generator ignored evidence it had: the score is symmetric between two failures whose fixes point in opposite directions. `groundedness-referee` breaks that symmetry using a counterfactual gold-injection probe, validates the protocol against an oracle-labeled corpus, and measures how much groundedness a score-only decision leaves on the table.

## 1. Twin Pipelines: Identical Score (`0.59`), Opposite Fixes (Experiment 1)

Evaluated on our seeded cohort of `240` oracle-labeled QA pairs across `696` corpus passages (`DEFAULT_SEED = 20261117`), both pipelines report a groundedness score of `0.59` to the second decimal place. A dashboard showing only groundedness treats them as identical. The counterfactual referee decomposes their unsupported answers into opposite root causes and recommends opposite engineering sprints:

| Metric / Verdict | `pipeline_a_retriever_bottleneck` | `pipeline_b_generator_bottleneck` |
| :--- | ---: | ---: |
| **Headline Groundedness Score (2 d.p.)** | **`0.59`** | **`0.59`** |
| Exact Groundedness Score (Grounded / Answered) | `0.5926` (`128 / 216`) | `0.5929` (`134 / 226`) |
| Declared Retriever Recall Knob | `0.60` | `0.95` |
| Declared Generator Faithfulness Knob | `0.95` | `0.635` |
| **Referee Blame: `retrieval-miss`** | **`90.9%`** | `7.6%` |
| **Referee Blame: `generation-ignore`** | `3.4%` | **`41.3%`** |
| **Referee Blame: `fabrication-with-evidence-present`** | `5.7%` | **`40.2%`** |
| **Referee Blame: `unattributable`** | `0.0%` | `10.9%` |
| **Referee Recommended Sprint** | **`fix_retriever`** | **`fix_generator`** |
| Groundedness Delta from Fixing Retriever (`+0.25` recall) | **`+0.2269`** | `+0.0265` |
| Groundedness Delta from Fixing Generator (`+0.25` faithfulness) | `+0.0231` | **`+0.2336`** |

Anchored verification figures from `docs/metrics.json`:
- Pipeline A exact groundedness score is `0.5926`, and Pipeline B exact groundedness score is `0.5929`.
- Pipeline A retrieval-miss share is `90.9%`, whereas Pipeline B generation-ignore share is `41.3%` and Pipeline B fabrication-with-evidence-present share is `40.2%`.
- Pipeline A retriever intervention delta is `+0.2269` (nearly 10x the `+0.0231` gain from tuning the generator), while Pipeline B generator intervention delta is `+0.2336` (nearly 9x the `+0.0265` gain from tuning the retriever).

## 2. Quickstart

Reproduce the twin pipelines, all five experiments, the property test suite (`98.4%` line coverage), and the anchor-phrase number checker from a fresh clone with zero network calls and no external packages required:

```bash
# 1. Run the one-command twin pipeline symmetry-break demo
PYTHONPATH=src python3 -m groundedness_referee.cli twin-demo

# 2. Run the full verification pipeline (fixtures, tests + coverage, metrics, number check)
make verify
```

## 3. System Architecture & Planted Fixture Structure

### 3.1 Oracle-Labeled Corpus (`src/groundedness_referee/corpus.py`)
Every passage and QA pair is deterministically synthesized from `DEFAULT_SEED = 20261117` and written to `data/corpus_passages.jsonl` and `data/qa_oracle_fixture.jsonl`. Because the fixture is constructed with known causal links, every unsupported answer carries an oracle ground-truth cause. The `240` questions span four planted classes, including three awkward cases:

1. **`single_passage` (`120` questions, 50%)**: One gold passage states the live production metric; two distractor passages describe adjacent operational metrics for the same service.
2. **`split_evidence` (`48` questions, 20%)**: Answering requires synthesizing two passages (`gold_a` for regional baseline quota and `gold_b` for the multi-region multiplier). When the retriever fetches only one of the two passages, a faithful generator produces a partially supported claim that fails the groundedness threshold (`retrieval-miss`).
3. **`distractor_trap` (`48` questions, 20%)**: Includes a lexical lookalike passage describing a deprecated staging proposal with the same attribute words but a rejected numeric value.
4. **`unanswerable` (`24` questions, 10%)**: Queries an attribute that does not exist anywhere in the corpus (`gold_passage_ids = []`). Faithful and abstaining generators emit `ABSTAIN`; unfaithful generators hallucinate a claim that cannot be rescued by gold passage injection (`unattributable`).

### 3.2 Scripted Retriever & Scripted Generator (`src/groundedness_referee/components.py`)
- **`ScriptedRetriever`**: Controlled by `recall`, `distractor_rate`, `ranking_noise`, and `top_k`.
- **`ScriptedGenerator`**: Controlled by `faithful_rate`, `ignore_rate`, `fabricate_rate`, `abstain_rate`, and `context_flip_rate`. Each question's behavioral mode is drawn via SHA-256 hashing of `(seed, "generator_mode", question_id)` so the generator's disposition remains stationary between the initial pass and the counterfactual probe unless `context_flip_rate > 0`.

### 3.3 Deterministic Groundedness Scorer (`src/groundedness_referee/scorer.py`)
`LexicalGroundednessScorer` computes a deterministic score in `[0.0, 1.0]` combining content-token span overlap (`0.40`), bigram claim alignment (`0.35`), and numeric literal match (`0.25`) against threshold `0.65`, capping the score below threshold whenever explicit numeric claims in the answer are absent from the retrieved context. Its three structural blind spots (negation polarity, cross-passage entity/value swapping, and synonym paraphrase penalty) and the numeric-stripping regression bug are pinned in `tests/test_suite.py`.

### 3.4 Counterfactual Referee (`src/groundedness_referee/referee.py`)
For every non-abstained answer flagged unsupported (`score < 0.65`), the referee force-injects `gold_passage_ids` at the top of the context and re-queries the generator:
- **Becomes grounded (`probed_score >= 0.65`)**: **`retrieval-miss`**.
- **Stays ungrounded (`probed_score < 0.65`) with low passage span overlap (`< 0.45`)**: **`generation-ignore`** (answered from parametric prior).
- **Stays ungrounded (`probed_score < 0.65`) with high passage span overlap (`>= 0.45`)**: **`fabrication-with-evidence-present`** (used passage framing, fabricated the value).
- **No gold passage exists in corpus or probed answer abstains**: **`unattributable`**.

## 4. Experimental Results (Experiments 2 through 5)

### 4.1 Experiment 2: Probe Validation & Context-Flip Failure Mode
Evaluated across `566` pooled unsupported answers in the five-mix grid (`mix_retrieval_heavy`, `mix_balanced`, `mix_ignore_heavy`, `mix_fabrication_heavy`, `mix_high_stress`), the baseline counterfactual probe attribution accuracy is `100.0%` across all four failure causes when generator disposition is stationary (`context_flip_rate = 0.00`).

**Where the probe fails**: Counterfactual attribution relies on a stationarity assumption that injecting gold evidence only changes the output if missing evidence was the bottleneck. If a brittle or chameleon generator randomly flips its behavior whenever prompt context changes (`context_flip_rate > 0`), an unfaithful generator spuriously grounds on the probe (misattributing generation faults to `retrieval-miss`), while a faithful generator destabilizes on injection:

| `context_flip_rate` | Unsupported Count | Attribution Accuracy | Probe Defeat Rate | Spurious Retrieval Blame Rate |
| ---: | ---: | ---: | ---: | ---: |
| `0.00` | `101` | `100.0%` | `0.0%` | `0.0%` |
| `0.05` | `101` | `97.0%` | `3.0%` | `5.3%` |
| `0.10` | `101` | `95.0%` | `5.0%` | `8.8%` |
| `0.20` | `101` | `88.1%` | `11.9%` | `19.3%` |
| `0.35` | `101` | `70.3%` | `29.7%` | `47.4%` |

In particular, at a 20% context-flip rate, probe defeat rate reaches `11.9%`, and at 35% context instability it reaches `29.7%`.

### 4.2 Experiment 3: Flag Precision at Operating Prevalence & Sample-Size Floors
With empirical lexical scorer sensitivity `TPR = 100.0%` and false-positive rate `FPR = 4.2%` (caused by faithful paraphrases falling below the `0.65` lexical threshold), Bayes' rule dictates that low-prevalence hallucination flags are dominated by false alarms:

| True Hallucination Prevalence | Flag Precision (PPV) | False Alarm Share | Min $n$ (`1/n` Floor) | Min $n$ (95% Zero-Event Bound) | Min $n$ (for 96 Flagged Events) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1.0%` | `19.5%` | `80.5%` | `100` | `300` | `9600` |
| `2.0%` | `32.9%` | `67.1%` | `50` | `150` | `4800` |
| `5.0%` | `55.8%` | `44.2%` | `20` | `60` | `1920` |
| `10.0%` | `72.7%` | `27.3%` | `10` | `30` | `960` |
| `20.0%` | `85.7%` | `14.3%` | `5` | `15` | `480` |
| `35.0%` | `92.8%` | `7.2%` | `3` | `9` | `275` |

Specifically, at 1.0% hallucination prevalence, flag precision (PPV) is `19.5%`, meaning more than four out of five unsupported flags are false alarms on paraphrased answers. Furthermore, the 95% rule-of-three zero-event sample floor at 1.0% prevalence is `300` queries (and estimating blame shares within $\pm 10\%$ at 95% confidence requires `96` flagged hallucinations, or `9600` total queries at `1.0%` prevalence).

### 4.3 Experiment 4: The Cost of Fixing the Wrong Thing
Across a seven-pipeline sweep spanning severe retrieval bottlenecks (`recall = 0.50, faithful = 0.94`) to severe generator bottlenecks (`recall = 0.96, faithful = 0.50`), we simulate spending one engineering sprint on either `+0.20` retriever recall or `+0.20` generator faithfulness:

- The mean referee-guided groundedness gain across the sweep is `+0.1896`, compared to `+0.0780` when applying the wrong intervention.
- Consequently, the mean wrong-intervention regret is `0.1116` (`11.16` percentage points of groundedness left on the table, and `0.0558` expected regret compared to a score-only 50/50 guess).

### 4.4 Experiment 5 (Stretch): Learned Neural Judge vs. Lexical Proxy
We train a two-layer neural classifier (`LearnedEntailmentJudge` in `src/groundedness_referee/judge.py`) on a disjoint training seed (`20262116`) and compare it against `LexicalGroundednessScorer` on `1176` benchmark pairs:

- The learned judge and lexical proxy agree on `67.3%` of pairs (`100.0%` agreement on exact entailment, numeric fabrication, prior ignore, distractor traps, and split partial claims).
- The learned judge resolves the lexical proxy's negation blindness (`100.0%` vs. `0.0%` accuracy) and paraphrase penalty (`100.0%` vs. `22.2%` accuracy), lifting overall pair accuracy from `67.3%` to `100.0%`.
- Number of experiment-level intervention conclusions changed: `0`.

## 5. What This Project Cannot Measure & Out-of-Scope Boundaries

### What this cannot measure:
- **Specific production LLM hallucination rates**: Our retriever and generator use declared probability knobs keyed from a seed to prove the attribution protocol's validity and failure boundaries; they do not measure any commercial or open-weight model's actual prior bias.
- **Real-world evidence messiness**: Production documents contain malformed tables, conflicting timestamps, and nuanced pragmatic implications beyond planted synthetic specifications.

### Out of scope, on purpose:
- **Multi-hop chained attribution**: Attributing blame across iterative retrieval rounds (`Retrieve -> Rewrite Query -> Retrieve -> Generate`) is out of scope. **Trigger for when this becomes necessary**: the first pipeline where single-round gold injection still fails because an upstream query-rewriter corrupted the intermediate hop before final generation.
- **Span-level inline citation checking**: Verifying `[1]` bracketed sentence citations is orthogonal to component-level sprint allocation.
- **Online evaluation against live traffic**: Production traffic lacks pre-known `gold_passage_ids`; running counterfactual probes online requires asynchronous shadow labeling on a curated golden set.

## 6. First-Person Answers to the Five Core Engineering Questions (Section 10)

1. **Your groundedness score dropped six points this week: which team should look at it?**
   I cannot tell from the six-point drop alone, and neither can any static groundedness dashboard. Before assigning the sprint, I run the counterfactual referee on the newly flagged unsupported answers by force-injecting their gold passages from our evaluation fixture: if the answers become grounded (`probed_score >= 0.65`), I assign the regression to the retrieval team (`retrieval-miss`); if they stay ungrounded, I assign it to the generation/prompting team (`generation-ignore` or `fabrication-with-evidence-present`).
2. **Why can two pipelines with the same score need opposite fixes?**
   Because a groundedness score $S(a, C_{\text{ret}})$ only measures alignment between the final answer and the retrieved passages. In Experiment 1, Pipeline A (`recall = 0.60, faithful = 0.95`) and Pipeline B (`recall = 0.95, faithful = 0.635`) both score `0.59` (`0.5926` vs. `0.5929`), yet `90.9%` of Pipeline A's failures occur because the retriever missed the gold passage (+`0.2269` gain from fixing retrieval), whereas `81.5%` of Pipeline B's failures occur because the generator ignored or fabricated over gold evidence that was already in context (+`0.2336` gain from fixing generation).
3. **How does a counterfactual probe attribute blame, and what kind of generator defeats it?**
   The probe re-asks the generator the exact same question with the gold evidence force-injected into context and reads blame off the score transition. A generator that flips its answer mode whenever context length or ordering changes (`context_flip_rate > 0`) defeats the probe: in Experiment 2, a `20%` context-flip rate causes an `11.9%` probe defeat rate (and a `19.3%` spurious retrieval blame rate) because unfaithful runs accidentally ground when re-prompted.
4. **At your prevalence, what does a hallucination flag actually mean?**
   With our lexical scorer's `100.0%` sensitivity and `4.2%` false-positive rate (driven by legitimate paraphrases), Bayes' theorem shows that at `1.0%` true hallucination prevalence, a hallucination flag has a positive predictive value of only `19.5%` (`80.5%` of flags are false alarms). Only when prevalence reaches `10.0%` (`72.7%` PPV) or `20.0%` (`85.7%` PPV) does a raw flag represent a likely hallucination.
5. **What sample size would the claim on your dashboard need before it means anything?**
   A measured zero over $n$ queries only bounds the failure rate below $1/n$ (the resolution floor) or $3/n$ at 95% confidence (rule of three). To claim a hallucination rate is below `1.0%` with 95% confidence upon observing zero failures, I need at least `300` evaluation queries (`100` just to clear the `1/n` single-event floor), and to estimate blame decomposition shares within $\pm 10\%$ at `1.0%` prevalence, I need `9600` queries so that at least `96` true unsupported events are observed.

## 7. Architecture Decision Records (ADRs)

- [ADR 0001: Blame by Counterfactual Injection Versus Decomposing the Groundedness Score](docs/adr/0001-blame-by-counterfactual-injection-vs-score-decomposition.md)
- [ADR 0002: Scripted Components with Declared Behavior Modes Versus a Live LLM](docs/adr/0002-scripted-components-with-declared-knobs-vs-live-llm.md)
- [ADR 0003: Lexical Entailment Proxy as Default with Learned Judge Behind an Optional Extra](docs/adr/0003-lexical-entailment-proxy-vs-learned-judge.md)
- [ADR 0004: Oracle Labels Planted at Generation Time Versus Annotated After the Fact](docs/adr/0004-oracle-labels-planted-at-generation-time.md)
- [ADR 0005: Deterministic JSONL Fixtures Regenerated from Seed (Nothing Hand-Edited)](docs/adr/0005-deterministic-jsonl-fixtures-from-seed.md)
- [Build Journal](docs/BUILD_JOURNAL.md)
