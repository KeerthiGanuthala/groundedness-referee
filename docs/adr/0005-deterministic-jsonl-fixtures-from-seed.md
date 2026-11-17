# ADR 0005: Deterministic JSONL Fixtures Regenerated from Seed (Nothing Hand-Edited)

- **Status**: Accepted
- **Date**: 2026-11-17

## Context

Hand-curated JSON or CSV test fixtures invite silent drift: a contributor edits one passage to fix a failing test, forgets to update the corresponding oracle metadata, and breaks reproducibility across cohort sizes.

## Decision

All corpus passages (`data/corpus_passages.jsonl`) and oracle-labeled QA pairs (`data/qa_oracle_fixture.jsonl`) are generated deterministically from `DEFAULT_SEED = 20261117` by `groundedness_referee.corpus.write_fixture_files()` and serialized in canonical key-sorted ASCII JSONL. Hand-editing fixture files is prohibited and checked in CI.

## Verification Receipts

- For `DEFAULT_SEED = 20261117` and `DEFAULT_N_QUESTIONS = 240`:
  - The canonical `corpus_passages.jsonl` SHA-256 prefix is `a94a7309491e` (`696` passages).
  - The canonical `qa_oracle_fixture.jsonl` SHA-256 prefix is `210376de6247` (`240` QA pairs: `120` `single_passage`, `48` `split_evidence`, `48` `distractor_trap`, `24` `unanswerable`).
- `test_m1_regenerating_fixture_from_same_seed_is_byte_identical` and `make verify` assert byte-for-byte identity on every run.
