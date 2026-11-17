"""Deterministic oracle-labeled corpus and QA fixture generator (M1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

from groundedness_referee.schema import (
    ALL_CASE_TYPES,
    CASE_DISTRACTOR_TRAP,
    CASE_SINGLE_PASSAGE,
    CASE_SPLIT_EVIDENCE,
    CASE_UNANSWERABLE,
    Passage,
    QAPair,
)

DEFAULT_SEED = 20261117
DEFAULT_N_QUESTIONS = 240

SYSTEM_PREFIXES = (
    "Atlas",
    "Borealis",
    "Chronos",
    "Delta",
    "Echelon",
    "Falcon",
    "Glacier",
    "Helios",
    "Icarus",
    "Janus",
    "Kestrel",
    "Lyra",
    "Meridian",
    "Nexus",
    "Orion",
    "Pulsar",
    "Quasar",
    "Rubicon",
    "Stratos",
    "Titan",
    "Umbra",
    "Vanguard",
    "Vortex",
    "Zephyr",
)

SYSTEM_DOMAINS = (
    "Gateway",
    "Indexer",
    "Ledger",
    "Router",
    "Scheduler",
    "Vault",
    "Aggregator",
    "Replicator",
    "Broker",
    "Coordinator",
)

ATTRIBUTES = (
    {
        "attr": "failover_timeout",
        "label": "leader failover timeout",
        "unit": "milliseconds",
        "adj_attr": "heartbeat_interval",
        "adj_label": "cluster gossip heartbeat interval",
        "unplanted_attr": "cold_boot_recovery_ceiling",
        "unplanted_label": "cold-boot recovery ceiling",
    },
    {
        "attr": "replication_quorum",
        "label": "write replication quorum",
        "unit": "nodes",
        "adj_attr": "read_replica_pool",
        "adj_label": "read replica connection pool size",
        "unplanted_attr": "cross_ocean_witness_count",
        "unplanted_label": "cross-ocean witness node count",
    },
    {
        "attr": "token_ttl",
        "label": "session token expiry window",
        "unit": "minutes",
        "adj_attr": "refresh_grace_period",
        "adj_label": "refresh token rotation grace period",
        "unplanted_attr": "hardware_dongle_reauth_limit",
        "unplanted_label": "hardware dongle reauthentication limit",
    },
    {
        "attr": "batch_flush_limit",
        "label": "write-ahead batch flush ceiling",
        "unit": "kilobytes",
        "adj_attr": "compaction_segment_size",
        "adj_label": "background compaction segment target",
        "unplanted_attr": "archive_tape_block_alignment",
        "unplanted_label": "archive tape block alignment size",
    },
    {
        "attr": "circuit_breaker_threshold",
        "label": "downstream circuit breaker trip threshold",
        "unit": "percent",
        "adj_attr": "canary_traffic_allocation",
        "adj_label": "canary deployment traffic allocation",
        "unplanted_attr": "manual_override_cooldown_hours",
        "unplanted_label": "manual override cooldown window",
    },
    {
        "attr": "checkpoint_cadence",
        "label": "state snapshot checkpoint cadence",
        "unit": "seconds",
        "adj_attr": "wal_retention_horizon",
        "adj_label": "write-ahead log retention horizon",
        "unplanted_attr": "glacier_deep_freeze_delay",
        "unplanted_label": "deep-freeze archival delay",
    },
)


def _deterministic_rng(seed: int, salt: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}::{salt}".encode("utf-8")).digest()
    int_seed = int.from_bytes(digest[:8], "big")
    return random.Random(int_seed)


def generate_oracle_fixture(
    seed: int = DEFAULT_SEED,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> tuple[list[Passage], list[QAPair]]:
    """Generate deterministic corpus passages and oracle-labeled QA pairs.

    Case distribution across `n_questions`:
      - 50% single_passage
      - 20% split_evidence (awkward case 1: requires two passages together)
      - 20% distractor_trap (awkward case 2: lexical lookalike staging distractor)
      - 10% unanswerable (awkward case 3: attribute absent from corpus)
    """
    if n_questions < 20:
        raise ValueError("n_questions must be at least 20 to populate all case types.")

    passages: list[Passage] = []
    qa_pairs: list[QAPair] = []

    n_single = int(round(n_questions * 0.50))
    n_split = int(round(n_questions * 0.20))
    n_distractor = int(round(n_questions * 0.20))
    n_unanswerable = n_questions - (n_single + n_split + n_distractor)

    schedule: list[str] = (
        [CASE_SINGLE_PASSAGE] * n_single
        + [CASE_SPLIT_EVIDENCE] * n_split
        + [CASE_DISTRACTOR_TRAP] * n_distractor
        + [CASE_UNANSWERABLE] * n_unanswerable
    )

    master_rng = _deterministic_rng(seed, "schedule")
    master_rng.shuffle(schedule)

    for idx, case_type in enumerate(schedule):
        q_rng = _deterministic_rng(seed, f"qa_{idx:04d}")
        prefix = SYSTEM_PREFIXES[idx % len(SYSTEM_PREFIXES)]
        domain = SYSTEM_DOMAINS[(idx // len(SYSTEM_PREFIXES)) % len(SYSTEM_DOMAINS)]
        cluster_tag = f"v{(idx % 9) + 1}{(idx % 4) + 1}"
        system_name = f"{prefix}-{domain}-{cluster_tag}"
        spec = ATTRIBUTES[idx % len(ATTRIBUTES)]

        attr = spec["attr"]
        label = spec["label"]
        unit = spec["unit"]
        adj_attr = spec["adj_attr"]
        adj_label = spec["adj_label"]

        primary_num = q_rng.randint(14, 96) * 5
        secondary_num = q_rng.randint(2, 9) * 3
        distractor_num = primary_num + q_rng.choice((35, 65, 95, 125))
        fabricated_num = primary_num + q_rng.choice((210, 340, 470))
        prior_num = q_rng.choice((1000, 2500, 5000, 9999))
        adj_num = q_rng.randint(11, 49) * 7

        doc_id = f"doc_{idx:04d}"
        qid = f"qa_{idx:04d}"

        if case_type == CASE_SINGLE_PASSAGE:
            p_gold_id = f"p_{idx:04d}_gold"
            p_dist1_id = f"p_{idx:04d}_dist1"
            p_dist2_id = f"p_{idx:04d}_dist2"

            val_str = f"{primary_num} {unit}"
            gold_text = (
                f"Production specification for {system_name}: the {label} is "
                f"enforced at exactly {primary_num} {unit} across all active zones."
            )
            dist1_text = (
                f"Telemetry note for {system_name}: the {adj_label} operates at "
                f"{adj_num} {unit} during routine health sweeps."
            )
            dist2_text = (
                f"Operational runbook for {system_name}: operators monitor queue "
                f"depth every {adj_num + 12} seconds via standard dashboards."
            )

            passages.extend(
                [
                    Passage(
                        passage_id=p_gold_id,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=attr,
                        value=val_str,
                        text=gold_text,
                        is_distractor=False,
                        distractor_kind="none",
                    ),
                    Passage(
                        passage_id=p_dist1_id,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=adj_attr,
                        value=f"{adj_num} {unit}",
                        text=dist1_text,
                        is_distractor=True,
                        distractor_kind="adjacent_attribute",
                    ),
                    Passage(
                        passage_id=p_dist2_id,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute="monitoring_cadence",
                        value=f"{adj_num + 12} seconds",
                        text=dist2_text,
                        is_distractor=True,
                        distractor_kind="adjacent_attribute",
                    ),
                ]
            )

            question = f"What is the production {label} for {system_name}?"
            expected_answer = (
                f"For {system_name}, the production {label} is enforced at "
                f"exactly {primary_num} {unit} across all active zones."
            )
            partial_answer = expected_answer
            prior_myth_answer = (
                f"Legacy cloud defaults typically assign a generic ceiling of "
                f"{prior_num} units based on vendor folklore."
            )
            fabricated_answer = (
                f"For {system_name}, the production {label} is enforced at "
                f"exactly {fabricated_num} {unit} across all active zones."
            )
            distractor_misled_answer = (
                f"For {system_name}, the production {label} is set to "
                f"{distractor_num} {unit} under legacy defaults."
            )
            paraphrased_answer = (
                f"{system_name} mandates a {primary_num}-{unit[:-1]} threshold "
                f"governing its {attr.replace('_', ' ')} in live clusters."
            )
            negated_answer = (
                f"For {system_name}, the production {label} is never enforced at "
                f"{primary_num} {unit} across active zones."
            )

            qa_pairs.append(
                QAPair(
                    question_id=qid,
                    question=question,
                    case_type=case_type,
                    system_name=system_name,
                    attribute=attr,
                    gold_passage_ids=[p_gold_id],
                    distractor_passage_ids=[p_dist1_id, p_dist2_id],
                    gold_values=[str(primary_num), unit],
                    expected_answer=expected_answer,
                    partial_answer=partial_answer,
                    prior_myth_answer=prior_myth_answer,
                    fabricated_answer=fabricated_answer,
                    distractor_misled_answer=distractor_misled_answer,
                    paraphrased_answer=paraphrased_answer,
                    negated_answer=negated_answer,
                )
            )

        elif case_type == CASE_SPLIT_EVIDENCE:
            p_gold_a = f"p_{idx:04d}_gold_a"
            p_gold_b = f"p_{idx:04d}_gold_b"
            p_dist1_id = f"p_{idx:04d}_dist1"

            gold_a_text = (
                f"Primary tier policy for {system_name}: the baseline {label} "
                f"starts at {primary_num} {unit} in regional hubs."
            )
            gold_b_text = (
                f"Failover addendum for {system_name}: multi-region mode applies "
                f"a strict surcharge factor of {secondary_num}x on top of baseline."
            )
            dist1_text = (
                f"Telemetry note for {system_name}: routine {adj_label} stays at "
                f"{adj_num} {unit} during calm periods."
            )

            passages.extend(
                [
                    Passage(
                        passage_id=p_gold_a,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=f"{attr}_base",
                        value=f"{primary_num} {unit}",
                        text=gold_a_text,
                        is_distractor=False,
                        distractor_kind="none",
                    ),
                    Passage(
                        passage_id=p_gold_b,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=f"{attr}_multiplier",
                        value=f"{secondary_num}x",
                        text=gold_b_text,
                        is_distractor=False,
                        distractor_kind="none",
                    ),
                    Passage(
                        passage_id=p_dist1_id,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=adj_attr,
                        value=f"{adj_num} {unit}",
                        text=dist1_text,
                        is_distractor=True,
                        distractor_kind="adjacent_attribute",
                    ),
                ]
            )

            question = (
                f"What are both the baseline {label} and the multi-region "
                f"surcharge factor for {system_name}?"
            )
            expected_answer = (
                f"For {system_name}, the baseline {label} starts at "
                f"{primary_num} {unit} in regional hubs and multi-region mode "
                f"applies a strict surcharge factor of {secondary_num}x."
            )
            partial_answer = (
                f"For {system_name}, the baseline {label} starts at "
                f"{primary_num} {unit}, while the multi-region factor defaults "
                f"to an unverified 99x legacy heuristic."
            )
            prior_myth_answer = (
                f"Standard distributed conventions assume a flat {prior_num} "
                f"quota with no regional multiplier."
            )
            fabricated_answer = (
                f"For {system_name}, the baseline {label} starts at "
                f"{fabricated_num} {unit} in regional hubs and multi-region mode "
                f"applies a strict surcharge factor of 88x."
            )
            distractor_misled_answer = (
                f"For {system_name}, the baseline {label} is {distractor_num} "
                f"{unit} with a 19x staging multiplier."
            )
            paraphrased_answer = (
                f"{system_name} combines a {primary_num} {unit} regional base "
                f"with a {secondary_num}x cross-region multiplier."
            )
            negated_answer = (
                f"For {system_name}, the baseline {label} never starts at "
                f"{primary_num} {unit} and excludes the {secondary_num}x factor."
            )

            qa_pairs.append(
                QAPair(
                    question_id=qid,
                    question=question,
                    case_type=case_type,
                    system_name=system_name,
                    attribute=attr,
                    gold_passage_ids=[p_gold_a, p_gold_b],
                    distractor_passage_ids=[p_dist1_id],
                    gold_values=[str(primary_num), unit, f"{secondary_num}x"],
                    expected_answer=expected_answer,
                    partial_answer=partial_answer,
                    prior_myth_answer=prior_myth_answer,
                    fabricated_answer=fabricated_answer,
                    distractor_misled_answer=distractor_misled_answer,
                    paraphrased_answer=paraphrased_answer,
                    negated_answer=negated_answer,
                )
            )

        elif case_type == CASE_DISTRACTOR_TRAP:
            p_gold_id = f"p_{idx:04d}_gold"
            p_trap_id = f"p_{idx:04d}_trap"
            p_dist_id = f"p_{idx:04d}_dist"

            gold_text = (
                f"Production specification for {system_name}: the live {label} "
                f"is locked at {primary_num} {unit} in verified production."
            )
            trap_text = (
                f"Deprecated staging draft for {system_name}: an obsolete sandbox "
                f"proposal tested {distractor_num} {unit} before rejection."
            )
            dist_text = (
                f"Telemetry note for {system_name}: the {adj_label} runs at "
                f"{adj_num} {unit} in background sweeps."
            )

            passages.extend(
                [
                    Passage(
                        passage_id=p_gold_id,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=attr,
                        value=f"{primary_num} {unit}",
                        text=gold_text,
                        is_distractor=False,
                        distractor_kind="none",
                    ),
                    Passage(
                        passage_id=p_trap_id,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=attr,
                        value=f"{distractor_num} {unit}",
                        text=trap_text,
                        is_distractor=True,
                        distractor_kind="lexical_lookalike_trap",
                    ),
                    Passage(
                        passage_id=p_dist_id,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=adj_attr,
                        value=f"{adj_num} {unit}",
                        text=dist_text,
                        is_distractor=True,
                        distractor_kind="adjacent_attribute",
                    ),
                ]
            )

            question = f"What is the verified live production {label} for {system_name}?"
            expected_answer = (
                f"For {system_name}, the live {label} is locked at "
                f"{primary_num} {unit} in verified production."
            )
            partial_answer = expected_answer
            prior_myth_answer = (
                f"Legacy cloud defaults typically assign a generic ceiling of "
                f"{prior_num} units based on vendor folklore."
            )
            fabricated_answer = (
                f"For {system_name}, the live {label} is locked at "
                f"{fabricated_num} {unit} in verified production."
            )
            distractor_misled_answer = (
                f"For {system_name}, the verified live production {label} is "
                f"locked at {distractor_num} {unit} across active production."
            )
            paraphrased_answer = (
                f"In live production, {system_name} fixes its {label} at "
                f"{primary_num} {unit}."
            )
            negated_answer = (
                f"For {system_name}, the live {label} is NOT locked at "
                f"{primary_num} {unit} in verified production."
            )

            qa_pairs.append(
                QAPair(
                    question_id=qid,
                    question=question,
                    case_type=case_type,
                    system_name=system_name,
                    attribute=attr,
                    gold_passage_ids=[p_gold_id],
                    distractor_passage_ids=[p_trap_id, p_dist_id],
                    gold_values=[str(primary_num), unit],
                    expected_answer=expected_answer,
                    partial_answer=partial_answer,
                    prior_myth_answer=prior_myth_answer,
                    fabricated_answer=fabricated_answer,
                    distractor_misled_answer=distractor_misled_answer,
                    paraphrased_answer=paraphrased_answer,
                    negated_answer=negated_answer,
                )
            )

        elif case_type == CASE_UNANSWERABLE:
            p_dist1_id = f"p_{idx:04d}_dist1"
            p_dist2_id = f"p_{idx:04d}_dist2"
            unplanted_attr = spec["unplanted_attr"]
            unplanted_label = spec["unplanted_label"]

            dist1_text = (
                f"Production specification for {system_name}: the {label} is "
                f"enforced at {primary_num} {unit} across active zones."
            )
            dist2_text = (
                f"Telemetry note for {system_name}: the {adj_label} operates at "
                f"{adj_num} {unit} during routine health sweeps."
            )

            passages.extend(
                [
                    Passage(
                        passage_id=p_dist1_id,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=attr,
                        value=f"{primary_num} {unit}",
                        text=dist1_text,
                        is_distractor=True,
                        distractor_kind="adjacent_attribute",
                    ),
                    Passage(
                        passage_id=p_dist2_id,
                        doc_id=doc_id,
                        system_name=system_name,
                        attribute=adj_attr,
                        value=f"{adj_num} {unit}",
                        text=dist2_text,
                        is_distractor=True,
                        distractor_kind="adjacent_attribute",
                    ),
                ]
            )

            question = f"What is the {unplanted_label} for {system_name}?"
            expected_answer = "ABSTAIN: insufficient evidence in context."
            partial_answer = expected_answer
            prior_myth_answer = (
                f"Legacy cloud defaults typically assign a {unplanted_label} of "
                f"{prior_num} hours based on vendor folklore."
            )
            fabricated_answer = (
                f"For {system_name}, the {unplanted_label} is enforced at "
                f"{fabricated_num} {unit} across active zones."
            )
            distractor_misled_answer = fabricated_answer
            paraphrased_answer = expected_answer
            negated_answer = fabricated_answer

            qa_pairs.append(
                QAPair(
                    question_id=qid,
                    question=question,
                    case_type=case_type,
                    system_name=system_name,
                    attribute=unplanted_attr,
                    gold_passage_ids=[],
                    distractor_passage_ids=[p_dist1_id, p_dist2_id],
                    gold_values=[],
                    expected_answer=expected_answer,
                    partial_answer=partial_answer,
                    prior_myth_answer=prior_myth_answer,
                    fabricated_answer=fabricated_answer,
                    distractor_misled_answer=distractor_misled_answer,
                    paraphrased_answer=paraphrased_answer,
                    negated_answer=negated_answer,
                )
            )
        else:
            raise ValueError(f"Unknown case_type: {case_type}")

    return passages, qa_pairs


def serialize_jsonl(items: Iterable[Passage | QAPair]) -> str:
    """Serialize fixture items to canonical newline-terminated JSONL."""
    lines = [
        json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=True)
        for item in items
    ]
    return "\n".join(lines) + "\n"


def write_fixture_files(
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    n_questions: int = DEFAULT_N_QUESTIONS,
) -> dict[str, str]:
    """Write corpus_passages.jsonl and qa_oracle_fixture.jsonl and return SHA-256 digests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    passages, qa_pairs = generate_oracle_fixture(seed=seed, n_questions=n_questions)

    passages_blob = serialize_jsonl(passages).encode("utf-8")
    qa_blob = serialize_jsonl(qa_pairs).encode("utf-8")

    passages_path = output_dir / "corpus_passages.jsonl"
    qa_path = output_dir / "qa_oracle_fixture.jsonl"

    passages_path.write_bytes(passages_blob)
    qa_path.write_bytes(qa_blob)

    return {
        "passages_sha256": hashlib.sha256(passages_blob).hexdigest(),
        "qa_sha256": hashlib.sha256(qa_blob).hexdigest(),
        "n_passages": str(len(passages)),
        "n_questions": str(len(qa_pairs)),
    }


def load_fixture_files(data_dir: Path) -> tuple[dict[str, Passage], list[QAPair]]:
    """Load passages dict and QA pairs list from JSONL files."""
    passages_path = data_dir / "corpus_passages.jsonl"
    qa_path = data_dir / "qa_oracle_fixture.jsonl"

    passages: dict[str, Passage] = {}
    for line in passages_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = Passage.from_dict(json.loads(line))
            passages[p.passage_id] = p

    qa_pairs: list[QAPair] = []
    for line in qa_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            qa_pairs.append(QAPair.from_dict(json.loads(line)))

    return passages, qa_pairs
