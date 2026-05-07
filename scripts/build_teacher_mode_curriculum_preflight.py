#!/usr/bin/env python3
"""Mix FineWeb-Edu, teacher-query, and GALT memory data into curriculum splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINEWEB = REPO_ROOT / "data" / "galt_teacher_mode" / "fineweb_edu_stage_a_sample_10k.jsonl"
DEFAULT_TEACHER = REPO_ROOT / "data" / "galt_teacher_mode" / "teacher_query_diverse_preflight.jsonl"
DEFAULT_GALT_PREFIX = "smallprod_20260501_070636_a2871d"
DEFAULT_CONTRAST = REPO_ROOT / "data" / "galt_contrast_lessons" / "accepted" / "glm_contrast_course_24x2_20260507_1643_accepted.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "galt_teacher_mode" / "curriculum_preflight"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def as_stage_a(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"curriculum/stage_a/{index:06d}",
        "schema_version": "galt_teacher_curriculum_v0",
        "stage": "stage_a_base_language",
        "stream": "fineweb_edu_lm",
        "text": row["text"],
        "source_id": row.get("source_id", ""),
        "source_dataset": row.get("source_dataset", "HuggingFaceFW/fineweb-edu"),
        "targets": row.get("targets", {}),
        "metadata": row.get("metadata", {}),
    }


def as_teacher(row: dict[str, Any], index: int) -> dict[str, Any]:
    out = dict(row)
    out["id"] = f"curriculum/stage_06_teacher_query/{index:06d}"
    out["schema_version"] = "galt_teacher_curriculum_v0"
    out["stream"] = "teacher_query_routing"
    return out


def as_galt(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"curriculum/{row['stage']}/{index:06d}",
        "schema_version": "galt_teacher_curriculum_v0",
        "stage": row["stage"],
        "stream": "galt_adjudicated_memory",
        "category": row["category"],
        "safety_layer": row["safety_layer"],
        "context": row["context"],
        "candidate_transition": row["candidate_transition"],
        "query": row["query"],
        "answer_target": row["answer_target"],
        "targets": {
            "gate_target": row["gate_target"],
            "action_target": row["action_target"],
            "memory_target": row["memory_target"],
            "residual_target": row["residual_target"],
        },
        "metadata": {
            "source_id": row["id"],
            "trace": row.get("trace", {}),
        },
    }


def safety_layer_for_packet(row: dict[str, Any], case: str) -> str:
    if case == "constraint":
        return "rule" if row.get("domain") in ("workflow", "safety_boundary") else "social"
    if row.get("domain") == "preference":
        return "social"
    return "rule"


def memory_layer_id_for_domain(domain: str) -> int:
    if domain == "safety_boundary":
        return 0
    if domain in ("tutoring", "workflow"):
        return 2
    if domain == "preference":
        return 3
    return 2


def violation_layer_id_for_case(domain: str, case: str) -> int:
    if case == "positive":
        return 4
    if case == "no_write":
        return memory_layer_id_for_domain(domain)
    if domain == "safety_boundary":
        return 0
    if domain in ("tutoring", "preference"):
        return 1
    if domain == "workflow":
        return 2
    return 1


def hierarchical_access_target(memory_layer_id: int, violation_layer_id: int) -> int:
    if violation_layer_id == 4:
        return 1
    return int(violation_layer_id > memory_layer_id)


def as_contrast_galt(row: dict[str, Any], index: int, case: str) -> dict[str, Any]:
    source_id = str(row["id"])
    concept = str(row["concept"])
    domain = str(row["domain"])
    base_context = (
        f"Concept: {concept}. Domain: {domain}. "
        f"Write condition: {row['write_condition']} "
        f"No-write condition: {row['no_write_condition']} "
        f"Constraint risk: {row['constraint_risk']}"
    )
    if case == "positive":
        stage = "stage_04_adjudicated_memory"
        category = f"contrast_positive_write_{domain}"
        candidate = row["positive_write_case"]
        answer = f"Accept the durable trace only under the stated write condition: {row['write_condition']}"
        targets = {
            "gate_target": [1.0, 1.0, 1.0],
            "action_target": "accept",
            "memory_target": "accepted_trace",
            "residual_target": "none",
        }
    elif case == "no_write":
        stage = "stage_03_safety_layers"
        category = f"contrast_no_write_{domain}"
        candidate = row["no_write_contrast"]
        answer = f"Reject persistent writing and remain silent because: {row['no_write_condition']}"
        targets = {
            "gate_target": [1.0, 0.0, 0.0],
            "action_target": "reject",
            "memory_target": "not_consolidated",
            "residual_target": "none",
        }
    elif case == "constraint":
        stage = "stage_03_safety_layers"
        category = f"contrast_constraint_{domain}"
        candidate = row["constraint_contrast"]
        answer = f"Reject consolidation because the constraint risk is: {row['constraint_risk']}"
        targets = {
            "gate_target": [1.0, 0.0, 0.0],
            "action_target": "reject",
            "memory_target": "not_consolidated",
            "residual_target": "hard_violation" if domain == "safety_boundary" else "boundary_ambiguity",
        }
    else:
        raise ValueError(f"unknown contrast case: {case}")
    memory_layer_id = memory_layer_id_for_domain(domain)
    violation_layer_id = violation_layer_id_for_case(domain, case)
    access_target = hierarchical_access_target(memory_layer_id, violation_layer_id)
    return {
        "id": f"curriculum/{stage}/contrast_lesson/{index:06d}_{case}",
        "schema_version": "galt_teacher_curriculum_v0",
        "stage": stage,
        "stream": "galt_adjudicated_memory",
        "category": category,
        "safety_layer": safety_layer_for_packet(row, case),
        "context": base_context,
        "candidate_transition": candidate,
        "query": row["retain_probe"],
        "answer_target": answer,
        "targets": targets,
        "metadata": {
            "source_id": source_id,
            "source_schema": row.get("schema_version", ""),
            "source_run": source_id.split("/")[1] if "/" in source_id else "",
            "case": case,
            "concept": concept,
            "domain": domain,
            "expected_site": row.get("expected_site", ""),
            "memory_layer_id": memory_layer_id,
            "violation_layer_id": violation_layer_id,
            "hierarchical_access_target": access_target,
            "write_condition": row.get("write_condition", ""),
            "no_write_condition": row.get("no_write_condition", ""),
            "constraint_risk": row.get("constraint_risk", ""),
            "trace": row.get("trace", ""),
            "judge": row.get("judge", {}),
        },
    }


def validate(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("id", "schema_version", "stage", "stream"):
        if not row.get(key):
            errors.append(f"missing:{key}")
    stream = row.get("stream")
    if stream == "fineweb_edu_lm" and not row.get("text"):
        errors.append("missing_text")
    if stream == "teacher_query_routing":
        for key in ("question", "knowledge_state", "action", "final_answer"):
            if not row.get(key):
                errors.append(f"missing:{key}")
    if stream == "galt_adjudicated_memory":
        for key in ("context", "candidate_transition", "query", "answer_target", "targets"):
            if not row.get(key):
                errors.append(f"missing:{key}")
    return errors


def split_group_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    source_id = metadata.get("source_id") or row.get("source_id") or row.get("id")
    if str(source_id).startswith("contrast_lesson/"):
        return "contrast:" + str(source_id)
    return str(row.get("stream", "")) + ":" + str(source_id)


def split_group_stratum(group: list[dict[str, Any]]) -> str:
    first = group[0]
    stream = str(first.get("stream", ""))
    if stream != "galt_adjudicated_memory":
        return stream
    residuals = sorted({str(row.get("targets", {}).get("residual_target", "none")) for row in group})
    metadata = first.get("metadata", {})
    if metadata.get("source_schema") == "galt_contrast_lesson_packet_v0":
        if "hard_violation" in residuals:
            return "galt_contrast:hard_violation"
        if "boundary_ambiguity" in residuals:
            return "galt_contrast:boundary_ambiguity"
        return "galt_contrast:none"
    return "galt_legacy:" + "+".join(residuals)


def split_stratum_groups(
    groups: list[list[dict[str, Any]]],
    rng: random.Random,
    train_frac: float,
    val_frac: float,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    groups = list(groups)
    rng.shuffle(groups)
    n = len(groups)
    if n <= 2:
        return groups, [], []
    test_frac = max(0.0, 1.0 - train_frac - val_frac)
    val_count = int(round(n * val_frac))
    test_count = int(round(n * test_frac))
    if val_count == 0:
        val_count = 1
    if test_count == 0:
        test_count = 1
    if val_count + test_count >= n:
        val_count = 1
        test_count = 1
    train_count = n - val_count - test_count
    return groups[:train_count], groups[train_count : train_count + val_count], groups[train_count + val_count :]


def grouped_split(
    rows: list[dict[str, Any]],
    rng: random.Random,
    train_frac: float,
    val_frac: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[split_group_key(row)].append(row)
    strata: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in grouped.values():
        strata[split_group_stratum(group)].append(group)
    train: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for groups in strata.values():
        train_groups, val_groups, test_groups = split_stratum_groups(groups, rng, train_frac, val_frac)
        for group in train_groups:
            train.extend(group)
        for group in val_groups:
            validation_rows.extend(group)
        for group in test_groups:
            test.extend(group)
    rng.shuffle(train)
    rng.shuffle(validation_rows)
    rng.shuffle(test)
    return train, validation_rows, test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb", type=Path, default=DEFAULT_FINEWEB)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--galt-prefix", default=DEFAULT_GALT_PREFIX)
    parser.add_argument("--contrast-lessons", type=Path, nargs="+", default=[DEFAULT_CONTRAST])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fineweb-limit", type=int, default=4000)
    parser.add_argument("--teacher-limit", type=int, default=1000)
    parser.add_argument("--galt-limit", type=int, default=0, help="0 means use all available accepted GALT rows for the prefix.")
    parser.add_argument("--contrast-limit", type=int, default=0, help="0 means use all accepted contrast lesson packets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    fineweb = read_jsonl(args.fineweb)
    rng.shuffle(fineweb)
    fineweb = fineweb[: args.fineweb_limit]

    teacher = read_jsonl(args.teacher)
    rng.shuffle(teacher)
    teacher = teacher[: args.teacher_limit]

    galt_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        path = REPO_ROOT / "data" / "galt_nl_corpus" / "accepted" / f"{args.galt_prefix}_{split}.jsonl"
        if path.exists():
            galt_rows.extend(read_jsonl(path))
    rng.shuffle(galt_rows)
    if args.galt_limit > 0:
        galt_rows = galt_rows[: args.galt_limit]

    contrast_rows: list[dict[str, Any]] = []
    contrast_packets: list[dict[str, Any]] = []
    for contrast_path in args.contrast_lessons:
        if contrast_path.exists():
            contrast_packets.extend(read_jsonl(contrast_path))
    rng.shuffle(contrast_packets)
    if args.contrast_limit > 0:
        contrast_packets = contrast_packets[: args.contrast_limit]
    for packet_idx, packet in enumerate(contrast_packets):
        for case in ("positive", "no_write", "constraint"):
            contrast_rows.append(as_contrast_galt(packet, packet_idx, case))

    rows: list[dict[str, Any]] = []
    rows.extend(as_stage_a(row, i) for i, row in enumerate(fineweb))
    rows.extend(as_teacher(row, i) for i, row in enumerate(teacher))
    rows.extend(as_galt(row, i) for i, row in enumerate(galt_rows))
    rows.extend(contrast_rows)
    rng.shuffle(rows)

    validation = Counter()
    for row in rows:
        validation.update(validate(row) or ["ok"])

    train, validation_rows, test = grouped_split(rows, rng, args.train_frac, args.val_frac)

    output_dir = args.output_dir.resolve()
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation_rows)
    write_jsonl(output_dir / "test.jsonl", test)

    counts = {
        "stage": Counter(row["stage"] for row in rows),
        "stream": Counter(row["stream"] for row in rows),
    }
    by_split = {
        "train": Counter(row["stream"] for row in train),
        "validation": Counter(row["stream"] for row in validation_rows),
        "test": Counter(row["stream"] for row in test),
    }
    manifest = {
        "timestamp": utc_now(),
        "output_dir": str(output_dir),
        "total_rows": len(rows),
        "splits": {"train": len(train), "validation": len(validation_rows), "test": len(test)},
        "validation": dict(validation),
        "counts": {key: dict(value) for key, value in counts.items()},
        "by_split_stream": {key: dict(value) for key, value in by_split.items()},
        "sources": {
            "fineweb": str(args.fineweb),
            "teacher": str(args.teacher),
            "galt_prefix": args.galt_prefix,
            "contrast_lessons": [str(path) for path in args.contrast_lessons],
        },
        "limits": {
            "fineweb": args.fineweb_limit,
            "teacher": args.teacher_limit,
            "galt": args.galt_limit,
            "contrast": args.contrast_limit,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if validation.get("ok") == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
