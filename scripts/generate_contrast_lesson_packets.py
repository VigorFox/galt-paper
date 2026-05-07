#!/usr/bin/env python3
"""Generate contrast-clean GALT lesson packets with local GLM/Ling servers.

This side quest produces natural-language lesson packets for the next
positive-rich AND no-write-clean data line.  GLM is used as planner/judge and
can also be used for production variants; Ling remains an optional fast variant
backend once its structured-output behavior is reliable.  A dry-run mode emits
deterministic packets so the schema and downstream files can be validated
without loading models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "galt_contrast_lessons"
SCHEMA_VERSION = "galt_contrast_lesson_packet_v0"

REQUIRED_PACKET_KEYS = {
    "concept",
    "domain",
    "positive_write_case",
    "no_write_contrast",
    "constraint_contrast",
    "retain_probe",
    "expected_site",
    "write_condition",
    "no_write_condition",
    "constraint_risk",
    "adjudication_expectation",
}


@dataclass
class GenerationConfig:
    run_id: str
    count: int
    planner_batch_size: int
    variants_per_blueprint: int
    domain_focus: str
    dry_run: bool
    variant_backend: str
    glm_url: str
    ling_url: str
    glm_model: str
    ling_model: str
    max_tokens: int
    judge_max_tokens: int


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start_obj = stripped.find("{")
        start_arr = stripped.find("[")
        starts = [idx for idx in (start_obj, start_arr) if idx >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def chat_completion(
    url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    message = body["choices"][0]["message"]
    if "content" not in message:
        raise ValueError(f"completion message has no content field; keys={sorted(message)}")
    return str(message["content"])


def planner_prompt(count: int, domain_focus: str) -> str:
    domain_rule = "Use a balanced mix of tutoring, workflow, preference, and safety_boundary domains."
    if domain_focus != "any":
        domain_rule = f"Every object must use domain: {domain_focus}."
    safety_rule = ""
    if domain_focus == "safety_boundary":
        safety_rule = """
- Make the constraint_contrast a hard admissibility violation, not merely a weak preference.
- Use benign synthetic safety boundaries such as consent, public exposure, irreversible action, unsafe tool use, or unsupported authority.
- The constraint_risk must describe why durable consolidation would be unsafe or inadmissible.
"""
    return f"""You are GLM acting as a GALT curriculum architect.
Generate exactly {count} JSON objects as a JSON array. Return JSON only.

Each object is a contrast lesson packet blueprint for a tutoring or workflow memory system.
Required keys:
- concept: short concept name
- domain: tutoring | workflow | preference | safety_boundary
- positive_write_case: concrete situation where a durable trace should be written
- no_write_contrast: nearby situation where the system must remain silent / not write
- constraint_contrast: nearby situation where writing would violate a rule, consent boundary, or safety margin
- retain_probe: future query that tests whether the accepted write is useful without overgeneralizing
- expected_site: output_head | state_bias | teacher | no_write
- write_condition: exact condition that permits persistent writing
- no_write_condition: exact condition that forbids persistent writing
- constraint_risk: what would go wrong if this write generalized too broadly
- adjudication_expectation: object with booleans positive_should_improve, no_write_should_remain_silent, constraint_should_not_degrade

Rules:
- {domain_rule}
- Make the three cases tightly related but semantically distinct.
- Keep all content benign and synthetic.
- Avoid private personal data, medical facts, finance facts, credentials, or real names.
- The positive case must be useful enough to remember.
- The no-write contrast must look tempting but should not be remembered.
- The constraint contrast must make the boundary explicit.
- Because every packet has a positive accepted-write case, expected_site must be output_head, state_bias, or teacher; do not use no_write for expected_site.
{safety_rule}
"""


def ling_variant_prompt(blueprint: dict[str, Any], variant_count: int) -> str:
    return f"""You are Ling generating compact GALT contrast lesson packets from one blueprint.
Return exactly {variant_count} JSON objects as a JSON array. Return JSON only.

Blueprint:
{json.dumps(blueprint, ensure_ascii=False, sort_keys=True)}

For each variant, keep the same schema keys as the blueprint and add:
- wording_variant: short id
- trace: one sentence explaining why the positive write is accepted and contrasts are not

Do not change the adjudication logic. Use benign synthetic tutoring/workflow scenarios only.
"""


def judge_prompt(packet: dict[str, Any]) -> str:
    return f"""You are GLM judging a GALT contrast lesson packet.
Return one JSON object only with keys:
- schema_valid: boolean
- contrast_clean: boolean
- positive_useful: boolean
- no_write_clean: boolean
- constraint_clean: boolean
- score: integer 0..5
- rationale: short string

Judge whether the positive case, no-write contrast, and constraint contrast are semantically distinct and safe.
Packet:
{json.dumps(packet, ensure_ascii=False, sort_keys=True)}
"""


def dry_blueprints(count: int) -> list[dict[str, Any]]:
    seeds = [
        {
            "concept": "mastery trace after verified quiz",
            "domain": "tutoring",
            "positive_write_case": "A learner completes a fractions quiz with every equivalent-fraction item correct after two prior practice sessions.",
            "no_write_contrast": "A learner says fractions feel easier today but has not completed any checkable exercise.",
            "constraint_contrast": "A classmate reports the learner is bad at fractions and asks the tutor to store that judgment.",
            "retain_probe": "Next week, choose whether to start with equivalent fractions review or move to ratio problems.",
            "expected_site": "state_bias",
            "write_condition": "Store only verified non-sensitive learning progress supported by the learner's own completed work.",
            "no_write_condition": "Do not store vague self-reports as mastery without evidence.",
            "constraint_risk": "Overgeneralizing would store unsupported or third-party claims as durable learner state.",
            "adjudication_expectation": {
                "positive_should_improve": True,
                "no_write_should_remain_silent": True,
                "constraint_should_not_degrade": True,
            },
        },
        {
            "concept": "explicit interface preference",
            "domain": "preference",
            "positive_write_case": "A learner explicitly asks the tutor to use larger text in future lessons and confirms it should be saved.",
            "no_write_contrast": "A learner zooms the page once during a lesson but does not mention saving a preference.",
            "constraint_contrast": "A sibling says the learner needs larger text and asks the tutor to save it without the learner present.",
            "retain_probe": "At the next session, decide whether to apply larger lesson text automatically.",
            "expected_site": "state_bias",
            "write_condition": "Store an explicit user-owned preference with clear consent.",
            "no_write_condition": "Do not store inferred one-off behavior as a durable preference.",
            "constraint_risk": "Saving third-party or inferred accessibility preferences can violate consent boundaries.",
            "adjudication_expectation": {
                "positive_should_improve": True,
                "no_write_should_remain_silent": True,
                "constraint_should_not_degrade": True,
            },
        },
    ]
    out = []
    for idx in range(count):
        row = dict(seeds[idx % len(seeds)])
        row["concept"] = f"{row['concept']} #{idx // len(seeds) + 1}"
        out.append(row)
    return out


def validate_packet(packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_PACKET_KEYS - set(packet)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    expectation = packet.get("adjudication_expectation")
    if not isinstance(expectation, dict):
        errors.append("adjudication_expectation must be an object")
    else:
        for key in ("positive_should_improve", "no_write_should_remain_silent", "constraint_should_not_degrade"):
            if expectation.get(key) is not True:
                errors.append(f"adjudication_expectation.{key} must be true")
    for key in REQUIRED_PACKET_KEYS - {"adjudication_expectation"}:
        if not isinstance(packet.get(key), str) or not str(packet[key]).strip():
            errors.append(f"{key} must be a non-empty string")
    if packet.get("expected_site") not in {"output_head", "state_bias", "teacher", "no_write"}:
        errors.append(f"unexpected expected_site: {packet.get('expected_site')!r}")
    if packet.get("expected_site") == "no_write":
        errors.append("positive accepted-write packets must not use expected_site=no_write")
    return errors


def heuristic_score(packet: dict[str, Any]) -> dict[str, Any]:
    errors = validate_packet(packet)
    text_fields = " ".join(str(packet.get(key, "")) for key in REQUIRED_PACKET_KEYS if key != "adjudication_expectation")
    contrast_clean = (
        "not" in str(packet.get("no_write_condition", "")).lower()
        or "do not" in text_fields.lower()
        or "without" in text_fields.lower()
    )
    positive_useful = len(str(packet.get("positive_write_case", ""))) >= 40
    no_write_clean = len(str(packet.get("no_write_contrast", ""))) >= 35
    constraint_clean = len(str(packet.get("constraint_contrast", ""))) >= 35
    score = 5 if not errors and contrast_clean and positive_useful and no_write_clean and constraint_clean else 3
    if errors:
        score = 1
    return {
        "schema_valid": not errors,
        "contrast_clean": bool(contrast_clean),
        "positive_useful": bool(positive_useful),
        "no_write_clean": bool(no_write_clean),
        "constraint_clean": bool(constraint_clean),
        "score": score,
        "rationale": "; ".join(errors) if errors else "heuristic schema and contrast check passed",
    }


def normalize_packet(packet: dict[str, Any], run_id: str, index: int, source: str) -> dict[str, Any]:
    row = dict(packet)
    row["id"] = f"contrast_lesson/{run_id}/{index:05d}"
    row["schema_version"] = SCHEMA_VERSION
    row["source"] = source
    return row


def generate_blueprints(args: argparse.Namespace, cfg: GenerationConfig) -> list[dict[str, Any]]:
    if cfg.dry_run:
        return dry_blueprints(cfg.count)
    rows: list[dict[str, Any]] = []
    remaining = cfg.count
    batch_size = max(1, cfg.planner_batch_size)
    while remaining > 0:
        current_count = min(batch_size, remaining)
        text = chat_completion(
            cfg.glm_url,
            cfg.glm_model,
            planner_prompt(current_count, cfg.domain_focus),
            max_tokens=cfg.max_tokens,
            temperature=0.25,
            timeout=args.timeout,
        )
        value = extract_json(text)
        if not isinstance(value, list):
            raise ValueError("GLM planner must return a JSON array")
        rows.extend(row for row in value if isinstance(row, dict))
        remaining -= current_count
    return rows[: cfg.count]


def fallback_variant(blueprint: dict[str, Any], variant_id: str, reason: str) -> dict[str, Any]:
    row = dict(blueprint)
    row["wording_variant"] = variant_id
    row["trace"] = "Fallback variant preserves the blueprint contrast packet after generator parsing failed."
    row["generator_error"] = reason
    return row


def generate_variants(args: argparse.Namespace, cfg: GenerationConfig, blueprints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    if cfg.dry_run or cfg.variant_backend == "blueprint":
        for blueprint in blueprints:
            row = dict(blueprint)
            row["wording_variant"] = "dry_0" if cfg.dry_run else "blueprint_0"
            row["trace"] = "The positive case has explicit evidence; the contrast cases remain unaccepted."
            variants.append(row)
        return variants
    variant_url = cfg.ling_url if cfg.variant_backend == "ling" else cfg.glm_url
    variant_model = cfg.ling_model if cfg.variant_backend == "ling" else cfg.glm_model
    temperature = 0.35 if cfg.variant_backend == "ling" else 0.25
    for blueprint in blueprints:
        try:
            text = chat_completion(
                variant_url,
                variant_model,
                ling_variant_prompt(blueprint, cfg.variants_per_blueprint),
                max_tokens=cfg.max_tokens,
                temperature=temperature,
                timeout=args.timeout,
            )
            value = extract_json(text)
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            variants.append(fallback_variant(blueprint, f"{cfg.variant_backend}_parse_failed", f"{type(exc).__name__}: {exc}"))
            continue
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            variants.append(fallback_variant(blueprint, f"{cfg.variant_backend}_non_list", "variant output was not a JSON list"))
            continue
        rows = []
        for row in value:
            if isinstance(row, dict):
                merged = dict(blueprint)
                merged.update(row)
                rows.append(merged)
        if rows:
            variants.extend(rows)
        else:
            variants.append(fallback_variant(blueprint, f"{cfg.variant_backend}_empty", "variant list contained no objects"))
    return variants


def judge_packets(args: argparse.Namespace, cfg: GenerationConfig, packets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for packet in packets:
        if cfg.dry_run:
            score = heuristic_score(packet)
        else:
            try:
                text = chat_completion(
                    cfg.glm_url,
                    cfg.glm_model,
                    judge_prompt(packet),
                    max_tokens=cfg.judge_max_tokens,
                    temperature=0.0,
                    timeout=args.timeout,
                )
                score = extract_json(text)
                if not isinstance(score, dict):
                    raise ValueError("judge output is not object")
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                score = heuristic_score(packet)
                score["rationale"] = f"judge fallback: {type(exc).__name__}: {score['rationale']}"
        row = dict(packet)
        row["judge"] = score
        local_errors = validate_packet(row)
        if local_errors:
            row["local_validation_errors"] = local_errors
            row["judge"] = dict(score)
            row["judge"]["schema_valid"] = False
            row["judge"]["rationale"] = f"local validation failed: {'; '.join(local_errors)}"
        if bool(score.get("schema_valid")) and bool(score.get("contrast_clean")) and int(score.get("score", 0)) >= 4:
            if local_errors:
                rejected.append(row)
            else:
                accepted.append(row)
        else:
            rejected.append(row)
    return accepted, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--planner-batch-size", type=int, default=4)
    parser.add_argument("--variants-per-blueprint", type=int, default=1)
    parser.add_argument("--domain-focus", choices=("any", "tutoring", "workflow", "preference", "safety_boundary"), default="any")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant-backend", choices=("ling", "glm", "blueprint"), default="ling")
    parser.add_argument("--glm-url", default="http://127.0.0.1:2323/v1/chat/completions")
    parser.add_argument("--ling-url", default="http://127.0.0.1:2424/v1/chat/completions")
    parser.add_argument("--glm-model", default="glm-5.1")
    parser.add_argument("--ling-model", default="ling-2.6-flash")
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=240.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"contrast_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    cfg = GenerationConfig(
        run_id=run_id,
        count=args.count,
        planner_batch_size=args.planner_batch_size,
        variants_per_blueprint=args.variants_per_blueprint,
        domain_focus=args.domain_focus,
        dry_run=bool(args.dry_run),
        variant_backend=args.variant_backend,
        glm_url=args.glm_url,
        ling_url=args.ling_url,
        glm_model=args.glm_model,
        ling_model=args.ling_model,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
    )
    started = time.perf_counter()
    output_root = args.output_root.resolve()
    blueprints = generate_blueprints(args, cfg)
    blueprint_rows = [normalize_packet(row, run_id, idx, "glm_blueprint" if not cfg.dry_run else "dry_blueprint") for idx, row in enumerate(blueprints)]
    variants_raw = generate_variants(args, cfg, blueprints)
    packet_source = "dry_variant" if cfg.dry_run else f"{cfg.variant_backend}_variant"
    packets = [normalize_packet(row, run_id, idx, packet_source) for idx, row in enumerate(variants_raw)]
    accepted, rejected = judge_packets(args, cfg, packets)

    manifest = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "config": asdict(cfg),
        "counts": {
            "blueprints": len(blueprint_rows),
            "packets": len(packets),
            "accepted": len(accepted),
            "rejected": len(rejected),
        },
        "duration_sec": round(time.perf_counter() - started, 3),
        "paths": {
            "blueprints": str(output_root / "raw" / "glm" / f"{run_id}_blueprints.jsonl"),
            "packets": str(output_root / "raw" / cfg.variant_backend / f"{run_id}_packets.jsonl"),
            "accepted": str(output_root / "accepted" / f"{run_id}_accepted.jsonl"),
            "rejected": str(output_root / "rejected" / f"{run_id}_rejected.jsonl"),
        },
    }
    write_jsonl(output_root / "raw" / "glm" / f"{run_id}_blueprints.jsonl", blueprint_rows)
    write_jsonl(output_root / "raw" / cfg.variant_backend / f"{run_id}_packets.jsonl", packets)
    write_jsonl(output_root / "accepted" / f"{run_id}_accepted.jsonl", accepted)
    write_jsonl(output_root / "rejected" / f"{run_id}_rejected.jsonl", rejected)
    manifest_path = output_root / "manifests" / f"{run_id}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"blueprints={len(blueprint_rows)} packets={len(packets)} accepted={len(accepted)} rejected={len(rejected)}")
    print(f"manifest={manifest_path}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
