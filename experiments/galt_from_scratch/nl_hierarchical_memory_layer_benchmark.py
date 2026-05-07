#!/usr/bin/env python3
"""Natural-language benchmark for safety-indexed hierarchical memory.

This connects the hierarchical-memory toy to generated contrast lesson packets.
The flat learner sees only natural-language memory/violation text.  The gate
policies use the packet-derived layer ledger.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRAST = [
    REPO_ROOT / "data" / "galt_contrast_lessons" / "accepted" / "glm_contrast_course_24x2_20260507_1643_accepted.jsonl",
    REPO_ROOT / "data" / "galt_contrast_lessons" / "accepted" / "glm_safety_boundary_course_12x2_20260507_1811_accepted.jsonl",
]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "galt_prework" / "nl_hierarchical_memory_layer_benchmark.json"

LAYER_NAMES = ["l0_hard_safety", "l1_social_consent", "l2_task_rule", "l3_preference"]
NONE_VIOLATION = 4


@dataclass
class Config:
    contrast_lessons: tuple[str, ...] = tuple(str(path) for path in DEFAULT_CONTRAST)
    output: str = str(DEFAULT_OUTPUT)
    seed: int = 42
    vocab_size: int = 5000
    steps: int = 1200
    lr: float = 0.25
    l2: float = 1e-4
    holdout_pairs: tuple[str, ...] = ("2:0", "3:0", "3:1", "0:2")
    packet_holdout_frac: float = 0.25


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_']+|[^\sA-Za-z0-9_]", text.lower())


def memory_layer(packet: dict[str, Any]) -> int:
    domain = str(packet.get("domain", ""))
    if domain == "safety_boundary":
        return 0
    if domain in {"tutoring", "workflow"}:
        return 2
    if domain == "preference":
        return 3
    return 2


def violation_layer(packet: dict[str, Any]) -> int:
    domain = str(packet.get("domain", ""))
    if domain == "safety_boundary":
        return 0
    if domain in {"tutoring", "preference"}:
        return 1
    if domain == "workflow":
        return 2
    return 1


def target_access(memory_layer_id: int, violation_layer_id: int) -> int:
    if violation_layer_id == NONE_VIOLATION:
        return 1
    return int(violation_layer_id > memory_layer_id)


def memory_text(packet: dict[str, Any]) -> str:
    return " ".join(
        str(packet.get(key, ""))
        for key in (
            "concept",
            "positive_write_case",
            "write_condition",
            "retain_probe",
            "trace",
        )
    )


def violation_text(packet: dict[str, Any] | None) -> str:
    if packet is None:
        return "No active violation is present. The candidate memory is being checked in an ordinary future context."
    return " ".join(
        str(packet.get(key, ""))
        for key in (
            "concept",
            "constraint_contrast",
            "constraint_risk",
            "no_write_condition",
        )
    )


def row_text(row: dict[str, Any]) -> str:
    return "Memory candidate: " + str(row["memory_text"]) + "\nCurrent boundary signal: " + str(row["violation_text"])


def pair_key(memory_layer_id: int, violation_layer_id: int) -> str:
    return f"{memory_layer_id}:{violation_layer_id}"


def regime(memory_layer_id: int, violation_layer_id: int) -> str:
    if violation_layer_id == NONE_VIOLATION:
        return "no_violation_access"
    if violation_layer_id <= memory_layer_id:
        return "bottom_up_invalidation"
    return "top_down_containment"


def build_rows(packets: list[dict[str, Any]], cfg: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    holdouts = set(cfg.holdout_pairs)
    rows: list[dict[str, Any]] = []
    for memory_packet in packets:
        m_layer = memory_layer(memory_packet)
        # Include the no-violation access case once per memory packet.
        for violation_packet in [None, *packets]:
            v_layer = NONE_VIOLATION if violation_packet is None else violation_layer(violation_packet)
            key = pair_key(m_layer, v_layer)
            rows.append(
                {
                    "memory_id": memory_packet["id"],
                    "violation_id": "none" if violation_packet is None else violation_packet["id"],
                    "memory_domain": memory_packet.get("domain"),
                    "violation_domain": "none" if violation_packet is None else violation_packet.get("domain"),
                    "memory_layer": m_layer,
                    "violation_layer": v_layer,
                    "pair": key,
                    "heldout_pair": key in holdouts,
                    "regime": regime(m_layer, v_layer),
                    "memory_text": memory_text(memory_packet),
                    "violation_text": violation_text(violation_packet),
                    "target_access": target_access(m_layer, v_layer),
                }
            )
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(rows)
    train = [row for row in rows if not row["heldout_pair"]]
    test = rows
    return train, test


def split_packet_holdouts(packets: list[dict[str, Any]], cfg: Config) -> set[str]:
    rng = np.random.default_rng(cfg.seed + 2048)
    by_domain: dict[str, list[str]] = {}
    for packet in packets:
        by_domain.setdefault(str(packet.get("domain", "")), []).append(str(packet["id"]))
    holdouts: set[str] = set()
    for ids in by_domain.values():
        ids = sorted(ids)
        if len(ids) <= 1:
            continue
        rng.shuffle(ids)
        count = max(1, int(round(len(ids) * cfg.packet_holdout_frac)))
        holdouts.update(ids[:count])
    return holdouts


def build_vocab(rows: list[dict[str, Any]], cfg: Config) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts.update(tokenize(row_text(row)))
    vocab = {}
    for token, count in counts.most_common(cfg.vocab_size):
        vocab[token] = len(vocab)
    return vocab


def vectorize(rows: list[dict[str, Any]], vocab: dict[str, int]) -> np.ndarray:
    return vectorize_texts([row_text(row) for row in rows], vocab)


def vectorize_texts(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    x = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for i, text in enumerate(texts):
        for token in tokenize(text):
            idx = vocab.get(token)
            if idx is not None:
                x[i, idx] += 1.0
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0)
    return x / norms


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def train_logistic(x: np.ndarray, y: np.ndarray, cfg: Config, *, balanced: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    w = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float64)
    b = 0.0
    yf = y.astype(np.float64)
    weights = np.ones_like(yf)
    if balanced:
        counts = np.bincount(y, minlength=2).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = len(y) / (2.0 * counts[y])
        weights = weights / np.mean(weights)
    for _ in range(cfg.steps):
        p = sigmoid(x @ w + b)
        grad = (p - yf) * weights / max(len(y), 1)
        w -= cfg.lr * (x.T @ grad + cfg.l2 * w)
        b -= cfg.lr * float(np.sum(grad))
    return {"weight": w, "bias": np.array(b, dtype=np.float64)}


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def train_softmax(x: np.ndarray, y: np.ndarray, labels: list[int], cfg: Config, *, balanced: bool = True) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed + len(labels) * 31)
    mapping = {label: idx for idx, label in enumerate(labels)}
    yi = np.array([mapping[int(label)] for label in y], dtype=np.int64)
    w = rng.normal(0.0, 0.01, size=(x.shape[1], len(labels))).astype(np.float64)
    b = np.zeros(len(labels), dtype=np.float64)
    onehot = np.eye(len(labels), dtype=np.float64)[yi]
    weights = np.ones(len(y), dtype=np.float64)
    if balanced:
        counts = np.bincount(yi, minlength=len(labels)).astype(np.float64)
        counts = np.maximum(counts, 1.0)
        weights = len(y) / (len(labels) * counts[yi])
        weights = weights / np.mean(weights)
    for _ in range(cfg.steps):
        probs = softmax(x @ w + b[None, :])
        grad = (probs - onehot) * weights[:, None] / max(len(y), 1)
        w -= cfg.lr * (x.T @ grad + cfg.l2 * w)
        b -= cfg.lr * np.sum(grad, axis=0)
    return {"weight": w, "bias": b, "labels": labels}


def predict_flat(model: dict[str, np.ndarray], rows: list[dict[str, Any]], vocab: dict[str, int]) -> np.ndarray:
    p = sigmoid(vectorize(rows, vocab) @ model["weight"] + float(model["bias"]))
    return (p >= 0.5).astype(np.int64)


def predict_softmax(model: dict[str, Any], texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    logits = vectorize_texts(texts, vocab) @ model["weight"] + model["bias"][None, :]
    pred = np.argmax(logits, axis=1)
    labels = np.array(model["labels"], dtype=np.int64)
    return labels[pred]


def predict_independent(rows: list[dict[str, Any]]) -> np.ndarray:
    out = []
    for row in rows:
        v = int(row["violation_layer"])
        m = int(row["memory_layer"])
        out.append(int(v == NONE_VIOLATION or v != m))
    return np.array(out, dtype=np.int64)


def predict_global_flush(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([int(int(row["violation_layer"]) == NONE_VIOLATION) for row in rows], dtype=np.int64)


def predict_hierarchical(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([target_access(int(row["memory_layer"]), int(row["violation_layer"])) for row in rows], dtype=np.int64)


def unique_layer_examples(rows: list[dict[str, Any]], key: str, text_key: str, layer_key: str) -> tuple[list[str], np.ndarray]:
    seen: dict[str, tuple[str, int]] = {}
    for row in rows:
        item_id = str(row[key])
        seen[item_id] = (str(row[text_key]), int(row[layer_key]))
    texts = [value[0] for value in seen.values()]
    labels = np.array([value[1] for value in seen.values()], dtype=np.int64)
    return texts, labels


def filter_layer_training_rows(rows: list[dict[str, Any]], key: str, holdout_packet_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row[key]) == "none" or str(row[key]) not in holdout_packet_ids]


def predict_layer_gate(
    rows: list[dict[str, Any]],
    memory_model: dict[str, Any],
    violation_model: dict[str, Any],
    vocab: dict[str, int],
) -> tuple[np.ndarray, dict[str, float]]:
    memory_texts = [str(row["memory_text"]) for row in rows]
    violation_texts = [str(row["violation_text"]) for row in rows]
    pred_m = predict_softmax(memory_model, memory_texts, vocab)
    pred_v = predict_softmax(violation_model, violation_texts, vocab)
    pred = np.array([target_access(int(m), int(v)) for m, v in zip(pred_m, pred_v)], dtype=np.int64)
    true_m = np.array([int(row["memory_layer"]) for row in rows], dtype=np.int64)
    true_v = np.array([int(row["violation_layer"]) for row in rows], dtype=np.int64)
    memory_holdout = np.array([bool(row.get("memory_packet_holdout")) for row in rows], dtype=bool)
    violation_holdout = np.array([bool(row.get("violation_packet_holdout")) for row in rows], dtype=bool)
    return pred, {
        "memory_layer_accuracy": float(np.mean(pred_m == true_m)),
        "violation_layer_accuracy": float(np.mean(pred_v == true_v)),
        "heldout_memory_layer_accuracy": float(np.mean(pred_m[memory_holdout] == true_m[memory_holdout])) if np.any(memory_holdout) else math.nan,
        "heldout_violation_layer_accuracy": float(np.mean(pred_v[violation_holdout] == true_v[violation_holdout])) if np.any(violation_holdout) else math.nan,
    }


def precision(pred: np.ndarray, y: np.ndarray, positive: int) -> float:
    mask = pred == positive
    if not np.any(mask):
        return math.nan
    return float(np.mean(y[mask] == positive))


def recall(pred: np.ndarray, y: np.ndarray, positive: int) -> float:
    mask = y == positive
    if not np.any(mask):
        return math.nan
    return float(np.mean(pred[mask] == positive))


def summarize(name: str, pred: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.array([int(row["target_access"]) for row in rows], dtype=np.int64)
    out: dict[str, Any] = {
        "policy": name,
        "accuracy": float(np.mean(pred == y)),
        "access_precision": precision(pred, y, 1),
        "access_recall": recall(pred, y, 1),
        "quarantine_precision": precision(pred, y, 0),
        "quarantine_recall": recall(pred, y, 0),
    }
    for split_name, mask in {
        "heldout_pair": np.array([bool(row["heldout_pair"]) for row in rows], dtype=bool),
        "heldout_packet": np.array([bool(row.get("heldout_packet")) for row in rows], dtype=bool),
        "bottom_up_invalidation": np.array([row["regime"] == "bottom_up_invalidation" for row in rows], dtype=bool),
        "top_down_containment": np.array([row["regime"] == "top_down_containment" for row in rows], dtype=bool),
        "no_violation_access": np.array([row["regime"] == "no_violation_access" for row in rows], dtype=bool),
    }.items():
        out[f"{split_name}_accuracy"] = float(np.mean(pred[mask] == y[mask])) if np.any(mask) else math.nan
    out["pred_counts"] = dict(Counter("access" if item else "quarantine" for item in pred))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast-lessons", type=Path, nargs="+", default=[Path(path) for path in Config.contrast_lessons])
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--vocab-size", type=int, default=Config.vocab_size)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--packet-holdout-frac", type=float, default=Config.packet_holdout_frac)
    args = parser.parse_args()
    cfg = Config(
        contrast_lessons=tuple(str(path) for path in args.contrast_lessons),
        output=args.output,
        seed=args.seed,
        vocab_size=args.vocab_size,
        steps=args.steps,
        lr=args.lr,
        packet_holdout_frac=args.packet_holdout_frac,
    )

    packets: list[dict[str, Any]] = []
    for path in args.contrast_lessons:
        packets.extend(read_jsonl(path))
    holdout_packet_ids = split_packet_holdouts(packets, cfg)
    train, test = build_rows(packets, cfg)
    for row in train + test:
        row["memory_packet_holdout"] = str(row["memory_id"]) in holdout_packet_ids
        row["violation_packet_holdout"] = str(row["violation_id"]) in holdout_packet_ids
        row["heldout_packet"] = bool(row["memory_packet_holdout"] or row["violation_packet_holdout"])
    vocab = build_vocab(train, cfg)
    y_train = np.array([int(row["target_access"]) for row in train], dtype=np.int64)
    flat = train_logistic(vectorize(train, vocab), y_train, cfg)
    flat_balanced = train_logistic(vectorize(train, vocab), y_train, cfg, balanced=True)
    memory_texts, memory_labels = unique_layer_examples(
        filter_layer_training_rows(train, "memory_id", holdout_packet_ids),
        "memory_id",
        "memory_text",
        "memory_layer",
    )
    violation_texts, violation_labels = unique_layer_examples(
        filter_layer_training_rows(train, "violation_id", holdout_packet_ids),
        "violation_id",
        "violation_text",
        "violation_layer",
    )
    memory_layer_model = train_softmax(
        vectorize_texts(memory_texts, vocab),
        memory_labels,
        sorted(set(int(label) for label in memory_labels)),
        cfg,
    )
    violation_layer_model = train_softmax(
        vectorize_texts(violation_texts, vocab),
        violation_labels,
        sorted(set(int(label) for label in violation_labels)),
        cfg,
    )
    predicted_layer_gate, predicted_layer_metrics = predict_layer_gate(test, memory_layer_model, violation_layer_model, vocab)

    policies = {
        "flat_bow_logistic": predict_flat(flat, test, vocab),
        "flat_bow_balanced": predict_flat(flat_balanced, test, vocab),
        "predicted_hierarchical_gate": predicted_layer_gate,
        "independent_layer_gate": predict_independent(test),
        "global_flush_gate": predict_global_flush(test),
        "hierarchical_gate": predict_hierarchical(test),
    }
    result = {
        "config": asdict(cfg),
        "layers": LAYER_NAMES,
        "packet_counts": {
            "total": len(packets),
            "domain": dict(Counter(str(packet.get("domain")) for packet in packets)),
        },
        "row_counts": {
            "train": len(train),
            "test": len(test),
            "pairs": dict(Counter(row["pair"] for row in test)),
            "regime": dict(Counter(row["regime"] for row in test)),
            "heldout_packet_rows": int(sum(1 for row in test if row["heldout_packet"])),
        },
        "packet_holdouts": {
            "count": len(holdout_packet_ids),
            "domain": dict(Counter(str(packet.get("domain")) for packet in packets if str(packet["id"]) in holdout_packet_ids)),
        },
        "target_rule": "violation_layer <= memory_layer quarantines memory; higher-layer violations are contained",
        "metrics": {name: summarize(name, pred, test) for name, pred in policies.items()},
        "layer_head_metrics": predicted_layer_metrics,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print("wrote " + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
