#!/usr/bin/env python3
"""Small MLX curriculum smoke for FineWeb + teacher-query + GALT memory data."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "galt_teacher_mode" / "curriculum_preflight"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "galt_prework" / "teacher_mode_curriculum" / "smoke.json"

ACTION_LABELS = ["answer", "ask_teacher", "ask_user", "defer_residual", "refuse"]
STATE_LABELS = ["known", "uncertain", "unknown", "underspecified", "unsafe", "ambiguous_boundary"]
GALT_ACTION_LABELS = ["accept", "reject", "accept_with_residual"]
GALT_MEMORY_LABELS = ["accepted_trace", "not_consolidated", "accepted_trace_with_residual", "not_applicable"]
GALT_RESIDUAL_LABELS = ["none", "boundary_ambiguity", "hard_violation", "soft_rule_debt"]
GALT_MEMORY_LAYER_LABELS = ["l0_hard_safety", "l1_social_consent", "l2_task_rule", "l3_preference"]
GALT_VIOLATION_LAYER_LABELS = ["l0_hard_safety", "l1_social_consent", "l2_task_rule", "l3_preference", "none"]


@dataclass
class Config:
    data_dir: str = str(DEFAULT_DATA_DIR)
    output: str = str(DEFAULT_OUTPUT)
    seed: int = 42
    max_len: int = 128
    vocab_size: int = 12000
    min_freq: int = 1
    hidden_dim: int = 128
    layers: int = 2
    heads: int = 4
    causal_mask: bool = True
    steps: int = 600
    batch_size: int = 32
    lr: float = 3e-4
    lm_weight: float = 1.0
    teacher_weight: float = 1.0
    galt_weight: float = 1.0
    galt_gate_loss_weight: float = 1.0
    galt_action_loss_weight: float = 1.0
    galt_memory_loss_weight: float = 1.0
    galt_residual_loss_weight: float = 1.0
    galt_memory_layer_loss_weight: float = 0.0
    galt_violation_layer_loss_weight: float = 0.0
    galt_hierarchical_access_loss_weight: float = 0.0
    enable_galt_class_weights: bool = False
    enable_galt_action_class_weights: bool = False
    enable_galt_memory_class_weights: bool = False
    enable_galt_residual_class_weights: bool = False
    enable_galt_violation_layer_class_weights: bool = False
    galt_class_weight_clip: float = 4.0
    galt_packet_balanced_sampling: bool = True
    schedule: str = "mixed"
    stage_a_steps: int = 300
    teacher_steps: int = 400
    galt_steps: int = 300


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_']+|[^\sA-Za-z0-9_]", text.lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_text(row: dict[str, Any]) -> str:
    stream = row.get("stream")
    if stream == "fineweb_edu_lm":
        return str(row.get("text", ""))
    if stream == "teacher_query_routing":
        return " ".join(str(row.get(key) or "") for key in ("question", "context"))
    if stream == "galt_adjudicated_memory":
        return " ".join(str(row.get(key) or "") for key in ("context", "candidate_transition", "query"))
    return json.dumps(row, ensure_ascii=False)


def build_vocab(rows: list[dict[str, Any]], cfg: Config) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts.update(tokenize(row_text(row)))
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for token, count in counts.most_common():
        if len(vocab) >= cfg.vocab_size:
            break
        if count >= cfg.min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [2]
    ids.extend(vocab.get(token, 1) for token in tokenize(text)[: max_len - 2])
    ids.append(3)
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids[:max_len]


def label_index(value: str, labels: list[str], default: str) -> int:
    value = value or default
    if value not in labels:
        value = default
    return labels.index(value)


def inverse_class_weights(labels: np.ndarray, num_classes: int, clip: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    weights = np.minimum(weights, max(float(clip), 1.0))
    weights = weights / max(np.mean(weights[labels]), 1e-6)
    return weights.astype(np.float32)


def masked_inverse_example_weights(labels: np.ndarray, mask: np.ndarray, num_classes: int, clip: float) -> np.ndarray:
    out = np.zeros(len(labels), dtype=np.float32)
    active = mask > 0
    if not np.any(active):
        return out
    active_weights = inverse_class_weights(labels[active], num_classes, clip)
    out[active] = active_weights[labels[active]]
    return out


def source_group_ids(rows: list[dict[str, Any]]) -> np.ndarray:
    group_map: dict[str, int] = {}
    ids = []
    for row in rows:
        metadata = row.get("metadata", {})
        source_id = str(metadata.get("source_id") or row.get("id"))
        if source_id not in group_map:
            group_map[source_id] = len(group_map)
        ids.append(group_map[source_id])
    return np.array(ids, dtype=np.int32)


def rows_to_arrays(rows: list[dict[str, Any]], vocab: dict[str, int], cfg: Config) -> dict[str, dict[str, mx.array]]:
    by_stream: dict[str, list[dict[str, Any]]] = {
        "fineweb_edu_lm": [],
        "teacher_query_routing": [],
        "galt_adjudicated_memory": [],
    }
    for row in rows:
        stream = row.get("stream")
        if stream in by_stream:
            by_stream[stream].append(row)

    arrays: dict[str, dict[str, mx.array]] = {}
    for stream, stream_rows in by_stream.items():
        if not stream_rows:
            continue
        tokens = np.array([encode(row_text(row), vocab, cfg.max_len) for row in stream_rows], dtype=np.int32)
        data: dict[str, mx.array] = {"tokens": mx.array(tokens)}
        if stream == "teacher_query_routing":
            data["action"] = mx.array(np.array([label_index(row["action"], ACTION_LABELS, "answer") for row in stream_rows], dtype=np.int32))
            data["state"] = mx.array(np.array([label_index(row["knowledge_state"], STATE_LABELS, "known") for row in stream_rows], dtype=np.int32))
            data["residual"] = mx.array(np.array([1 if row.get("residual_target") not in (None, "none") else 0 for row in stream_rows], dtype=np.int32))
        elif stream == "galt_adjudicated_memory":
            targets = [row["targets"] for row in stream_rows]
            data["gate"] = mx.array(np.array([target["gate_target"] for target in targets], dtype=np.float32))
            action = np.array([label_index(target["action_target"], GALT_ACTION_LABELS, "accept") for target in targets], dtype=np.int32)
            memory = np.array([label_index(target["memory_target"], GALT_MEMORY_LABELS, "not_applicable") for target in targets], dtype=np.int32)
            residual = np.array([label_index(target["residual_target"], GALT_RESIDUAL_LABELS, "none") for target in targets], dtype=np.int32)
            data["action"] = mx.array(action)
            data["memory"] = mx.array(memory)
            data["residual"] = mx.array(residual)
            data["packet_group"] = mx.array(source_group_ids(stream_rows))
            metadata_rows = [row.get("metadata", {}) for row in stream_rows]
            layer_mask = np.array(
                [
                    int(
                        "memory_layer_id" in metadata
                        and "violation_layer_id" in metadata
                        and "hierarchical_access_target" in metadata
                    )
                    for metadata in metadata_rows
                ],
                dtype=np.float32,
            )
            data["layer_mask"] = mx.array(layer_mask)
            data["memory_layer"] = mx.array(
                np.array([int(metadata.get("memory_layer_id", 0)) for metadata in metadata_rows], dtype=np.int32)
            )
            data["violation_layer"] = mx.array(
                np.array([int(metadata.get("violation_layer_id", 0)) for metadata in metadata_rows], dtype=np.int32)
            )
            data["hierarchical_access"] = mx.array(
                np.array([int(metadata.get("hierarchical_access_target", 0)) for metadata in metadata_rows], dtype=np.int32)
            )
            if cfg.enable_galt_violation_layer_class_weights:
                data["violation_layer_weight"] = mx.array(
                    masked_inverse_example_weights(
                        np.array([int(metadata.get("violation_layer_id", 0)) for metadata in metadata_rows], dtype=np.int32),
                        layer_mask,
                        len(GALT_VIOLATION_LAYER_LABELS),
                        cfg.galt_class_weight_clip,
                    )
                )
            if cfg.enable_galt_class_weights or cfg.enable_galt_action_class_weights:
                action_weights = inverse_class_weights(action, len(GALT_ACTION_LABELS), cfg.galt_class_weight_clip)
                data["action_weight"] = mx.array(action_weights[action])
            if cfg.enable_galt_class_weights or cfg.enable_galt_memory_class_weights:
                memory_weights = inverse_class_weights(memory, len(GALT_MEMORY_LABELS), cfg.galt_class_weight_clip)
                data["memory_weight"] = mx.array(memory_weights[memory])
            if cfg.enable_galt_class_weights or cfg.enable_galt_residual_class_weights:
                residual_weights = inverse_class_weights(residual, len(GALT_RESIDUAL_LABELS), cfg.galt_class_weight_clip)
                data["residual_weight"] = mx.array(residual_weights[residual])
        arrays[stream] = data
    return arrays


def batch_take(arrays: dict[str, mx.array], indices: np.ndarray) -> dict[str, mx.array]:
    idx = mx.array(indices.astype(np.int32))
    return {key: value[idx] for key, value in arrays.items()}


class CurriculumModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.hidden_dim)
        self.pos = nn.Embedding(cfg.max_len, cfg.hidden_dim)
        self.layers = [
            nn.TransformerEncoderLayer(
                cfg.hidden_dim,
                cfg.heads,
                mlp_dims=cfg.hidden_dim * 4,
                norm_first=True,
            )
            for _ in range(cfg.layers)
        ]
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, vocab_size)
        self.teacher_action_head = nn.Linear(cfg.hidden_dim, len(ACTION_LABELS))
        self.teacher_state_head = nn.Linear(cfg.hidden_dim, len(STATE_LABELS))
        self.teacher_residual_head = nn.Linear(cfg.hidden_dim, 2)
        self.galt_gate_head = nn.Linear(cfg.hidden_dim, 3)
        self.galt_action_head = nn.Linear(cfg.hidden_dim, len(GALT_ACTION_LABELS))
        self.galt_memory_head = nn.Linear(cfg.hidden_dim, len(GALT_MEMORY_LABELS))
        self.galt_residual_head = nn.Linear(cfg.hidden_dim, len(GALT_RESIDUAL_LABELS))
        self.galt_memory_layer_head = nn.Linear(cfg.hidden_dim, len(GALT_MEMORY_LAYER_LABELS))
        self.galt_violation_layer_head = nn.Linear(cfg.hidden_dim, len(GALT_VIOLATION_LAYER_LABELS))
        self.galt_hierarchical_access_head = nn.Linear(cfg.hidden_dim, 2)
        self.causal_mask = cfg.causal_mask

    def encode(self, tokens: mx.array) -> mx.array:
        pos = mx.arange(tokens.shape[1])[None, :]
        x = self.embed(tokens) + self.pos(pos)
        attn_mask = None
        if self.causal_mask:
            attn_mask = nn.MultiHeadAttention.create_additive_causal_mask(tokens.shape[1], x.dtype)
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.norm(x)

    def __call__(self, tokens: mx.array) -> dict[str, mx.array]:
        hidden = self.encode(tokens)
        mask = (tokens != 0).astype(mx.float32)
        pooled = (hidden * mask[..., None]).sum(axis=1) / mx.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        return {
            "hidden": hidden,
            "lm": self.lm_head(hidden),
            "teacher_action": self.teacher_action_head(pooled),
            "teacher_state": self.teacher_state_head(pooled),
            "teacher_residual": self.teacher_residual_head(pooled),
            "galt_gate": self.galt_gate_head(pooled),
            "galt_action": self.galt_action_head(pooled),
            "galt_memory": self.galt_memory_head(pooled),
            "galt_residual": self.galt_residual_head(pooled),
            "galt_memory_layer": self.galt_memory_layer_head(pooled),
            "galt_violation_layer": self.galt_violation_layer_head(pooled),
            "galt_hierarchical_access": self.galt_hierarchical_access_head(pooled),
        }


def masked_lm_loss(logits: mx.array, tokens: mx.array) -> mx.array:
    pred = logits[:, :-1, :]
    target = tokens[:, 1:]
    mask = (target != 0).astype(mx.float32)
    losses = nn.losses.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none")
    losses = losses.reshape(target.shape)
    return (losses * mask).sum() / mx.maximum(mask.sum(), 1.0)


def ce(logits: mx.array, labels: mx.array) -> mx.array:
    return nn.losses.cross_entropy(logits, labels, reduction="mean")


def weighted_ce(logits: mx.array, labels: mx.array, weights: mx.array | None = None) -> mx.array:
    if weights is None:
        return ce(logits, labels)
    losses = nn.losses.cross_entropy(logits, labels, reduction="none")
    return (losses * weights).sum() / mx.maximum(weights.sum(), 1.0)


def masked_ce(logits: mx.array, labels: mx.array, mask: mx.array) -> mx.array:
    losses = nn.losses.cross_entropy(logits, labels, reduction="none")
    return (losses * mask).sum() / mx.maximum(mask.sum(), 1.0)


def masked_weighted_ce(logits: mx.array, labels: mx.array, mask: mx.array, weights: mx.array | None = None) -> mx.array:
    if weights is None:
        return masked_ce(logits, labels, mask)
    losses = nn.losses.cross_entropy(logits, labels, reduction="none")
    active_weights = weights * mask
    return (losses * active_weights).sum() / mx.maximum(active_weights.sum(), 1.0)


def bce_logits(logits: mx.array, labels: mx.array) -> mx.array:
    return nn.losses.binary_cross_entropy(logits, labels, reduction="mean")


def acc(logits: mx.array, labels: mx.array) -> float:
    return float(mx.mean((mx.argmax(logits, axis=-1) == labels).astype(mx.float32)))


def masked_acc(logits: mx.array, labels: mx.array, mask: mx.array) -> float:
    count = float(mask.astype(mx.float32).sum())
    if count == 0.0:
        return math.nan
    pred = mx.argmax(logits, axis=-1)
    return float((((pred == labels).astype(mx.float32)) * mask).sum() / count)


def masked_pred_acc(pred: mx.array, labels: mx.array, mask: mx.array) -> float:
    count = float(mask.astype(mx.float32).sum())
    if count == 0.0:
        return math.nan
    return float((((pred == labels).astype(mx.float32)) * mask).sum() / count)


def gate_acc(logits: mx.array, labels: mx.array) -> float:
    pred = (mx.sigmoid(logits) >= 0.5).astype(mx.float32)
    return float(mx.mean((pred == labels).astype(mx.float32)))


def class_recalls(logits: mx.array, labels: mx.array, names: list[str], prefix: str) -> dict[str, float]:
    pred = mx.argmax(logits, axis=-1)
    out = {}
    for idx, name in enumerate(names):
        mask = labels == idx
        count = float(mask.astype(mx.float32).sum())
        if count == 0.0:
            out[f"{prefix}_{name}_recall"] = math.nan
            continue
        hits = ((pred == idx) & mask).astype(mx.float32).sum()
        out[f"{prefix}_{name}_recall"] = float(hits / count)
    return out


def sample_packet_balanced_batch(stream_arrays: dict[str, mx.array], batch_size: int, rng: np.random.Generator) -> dict[str, mx.array]:
    groups = np.array(stream_arrays["packet_group"])
    unique = np.unique(groups)
    if len(unique) == 0:
        return sample_random_batch(stream_arrays, batch_size, rng)
    packet_count = max(1, batch_size // 3)
    chosen = rng.choice(unique, size=min(packet_count, len(unique)), replace=len(unique) < packet_count)
    indices = np.concatenate([np.where(groups == group)[0] for group in chosen]).astype(np.int32)
    rng.shuffle(indices)
    return batch_take(stream_arrays, indices)


def sample_random_batch(stream_arrays: dict[str, mx.array], batch_size: int, rng: np.random.Generator) -> dict[str, mx.array]:
    n = int(stream_arrays["tokens"].shape[0])
    idx = rng.integers(0, n, size=batch_size)
    return batch_take(stream_arrays, idx)


def sample_batch(
    stream: str,
    stream_arrays: dict[str, mx.array],
    batch_size: int,
    rng: np.random.Generator,
    cfg: Config,
) -> dict[str, mx.array]:
    if stream == "galt_adjudicated_memory" and cfg.galt_packet_balanced_sampling and "packet_group" in stream_arrays:
        return sample_packet_balanced_batch(stream_arrays, batch_size, rng)
    return sample_random_batch(stream_arrays, batch_size, rng)


def active_streams_for_step(cfg: Config, step: int, available: set[str]) -> tuple[str, list[str]]:
    if cfg.schedule == "mixed":
        return "mixed", sorted(available)
    if cfg.schedule != "staged":
        raise ValueError(f"unknown schedule: {cfg.schedule}")
    if step <= cfg.stage_a_steps:
        return "stage_a_lm", [stream for stream in ("fineweb_edu_lm",) if stream in available]
    if step <= cfg.stage_a_steps + cfg.teacher_steps:
        return "stage_c_teacher_query", [
            stream for stream in ("fineweb_edu_lm", "teacher_query_routing") if stream in available
        ]
    if step <= cfg.stage_a_steps + cfg.teacher_steps + cfg.galt_steps:
        return "stage_d_galt", [
            stream
            for stream in ("fineweb_edu_lm", "teacher_query_routing", "galt_adjudicated_memory")
            if stream in available
        ]
    return "stage_e_mixed_replay", sorted(available)


def loss_fn(model: CurriculumModel, batches: dict[str, dict[str, mx.array]], cfg: Config) -> mx.array:
    total = mx.array(0.0)
    if "fineweb_edu_lm" in batches:
        out = model(batches["fineweb_edu_lm"]["tokens"])
        total = total + cfg.lm_weight * masked_lm_loss(out["lm"], batches["fineweb_edu_lm"]["tokens"])
    if "teacher_query_routing" in batches:
        batch = batches["teacher_query_routing"]
        out = model(batch["tokens"])
        total = total + cfg.teacher_weight * (
            ce(out["teacher_action"], batch["action"])
            + ce(out["teacher_state"], batch["state"])
            + 0.5 * ce(out["teacher_residual"], batch["residual"])
        )
    if "galt_adjudicated_memory" in batches:
        batch = batches["galt_adjudicated_memory"]
        out = model(batch["tokens"])
        total = total + cfg.galt_weight * (
            cfg.galt_gate_loss_weight * bce_logits(out["galt_gate"], batch["gate"])
            + cfg.galt_action_loss_weight * weighted_ce(out["galt_action"], batch["action"], batch.get("action_weight"))
            + cfg.galt_memory_loss_weight * weighted_ce(out["galt_memory"], batch["memory"], batch.get("memory_weight"))
            + cfg.galt_residual_loss_weight * weighted_ce(out["galt_residual"], batch["residual"], batch.get("residual_weight"))
            + cfg.galt_memory_layer_loss_weight * masked_ce(out["galt_memory_layer"], batch["memory_layer"], batch["layer_mask"])
            + cfg.galt_violation_layer_loss_weight
            * masked_weighted_ce(
                out["galt_violation_layer"],
                batch["violation_layer"],
                batch["layer_mask"],
                batch.get("violation_layer_weight"),
            )
            + cfg.galt_hierarchical_access_loss_weight
            * masked_ce(out["galt_hierarchical_access"], batch["hierarchical_access"], batch["layer_mask"])
        )
    return total


def evaluate_stream(model: CurriculumModel, arrays: dict[str, dict[str, mx.array]], stream: str) -> dict[str, float]:
    if stream not in arrays:
        return {}
    batch = arrays[stream]
    out = model(batch["tokens"])
    mx.eval(out)
    if stream == "fineweb_edu_lm":
        return {"lm_loss": float(masked_lm_loss(out["lm"], batch["tokens"]))}
    if stream == "teacher_query_routing":
        return {
            "teacher_action_acc": acc(out["teacher_action"], batch["action"]),
            "teacher_state_acc": acc(out["teacher_state"], batch["state"]),
            "teacher_residual_acc": acc(out["teacher_residual"], batch["residual"]),
        }
    if stream == "galt_adjudicated_memory":
        pred_memory_layer = mx.argmax(out["galt_memory_layer"], axis=-1)
        pred_violation_layer = mx.argmax(out["galt_violation_layer"], axis=-1)
        pred_access_from_layers = ((pred_violation_layer == 4) | (pred_violation_layer > pred_memory_layer)).astype(mx.int32)
        return {
            "galt_gate_acc": gate_acc(out["galt_gate"], batch["gate"]),
            "galt_action_acc": acc(out["galt_action"], batch["action"]),
            "galt_memory_acc": acc(out["galt_memory"], batch["memory"]),
            "galt_residual_acc": acc(out["galt_residual"], batch["residual"]),
            "galt_memory_layer_acc": masked_acc(out["galt_memory_layer"], batch["memory_layer"], batch["layer_mask"]),
            "galt_violation_layer_acc": masked_acc(out["galt_violation_layer"], batch["violation_layer"], batch["layer_mask"]),
            "galt_hierarchical_access_acc": masked_acc(
                out["galt_hierarchical_access"], batch["hierarchical_access"], batch["layer_mask"]
            ),
            "galt_predicted_layer_access_acc": masked_pred_acc(
                pred_access_from_layers, batch["hierarchical_access"], batch["layer_mask"]
            ),
            **class_recalls(out["galt_action"], batch["action"], GALT_ACTION_LABELS, "galt_action"),
            **class_recalls(out["galt_memory"], batch["memory"], GALT_MEMORY_LABELS, "galt_memory"),
            **class_recalls(out["galt_residual"], batch["residual"], GALT_RESIDUAL_LABELS, "galt_residual"),
        }
    return {}


def evaluate(model: CurriculumModel, arrays: dict[str, dict[str, mx.array]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for stream in ("fineweb_edu_lm", "teacher_query_routing", "galt_adjudicated_memory"):
        metrics.update(evaluate_stream(model, arrays, stream))
    return metrics


def parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    for field, default in Config().__dict__.items():
        arg_type = parse_bool_arg if isinstance(default, bool) else type(default)
        parser.add_argument("--" + field.replace("_", "-"), type=arg_type, default=default)
    cfg = Config(**vars(parser.parse_args()))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    mx.random.seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "validation.jsonl")
    test_path = data_dir / "test.jsonl"
    test_rows = read_jsonl(test_path) if test_path.exists() else []
    vocab = build_vocab(train_rows, cfg)
    train = rows_to_arrays(train_rows, vocab, cfg)
    val = rows_to_arrays(val_rows, vocab, cfg)
    test = rows_to_arrays(test_rows, vocab, cfg) if test_rows else {}

    model = CurriculumModel(len(vocab), cfg)
    optimizer = optim.AdamW(learning_rate=cfg.lr, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(cfg.seed)

    history = []
    available_streams = set(train)
    for step in range(1, cfg.steps + 1):
        phase, streams = active_streams_for_step(cfg, step, available_streams)
        stream_batch = max(1, cfg.batch_size // max(1, len(streams)))
        batches = {
            stream: sample_batch(stream, stream_arrays, stream_batch, rng, cfg)
            for stream, stream_arrays in train.items()
            if stream in streams
        }
        loss, grads = loss_and_grad(model, batches, cfg)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if step == 1 or step % max(1, cfg.steps // 6) == 0:
            metrics = {f"val_{key}": value for key, value in evaluate(model, val).items()}
            metrics["step"] = step
            metrics["phase"] = phase
            metrics["loss"] = float(loss)
            history.append(metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

    final_val = evaluate(model, val)
    final_test = evaluate(model, test) if test else {}
    result = {
        "config": asdict(cfg),
        "vocab_size": len(vocab),
        "train_counts": {stream: int(values["tokens"].shape[0]) for stream, values in train.items()},
        "val_counts": {stream: int(values["tokens"].shape[0]) for stream, values in val.items()},
        "test_counts": {stream: int(values["tokens"].shape[0]) for stream, values in test.items()},
        "history": history,
        "final": final_val,
        "final_test": final_test,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote " + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
