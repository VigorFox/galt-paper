#!/usr/bin/env python3
"""Benchmark safety-indexed hierarchical memory access.

The task is intentionally small and synthetic.  It tests whether a memory item
should remain accessible when a rule at some layer is violated.

Layer semantics:
  L0 hard admissibility / basic safety
  L1 social / consent boundary
  L2 task or workflow rule
  L3 preference / durable memory

Target rule:
  A violation at layer v invalidates memory at layer m when v <= m.
  A higher-layer violation v > m should not invalidate lower-layer memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "galt_prework" / "hierarchical_memory_layer_benchmark.json"

LAYER_NAMES = ["l0_hard_safety", "l1_social", "l2_task_rule", "l3_preference"]
NONE_VIOLATION = 4


@dataclass
class Config:
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    train_per_pair: int = 24
    test_per_pair: int = 64
    hidden_dim: int = 24
    steps: int = 1200
    lr: float = 0.05
    weight_decay: float = 1e-4
    output: str = str(DEFAULT_OUTPUT)


HOLDOUT_PAIRS = {
    # Bottom-up invalidation pairs: these require knowing that lower layers
    # invalidate higher memory, not only same-layer matches.
    (2, 0),
    (3, 0),
    (3, 1),
    # Top-down containment pairs: these require knowing that higher-layer
    # failures should not destroy lower-layer memory.
    (0, 3),
    (1, 3),
}


def one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    out[index] = 1.0
    return out


def target_access(memory_layer: int, violation_layer: int) -> int:
    if violation_layer == NONE_VIOLATION:
        return 1
    return int(violation_layer > memory_layer)


def build_feature(memory_layer: int, violation_layer: int, rng: np.random.Generator) -> np.ndarray:
    layer_feat = one_hot(memory_layer, 4)
    violation_feat = one_hot(violation_layer, 5)
    # Small nuisance features make this a learned classification problem while
    # preserving the clean layer symbols needed by the structured policy.
    nuisance = rng.normal(0.0, 0.03, size=6).astype(np.float32)
    return np.concatenate([layer_feat, violation_feat, nuisance]).astype(np.float32)


def make_rows(cfg: Config, seed: int, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    rng = np.random.default_rng(seed + (0 if split == "train" else 10_000))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    meta: list[dict[str, int | str]] = []
    per_pair = cfg.train_per_pair if split == "train" else cfg.test_per_pair
    for memory_layer in range(4):
        for violation_layer in range(5):
            if split == "train" and (memory_layer, violation_layer) in HOLDOUT_PAIRS:
                continue
            for _ in range(per_pair):
                xs.append(build_feature(memory_layer, violation_layer, rng))
                ys.append(target_access(memory_layer, violation_layer))
                if violation_layer == NONE_VIOLATION:
                    regime = "no_violation_access"
                elif violation_layer <= memory_layer:
                    regime = "bottom_up_invalidation"
                else:
                    regime = "top_down_containment"
                meta.append(
                    {
                        "memory_layer": memory_layer,
                        "violation_layer": violation_layer,
                        "heldout_pair": int((memory_layer, violation_layer) in HOLDOUT_PAIRS),
                        "regime": regime,
                    }
                )
    order = rng.permutation(len(xs))
    return np.stack(xs)[order], np.array(ys, dtype=np.int64)[order], [meta[i] for i in order]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def train_flat_mlp(cfg: Config, seed: int, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.15, size=(x.shape[1], cfg.hidden_dim)).astype(np.float32)
    b1 = np.zeros(cfg.hidden_dim, dtype=np.float32)
    w2 = rng.normal(0.0, 0.15, size=(cfg.hidden_dim, 1)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    yf = y.astype(np.float32)[:, None]
    n = len(x)
    for _ in range(cfg.steps):
        h_pre = x @ w1 + b1
        h = np.tanh(h_pre)
        logits = h @ w2 + b2
        p = sigmoid(logits)
        dlogits = (p - yf) / n
        dw2 = h.T @ dlogits + cfg.weight_decay * w2
        db2 = dlogits.sum(axis=0)
        dh = dlogits @ w2.T
        dh_pre = dh * (1.0 - h * h)
        dw1 = x.T @ dh_pre + cfg.weight_decay * w1
        db1 = dh_pre.sum(axis=0)
        w1 -= cfg.lr * dw1
        b1 -= cfg.lr * db1
        w2 -= cfg.lr * dw2
        b2 -= cfg.lr * db2
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def predict_flat(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = np.tanh(x @ model["w1"] + model["b1"])
    p = sigmoid(h @ model["w2"] + model["b2"])[:, 0]
    return (p >= 0.5).astype(np.int64)


def predict_independent(meta: list[dict[str, int | str]]) -> np.ndarray:
    out = []
    for row in meta:
        violation_layer = int(row["violation_layer"])
        memory_layer = int(row["memory_layer"])
        out.append(int(violation_layer == NONE_VIOLATION or violation_layer != memory_layer))
    return np.array(out, dtype=np.int64)


def predict_global_flush(meta: list[dict[str, int | str]]) -> np.ndarray:
    return np.array([int(int(row["violation_layer"]) == NONE_VIOLATION) for row in meta], dtype=np.int64)


def predict_hierarchical(meta: list[dict[str, int | str]]) -> np.ndarray:
    return np.array([target_access(int(row["memory_layer"]), int(row["violation_layer"])) for row in meta], dtype=np.int64)


def summarize(pred: np.ndarray, y: np.ndarray, meta: list[dict[str, int | str]]) -> dict[str, float]:
    metrics: dict[str, float] = {"accuracy": float(np.mean(pred == y))}
    for regime in ("no_violation_access", "bottom_up_invalidation", "top_down_containment"):
        mask = np.array([row["regime"] == regime for row in meta], dtype=bool)
        metrics[f"{regime}_accuracy"] = float(np.mean(pred[mask] == y[mask]))
    hard = np.array([bool(row["heldout_pair"]) for row in meta], dtype=bool)
    metrics["heldout_pair_accuracy"] = float(np.mean(pred[hard] == y[hard]))
    bottom_hard = hard & np.array([row["regime"] == "bottom_up_invalidation" for row in meta], dtype=bool)
    top_hard = hard & np.array([row["regime"] == "top_down_containment" for row in meta], dtype=bool)
    metrics["heldout_bottom_up_accuracy"] = float(np.mean(pred[bottom_hard] == y[bottom_hard]))
    metrics["heldout_top_down_accuracy"] = float(np.mean(pred[top_hard] == y[top_hard]))
    return metrics


def run_seed(cfg: Config, seed: int) -> dict[str, dict[str, float]]:
    train_x, train_y, _ = make_rows(cfg, seed, "train")
    test_x, test_y, test_meta = make_rows(cfg, seed, "test")
    flat = train_flat_mlp(cfg, seed, train_x, train_y)
    return {
        "flat_mlp": summarize(predict_flat(flat, test_x), test_y, test_meta),
        "global_flush_gate": summarize(predict_global_flush(test_meta), test_y, test_meta),
        "independent_layer_gate": summarize(predict_independent(test_meta), test_y, test_meta),
        "hierarchical_gate": summarize(predict_hierarchical(test_meta), test_y, test_meta),
    }


def aggregate(per_seed: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    policies = sorted(next(iter(per_seed.values())).keys())
    out: dict[str, dict[str, dict[str, float]]] = {}
    for policy in policies:
        out[policy] = {}
        metric_names = sorted(next(iter(per_seed.values()))[policy].keys())
        for metric in metric_names:
            values = np.array([per_seed[str(seed)][policy][metric] for seed in per_seed], dtype=np.float64)
            out[policy][metric] = {"mean": float(values.mean()), "std": float(values.std())}
    return out


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--train-per-pair", type=int, default=Config.train_per_pair)
    parser.add_argument("--test-per-pair", type=int, default=Config.test_per_pair)
    args = parser.parse_args()
    return Config(
        output=args.output,
        steps=args.steps,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        train_per_pair=args.train_per_pair,
        test_per_pair=args.test_per_pair,
    )


def main() -> int:
    cfg = parse_args()
    per_seed = {str(seed): run_seed(cfg, seed) for seed in cfg.seeds}
    result = {
        "config": asdict(cfg),
        "layers": LAYER_NAMES,
        "target_rule": "violation_layer <= memory_layer invalidates memory; higher-layer violations are contained",
        "holdout_pairs": sorted([list(pair) for pair in HOLDOUT_PAIRS]),
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print("wrote " + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
