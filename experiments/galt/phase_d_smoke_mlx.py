"""Phase D smoke on a real MLX Transformer with all-layer LoRA blocks.

This is the first real-model bridge after Phase C + Phase D profiling.
It does NOT implement full block-local ADMM yet. Instead, it validates the
practical scaffolding required for Phase D:

1. all-layer LoRA block partitioning on a live Qwen-MLX model
2. exact last-token hidden boundary messages (chosen by phase_d_profile_mlx.py)
3. one forward-consistency edge over all adapted boundaries
4. one safety edge + one knowledge-preservation edge
5. AVBD-GALT multi-constraint optimizer wiring on a live training loop
6. modeled per-block local-solve accounting with d_v >> k diagnostics

The success bar for this smoke is operational:
- finite live steps on the real model
- forward/safety/retain constraints all produce usable gradients
- boundary snapshots refresh on outer steps
- per-layer block bookkeeping is stable and SM-friendly (d_v / k well above threshold)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm.tuner.utils import linear_to_lora_layers

PORTABLE_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PORTABLE_ROOT / "optimizer"))
sys.path.insert(0, str(PORTABLE_ROOT / "data_utils"))
sys.path.insert(0, str(PORTABLE_ROOT / "experiments" / "shared_runtime"))
sys.path.insert(0, str(THIS_DIR))

from avbd_galt_optimizer_mlx import AVBDGALTOptimizer
from data import default_categories_for_source, load_continual_tasks, load_edit_samples, load_safety_samples
from hidden_collector_mlx import forward_collect_hiddens, get_hidden_size, get_num_layers
from mlx_utils import flatten_tree, scalar
from continual_runtime_mlx import (
    DEFAULT_MODEL_NAME,
    ExperimentConfig,
    batch_labels,
    compute_choice_loss,
    compute_choice_scores_batch,
    compute_distillation_kl,
    compute_replay_anchor_probs,
    evaluate,
    get_choice_token_ids,
    sample_replay_buffer,
    set_seed,
    tokenize_prompt,
    tokenize_prompts,
)


RESULTS_DIR = PORTABLE_ROOT / "results" / "galt_prework" / "phase_d_smoke"
LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")


@dataclass
class SmokeConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_source: str = "ag_news"
    local_files_only: bool = True
    max_train_per_task: int = 16
    max_eval_per_task: int = 32
    eval_batch_size: int = 4
    batch_size: int = 1
    max_length: int = 256
    lora_rank: int = 8
    lora_scale: float = 16.0
    lora_dropout: float = 0.0
    lr: float = 5e-5
    seed: int = 42
    smoke_steps: int = 6
    outer_step_freq: int = 3
    replay_size: int = 4
    safety_eval_size: int = 32
    retain_eval_size: int = 32
    avbd_constraint_temperature: float = 1.5
    forward_margin: float = 0.0
    safety_margin: float = 0.0
    retain_margin: float = 0.0
    rho_p_init: float = 0.5
    rho_p_min: float = 0.5
    rho_p_max: float = 5.0
    rho_p_growth: float = 1.5
    rho_i: float = 0.1
    rho_anchor: float = 0.3
    weight_decay: float = 0.01
    do_safety_warmup: bool = False
    warmup_epochs: int = 1
    warmup_lr: float = 1e-4
    output: str = str(RESULTS_DIR / "summary.json")


def _load_model_all_layer_lora(cfg: SmokeConfig):
    if cfg.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model, tokenizer = mlx_load(cfg.model_name, tokenizer_config={"trust_remote_code": True})
    model.freeze()
    linear_to_lora_layers(
        model,
        len(model.layers),
        {"rank": cfg.lora_rank, "scale": cfg.lora_scale, "dropout": cfg.lora_dropout},
    )
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.eval(model.parameters())
    return model, tokenizer


def _param_count(array: mx.array) -> int:
    count = 1
    for dim in array.shape:
        count *= int(dim)
    return count


def _layer_param_counts(model) -> list[dict]:
    flat = flatten_tree(model.trainable_parameters())
    buckets: dict[int, dict] = {}
    for name, value in flat.items():
        match = LAYER_RE.match(name)
        if not match:
            continue
        layer_idx = int(match.group(1))
        bucket = buckets.setdefault(layer_idx, {"n_params": 0, "names": []})
        bucket["n_params"] += _param_count(value)
        if len(bucket["names"]) < 4:
            bucket["names"].append(name)
    return [
        {
            "layer_index": layer_idx,
            "n_params": info["n_params"],
            "example_names": info["names"],
        }
        for layer_idx, info in sorted(buckets.items())
    ]


def _group_grad_norms_by_layer(flat_grads: dict[str, mx.array]) -> dict[int, float]:
    layer_sq: dict[int, float] = {}
    for name, grad in flat_grads.items():
        if grad is None:
            continue
        match = LAYER_RE.match(name)
        if not match:
            continue
        layer_idx = int(match.group(1))
        sq = scalar(mx.sum(grad * grad))
        layer_sq[layer_idx] = layer_sq.get(layer_idx, 0.0) + sq
    return {layer: math.sqrt(max(0.0, value)) for layer, value in sorted(layer_sq.items())}


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _capture_last_token_hiddens(model, prompt_ids: list[int], boundary_layers: list[int]) -> dict[int, mx.array]:
    token_array = mx.array([prompt_ids], dtype=mx.int32)
    hiddens, _ = forward_collect_hiddens(model, token_array, boundary_layers, return_logits=False)
    last_token = {layer: hiddens[layer][:, -1, :] for layer in boundary_layers}
    mx.eval(*last_token.values())
    return last_token


def _forward_probe_residuals(
    model,
    prompt_ids: list[int],
    boundary_layers: list[int],
    targets: dict[int, mx.array],
) -> tuple[float, dict[int, float]]:
    current = _capture_last_token_hiddens(model, prompt_ids, boundary_layers)
    per_layer = {}
    for layer in boundary_layers:
        diff = current[layer] - targets[layer]
        per_layer[layer] = scalar(mx.mean(diff * diff))
    mean_residual = sum(per_layer.values()) / max(1, len(per_layer))
    return mean_residual, per_layer


def _make_forward_loss_fn(
    prompt_ids: list[int],
    boundary_layers: list[int],
    targets: dict[int, mx.array],
):
    token_array = mx.array([prompt_ids], dtype=mx.int32)

    def loss_fn(module):
        hiddens, _ = forward_collect_hiddens(module, token_array, boundary_layers, return_logits=False)
        loss = mx.array(0.0, dtype=mx.float32)
        for layer in boundary_layers:
            current = hiddens[layer][:, -1, :]
            diff = current - targets[layer]
            loss = loss + mx.mean(diff * diff)
        return loss / max(1, len(boundary_layers))

    return loss_fn


def _make_kl_constraint_fn(tokenizer, replay: list[dict], anchor_probs: mx.array, choice_token_ids, cfg: SmokeConfig):
    prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in replay], cfg.max_length)

    def loss_fn(module):
        scores = compute_choice_scores_batch(module, prompt_tokens, choice_token_ids)
        return compute_distillation_kl(scores, anchor_probs, temperature=cfg.avbd_constraint_temperature)

    return loss_fn


def _eval_constraint_value(model, loss_fn) -> float:
    value = loss_fn(model)
    mx.eval(value)
    return scalar(value)


def _load_smoke_data(cfg: SmokeConfig):
    tasks = load_continual_tasks(
        dataset_source=cfg.dataset_source,
        categories=default_categories_for_source(cfg.dataset_source),
        max_train_per_task=cfg.max_train_per_task,
        max_eval_per_task=cfg.max_eval_per_task,
        seed=cfg.seed,
        local_files_only=cfg.local_files_only,
    )
    safety_samples = load_safety_samples(str(PORTABLE_ROOT / "prompts" / "safety_prompts.json"))
    retain_samples = load_edit_samples(str(PORTABLE_ROOT / "prompts" / "retain_set.json"))
    return tasks, safety_samples, retain_samples


def _modeled_local_solve_stats(layer_counts: list[dict], k_boundary: int) -> dict:
    works = [layer["n_params"] * k_boundary for layer in layer_counts]
    return {
        "k_boundary": k_boundary,
        "sum_d_times_k": int(sum(works)),
        "max_d_times_k": int(max(works) if works else 0),
        "parallel_gain_upper_bound": float(sum(works) / max(1, max(works) if works else 1)),
    }


def run_smoke(cfg: SmokeConfig) -> dict:
    set_seed(cfg.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_all_layer_lora(cfg)
    choice_token_ids = get_choice_token_ids(tokenizer)
    tasks, safety_samples, retain_samples = _load_smoke_data(cfg)
    task = tasks[0]

    eval_cfg = ExperimentConfig(
        model_name=cfg.model_name,
        dataset_source=cfg.dataset_source,
        local_files_only=cfg.local_files_only,
        eval_batch_size=cfg.eval_batch_size,
        max_length=cfg.max_length,
    )

    if cfg.do_safety_warmup:
        from run_sfb_experiment_mlx import safety_warmup

        print("[phase-d-smoke] running brief safety warmup")
        safety_warmup(
            model,
            tokenizer,
            safety_samples[: cfg.safety_eval_size],
            choice_token_ids,
            eval_cfg,
            warmup_epochs=cfg.warmup_epochs,
            warmup_lr=cfg.warmup_lr,
        )

    num_layers = get_num_layers(model)
    hidden_size = get_hidden_size(model)
    boundary_layers = list(range(num_layers))
    layer_counts = _layer_param_counts(model)
    d_over_k = {
        layer["layer_index"]: float(layer["n_params"] / hidden_size)
        for layer in layer_counts
    }

    safety_replay = sample_replay_buffer(safety_samples, min(cfg.replay_size, len(safety_samples)), cfg.seed)
    retain_replay = sample_replay_buffer(retain_samples, min(cfg.replay_size, len(retain_samples)), cfg.seed + 17)
    safety_anchor = compute_replay_anchor_probs(model, tokenizer, safety_replay, choice_token_ids, eval_cfg)
    retain_anchor = compute_replay_anchor_probs(model, tokenizer, retain_replay, choice_token_ids, eval_cfg)
    mx.eval(safety_anchor, retain_anchor)

    probe_prompt = task.train_samples[0]["prompt"]
    probe_tokens = tokenize_prompt(tokenizer, probe_prompt, cfg.max_length)
    forward_targets = _capture_last_token_hiddens(model, probe_tokens, boundary_layers)

    task_eval_samples = task.eval_samples
    safety_eval_samples = safety_samples[: cfg.safety_eval_size]
    retain_eval_samples = retain_samples[: cfg.retain_eval_size]

    pre_metrics = {
        "task_acc": evaluate(model, tokenizer, task_eval_samples, choice_token_ids, eval_cfg),
        "safety_acc": evaluate(model, tokenizer, safety_eval_samples, choice_token_ids, eval_cfg),
        "retain_acc": evaluate(model, tokenizer, retain_eval_samples, choice_token_ids, eval_cfg),
    }

    safety_constraint_fn = _make_kl_constraint_fn(tokenizer, safety_replay, safety_anchor, choice_token_ids, cfg)
    retain_constraint_fn = _make_kl_constraint_fn(tokenizer, retain_replay, retain_anchor, choice_token_ids, cfg)

    optimizer = AVBDGALTOptimizer(
        model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        rho_p_init=cfg.rho_p_init,
        rho_p_min=cfg.rho_p_min,
        rho_p_max=cfg.rho_p_max,
        rho_p_growth=cfg.rho_p_growth,
        rho_i=cfg.rho_i,
        outer_step_freq=cfg.outer_step_freq,
        rho_anchor=cfg.rho_anchor,
        use_multi_constraint_woodbury=True,
    )
    forward_ci = optimizer.add_constraint("forward_last_token")
    safety_ci = optimizer.add_constraint("safety_alignment")
    retain_ci = optimizer.add_constraint("knowledge_retain")

    def task_loss(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        return compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)

    task_grad_fn = nn.value_and_grad(model, task_loss)
    step_trace = []
    refresh_steps = []
    saw_nonzero_forward_grad_layers = set()
    saw_nonzero_task_grad_layers = set()

    train_samples = list(task.train_samples)
    random.Random(cfg.seed).shuffle(train_samples)

    for step_idx in range(cfg.smoke_steps):
        batch = [train_samples[step_idx % len(train_samples)]]
        task_value, task_grads = task_grad_fn(model, batch)
        flat_task_grads = flatten_tree(task_grads)
        task_layer_norms = _group_grad_norms_by_layer(flat_task_grads)
        saw_nonzero_task_grad_layers.update(layer for layer, norm in task_layer_norms.items() if norm > 0.0)

        forward_constraint_fn = _make_forward_loss_fn(probe_tokens, boundary_layers, forward_targets)
        forward_grad_fn = nn.value_and_grad(model, forward_constraint_fn)
        forward_value_tensor, forward_grads_tree = forward_grad_fn(model)
        forward_grads = flatten_tree(forward_grads_tree)
        forward_layer_norms = _group_grad_norms_by_layer(forward_grads)
        saw_nonzero_forward_grad_layers.update(layer for layer, norm in forward_layer_norms.items() if norm > 0.0)

        safety_value = _eval_constraint_value(model, safety_constraint_fn)
        retain_value = _eval_constraint_value(model, retain_constraint_fn)
        forward_value = scalar(forward_value_tensor)
        mx.eval(task_value, forward_value_tensor)

        forward_mean_resid, per_layer_forward = _forward_probe_residuals(
            model, probe_tokens, boundary_layers, forward_targets
        )

        safety_grad_fn = nn.value_and_grad(model, safety_constraint_fn)
        retain_grad_fn = nn.value_and_grad(model, retain_constraint_fn)
        _safety_value_tensor, safety_grads_tree = safety_grad_fn(model)
        _retain_value_tensor, retain_grads_tree = retain_grad_fn(model)
        safety_grads = flatten_tree(safety_grads_tree)
        retain_grads = flatten_tree(retain_grads_tree)

        optimizer.set_constraint_grads(
            forward_ci,
            forward_value - cfg.forward_margin,
            forward_grads,
        )
        optimizer.set_constraint_grads(
            safety_ci,
            safety_value - cfg.safety_margin,
            safety_grads,
        )
        optimizer.set_constraint_grads(
            retain_ci,
            retain_value - cfg.retain_margin,
            retain_grads,
        )
        optimizer.step(flat_task_grads)

        refreshed = False
        if (step_idx + 1) % cfg.outer_step_freq == 0:
            forward_targets = _capture_last_token_hiddens(model, probe_tokens, boundary_layers)
            refresh_steps.append(step_idx + 1)
            refreshed = True

        cstate = optimizer.get_constraint_info()
        per_layer_values = list(per_layer_forward.values())
        step_trace.append(
            {
                "step": step_idx + 1,
                "task_loss": scalar(task_value),
                "forward_raw": float(forward_value - cfg.forward_margin),
                "safety_raw": float(safety_value - cfg.safety_margin),
                "retain_raw": float(retain_value - cfg.retain_margin),
                "forward_mean_residual": float(forward_mean_resid),
                "forward_residual_stats": _summary_stats(per_layer_values),
                "task_grad_layer_stats": _summary_stats(list(task_layer_norms.values())),
                "forward_grad_layer_stats": _summary_stats(list(forward_layer_norms.values())),
                "constraint_info": cstate,
                "refreshed_forward_targets": refreshed,
            }
        )

    post_metrics = {
        "task_acc": evaluate(model, tokenizer, task_eval_samples, choice_token_ids, eval_cfg),
        "safety_acc": evaluate(model, tokenizer, safety_eval_samples, choice_token_ids, eval_cfg),
        "retain_acc": evaluate(model, tokenizer, retain_eval_samples, choice_token_ids, eval_cfg),
    }
    final_forward_mean, final_forward_per_layer = _forward_probe_residuals(
        model, probe_tokens, boundary_layers, forward_targets
    )

    all_scalars = []
    for entry in step_trace:
        all_scalars.extend(
            [entry["task_loss"], entry["forward_raw"], entry["safety_raw"], entry["retain_raw"], entry["forward_mean_residual"]]
        )
    finite_scalars = all(math.isfinite(value) for value in all_scalars)
    min_ratio = min(d_over_k.values()) if d_over_k else 0.0

    success_checks = {
        "finite_step_scalars": finite_scalars,
        "all_layers_have_forward_params": len(layer_counts) == num_layers,
        "all_layers_receive_forward_grad": len(saw_nonzero_forward_grad_layers) == num_layers,
        "all_layers_receive_task_grad": len(saw_nonzero_task_grad_layers) == num_layers,
        "forward_targets_refreshed": len(refresh_steps) >= 1,
        "d_over_k_above_10": min_ratio >= 10.0,
    }

    summary = {
        "config": asdict(cfg),
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "model": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "boundary_surface": "last_token_hidden",
            "probe_prompt_length": len(probe_tokens),
        },
        "layer_param_counts": layer_counts,
        "d_over_k_by_layer": d_over_k,
        "modeled_local_solve": _modeled_local_solve_stats(layer_counts, hidden_size),
        "constraint_setup": {
            "forward_constraint": "mean MSE over last-token hidden states across all adapted layers",
            "safety_constraint": "KL(student || anchor) over safety replay anchor snapshot",
            "retain_constraint": "KL(student || anchor) over retain replay anchor snapshot",
            "forward_margin": cfg.forward_margin,
            "safety_margin": cfg.safety_margin,
            "retain_margin": cfg.retain_margin,
        },
        "refresh_steps": refresh_steps,
        "final_forward_residual": {
            "mean": final_forward_mean,
            "stats": _summary_stats(list(final_forward_per_layer.values())),
        },
        "step_trace": step_trace,
        "success_checks": success_checks,
        "overall_pass": all(success_checks.values()),
    }

    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[phase-d-smoke] wrote {out_path}")
    print(
        f"[phase-d-smoke] pre task/safety/retain = "
        f"{pre_metrics['task_acc']:.3f}/{pre_metrics['safety_acc']:.3f}/{pre_metrics['retain_acc']:.3f}"
    )
    print(
        f"[phase-d-smoke] post task/safety/retain = "
        f"{post_metrics['task_acc']:.3f}/{post_metrics['safety_acc']:.3f}/{post_metrics['retain_acc']:.3f}"
    )
    print(
        f"[phase-d-smoke] min d/k = {min_ratio:.1f}, refreshed_forward_targets={refresh_steps}, "
        f"overall_pass={summary['overall_pass']}"
    )
    return summary


def parse_args() -> SmokeConfig:
    parser = argparse.ArgumentParser(description="Phase D smoke on real Qwen-MLX with all-layer LoRA.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-source", default="ag_news")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--max-train-per-task", type=int, default=16)
    parser.add_argument("--max-eval-per-task", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-steps", type=int, default=6)
    parser.add_argument("--outer-step-freq", type=int, default=3)
    parser.add_argument("--replay-size", type=int, default=4)
    parser.add_argument("--safety-eval-size", type=int, default=32)
    parser.add_argument("--retain-eval-size", type=int, default=32)
    parser.add_argument("--constraint-temperature", type=float, default=1.5)
    parser.add_argument("--forward-margin", type=float, default=0.0)
    parser.add_argument("--safety-margin", type=float, default=0.0)
    parser.add_argument("--retain-margin", type=float, default=0.0)
    parser.add_argument("--rho-p-init", type=float, default=0.5)
    parser.add_argument("--rho-p-min", type=float, default=0.5)
    parser.add_argument("--rho-p-max", type=float, default=5.0)
    parser.add_argument("--rho-p-growth", type=float, default=1.5)
    parser.add_argument("--rho-i", type=float, default=0.1)
    parser.add_argument("--rho-anchor", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--do-safety-warmup", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-lr", type=float, default=1e-4)
    parser.add_argument("--output", default=str(RESULTS_DIR / "summary.json"))
    args = parser.parse_args()
    return SmokeConfig(
        model_name=args.model_name,
        dataset_source=args.dataset_source,
        local_files_only=args.local_files_only,
        max_train_per_task=args.max_train_per_task,
        max_eval_per_task=args.max_eval_per_task,
        eval_batch_size=args.eval_batch_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        seed=args.seed,
        smoke_steps=args.smoke_steps,
        outer_step_freq=args.outer_step_freq,
        replay_size=args.replay_size,
        safety_eval_size=args.safety_eval_size,
        retain_eval_size=args.retain_eval_size,
        avbd_constraint_temperature=args.constraint_temperature,
        forward_margin=args.forward_margin,
        safety_margin=args.safety_margin,
        retain_margin=args.retain_margin,
        rho_p_init=args.rho_p_init,
        rho_p_min=args.rho_p_min,
        rho_p_max=args.rho_p_max,
        rho_p_growth=args.rho_p_growth,
        rho_i=args.rho_i,
        rho_anchor=args.rho_anchor,
        weight_decay=args.weight_decay,
        do_safety_warmup=args.do_safety_warmup,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        output=args.output,
    )


if __name__ == "__main__":
    run_smoke(parse_args())
