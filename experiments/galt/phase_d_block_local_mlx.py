"""Phase D block-local smoke on real Qwen-MLX.

This is the next step after `phase_d_smoke_mlx.py`. It upgrades the global
multi-constraint smoke into a block-local Gauss-Seidel-style prototype:

- group all-layer LoRA parameters into macro-blocks
- assign one AVBD-GALT optimizer per block
- use exact last-token boundary states at block exits
- mask task / safety / retain / forward gradients to the active block

This is still a smoke/validation prototype, not the final full ADMM paper
implementation. Its goal is to validate that explicit block partitioning,
per-block forward edges, and masked local solves are operational on a real
Transformer carrier.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

PORTABLE_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PORTABLE_ROOT / "optimizer"))
sys.path.insert(0, str(PORTABLE_ROOT / "data_utils"))
sys.path.insert(0, str(PORTABLE_ROOT / "experiments" / "shared_runtime"))
sys.path.insert(0, str(THIS_DIR))

from avbd_galt_optimizer_mlx import AVBDGALTOptimizer
from phase_d_smoke_mlx import (
    SmokeConfig,
    _capture_last_token_hiddens,
    _eval_constraint_value,
    _group_grad_norms_by_layer,
    _layer_param_counts,
    _load_model_all_layer_lora,
    _load_smoke_data,
    _make_forward_loss_fn,
    _make_kl_constraint_fn,
    _summary_stats,
)
from continual_runtime_mlx import (
    ExperimentConfig,
    batch_labels,
    compute_choice_loss,
    compute_replay_anchor_probs,
    evaluate,
    get_choice_token_ids,
    sample_replay_buffer,
    set_seed,
    tokenize_prompt,
    tokenize_prompts,
)
from mlx_utils import flatten_tree, scalar

RESULTS_DIR = PORTABLE_ROOT / "results" / "galt_prework" / "phase_d_block_local"
LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")


@dataclass
class BlockLocalConfig(SmokeConfig):
    block_size: int = 4
    output: str = str(RESULTS_DIR / "summary.json")


def _build_blocks(layer_counts: list[dict], block_size: int) -> list[dict]:
    blocks = []
    sorted_layers = sorted(layer_counts, key=lambda item: item["layer_index"])
    for block_index, start in enumerate(range(0, len(sorted_layers), block_size)):
        chunk = sorted_layers[start : start + block_size]
        layers = [item["layer_index"] for item in chunk]
        blocks.append(
            {
                "block_index": block_index,
                "layers": layers,
                "boundary_layer": layers[-1],
                "n_params": int(sum(item["n_params"] for item in chunk)),
                "example_names": [name for item in chunk for name in item["example_names"][:1]][:4],
            }
        )
    return blocks


def _mask_flat_to_layers(flat_map: dict[str, mx.array], layers: set[int]) -> dict[str, mx.array]:
    masked = {}
    for name, value in flat_map.items():
        match = LAYER_RE.match(name)
        if match and int(match.group(1)) in layers:
            masked[name] = value
    return masked


def _block_norm(flat_map: dict[str, mx.array]) -> float:
    total = 0.0
    for value in flat_map.values():
        total += scalar(mx.sum(value * value))
    return math.sqrt(max(0.0, total))


def _modeled_block_local_stats(blocks: list[dict], k_boundary: int) -> dict:
    works = [block["n_params"] * k_boundary for block in blocks]
    return {
        "k_boundary": k_boundary,
        "sum_d_times_k": int(sum(works)),
        "max_d_times_k": int(max(works) if works else 0),
        "parallel_gain_upper_bound": float(sum(works) / max(1, max(works) if works else 1)),
    }


def run_block_local(cfg: BlockLocalConfig) -> dict:
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
        avbd_constraint_temperature=cfg.avbd_constraint_temperature,
    )

    layer_counts = _layer_param_counts(model)
    blocks = _build_blocks(layer_counts, cfg.block_size)
    boundary_layers = [block["boundary_layer"] for block in blocks]
    hidden_size = int(model.args.hidden_size)
    d_over_k_by_block = {
        block["block_index"]: float(block["n_params"] / hidden_size)
        for block in blocks
    }

    safety_replay = sample_replay_buffer(safety_samples, min(cfg.replay_size, len(safety_samples)), cfg.seed)
    retain_replay = sample_replay_buffer(retain_samples, min(cfg.replay_size, len(retain_samples)), cfg.seed + 17)
    safety_anchor = compute_replay_anchor_probs(model, tokenizer, safety_replay, choice_token_ids, eval_cfg)
    retain_anchor = compute_replay_anchor_probs(model, tokenizer, retain_replay, choice_token_ids, eval_cfg)
    mx.eval(safety_anchor, retain_anchor)

    safety_constraint_fn = _make_kl_constraint_fn(tokenizer, safety_replay, safety_anchor, choice_token_ids, cfg)
    retain_constraint_fn = _make_kl_constraint_fn(tokenizer, retain_replay, retain_anchor, choice_token_ids, cfg)

    probe_prompt = task.train_samples[0]["prompt"]
    probe_tokens = tokenize_prompt(tokenizer, probe_prompt, cfg.max_length)
    forward_targets = _capture_last_token_hiddens(model, probe_tokens, boundary_layers)

    pre_metrics = {
        "task_acc": evaluate(model, tokenizer, task.eval_samples, choice_token_ids, eval_cfg),
        "safety_acc": evaluate(model, tokenizer, safety_samples[: cfg.safety_eval_size], choice_token_ids, eval_cfg),
        "retain_acc": evaluate(model, tokenizer, retain_samples[: cfg.retain_eval_size], choice_token_ids, eval_cfg),
    }

    block_optimizers = []
    for block in blocks:
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
        block_optimizers.append(
            {
                "block": block,
                "optimizer": optimizer,
                "forward_ci": optimizer.add_constraint(f"forward_block_{block['block_index']}"),
                "safety_ci": optimizer.add_constraint("safety_alignment"),
                "retain_ci": optimizer.add_constraint("knowledge_retain"),
            }
        )

    def task_loss(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        return compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)

    task_grad_fn = nn.value_and_grad(model, task_loss)
    safety_grad_fn = nn.value_and_grad(model, safety_constraint_fn)
    retain_grad_fn = nn.value_and_grad(model, retain_constraint_fn)

    train_samples = list(task.train_samples)
    random.Random(cfg.seed).shuffle(train_samples)
    refresh_steps = []
    step_trace = []
    saw_task_blocks = set()
    saw_forward_blocks = set()

    for step_idx in range(cfg.smoke_steps):
        batch = [train_samples[step_idx % len(train_samples)]]
        task_value_tensor, task_grads_tree = task_grad_fn(model, batch)
        flat_task_grads = flatten_tree(task_grads_tree)
        _safety_val_tensor, safety_grads_tree = safety_grad_fn(model)
        _retain_val_tensor, retain_grads_tree = retain_grad_fn(model)
        safety_grads = flatten_tree(safety_grads_tree)
        retain_grads = flatten_tree(retain_grads_tree)
        safety_value = _eval_constraint_value(model, safety_constraint_fn)
        retain_value = _eval_constraint_value(model, retain_constraint_fn)
        mx.eval(task_value_tensor)

        step_blocks = []
        for entry in block_optimizers:
            block = entry["block"]
            layer_set = set(block["layers"])
            boundary_layer = block["boundary_layer"]

            forward_fn = _make_forward_loss_fn(probe_tokens, [boundary_layer], {boundary_layer: forward_targets[boundary_layer]})
            forward_grad_fn = nn.value_and_grad(model, forward_fn)
            forward_value_tensor, forward_grads_tree = forward_grad_fn(model)
            forward_grads = flatten_tree(forward_grads_tree)
            mx.eval(forward_value_tensor)

            task_block = _mask_flat_to_layers(flat_task_grads, layer_set)
            forward_block = _mask_flat_to_layers(forward_grads, layer_set)
            safety_block = _mask_flat_to_layers(safety_grads, layer_set)
            retain_block = _mask_flat_to_layers(retain_grads, layer_set)

            task_norm = _block_norm(task_block)
            forward_norm = _block_norm(forward_block)
            safety_norm = _block_norm(safety_block)
            retain_norm = _block_norm(retain_block)
            if task_norm > 0.0:
                saw_task_blocks.add(block["block_index"])
            if forward_norm > 0.0:
                saw_forward_blocks.add(block["block_index"])

            optimizer = entry["optimizer"]
            optimizer.set_constraint_grads(
                entry["forward_ci"],
                scalar(forward_value_tensor) - cfg.forward_margin,
                forward_block,
            )
            optimizer.set_constraint_grads(
                entry["safety_ci"],
                safety_value - cfg.safety_margin,
                safety_block,
            )
            optimizer.set_constraint_grads(
                entry["retain_ci"],
                retain_value - cfg.retain_margin,
                retain_block,
            )
            optimizer.step(task_block)

            boundary_now = _capture_last_token_hiddens(model, probe_tokens, [boundary_layer])[boundary_layer]
            diff = boundary_now - forward_targets[boundary_layer]
            residual = scalar(mx.mean(diff * diff))
            block_state = optimizer.get_constraint_info()
            step_blocks.append(
                {
                    "block_index": block["block_index"],
                    "layers": block["layers"],
                    "boundary_layer": boundary_layer,
                    "task_grad_norm": task_norm,
                    "forward_grad_norm": forward_norm,
                    "safety_grad_norm": safety_norm,
                    "retain_grad_norm": retain_norm,
                    "forward_raw": scalar(forward_value_tensor) - cfg.forward_margin,
                    "forward_residual": residual,
                    "constraint_info": block_state,
                }
            )

        refreshed = False
        if (step_idx + 1) % cfg.outer_step_freq == 0:
            forward_targets = _capture_last_token_hiddens(model, probe_tokens, boundary_layers)
            refresh_steps.append(step_idx + 1)
            refreshed = True

        step_trace.append(
            {
                "step": step_idx + 1,
                "task_loss": scalar(task_value_tensor),
                "safety_raw": float(safety_value - cfg.safety_margin),
                "retain_raw": float(retain_value - cfg.retain_margin),
                "blocks": step_blocks,
                "refreshed_forward_targets": refreshed,
            }
        )

    post_metrics = {
        "task_acc": evaluate(model, tokenizer, task.eval_samples, choice_token_ids, eval_cfg),
        "safety_acc": evaluate(model, tokenizer, safety_samples[: cfg.safety_eval_size], choice_token_ids, eval_cfg),
        "retain_acc": evaluate(model, tokenizer, retain_samples[: cfg.retain_eval_size], choice_token_ids, eval_cfg),
    }
    final_targets = _capture_last_token_hiddens(model, probe_tokens, boundary_layers)
    final_block_residuals = {}
    for block in blocks:
        layer = block["boundary_layer"]
        diff = final_targets[layer] - forward_targets[layer]
        final_block_residuals[block["block_index"]] = scalar(mx.mean(diff * diff))

    all_scalars = []
    for entry in step_trace:
        all_scalars.extend([entry["task_loss"], entry["safety_raw"], entry["retain_raw"]])
        for block in entry["blocks"]:
            all_scalars.extend([block["task_grad_norm"], block["forward_grad_norm"], block["forward_raw"], block["forward_residual"]])
    success_checks = {
        "finite_step_scalars": all(math.isfinite(value) for value in all_scalars),
        "all_blocks_have_params": all(block["n_params"] > 0 for block in blocks),
        "all_blocks_receive_forward_grad": len(saw_forward_blocks) == len(blocks),
        "all_blocks_receive_task_grad": len(saw_task_blocks) == len(blocks),
        "forward_targets_refreshed": len(refresh_steps) >= 1,
        "d_over_k_above_10": min(d_over_k_by_block.values()) >= 10.0,
    }

    summary = {
        "config": asdict(cfg),
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "model": {
            "num_layers": len(layer_counts),
            "hidden_size": hidden_size,
            "boundary_surface": "last_token_hidden",
            "probe_prompt_length": len(probe_tokens),
        },
        "blocks": blocks,
        "d_over_k_by_block": d_over_k_by_block,
        "modeled_local_solve": _modeled_block_local_stats(blocks, hidden_size),
        "constraint_setup": {
            "forward_constraint": "per-block last-token boundary MSE with block-masked grads",
            "safety_constraint": "KL(student || anchor) over safety replay anchor snapshot, block-masked",
            "retain_constraint": "KL(student || anchor) over retain replay anchor snapshot, block-masked",
        },
        "refresh_steps": refresh_steps,
        "final_block_forward_residuals": final_block_residuals,
        "final_block_forward_residual_stats": _summary_stats(list(final_block_residuals.values())),
        "step_trace": step_trace,
        "success_checks": success_checks,
        "overall_pass": all(success_checks.values()),
    }

    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[phase-d-block-local] wrote {out_path}")
    print(
        f"[phase-d-block-local] pre task/safety/retain = "
        f"{pre_metrics['task_acc']:.3f}/{pre_metrics['safety_acc']:.3f}/{pre_metrics['retain_acc']:.3f}"
    )
    print(
        f"[phase-d-block-local] post task/safety/retain = "
        f"{post_metrics['task_acc']:.3f}/{post_metrics['safety_acc']:.3f}/{post_metrics['retain_acc']:.3f}"
    )
    print(
        f"[phase-d-block-local] blocks={len(blocks)} min d/k={min(d_over_k_by_block.values()):.1f} "
        f"refresh={refresh_steps} overall_pass={summary['overall_pass']}"
    )
    return summary


def parse_args() -> BlockLocalConfig:
    parser = argparse.ArgumentParser(description="Phase D block-local smoke on real Qwen-MLX.")
    parser.add_argument("--model-name", default=SmokeConfig.model_name)
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
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--output", default=str(RESULTS_DIR / "summary.json"))
    args = parser.parse_args()
    return BlockLocalConfig(
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
        block_size=args.block_size,
        output=args.output,
    )


if __name__ == "__main__":
    run_block_local(parse_args())
