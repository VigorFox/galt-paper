"""MLX/MLX-LM continual-learning experiment for macOS.

This port preserves the project's prompt-format and continual-learning setup
while moving the executable path to MLX. For the LLM path it uses MLX-LM LoRA
adapters on the last N layers rather than PyTorch-specific expert surgery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

PORTABLE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
sys.path.insert(0, str(PORTABLE_ROOT / "optimizer"))
sys.path.insert(0, str(PORTABLE_ROOT / "data_utils"))

from avbd_hessian_optimizer_mlx import AVBDHessianOptimizer
from data import (
    CHOICE_LETTERS,
    DEFAULT_CATEGORIES_BY_SOURCE,
    DEFAULT_DATASET_SOURCE,
    ContinualTask,
    default_categories_for_source,
    describe_dataset_source,
    load_continual_tasks,
)
from mlx_utils import clone_flat_dict, flatten_tree, scalar, unflatten_tree


@dataclass
class ExperimentConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_source: str = DEFAULT_DATASET_SOURCE
    dataset_eval_fraction: float | None = None
    local_files_only: bool = True
    batch_size: int = 1
    eval_batch_size: int = 2
    epochs_per_task: int = 1
    lr: float = 2e-4
    ewc_lambda: float = 50.0
    fisher_batches: int = 8
    replay_size: int = 8
    max_train_per_task: int = 64
    max_eval_per_task: int = 64
    seed: int = 42
    data_seed: int = 42
    train_seed: int = 42
    max_length: int = 256
    local_head_count: int = 2
    avbd_first_task_global_only: bool = True
    avbd_warmup_global_steps: int = 4
    lora_num_layers: int = 8
    lora_rank: int = 8
    lora_scale: float = 16.0
    lora_dropout: float = 0.0
    trainable_surface: str = "all_lora"
    avbd_constraint_margin: float = 0.02
    avbd_constraint_temperature: float = 1.5
    avbd_refresh_period: int = 10
    avbd_refresh_cstr_trigger: float = 0.3
    avbd_adaptive_refresh: bool = False
    avbd_adaptive_refresh_increment: int = 2
    avbd_adaptive_refresh_max_period: int = 24
    avbd_adaptive_refresh_safe_ratio: float = 0.5
    avbd_rho_init: float = 1.0
    avbd_rho_growth: float = 1.5
    avbd_rho_max: float = 2.0
    avbd_local_head_lr: float = 1e-3
    avbd_local_global_distill_weight: float = 0.25


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def iter_batches(samples: list[dict], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def tokenize_prompt(tokenizer, prompt: str, max_length: int) -> list[int]:
    token_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    if len(token_ids) > max_length:
        token_ids = token_ids[-max_length:]
    return token_ids


def tokenize_prompts(tokenizer, prompts: list[str], max_length: int) -> list[list[int]]:
    return [tokenize_prompt(tokenizer, prompt, max_length) for prompt in prompts]


def batch_labels(samples: list[dict]) -> mx.array:
    return mx.array([sample["label"] for sample in samples], dtype=mx.int32)


def get_choice_token_ids(tokenizer) -> list[list[int]]:
    probe_prompt = "Answer: "
    probe_ids = list(tokenizer.encode(probe_prompt, add_special_tokens=False))
    token_ids = []
    for letter in CHOICE_LETTERS:
        encoded = list(tokenizer.encode(probe_prompt + letter, add_special_tokens=False))
        common_prefix = 0
        max_common = min(len(probe_ids), len(encoded))
        while common_prefix < max_common and probe_ids[common_prefix] == encoded[common_prefix]:
            common_prefix += 1
        choice_ids = encoded[common_prefix:] or list(tokenizer.encode(letter, add_special_tokens=False))
        if not choice_ids:
            raise ValueError(f"Choice token {letter!r} could not be encoded for tokenizer {tokenizer.name_or_path}.")
        token_ids.append(choice_ids)
    return token_ids


def score_choice_sequence(model, prompt_ids: list[int], choice_ids: list[int]) -> mx.array:
    full_sequence = prompt_ids + choice_ids
    if len(full_sequence) < 2:
        raise ValueError("Prompt must contain at least one token before scoring choices.")
    logits = model(mx.array([full_sequence[:-1]], dtype=mx.int32))[0]
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    prompt_length = len(prompt_ids)
    score = mx.array(0.0)
    for offset, token_id in enumerate(choice_ids):
        score = score + log_probs[prompt_length - 1 + offset, token_id]
    return score


def _all_choices_are_single_token(choice_token_ids: list[list[int]]) -> bool:
    return all(len(choice_ids) == 1 for choice_ids in choice_token_ids)


def _compute_choice_scores_from_logits(logits: mx.array, choice_token_ids: list[list[int]]) -> mx.array:
    choice_ids = mx.array([choice_ids[0] for choice_ids in choice_token_ids], dtype=mx.int32)
    last_log_probs = logits[-1] - mx.logsumexp(logits[-1], axis=-1, keepdims=True)
    return mx.take(last_log_probs, choice_ids, axis=0)


def compute_choice_scores(model, prompt_ids: list[int], choice_token_ids: list[list[int]]) -> mx.array:
    if _all_choices_are_single_token(choice_token_ids):
        if not prompt_ids:
            raise ValueError("Prompt must contain at least one token before scoring choices.")
        logits = model(mx.array([prompt_ids], dtype=mx.int32))[0]
        return _compute_choice_scores_from_logits(logits, choice_token_ids)
    return mx.stack(
        [score_choice_sequence(model, prompt_ids, choice_ids) for choice_ids in choice_token_ids],
        axis=0,
    )


def compute_choice_scores_batch(model, prompt_token_batches: list[list[int]], choice_token_ids: list[list[int]]) -> mx.array:
    return mx.stack(
        [compute_choice_scores(model, prompt_ids, choice_token_ids) for prompt_ids in prompt_token_batches],
        axis=0,
    )


def cross_entropy_from_scores(scores: mx.array, labels: mx.array) -> mx.array:
    log_probs = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    picked = mx.take_along_axis(log_probs, labels.reshape((-1, 1)), axis=-1)
    return -picked.mean()


def compute_choice_loss(
    model,
    prompt_token_batches: list[list[int]],
    labels: mx.array,
    choice_token_ids: list[list[int]],
) -> mx.array:
    scores = compute_choice_scores_batch(model, prompt_token_batches, choice_token_ids)
    return cross_entropy_from_scores(scores, labels)


def compute_choice_predictions(
    model,
    prompt_token_batches: list[list[int]],
    choice_token_ids: list[list[int]],
) -> mx.array:
    return mx.argmax(compute_choice_scores_batch(model, prompt_token_batches, choice_token_ids), axis=-1)


def compute_choice_distribution(
    model,
    prompt_token_batches: list[list[int]],
    choice_token_ids: list[list[int]],
    temperature: float,
) -> mx.array:
    scores = compute_choice_scores_batch(model, prompt_token_batches, choice_token_ids)
    scaled = scores / temperature
    return mx.softmax(scaled, axis=-1)


def compute_distillation_kl(student_scores: mx.array, teacher_probs: mx.array, temperature: float) -> mx.array:
    # Cast to float32 to avoid float16 underflow: 1e-12 → 0 in fp16, causing log(0) = -inf → NaN
    scaled_scores = student_scores.astype(mx.float32) / temperature
    student_log_probs = scaled_scores - mx.logsumexp(scaled_scores, axis=-1, keepdims=True)
    teacher_probs_f32 = teacher_probs.astype(mx.float32)
    teacher_log_probs = mx.log(mx.maximum(teacher_probs_f32, 1e-12))
    kl = mx.sum(teacher_probs_f32 * (teacher_log_probs - student_log_probs), axis=-1)
    return kl.mean() * (temperature**2)


def format_accs(accs) -> str:
    return "[" + ", ".join(f"{acc:.3f}" for acc in accs) + "]"


def sample_replay_buffer(samples: list[dict], replay_size: int, seed: int) -> list[dict]:
    items = list(samples)
    random.Random(seed).shuffle(items)
    return items[: min(replay_size, len(items))]


def clone_trainable_params(model) -> dict[str, mx.array]:
    return clone_flat_dict(flatten_tree(model.trainable_parameters()))


def restore_trainable_params(model, flat_params: dict[str, mx.array]):
    model.update(unflatten_tree(clone_flat_dict(flat_params)), strict=False)
    mx.eval(model.parameters())


def configure_trainable_surface(model, adapted_layers: list[int], trainable_surface: str):
    if trainable_surface == "all_lora":
        return
    if trainable_surface == "moe_mlp_only":
        for layer_index in adapted_layers:
            layer = model.layers[layer_index]
            layer.self_attn.freeze()
            layer.block_sparse_moe.gate.freeze()
        return
    raise ValueError(f"Unsupported MLX trainable surface: {trainable_surface!r}")


def load_model_and_tokenizer(cfg: ExperimentConfig):
    if cfg.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model, tokenizer = load(
        cfg.model_name,
        tokenizer_config={"trust_remote_code": True},
    )
    model.freeze()
    linear_to_lora_layers(
        model,
        cfg.lora_num_layers,
        {
            "rank": cfg.lora_rank,
            "scale": cfg.lora_scale,
            "dropout": cfg.lora_dropout,
        },
    )
    adapted_layers = list(range(max(0, len(model.layers) - cfg.lora_num_layers), len(model.layers)))
    configure_trainable_surface(model, adapted_layers, cfg.trainable_surface)
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.eval(model.parameters())
    return model, tokenizer, adapted_layers


def evaluate(
    model,
    tokenizer,
    samples: list[dict],
    choice_token_ids: list[list[int]],
    cfg: ExperimentConfig,
) -> float:
    correct = 0
    total = 0
    for batch in iter_batches(samples, cfg.eval_batch_size):
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch], cfg.max_length)
        labels = batch_labels(batch)
        predictions = compute_choice_predictions(model, prompt_tokens, choice_token_ids)
        correct += int(mx.sum(predictions == labels).item())
        total += len(batch)
    return correct / max(1, total)


def evaluate_all(
    model,
    tokenizer,
    tasks: list[ContinualTask],
    choice_token_ids: list[list[int]],
    cfg: ExperimentConfig,
    num_seen: int,
):
    return [
        evaluate(model, tokenizer, tasks[index].eval_samples, choice_token_ids, cfg)
        for index in range(num_seen)
    ]


def compute_replay_anchor_probs(
    model,
    tokenizer,
    replay_samples: list[dict],
    choice_token_ids: list[list[int]],
    cfg: ExperimentConfig,
) -> mx.array:
    prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in replay_samples], cfg.max_length)
    return compute_choice_distribution(
        model,
        prompt_tokens,
        choice_token_ids,
        temperature=cfg.avbd_constraint_temperature,
    )


def compute_replay_loss(
    model,
    tokenizer,
    replay_samples: list[dict],
    choice_token_ids: list[list[int]],
    cfg: ExperimentConfig,
) -> float:
    prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in replay_samples], cfg.max_length)
    labels = batch_labels(replay_samples)
    loss = compute_choice_loss(model, prompt_tokens, labels, choice_token_ids)
    return scalar(loss)


def build_task_batch_loss(choice_token_ids, cfg: ExperimentConfig):
    def loss_fn(model, batch_samples: list[dict]):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(
            tokenizer=loss_fn.tokenizer,
            prompts=[sample["prompt"] for sample in batch_samples],
            max_length=cfg.max_length,
        )
        return compute_choice_loss(model, prompt_tokens, labels, choice_token_ids)

    return loss_fn


def train_adam(
    model,
    tokenizer,
    tasks: list[ContinualTask],
    choice_token_ids: list[list[int]],
    cfg: ExperimentConfig,
):
    results = {"name": "Adam+LoRA", "accs_after_task": []}
    optimizer = optim.AdamW(learning_rate=cfg.lr, weight_decay=0.01)

    def loss_fn(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        return compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)

    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    total_steps = 0

    for task_index, task in enumerate(tasks):
        print(f"\n  [Adam] {task.name} ...")
        for epoch in range(cfg.epochs_per_task):
            for batch in tqdm(list(iter_batches(task.train_samples, cfg.batch_size)), leave=False):
                _loss, grads = loss_grad_fn(model, batch)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                total_steps += 1

        accs = evaluate_all(model, tokenizer, tasks, choice_token_ids, cfg, task_index + 1)
        results["accs_after_task"].append(accs)
        print(f"  [Adam] accs={format_accs(accs)}  avg={np.mean(accs):.3f}")

    results["total_backprop_calls"] = total_steps
    return results


def train_ewc(
    model,
    tokenizer,
    tasks: list[ContinualTask],
    choice_token_ids: list[list[int]],
    cfg: ExperimentConfig,
):
    results = {"name": "Adam+LoRA+EWC", "accs_after_task": []}
    optimizer = optim.AdamW(learning_rate=cfg.lr, weight_decay=0.01)
    fisher = {}
    anchors = {}
    total_steps = 0

    def loss_fn(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        loss = compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)
        if fisher:
            for name, param in flatten_tree(module.trainable_parameters()).items():
                if name in fisher:
                    diff = param - anchors[name]
                    loss = loss + cfg.ewc_lambda * mx.sum(fisher[name] * diff * diff)
        return loss

    loss_grad_fn = nn.value_and_grad(model, loss_fn)

    def fisher_loss(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        return compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)

    fisher_grad_fn = nn.value_and_grad(model, fisher_loss)

    for task_index, task in enumerate(tasks):
        print(f"\n  [EWC] {task.name} ...")
        for epoch in range(cfg.epochs_per_task):
            for batch in tqdm(list(iter_batches(task.train_samples, cfg.batch_size)), leave=False):
                _loss, grads = loss_grad_fn(model, batch)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                total_steps += 1

        fisher_accumulator = {}
        fisher_count = 0
        for batch in iter_batches(task.train_samples, cfg.batch_size):
            _loss, grads = fisher_grad_fn(model, batch)
            for name, grad in flatten_tree(grads).items():
                fisher_accumulator[name] = fisher_accumulator.get(name, mx.zeros_like(grad)) + grad * grad
            fisher_count += 1
            if fisher_count >= cfg.fisher_batches:
                break
        fisher = {
            name: value / max(1, fisher_count)
            for name, value in fisher_accumulator.items()
        }
        anchors = clone_trainable_params(model)

        accs = evaluate_all(model, tokenizer, tasks, choice_token_ids, cfg, task_index + 1)
        results["accs_after_task"].append(accs)
        print(f"  [EWC] accs={format_accs(accs)}  avg={np.mean(accs):.3f}")

    results["total_backprop_calls"] = total_steps
    return results


def train_avbd_hessian(
    model,
    tokenizer,
    tasks: list[ContinualTask],
    choice_token_ids: list[list[int]],
    cfg: ExperimentConfig,
):
    results = {"name": "AVBD-Hessian+LoRA", "accs_after_task": []}
    optimizer = AVBDHessianOptimizer(
        model,
        lr=cfg.lr,
        rho_init=cfg.avbd_rho_init,
        rho_max=cfg.avbd_rho_max,
        rho_growth=cfg.avbd_rho_growth,
    )

    def task_loss(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        return compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)

    task_grad_fn = nn.value_and_grad(model, task_loss)
    total_steps = 0
    constraint_infos = []

    for task_index, task in enumerate(tasks):
        print(f"\n  [AVBD-Hessian] {task.name} ...")
        for epoch in range(cfg.epochs_per_task):
            for batch in tqdm(list(iter_batches(task.train_samples, cfg.batch_size)), leave=False):
                _loss, task_grads = task_grad_fn(model, batch)
                task_grad_flat = flatten_tree(task_grads)

                for cinfo in constraint_infos:
                    def constraint_loss(module, replay_samples=cinfo["replay"], anchor_probs=cinfo["anchor_probs"]):
                        prompt_tokens = tokenize_prompts(
                            tokenizer,
                            [sample["prompt"] for sample in replay_samples],
                            cfg.max_length,
                        )
                        scores = compute_choice_scores_batch(module, prompt_tokens, choice_token_ids)
                        return compute_distillation_kl(
                            scores,
                            anchor_probs,
                            temperature=cfg.avbd_constraint_temperature,
                        )

                    constraint_grad_fn = nn.value_and_grad(model, constraint_loss)
                    kl_value, constraint_grads = constraint_grad_fn(model)
                    constraint_value = scalar(kl_value) - cfg.avbd_constraint_margin
                    optimizer.set_constraint_grads(
                        cinfo["ci"],
                        constraint_value,
                        flatten_tree(constraint_grads),
                    )

                optimizer.step(task_grad_flat)
                total_steps += 1

        accs = evaluate_all(model, tokenizer, tasks, choice_token_ids, cfg, task_index + 1)
        results["accs_after_task"].append(accs)
        print(f"  [AVBD-Hessian] accs={format_accs(accs)}  avg={np.mean(accs):.3f}")

        if task_index < len(tasks) - 1:
            replay = sample_replay_buffer(task.train_samples, cfg.replay_size, cfg.train_seed + task_index + 100)
            anchor_probs = compute_replay_anchor_probs(model, tokenizer, replay, choice_token_ids, cfg)
            ci = optimizer.add_constraint(f"retain_task_{task_index}")
            constraint_infos.append({"ci": ci, "replay": replay, "anchor_probs": anchor_probs})

    results["total_backprop_calls"] = total_steps
    results["constraint_info"] = optimizer.get_constraint_info()
    return results


def run_experiment(args):
    cfg = ExperimentConfig(
        model_name=args.model_name,
        dataset_source=args.dataset_source,
        dataset_eval_fraction=args.dataset_eval_fraction,
        local_files_only=not args.allow_online_hf_load,
        batch_size=args.batch_size,
        eval_batch_size=max(1, args.eval_batch_size),
        epochs_per_task=args.epochs_per_task,
        lr=args.lr,
        ewc_lambda=args.ewc_lambda,
        fisher_batches=args.fisher_batches,
        replay_size=args.replay_size,
        max_train_per_task=args.max_train_per_task,
        max_eval_per_task=args.max_eval_per_task,
        seed=args.seed,
        data_seed=args.seed,
        train_seed=args.seed,
        max_length=args.max_length,
        lora_num_layers=args.lora_num_layers,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        lora_dropout=args.lora_dropout,
        trainable_surface=args.trainable_surface,
        avbd_constraint_margin=args.avbd_constraint_margin,
        avbd_constraint_temperature=args.avbd_constraint_temperature,
        avbd_rho_init=args.avbd_rho_init,
        avbd_rho_growth=args.avbd_rho_growth,
        avbd_rho_max=args.avbd_rho_max,
    )
    set_seed(cfg.seed)

    categories = args.categories or default_categories_for_source(cfg.dataset_source)
    print(f"Model: {cfg.model_name}")
    print(f"Dataset: {describe_dataset_source(cfg.dataset_source)}")
    print(f"Categories: {categories}")
    print(f"Device: {mx.default_device()}")

    tasks = load_continual_tasks(
        dataset_source=cfg.dataset_source,
        categories=categories,
        max_train_per_task=cfg.max_train_per_task,
        max_eval_per_task=cfg.max_eval_per_task,
        seed=cfg.data_seed,
        eval_fraction=cfg.dataset_eval_fraction,
        local_files_only=cfg.local_files_only,
    )
    for task in tasks:
        print(f"  {task.name}  train={len(task.train_samples)}  eval={len(task.eval_samples)}")

    print("\n=== Loading MLX model ===")
    model, tokenizer, adapted_layers = load_model_and_tokenizer(cfg)
    choice_token_ids = get_choice_token_ids(tokenizer)
    print(f"Adapted layers: {adapted_layers}")

    baseline = [evaluate(model, tokenizer, task.eval_samples, choice_token_ids, cfg) for task in tasks]
    print(f"Baseline accs={format_accs(baseline)}  avg={np.mean(baseline):.3f}")
    initial_trainable = clone_trainable_params(model)

    methods = [
        ("Adam+LoRA", train_adam),
        ("Adam+LoRA+EWC", train_ewc),
        ("AVBD-Hessian+LoRA", train_avbd_hessian),
    ]
    all_results = {
        "config": vars(cfg),
        "baseline_accs": baseline,
        "methods": [],
    }

    for name, train_fn in methods:
        print("\n" + "=" * 70)
        print(f"=== {name} ===")
        print("=" * 70)
        restore_trainable_params(model, initial_trainable)
        start = time.time()
        result = train_fn(model, tokenizer, tasks, choice_token_ids, cfg)
        result["wall_time"] = time.time() - start
        all_results["methods"].append(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    return all_results


def build_parser():
    parser = argparse.ArgumentParser(description="MLX continual-learning experiment.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-source", type=str, default=DEFAULT_DATASET_SOURCE, choices=list(DEFAULT_CATEGORIES_BY_SOURCE))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--dataset-eval-fraction", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ewc-lambda", type=float, default=50.0)
    parser.add_argument("--fisher-batches", type=int, default=8)
    parser.add_argument("--replay-size", type=int, default=8)
    parser.add_argument("--max-train-per-task", type=int, default=64)
    parser.add_argument("--max-eval-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-num-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--trainable-surface", type=str, default="all_lora", choices=["all_lora", "moe_mlp_only"])
    parser.add_argument("--avbd-constraint-margin", type=float, default=0.02)
    parser.add_argument("--avbd-constraint-temperature", type=float, default=1.5)
    parser.add_argument("--avbd-rho-init", type=float, default=1.0)
    parser.add_argument("--avbd-rho-growth", type=float, default=1.5)
    parser.add_argument("--avbd-rho-max", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--allow-online-hf-load", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
