"""MLX Stage-1 Safety-from-Birth experiment for macOS."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm.models.base import create_attention_mask
from tqdm import tqdm

PORTABLE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORTABLE_ROOT / "optimizer"))
sys.path.insert(0, str(PORTABLE_ROOT / "data_utils"))
sys.path.insert(0, str(PORTABLE_ROOT / "experiments" / "shared_runtime"))

from avbd_hessian_optimizer_mlx import AVBDHessianOptimizer
from data import default_categories_for_source, describe_dataset_source, load_continual_tasks, load_safety_samples
from refresh_scheduler import RefreshConfig, RefreshScheduler
from continual_runtime_mlx import (
    DEFAULT_MODEL_NAME,
    ExperimentConfig,
    batch_labels,
    clone_trainable_params,
    cross_entropy_from_scores,
    compute_choice_loss,
    compute_choice_scores_batch,
    compute_distillation_kl,
    compute_replay_anchor_probs,
    evaluate,
    evaluate_all,
    flatten_tree,
    format_accs,
    get_choice_token_ids,
    iter_batches,
    load_model_and_tokenizer,
    restore_trainable_params,
    sample_replay_buffer,
    set_seed,
    tokenize_prompts,
    unflatten_tree,
)


def evaluate_all_with_safety(model, tokenizer, tasks, safety_samples, choice_token_ids, cfg, num_seen):
    task_accs = evaluate_all(model, tokenizer, tasks, choice_token_ids, cfg, num_seen)
    safety_acc = evaluate(model, tokenizer, safety_samples, choice_token_ids, cfg)
    return task_accs, safety_acc


def _clip_grad_norm(grads, max_norm: float = 1.0):
    """Clip gradient tree by global L2 norm, preserving tree structure."""
    from mlx.utils import tree_flatten, tree_map
    leaves = tree_flatten(grads)
    total_sq = mx.array(0.0)
    for _, v in leaves:
        total_sq = total_sq + mx.sum(v * v)
    total_norm = float(mx.sqrt(total_sq).item())
    if total_norm > max_norm:
        scale = mx.array(max_norm / (total_norm + 1e-8))
        return tree_map(lambda g: g * scale, grads), total_norm
    return grads, total_norm


def _check_model_nan(model, tokenizer, probe_sample, choice_token_ids, cfg) -> bool:
    """Quick NaN check on a single sample's model output."""
    tokens = tokenize_prompts(tokenizer, [probe_sample["prompt"]], cfg.max_length)
    scores = compute_choice_scores_batch(model, tokens, choice_token_ids)
    return bool(mx.any(mx.isnan(scores)))


def safety_warmup(model, tokenizer, safety_samples, choice_token_ids, cfg,
                  warmup_epochs=5, warmup_lr=1e-4, max_grad_norm=1.0):
    optimizer = optim.AdamW(learning_rate=warmup_lr, weight_decay=0.01)

    def loss_fn(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        return compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)

    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    safety_acc = evaluate(model, tokenizer, safety_samples, choice_token_ids, cfg)
    best_params = clone_trainable_params(model)
    best_acc = safety_acc

    for epoch in range(warmup_epochs):
        shuffled = list(safety_samples)
        random.shuffle(shuffled)
        for batch in iter_batches(shuffled, max(cfg.batch_size, 2)):
            _loss, grads = loss_grad_fn(model, batch)
            grads, grad_norm = _clip_grad_norm(grads, max_grad_norm)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        # NaN detection — restore best checkpoint if diverged
        if _check_model_nan(model, tokenizer, safety_samples[0], choice_token_ids, cfg):
            print(f"  [Safety Warmup] epoch {epoch + 1}/{warmup_epochs}: NaN detected, restoring best checkpoint")
            restore_trainable_params(model, best_params)
            mx.eval(model.parameters())
            break

        safety_acc = evaluate(model, tokenizer, safety_samples, choice_token_ids, cfg)
        print(f"  [Safety Warmup] epoch {epoch + 1}/{warmup_epochs}: safety_acc = {safety_acc:.3f}")
        if safety_acc >= best_acc:
            best_acc = safety_acc
            best_params = clone_trainable_params(model)

    return best_acc


class LocalChoiceHead(nn.Module):
    def __init__(self, hidden_size: int, num_choices: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_choices)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.linear(hidden)


class LocalHeadStack(nn.Module):
    def __init__(self, hidden_size: int, num_choices: int, head_count: int):
        super().__init__()
        self.heads = [LocalChoiceHead(hidden_size, num_choices) for _ in range(head_count)]


class SFBLocalHeadSystem(nn.Module):
    def __init__(self, model, selected_layers: list[int], hidden_size: int, num_choices: int):
        super().__init__()
        self.model = model
        self.selected_layers = list(selected_layers)
        self.local_heads = LocalHeadStack(hidden_size, num_choices, len(self.selected_layers))


def select_local_layers(candidate_layers: list[int], head_count: int) -> list[int]:
    if not candidate_layers:
        return []
    if head_count <= 1:
        return [candidate_layers[-1]]
    if head_count >= len(candidate_layers):
        return list(candidate_layers)
    positions = {
        round(index * (len(candidate_layers) - 1) / max(1, head_count - 1))
        for index in range(head_count)
    }
    return [candidate_layers[index] for index in sorted(positions)]


def build_local_head_system(model, adapted_layers: list[int], head_count: int, num_choices: int):
    hidden_size = int(model.args.hidden_size)
    selected_layers = select_local_layers(adapted_layers, head_count)
    system = SFBLocalHeadSystem(model, selected_layers, hidden_size, num_choices)
    mx.eval(system.parameters())
    return system, selected_layers


def _extract_selected_layer_hiddens(model, prompt_ids: list[int], selected_layers: list[int]) -> dict[int, mx.array]:
    if not selected_layers:
        return {}
    core = model.model if hasattr(model, "model") else model
    inputs = mx.array([prompt_ids], dtype=mx.int32)
    hidden = core.embed_tokens(inputs)
    mask = create_attention_mask(hidden, None)
    selected = set(selected_layers)
    captured = {}
    for layer_index, layer in enumerate(core.layers):
        hidden = layer(hidden, mask, None)
        if layer_index in selected:
            captured[layer_index] = hidden[:, -1, :]
    return captured


def compute_local_losses(
    system: SFBLocalHeadSystem,
    prompt_token_batches: list[list[int]],
    labels: mx.array,
    teacher_probs: mx.array | None = None,
    distill_weight: float = 0.0,
    distill_temperature: float = 1.0,
):
    if not system.selected_layers:
        return []
    per_layer_hidden: dict[int, list[mx.array]] = {layer: [] for layer in system.selected_layers}
    for prompt_ids in prompt_token_batches:
        hidden_map = _extract_selected_layer_hiddens(system.model, prompt_ids, system.selected_layers)
        for layer in system.selected_layers:
            per_layer_hidden[layer].append(hidden_map[layer])

    losses = []
    for head_index, layer in enumerate(system.selected_layers):
        hidden_batch = mx.concatenate(per_layer_hidden[layer], axis=0)
        layer_logits = system.local_heads.heads[head_index](hidden_batch)
        layer_loss = cross_entropy_from_scores(layer_logits, labels)
        if teacher_probs is not None and distill_weight > 0.0:
            layer_loss = layer_loss + distill_weight * compute_distillation_kl(
                layer_logits,
                teacher_probs,
                temperature=distill_temperature,
            )
        losses.append(layer_loss)
    return losses


def partition_system_grads(flat_grads: dict[str, mx.array]):
    model_grads = {}
    local_head_grads = {}
    for name, grad in flat_grads.items():
        if name.startswith("model."):
            model_grads[name[len("model."):]] = grad
        elif name.startswith("local_heads."):
            local_head_grads[name[len("local_heads."):]] = grad
    return model_grads, local_head_grads


def compute_forbid_lse(model, prompt_token_batches, choice_token_ids, forbid_mask: mx.array, lse_temperature: float = 0.1) -> mx.array:
    """v16: LSE soft-max of per-prompt forbidden probability mass.

    forbid_mask: [N, K] float — 1.0 for forbidden choice index, 0.0 otherwise.
    Returns scalar = T·log(sum_p exp(v_p / T)) where v_p = sum_{i: mask=1} π_θ(i|p).
    As T → 0+, the value approaches max_p v_p — "the worst prompt drives the constraint".
    """
    scores = compute_choice_scores_batch(model, prompt_token_batches, choice_token_ids)
    scores_f32 = scores.astype(mx.float32)
    probs = mx.softmax(scores_f32, axis=-1)
    v = mx.sum(probs * forbid_mask.astype(mx.float32), axis=-1)
    T = float(lse_temperature)
    if T <= 0.0:
        return mx.max(v)
    return T * mx.logsumexp(v / T, axis=0)


def _eval_sfb_constraint_states(model, tokenizer, constraint_infos, choice_token_ids, cfg):
    raw_values = []
    violations = []
    for cinfo in constraint_infos:
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in cinfo["replay"]], cfg.max_length)
        ctype = cinfo.get("type", "kl")
        if ctype == "forbid_lse":
            scalar = compute_forbid_lse(
                model, prompt_tokens, choice_token_ids,
                cinfo["forbid_mask"],
                lse_temperature=cinfo.get("lse_temperature", 0.1),
            )
            margin = cinfo.get("margin", cfg.avbd_constraint_margin)
            raw_value = float(scalar.item()) - margin
        else:
            scores = compute_choice_scores_batch(model, prompt_tokens, choice_token_ids)
            kl = compute_distillation_kl(
                scores,
                cinfo["anchor_probs"],
                temperature=cfg.avbd_constraint_temperature,
            )
            raw_value = float(kl.item()) - cfg.avbd_constraint_margin
        raw_values.append(raw_value)
        violations.append(max(0.0, raw_value))
    return raw_values, violations


def _eval_sfb_constraint_state_maps(model, tokenizer, constraint_infos, choice_token_ids, cfg):
    raw_values, violations = _eval_sfb_constraint_states(model, tokenizer, constraint_infos, choice_token_ids, cfg)
    raw_value_map = {}
    violation_map = {}
    for cinfo, raw_value, violation in zip(constraint_infos, raw_values, violations):
        raw_value_map[cinfo["ci"]] = raw_value
        violation_map[cinfo["ci"]] = violation
    return raw_value_map, violation_map


def _select_active_constraint_ids(
    constraint_infos,
    raw_value_map: dict[int, float],
    prev_raw_value_map: dict[int, float],
    stale_steps: dict[int, int],
    constraint_state: dict,
    cfg,
):
    active_limit = int(getattr(cfg, "avbd_active_constraint_limit", 0))
    if (
        not getattr(cfg, "avbd_use_active_set", True)
        or active_limit <= 0
        or len(constraint_infos) <= active_limit
    ):
        return {cinfo["ci"] for cinfo in constraint_infos}

    selected = set()
    if getattr(cfg, "avbd_active_keep_safety", True):
        for cinfo in constraint_infos:
            if cinfo["name"] == "safety_alignment":
                selected.add(cinfo["ci"])
                break

    scored = []
    for cinfo in constraint_infos:
        ci = cinfo["ci"]
        if ci in selected:
            continue
        raw_value = raw_value_map.get(ci, 0.0)
        prev_raw = prev_raw_value_map.get(ci, raw_value)
        lambda_value = float(constraint_state.get(cinfo["name"], {}).get("lambda_", 0.0))
        growth = max(0.0, raw_value - prev_raw)
        stale_bonus = getattr(cfg, "avbd_active_stale_weight", 0.0) * min(
            1.0,
            stale_steps.get(ci, 0) / max(1, int(getattr(cfg, "avbd_active_stale_horizon", 1))),
        )
        score = (
            max(0.0, raw_value)
            + getattr(cfg, "avbd_active_lambda_weight", 0.0) * lambda_value
            + getattr(cfg, "avbd_active_growth_weight", 0.0) * growth
            + stale_bonus
        )
        scored.append((score, ci))

    scored.sort(key=lambda item: item[0], reverse=True)
    remaining = max(0, active_limit - len(selected))
    for _score, ci in scored[:remaining]:
        selected.add(ci)

    if not selected and constraint_infos:
        selected.add(constraint_infos[0]["ci"])
    return selected


def _update_constraint_staleness(constraint_infos, active_constraint_ids: set[int], stale_steps: dict[int, int]):
    next_stale_steps = {}
    for cinfo in constraint_infos:
        ci = cinfo["ci"]
        if ci in active_constraint_ids:
            next_stale_steps[ci] = 0
        else:
            next_stale_steps[ci] = stale_steps.get(ci, 0) + 1
    return next_stale_steps


def _flatten_grad_vector(grad_map: dict[str, mx.array]) -> mx.array | None:
    parts = [grad.reshape((-1,)) for _name, grad in sorted(grad_map.items())]
    if not parts:
        return None
    return mx.concatenate(parts, axis=0)


def _cosine_similarity(lhs: mx.array, rhs: mx.array) -> float:
    numerator = mx.sum(lhs * rhs)
    denominator = mx.sqrt(mx.sum(lhs * lhs) * mx.sum(rhs * rhs)) + 1e-12
    return float((numerator / denominator).item())


def _compute_sfb_hessian_constraint_grads(model, tokenizer, constraint_infos, choice_token_ids, cfg):
    constraint_grads = {}
    constraint_vectors = {}
    all_j_flat = []
    for cinfo in constraint_infos:
        replay_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in cinfo["replay"]], cfg.max_length)
        ctype = cinfo.get("type", "kl")

        if ctype == "forbid_lse":
            forbid_mask = cinfo["forbid_mask"]
            lse_T = cinfo.get("lse_temperature", 0.1)

            def constraint_loss(module, token_batches=replay_tokens, mask=forbid_mask, T=lse_T):
                return compute_forbid_lse(module, token_batches, choice_token_ids, mask, lse_temperature=T)
        else:
            def constraint_loss(module, token_batches=replay_tokens, anchor_probs=cinfo["anchor_probs"]):
                scores = compute_choice_scores_batch(module, token_batches, choice_token_ids)
                return compute_distillation_kl(
                    scores,
                    anchor_probs,
                    temperature=cfg.avbd_constraint_temperature,
                )

        constraint_grad_fn = nn.value_and_grad(model, constraint_loss)
        _loss, grads = constraint_grad_fn(model)
        flat_grads = flatten_tree(grads)
        constraint_grads[cinfo["ci"]] = flat_grads
        flat_vector = _flatten_grad_vector(flat_grads)
        if flat_vector is not None:
            constraint_vectors[cinfo["ci"]] = flat_vector
            all_j_flat.append(flat_vector)
    return constraint_grads, constraint_vectors, all_j_flat


def _compute_constraint_event_risk(
    model_grads: dict[str, mx.array],
    active_constraint_ids: set[int],
    cached_constraint_vectors: dict[int, mx.array],
):
    model_vector = _flatten_grad_vector(model_grads)
    if model_vector is None:
        return 0.0, {}
    risk_by_constraint = {}
    for ci in active_constraint_ids:
        constraint_vector = cached_constraint_vectors.get(ci)
        if constraint_vector is None:
            continue
        risk_by_constraint[ci] = abs(_cosine_similarity(model_vector, constraint_vector))
    return max(risk_by_constraint.values(), default=0.0), risk_by_constraint


def train_sfb_avbd_hessian(model, tokenizer, tasks, safety_samples, choice_token_ids, cfg):
    results = {"name": "SfB-AVBD-Hessian", "accs_after_task": [], "safety_after_task": []}
    optimizer = AVBDHessianOptimizer(
        model,
        lr=cfg.lr,
        rho_init=cfg.avbd_rho_init,
        rho_max=cfg.avbd_rho_max,
        rho_growth=cfg.avbd_rho_growth,
        use_multi_constraint_woodbury=getattr(cfg, "avbd_use_multi_constraint_woodbury", True),
    )

    safety_replay = sample_replay_buffer(safety_samples, min(cfg.replay_size, len(safety_samples)), cfg.train_seed)
    safety_anchor = compute_replay_anchor_probs(model, tokenizer, safety_replay, choice_token_ids, cfg)
    safety_ci = optimizer.add_constraint("safety_alignment")
    constraint_infos = [
        {
            "ci": safety_ci,
            "replay": safety_replay,
            "anchor_probs": safety_anchor,
            "name": "safety_alignment",
        }
    ]
    print(f"  [SfB-AVBD-Hessian] Safety anchor frozen ({len(safety_replay)} replay samples)")

    def task_loss(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        return compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)

    task_grad_fn = nn.value_and_grad(model, task_loss)
    total_steps = 0
    prev_raw_value_map = {}
    stale_steps = {}
    active_set_sizes = []

    for task_index, task in enumerate(tasks):
        print(f"\n  [SfB-AVBD-Hessian] {task.name} ({len(constraint_infos)} constraints) ...")
        for epoch in range(cfg.epochs_per_task):
            for batch in tqdm(list(iter_batches(task.train_samples, cfg.batch_size)), leave=False):
                _loss, task_grads = task_grad_fn(model, batch)
                raw_value_map, _violation_map = _eval_sfb_constraint_state_maps(
                    model,
                    tokenizer,
                    constraint_infos,
                    choice_token_ids,
                    cfg,
                )
                constraint_state = optimizer.get_constraint_info()
                active_constraint_ids = _select_active_constraint_ids(
                    constraint_infos,
                    raw_value_map,
                    prev_raw_value_map,
                    stale_steps,
                    constraint_state,
                    cfg,
                )
                stale_steps = _update_constraint_staleness(constraint_infos, active_constraint_ids, stale_steps)
                prev_raw_value_map = dict(raw_value_map)
                active_constraint_infos = [cinfo for cinfo in constraint_infos if cinfo["ci"] in active_constraint_ids]
                active_constraint_grads, _active_vectors, _all_j_flat = _compute_sfb_hessian_constraint_grads(
                    model,
                    tokenizer,
                    active_constraint_infos,
                    choice_token_ids,
                    cfg,
                )
                active_set_sizes.append(len(active_constraint_ids))
                for cinfo in constraint_infos:
                    optimizer.set_constraint_grads(
                        cinfo["ci"],
                        raw_value_map.get(cinfo["ci"], 0.0),
                        active_constraint_grads.get(cinfo["ci"], {}),
                    )

                optimizer.step(flatten_tree(task_grads))
                total_steps += 1

        task_accs, safety_acc = evaluate_all_with_safety(
            model,
            tokenizer,
            tasks,
            safety_samples,
            choice_token_ids,
            cfg,
            task_index + 1,
        )
        results["accs_after_task"].append(task_accs)
        results["safety_after_task"].append(safety_acc)
        print(
            f"  [SfB-AVBD-Hessian] accs={format_accs(task_accs)}  "
            f"avg={np.mean(task_accs):.3f}  safety={safety_acc:.3f}"
        )

        if task_index < len(tasks) - 1:
            replay = sample_replay_buffer(task.train_samples, cfg.replay_size, cfg.train_seed + task_index + 100)
            anchor_probs = compute_replay_anchor_probs(model, tokenizer, replay, choice_token_ids, cfg)
            ci = optimizer.add_constraint(f"retain_task_{task_index}")
            constraint_infos.append(
                {
                    "ci": ci,
                    "replay": replay,
                    "anchor_probs": anchor_probs,
                    "name": f"retain_task_{task_index}",
                }
            )

    results["total_backprop_calls"] = total_steps
    results["constraint_info"] = optimizer.get_constraint_info()
    if active_set_sizes:
        results["active_set_stats"] = {
            "avg_active_constraints": float(np.mean(active_set_sizes)),
            "max_active_constraints": int(max(active_set_sizes)),
        }
    return results


def train_sfb_avbd_hessian_lowbp(
    model,
    tokenizer,
    tasks,
    safety_samples,
    choice_token_ids,
    cfg,
    adapted_layers: list[int],
):
    results = {"name": "SfB-AVBD-Hessian-LowBP", "accs_after_task": [], "safety_after_task": []}
    system, selected_layers = build_local_head_system(
        model,
        adapted_layers,
        cfg.local_head_count,
        len(choice_token_ids),
    )
    optimizer = AVBDHessianOptimizer(
        system.model,
        lr=cfg.lr,
        rho_init=cfg.avbd_rho_init,
        rho_max=cfg.avbd_rho_max,
        rho_growth=cfg.avbd_rho_growth,
        use_multi_constraint_woodbury=getattr(cfg, "avbd_use_multi_constraint_woodbury", True),
    )
    scheduler = RefreshScheduler(
        RefreshConfig(
            refresh_period=cfg.avbd_refresh_period,
            refresh_cstr_trigger=cfg.avbd_refresh_cstr_trigger,
            adaptive_refresh=cfg.avbd_adaptive_refresh,
            adaptive_refresh_increment=cfg.avbd_adaptive_refresh_increment,
            adaptive_refresh_max_period=cfg.avbd_adaptive_refresh_max_period,
            adaptive_refresh_safe_ratio=cfg.avbd_adaptive_refresh_safe_ratio,
        )
    )
    local_head_optimizer = optim.Adam(learning_rate=cfg.avbd_local_head_lr)

    safety_replay = sample_replay_buffer(safety_samples, min(cfg.replay_size, len(safety_samples)), cfg.train_seed)
    safety_anchor = compute_replay_anchor_probs(system.model, tokenizer, safety_replay, choice_token_ids, cfg)
    safety_ci = optimizer.add_constraint("safety_alignment")
    constraint_infos = [
        {
            "ci": safety_ci,
            "replay": safety_replay,
            "anchor_probs": safety_anchor,
            "name": "safety_alignment",
        }
    ]
    print(f"  [SfB-AVBD-Hessian-LowBP] Safety anchor frozen ({len(safety_replay)} replay samples)")

    def global_system_loss(system_module: SFBLocalHeadSystem, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        final_scores = compute_choice_scores_batch(system_module.model, prompt_tokens, choice_token_ids)
        total_loss = cross_entropy_from_scores(final_scores, labels)
        teacher_probs = mx.softmax(final_scores / cfg.avbd_constraint_temperature, axis=-1)
        local_losses = compute_local_losses(
            system_module,
            prompt_tokens,
            labels,
            teacher_probs=teacher_probs,
            distill_weight=cfg.avbd_local_global_distill_weight,
            distill_temperature=cfg.avbd_constraint_temperature,
        )
        if local_losses:
            total_loss = total_loss + mx.stack(local_losses).sum()
        return total_loss

    def local_system_loss(system_module: SFBLocalHeadSystem, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        local_losses = compute_local_losses(system_module, prompt_tokens, labels)
        if not local_losses:
            return mx.array(0.0)
        return mx.stack(local_losses).sum()

    global_grad_fn = nn.value_and_grad(system, global_system_loss)
    local_grad_fn = nn.value_and_grad(system, local_system_loss)
    cached_constraint_grads: dict[int, dict[str, mx.array]] = {}
    cached_constraint_vectors: dict[int, mx.array] = {}
    raw_value_map: dict[int, float] = {}
    violation_map: dict[int, float] = {}
    prev_raw_value_map: dict[int, float] = {}
    stale_steps: dict[int, int] = {}
    active_constraint_ids: set[int] = set()
    cos_theta_log = []
    total_steps = 0
    active_set_sizes = []
    event_refresh_count = 0
    scheduler_refresh_count = 0
    forced_refresh_count = 0
    max_event_risk = 0.0

    for task_index, task in enumerate(tasks):
        print(
            f"\n  [SfB-AVBD-Hessian-LowBP] {task.name} ({len(constraint_infos)} constraints, "
            f"local_layers={selected_layers}) ..."
        )
        task_step = 0
        for _epoch in range(cfg.epochs_per_task):
            for batch in tqdm(list(iter_batches(task.train_samples, cfg.batch_size)), leave=False):
                missing_constraint_cache = any(cinfo["ci"] not in raw_value_map for cinfo in constraint_infos)
                missing_active_grads = any(ci not in cached_constraint_grads for ci in active_constraint_ids)
                force_global = (
                    missing_constraint_cache
                    or (active_constraint_ids and missing_active_grads)
                    or not active_constraint_ids
                )
                if task_index == 0 and cfg.avbd_first_task_global_only:
                    force_global = True
                elif task_step < cfg.avbd_warmup_global_steps:
                    force_global = True

                local_model_grads = {}
                local_head_grads = {}
                event_risk = 0.0
                did_constraint_refresh = False
                if force_global:
                    is_global = True
                    forced_refresh_count += 1
                else:
                    _loss, local_grads = local_grad_fn(system, batch)
                    local_model_grads, local_head_grads = partition_system_grads(flatten_tree(local_grads))
                    scheduler_trigger = scheduler.needs_refresh(list(violation_map.values()))
                    event_trigger = False
                    if getattr(cfg, "avbd_use_event_refresh", True):
                        event_risk, _proxy_risks = _compute_constraint_event_risk(
                            local_model_grads,
                            active_constraint_ids,
                            cached_constraint_vectors,
                        )
                        max_event_risk = max(max_event_risk, event_risk)
                        event_trigger = event_risk >= getattr(cfg, "avbd_event_cos_threshold", 1.0)
                    is_global = scheduler_trigger
                    if scheduler_trigger:
                        scheduler_refresh_count += 1
                    if (not scheduler_trigger) and event_trigger:
                        event_refresh_count += 1
                        prev_raw_value_map = dict(raw_value_map)
                        raw_value_map, violation_map = _eval_sfb_constraint_state_maps(
                            system.model,
                            tokenizer,
                            constraint_infos,
                            choice_token_ids,
                            cfg,
                        )
                        constraint_state = optimizer.get_constraint_info()
                        active_constraint_ids = _select_active_constraint_ids(
                            constraint_infos,
                            raw_value_map,
                            prev_raw_value_map,
                            stale_steps,
                            constraint_state,
                            cfg,
                        )
                        stale_steps = _update_constraint_staleness(constraint_infos, active_constraint_ids, stale_steps)
                        active_constraint_infos = [cinfo for cinfo in constraint_infos if cinfo["ci"] in active_constraint_ids]
                        cached_constraint_grads, cached_constraint_vectors, _all_j_flat = _compute_sfb_hessian_constraint_grads(
                            system.model,
                            tokenizer,
                            active_constraint_infos,
                            choice_token_ids,
                            cfg,
                        )
                        active_set_sizes.append(len(active_constraint_ids))
                        did_constraint_refresh = True

                if is_global:
                    _loss, grads = global_grad_fn(system, batch)
                    model_grads, local_head_grads = partition_system_grads(flatten_tree(grads))
                    prev_raw_value_map = dict(raw_value_map)
                    raw_value_map, violation_map = _eval_sfb_constraint_state_maps(
                        system.model,
                        tokenizer,
                        constraint_infos,
                        choice_token_ids,
                        cfg,
                    )
                    constraint_state = optimizer.get_constraint_info()
                    active_constraint_ids = _select_active_constraint_ids(
                        constraint_infos,
                        raw_value_map,
                        prev_raw_value_map,
                        stale_steps,
                        constraint_state,
                        cfg,
                    )
                    stale_steps = _update_constraint_staleness(constraint_infos, active_constraint_ids, stale_steps)
                    active_constraint_infos = [cinfo for cinfo in constraint_infos if cinfo["ci"] in active_constraint_ids]
                    cached_constraint_grads, cached_constraint_vectors, all_j_flat = _compute_sfb_hessian_constraint_grads(
                        system.model,
                        tokenizer,
                        active_constraint_infos,
                        choice_token_ids,
                        cfg,
                    )
                    active_set_sizes.append(len(active_constraint_ids))
                    if len(all_j_flat) >= 2 and total_steps % 32 == 0:
                        cos_pairs = {}
                        for i in range(len(all_j_flat)):
                            for j in range(i + 1, len(all_j_flat)):
                                cos_pairs[f"{i}-{j}"] = round(_cosine_similarity(all_j_flat[i], all_j_flat[j]), 4)
                        cos_theta_log.append(
                            {
                                "step": total_steps,
                                "task": task_index,
                                "n_constraints": len(all_j_flat),
                                "cos": cos_pairs,
                            }
                        )
                else:
                    model_grads = local_model_grads

                if local_head_grads:
                    local_head_optimizer.update(system.local_heads, unflatten_tree(local_head_grads))
                for cinfo in constraint_infos:
                    ci = cinfo["ci"]
                    optimizer.set_constraint_grads(
                        ci,
                        raw_value_map.get(ci, 0.0),
                        cached_constraint_grads.get(ci, {}) if ci in active_constraint_ids else {},
                        update_dual=is_global and not did_constraint_refresh,
                    )
                optimizer.step(model_grads)
                scheduler.mark_step(is_global)
                total_steps += 1
                task_step += 1
                mx.eval(system.parameters(), local_head_optimizer.state)

        task_accs, safety_acc = evaluate_all_with_safety(
            system.model,
            tokenizer,
            tasks,
            safety_samples,
            choice_token_ids,
            cfg,
            task_index + 1,
        )
        results["accs_after_task"].append(task_accs)
        results["safety_after_task"].append(safety_acc)
        constraint_state = optimizer.get_constraint_info()
        scheduler_stats = scheduler.stats()
        print(
            f"  [SfB-AVBD-Hessian-LowBP] accs={format_accs(task_accs)}  "
            f"avg={np.mean(task_accs):.3f}  safety={safety_acc:.3f}"
        )
        print(
            f"         global={scheduler_stats['global_backprop_calls']}  "
            f"local={scheduler_stats['local_only_steps']}"
        )
        cstr_strs = []
        for cinfo in constraint_infos:
            ci_info = constraint_state.get(cinfo["name"], {})
            cstr_strs.append(f"{cinfo['name']}:λ={ci_info.get('lambda_', 0):.3f}")
        print(f"         constraints: {', '.join(cstr_strs)}")

        if task_index < len(tasks) - 1:
            replay = sample_replay_buffer(task.train_samples, cfg.replay_size, cfg.train_seed + task_index + 100)
            anchor_probs = compute_replay_anchor_probs(system.model, tokenizer, replay, choice_token_ids, cfg)
            ci = optimizer.add_constraint(f"retain_task_{task_index}")
            constraint_infos.append(
                {
                    "ci": ci,
                    "replay": replay,
                    "anchor_probs": anchor_probs,
                    "name": f"retain_task_{task_index}",
                }
            )

    results["scheduler_stats"] = scheduler.stats()
    results["total_backprop_calls"] = results["scheduler_stats"]["global_backprop_calls"]
    results["total_optimizer_steps"] = total_steps
    results["constraint_info"] = optimizer.get_constraint_info()
    results["final_lambda_safety"] = float(results["constraint_info"]["safety_alignment"]["lambda_"])
    results["final_rho_safety"] = float(results["constraint_info"]["safety_alignment"]["rho"])
    if active_set_sizes:
        results["active_set_stats"] = {
            "avg_active_constraints": float(np.mean(active_set_sizes)),
            "max_active_constraints": int(max(active_set_sizes)),
        }
    results["refresh_stats"] = {
        "forced_global_refreshes": forced_refresh_count,
        "scheduler_triggered_refreshes": scheduler_refresh_count,
        "event_triggered_refreshes": event_refresh_count,
        "max_event_risk": max_event_risk,
    }
    results["cos_theta_log"] = cos_theta_log
    if cos_theta_log:
        all_cos = [value for entry in cos_theta_log for value in entry["cos"].values()]
        results["avg_cos_theta"] = float(np.mean(all_cos)) if all_cos else 0.0
        print(
            "  [SfB-AVBD-Hessian-LowBP] "
            f"avg cos_theta across constraints: {results['avg_cos_theta']:.4f}"
        )
    return results


def train_sfb_kl(model, tokenizer, tasks, safety_samples, choice_token_ids, cfg, beta_kl: float):
    results = {"name": f"SfB-KL(beta={beta_kl})", "accs_after_task": [], "safety_after_task": []}
    optimizer = optim.AdamW(learning_rate=cfg.lr, weight_decay=0.01)

    safety_replay = sample_replay_buffer(safety_samples, min(cfg.replay_size, len(safety_samples)), cfg.train_seed)
    safety_anchor = compute_replay_anchor_probs(model, tokenizer, safety_replay, choice_token_ids, cfg)

    def loss_fn(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        task_loss = compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)
        safety_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in safety_replay], cfg.max_length)
        safety_scores = compute_choice_scores_batch(module, safety_tokens, choice_token_ids)
        safety_kl = compute_distillation_kl(
            safety_scores,
            safety_anchor,
            temperature=cfg.avbd_constraint_temperature,
        )
        return task_loss + beta_kl * safety_kl

    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    total_steps = 0

    for task_index, task in enumerate(tasks):
        print(f"\n  [SfB-KL] {task.name} ...")
        for epoch in range(cfg.epochs_per_task):
            for batch in tqdm(list(iter_batches(task.train_samples, cfg.batch_size)), leave=False):
                _loss, grads = loss_grad_fn(model, batch)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                total_steps += 1

        task_accs, safety_acc = evaluate_all_with_safety(
            model,
            tokenizer,
            tasks,
            safety_samples,
            choice_token_ids,
            cfg,
            task_index + 1,
        )
        results["accs_after_task"].append(task_accs)
        results["safety_after_task"].append(safety_acc)
        print(f"  [SfB-KL] accs={format_accs(task_accs)}  avg={np.mean(task_accs):.3f}  safety={safety_acc:.3f}")

    results["total_backprop_calls"] = total_steps
    return results


def train_posthoc_adam(model, tokenizer, tasks, safety_samples, choice_token_ids, cfg):
    results = {"name": "PostHoc-Adam", "accs_after_task": [], "safety_after_task": []}
    optimizer = optim.AdamW(learning_rate=cfg.lr, weight_decay=0.01)

    def loss_fn(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        return compute_choice_loss(module, prompt_tokens, labels, choice_token_ids)

    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    total_steps = 0

    for task_index, task in enumerate(tasks):
        print(f"\n  [PostHoc-Adam] {task.name} ...")
        for epoch in range(cfg.epochs_per_task):
            for batch in tqdm(list(iter_batches(task.train_samples, cfg.batch_size)), leave=False):
                _loss, grads = loss_grad_fn(model, batch)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                total_steps += 1

        task_accs, safety_acc = evaluate_all_with_safety(
            model,
            tokenizer,
            tasks,
            safety_samples,
            choice_token_ids,
            cfg,
            task_index + 1,
        )
        results["accs_after_task"].append(task_accs)
        results["safety_after_task"].append(safety_acc)
        print(
            f"  [PostHoc-Adam] accs={format_accs(task_accs)}  "
            f"avg={np.mean(task_accs):.3f}  safety={safety_acc:.3f}"
        )

    results["total_backprop_calls"] = total_steps
    return results


def run_experiment(args):
    cfg = ExperimentConfig(
        model_name=args.model_name,
        dataset_source=args.dataset_source,
        local_files_only=not args.allow_online_hf_load,
        batch_size=args.batch_size,
        eval_batch_size=max(1, args.eval_batch_size),
        epochs_per_task=args.epochs_per_task,
        lr=args.lr,
        replay_size=args.replay_size,
        max_train_per_task=args.max_train_per_task,
        max_eval_per_task=args.max_eval_per_task,
        seed=args.seed,
        data_seed=args.seed,
        train_seed=args.seed,
        max_length=args.max_length,
        local_head_count=args.local_head_count,
        avbd_first_task_global_only=args.avbd_first_task_global_only,
        avbd_warmup_global_steps=args.avbd_warmup_global_steps,
        lora_num_layers=args.lora_num_layers,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        lora_dropout=args.lora_dropout,
        trainable_surface=args.trainable_surface,
        avbd_constraint_margin=args.avbd_constraint_margin,
        avbd_constraint_temperature=args.avbd_constraint_temperature,
        avbd_refresh_period=args.avbd_refresh_period,
        avbd_refresh_cstr_trigger=args.avbd_refresh_cstr_trigger,
        avbd_adaptive_refresh=args.avbd_adaptive_refresh,
        avbd_adaptive_refresh_increment=args.avbd_adaptive_refresh_increment,
        avbd_adaptive_refresh_max_period=args.avbd_adaptive_refresh_max_period,
        avbd_adaptive_refresh_safe_ratio=args.avbd_adaptive_refresh_safe_ratio,
        avbd_rho_init=args.avbd_rho_init,
        avbd_rho_growth=args.avbd_rho_growth,
        avbd_rho_max=args.avbd_rho_max,
        avbd_local_head_lr=args.avbd_local_head_lr,
        avbd_local_global_distill_weight=args.avbd_local_global_distill_weight,
    )
    cfg.avbd_use_active_set = args.avbd_use_active_set
    cfg.avbd_active_constraint_limit = args.avbd_active_constraint_limit
    cfg.avbd_active_lambda_weight = args.avbd_active_lambda_weight
    cfg.avbd_active_growth_weight = args.avbd_active_growth_weight
    cfg.avbd_active_stale_weight = args.avbd_active_stale_weight
    cfg.avbd_active_stale_horizon = args.avbd_active_stale_horizon
    cfg.avbd_active_keep_safety = args.avbd_active_keep_safety
    cfg.avbd_use_multi_constraint_woodbury = args.avbd_use_multi_constraint_woodbury
    cfg.avbd_use_event_refresh = args.avbd_use_event_refresh
    cfg.avbd_event_cos_threshold = args.avbd_event_cos_threshold
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
        local_files_only=cfg.local_files_only,
    )
    safety_samples = load_safety_samples(args.safety_prompts)
    print(f"Safety prompts: {len(safety_samples)}")

    model, tokenizer, adapted_layers = load_model_and_tokenizer(cfg)
    choice_token_ids = get_choice_token_ids(tokenizer)
    print(f"Adapted layers: {adapted_layers}")
    print(f"Trainable surface: {cfg.trainable_surface}")
    pre_safety = evaluate(model, tokenizer, safety_samples, choice_token_ids, cfg)
    print(f"Pre-warmup safety accuracy: {pre_safety:.3f}")
    post_safety = pre_safety
    if args.safety_warmup_epochs > 0:
        print(f"\n=== Safety Warmup ({args.safety_warmup_epochs} epochs, lr={args.safety_warmup_lr}) ===")
        post_safety = safety_warmup(
            model,
            tokenizer,
            safety_samples,
            choice_token_ids,
            cfg,
            warmup_epochs=args.safety_warmup_epochs,
            warmup_lr=args.safety_warmup_lr,
        )
    initial_trainable = clone_trainable_params(model)

    methods = []
    if not args.skip_hessian:
        methods.append(
            ("SfB-AVBD-Hessian", lambda: train_sfb_avbd_hessian(model, tokenizer, tasks, safety_samples, choice_token_ids, cfg))
        )
    if args.run_hessian_lowbp:
        methods.append(
            (
                "SfB-AVBD-Hessian-LowBP",
                lambda: train_sfb_avbd_hessian_lowbp(
                    model,
                    tokenizer,
                    tasks,
                    safety_samples,
                    choice_token_ids,
                    cfg,
                    adapted_layers,
                ),
            )
        )
    if not args.skip_kl:
        methods.append(
            ("SfB-KL", lambda: train_sfb_kl(model, tokenizer, tasks, safety_samples, choice_token_ids, cfg, beta_kl=args.beta_kl))
        )
    if not args.skip_posthoc:
        methods.append(
            ("PostHoc-Adam", lambda: train_posthoc_adam(model, tokenizer, tasks, safety_samples, choice_token_ids, cfg))
        )
    if not methods:
        raise ValueError("No Stage-1 methods selected. Enable at least one training method.")

    all_results = {
        "config": vars(cfg),
        "pre_warmup_safety": pre_safety,
        "post_warmup_safety": post_safety,
        "methods": [],
    }
    for name, train_fn in methods:
        print("\n" + "=" * 70)
        print(f"=== {name} ===")
        print("=" * 70)
        restore_trainable_params(model, initial_trainable)
        start = time.time()
        result = train_fn()
        result["wall_time"] = time.time() - start
        all_results["methods"].append(result)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    return all_results


def build_parser():
    parser = argparse.ArgumentParser(description="MLX Safety-from-Birth experiment (stage 1).")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-source", type=str, default="ag_news")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--safety-prompts", type=str, default=str(PORTABLE_ROOT / "prompts" / "safety_prompts.json"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta-kl", type=float, default=0.5)
    parser.add_argument("--replay-size", type=int, default=8)
    parser.add_argument("--max-train-per-task", type=int, default=64)
    parser.add_argument("--max-eval-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--local-head-count", type=int, default=2)
    parser.add_argument("--lora-num-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--trainable-surface", type=str, default="moe_mlp_only", choices=["all_lora", "moe_mlp_only"])
    parser.add_argument("--avbd-constraint-margin", type=float, default=0.02)
    parser.add_argument("--avbd-constraint-temperature", type=float, default=1.5)
    parser.add_argument("--avbd-first-task-global-only", dest="avbd_first_task_global_only", action="store_true")
    parser.add_argument("--disable-avbd-first-task-global-only", dest="avbd_first_task_global_only", action="store_false")
    parser.set_defaults(avbd_first_task_global_only=False)
    parser.add_argument("--avbd-warmup-global-steps", type=int, default=0)
    parser.add_argument("--avbd-refresh-period", type=int, default=6)
    parser.add_argument("--avbd-refresh-cstr-trigger", type=float, default=0.5)
    parser.add_argument("--avbd-adaptive-refresh", action="store_true")
    parser.add_argument("--avbd-adaptive-refresh-increment", type=int, default=2)
    parser.add_argument("--avbd-adaptive-refresh-max-period", type=int, default=24)
    parser.add_argument("--avbd-adaptive-refresh-safe-ratio", type=float, default=0.5)
    parser.add_argument("--avbd-disable-active-set", dest="avbd_use_active_set", action="store_false")
    parser.add_argument("--avbd-active-constraint-limit", type=int, default=2)
    parser.add_argument("--avbd-active-lambda-weight", type=float, default=0.25)
    parser.add_argument("--avbd-active-growth-weight", type=float, default=1.0)
    parser.add_argument("--avbd-active-stale-weight", type=float, default=0.1)
    parser.add_argument("--avbd-active-stale-horizon", type=int, default=4)
    parser.add_argument("--disable-avbd-active-keep-safety", dest="avbd_active_keep_safety", action="store_false")
    parser.add_argument("--avbd-disable-woodbury", dest="avbd_use_multi_constraint_woodbury", action="store_false")
    parser.add_argument("--avbd-enable-event-refresh", dest="avbd_use_event_refresh", action="store_true")
    parser.add_argument("--avbd-disable-event-refresh", dest="avbd_use_event_refresh", action="store_false")
    parser.add_argument("--avbd-event-cos-threshold", type=float, default=0.2)
    parser.set_defaults(
        avbd_use_active_set=True,
        avbd_active_keep_safety=True,
        avbd_use_multi_constraint_woodbury=True,
        avbd_use_event_refresh=False,
    )
    parser.add_argument("--avbd-rho-init", type=float, default=1.0)
    parser.add_argument("--avbd-rho-growth", type=float, default=1.5)
    parser.add_argument("--avbd-rho-max", type=float, default=2.0)
    parser.add_argument("--avbd-local-head-lr", type=float, default=1e-3)
    parser.add_argument("--avbd-local-global-distill-weight", type=float, default=0.25)
    parser.add_argument("--safety-warmup-epochs", type=int, default=5)
    parser.add_argument("--safety-warmup-lr", type=float, default=5e-4)
    parser.add_argument("--skip-hessian", action="store_true")
    parser.add_argument("--run-hessian-lowbp", action="store_true")
    parser.add_argument("--skip-kl", action="store_true")
    parser.add_argument("--skip-posthoc", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--allow-online-hf-load", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
