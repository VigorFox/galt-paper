"""Stage D native policy smoke on a real Qwen-MLX carrier.

This is the first real-carrier Stage D prototype after the native routing toy.
Its goal is not to solve full Stage D optimization, but to validate the
architectural principle on a live Transformer:

1. maintain an explicit narrow policy state
2. force downstream specialized operators to consume that policy
3. test whether zeroing / scrambling policy now affects useful behavior

Unlike Stage C, the narrow state is not treated as a summary readout target.
It is used directly to mix specialized per-block expert adapters.
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
from mlx_lm.models.base import create_attention_mask

PORTABLE_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PORTABLE_ROOT / "optimizer"))
sys.path.insert(0, str(PORTABLE_ROOT / "data_utils"))
sys.path.insert(0, str(PORTABLE_ROOT / "experiments" / "shared_runtime"))
sys.path.insert(0, str(THIS_DIR))

from avbd_galt_optimizer_mlx import AVBDGALTOptimizer
from phase_d_block_local_mlx import (
    BlockLocalConfig,
    _block_norm,
    _build_blocks,
    _modeled_block_local_stats,
)
from phase_d_smoke_mlx import (
    _eval_constraint_value,
    _layer_param_counts,
    _load_model_all_layer_lora,
    _load_smoke_data,
    _param_count,
    _summary_stats,
)
from mlx_utils import flatten_tree, scalar
from continual_runtime_mlx import (
    ExperimentConfig,
    batch_labels,
    compute_choice_scores_batch,
    compute_distillation_kl,
    compute_replay_anchor_probs,
    cross_entropy_from_scores,
    evaluate,
    get_choice_token_ids,
    sample_replay_buffer,
    set_seed,
    tokenize_prompt,
    tokenize_prompts,
)


RESULTS_DIR = PORTABLE_ROOT / "results" / "galt_prework" / "stage_d_native_policy_smoke"
LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")
POLICY_READ_RE = re.compile(r"^policy_reads\.(\d+)\.")
POLICY_CARRY_RE = re.compile(r"^policy_carries\.(\d+)\.")
POLICY_DOWN_RE = re.compile(r"^policy_downs\.(\d+)\.")
POLICY_UP_RE = re.compile(r"^policy_ups\.(\d+)\.")
OUTPUT_HEAD_RE = re.compile(r"^output_choice_heads\.(\d+)\.")
TASK_OUTPUT_HEAD_RE = re.compile(r"^task_output_choice_heads\.(\d+)\.")
SAFETY_OUTPUT_HEAD_RE = re.compile(r"^safety_output_choice_heads\.(\d+)\.")
MEMORY_OUTPUT_HEAD_RE = re.compile(r"^memory_output_choice_heads\.(\d+)\.")
OUTPUT_BRANCH_NAMES = ("task", "safety", "memory")


@dataclass
class StageDNativePolicyConfig(BlockLocalConfig):
    num_policies: int = 4
    expert_rank: int = 16
    policy_scale: float = 0.08
    distill_weight: float = 0.5
    route_task_weight: float = 0.5
    route_entropy_weight: float = 0.01
    route_block_weight_power: float = 2.0
    safety_branch_weight: float = 0.0
    memory_branch_weight: float = 0.0
    branch_preference_weight: float = 0.0
    branch_preference_margin: float = 0.0
    task_branch_preference: bool = True
    task_shadow_suppression_weight: float = 0.0
    task_shadow_suppression_margin: float = 0.0
    safety_shadow_suppression_weight: float = 0.0
    safety_shadow_suppression_margin: float = 0.0
    memory_shadow_suppression_weight: float = 0.0
    memory_shadow_suppression_margin: float = 0.0
    output_expert_scale: float = 0.0
    base_choice_scale: float = 1.0
    typed_output_branches: bool = False
    hard_routing: bool = True
    policy_only_warmup_steps: int = 4
    output: str = str(RESULTS_DIR / "summary.json")


class MacroBlockPolicyCarrier(nn.Module):
    """Real-carrier prototype with explicit policy-conditioned experts."""

    def __init__(
        self,
        base_model,
        block_size: int,
        num_policies: int,
        expert_rank: int,
        policy_scale: float,
        output_expert_scale: float,
        base_choice_scale: float,
        typed_output_branches: bool,
        hard_routing: bool,
        task_choice_token_ids: list[int],
    ):
        super().__init__()
        self.args = base_model.args
        self.model_type = getattr(base_model, "model_type", self.args.model_type)
        self.model = base_model.model
        self.block_size = block_size
        self.num_policies = num_policies
        self.expert_rank = expert_rank
        self.policy_scale = policy_scale
        self.output_expert_scale = output_expert_scale
        self.base_choice_scale = base_choice_scale
        self.typed_output_branches = typed_output_branches
        self.hard_routing = hard_routing
        self.task_choice_token_ids = task_choice_token_ids
        self.route_mode = "normal"
        self.block_ranges = [
            (start, min(start + block_size, len(self.model.layers)))
            for start in range(0, len(self.model.layers), block_size)
        ]
        self.policy_reads = [nn.Linear(int(self.args.hidden_size), num_policies) for _ in self.block_ranges]
        self.policy_carries = [nn.Linear(num_policies, num_policies) for _ in self.block_ranges]
        n_experts = len(self.block_ranges) * num_policies
        self.policy_downs = [nn.Linear(int(self.args.hidden_size), expert_rank) for _ in range(n_experts)]
        self.policy_ups = [nn.Linear(expert_rank, int(self.args.hidden_size)) for _ in range(n_experts)]
        if self.typed_output_branches:
            self.task_output_choice_heads = [nn.Linear(int(self.args.hidden_size), num_policies) for _ in range(num_policies)]
            self.safety_output_choice_heads = [nn.Linear(int(self.args.hidden_size), num_policies) for _ in range(num_policies)]
            self.memory_output_choice_heads = [nn.Linear(int(self.args.hidden_size), num_policies) for _ in range(num_policies)]
        else:
            self.output_choice_heads = [nn.Linear(int(self.args.hidden_size), num_policies) for _ in range(num_policies)]
        self.route_permutations = []
        for block_index in range(len(self.block_ranges)):
            perm = list(range(num_policies))
            random.Random(3000 + block_index).shuffle(perm)
            self.route_permutations.append(mx.array(perm, dtype=mx.int32))
        if not self.args.tie_word_embeddings and hasattr(base_model, "lm_head"):
            self.lm_head = base_model.lm_head
        self._reset_policy_modules()

    def output_branch_names(self) -> tuple[str, ...]:
        return OUTPUT_BRANCH_NAMES if self.typed_output_branches else ("shared",)

    def _output_heads_for_branch(self, branch_name: str):
        if not self.typed_output_branches:
            return self.output_choice_heads
        branch_map = {
            "task": self.task_output_choice_heads,
            "safety": self.safety_output_choice_heads,
            "memory": self.memory_output_choice_heads,
        }
        if branch_name not in branch_map:
            raise ValueError(f"Unsupported output branch: {branch_name}")
        return branch_map[branch_name]

    def _reset_policy_modules(self):
        hidden_size = int(self.args.hidden_size)
        read_scale = 0.02 / math.sqrt(hidden_size)
        down_scale = 0.02 / math.sqrt(hidden_size)
        up_scale = 0.02 / math.sqrt(max(1, self.expert_rank))
        carry_eye = 0.5 * mx.eye(self.num_policies)
        for reader in self.policy_reads:
            reader.weight = mx.random.normal(shape=reader.weight.shape) * read_scale
            reader.bias = mx.zeros(reader.bias.shape)
        for carry in self.policy_carries:
            carry.weight = carry_eye
            carry.bias = mx.zeros(carry.bias.shape)
        for down in self.policy_downs:
            down.weight = mx.random.normal(shape=down.weight.shape) * down_scale
            down.bias = mx.zeros(down.bias.shape)
        for up in self.policy_ups:
            up.weight = mx.random.normal(shape=up.weight.shape) * up_scale
            up.bias = mx.zeros(up.bias.shape)
        for branch_name in self.output_branch_names():
            for head in self._output_heads_for_branch(branch_name):
                head.weight = mx.zeros(head.weight.shape)
                head.bias = mx.zeros(head.bias.shape)

    def set_route_mode(self, mode: str):
        if mode not in {"normal", "zero", "scramble"}:
            raise ValueError(f"Unsupported route mode: {mode}")
        self.route_mode = mode

    def _transform_route(self, block_index: int, route: mx.array) -> mx.array:
        if self.route_mode == "normal":
            return route
        if self.route_mode == "zero":
            return mx.ones(route.shape, dtype=route.dtype) * (1.0 / self.num_policies)
        return mx.take(route, self.route_permutations[block_index], axis=-1)

    def _expert_index(self, block_index: int, expert_index: int) -> int:
        return block_index * self.num_policies + expert_index

    def _apply_policy_experts(self, block_index: int, h: mx.array, route: mx.array) -> mx.array:
        mixed = mx.zeros_like(h)
        route_indices = mx.argmax(route, axis=-1) if self.hard_routing else None
        for expert_index in range(self.num_policies):
            flat_index = self._expert_index(block_index, expert_index)
            expert_hidden = mx.tanh(self.policy_downs[flat_index](h))
            expert_out = self.policy_ups[flat_index](expert_hidden)
            if self.hard_routing:
                expert_mask = (route_indices == expert_index).astype(h.dtype)[:, None, None]
                mixed = mixed + expert_mask * expert_out
            else:
                mixed = mixed + route[:, expert_index][:, None, None] * expert_out
        return h + self.policy_scale * mixed

    def _run_blocks(self, inputs: mx.array, input_embeddings: mx.array | None = None):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.model.embed_tokens(inputs)
        route = mx.ones((h.shape[0], self.num_policies), dtype=h.dtype) * (1.0 / self.num_policies)
        route_map: dict[int, mx.array] = {}
        route_logits_map: dict[int, mx.array] = {}

        for block_index, (start, end) in enumerate(self.block_ranges):
            consumed_route = self._transform_route(block_index, route)
            h = self._apply_policy_experts(block_index, h, consumed_route)
            mask = create_attention_mask(h, None)
            for layer in self.model.layers[start:end]:
                h = layer(h, mask, None)
            summary = h[:, -1, :]
            route_logits = self.policy_reads[block_index](summary) + self.policy_carries[block_index](consumed_route)
            route = mx.softmax(route_logits.astype(mx.float32), axis=-1).astype(h.dtype)
            route_map[block_index] = route
            route_logits_map[block_index] = route_logits

        h = self.model.norm(h)
        return h, route_map, route_logits_map

    def _output_choice_scores(self, summary: mx.array, route: mx.array, branch_name: str) -> mx.array:
        mixed_scores = mx.zeros((summary.shape[0], self.num_policies), dtype=summary.dtype)
        route_indices = mx.argmax(route, axis=-1) if self.hard_routing else None
        for expert_index, head in enumerate(self._output_heads_for_branch(branch_name)):
            head_scores = head(summary)
            if self.hard_routing:
                expert_mask = (route_indices == expert_index).astype(summary.dtype)[:, None]
                mixed_scores = mixed_scores + expert_mask * head_scores
            else:
                mixed_scores = mixed_scores + route[:, expert_index][:, None] * head_scores
        return self.output_expert_scale * mixed_scores

    def __call__(self, inputs: mx.array, cache=None, input_embeddings: mx.array | None = None):
        if cache is not None:
            raise ValueError("Stage D native policy smoke does not support KV cache.")
        out, _, _ = self._run_blocks(inputs, input_embeddings=input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def forward_with_policy_states(self, inputs: mx.array, input_embeddings: mx.array | None = None, branch_name: str = "task"):
        out, route_map, route_logits_map = self._run_blocks(inputs, input_embeddings=input_embeddings)
        final_block_index = max(route_map)
        final_route = self._transform_route(final_block_index, route_map[final_block_index])
        final_summary = out[:, -1, :]
        output_choice_scores = self._output_choice_scores(final_summary, final_route, branch_name=branch_name)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        return logits, route_map, route_logits_map, output_choice_scores


def _policy_block_index(name: str, num_policies: int, num_blocks: int) -> int | None:
    match = POLICY_READ_RE.match(name)
    if match:
        return int(match.group(1))
    match = POLICY_CARRY_RE.match(name)
    if match:
        return int(match.group(1))
    match = POLICY_DOWN_RE.match(name)
    if match:
        return int(match.group(1)) // num_policies
    match = POLICY_UP_RE.match(name)
    if match:
        return int(match.group(1)) // num_policies
    match = OUTPUT_HEAD_RE.match(name)
    if match:
        return num_blocks - 1
    match = TASK_OUTPUT_HEAD_RE.match(name)
    if match:
        return num_blocks - 1
    match = SAFETY_OUTPUT_HEAD_RE.match(name)
    if match:
        return num_blocks - 1
    match = MEMORY_OUTPUT_HEAD_RE.match(name)
    if match:
        return num_blocks - 1
    return None


def _augment_blocks_with_policy_params(model: MacroBlockPolicyCarrier, blocks: list[dict]) -> list[dict]:
    policy_counts = {block["block_index"]: 0 for block in blocks}
    for name, value in flatten_tree(model.trainable_parameters()).items():
        block_index = _policy_block_index(name, model.num_policies, len(model.block_ranges))
        if block_index is not None:
            policy_counts[block_index] += _param_count(value)
    augmented = []
    for block in blocks:
        policy_n_params = policy_counts.get(block["block_index"], 0)
        augmented.append(
            {
                **block,
                "layer_n_params": block["n_params"],
                "policy_n_params": policy_n_params,
                "n_params": int(block["n_params"] + policy_n_params),
            }
        )
    return augmented


def _mask_flat_to_stage_d_block(
    flat_map: dict[str, mx.array],
    layers: set[int],
    block_index: int,
    num_policies: int,
    num_blocks: int,
) -> dict[str, mx.array]:
    masked = {}
    for name, value in flat_map.items():
        layer_match = LAYER_RE.match(name)
        if layer_match and int(layer_match.group(1)) in layers:
            masked[name] = value
            continue
        policy_block = _policy_block_index(name, num_policies, num_blocks)
        if policy_block == block_index:
            masked[name] = value
    return masked


def _mask_policy_only(flat_map: dict[str, mx.array], block_index: int, num_policies: int, num_blocks: int) -> dict[str, mx.array]:
    masked = {}
    for name, value in flat_map.items():
        if _policy_block_index(name, num_policies, num_blocks) == block_index:
            masked[name] = value
    return masked


def _all_choices_are_single_token(choice_token_ids: list[list[int]]) -> bool:
    return all(len(choice_ids) == 1 for choice_ids in choice_token_ids)


def _infer_choice_count_from_samples(samples: list[dict]) -> int:
    for sample in samples:
        prompt = sample.get("prompt", "")
        count = sum(1 for line in prompt.splitlines() if re.match(r"^[A-J]\.\s", line.strip()))
        if count:
            return count
    raise ValueError("Could not infer task choice count from the provided samples.")


def _compute_choice_scores_from_logits(logits: mx.array, choice_token_ids: list[list[int]]) -> mx.array:
    choice_ids = mx.array([choice_ids[0] for choice_ids in choice_token_ids], dtype=mx.int32)
    last_log_probs = logits[-1] - mx.logsumexp(logits[-1], axis=-1, keepdims=True)
    return mx.take(last_log_probs, choice_ids, axis=0)


def _compute_choice_scores(
    module: MacroBlockPolicyCarrier,
    prompt_ids: list[int],
    choice_token_ids: list[list[int]],
    branch_name: str = "task",
) -> mx.array:
    if not _all_choices_are_single_token(choice_token_ids):
        raise ValueError("Stage D native policy smoke currently expects single-token choices.")
    logits, _route_map, _route_logits, output_choice_scores = module.forward_with_policy_states(
        mx.array([prompt_ids], dtype=mx.int32),
        branch_name=branch_name,
    )
    base_scores = _compute_choice_scores_from_logits(logits[0], choice_token_ids)
    if module.base_choice_scale != 1.0:
        base_scores = module.base_choice_scale * base_scores
    if module.output_expert_scale > 0.0:
        policy_count = output_choice_scores.shape[-1]
        policy_bonus = output_choice_scores[0]
        padded = mx.concatenate(
            [
                policy_bonus,
                mx.zeros((base_scores.shape[0] - policy_count,), dtype=policy_bonus.dtype),
            ],
            axis=0,
        )
        return base_scores + padded
    return base_scores


def _compute_choice_scores_batch(
    module: MacroBlockPolicyCarrier,
    prompt_token_batches: list[list[int]],
    choice_token_ids: list[list[int]],
    branch_name: str = "task",
) -> mx.array:
    return mx.stack(
        [_compute_choice_scores(module, prompt_ids, choice_token_ids, branch_name=branch_name) for prompt_ids in prompt_token_batches],
        axis=0,
    )


def _compute_choice_distribution(
    module: MacroBlockPolicyCarrier,
    prompt_token_batches: list[list[int]],
    choice_token_ids: list[list[int]],
    temperature: float,
    branch_name: str = "task",
) -> mx.array:
    scores = _compute_choice_scores_batch(module, prompt_token_batches, choice_token_ids, branch_name=branch_name)
    return mx.softmax(scores / temperature, axis=-1)


def _compute_replay_anchor_probs_local(
    model,
    tokenizer,
    replay_samples: list[dict],
    choice_token_ids: list[list[int]],
    cfg: ExperimentConfig,
    branch_name: str = "task",
) -> mx.array:
    prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in replay_samples], cfg.max_length)
    return _compute_choice_distribution(
        model,
        prompt_tokens,
        choice_token_ids,
        temperature=cfg.avbd_constraint_temperature,
        branch_name=branch_name,
    )


def _make_kl_constraint_fn_local(
    tokenizer,
    replay: list[dict],
    anchor_probs: mx.array,
    choice_token_ids,
    cfg: StageDNativePolicyConfig,
    branch_name: str = "task",
):
    prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in replay], cfg.max_length)

    def loss_fn(module):
        scores = _compute_choice_scores_batch(module, prompt_tokens, choice_token_ids, branch_name=branch_name)
        return compute_distillation_kl(scores, anchor_probs, temperature=cfg.avbd_constraint_temperature)

    return loss_fn


def _evaluate_local(
    model,
    tokenizer,
    samples: list[dict],
    choice_token_ids: list[list[int]],
    eval_cfg: ExperimentConfig,
    branch_name: str = "task",
) -> float:
    batch_size = max(1, eval_cfg.eval_batch_size)
    correct = 0
    total = 0
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], eval_cfg.max_length)
        scores = _compute_choice_scores_batch(model, prompt_tokens, choice_token_ids, branch_name=branch_name)
        preds = mx.argmax(scores, axis=-1)
        mx.eval(preds, labels)
        correct += int(mx.sum(preds == labels).item())
        total += len(batch_samples)
    return float(correct / max(1, total))


def _correct_choice_scores(scores: mx.array, labels: mx.array) -> mx.array:
    return mx.take_along_axis(scores, labels.reshape((-1, 1)), axis=-1).reshape((-1,))


def _branch_preference_loss(
    module: MacroBlockPolicyCarrier,
    prompt_token_batches: list[list[int]],
    labels: mx.array,
    choice_token_ids: list[list[int]],
    preferred_branch: str,
    margin: float,
) -> mx.array:
    if not module.typed_output_branches:
        return mx.array(0.0, dtype=mx.float32)
    preferred_scores = _compute_choice_scores_batch(
        module,
        prompt_token_batches,
        choice_token_ids,
        branch_name=preferred_branch,
    )
    preferred_correct = _correct_choice_scores(preferred_scores, labels)
    loss = mx.array(0.0, dtype=mx.float32)
    for branch_name in OUTPUT_BRANCH_NAMES:
        if branch_name == preferred_branch:
            continue
        branch_scores = _compute_choice_scores_batch(
            module,
            prompt_token_batches,
            choice_token_ids,
            branch_name=branch_name,
        )
        branch_correct = _correct_choice_scores(branch_scores, labels)
        loss = loss + mx.mean(mx.maximum(0.0, margin + branch_correct - preferred_correct))
    return loss


def _branch_vs_branch_margin_loss(
    module: MacroBlockPolicyCarrier,
    prompt_token_batches: list[list[int]],
    labels: mx.array,
    choice_token_ids: list[list[int]],
    preferred_branch: str,
    competing_branch: str,
    margin: float,
) -> mx.array:
    if not module.typed_output_branches:
        return mx.array(0.0, dtype=mx.float32)
    preferred_scores = _compute_choice_scores_batch(
        module,
        prompt_token_batches,
        choice_token_ids,
        branch_name=preferred_branch,
    )
    competing_scores = _compute_choice_scores_batch(
        module,
        prompt_token_batches,
        choice_token_ids,
        branch_name=competing_branch,
    )
    preferred_correct = _correct_choice_scores(preferred_scores, labels)
    competing_correct = _correct_choice_scores(competing_scores, labels)
    return mx.mean(mx.maximum(0.0, margin + competing_correct - preferred_correct))


def _route_block_weights(num_blocks: int, power: float) -> list[float]:
    if num_blocks <= 0:
        return []
    raw = [float((index + 1) ** power) for index in range(num_blocks)]
    total = sum(raw)
    return [value / total for value in raw]


def _route_logits_batch(module: MacroBlockPolicyCarrier, prompt_token_batches: list[list[int]]) -> list[mx.array]:
    per_block_logits: list[list[mx.array]] | None = None
    for prompt_ids in prompt_token_batches:
        token_array = mx.array([prompt_ids], dtype=mx.int32)
        _logits, _route_map, route_logits_map, _output_choice_scores = module.forward_with_policy_states(token_array)
        block_logits = [route_logits_map[block_index][0] for block_index in sorted(route_logits_map)]
        if per_block_logits is None:
            per_block_logits = [[] for _ in block_logits]
        for block_index, score in enumerate(block_logits):
            per_block_logits[block_index].append(score)
    if per_block_logits is None:
        return []
    return [mx.stack(block_logits, axis=0) for block_logits in per_block_logits]


def _route_policy_loss(
    module: MacroBlockPolicyCarrier,
    prompt_token_batches: list[list[int]],
    labels: mx.array,
    cfg: StageDNativePolicyConfig,
) -> mx.array:
    loss = mx.array(0.0, dtype=mx.float32)
    if cfg.route_task_weight <= 0.0 and cfg.route_entropy_weight <= 0.0:
        return loss
    block_logits_batch = _route_logits_batch(module, prompt_token_batches)
    weights = _route_block_weights(len(block_logits_batch), cfg.route_block_weight_power)
    for weight, scores in zip(weights, block_logits_batch):
        if cfg.route_task_weight > 0.0:
            loss = loss + (cfg.route_task_weight * weight) * cross_entropy_from_scores(scores, labels)
        if cfg.route_entropy_weight > 0.0:
            probs = mx.softmax(scores.astype(mx.float32), axis=-1)
            entropy = -mx.mean(mx.sum(probs * mx.log(probs + 1e-8), axis=-1))
            loss = loss + (cfg.route_entropy_weight * weight) * entropy
    return loss


def _make_route_forward_loss_fn(prompt_ids: list[int], block_index: int, target_route: mx.array):
    tokens = mx.array([prompt_ids], dtype=mx.int32)

    def loss_fn(module: MacroBlockPolicyCarrier):
        _logits, route_map, _route_logits, _output_choice_scores = module.forward_with_policy_states(tokens)
        current = route_map[block_index]
        diff = current - target_route
        return mx.mean(diff * diff)

    return loss_fn


def _route_residual(model: MacroBlockPolicyCarrier, prompt_ids: list[int], block_index: int, target_route: mx.array) -> float:
    tokens = mx.array([prompt_ids], dtype=mx.int32)
    _logits, route_map, _route_logits, _output_choice_scores = model.forward_with_policy_states(tokens)
    current = route_map[block_index]
    diff = current - target_route
    return scalar(mx.mean(diff * diff))


def _evaluate_route_metrics(
    model,
    tokenizer,
    samples: list[dict],
    eval_cfg: ExperimentConfig,
) -> dict[str, float]:
    batch_size = max(1, eval_cfg.eval_batch_size)
    correct = 0
    total = 0
    entropies = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], eval_cfg.max_length)
        route_logits = _route_logits_batch(model, prompt_tokens)[-1]
        route_probs = mx.softmax(route_logits.astype(mx.float32), axis=-1)
        preds = mx.argmax(route_logits, axis=-1)
        entropy = -mx.mean(mx.sum(route_probs.astype(mx.float32) * mx.log(route_probs.astype(mx.float32) + 1e-8), axis=-1))
        mx.eval(preds, labels, entropy)
        correct += int(mx.sum(preds == labels).item())
        total += len(batch_samples)
        entropies.append(float(entropy.item()))
    return {
        "route_acc": float(correct / max(1, total)),
        "route_entropy": float(sum(entropies) / max(1, len(entropies))),
    }


def _evaluate_route_modes(model, tokenizer, task, safety_samples, retain_samples, choice_token_ids, eval_cfg):
    metrics = {}
    original_mode = getattr(model, "route_mode", "normal")
    for mode in ("normal", "zero", "scramble"):
        model.set_route_mode(mode)
        metrics[mode] = {
            "task_acc": _evaluate_local(model, tokenizer, task.eval_samples, choice_token_ids, eval_cfg, branch_name="task"),
            "safety_acc": _evaluate_local(model, tokenizer, safety_samples, choice_token_ids, eval_cfg, branch_name="safety"),
            "retain_acc": _evaluate_local(model, tokenizer, retain_samples, choice_token_ids, eval_cfg, branch_name="memory"),
        }
    model.set_route_mode(original_mode)
    return metrics


def _evaluate_branch_matrix(model, tokenizer, datasets: dict[str, list[dict]], choice_token_ids, eval_cfg) -> dict[str, dict[str, float]]:
    branch_names = model.output_branch_names()
    matrix: dict[str, dict[str, float]] = {}
    for dataset_name, samples in datasets.items():
        matrix[dataset_name] = {}
        for branch_name in branch_names:
            matrix[dataset_name][branch_name] = _evaluate_local(
                model,
                tokenizer,
                samples,
                choice_token_ids,
                eval_cfg,
                branch_name=branch_name if branch_name != "shared" else "task",
            )
    return matrix


def _branch_specialization_summary(branch_matrix: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    expected = {
        "task": "task",
        "safety": "safety",
        "retain": "memory",
    }
    summary: dict[str, dict[str, float]] = {}
    for dataset_name, preferred_branch in expected.items():
        dataset_scores = branch_matrix.get(dataset_name, {})
        if preferred_branch not in dataset_scores:
            continue
        preferred = dataset_scores[preferred_branch]
        wrong_scores = [score for branch_name, score in dataset_scores.items() if branch_name != preferred_branch]
        summary[dataset_name] = {
            "preferred_branch_acc": preferred,
            "best_wrong_branch_acc": max(wrong_scores) if wrong_scores else preferred,
            "preferred_margin_vs_best_wrong": preferred - (max(wrong_scores) if wrong_scores else preferred),
        }
    return summary


def run_stage_d_native_policy_smoke(cfg: StageDNativePolicyConfig) -> dict:
    set_seed(cfg.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_model, tokenizer = _load_model_all_layer_lora(cfg)
    choice_token_ids = get_choice_token_ids(tokenizer)
    tasks, safety_samples, retain_samples = _load_smoke_data(cfg)
    task = tasks[0]
    task_samples = list(task.train_samples) + list(task.eval_samples)
    observed_label_count = int(mx.max(batch_labels(task_samples)).item()) + 1
    task_label_count = max(observed_label_count, _infer_choice_count_from_samples(task_samples))
    if cfg.num_policies != task_label_count:
        raise ValueError(
            f"num_policies={cfg.num_policies} must match number of task labels={task_label_count} for this smoke."
        )
    if not _all_choices_are_single_token(choice_token_ids):
        raise ValueError("Stage D native policy smoke currently expects single-token choice tokens.")
    task_choice_token_ids = [choice_ids[0] for choice_ids in choice_token_ids[: cfg.num_policies]]
    train_prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in task.train_samples], cfg.max_length)
    teacher_task_scores = compute_choice_scores_batch(base_model, train_prompt_tokens, choice_token_ids)
    teacher_task_probs = mx.softmax(teacher_task_scores / cfg.avbd_constraint_temperature, axis=-1)
    teacher_task_probs_by_prompt = {
        sample["prompt"]: teacher_task_probs[index]
        for index, sample in enumerate(task.train_samples)
    }

    model = MacroBlockPolicyCarrier(
        base_model,
        block_size=cfg.block_size,
        num_policies=cfg.num_policies,
        expert_rank=cfg.expert_rank,
        policy_scale=cfg.policy_scale,
        output_expert_scale=cfg.output_expert_scale,
        base_choice_scale=cfg.base_choice_scale,
        typed_output_branches=cfg.typed_output_branches,
        hard_routing=cfg.hard_routing,
        task_choice_token_ids=task_choice_token_ids,
    )
    mx.eval(model.parameters())

    eval_cfg = ExperimentConfig(
        model_name=cfg.model_name,
        dataset_source=cfg.dataset_source,
        local_files_only=cfg.local_files_only,
        eval_batch_size=cfg.eval_batch_size,
        max_length=cfg.max_length,
        avbd_constraint_temperature=cfg.avbd_constraint_temperature,
    )

    layer_counts = _layer_param_counts(model)
    blocks = _augment_blocks_with_policy_params(model, _build_blocks(layer_counts, cfg.block_size))
    k_policy = cfg.num_policies
    d_over_k_by_block = {block["block_index"]: float(block["n_params"] / k_policy) for block in blocks}

    safety_replay = sample_replay_buffer(safety_samples, min(cfg.replay_size, len(safety_samples)), cfg.seed)
    retain_replay = sample_replay_buffer(retain_samples, min(cfg.replay_size, len(retain_samples)), cfg.seed + 17)
    safety_replay_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in safety_replay], cfg.max_length)
    safety_replay_labels = batch_labels(safety_replay)
    retain_replay_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in retain_replay], cfg.max_length)
    retain_replay_labels = batch_labels(retain_replay)
    safety_anchor = _compute_replay_anchor_probs_local(
        model,
        tokenizer,
        safety_replay,
        choice_token_ids,
        eval_cfg,
        branch_name="safety",
    )
    retain_anchor = _compute_replay_anchor_probs_local(
        model,
        tokenizer,
        retain_replay,
        choice_token_ids,
        eval_cfg,
        branch_name="memory",
    )
    mx.eval(safety_anchor, retain_anchor)

    safety_constraint_fn = _make_kl_constraint_fn_local(
        tokenizer,
        safety_replay,
        safety_anchor,
        choice_token_ids,
        cfg,
        branch_name="safety",
    )
    retain_constraint_fn = _make_kl_constraint_fn_local(
        tokenizer,
        retain_replay,
        retain_anchor,
        choice_token_ids,
        cfg,
        branch_name="memory",
    )

    probe_prompt = task.train_samples[0]["prompt"]
    probe_tokens = tokenize_prompt(tokenizer, probe_prompt, cfg.max_length)
    probe_array = mx.array([probe_tokens], dtype=mx.int32)
    _probe_logits, initial_route_map, _probe_route_logits, _probe_choice_scores = model.forward_with_policy_states(probe_array)
    forward_targets = {block_idx: route for block_idx, route in initial_route_map.items()}
    mx.eval(*forward_targets.values())

    pre_metrics = {
        "task_acc": _evaluate_local(model, tokenizer, task.eval_samples, choice_token_ids, eval_cfg, branch_name="task"),
        "safety_acc": _evaluate_local(model, tokenizer, safety_samples[: cfg.safety_eval_size], choice_token_ids, eval_cfg, branch_name="safety"),
        "retain_acc": _evaluate_local(model, tokenizer, retain_samples[: cfg.retain_eval_size], choice_token_ids, eval_cfg, branch_name="memory"),
    }
    pre_route_metrics = _evaluate_route_metrics(model, tokenizer, task.eval_samples, eval_cfg)

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
                "forward_ci": optimizer.add_constraint(f"route_forward_block_{block['block_index']}"),
                "safety_ci": optimizer.add_constraint("safety_alignment"),
                "retain_ci": optimizer.add_constraint("knowledge_retain"),
            }
        )

    def task_loss(module, batch_samples):
        labels = batch_labels(batch_samples)
        prompt_tokens = tokenize_prompts(tokenizer, [sample["prompt"] for sample in batch_samples], cfg.max_length)
        token_scores = _compute_choice_scores_batch(module, prompt_tokens, choice_token_ids, branch_name="task")
        loss = cross_entropy_from_scores(token_scores, labels)
        teacher_probs = mx.stack([teacher_task_probs_by_prompt[sample["prompt"]] for sample in batch_samples], axis=0)
        if cfg.distill_weight > 0.0:
            loss = loss + cfg.distill_weight * compute_distillation_kl(
                token_scores,
                teacher_probs,
                temperature=cfg.avbd_constraint_temperature,
            )
        if cfg.typed_output_branches and cfg.safety_branch_weight > 0.0:
            safety_scores = _compute_choice_scores_batch(
                module,
                safety_replay_tokens,
                choice_token_ids,
                branch_name="safety",
            )
            loss = loss + cfg.safety_branch_weight * cross_entropy_from_scores(safety_scores, safety_replay_labels)
        if cfg.typed_output_branches and cfg.memory_branch_weight > 0.0:
            memory_scores = _compute_choice_scores_batch(
                module,
                retain_replay_tokens,
                choice_token_ids,
                branch_name="memory",
            )
            loss = loss + cfg.memory_branch_weight * cross_entropy_from_scores(memory_scores, retain_replay_labels)
        if cfg.typed_output_branches and cfg.branch_preference_weight > 0.0:
            if cfg.task_branch_preference:
                loss = loss + cfg.branch_preference_weight * _branch_preference_loss(
                    module,
                    prompt_tokens,
                    labels,
                    choice_token_ids,
                    preferred_branch="task",
                    margin=cfg.branch_preference_margin,
                )
            loss = loss + cfg.branch_preference_weight * _branch_preference_loss(
                module,
                safety_replay_tokens,
                safety_replay_labels,
                choice_token_ids,
                preferred_branch="safety",
                margin=cfg.branch_preference_margin,
            )
            loss = loss + cfg.branch_preference_weight * _branch_preference_loss(
                module,
                retain_replay_tokens,
                retain_replay_labels,
                choice_token_ids,
                preferred_branch="memory",
                margin=cfg.branch_preference_margin,
            )
        if cfg.typed_output_branches and cfg.task_shadow_suppression_weight > 0.0:
            loss = loss + cfg.task_shadow_suppression_weight * _branch_vs_branch_margin_loss(
                module,
                safety_replay_tokens,
                safety_replay_labels,
                choice_token_ids,
                preferred_branch="safety",
                competing_branch="task",
                margin=cfg.task_shadow_suppression_margin,
            )
            loss = loss + cfg.task_shadow_suppression_weight * _branch_vs_branch_margin_loss(
                module,
                retain_replay_tokens,
                retain_replay_labels,
                choice_token_ids,
                preferred_branch="memory",
                competing_branch="task",
                margin=cfg.task_shadow_suppression_margin,
            )
        if cfg.typed_output_branches and cfg.safety_shadow_suppression_weight > 0.0:
            loss = loss + cfg.safety_shadow_suppression_weight * _branch_vs_branch_margin_loss(
                module,
                safety_replay_tokens,
                safety_replay_labels,
                choice_token_ids,
                preferred_branch="safety",
                competing_branch="task",
                margin=cfg.safety_shadow_suppression_margin,
            )
        if cfg.typed_output_branches and cfg.memory_shadow_suppression_weight > 0.0:
            loss = loss + cfg.memory_shadow_suppression_weight * _branch_vs_branch_margin_loss(
                module,
                retain_replay_tokens,
                retain_replay_labels,
                choice_token_ids,
                preferred_branch="memory",
                competing_branch="task",
                margin=cfg.memory_shadow_suppression_margin,
            )
        return loss + _route_policy_loss(module, prompt_tokens, labels, cfg)

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
        route_only_phase = step_idx < cfg.policy_only_warmup_steps
        task_value_tensor, task_grads_tree = task_grad_fn(model, batch)
        flat_task_grads = flatten_tree(task_grads_tree)
        _safety_value_tensor, safety_grads_tree = safety_grad_fn(model)
        _retain_value_tensor, retain_grads_tree = retain_grad_fn(model)
        safety_grads = flatten_tree(safety_grads_tree)
        retain_grads = flatten_tree(retain_grads_tree)
        safety_value = _eval_constraint_value(model, safety_constraint_fn)
        retain_value = _eval_constraint_value(model, retain_constraint_fn)
        mx.eval(task_value_tensor)

        step_blocks = []
        for entry in block_optimizers:
            block = entry["block"]
            block_index = block["block_index"]
            layer_set = set(block["layers"])
            target_route = forward_targets[block_index]

            forward_fn = _make_route_forward_loss_fn(probe_tokens, block_index, target_route)
            forward_grad_fn = nn.value_and_grad(model, forward_fn)
            forward_value_tensor, forward_grads_tree = forward_grad_fn(model)
            forward_grads = flatten_tree(forward_grads_tree)
            mx.eval(forward_value_tensor)

            task_block = _mask_flat_to_stage_d_block(flat_task_grads, layer_set, block_index, cfg.num_policies, len(blocks))
            forward_block = _mask_flat_to_stage_d_block(forward_grads, layer_set, block_index, cfg.num_policies, len(blocks))
            safety_block = _mask_flat_to_stage_d_block(safety_grads, layer_set, block_index, cfg.num_policies, len(blocks))
            retain_block = _mask_flat_to_stage_d_block(retain_grads, layer_set, block_index, cfg.num_policies, len(blocks))

            if route_only_phase:
                task_block = _mask_policy_only(task_block, block_index, cfg.num_policies, len(blocks))
                forward_block = _mask_policy_only(forward_block, block_index, cfg.num_policies, len(blocks))
                safety_block = _mask_policy_only(safety_block, block_index, cfg.num_policies, len(blocks))
                retain_block = _mask_policy_only(retain_block, block_index, cfg.num_policies, len(blocks))

            task_norm = _block_norm(task_block)
            forward_norm = _block_norm(forward_block)
            safety_norm = _block_norm(safety_block)
            retain_norm = _block_norm(retain_block)
            if task_norm > 0.0:
                saw_task_blocks.add(block_index)
            if forward_norm > 0.0:
                saw_forward_blocks.add(block_index)

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

            residual = _route_residual(model, probe_tokens, block_index, target_route)
            block_state = optimizer.get_constraint_info()
            step_blocks.append(
                {
                    "block_index": block_index,
                    "layers": block["layers"],
                    "layer_n_params": block["layer_n_params"],
                    "policy_n_params": block["policy_n_params"],
                    "route_only_phase": route_only_phase,
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
            _probe_logits, refreshed_route_map, _probe_route_logits, _probe_choice_scores = model.forward_with_policy_states(probe_array)
            forward_targets = {block_idx: route for block_idx, route in refreshed_route_map.items()}
            refresh_steps.append(step_idx + 1)
            refreshed = True

        step_trace.append(
            {
                "step": step_idx + 1,
                "route_only_phase": route_only_phase,
                "task_loss": scalar(task_value_tensor),
                "safety_raw": float(safety_value - cfg.safety_margin),
                "retain_raw": float(retain_value - cfg.retain_margin),
                "blocks": step_blocks,
                "refreshed_forward_targets": refreshed,
            }
        )

    post_metrics = {
        "task_acc": _evaluate_local(model, tokenizer, task.eval_samples, choice_token_ids, eval_cfg, branch_name="task"),
        "safety_acc": _evaluate_local(model, tokenizer, safety_samples[: cfg.safety_eval_size], choice_token_ids, eval_cfg, branch_name="safety"),
        "retain_acc": _evaluate_local(model, tokenizer, retain_samples[: cfg.retain_eval_size], choice_token_ids, eval_cfg, branch_name="memory"),
    }
    post_route_metrics = _evaluate_route_metrics(model, tokenizer, task.eval_samples, eval_cfg)
    _probe_logits, final_route_map, _probe_route_logits, _probe_choice_scores = model.forward_with_policy_states(probe_array)
    final_block_residuals = {}
    for block in blocks:
        block_index = block["block_index"]
        diff = final_route_map[block_index] - forward_targets[block_index]
        final_block_residuals[block_index] = scalar(mx.mean(diff * diff))

    coord_mode_metrics = _evaluate_route_modes(
        model,
        tokenizer,
        task,
        safety_samples[: cfg.safety_eval_size],
        retain_samples[: cfg.retain_eval_size],
        choice_token_ids,
        eval_cfg,
    )
    branch_matrix = _evaluate_branch_matrix(
        model,
        tokenizer,
        {
            "task": task.eval_samples,
            "safety": safety_samples[: cfg.safety_eval_size],
            "retain": retain_samples[: cfg.retain_eval_size],
        },
        choice_token_ids,
        eval_cfg,
    )
    branch_specialization = _branch_specialization_summary(branch_matrix)
    necessity_gaps = {
        "task_zero_gap": coord_mode_metrics["normal"]["task_acc"] - coord_mode_metrics["zero"]["task_acc"],
        "task_scramble_gap": coord_mode_metrics["normal"]["task_acc"] - coord_mode_metrics["scramble"]["task_acc"],
        "safety_zero_gap": coord_mode_metrics["normal"]["safety_acc"] - coord_mode_metrics["zero"]["safety_acc"],
        "safety_scramble_gap": coord_mode_metrics["normal"]["safety_acc"] - coord_mode_metrics["scramble"]["safety_acc"],
        "retain_zero_gap": coord_mode_metrics["normal"]["retain_acc"] - coord_mode_metrics["zero"]["retain_acc"],
        "retain_scramble_gap": coord_mode_metrics["normal"]["retain_acc"] - coord_mode_metrics["scramble"]["retain_acc"],
    }

    all_scalars = []
    for entry in step_trace:
        all_scalars.extend([entry["task_loss"], entry["safety_raw"], entry["retain_raw"]])
        for block in entry["blocks"]:
            all_scalars.extend(
                [
                    block["task_grad_norm"],
                    block["forward_grad_norm"],
                    block["safety_grad_norm"],
                    block["retain_grad_norm"],
                    block["forward_raw"],
                    block["forward_residual"],
                ]
            )
    success_checks = {
        "finite_step_scalars": all(math.isfinite(value) for value in all_scalars),
        "all_blocks_have_params": all(block["n_params"] > 0 for block in blocks),
        "all_blocks_receive_forward_grad": len(saw_forward_blocks) == len(blocks),
        "all_blocks_receive_task_grad": len(saw_task_blocks) == len(blocks),
        "forward_targets_refreshed": len(refresh_steps) >= 1,
        "d_over_k_above_10": min(d_over_k_by_block.values()) >= 10.0,
        "route_readout_above_chance": post_route_metrics["route_acc"] > (1.0 / cfg.num_policies),
        "positive_route_zero_gap": necessity_gaps["task_zero_gap"] > 0.0,
        "positive_route_scramble_gap": necessity_gaps["task_scramble_gap"] > 0.0,
    }

    summary = {
        "config": asdict(cfg),
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
        "pre_route_metrics": pre_route_metrics,
        "post_route_metrics": post_route_metrics,
        "model": {
            "num_layers": len(layer_counts),
            "hidden_size": int(model.args.hidden_size),
            "architecture": "MacroBlock native policy carrier",
            "num_policies": cfg.num_policies,
            "expert_rank": cfg.expert_rank,
            "policy_scale": cfg.policy_scale,
            "distill_weight": cfg.distill_weight,
            "route_task_weight": cfg.route_task_weight,
            "route_entropy_weight": cfg.route_entropy_weight,
            "route_block_weight_power": cfg.route_block_weight_power,
            "safety_branch_weight": cfg.safety_branch_weight,
            "memory_branch_weight": cfg.memory_branch_weight,
            "branch_preference_weight": cfg.branch_preference_weight,
            "branch_preference_margin": cfg.branch_preference_margin,
            "task_branch_preference": cfg.task_branch_preference,
            "task_shadow_suppression_weight": cfg.task_shadow_suppression_weight,
            "task_shadow_suppression_margin": cfg.task_shadow_suppression_margin,
            "safety_shadow_suppression_weight": cfg.safety_shadow_suppression_weight,
            "safety_shadow_suppression_margin": cfg.safety_shadow_suppression_margin,
            "memory_shadow_suppression_weight": cfg.memory_shadow_suppression_weight,
            "memory_shadow_suppression_margin": cfg.memory_shadow_suppression_margin,
            "output_expert_scale": cfg.output_expert_scale,
            "base_choice_scale": cfg.base_choice_scale,
            "typed_output_branches": cfg.typed_output_branches,
            "hard_routing": cfg.hard_routing,
            "policy_only_warmup_steps": cfg.policy_only_warmup_steps,
            "probe_prompt_length": len(probe_tokens),
        },
        "blocks": blocks,
        "d_over_k_by_block": d_over_k_by_block,
        "modeled_local_solve": _modeled_block_local_stats(blocks, k_policy),
        "constraint_setup": {
            "forward_constraint": f"per-block policy-state MSE on the probe prompt (num_policies={cfg.num_policies})",
            "safety_constraint": "KL(student || anchor) over safety replay anchor snapshot, block-masked",
            "retain_constraint": "KL(student || anchor) over retain replay anchor snapshot, block-masked",
            "distill_objective": f"teacher choice KL weight={cfg.distill_weight}" if cfg.distill_weight > 0.0 else "disabled",
            "route_objective": f"all-block route CE weight={cfg.route_task_weight}, power={cfg.route_block_weight_power}" if cfg.route_task_weight > 0.0 else "disabled",
            "route_entropy_objective": f"all-block route entropy weight={cfg.route_entropy_weight}, power={cfg.route_block_weight_power}" if cfg.route_entropy_weight > 0.0 else "disabled",
            "safety_branch_objective": (
                f"safety-branch replay CE weight={cfg.safety_branch_weight}" if cfg.typed_output_branches and cfg.safety_branch_weight > 0.0 else "disabled"
            ),
            "memory_branch_objective": (
                f"memory-branch replay CE weight={cfg.memory_branch_weight}" if cfg.typed_output_branches and cfg.memory_branch_weight > 0.0 else "disabled"
            ),
            "branch_preference_objective": (
                f"preferred-branch margin weight={cfg.branch_preference_weight}, margin={cfg.branch_preference_margin}, task_enabled={cfg.task_branch_preference}"
                if cfg.typed_output_branches and cfg.branch_preference_weight > 0.0
                else "disabled"
            ),
            "task_shadow_suppression_objective": (
                f"safety/memory must beat task branch weight={cfg.task_shadow_suppression_weight}, margin={cfg.task_shadow_suppression_margin}"
                if cfg.typed_output_branches and cfg.task_shadow_suppression_weight > 0.0
                else "disabled"
            ),
            "safety_shadow_suppression_objective": (
                f"safety must beat task branch weight={cfg.safety_shadow_suppression_weight}, margin={cfg.safety_shadow_suppression_margin}"
                if cfg.typed_output_branches and cfg.safety_shadow_suppression_weight > 0.0
                else "disabled"
            ),
            "memory_shadow_suppression_objective": (
                f"memory must beat task branch weight={cfg.memory_shadow_suppression_weight}, margin={cfg.memory_shadow_suppression_margin}"
                if cfg.typed_output_branches and cfg.memory_shadow_suppression_weight > 0.0
                else "disabled"
            ),
            "output_experts": (
                f"typed route-conditioned output choice experts scale={cfg.output_expert_scale}"
                if cfg.output_expert_scale > 0.0 and cfg.typed_output_branches
                else f"route-conditioned output choice experts scale={cfg.output_expert_scale}"
                if cfg.output_expert_scale > 0.0
                else "disabled"
            ),
            "base_choice_path": f"dense base choice scores scaled by {cfg.base_choice_scale}",
        },
        "refresh_steps": refresh_steps,
        "final_block_forward_residuals": final_block_residuals,
        "final_block_forward_residual_stats": _summary_stats(list(final_block_residuals.values())),
        "route_mode_metrics": coord_mode_metrics,
        "branch_matrix": branch_matrix,
        "branch_specialization": branch_specialization,
        "necessity_gaps": necessity_gaps,
        "step_trace": step_trace,
        "success_checks": success_checks,
        "overall_pass": all(success_checks.values()),
        "architecture_note": (
            "Each macro-block consumes a narrow routing state through specialized expert adapters before local Transformer computation, then updates the routing state from the block summary. "
            "Optional route-conditioned output experts inject branch-specific choice scores so task, safety, and memory paths can specialize on the same routing carrier."
            if cfg.typed_output_branches
            else "Each macro-block consumes a narrow routing state through specialized expert adapters before local Transformer computation, then updates the routing state from the block summary. Optional route-conditioned output experts can inject task-bearing choice scores directly so route can affect final decisions instead of only hidden features."
        ),
    }

    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[stage-d-native-policy] wrote {out_path}")
    print(
        f"[stage-d-native-policy] pre task/safety/retain = "
        f"{pre_metrics['task_acc']:.3f}/{pre_metrics['safety_acc']:.3f}/{pre_metrics['retain_acc']:.3f}"
    )
    print(
        f"[stage-d-native-policy] post task/safety/retain = "
        f"{post_metrics['task_acc']:.3f}/{post_metrics['safety_acc']:.3f}/{post_metrics['retain_acc']:.3f}"
    )
    print(
        f"[stage-d-native-policy] route normal/zero/scramble task = "
        f"{coord_mode_metrics['normal']['task_acc']:.3f}/"
        f"{coord_mode_metrics['zero']['task_acc']:.3f}/"
        f"{coord_mode_metrics['scramble']['task_acc']:.3f}"
    )
    print(
        f"[stage-d-native-policy] blocks={len(blocks)} min d/k={min(d_over_k_by_block.values()):.1f} "
        f"refresh={refresh_steps} overall_pass={summary['overall_pass']}"
    )
    return summary


def parse_args() -> StageDNativePolicyConfig:
    parser = argparse.ArgumentParser(description="Stage D native policy smoke on real Qwen-MLX.")
    parser.add_argument("--model-name", default=BlockLocalConfig.model_name)
    parser.add_argument("--dataset-source", default="ag_news")
    parser.add_argument("--local-files-only", dest="local_files_only", action="store_true")
    parser.add_argument("--allow-online-hf-load", dest="local_files_only", action="store_false")
    parser.set_defaults(local_files_only=True)
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
    parser.add_argument("--smoke-steps", type=int, default=8)
    parser.add_argument("--outer-step-freq", type=int, default=4)
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
    parser.add_argument("--num-policies", type=int, default=4)
    parser.add_argument("--expert-rank", type=int, default=16)
    parser.add_argument("--policy-scale", type=float, default=0.08)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--route-task-weight", type=float, default=0.5)
    parser.add_argument("--route-entropy-weight", type=float, default=0.01)
    parser.add_argument("--route-block-weight-power", type=float, default=2.0)
    parser.add_argument("--safety-branch-weight", type=float, default=0.0)
    parser.add_argument("--memory-branch-weight", type=float, default=0.0)
    parser.add_argument("--branch-preference-weight", type=float, default=0.0)
    parser.add_argument("--branch-preference-margin", type=float, default=0.0)
    parser.add_argument("--skip-task-branch-preference", action="store_true")
    parser.add_argument("--task-shadow-suppression-weight", type=float, default=0.0)
    parser.add_argument("--task-shadow-suppression-margin", type=float, default=0.0)
    parser.add_argument("--safety-shadow-suppression-weight", type=float, default=0.0)
    parser.add_argument("--safety-shadow-suppression-margin", type=float, default=0.0)
    parser.add_argument("--memory-shadow-suppression-weight", type=float, default=0.0)
    parser.add_argument("--memory-shadow-suppression-margin", type=float, default=0.0)
    parser.add_argument("--output-expert-scale", type=float, default=0.0)
    parser.add_argument("--base-choice-scale", type=float, default=1.0)
    parser.add_argument("--typed-output-branches", action="store_true")
    parser.add_argument("--soft-routing", action="store_true")
    parser.add_argument("--policy-only-warmup-steps", type=int, default=4)
    parser.add_argument("--output", default=str(RESULTS_DIR / "summary.json"))
    args = parser.parse_args()
    return StageDNativePolicyConfig(
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
        num_policies=args.num_policies,
        expert_rank=args.expert_rank,
        policy_scale=args.policy_scale,
        distill_weight=args.distill_weight,
        route_task_weight=args.route_task_weight,
        route_entropy_weight=args.route_entropy_weight,
        route_block_weight_power=args.route_block_weight_power,
        safety_branch_weight=args.safety_branch_weight,
        memory_branch_weight=args.memory_branch_weight,
        branch_preference_weight=args.branch_preference_weight,
        branch_preference_margin=args.branch_preference_margin,
        task_branch_preference=not args.skip_task_branch_preference,
        task_shadow_suppression_weight=args.task_shadow_suppression_weight,
        task_shadow_suppression_margin=args.task_shadow_suppression_margin,
        safety_shadow_suppression_weight=args.safety_shadow_suppression_weight,
        safety_shadow_suppression_margin=args.safety_shadow_suppression_margin,
        memory_shadow_suppression_weight=args.memory_shadow_suppression_weight,
        memory_shadow_suppression_margin=args.memory_shadow_suppression_margin,
        output_expert_scale=args.output_expert_scale,
        base_choice_scale=args.base_choice_scale,
        typed_output_branches=args.typed_output_branches,
        hard_routing=not args.soft_routing,
        policy_only_warmup_steps=args.policy_only_warmup_steps,
        output=args.output,
    )


if __name__ == "__main__":
    run_stage_d_native_policy_smoke(parse_args())
