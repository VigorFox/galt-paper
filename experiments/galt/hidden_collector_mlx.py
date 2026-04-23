"""Hidden-state collection wrapper for MLX TinyLlama-MoE (Mixtral).

GALT pre-work M2: thin wrapper exposing per-layer hidden states without modifying
the upstream `mlx_lm.models.mixtral` source. Used by V1-V3 multi-layer anchoring
constraints. Not imported by any public CSAT code path.

Design constraints (from anchoring_theory_v2.md §3):
- Batched: tokens shape (B, L) → returns dict[layer_idx -> (B, L, D)] hidden states.
- Gradients flow through hidden tensors back to model parameters.
- Optionally also returns logits in the same forward pass (single embed/attn pass).
- Captures hidden state at the OUTPUT of each requested decoder layer (post-residual,
  pre-final-RMSNorm). This is the convention used by HuggingFace `output_hidden_states`
  and is the right anchor surface for A1 (Coverage) constraints.

Public API:
- `forward_collect_hiddens(model, tokens, layer_indices, return_logits=False)`
- `LAYER_KEY_LAST = -1` sentinel resolves to the post-final-norm hidden (just before lm_head).
"""

from __future__ import annotations

from typing import Iterable

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask


LAYER_KEY_LAST = -1  # sentinel: post-final-RMSNorm hidden state (pre-lm_head)


def _get_core(model):
    """Return the inner MixtralModel (with .embed_tokens, .layers, .norm)."""
    return model.model if hasattr(model, "model") else model


def _resolve_layer_indices(num_layers: int, layer_indices: Iterable[int]) -> list[int]:
    resolved: list[int] = []
    seen: set[int] = set()
    for idx in layer_indices:
        if idx == LAYER_KEY_LAST:
            real = LAYER_KEY_LAST  # keep sentinel; handled separately
        else:
            real = int(idx)
            if real < 0:
                real = num_layers + real
            if not (0 <= real < num_layers):
                raise IndexError(f"layer index {idx} out of range for num_layers={num_layers}")
        if real in seen:
            continue
        seen.add(real)
        resolved.append(real)
    return resolved


def forward_collect_hiddens(
    model,
    tokens: mx.array,
    layer_indices: Iterable[int],
    return_logits: bool = False,
) -> tuple[dict[int, mx.array], mx.array | None]:
    """Run forward and capture hidden states at requested decoder layers.

    Args:
        model: TinyLlama-MoE Mixtral model (outer Model with .model and optional .lm_head).
        tokens: int32 token ids of shape (B, L).
        layer_indices: iterable of layer indices to capture (0-based; -1 = post-final-norm).
        return_logits: if True, also return logits (B, L, vocab) in same forward pass.

    Returns:
        (hidden_dict, logits_or_None) where hidden_dict[layer] has shape (B, L, hidden).

    Notes:
        - Layer indices use LAYER_KEY_LAST (=-1) for the post-final-norm hidden, NOT
          the last decoder layer's pre-norm output. Use num_layers-1 for the latter.
        - No KV cache used (training mode); attention mask reconstructed each call.
        - Gradients flow through all returned tensors.
    """
    core = _get_core(model)
    num_layers = len(core.layers)
    requested = _resolve_layer_indices(num_layers, layer_indices)
    requested_decoder = {i for i in requested if i != LAYER_KEY_LAST}
    want_post_norm = LAYER_KEY_LAST in requested

    if tokens.dtype != mx.int32:
        tokens = tokens.astype(mx.int32)

    h = core.embed_tokens(tokens)
    mask = create_attention_mask(h, None)

    captured: dict[int, mx.array] = {}
    for layer_index, layer in enumerate(core.layers):
        h = layer(h, mask, None)
        if layer_index in requested_decoder:
            captured[layer_index] = h

    h_norm = core.norm(h)
    if want_post_norm:
        captured[LAYER_KEY_LAST] = h_norm

    logits = None
    if return_logits:
        if hasattr(model, "args") and getattr(model.args, "tie_word_embeddings", False):
            logits = core.embed_tokens.as_linear(h_norm)
        elif hasattr(model, "lm_head"):
            logits = model.lm_head(h_norm)
        else:
            logits = core.embed_tokens.as_linear(h_norm)

    return captured, logits


def get_num_layers(model) -> int:
    return len(_get_core(model).layers)


def get_hidden_size(model) -> int:
    return int(model.args.hidden_size)
