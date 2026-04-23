"""Helpers for MLX parameter trees used by the Mac/MLX experiment path."""

from __future__ import annotations

from typing import Dict

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def flatten_tree(tree) -> Dict[str, mx.array]:
    if not tree:
        return {}
    return {name: value for name, value in tree_flatten(tree)}


def unflatten_tree(flat: Dict[str, mx.array]):
    if not flat:
        return {}
    return tree_unflatten(list(flat.items()))


def clone_flat_dict(flat: Dict[str, mx.array]) -> Dict[str, mx.array]:
    return {name: mx.array(value) for name, value in flat.items()}


def zeros_like_flat(flat: Dict[str, mx.array]) -> Dict[str, mx.array]:
    return {name: mx.zeros_like(value) for name, value in flat.items()}


def scalar(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(mx.array(value).item())


def filter_prefix(flat: Dict[str, mx.array], prefix: str) -> Dict[str, mx.array]:
    return {
        name: value
        for name, value in flat.items()
        if name.startswith(prefix)
    }


def strip_prefix(flat: Dict[str, mx.array], prefix: str) -> Dict[str, mx.array]:
    stripped = {}
    for name, value in flat.items():
        if name.startswith(prefix):
            stripped[name[len(prefix):]] = value
    return stripped

