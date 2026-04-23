"""MLX model definitions for the dense Split-MNIST experiment."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.lora_A = mx.random.normal(shape=(rank, in_features)) * (0.02 / rank**0.5)
        self.lora_B = mx.zeros((out_features, rank))
        self.scaling = 1.0 / rank

    def __call__(self, x):
        return self.base(x) + self.lora_delta(x)

    def lora_delta(self, x):
        return ((x @ self.lora_A.T) @ self.lora_B.T) * self.scaling

    def freeze_base(self):
        self.base.freeze()


class LoRAMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (256, 128),
        num_classes: int = 10,
        lora_rank: int = 4,
    ):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        dims = [input_dim] + self.hidden_dims
        self.layers = [
            LoRALinear(dims[index], dims[index + 1], rank=lora_rank)
            for index in range(len(dims) - 1)
        ]
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def __call__(self, x):
        hidden = x.reshape((x.shape[0], -1))
        for layer in self.layers:
            hidden = mx.maximum(layer(hidden), 0.0)
        return self.head(hidden)

    def get_pre_activations(self, x):
        hidden = x.reshape((x.shape[0], -1))
        pre_acts = [mx.stop_gradient(hidden)]
        for layer in self.layers:
            hidden = mx.maximum(layer(hidden), 0.0)
            pre_acts.append(mx.stop_gradient(hidden))
        return pre_acts

    def freeze_backbone(self):
        for layer in self.layers:
            layer.freeze_base()

    def get_lora_blocks(self, prefix: str = "") -> list[dict]:
        return [
            {
                "name": f"lora_{index}",
                "params": [
                    f"{prefix}layers.{index}.lora_A",
                    f"{prefix}layers.{index}.lora_B",
                ],
            }
            for index, _layer in enumerate(self.layers)
        ]

    def head_param_paths(self, prefix: str = "") -> list[str]:
        return [f"{prefix}head.weight", f"{prefix}head.bias"]


class LocalReadout(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def __call__(self, hidden):
        return self.fc(hidden)


class LocalHeadStack(nn.Module):
    def __init__(self, hidden_dims: list[int], num_classes: int):
        super().__init__()
        self.heads = [LocalReadout(dim, num_classes) for dim in hidden_dims]


class DenseAVBDSystem(nn.Module):
    def __init__(self, model: LoRAMLP, num_classes: int = 10):
        super().__init__()
        self.model = model
        self.local_heads = LocalHeadStack(model.hidden_dims, num_classes)
