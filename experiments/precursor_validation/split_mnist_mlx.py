"""MLX/Mac Split-MNIST continual-learning experiment using AVBD-Hessian."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

PORTABLE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORTABLE_ROOT / "optimizer"))

from avbd_hessian_optimizer_mlx import AVBDHessianOptimizer
from mlx_utils import clone_flat_dict, flatten_tree, scalar, unflatten_tree
from split_mnist_model_mlx import DenseAVBDSystem, LoRAMLP
from refresh_scheduler import RefreshConfig, RefreshScheduler


def cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(log_probs, labels.reshape((-1, 1)), axis=-1)
    return -picked.mean()


def compute_distillation_kl(student_logits: mx.array, teacher_probs: mx.array, temperature: float = 1.0) -> mx.array:
    scaled_logits = student_logits / temperature
    student_log_probs = scaled_logits - mx.logsumexp(scaled_logits, axis=-1, keepdims=True)
    teacher_probs = mx.array(teacher_probs, dtype=student_log_probs.dtype)
    teacher_log_probs = mx.log(mx.maximum(teacher_probs, 1e-12))
    kl = mx.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    return kl.mean() * (temperature**2)


def _load_mnist_arrays(local_files_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from datasets import DownloadConfig, load_dataset
    from torchvision import datasets as tv_datasets

    download_config = DownloadConfig(local_files_only=local_files_only)
    try:
        dataset = load_dataset("ylecun/mnist", download_config=download_config)
    except Exception:
        dataset = None

    def normalize_images(images: np.ndarray) -> np.ndarray:
        images = images.astype(np.float32) / 255.0
        images = (images - 0.1307) / 0.3081
        return images.reshape(images.shape[0], -1)

    def convert_hf_split(split_name: str) -> tuple[np.ndarray, np.ndarray]:
        images = []
        labels = []
        for sample in dataset[split_name]:
            images.append(np.asarray(sample["image"], dtype=np.float32))
            labels.append(int(sample["label"]))
        return normalize_images(np.stack(images)), np.asarray(labels, dtype=np.int32)

    if dataset is not None:
        train_x, train_y = convert_hf_split("train")
        test_x, test_y = convert_hf_split("test")
        return train_x, train_y, test_x, test_y

    try:
        data_dir = PORTABLE_ROOT / "data"
        train = tv_datasets.MNIST(root=str(data_dir), train=True, download=not local_files_only)
        test = tv_datasets.MNIST(root=str(data_dir), train=False, download=not local_files_only)
    except Exception as exc:
        if local_files_only:
            raise RuntimeError(
                "MNIST is not available locally through either Hugging Face datasets or torchvision. "
                "Rerun with --allow-online-hf-load after confirming network access."
            ) from exc
        raise RuntimeError("Failed to load MNIST through both Hugging Face datasets and torchvision.") from exc

    train_x = normalize_images(train.data.numpy())
    train_y = train.targets.numpy().astype(np.int32)
    test_x = normalize_images(test.data.numpy())
    test_y = test.targets.numpy().astype(np.int32)
    return train_x, train_y, test_x, test_y


def get_split_mnist(num_tasks: int = 5, local_files_only: bool = False):
    train_x, train_y, test_x, test_y = _load_mnist_arrays(local_files_only)
    tasks = []
    classes_per_group = 10 // num_tasks
    for task_index in range(num_tasks):
        classes = list(range(task_index * classes_per_group, (task_index + 1) * classes_per_group))
        train_mask = np.isin(train_y, classes)
        test_mask = np.isin(test_y, classes)
        tasks.append(
            {
                "name": f"Task {task_index} (digits {classes})",
                "classes": classes,
                "train_x": train_x[train_mask],
                "train_y": train_y[train_mask],
                "test_x": test_x[test_mask],
                "test_y": test_y[test_mask],
            }
        )
    return tasks, train_x, train_y, test_x, test_y


def iter_array_batches(xs, ys, batch_size: int, shuffle: bool, seed: int):
    indices = np.arange(len(xs))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield mx.array(xs[batch_indices]), mx.array(ys[batch_indices])


def make_replay_buffer(xs, ys, size: int, seed: int):
    rng = np.random.default_rng(seed)
    count = min(size, len(xs))
    indices = rng.choice(len(xs), size=count, replace=False)
    return mx.array(xs[indices]), mx.array(ys[indices])


def evaluate(model, xs, ys, batch_size: int = 256) -> float:
    correct = 0
    total = 0
    for batch_x, batch_y in iter_array_batches(xs, ys, batch_size, shuffle=False, seed=0):
        logits = model(batch_x)
        predictions = mx.argmax(logits, axis=-1)
        correct += int(mx.sum(predictions == batch_y).item())
        total += batch_y.shape[0]
    return correct / max(total, 1)


def pretrain(model: LoRAMLP, train_x, train_y, epochs: int = 3, batch_size: int = 128):
    optimizer = optim.Adam(learning_rate=1e-3)
    loss_grad_fn = nn.value_and_grad(model, lambda module, x, y: cross_entropy(module(x), y))
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for batch_x, batch_y in iter_array_batches(train_x, train_y, batch_size, shuffle=True, seed=epoch):
            loss, grads = loss_grad_fn(model, batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += scalar(loss)
            count += 1
        print(f"  Pretrain epoch {epoch + 1}/{epochs}  loss={total_loss / max(count, 1):.4f}")


def train_adam(model: LoRAMLP, tasks, epochs_per_task: int = 5, lr: float = 1e-3):
    results = {"accs_after_task": [], "name": "Adam+LoRA"}
    optimizer = optim.Adam(learning_rate=lr)
    loss_grad_fn = nn.value_and_grad(model, lambda module, x, y: cross_entropy(module(x), y))
    total_steps = 0

    for task_index, task in enumerate(tasks):
        print(f"\n  [Adam] {task['name']} ...")
        for epoch in range(epochs_per_task):
            for batch_x, batch_y in iter_array_batches(
                task["train_x"], task["train_y"], 64, shuffle=True, seed=task_index * 100 + epoch
            ):
                _loss, grads = loss_grad_fn(model, batch_x, batch_y)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                total_steps += 1

        accs = [evaluate(model, tasks[index]["test_x"], tasks[index]["test_y"]) for index in range(task_index + 1)]
        results["accs_after_task"].append(accs)
        print(f"  [Adam] accs={format_accs(accs)}  avg={np.mean(accs):.3f}")

    results["total_backprop_calls"] = total_steps
    return results


def train_ewc(model: LoRAMLP, tasks, epochs_per_task: int = 5, lr: float = 1e-3, ewc_lambda: float = 400.0):
    results = {"accs_after_task": [], "name": "Adam+LoRA+EWC"}
    optimizer = optim.Adam(learning_rate=lr)
    fisher = {}
    anchors = {}
    total_steps = 0

    def loss_fn(module, x, y):
        loss = cross_entropy(module(x), y)
        if fisher:
            for name, param in flatten_tree(module.trainable_parameters()).items():
                if name in fisher:
                    diff = param - anchors[name]
                    loss = loss + ewc_lambda * mx.sum(fisher[name] * diff * diff)
        return loss

    loss_grad_fn = nn.value_and_grad(model, loss_fn)
    fisher_grad_fn = nn.value_and_grad(model, lambda module, x, y: cross_entropy(module(x), y))

    for task_index, task in enumerate(tasks):
        print(f"\n  [EWC] {task['name']} ...")
        for epoch in range(epochs_per_task):
            for batch_x, batch_y in iter_array_batches(
                task["train_x"], task["train_y"], 64, shuffle=True, seed=task_index * 100 + epoch
            ):
                _loss, grads = loss_grad_fn(model, batch_x, batch_y)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                total_steps += 1

        fisher_accumulator = {}
        fisher_batches = 0
        for batch_x, batch_y in iter_array_batches(
            task["train_x"], task["train_y"], 64, shuffle=True, seed=task_index + 999
        ):
            _loss, grads = fisher_grad_fn(model, batch_x, batch_y)
            for name, grad in flatten_tree(grads).items():
                fisher_accumulator[name] = fisher_accumulator.get(name, mx.zeros_like(grad)) + grad * grad
            fisher_batches += 1
            if fisher_batches >= 20:
                break

        fisher = {name: value / max(fisher_batches, 1) for name, value in fisher_accumulator.items()}
        anchors = clone_flat_dict(flatten_tree(model.trainable_parameters()))

        accs = [evaluate(model, tasks[index]["test_x"], tasks[index]["test_y"]) for index in range(task_index + 1)]
        results["accs_after_task"].append(accs)
        print(f"  [EWC] accs={format_accs(accs)}  avg={np.mean(accs):.3f}")

    results["total_backprop_calls"] = total_steps
    return results


def compute_replay_anchor_probs(model, replay_x: mx.array, temperature: float = 1.0) -> mx.array:
    logits = model(replay_x)
    return mx.softmax(logits / temperature, axis=-1)


def partition_system_grads(flat_grads: dict[str, mx.array]):
    model_grads = {}
    local_head_grads = {}
    for name, grad in flat_grads.items():
        if name.startswith("model."):
            model_grads[name[len("model."):]] = grad
        elif name.startswith("local_heads."):
            local_head_grads[name[len("local_heads."):]] = grad
    return model_grads, local_head_grads


def global_system_loss(system: DenseAVBDSystem, batch_x: mx.array, batch_y: mx.array):
    total_loss = cross_entropy(system.model(batch_x), batch_y)
    hidden = mx.stop_gradient(batch_x.reshape((batch_x.shape[0], -1)))
    for index, layer in enumerate(system.model.layers):
        hidden = mx.stop_gradient(mx.maximum(layer(hidden), 0.0))
        total_loss = total_loss + cross_entropy(system.local_heads.heads[index](hidden), batch_y)
    return total_loss


def local_system_loss(system: DenseAVBDSystem, batch_x: mx.array, batch_y: mx.array):
    pre_acts = system.model.get_pre_activations(batch_x)
    total_loss = mx.array(0.0)
    for index, layer in enumerate(system.model.layers):
        hidden_in = pre_acts[index]
        base_out = mx.stop_gradient(layer.base(hidden_in))
        lora_out = layer.lora_delta(hidden_in)
        hidden_out = mx.maximum(base_out + lora_out, 0.0)
        total_loss = total_loss + cross_entropy(system.local_heads.heads[index](hidden_out), batch_y)
    total_loss = total_loss + cross_entropy(system.model.head(pre_acts[-1]), batch_y)
    return total_loss


def eval_constraint_states(model, replay_entries, temperature: float = 1.0, margin: float = 0.02):
    raw_values = []
    violations = []
    for entry in replay_entries:
        logits = model(entry["x"])
        kl = compute_distillation_kl(
            logits,
            entry["anchor_probs"],
            temperature=temperature,
        )
        raw_value = scalar(kl) - margin
        raw_values.append(raw_value)
        violations.append(max(0.0, raw_value))
    return raw_values, violations


def compute_constraint_grads(model, replay_entries, temperature: float = 1.0):
    constraint_grads = {}
    for entry in replay_entries:
        def constraint_loss(module, replay_x=entry["x"], anchor_probs=entry["anchor_probs"]):
            logits = module(replay_x)
            return compute_distillation_kl(
                logits,
                anchor_probs,
                temperature=temperature,
            )

        constraint_grad_fn = nn.value_and_grad(model, constraint_loss)
        _kl_value, grads = constraint_grad_fn(model)
        constraint_grads[entry["ci"]] = flatten_tree(grads)
    return constraint_grads


def train_avbd_hessian_lowbp(
    model: LoRAMLP,
    tasks,
    epochs_per_task: int = 5,
    lr: float = 1e-3,
    replay_size: int = 200,
    replay_margin: float = 0.02,
    constraint_temperature: float = 1.0,
):
    results = {"accs_after_task": [], "name": "AVBD-Hessian-LowBP"}
    system = DenseAVBDSystem(model)
    mx.eval(system.parameters())
    optimizer = AVBDHessianOptimizer(
        system.model,
        lr=lr,
        rho_init=1.0,
        rho_max=2.0,
        rho_growth=1.5,
    )
    scheduler = RefreshScheduler(RefreshConfig(refresh_period=10, refresh_cstr_trigger=0.3))
    local_head_optimizer = optim.Adam(learning_rate=1e-3)
    global_grad_fn = nn.value_and_grad(system, global_system_loss)
    local_grad_fn = nn.value_and_grad(system, local_system_loss)
    replay_entries = []
    cached_constraint_grads = {}
    total_steps = 0

    for task_index, task in enumerate(tasks):
        print(f"\n  [AVBD-Hessian-LowBP] {task['name']} ...")
        for epoch in range(epochs_per_task):
            for batch_x, batch_y in iter_array_batches(
                task["train_x"], task["train_y"], 64, shuffle=True, seed=task_index * 100 + epoch
            ):
                raw_constraint_values, violations = eval_constraint_states(
                    system.model,
                    replay_entries,
                    temperature=constraint_temperature,
                    margin=replay_margin,
                )
                is_global = scheduler.needs_refresh(violations)
                if any(entry["ci"] not in cached_constraint_grads for entry in replay_entries):
                    is_global = True

                if is_global:
                    _loss, grads = global_grad_fn(system, batch_x, batch_y)
                    cached_constraint_grads = compute_constraint_grads(
                        system.model,
                        replay_entries,
                        temperature=constraint_temperature,
                    )
                else:
                    _loss, grads = local_grad_fn(system, batch_x, batch_y)

                model_grads, local_head_grads = partition_system_grads(flatten_tree(grads))
                if local_head_grads:
                    local_head_optimizer.update(system.local_heads, unflatten_tree(local_head_grads))
                for entry, raw_value in zip(replay_entries, raw_constraint_values):
                    optimizer.set_constraint_grads(
                        entry["ci"],
                        raw_value,
                        cached_constraint_grads.get(entry["ci"], {}),
                    )

                optimizer.step(model_grads)
                scheduler.mark_step(is_global)
                total_steps += 1
                mx.eval(system.parameters(), local_head_optimizer.state)

        if task_index < len(tasks) - 1:
            replay_x, replay_y = make_replay_buffer(
                task["train_x"],
                task["train_y"],
                size=replay_size,
                seed=task_index + 123,
            )
            anchor_probs = compute_replay_anchor_probs(
                system.model,
                replay_x,
                temperature=constraint_temperature,
            )
            ci = optimizer.add_constraint(f"retain_task_{task_index}")
            replay_entries.append(
                {
                    "ci": ci,
                    "x": replay_x,
                    "y": replay_y,
                    "anchor_probs": anchor_probs,
                }
            )

        accs = [
            evaluate(system.model, tasks[index]["test_x"], tasks[index]["test_y"])
            for index in range(task_index + 1)
        ]
        results["accs_after_task"].append(accs)
        constraint_info = optimizer.get_constraint_info()
        lambda_str = ", ".join(f"{name}:λ={info['lambda_']:.3f}" for name, info in constraint_info.items())
        scheduler_stats = scheduler.stats()
        print(
            f"  [AVBD-Hessian-LowBP] accs={format_accs(accs)}  avg={np.mean(accs):.3f}"
        )
        print(
            f"                       global={scheduler_stats['global_backprop_calls']}  "
            f"local={scheduler_stats['local_only_steps']}"
        )
        if lambda_str:
            print(f"                       constraints: {lambda_str}")

    results["scheduler_stats"] = scheduler.stats()
    results["total_backprop_calls"] = results["scheduler_stats"]["global_backprop_calls"]
    results["total_optimizer_steps"] = total_steps
    results["constraint_info"] = optimizer.get_constraint_info()
    return results


def format_accs(accs):
    return "[" + ", ".join(f"{acc:.3f}" for acc in accs) + "]"


def average_forgetting(result, num_tasks: int) -> float:
    final = result["accs_after_task"][-1]
    forgetting = []
    for task_index in range(num_tasks - 1):
        best = result["accs_after_task"][task_index][task_index]
        forgetting.append(best - final[task_index])
    return float(np.mean(forgetting)) if forgetting else 0.0


def main():
    parser = argparse.ArgumentParser(description="MLX Split-MNIST continual-learning experiment.")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--epochs-per-task", type=int, default=5)
    parser.add_argument("--allow-online-hf-load", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    print(f"Device: {mx.default_device()}\n")
    mx.random.seed(42)
    np.random.seed(42)

    print("=== Loading Split-MNIST ===")
    tasks, full_train_x, full_train_y, full_test_x, full_test_y = get_split_mnist(
        num_tasks=args.num_tasks,
        local_files_only=not args.allow_online_hf_load,
    )
    for task in tasks:
        print(f"  {task['name']}  train={len(task['train_x'])}  test={len(task['test_x'])}")

    print("\n=== Pretraining backbone ===")
    base_model = LoRAMLP(lora_rank=4)
    mx.eval(base_model.parameters())
    pretrain(base_model, full_train_x, full_train_y, epochs=3)
    pretrained = clone_flat_dict(flatten_tree(base_model.parameters()))
    pre_acc = evaluate(base_model, full_test_x, full_test_y)
    print(f"  Pretrained full-MNIST accuracy: {pre_acc:.3f}")

    all_results = []
    experiment_matrix = [
        ("Adam+LoRA", lambda model: train_adam(model, tasks, epochs_per_task=args.epochs_per_task)),
        ("Adam+LoRA+EWC", lambda model: train_ewc(model, tasks, epochs_per_task=args.epochs_per_task)),
        ("AVBD-Hessian-LowBP", lambda model: train_avbd_hessian_lowbp(model, tasks, epochs_per_task=args.epochs_per_task)),
    ]

    for name, train_fn in experiment_matrix:
        print("\n" + "=" * 60)
        print(f"=== {name} ===")
        print("=" * 60)
        model = LoRAMLP(lora_rank=4)
        model.update(unflatten_tree(pretrained), strict=False)
        model.freeze_backbone()
        mx.eval(model.parameters())
        start = time.time()
        result = train_fn(model)
        result["wall_time"] = time.time() - start
        all_results.append(result)

    print("\n" + "=" * 60)
    print("=== FINAL COMPARISON ===")
    print("=" * 60)
    for result in all_results:
        final = result["accs_after_task"][-1]
        avg_acc = float(np.mean(final))
        avg_fgt = average_forgetting(result, len(tasks))
        print(f"\n{result['name']}  (wall {result['wall_time']:.1f}s)")
        print(f"  Final accs : {format_accs(final)}")
        print(f"  Avg acc    : {avg_acc:.3f}")
        print(f"  Avg forget : {avg_fgt:.3f}")
        if "constraint_info" in result:
            for name, info in result["constraint_info"].items():
                print(f"  {name}     : λ={info['lambda_']:.3f}  ρ={info['rho']:.3f}")
        if "scheduler_stats" in result:
            stats = result["scheduler_stats"]
            total = stats["global_backprop_calls"] + stats["local_only_steps"]
            print(
                f"  Global BP  : {stats['global_backprop_calls']}  "
                f"({stats['global_backprop_calls'] / max(1, total):.0%} of steps)"
            )
            print(f"  Local only : {stats['local_only_steps']}")
        else:
            print(f"  Total BP   : {result['total_backprop_calls']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
