# GALT Initial GitHub Release

This directory is a curated first public snapshot of the project. It keeps the
paper and the smallest set of experiment code and result artifacts needed to
understand the current claim:

> GALT is not just a constrained optimizer; it is the beginning of a broader
> training regime in which task, safety, and memory can start to live in typed,
> routed, increasingly necessary internal channels.

## What is included

- `paper/galt.pdf`
  - the current GALT-first manuscript
- `experiments/galt/stage_d_native_policy_smoke_mlx.py`
  - the key real-carrier Stage D validation script
- `experiments/galt/{hidden_collector_mlx,phase_d_smoke_mlx,phase_d_block_local_mlx}.py`
  - the direct dependency chain needed by the Stage D script
- `experiments/shared_runtime/continual_runtime_mlx.py`
  - shared MLX continual-learning helpers used by the public scripts
- `results/galt_prework/stage_d_native_policy_smoke/`
  - selected typed-channel result files, including the strongest single-seed run
    and the small multiseed check
- `paper/galt_prework/`
  - short experiment reports for typed branches, anti-shadowing, and multiseed
- `experiments/precursor_validation/safety_from_birth_mlx.py`
  - the Safety-from-Birth precursor validation path
- `results/multi_constraint_4task_3seed.json`
  - the AG News 4-task x 3-seed precursor result summary
- `experiments/precursor_validation/split_mnist_mlx.py`
  - the Split-MNIST precursor validation path
- `experiments/precursor_validation/split_mnist_model_mlx.py`
  - the compact model definition used by the Split-MNIST script
- `results/dense_mnist_mlx_hybrid_hessian_smoke.json`
  - the compact MLX Split-MNIST result artifact

The copied support modules under `optimizer/`, `data_utils/`, `prompts/`, and
the included experiment subdirectories preserve the original relative imports so
the bundled scripts can still be inspected and run from inside `release/`. The
`experiments/galt/` subtree is intentionally trimmed to only the Stage D script
and its immediate runtime dependencies.

## Recommended reading order

1. `paper/galt.pdf`
2. `paper/galt_prework/27_stage_d_typed_multiseed_report.md`
3. `results/galt_prework/stage_d_native_policy_smoke/typed_output_branches_lr2e4_w05.json`
4. `results/galt_prework/stage_d_native_policy_smoke/typed_multiseed_baseline_seed41.json`

## Key result files

### GALT Stage D typed-channel line

- `results/galt_prework/stage_d_native_policy_smoke/typed_output_branches_lr2e4_w05.json`
  - strongest single-seed typed routed result
- `results/galt_prework/stage_d_native_policy_smoke/typed_output_branches_safetyshadow05_m01.json`
  - safety-only anti-shadowing result
- `results/galt_prework/stage_d_native_policy_smoke/typed_multiseed_baseline_seed41.json`
- `results/galt_prework/stage_d_native_policy_smoke/typed_multiseed_baseline_seed42.json`
- `results/galt_prework/stage_d_native_policy_smoke/typed_multiseed_baseline_seed43.json`
- `results/galt_prework/stage_d_native_policy_smoke/typed_multiseed_safetyshadow_seed41.json`
- `results/galt_prework/stage_d_native_policy_smoke/typed_multiseed_safetyshadow_seed42.json`
- `results/galt_prework/stage_d_native_policy_smoke/typed_multiseed_safetyshadow_seed43.json`

### CSAT / precursor validation line

- `results/multi_constraint_4task_3seed.json`
  - AG News Safety-from-Birth precursor comparison
- `results/dense_mnist_mlx_hybrid_hessian_smoke.json`
  - compact Split-MNIST proof-of-concept result

## Minimal run commands

Install dependencies:

```bash
pip install -r requirements-mlx.txt
```

Run the key GALT validation:

```bash
python experiments/galt/stage_d_native_policy_smoke_mlx.py \
  --allow-online-hf-load \
  --typed-output-branches \
  --output-expert-scale 2.0 \
  --base-choice-scale 0.0 \
  --route-task-weight 4.0 \
  --route-entropy-weight 0.02 \
  --safety-branch-weight 0.5 \
  --memory-branch-weight 0.5 \
  --policy-only-warmup-steps 40 \
  --smoke-steps 40 \
  --seed 42
```

Run the precursor Safety-from-Birth validation:

```bash
python experiments/precursor_validation/safety_from_birth_mlx.py \
  --dataset-source ag_news \
  --categories batch_0 batch_1 batch_2 batch_3 \
  --max-train-per-task 64 \
  --max-eval-per-task 64 \
  --allow-online-hf-load
```

Run the compact Split-MNIST proof of concept:

```bash
python experiments/precursor_validation/split_mnist_mlx.py --allow-online-hf-load
```

## Notes

- This is a curated release snapshot, not the full historical repository.
- Run commands from inside the `release/` directory so the preserved relative
  imports resolve correctly.
- `experiments/galt/` is the public-facing Stage D folder name for this release.
- `experiments/shared_runtime/` holds shared helper code rather than a separate
  narrative line.
- `experiments/precursor_validation/` collects non-mainline historical
  validation scripts that still support the paper's precursor story.
- The Stage D MLX script defaults to `--local-files-only`; pass
  `--allow-online-hf-load` if you want it to fetch the model when needed.
- The main public story should center on the paper plus the Stage D typed-channel
  evidence.
- The Safety-from-Birth and Split-MNIST paths are included as important
  precursor validation, not as the headline architectural claim.
