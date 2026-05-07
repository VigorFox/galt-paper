# Curriculum Layer Heads Smoke Report

This pass moves the hierarchical memory direction into the MLX curriculum
model.

## Data Update

Updated:

```text
scripts/build_teacher_mode_curriculum_preflight.py
```

Contrast-expanded curriculum rows now include:

```text
metadata.memory_layer_id
metadata.violation_layer_id
metadata.hierarchical_access_target
```

Layer rule:

```text
violation_layer_id == 4 -> access
violation_layer_id <= memory_layer_id -> quarantine
violation_layer_id > memory_layer_id -> access
```

Regenerated:

```text
data/galt_teacher_mode/curriculum_contrast_24x2_plus_safety_12x2_stratified
```

Held-out contrast coverage:

```text
validation:
  contrast rows 21
  memory_layer 0:12, 2:6, 3:3
  violation_layer 0:8, 4:7, 1:3, 2:2, 3:1
  access_target 0:14, 1:7

test:
  contrast rows 21
  memory_layer 0:12, 2:6, 3:3
  violation_layer 0:8, 4:7, 2:4, 1:1, 3:1
  access_target 0:14, 1:7
```

## Model Update

Updated:

```text
experiments/galt_from_scratch/teacher_mode_curriculum_mlx.py
```

New heads:

```text
galt_memory_layer_head
galt_violation_layer_head
galt_hierarchical_access_head
```

New loss weights:

```text
galt_memory_layer_loss_weight
galt_violation_layer_loss_weight
galt_hierarchical_access_loss_weight
```

New metrics:

```text
galt_memory_layer_acc
galt_violation_layer_acc
galt_hierarchical_access_acc
galt_predicted_layer_access_acc
```

Defaults keep the new layer losses disabled, so the prior mainline is not
silently changed.

## MLX Smokes

All use:

```text
data/galt_teacher_mode/curriculum_contrast_24x2_plus_safety_12x2_stratified
hidden_dim 96
layers 1
heads 4
batch_size 24
packet-balanced sampling true
seed 42
```

### 240-step layer-head smoke

Output:

```text
results/galt_prework/teacher_mode_curriculum/contrast_24x2_plus_safety_12x2_stratified_layer_heads_smoke.json
```

Layer losses:

```text
memory_layer 0.5
violation_layer 0.5
hierarchical_access 0.5
```

Test:

```text
action_acc                       0.719
memory_acc                       0.688
memory_layer_acc                 0.857
violation_layer_acc              0.476
hierarchical_access_acc          0.571
predicted_layer_access_acc       0.476
```

Interpretation: the memory layer begins to learn, but violation-layer
prediction is too weak for the predicted gate to work.

### 400-step no-layer-loss control

Output:

```text
results/galt_prework/teacher_mode_curriculum/contrast_24x2_plus_safety_12x2_stratified_no_layer_400step_smoke.json
```

Test:

```text
action_acc                 1.000
memory_acc                 1.000
residual_acc               0.813
hard_violation_recall      0.250
memory_layer_acc           0.048
violation_layer_acc        0.143
predicted_layer_access_acc 0.619
```

Interpretation: longer GALT training alone strongly improves action/memory,
but layer-head metrics remain effectively untrained.

### 400-step layer-head smoke

Output:

```text
results/galt_prework/teacher_mode_curriculum/contrast_24x2_plus_safety_12x2_stratified_layer_heads_400step_smoke.json
```

Layer losses:

```text
memory_layer 1.0
violation_layer 1.0
hierarchical_access 1.0
```

Test:

```text
action_acc                       0.875
accept_recall                    0.909
reject_recall                    0.857
memory_acc                       0.844
accepted_trace_recall            0.909
not_consolidated_recall          0.810
residual_acc                     0.781
hard_violation_recall            0.000
memory_layer_acc                 1.000
violation_layer_acc              0.667
hierarchical_access_acc          0.810
predicted_layer_access_acc       0.667
```

## Interpretation

The MLX model now supports trainable layer heads, and the data path works.
However, the current layer-head formulation is not yet good enough:

```text
memory_layer is easy and reaches 1.000
violation_layer is the bottleneck at about 0.667
predicted hierarchical gate is limited by violation_layer prediction
```

The 400-step no-layer control is important:

```text
longer GALT training alone improves action/memory more than the first layer-head
run, so layer-head gains must be judged against equal training budget.
```

The positive result is:

```text
hierarchical layer supervision can be integrated into the curriculum model.
```

The unresolved problem is:

```text
violation-layer prediction needs a cleaner supervision path.
```

## Next Step

Do not claim the MLX predicted hierarchical gate is solved yet.

Next implementation should focus on violation-layer supervision:

```text
1. Train violation layer only on contrast rows, not mixed legacy GALT rows.
2. Add per-class weighting for violation_layer_id.
3. Consider a separate violation encoder/path using candidate_transition +
   constraint/no_write text, rather than the pooled shared row encoding.
4. Compare equal-budget:
   - no layer loss 400
   - layer loss 400
   - violation-only weighted layer loss 400
```

Adoption gate:

```text
violation_layer_acc >= 0.80
predicted_layer_access_acc >= 0.80
action/memory reject recall does not collapse
```
