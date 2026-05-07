# Violation-Weighted Hierarchical Gate Report

This pass closes the immediate evidence gap left by the first curriculum
layer-head smoke.

Follow-up note: report 82 extends this single-seed gate to a three-seed
violation-weighted check over seeds 42/43/44.

## Starting Point

Report 80 showed that the MLX curriculum model could train memory-layer,
violation-layer, and hierarchical-access heads, but the predicted gate was
still limited by violation-layer prediction:

| Run | action acc | memory acc | violation layer acc | predicted access acc | hard recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 400-step no layer loss | 1.000 | 1.000 | 0.143 | 0.619 | 0.250 |
| 400-step layer heads | 0.875 | 0.844 | 0.667 | 0.667 | 0.000 |
| 400-step layer heads + violation weights | 0.844 | 0.844 | 0.762 | 0.762 | 0.000 |

The bottleneck was therefore not the memory-layer label, but the boundary
classification needed to decide whether a memory should remain accessible or
be quarantined.

## Final Smoke

Run output:

```text
results/galt_prework/teacher_mode_curriculum/contrast_24x2_plus_safety_12x2_stratified_layer_heads_600step_violation_weighted_smoke.json
```

Configuration:

```text
data_dir: data/galt_teacher_mode/curriculum_contrast_24x2_plus_safety_12x2_stratified
steps: 600
schedule: staged
stage_a_steps: 80
teacher_steps: 80
galt_steps: 440
batch_size: 24
hidden_dim: 96
layers: 1
heads: 4
packet_balanced_sampling: true
enable_galt_violation_layer_class_weights: true
galt_class_weight_clip: 4.0
memory_layer_loss_weight: 1.0
violation_layer_loss_weight: 1.0
hierarchical_access_loss_weight: 1.0
seed: 42
```

Final test metrics:

| Metric | Value |
| --- | ---: |
| action acc | 1.000 |
| action accept recall | 1.000 |
| action reject recall | 1.000 |
| gate acc | 1.000 |
| memory acc | 1.000 |
| accepted trace recall | 1.000 |
| not consolidated recall | 1.000 |
| residual acc | 0.969 |
| residual boundary ambiguity recall | 0.900 |
| residual hard violation recall | 1.000 |
| residual none recall | 1.000 |
| memory layer acc | 1.000 |
| violation layer acc | 1.000 |
| direct hierarchical access acc | 1.000 |
| predicted layer access acc | 1.000 |

## Interpretation

This run passes the adoption gate defined in report 80:

```text
violation_layer_acc >= 0.80
predicted_layer_access_acc >= 0.80
action/memory reject recall does not collapse
```

It also fixes the earlier residual failure mode:

```text
hard_violation_recall: 0.000 -> 1.000
```

without sacrificing accepted-trace consolidation or no-write rejection.

The key result is not merely higher aggregate accuracy.  The model now learns
the missing interface required for safety-indexed hierarchical memory:

```text
predict memory layer
predict active violation layer
apply hierarchical access rule
preserve access for higher-layer violations
quarantine dependent upper memory for lower/same-layer violations
```

## Current Evidence Chain

The hierarchical-memory claim now has a three-step chain:

1. Symbolic benchmark:
   - independent same-layer gate misses bottom-up invalidation;
   - global flush gate destroys top-down containment;
   - hierarchical gate satisfies both.
2. Natural-language benchmark:
   - generated contrast packets preserve the same asymmetry;
   - flat text-only learners partially learn the rule but remain below the
     structural hierarchical gate;
   - predicted natural-language layer heads can drive the gate on the BoW
     diagnostic.
3. MLX curriculum smoke:
   - the actual teacher-mode curriculum model learns memory/violation layers;
   - predicted layer access reaches `1.000` on the held-out test split in the
     600-step violation-weighted smoke;
   - accept/reject and memory/not-consolidated heads remain intact.

## Caveats

This should be reported as a small-scale proof of mechanism, not as a mature
scaled result:

```text
seed: one seed only for the final 600-step gate
data: 68 generated contrast packets expanded into a small curriculum
heldout support: useful but still small
model: compact MLX smoke, not a full language model
```

The next robustness step is multiseed coverage over the 600-step recipe and a
larger safety-boundary / preference / workflow lesson set.  But the evidence is
now strong enough to update the paper from "stratified adjudicated memory is
the next experimental target" to "small-scale evidence supports a
safety-indexed hierarchical memory ledger."

## Paper Implication

The paper should now present memory as:

```text
trace-bearing structure indexed by safety/admissibility layer
```

not as:

```text
a symmetric memory branch or retain loss
```

The revised mechanism is:

```text
neural map chooses where correction can be expressed
safety stack defines admissible region
layer heads identify dependency and violation level
adjudication gate accepts, rejects, or quarantines the trace
residual ledger stores unresolved cross-layer debt
```

This keeps memory downstream of the core GALT thesis while making the memory
derivative much more structurally precise.
