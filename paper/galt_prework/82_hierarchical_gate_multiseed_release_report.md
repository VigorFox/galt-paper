# Hierarchical Gate Multiseed and Release Report

This pass strengthens the evidence chain after the first 600-step
violation-weighted layer-head gate.

## Goal

The previous strongest result was a single seed:

```text
contrast_24x2_plus_safety_12x2_stratified_layer_heads_600step_violation_weighted_smoke.json
```

The question was whether that result was only a seed-42 accident or a stable
small-curriculum mechanism.

## New Runs

Primary recipe:

```text
data_dir: data/galt_teacher_mode/curriculum_contrast_24x2_plus_safety_12x2_stratified
steps: 600
schedule: staged 80 / 80 / 440
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
```

New outputs:

```text
results/galt_prework/teacher_mode_curriculum/contrast_24x2_plus_safety_12x2_stratified_layer_heads_600step_violation_weighted_seed43.json
results/galt_prework/teacher_mode_curriculum/contrast_24x2_plus_safety_12x2_stratified_layer_heads_600step_violation_weighted_seed44.json
results/galt_prework/teacher_mode_curriculum/contrast_24x2_plus_safety_12x2_stratified_layer_heads_600step_no_violation_weight_seed42.json
results/galt_prework/teacher_mode_curriculum/hierarchical_gate_multiseed_summary.json
```

## Multiseed Result

Final-test metrics for the violation-weighted 600-step recipe:

| Metric | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| action acc | 1.000 | 1.000 | 1.000 |
| accept recall | 1.000 | 1.000 | 1.000 |
| reject recall | 1.000 | 1.000 | 1.000 |
| gate acc | 1.000 | 1.000 | 1.000 |
| memory acc | 1.000 | 1.000 | 1.000 |
| accepted trace recall | 1.000 | 1.000 | 1.000 |
| not consolidated recall | 1.000 | 1.000 | 1.000 |
| residual acc | 0.984 | 0.969 | 1.000 |
| boundary ambiguity recall | 0.950 | 0.900 | 1.000 |
| hard violation recall | 1.000 | 1.000 | 1.000 |
| memory layer acc | 1.000 | 1.000 | 1.000 |
| violation layer acc | 0.976 | 0.952 | 1.000 |
| direct hierarchical access acc | 1.000 | 1.000 | 1.000 |
| predicted hierarchical access acc | 1.000 | 1.000 | 1.000 |

This passes the adoption gate across seeds 42/43/44:

```text
violation_layer_acc >= 0.80
predicted_layer_access_acc >= 0.80
accept/reject and memory/not-consolidated do not collapse
hard_violation_recall remains nonzero and in fact reaches 1.000
```

## Ablation Context

Seed-42 controls:

| Run | action | memory | residual | hard | memory layer | violation layer | predicted access |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 400-step no layer loss | 1.000 | 1.000 | 0.812 | 0.250 | 0.048 | 0.143 | 0.619 |
| 400-step layer heads | 0.875 | 0.844 | 0.781 | 0.000 | 1.000 | 0.667 | 0.667 |
| 600-step no violation weights | 1.000 | 1.000 | 0.969 | 1.000 | 1.000 | 0.952 | 1.000 |
| 600-step violation weights | 1.000 | 1.000 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation:

```text
longer GALT-stage training is essential;
violation-layer weighting improves margin/stability;
the hierarchical access decision becomes reliable only after the model learns
the violation-layer interface, not merely action/memory labels.
```

## Updated Evidence Claim

The strong evidence chain is now:

```text
symbolic hierarchical gate:
  structural asymmetry is required

natural-language contrast gate:
  the same asymmetry survives generated text packets

MLX trainable layer-head gate:
  predicted hierarchical access passes the adoption gate across seeds 42/43/44
```

This is enough to support the paper statement:

```text
small-scale evidence supports safety-indexed hierarchical memory as the current
trace/residual derivative of GALT.
```

It is not enough to claim:

```text
large-scale language-model memory governance is solved.
```

The remaining robustness work is larger generated curricula, broader
layer-combination coverage, and full language-bearing student integration.

## Release Implication

The release snapshot should include:

```text
paper/galt.pdf
paper/galt_prework/78_hierarchical_safety_indexed_memory_report.md
paper/galt_prework/79_nl_hierarchical_memory_gate_report.md
paper/galt_prework/80_curriculum_layer_heads_smoke_report.md
paper/galt_prework/81_violation_weighted_hierarchical_gate_report.md
paper/galt_prework/82_hierarchical_gate_multiseed_release_report.md
experiments/galt_from_scratch/hierarchical_memory_layer_benchmark.py
experiments/galt_from_scratch/nl_hierarchical_memory_layer_benchmark.py
experiments/galt_from_scratch/teacher_mode_curriculum_mlx.py
scripts/build_teacher_mode_curriculum_preflight.py
scripts/generate_contrast_lesson_packets.py
results/galt_prework/teacher_mode_curriculum/hierarchical_gate_multiseed_summary.json
```
