# GALT v4 Release Notes

## Release Identity

This is the **GALT v4** release snapshot.

Version meaning:

```text
v1: CSAT / AVBD optimizer precursor
v2: Stage D typed routed channels
v3: GALT-vs-BP correction geometry and neural-map thesis
v4: safety-indexed hierarchical trace memory with compact multiseed evidence
```

## Main Claim

The v4 claim is:

```text
GALT is a map-guided, constraint-native local correction framework.
Memory is a downstream trace-bearing structure indexed by safety/admissibility
layer, not a symmetric retain branch.
```

The memory rule tested in v4 is:

```text
lower/same-layer violation -> quarantine dependent memory
higher-layer violation     -> local containment; lower memory remains accessible
no active violation        -> access
```

## New Evidence Chain

v4 includes a three-stage hierarchical-memory evidence chain:

1. **Symbolic hierarchical gate**
   - independent same-layer gates miss bottom-up invalidation;
   - global flush gates destroy top-down containment;
   - hierarchical gates preserve both.
2. **Natural-language contrast gate**
   - generated contrast packets preserve the same asymmetry;
   - flat text-only learners partially learn the rule but remain below a
     structural ledger/gate guarantee.
3. **Trainable MLX layer-head gate**
   - memory-layer and violation-layer heads drive predicted hierarchical access;
   - the 600-step violation-weighted recipe passes seeds 42/43/44.

## Key Multiseed Result

Summary artifact:

```text
results/galt_prework/teacher_mode_curriculum/hierarchical_gate_multiseed_summary.json
```

Final-test means for the 600-step violation-weighted recipe:

| Metric | Mean |
| --- | ---: |
| action acc | 1.000 |
| gate acc | 1.000 |
| memory acc | 1.000 |
| accepted trace recall | 1.000 |
| not consolidated recall | 1.000 |
| hard-violation recall | 1.000 |
| predicted hierarchical access acc | 1.000 |
| residual acc | 0.984 |
| violation-layer acc | 0.976 |

## Included Artifacts

Core manuscript:

```text
paper/galt.pdf
```

Reports:

```text
paper/galt_prework/78_hierarchical_safety_indexed_memory_report.md
paper/galt_prework/79_nl_hierarchical_memory_gate_report.md
paper/galt_prework/80_curriculum_layer_heads_smoke_report.md
paper/galt_prework/81_violation_weighted_hierarchical_gate_report.md
paper/galt_prework/82_hierarchical_gate_multiseed_release_report.md
```

Core scripts:

```text
experiments/galt_from_scratch/hierarchical_memory_layer_benchmark.py
experiments/galt_from_scratch/nl_hierarchical_memory_layer_benchmark.py
experiments/galt_from_scratch/teacher_mode_curriculum_mlx.py
scripts/build_teacher_mode_curriculum_preflight.py
scripts/generate_contrast_lesson_packets.py
```

## Caveat

v4 is a compact mechanism release.  It supports the safety-indexed
hierarchical-memory derivative of GALT at small scale, with multiseed evidence
on a generated curriculum.  It does not claim that large language-model memory
governance is solved.
