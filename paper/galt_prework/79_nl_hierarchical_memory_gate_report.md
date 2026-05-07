# Natural-Language Hierarchical Memory Gate Report

This pass connects the synthetic hierarchical-memory benchmark to the generated
natural-language contrast lesson packets.

## Goal

Validate the revised claim:

```text
Memory should be safety-indexed and hierarchical.
Lower-layer violations quarantine dependent upper memories.
Higher-layer violations should remain locally contained and preserve lower
memory/capability layers.
```

The key question is whether this remains visible on generated natural-language
contrast packets, not only on symbolic layer ids.

## Implementation

Added:

```text
experiments/galt_from_scratch/nl_hierarchical_memory_layer_benchmark.py
```

Output:

```text
results/galt_prework/nl_hierarchical_memory_layer_benchmark.json
```

Sources:

```text
data/galt_contrast_lessons/accepted/glm_contrast_course_24x2_20260507_1643_accepted.jsonl
data/galt_contrast_lessons/accepted/glm_safety_boundary_course_12x2_20260507_1811_accepted.jsonl
```

Packet count:

```text
total           68
safety_boundary 36
workflow        12
preference      12
tutoring         8
```

The benchmark creates memory-access rows by pairing each accepted positive
memory candidate with each active violation context.

Layer mapping:

```text
memory layer:
  safety_boundary -> L0 hard safety/admissibility
  tutoring/workflow -> L2 task rule
  preference -> L3 preference

violation layer:
  safety_boundary -> L0 hard safety/admissibility
  tutoring/preference -> L1 social/consent
  workflow -> L2 task rule
```

Target rule:

```text
violation_layer <= memory_layer -> quarantine
violation_layer > memory_layer  -> access
none violation                  -> access
```

The flat learners see only natural-language text:

```text
memory candidate text + current boundary signal text
```

They do not receive explicit layer ids.

## Policies

Compared:

```text
flat_bow_logistic
flat_bow_balanced
independent_layer_gate
global_flush_gate
hierarchical_gate
```

Interpretation of structural baselines:

```text
independent_layer_gate:
  only same-layer violation invalidates memory

global_flush_gate:
  any violation invalidates every memory

hierarchical_gate:
  lower/same layer invalidates, higher layer is contained
```

## Results

| Policy | accuracy | access recall | quarantine recall | bottom-up invalidation | top-down containment | heldout pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flat BoW logistic | 0.752 | 0.046 | 1.000 | 1.000 | 0.049 | 0.763 |
| flat BoW balanced | 0.842 | 0.670 | 0.902 | 0.902 | 0.674 | 0.681 |
| independent layer gate | 0.587 | 1.000 | 0.442 | 0.442 | 1.000 | 0.237 |
| global flush gate | 0.754 | 0.056 | 1.000 | 1.000 | 0.000 | 0.763 |
| hierarchical gate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Predicted Layer Gate

Follow-up implementation:

```text
memory_layer_head
violation_layer_head
predicted_hierarchical_gate
```

The layer heads are trained from natural-language packet text and then their
predicted layer ids drive the same hierarchical ledger rule.

To avoid only testing memorized packet texts, a packet-level holdout is used:

```text
heldout packets: 17
preference:       3
safety_boundary:  9
tutoring:         2
workflow:         3
```

Layer-head metrics:

```text
memory_layer_accuracy          0.971
violation_layer_accuracy       0.971
heldout_memory_layer_accuracy  0.882
heldout_violation_layer_acc    0.882
```

Despite layer-head errors, the predicted hierarchical gate preserves the access
decision in this benchmark:

| Policy | accuracy | bottom-up invalidation | top-down containment | heldout pair | heldout packet |
| --- | ---: | ---: | ---: | ---: | ---: |
| predicted hierarchical gate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| oracle hierarchical gate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

This is the first evidence that the hierarchical memory rule can be driven by
trainable natural-language layer heads, not only by hand-coded oracle labels.

## Interpretation

The natural-language result matches the symbolic result qualitatively:

```text
independent layer gate is too weak:
  it preserves top-down containment but misses bottom-up invalidation.

global flush gate is too strong:
  it catches bottom-up invalidation but destroys lower memory under higher
  violations.

hierarchical gate matches the desired asymmetry.
```

The flat BoW learner partially learns from text when class-balanced:

```text
accuracy 0.842
bottom-up 0.902
top-down 0.674
heldout 0.681
```

This is useful evidence but also a warning.  Natural-language text alone can
learn part of the rule, but it does not provide the governance guarantee:

```text
bottom-up invalidation and top-down containment should be enforced by the
memory ledger/gate, not left only to a flat classifier.
```

## Paper Implication

The paper can now state a stronger memory thesis:

```text
Memory is not a single append-only store.
It is a safety-indexed hierarchy of trace-bearing structures.
Lower safety layers define admissibility for higher memories, and violations
create residual/quarantine debt for dependent upper traces.
```

This ties together:

```text
safety-defined feasible region
adjudicated trace
memory consolidation
residual debt
teacher re-adjudication
```

## Next Step

Turn this from an external deterministic gate into a trainable kernel component:

```text
1. Add explicit safety_layer_id and memory_layer_id fields into curriculum rows.
2. Connect layer prediction heads to the MLX curriculum model.
3. Use predicted layers to drive a hierarchical memory ledger gate.
4. Compare:
   - oracle hierarchical gate
   - predicted hierarchical gate
   - flat action/memory/residual heads
5. Measure:
   - bottom_up_invalidation_accuracy
   - top_down_containment_accuracy
   - quarantine_precision
   - access_recall
   - re_adjudication_recovery
```

The success condition is not merely high aggregate accuracy.  The gate must
preserve both:

```text
lower-layer invalidates upper memory
higher-layer does not destroy lower memory
```
