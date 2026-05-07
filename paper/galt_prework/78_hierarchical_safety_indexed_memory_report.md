# Hierarchical Safety-Indexed Memory Report

This pass explores the user's revised hypothesis:

```text
Memory should be layered by safety/admissibility dependency.
Breaking a lower-level rule invalidates memories at that layer and above.
Breaking a higher-level rule should not destroy lower-level memory.
```

This is stronger than a simple parallel separation between residual/safety
pressure and memory consolidation pressure.

## Conceptual Update

The right structure is not:

```text
safety head || residual head || memory head
```

as independent peers.

The better structure is:

```text
L0 hard admissibility / basic safety
  -> L1 social / consent boundary
    -> L2 task or workflow rule
      -> L3 preference / durable memory
```

Memory entries should be indexed by the safety layer they depend on:

```text
memory_entry = {
  layer_id,
  dependency_rules,
  accepted_trace,
  adjudication_state,
  residual_state
}
```

Access rule:

```text
a violation at layer v invalidates memory at layer m when v <= m
a higher-layer violation v > m should not invalidate lower-layer memory
```

Operationally:

```text
lower-layer break -> upper memories quarantine / re-adjudication
higher-layer break -> local containment only
```

## Benchmark

Added:

```text
experiments/galt_from_scratch/hierarchical_memory_layer_benchmark.py
```

Output:

```text
results/galt_prework/hierarchical_memory_layer_benchmark.json
```

The benchmark compares four policies:

```text
independent_layer_gate
  invalidates only the same layer

global_flush_gate
  invalidates every memory after any violation

hierarchical_gate
  invalidates same and higher memory layers only

flat_mlp
  learned classifier over explicit layer symbols
```

Metrics:

```text
bottom_up_invalidation_accuracy
top_down_containment_accuracy
heldout_bottom_up_accuracy
heldout_top_down_accuracy
```

## Results

Aggregate over seeds 42/43/44/45/46:

| Policy | accuracy | bottom-up invalidation | top-down containment | heldout bottom-up | heldout top-down |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent layer gate | 0.700 | 0.400 | 1.000 | 0.000 | 1.000 |
| global flush gate | 0.700 | 1.000 | 0.000 | 1.000 | 0.000 |
| hierarchical gate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| flat MLP | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

The two naive structural policies fail in opposite ways:

```text
independent layer gate:
  protects lower memories from higher violations,
  but misses lower-to-upper invalidation.

global flush gate:
  catches lower-to-upper invalidation,
  but destroys lower memories when only higher rules break.
```

The hierarchical gate has the intended asymmetry:

```text
bottom-up invalidation + top-down containment
```

The flat MLP result is an important caveat.  With explicit symbolic layer
features and a clean synthetic target, a generic learner can fit the rule.
This does not remove the need for hierarchical memory structure.  It means the
rule is learnable, while the architecture question is whether memory access and
quarantine are guaranteed by the ledger/gate rather than left as an ordinary
classification habit.

## Paper Implication

The paper should say:

```text
Memory grows on top of safety-indexed admissibility layers.
Lower-layer violations invalidate or quarantine dependent upper memories.
Higher-layer violations are contained locally and should not destroy lower
capabilities or lower-layer memories.
Residual is the ledger of unresolved cross-layer debt.
```

This gives the memory story a stronger causal structure:

```text
safety layer -> admissible region -> adjudicated trace -> memory access
                                  -> residual/quarantine if unresolved
```

## Next Experiment

Connect this deterministic layer-gate benchmark back to the natural-language
contrast curriculum:

```text
1. Add explicit safety_layer_id / memory_layer_id targets to contrast rows.
2. Add dependency-layer metadata to accepted_trace rows.
3. Train/evaluate:
   - flat action/memory/residual heads
   - independent layer gate
   - global flush gate
   - hierarchical safety-indexed memory gate
4. Measure:
   - bottom_up_invalidation_accuracy
   - top_down_containment_accuracy
   - quarantine_precision
   - re_adjudication_recovery
   - lower_layer_retention_after_high_layer_violation
```

The immediate architecture target is therefore:

```text
safety-indexed hierarchical memory ledger
```

not merely:

```text
parallel residual/safety/memory head isolation
```
