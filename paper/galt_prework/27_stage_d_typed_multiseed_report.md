# Stage D typed multiseed report

## Goal

The single-seed typed Stage D results were already strong, but for the paper the
next question was obvious:

> are the typed real-carrier results just a lucky seed,
> or do they survive a small multiseed check?

This round ran a minimal paper-oriented multiseed sweep over the two most
important typed configurations:

1. typed baseline
2. typed baseline + moderate safety-only anti-shadowing

---

## Artifact

- `experiments/galt/stage_d_native_policy_smoke_mlx.py`

Result files:

- `typed_multiseed_baseline_seed41.json`
- `typed_multiseed_baseline_seed42.json`
- `typed_multiseed_baseline_seed43.json`
- `typed_multiseed_safetyshadow_seed41.json`
- `typed_multiseed_safetyshadow_seed42.json`
- `typed_multiseed_safetyshadow_seed43.json`

Shared config:

- `lr=2e-4`
- `typed_output_branches=true`
- `safety_branch_weight=0.5`
- `memory_branch_weight=0.5`
- `output_expert_scale=2.0`
- `base_choice_scale=0.0`
- hard routing
- `smoke_steps=40`
- `policy_only_warmup_steps=40`

Safety-shadow variant:

- `safety_shadow_suppression_weight=0.5`
- `safety_shadow_suppression_margin=0.1`

---

## 1. Typed baseline across 3 seeds

Per-seed post metrics:

- seed 41: `0.875 / 0.750 / 0.600`
- seed 42: `0.875 / 0.875 / 0.900`
- seed 43: `0.6875 / 0.875 / 0.600`

Aggregate:

- post task = `0.8125 ± 0.088`
- post safety = `0.8333 ± 0.059`
- post retain = `0.7000 ± 0.141`
- route acc = `0.7396 ± 0.131`

Necessity aggregates:

- task zero gap = `0.4792 ± 0.097`
- task scramble gap = `0.4583 ± 0.082`
- safety zero gap = `0.3958 ± 0.280`
- safety scramble gap = `0.4063 ± 0.288`
- retain zero gap = `0.2667 ± 0.450`
- retain scramble gap = `-0.0167 ± 0.024`

Interpretation:

- the typed real-carrier regime is **not a single-seed mirage**
- all three seeds still passed the current success gate
- but the regime is not yet tightly concentrated:
  - task and retain utility vary materially
  - retain necessity remains especially unstable

So for the paper, the right claim is:

> typed routed channels are real and repeatable,
> but still seed-sensitive rather than fully mature

---

## 2. Typed baseline + safety-only anti-shadowing across 3 seeds

Per-seed post metrics:

- seed 41: `0.875 / 0.750 / 0.600`
- seed 42: `0.875 / 0.875 / 0.900`
- seed 43: `0.6875 / 0.84375 / 0.600`

Aggregate:

- post task = `0.8125 ± 0.088`
- post safety = `0.8229 ± 0.053`
- post retain = `0.7000 ± 0.141`
- route acc = `0.7396 ± 0.131`

Necessity aggregates:

- task zero gap = `0.4792 ± 0.097`
- task scramble gap = `0.4583 ± 0.082`
- safety zero gap = `0.3854 ± 0.273`
- safety scramble gap = `0.3854 ± 0.275`
- retain zero gap = `0.2667 ± 0.450`
- retain scramble gap = `-0.0167 ± 0.024`

Interpretation:

- moderate safety anti-shadowing does **not** materially raise the multiseed
  headline averages
- its value is therefore not "better average benchmark score"
- its value is mainly in the **branch-separation story**

This is confirmed by the branch margins:

- task margin:
  - baseline mean = `+0.1771`
  - safety-shadow mean = `+0.1875`
- safety margin:
  - baseline mean = `-0.0521`
  - safety-shadow mean = `+0.0729`
- retain margin:
  - baseline mean = `-0.2000`
  - safety-shadow mean = `-0.1667`

The most important change is:

> the safety branch is closer to, and sometimes above, task-branch dominance
> under the anti-shadowed regime

So the safety anti-shadow term is paper-relevant not because it clearly boosts
utility, but because it sharpens the **responsibility-separated-channel** claim.

---

## Main conclusion

This multiseed round strengthens the paper in two ways:

1. the typed real-carrier regime is repeatable enough to count as real evidence,
   not just a one-seed curiosity
2. safety anti-shadowing is best understood as a **separation mechanism**, not a
   utility-boost mechanism

In plain language:

> the typed GALT story survives beyond one lucky run,
> but the branches are still not fully stable or fully disentangled

That is a strong and honest paper position.

---

## Strategic consequence

For the GALT paper, the best use of this result is:

1. report multiseed evidence for the typed regime
2. present safety anti-shadowing as branch-boundary sharpening
3. keep explicit caveats that retain/memory remains the weaker and more variable path

This supports a sharper narrative:

- the paradigm is real
- the current implementation is not yet final
- the next work is about tightening robustness and completing separation
