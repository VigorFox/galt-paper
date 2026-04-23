# Stage D typed output branches report

## Goal

After expert-only routed outputs worked on the real carrier, the next question was:

> can we move from generic routed output experts to **typed task / safety / memory branches**
> while preserving route necessity?

This round tested whether the Stage D carrier can support branch-specialized routed outputs
 rather than a single shared output-expert bank.

---

## Artifact

- `experiments/galt/stage_d_native_policy_smoke_mlx.py`

New architecture additions:

- optional `--typed-output-branches`
- separate routed output heads for:
  - `task`
  - `safety`
  - `memory`
- branch-aware local scoring / replay-anchor / KL-constraint / route-mode evaluation
- optional direct branch supervision:
  - `--safety-branch-weight`
  - `--memory-branch-weight`

---

## Probe 1: typed branches without direct branch supervision

Artifact:
- `results/galt_prework/stage_d_native_policy_smoke/typed_output_branches_smoke12.json`

Config highlight:

- `typed_output_branches=true`
- `output_expert_scale=2.0`
- `base_choice_scale=0.0`
- no direct safety / memory branch CE

Result:

- post task/safety/retain = `0.25 / 0.25 / 0.0`
- all route necessity gaps = `0.0`

Interpretation:

- the architecture path itself is valid
- but typed branches fail when safety / memory heads receive only anchor-preservation constraints
- unlike the shared-output regime, non-task heads do not get enough positive learning signal from task training alone

This is an important negative result:

> typed branches cannot be treated as "just split the heads and let constraints do the rest."
> once branches are isolated, they need their own native learning signal.

---

## Probe 2: add direct branch supervision

Artifact:
- `results/galt_prework/stage_d_native_policy_smoke/typed_output_branches_smoke12_supervised.json`

Config highlight:

- `typed_output_branches=true`
- `safety_branch_weight=1.0`
- `memory_branch_weight=1.0`

Result:

- post task/safety/retain = `0.25 / 0.50 / 0.875`

Branch specialization snapshot:

- safety branch margin vs best wrong branch = `+0.125`
- memory branch margin vs best wrong branch = `+0.375`

Interpretation:

- as soon as the safety / memory branches receive direct CE supervision, they become functional
- this confirms that the failure in Probe 1 was not caused by the typed architecture itself
- the issue was missing positive branch-specific training signal

---

## Probe 3: full typed-branch run in the positive Stage D regime

Artifact:
- `results/galt_prework/stage_d_native_policy_smoke/typed_output_branches_lr2e4_w05.json`

Config highlight:

- `lr=2e-4`
- `smoke_steps=40`
- `policy_only_warmup_steps=40`
- `typed_output_branches=true`
- `safety_branch_weight=0.5`
- `memory_branch_weight=0.5`
- `output_expert_scale=2.0`
- `base_choice_scale=0.0`
- hard routing

Result:

- pre task/safety/retain = `0.3125 / 0.28125 / 0.0`
- post task/safety/retain = `0.875 / 0.875 / 0.9`
- route accuracy = `0.875`
- route entropy = `0.661`
- task gaps:
  - `task_zero_gap = 0.5625`
  - `task_scramble_gap = 0.53125`
- safety gaps:
  - `safety_zero_gap = 0.59375`
  - `safety_scramble_gap = 0.625`
- retain gaps:
  - `retain_zero_gap = 0.9`
  - `retain_scramble_gap = 0.0`
- `overall_pass = True`

Branch matrix highlights:

- task dataset:
  - task branch = `0.875`
  - best wrong branch = `0.65625`
  - margin = `+0.21875`
- safety dataset:
  - safety branch = `0.875`
  - task branch = `1.0`
- retain dataset:
  - memory branch = `0.9`
  - task branch = `1.0`

Interpretation:

- typed routed outputs **do** preserve the main Stage D breakthrough on the real carrier
- task branch specialization is now visible
- safety and memory branches are useful and high-performing
- but branch separation is still incomplete because the task branch remains very strong on safety and retain prompts
- retain is highly sensitive to route zeroing, but not yet to route scrambling

In plain language:

> Stage D can now carry not just one generic routed output path,
> but a typed family of routed paths.
> However, typed branching has not yet become fully disentangled.

---

## Main conclusion

This is a positive Stage D extension, not just a cosmetic variant.

We now have evidence for the following chain:

1. generic routed output experts can become necessary on the real carrier
2. typed routed branches also work
3. typed branches require native positive supervision, not only anchor constraints

The main new lesson is:

> once GALT moves from "shared routed carrier" to "typed routed carrier",
> each branch must be treated as a real capability path with its own training signal.

---

## Strategic consequence

This justifies the next mainline:

1. robustness sweeps around the positive typed regime
2. stronger branch-separation objectives so safety / memory no longer ride on the task branch
3. architecture-level removal of any remaining shared fallback behavior

Typed routing is now a real path forward, not a speculative extension.
