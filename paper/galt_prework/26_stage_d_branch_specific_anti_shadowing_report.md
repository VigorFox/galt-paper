# Stage D branch-specific anti-shadowing report

## Goal

The earlier typed separation pilot showed that moderate anti-shadowing could
improve branch separation without sacrificing the positive typed regime.

But one question remained unresolved:

> was the gain coming from suppressing task-branch shadowing on both safety and
> memory together, or was one branch doing all the work?

This round decomposed anti-shadowing into branch-specific objectives.

---

## Artifact

- `experiments/galt/stage_d_native_policy_smoke_mlx.py`

New controls:

- `--safety-shadow-suppression-weight`
- `--safety-shadow-suppression-margin`
- `--memory-shadow-suppression-weight`
- `--memory-shadow-suppression-margin`

These make it possible to suppress:

- task-overreach on safety replay only
- task-overreach on retain replay only

instead of tying both into one shared anti-shadow term.

---

## Baseline reference

Artifact:

- `typed_output_branches_lr2e4_w05.json`

Baseline typed regime:

- post task/safety/retain = `0.875 / 0.875 / 0.9`
- branch margins:
  - task = `+0.21875`
  - safety = `-0.125`
  - retain = `-0.1`

The shadowing pattern was clear:

- safety prompts: task branch = `1.0`, safety branch = `0.875`
- retain prompts: task branch = `1.0`, memory branch = `0.9`

---

## 1. Safety-only anti-shadowing

Artifact:

- `typed_output_branches_safetyshadow05_m01.json`

Config highlight:

- `safety_shadow_suppression_weight = 0.5`
- `safety_shadow_suppression_margin = 0.1`

Result:

- post task/safety/retain = `0.875 / 0.875 / 0.9`
- all main necessity gaps unchanged

Branch matrix:

- safety prompts:
  - task branch: `1.0 -> 0.65625`
  - safety branch: stays `0.875`
- retain prompts:
  - task branch: `1.0 -> 0.9`
  - memory branch: stays `0.9`

Branch margins:

- task = `+0.21875`
- safety = `-0.125 -> +0.21875`
- retain = `-0.1 -> 0.0`

Most importantly:

> this run is exactly identical to the earlier shared moderate anti-shadowing
> result (`typed_output_branches_shadow05_m01.json`)

Interpretation:

- the earlier moderate gain came entirely from the **safety-side**
  anti-shadowing term
- suppressing task overreach on safety prompts is enough to improve both:
  - safety branch standing
  - retain branch standing up to parity

This is a strong causal attribution result.

---

## 2. Memory-only anti-shadowing

Artifacts:

- `typed_output_branches_memoryshadow05_m01.json`
- `typed_output_branches_memoryshadow10_m02.json`

Config highlights:

- moderate:
  - `memory_shadow_suppression_weight = 0.5`
  - `memory_shadow_suppression_margin = 0.1`
- stronger:
  - `memory_shadow_suppression_weight = 1.0`
  - `memory_shadow_suppression_margin = 0.2`

Result:

Both runs were exactly unchanged relative to baseline:

- post task/safety/retain = `0.875 / 0.875 / 0.9`
- same necessity gaps
- same branch matrix
- same branch margins

Interpretation:

- in the current regime, **memory-only anti-shadowing does not move the system**
- this suggests the retain path is not limited by the same simple task-shadow
  mechanism that governs safety
- memory separation is therefore likely a harder problem than safety separation

In plain language:

> safety branch disentanglement responds to targeted pressure;
> memory branch disentanglement currently does not.

---

## Main conclusion

This round identifies where the current typed-separation progress is actually
coming from.

1. the earlier moderate anti-shadowing gain is fully explained by
   **safety-only anti-shadowing**
2. memory-only anti-shadowing does nothing in the current setup

So the next picture is sharper:

- **safety branch separation is now controllable**
- **memory branch separation remains the harder unresolved frontier**

---

## Strategic consequence

The next typed-separation work should stop treating safety and memory as one
combined problem.

Instead:

1. keep the moderate safety anti-shadowing term as a stable positive ingredient
2. treat memory separation as its own subproblem
3. investigate more structural memory-specific interventions, especially those
   that may affect `retain_scramble_gap`, which remains `0.0`

This is progress because it reduces one big vague separation problem into two
very different subproblems:

- a solvable safety-shadow problem
- a harder memory-carrier problem
