"""AVBD-GALT Optimizer (v13): PI + ADMM Proximal Anchor.

First instance of GALT single-block form (galt_theory.md §12.2). Extends the
v12 PI structure with the proximal anchor mechanism transcribed from PhysX
AVBD (`avbd_solver.cpp:1035-1036, 1132-1135`):

    outer (every K_in steps):
        θ_anchor ← θ_current              # snapshot
        λ       ← max(0, λ + ρ_I · ⟨C_post⟩)  # dual update
        reset integrator EMA

    inner (every step, j = 1..K_in):
        base_δ  = -m̂/D + ρ_a · (θ_anchor − θ)   # Adam + anchor pull
        δ       = SM_project(base_δ, J, ρ_P, λ, C_raw)
        θ      ← θ + lr · δ

The anchor pull is the proximal term that makes nonconvex ADMM converge; it
contracts θ toward the outer snapshot so inner iterations cannot drift
between dual updates. This directly addresses the v12 bipolar variance mode
where raw-C noise drove ρ_P saturation into a positive-feedback loop.

Beyond v12 (PI only), v13 adds:
  - θ_anchor snapshot & proximal pull (§12.2 Outer/Inner Anchor)
  - ρ_P lower floor  (§12.6 PENALTY_MIN)
  - λ updates synchronized with outer step (PhysX semantics, not per-step)
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_utils import clone_flat_dict, flatten_tree, scalar, unflatten_tree


class AVBDGALTOptimizer:
    """v13: Adam + SM projection + PI dual + ADMM proximal anchor (single-block GALT)."""

    def __init__(
        self,
        model,
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        # ----- Proportional gain (primal projection) -----
        rho_p_init: float = 1.0,
        rho_p_min: float = 1.0,
        rho_p_max: float = 50.0,
        rho_p_growth: float = 1.5,
        rho_p_decay: float = 1.0,
        rho_p_tau: float = 0.9,
        # ----- Integral gain (dual update) -----
        rho_i: float = 0.5,
        integral_smoothing: float = 0.9,
        use_post_residual_for_integral: bool = True,
        # ----- ADMM proximal anchor (§12.2) -----
        outer_step_freq: int = 5,
        rho_anchor: float = 0.1,
        rho_anchor_decay: float = 1.0,
        # ----- λ persistence (§12.3 style, cross-task) -----
        lambda_warmstart: float = 0.0,
        lambda_carry_factor: float = 1.0,
        lambda_max: float = float("inf"),
        # ----- Misc -----
        use_multi_constraint_woodbury: bool = True,
    ):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.rho_p_init = rho_p_init
        self.rho_p_min = rho_p_min
        self.rho_p_max = rho_p_max
        self.rho_p_growth = rho_p_growth
        self.rho_p_decay = rho_p_decay
        self.rho_p_tau = rho_p_tau

        self.rho_i = rho_i
        self.integral_smoothing = integral_smoothing
        self.use_post_residual_for_integral = use_post_residual_for_integral

        self.outer_step_freq = max(1, outer_step_freq)
        self.rho_anchor_init = rho_anchor
        self.rho_anchor_decay = rho_anchor_decay
        self.rho_anchor = rho_anchor

        self.lambda_warmstart = lambda_warmstart
        self.lambda_carry_factor = lambda_carry_factor
        self.lambda_max = lambda_max

        self.use_multi_constraint_woodbury = use_multi_constraint_woodbury

        self.constraints: list[dict] = []
        self._constraint_data: list[Optional[tuple[float, dict[str, mx.array], bool]]] = []
        self.state: dict[str, dict] = {}

        # Global inner-step counter; outer step fires every `outer_step_freq` calls to step().
        self._inner_step_count: int = 0

    # ------------------------------------------------------------------ API
    def add_constraint(self, name: str, lambda_init: Optional[float] = None) -> int:
        lam0 = self.lambda_warmstart if lambda_init is None else lambda_init
        self.constraints.append(
            {
                "name": name,
                "lambda_": float(lam0),
                "rho_p": self.rho_p_init,
                "prev_violation": None,
                "integral_ema": 0.0,
                "n_dual_updates": 0,
            }
        )
        self._constraint_data.append(None)
        return len(self.constraints) - 1

    def set_constraint_grads(
        self,
        constraint_idx: int,
        constraint_value: float,
        grad_map: dict[str, mx.array],
        *,
        update_dual: bool = True,
    ):
        self._constraint_data[constraint_idx] = (
            constraint_value,
            clone_flat_dict(grad_map),
            update_dual,
        )

    def reset_for_new_task(self, carry_factor: Optional[float] = None) -> None:
        """Apply λ persistence at task boundary.

        Also re-snaps θ_anchor to current params (a new task is a new outer cycle)
        and resets integrator/inner counter.
        """
        cf = self.lambda_carry_factor if carry_factor is None else carry_factor
        for ct in self.constraints:
            ct["lambda_"] = float(ct["lambda_"]) * cf
            ct["integral_ema"] = 0.0
            ct["prev_violation"] = None
        self._inner_step_count = 0
        self.rho_anchor = self.rho_anchor_init
        self._snapshot_anchor()

    # -------------------------------------------------------------- internal
    def _ensure_state(self, params_flat: dict[str, mx.array]):
        for path, param in params_flat.items():
            if path not in self.state:
                self.state[path] = {
                    "step": 0,
                    "exp_avg": mx.zeros_like(param),
                    "exp_avg_sq": mx.zeros_like(param),
                    "theta_anchor": mx.array(param),  # initial anchor = initial param
                }

    def _snapshot_anchor(self) -> None:
        """Snapshot current trainable params into theta_anchor for each state entry."""
        params_flat = flatten_tree(self.model.trainable_parameters())
        for path, param in params_flat.items():
            if path in self.state:
                self.state[path]["theta_anchor"] = mx.array(param)

    # ----------------------------------------------------------------- step
    def step(self, task_grads: dict[str, mx.array]):
        params_flat = flatten_tree(self.model.trainable_parameters())
        self._ensure_state(params_flat)

        beta1, beta2 = self.betas
        updated: dict[str, mx.array] = {}
        base_delta: dict[str, mx.array] = {}
        denom_map: dict[str, mx.array] = {}
        final_deltas: dict[str, mx.array] = {}

        # Resolve primal force per constraint using RAW violation (P-term, full bandwidth).
        active_projection_constraints: list[tuple[int, float, dict[str, mx.array], float]] = []
        for ci, cdata in enumerate(self._constraint_data):
            if cdata is None:
                continue
            constraint_value, grad_map, _ = cdata
            ct = self.constraints[ci]
            raw_violation = max(0.0, constraint_value)
            f_c = ct["lambda_"] + ct["rho_p"] * raw_violation
            if grad_map:
                active_projection_constraints.append((ci, constraint_value, grad_map, f_c))

        use_woodbury = (
            self.use_multi_constraint_woodbury and len(active_projection_constraints) > 1
        )
        woodbury_alpha = None
        woodbury_tau = None
        rho_p_inv_vec = None
        f_vec = None
        if use_woodbury:
            k = len(active_projection_constraints)
            woodbury_alpha = [[0.0 for _ in range(k)] for _ in range(k)]
            woodbury_tau = [0.0 for _ in range(k)]
            rho_p_inv = [
                1.0 / max(self.constraints[ci]["rho_p"], 1e-10)
                for (ci, _, _, _) in active_projection_constraints
            ]
            f_vals = [f_c for (_, _, _, f_c) in active_projection_constraints]
            rho_p_inv_vec = mx.array(rho_p_inv)
            f_vec = mx.array(f_vals)

        # ----- Per-parameter: Adam + Anchor pull + SM projection -----
        for path, param in params_flat.items():
            grad = task_grads.get(path)
            if grad is None:
                continue

            state = self.state[path]
            state["step"] += 1
            exp_avg = state["exp_avg"] * beta1 + grad * (1.0 - beta1)
            exp_avg_sq = state["exp_avg_sq"] * beta2 + (grad * grad) * (1.0 - beta2)
            state["exp_avg"] = exp_avg
            state["exp_avg_sq"] = exp_avg_sq

            bc1 = 1.0 - beta1 ** state["step"]
            bc2 = 1.0 - beta2 ** state["step"]
            m_hat = exp_avg / max(bc1, 1e-12)
            v_hat = exp_avg_sq / max(bc2, 1e-12)
            D = mx.sqrt(v_hat) + self.eps

            # Adam direction
            delta = -m_hat / D
            # ADMM proximal anchor: elastic pull toward θ_anchor (GALT §12.2).
            # Mass-weighted spring: f_anchor = ρ_a · D · (θ_anchor − θ); after /D it's ρ_a · (θ_anchor − θ).
            if self.rho_anchor > 0.0:
                delta = delta + self.rho_anchor * (state["theta_anchor"] - param)

            base_delta[path] = delta
            denom_map[path] = D

            if use_woodbury:
                j_tildes: list[Optional[mx.array]] = []
                for li, (_ci, _cv, grad_map, _fc) in enumerate(active_projection_constraints):
                    J = grad_map.get(path)
                    if J is None:
                        j_tildes.append(None)
                        continue
                    J_tilde = J / D
                    woodbury_tau[li] += scalar(mx.sum(J * delta))
                    j_tildes.append(J_tilde)
                for i, (_ci_i, _cv_i, gi, _fi) in enumerate(active_projection_constraints):
                    J_i = gi.get(path)
                    if J_i is None:
                        continue
                    for j, J_tilde_j in enumerate(j_tildes):
                        if J_tilde_j is None:
                            continue
                        woodbury_alpha[i][j] += scalar(mx.sum(J_i * J_tilde_j))
                continue

            for ci, _cv, grad_map, f_c in active_projection_constraints:
                J = grad_map.get(path)
                if J is None:
                    continue
                rho_p = self.constraints[ci]["rho_p"]
                J_tilde = J / D
                alpha_val = scalar(mx.sum(J * J_tilde))
                tau_val = scalar(mx.sum(J * delta))
                correction = (f_c + rho_p * tau_val) / (1.0 + rho_p * alpha_val + 1e-10)
                delta = delta - correction * J_tilde

            next_param = param
            if self.weight_decay > 0:
                next_param = next_param * (1.0 - self.lr * self.weight_decay)
            updated[path] = next_param + self.lr * delta
            final_deltas[path] = delta

        # ----- Solve Woodbury (multi-constraint path) -----
        if use_woodbury:
            rhs = rho_p_inv_vec * f_vec + mx.array(woodbury_tau)
            system = mx.diag(rho_p_inv_vec) + mx.array(woodbury_alpha)
            system_np = np.array(system)
            cond = np.linalg.cond(system_np)
            if cond > 1e6:
                for path, param in params_flat.items():
                    delta = base_delta.get(path)
                    if delta is None:
                        continue
                    D = denom_map[path]
                    for ci, _cv, grad_map, f_c in active_projection_constraints:
                        J = grad_map.get(path)
                        if J is None:
                            continue
                        rho_p = self.constraints[ci]["rho_p"]
                        J_tilde = J / D
                        alpha_val = scalar(mx.sum(J * J_tilde))
                        tau_val = scalar(mx.sum(J * delta))
                        correction = (f_c + rho_p * tau_val) / (1.0 + rho_p * alpha_val + 1e-10)
                        delta = delta - correction * J_tilde
                    next_param = param
                    if self.weight_decay > 0:
                        next_param = next_param * (1.0 - self.lr * self.weight_decay)
                    updated[path] = next_param + self.lr * delta
                    final_deltas[path] = delta
            else:
                coeffs = mx.array(np.linalg.solve(system_np, np.array(rhs)), dtype=rhs.dtype)
                for path, param in params_flat.items():
                    delta = base_delta.get(path)
                    if delta is None:
                        continue
                    D = denom_map[path]
                    for coeff, (_ci, _cv, grad_map, _fc) in zip(coeffs, active_projection_constraints):
                        J = grad_map.get(path)
                        if J is None:
                            continue
                        delta = delta - coeff * (J / D)
                    next_param = param
                    if self.weight_decay > 0:
                        next_param = next_param * (1.0 - self.lr * self.weight_decay)
                    updated[path] = next_param + self.lr * delta
                    final_deltas[path] = delta

        if updated:
            self.model.update(unflatten_tree(updated), strict=False)

        # ----- Integrator: smoothed post-residual for the dual integrator -----
        post_residuals: list[Optional[float]] = [None] * len(self.constraints)
        if self.use_post_residual_for_integral and final_deltas:
            for ci, cdata in enumerate(self._constraint_data):
                if cdata is None:
                    continue
                constraint_value, grad_map, _ = cdata
                if not grad_map:
                    continue
                j_dot_delta = 0.0
                for path, J in grad_map.items():
                    d = final_deltas.get(path)
                    if d is None:
                        continue
                    j_dot_delta += scalar(mx.sum(J * d))
                post_residuals[ci] = constraint_value + self.lr * j_dot_delta

        for ci, cdata in enumerate(self._constraint_data):
            if cdata is None:
                continue
            constraint_value, _grad_map, update_dual = cdata
            if not update_dual:
                continue
            ct = self.constraints[ci]

            if self.use_post_residual_for_integral and post_residuals[ci] is not None:
                signal = max(0.0, post_residuals[ci])
            else:
                signal = max(0.0, constraint_value)

            beta_i = self.integral_smoothing
            ct["integral_ema"] = beta_i * ct["integral_ema"] + (1.0 - beta_i) * signal

            # ρ_P adaptation (with FLOOR per §12.6, not just ceiling).
            raw_violation = max(0.0, constraint_value)
            if ct["prev_violation"] is not None:
                if abs(raw_violation) > self.rho_p_tau * abs(ct["prev_violation"]):
                    ct["rho_p"] = min(ct["rho_p"] * self.rho_p_growth, self.rho_p_max)
                elif self.rho_p_decay < 1.0:
                    ct["rho_p"] = max(self.rho_p_min, ct["rho_p"] * self.rho_p_decay)
            ct["prev_violation"] = raw_violation

        # ----- OUTER STEP: fires every `outer_step_freq` inner steps -----
        self._inner_step_count += 1
        if self._inner_step_count >= self.outer_step_freq:
            # 1. Dual update (λ ← max(0, λ + ρ_I · integral_ema)) per constraint
            for ct in self.constraints:
                lam_new = ct["lambda_"] + self.rho_i * ct["integral_ema"]
                ct["lambda_"] = min(self.lambda_max, max(0.0, lam_new))
                ct["n_dual_updates"] += 1
                ct["integral_ema"] = 0.0  # reset integrator for next outer window

            # 2. Snapshot θ_anchor at the NEW primal position (post-update).
            self._snapshot_anchor()

            # 3. Mild anchor-stiffness decay (optional; 1.0 = disabled).
            if self.rho_anchor_decay < 1.0:
                self.rho_anchor = max(0.0, self.rho_anchor * self.rho_anchor_decay)

            self._inner_step_count = 0

        self._constraint_data = [None] * len(self.constraints)
        mx.eval(
            self.model.parameters(),
            [s["exp_avg"] for s in self.state.values()],
            [s["exp_avg_sq"] for s in self.state.values()],
            [s["theta_anchor"] for s in self.state.values()],
        )

    # ------------------------------------------------------------ diagnostics
    def get_constraint_info(self) -> dict:
        return {
            ct["name"]: {
                "lambda_": ct["lambda_"],
                "rho": ct["rho_p"],
                "rho_p": ct["rho_p"],
                "rho_i": self.rho_i,
                "rho_anchor": self.rho_anchor,
                "violation_ema": ct["integral_ema"],
                "integral_ema": ct["integral_ema"],
                "n_dual_updates": ct["n_dual_updates"],
                "inner_step_count": self._inner_step_count,
            }
            for ct in self.constraints
        }
