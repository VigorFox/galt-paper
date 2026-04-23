"""MLX port of the Sherman-Morrison AVBD-Hessian optimizer."""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_utils import clone_flat_dict, flatten_tree, scalar, unflatten_tree


class AVBDHessianOptimizer:
    """Adam + Sherman-Morrison constraint projection for MLX models."""

    def __init__(
        self,
        model,
        lr: float = 2e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rho_init: float = 1.0,
        rho_max: float = 50.0,
        rho_growth: float = 1.5,
        rho_decay: float = 1.0,
        tau_decrease: float = 0.9,
        use_multi_constraint_woodbury: bool = True,
        violation_ema_beta: float = 0.9,
        lambda_damping: float = 1.0,
        lambda_max: float = float("inf"),
        # Frame-age reset (inspired by PhysX AVBD warm-start mechanism)
        lambda_max_age: int = 0,
        lambda_age_decay: float = 0.95,
        # Inner/outer loop: batch dual updates (inspired by PhysX innerIterations)
        # dual_update_freq=K means K primal steps per 1 dual update (physics: K=10)
        # Fixes: stochastic noise averaging, primal convergence, diminishing step
        dual_update_freq: int = 1,
        dual_step_diminish: bool = False,
        # v10: post-projection residual for dual update (physics-correct)
        # Physics AVBD updates lambda on POST-solve residual, not pre-projection violation.
        # We compute analytical predicted residual: C_post = C + lr * J . delta_final
        # When projection succeeds: residual ~ 0, lambda doesn't grow (regime-adaptive)
        # When projection fails (real persistent violation): residual stays high, lambda grows
        use_post_projection_residual: bool = False,
    ):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.rho_init = rho_init
        self.rho_max = rho_max
        self.rho_growth = rho_growth
        self.rho_decay = rho_decay
        self.tau_decrease = tau_decrease
        self.use_multi_constraint_woodbury = use_multi_constraint_woodbury
        self.violation_ema_beta = violation_ema_beta
        self.lambda_damping = lambda_damping
        self.lambda_max = lambda_max
        # Frame-age: steps since last active violation (> 0 means constraint was active)
        # When age exceeds max_age, lambda is hard-reset to 0 (prevents runaway on sparse violation)
        # 0 = disabled (legacy behavior)
        self.lambda_max_age = lambda_max_age
        self.lambda_age_decay = lambda_age_decay
        self.dual_update_freq = max(1, dual_update_freq)
        self.dual_step_diminish = dual_step_diminish
        self.use_post_projection_residual = use_post_projection_residual

        self.constraints: list[dict] = []
        self._constraint_data: list[Optional[tuple[float, dict[str, mx.array], bool]]] = []
        self.state: dict[str, dict] = {}

    def add_constraint(self, name: str) -> int:
        self.constraints.append(
            {
                "name": name,
                "lambda_": 0.0,
                "rho": self.rho_init,
                "prev_violation": None,
                "violation_ema": 0.0,
                "steps_since_violation": 0,
                # Inner/outer loop state
                "violation_accum": 0.0,
                "inner_step_count": 0,
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
        self._constraint_data[constraint_idx] = (constraint_value, clone_flat_dict(grad_map), update_dual)

    def _ensure_state(self, params_flat: dict[str, mx.array]):
        for path, param in params_flat.items():
            if path not in self.state:
                self.state[path] = {
                    "step": 0,
                    "exp_avg": mx.zeros_like(param),
                    "exp_avg_sq": mx.zeros_like(param),
                }

    def step(self, task_grads: dict[str, mx.array]):
        params_flat = flatten_tree(self.model.trainable_parameters())
        self._ensure_state(params_flat)

        beta1, beta2 = self.betas
        updated = {}
        base_delta = {}
        denom_map = {}
        # v10: track final delta per param for post-projection residual computation
        final_deltas: dict[str, mx.array] = {}

        active_projection_constraints = []
        for ci, cdata in enumerate(self._constraint_data):
            if cdata is None:
                continue
            constraint_value, grad_map, _update_dual = cdata
            # Update violation EMA (once per step, before primal use)
            ct = self.constraints[ci]
            violation = max(0.0, constraint_value)
            ct["violation_ema"] = (
                self.violation_ema_beta * ct["violation_ema"]
                + (1.0 - self.violation_ema_beta) * violation
            )
            if grad_map:
                active_projection_constraints.append((ci, constraint_value, grad_map))

        woodbury_coeffs = None
        use_woodbury_step = False
        rho_inv_vec = None
        f_vec = None
        if self.use_multi_constraint_woodbury and len(active_projection_constraints) > 1:
            use_woodbury_step = True
            k = len(active_projection_constraints)
            alpha = [[0.0 for _ in range(k)] for _ in range(k)]
            tau = [0.0 for _ in range(k)]
            rho_inv = []
            f_vals = []
            for local_index, (ci, constraint_value, grad_map) in enumerate(active_projection_constraints):
                ct = self.constraints[ci]
                rho = ct["rho"]
                f_c = ct["lambda_"] + rho * ct["violation_ema"]
                rho_inv.append(1.0 / max(rho, 1e-10))
                f_vals.append(f_c)
            rho_inv_vec = mx.array(rho_inv)
            f_vec = mx.array(f_vals)

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
            delta = -m_hat / D
            base_delta[path] = delta
            denom_map[path] = D

            if use_woodbury_step:
                j_tildes = []
                for local_index, (_ci, _constraint_value, grad_map) in enumerate(active_projection_constraints):
                    J = grad_map.get(path)
                    if J is None:
                        j_tildes.append(None)
                        continue
                    J_tilde = J / D
                    tau[local_index] += scalar(mx.sum(J * delta))
                    j_tildes.append(J_tilde)
                for i, (_ci_i, _constraint_value_i, grad_map_i) in enumerate(active_projection_constraints):
                    J_i = grad_map_i.get(path)
                    if J_i is None:
                        continue
                    for j, J_tilde_j in enumerate(j_tildes):
                        if J_tilde_j is None:
                            continue
                        alpha[i][j] += scalar(mx.sum(J_i * J_tilde_j))
                continue

            for ci, cdata in enumerate(self._constraint_data):
                if cdata is None:
                    continue
                constraint_value, grad_map, _update_dual = cdata
                J = grad_map.get(path)
                if J is None:
                    continue

                ct = self.constraints[ci]
                rho = ct["rho"]
                f_c = ct["lambda_"] + rho * ct["violation_ema"]

                J_tilde = J / D
                alpha_val = scalar(mx.sum(J * J_tilde))
                tau_val = scalar(mx.sum(J * delta))
                correction = (f_c + rho * tau_val) / (1.0 + rho * alpha_val + 1e-10)
                delta = delta - correction * J_tilde

            next_param = param
            if self.weight_decay > 0:
                next_param = next_param * (1.0 - self.lr * self.weight_decay)
            updated[path] = next_param + self.lr * delta
            final_deltas[path] = delta

        if use_woodbury_step:
            rhs = rho_inv_vec * f_vec + mx.array(tau)
            system = mx.diag(rho_inv_vec) + mx.array(alpha)
            system_np = np.array(system)
            cond = np.linalg.cond(system_np)
            if cond > 1e6:
                # Ill-conditioned: fallback to sequential SM
                use_woodbury_step = False
                for path, param in params_flat.items():
                    delta = base_delta.get(path)
                    if delta is None:
                        continue
                    D = denom_map[path]
                    for ci_local, (ci, constraint_value, grad_map) in enumerate(active_projection_constraints):
                        J = grad_map.get(path)
                        if J is None:
                            continue
                        ct = self.constraints[ci]
                        rho = ct["rho"]
                        f_c = ct["lambda_"] + rho * ct["violation_ema"]
                        J_tilde = J / D
                        alpha_val = scalar(mx.sum(J * J_tilde))
                        tau_val = scalar(mx.sum(J * delta))
                        correction = (f_c + rho * tau_val) / (1.0 + rho * alpha_val + 1e-10)
                        delta = delta - correction * J_tilde
                    next_param = param
                    if self.weight_decay > 0:
                        next_param = next_param * (1.0 - self.lr * self.weight_decay)
                    updated[path] = next_param + self.lr * delta
                    final_deltas[path] = delta
            else:
                woodbury_coeffs = mx.array(
                    np.linalg.solve(system_np, np.array(rhs)),
                    dtype=rhs.dtype,
                )
                for path, param in params_flat.items():
                    delta = base_delta.get(path)
                    if delta is None:
                        continue
                    D = denom_map[path]
                    for coeff, (_ci, _constraint_value, grad_map) in zip(woodbury_coeffs, active_projection_constraints):
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

        # v10: compute predicted post-projection residual per constraint
        # C_post ~ C_pre + lr * J . delta_final  (linearization of C around current params)
        # This is the analytical residual the SM step should leave; use it for dual update
        post_residuals = [None] * len(self.constraints)
        if self.use_post_projection_residual and final_deltas:
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
            rho = ct["rho"]
            rho_dual = min(self.rho_max, rho * rho / (rho + self.rho_max))

            # Frame-age-aware dual update (inspired by PhysX AVBD lambda warm-start)
            # v10: optionally use predicted post-projection residual instead of raw violation
            if self.use_post_projection_residual and post_residuals[ci] is not None:
                violation = max(0.0, post_residuals[ci])
            else:
                violation = max(0.0, constraint_value)
            is_active = violation > 0.0

            # Inner/outer loop: accumulate violation, batch dual update every K steps
            ct["violation_accum"] += violation
            ct["inner_step_count"] += 1
            do_dual_update = ct["inner_step_count"] >= self.dual_update_freq

            if do_dual_update:
                avg_violation = ct["violation_accum"] / ct["inner_step_count"]
                ct["n_dual_updates"] += 1
                # Diminishing step size: ρ/√n (Robbins-Monro condition for stochastic AL)
                if self.dual_step_diminish and ct["n_dual_updates"] > 1:
                    import math
                    effective_rho_dual = rho_dual / math.sqrt(ct["n_dual_updates"])
                else:
                    effective_rho_dual = rho_dual

                if self.lambda_max_age > 0:
                    # Frame-age mechanism enabled
                    if avg_violation > 0.0:
                        ct["lambda_"] = min(
                            self.lambda_max,
                            max(0.0, ct["lambda_"] + effective_rho_dual * avg_violation),
                        )
                        ct["steps_since_violation"] = 0
                    else:
                        ct["steps_since_violation"] += self.dual_update_freq
                        if ct["steps_since_violation"] > self.lambda_max_age:
                            ct["lambda_"] = 0.0
                        else:
                            ct["lambda_"] *= self.lambda_age_decay ** self.dual_update_freq
                else:
                    # Standard AL dual update (batched)
                    ct["lambda_"] = min(
                        self.lambda_max,
                        max(0.0, self.lambda_damping * ct["lambda_"] + effective_rho_dual * avg_violation),
                    )

                # Reset accumulator for next inner window
                ct["violation_accum"] = 0.0
                ct["inner_step_count"] = 0

            if ct["prev_violation"] is not None:
                if abs(violation) > self.tau_decrease * abs(ct["prev_violation"]):
                    ct["rho"] = min(ct["rho"] * self.rho_growth, self.rho_max)
                elif self.rho_decay < 1.0:
                    ct["rho"] = max(self.rho_init, ct["rho"] * self.rho_decay)
            ct["prev_violation"] = violation

        self._constraint_data = [None] * len(self.constraints)
        mx.eval(
            self.model.parameters(),
            [state["exp_avg"] for state in self.state.values()],
            [state["exp_avg_sq"] for state in self.state.values()],
        )

    def get_constraint_info(self) -> dict:
        return {
            ct["name"]: {
                "lambda_": ct["lambda_"],
                "rho": ct["rho"],
                "violation_ema": ct["violation_ema"],
                "steps_since_violation": ct["steps_since_violation"],
                "n_dual_updates": ct["n_dual_updates"],
            }
            for ct in self.constraints
        }
