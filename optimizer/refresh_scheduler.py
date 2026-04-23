"""Refresh scheduling helpers shared by dense AVBD-Hessian experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RefreshConfig:
    refresh_period: int = 10
    refresh_cstr_trigger: float = 0.3
    adaptive_refresh: bool = False
    adaptive_refresh_increment: int = 2
    adaptive_refresh_max_period: int = 24
    adaptive_refresh_safe_ratio: float = 0.5


class RefreshScheduler:
    """Decide when to do a full global refresh versus a local-only step."""

    def __init__(self, config: RefreshConfig | None = None):
        self.config = config or RefreshConfig()
        self.global_count = 0
        self.local_count = 0
        self.step_num = 0
        self.steps_since_global = 0
        self.safe_local_streak = 0
        self.last_effective_refresh_period = self.config.refresh_period
        self.max_effective_refresh_period = self.config.refresh_period

    def needs_refresh(self, constraint_vals: list[float] | None = None) -> bool:
        self.step_num += 1
        self.steps_since_global += 1
        max_c = max((abs(value) for value in (constraint_vals or [])), default=0.0)

        if max_c > self.config.refresh_cstr_trigger:
            self.safe_local_streak = 0
            self.last_effective_refresh_period = self.config.refresh_period
            return True

        effective_period = self.config.refresh_period
        if self.config.adaptive_refresh:
            safe_threshold = self.config.refresh_cstr_trigger * self.config.adaptive_refresh_safe_ratio
            if max_c <= safe_threshold:
                self.safe_local_streak += 1
            else:
                self.safe_local_streak = 0
            effective_period = min(
                self.config.adaptive_refresh_max_period,
                self.config.refresh_period + self.safe_local_streak * self.config.adaptive_refresh_increment,
            )
            self.last_effective_refresh_period = effective_period
            self.max_effective_refresh_period = max(self.max_effective_refresh_period, effective_period)

        return self.steps_since_global >= effective_period

    def mark_step(self, is_global: bool):
        if is_global:
            self.global_count += 1
            self.steps_since_global = 0
            self.safe_local_streak = 0
        else:
            self.local_count += 1

    def stats(self) -> dict:
        total = self.global_count + self.local_count
        return {
            "global_backprop_calls": self.global_count,
            "local_only_steps": self.local_count,
            "global_ratio": self.global_count / max(1, total),
            "last_effective_refresh_period": self.last_effective_refresh_period,
            "max_effective_refresh_period": self.max_effective_refresh_period,
        }
