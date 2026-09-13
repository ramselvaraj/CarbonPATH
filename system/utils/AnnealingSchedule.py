"""Pure feedback control for adaptive simulated annealing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True)
class AdaptiveScheduleConfig:
    max_total_moves: int = 75650
    moves_per_level: int = 100
    target_acceptance_start: float = 0.80
    target_acceptance_end: float = 0.05
    minimum_temperature: float = 1e-3
    maximum_temperature_factor: float = 4.0
    feedback_gain: float = 1.0
    minimum_uphill_proposals: int = 5
    minimum_multiplier: float = 0.5
    maximum_multiplier: float = 1.5
    fallback_cooling_rate: float = 0.98
    stagnation_moves: int = 5000
    maximum_reheats: int = 2
    reheat_factor: float = 2.0
    reheat_progress_limit: float = 0.80
    cost_epsilon: float = 1e-12

    def validate(self) -> None:
        if self.max_total_moves <= 0 or self.moves_per_level <= 0:
            raise ValueError("move budgets must be positive")
        if not 0 < self.target_acceptance_end < self.target_acceptance_start < 1:
            raise ValueError("acceptance targets must satisfy 0 < end < start < 1")
        if self.minimum_temperature <= 0:
            raise ValueError("minimum_temperature must be positive")
        if self.maximum_temperature_factor <= 1:
            raise ValueError("maximum_temperature_factor must exceed one")
        if self.feedback_gain < 0 or self.minimum_uphill_proposals < 1:
            raise ValueError("feedback gain and evidence threshold are invalid")
        if not 0 < self.minimum_multiplier <= self.maximum_multiplier:
            raise ValueError("temperature multipliers are invalid")
        if not 0 < self.fallback_cooling_rate < 1:
            raise ValueError("fallback_cooling_rate must be between zero and one")
        if self.stagnation_moves <= 0 or self.maximum_reheats < 0:
            raise ValueError("stagnation and reheats must be non-negative")
        if self.reheat_factor <= 1 or not 0 <= self.reheat_progress_limit <= 1:
            raise ValueError("reheat settings are invalid")
        if self.cost_epsilon <= 0:
            raise ValueError("cost_epsilon must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemperatureDecision:
    level: int
    moves_before: int
    moves_after: int
    temperature_before: float
    target_acceptance: float
    observed_acceptance: float | None
    evidence_count: int
    multiplier: float
    temperature_after: float
    best_cost_before: float
    best_cost_after: float
    stagnation_moves: int
    reheat_count: int
    action: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveTemperatureController:
    """Adjust temperature from observed uphill acceptance and stagnation."""

    def __init__(self, initial_temperature: float, config: AdaptiveScheduleConfig):
        config.validate()
        if not math.isfinite(initial_temperature) or initial_temperature <= 0:
            raise ValueError("initial_temperature must be finite and positive")
        self.config = config
        self.initial_temperature = initial_temperature
        self.maximum_temperature = (
            initial_temperature * config.maximum_temperature_factor
        )
        self.reheat_count = 0
        self.stagnation_moves = 0
        self._moves_since_improvement = 0
        self.history: list[TemperatureDecision] = []

    def target_acceptance(self, attempted_moves: int) -> float:
        progress = min(max(attempted_moves / self.config.max_total_moves, 0.0), 1.0)
        return self.config.target_acceptance_start * (
            self.config.target_acceptance_end
            / self.config.target_acceptance_start
        ) ** progress

    def _statistics(self, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
        uphill = 0
        accepted = 0
        epsilon = self.config.cost_epsilon
        for row in rows:
            diff = row.get("cost_diff")
            if not row.get("proposal_valid", row.get("new_cost") is not None):
                continue
            if not row.get("proposal_changed", True) or diff is None or diff <= epsilon:
                continue
            uphill += 1
            if row.get("move_accepted") is True:
                accepted += 1
        return uphill, accepted

    def next_temperature(
        self,
        *,
        level_index: int,
        temperature: float,
        rows: tuple[dict[str, Any], ...],
        attempted_moves: int,
        current_cost: float,
        best_cost_before: float,
        best_cost_after: float,
    ) -> float:
        del current_cost
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if attempted_moves > self.config.max_total_moves:
            raise ValueError("attempted moves exceed adaptive budget")

        uphill, accepted = self._statistics(rows)
        target = self.target_acceptance(attempted_moves)
        observed = None
        action = "fallback_cool"
        multiplier = self.config.fallback_cooling_rate
        if uphill >= self.config.minimum_uphill_proposals:
            observed = (accepted + 0.5) / (uphill + 1)
            error = target - observed
            multiplier = min(
                self.config.maximum_multiplier,
                max(self.config.minimum_multiplier, math.exp(self.config.feedback_gain * error)),
            )
            action = "feedback_heat" if multiplier > 1 else "feedback_cool"
            if math.isclose(multiplier, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                action = "feedback_hold"

        if best_cost_after < best_cost_before - self.config.cost_epsilon:
            self._moves_since_improvement = 0
        else:
            self._moves_since_improvement += len(rows)
        self.stagnation_moves = self._moves_since_improvement

        progress = attempted_moves / self.config.max_total_moves
        should_reheat = (
            self.stagnation_moves >= self.config.stagnation_moves
            and progress < self.config.reheat_progress_limit
            and self.reheat_count < self.config.maximum_reheats
        )
        before = temperature
        next_temperature = temperature * multiplier
        if should_reheat:
            next_temperature = max(next_temperature, temperature * self.config.reheat_factor)
            self.reheat_count += 1
            self.stagnation_moves = 0
            self._moves_since_improvement = 0
            action = "reheat"
            multiplier = next_temperature / temperature
        next_temperature = min(
            self.maximum_temperature,
            max(self.config.minimum_temperature, next_temperature),
        )
        multiplier = next_temperature / before
        self.history.append(
            TemperatureDecision(
                level=level_index,
                moves_before=attempted_moves - len(rows),
                moves_after=attempted_moves,
                temperature_before=before,
                target_acceptance=target,
                observed_acceptance=observed,
                evidence_count=uphill,
                multiplier=multiplier,
                temperature_after=next_temperature,
                best_cost_before=best_cost_before,
                best_cost_after=best_cost_after,
                stagnation_moves=self.stagnation_moves,
                reheat_count=self.reheat_count,
                action=action,
            )
        )
        return next_temperature
