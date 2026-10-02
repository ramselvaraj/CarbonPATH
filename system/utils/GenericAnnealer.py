"""Domain-independent simulated annealing control flow.

Hardware construction, workload evaluation, and objective calculation enter as
callables.  This module owns only proposal accounting, Metropolis acceptance,
current/best state, cooling, and the trace.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


@dataclass(frozen=True)
class AnnealingResult:
    best_design: Any
    best_evaluation: Any
    best_cost: float
    current_design: Any
    current_evaluation: Any
    current_cost: float
    trace: tuple[dict, ...]


def _validate_schedule(
    initial_temperature,
    freezing_temperature,
    moves_per_temperature,
    cooling_rate,
):
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be positive")
    if freezing_temperature <= 0 or freezing_temperature >= initial_temperature:
        raise ValueError(
            "freezing_temperature must be positive and below initial_temperature"
        )
    if moves_per_temperature <= 0:
        raise ValueError("moves_per_temperature must be positive")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between zero and one")


def anneal(
    *,
    initial_design,
    propose: Callable,
    evaluate: Callable,
    identity: Callable,
    rng,
    initial_temperature: float,
    freezing_temperature: float,
    moves_per_temperature: int,
    cooling_rate: float,
    temperature_controller=None,
    max_total_moves=None,
    level_callback=None,
):
    """Search from ``initial_design`` using injected domain operations.

    ``propose(current, rng)`` returns ``(candidate, move_name)`` and may raise
    ``UnsupportedEvaluation`` when no valid candidate can be formed.
    ``evaluate(design)`` returns ``(evaluation, scalar_cost)``.
    """
    _validate_schedule(
        initial_temperature,
        freezing_temperature,
        moves_per_temperature,
        cooling_rate,
    )
    if temperature_controller is not None and (
        max_total_moves is None or max_total_moves <= 0
    ):
        raise ValueError("adaptive annealing requires a positive move budget")

    current_design = copy.deepcopy(initial_design)
    current_evaluation, current_cost = evaluate(current_design)
    current_cost = float(current_cost)
    if not math.isfinite(current_cost):
        raise ValueError("initial design has a non-finite objective")

    best_design = copy.deepcopy(current_design)
    best_evaluation = current_evaluation
    best_cost = current_cost
    best_identity = identity(best_design)
    temperature = float(initial_temperature)
    attempt = 0
    trace = []

    while (
        attempt < max_total_moves
        if temperature_controller is not None
        else temperature > freezing_temperature
    ):
        level_rows = []
        remaining = (
            max_total_moves - attempt
            if temperature_controller is not None
            else moves_per_temperature
        )
        best_cost_before_level = best_cost
        for inner_iteration in range(min(moves_per_temperature, remaining)):
            attempt += 1
            row = {
                "temperature": temperature,
                "inner_iter": inner_iteration,
                "attempt": attempt,
                "current_cost_before": current_cost,
                "best_cost_before": best_cost,
            }
            try:
                candidate, move_name = propose(current_design, rng)
            except UnsupportedEvaluation as error:
                row.update(
                    {
                        "move_name": None,
                        "proposal_valid": False,
                        "proposal_changed": False,
                        "proposal_error": str(error),
                        "candidate_cost": None,
                        "cost_diff": None,
                        "move_accepted": False,
                        "acceptance_reason": "invalid",
                        "current_cost_after": current_cost,
                        "best_cost_after": best_cost,
                    }
                )
                trace.append(row)
                level_rows.append(row)
                continue

            candidate_identity = identity(candidate)
            changed = candidate_identity != identity(current_design)
            row.update(
                {
                    "move_name": move_name,
                    "proposal_valid": True,
                    "proposal_changed": changed,
                    "proposal_error": None,
                }
            )
            if not changed:
                row.update(
                    {
                        "candidate_cost": None,
                        "cost_diff": None,
                        "move_accepted": False,
                        "acceptance_reason": "unchanged",
                        "current_cost_after": current_cost,
                        "best_cost_after": best_cost,
                    }
                )
                trace.append(row)
                level_rows.append(row)
                continue

            candidate_evaluation, candidate_cost = evaluate(candidate)
            candidate_cost = float(candidate_cost)
            if not math.isfinite(candidate_cost):
                raise ValueError("candidate design has a non-finite objective")
            cost_diff = candidate_cost - current_cost
            if cost_diff < 0:
                accepted = True
                reason = "improvement"
            elif rng.random() < math.exp(-cost_diff / temperature):
                accepted = True
                reason = "metropolis"
            else:
                accepted = False
                reason = "rejected"

            if accepted:
                current_design = copy.deepcopy(candidate)
                current_evaluation = candidate_evaluation
                current_cost = candidate_cost

                improves_best = candidate_cost < best_cost
                wins_tie = (
                    candidate_cost == best_cost
                    and candidate_identity < best_identity
                )
                if improves_best or wins_tie:
                    best_design = copy.deepcopy(candidate)
                    best_evaluation = candidate_evaluation
                    best_cost = candidate_cost
                    best_identity = candidate_identity

            row.update(
                {
                    "candidate_identity": candidate_identity,
                    "candidate_evaluation": candidate_evaluation,
                    "candidate_cost": candidate_cost,
                    "cost_diff": cost_diff,
                    "move_accepted": accepted,
                    "acceptance_reason": reason,
                    "current_cost_after": current_cost,
                    "best_cost_after": best_cost,
                }
            )
            trace.append(row)
            level_rows.append(row)

        if temperature_controller is None:
            temperature *= cooling_rate
            if level_callback is not None:
                level_callback(None, tuple(level_rows))
        else:
            temperature = temperature_controller.next_temperature(
                level_index=len(temperature_controller.history),
                temperature=temperature,
                rows=tuple(level_rows),
                attempted_moves=attempt,
                current_cost=current_cost,
                best_cost_before=best_cost_before_level,
                best_cost_after=best_cost,
            )
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("adaptive controller returned invalid temperature")
            if level_callback is not None:
                level_callback(temperature_controller.history[-1], tuple(level_rows))

    return AnnealingResult(
        best_design=best_design,
        best_evaluation=best_evaluation,
        best_cost=best_cost,
        current_design=current_design,
        current_evaluation=current_evaluation,
        current_cost=current_cost,
        trace=tuple(trace),
    )
