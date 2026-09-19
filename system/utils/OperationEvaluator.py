"""Operation evaluator contract and registry.

An operation evaluator estimates one ATLAS operation at an assigned endpoint.
Every evaluator returns only its compute latency and dynamic energy. The
executor owns operation identity, placement, tensor residency, tensor movement,
totals, and reporting.

Evaluators are replaceable. A new GEMM or non-GEMM model is a new evaluator
registration, not a new branch in the executor.
"""

from __future__ import annotations

from dataclasses import dataclass

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


@dataclass(frozen=True)
class OperationEstimate:
    operation_id: str
    evaluator_id: str
    compute_latency_ns: float
    dynamic_energy_pj: float


@dataclass
class EvaluationContext:
    cache: object
    architecture: dict
    system: object
    activation_from_dram: bool = True
    output_to_dram: bool = True


class OperationEvaluator:
    """Base contract for one operation's compute estimate."""

    evaluator_id = ""

    def evaluate(
        self, operation, placement, context
    ) -> OperationEstimate:  # pragma: no cover - interface
        raise NotImplementedError


class EvaluatorRegistry:
    """Maps evaluation-profile evaluator IDs to evaluator implementations."""

    def __init__(self):
        self._evaluators = {}

    def register(self, evaluator):
        if not isinstance(evaluator, OperationEvaluator):
            raise ValueError("registered evaluator must be an OperationEvaluator")
        if not evaluator.evaluator_id:
            raise ValueError("registered evaluator must declare an evaluator_id")
        self._evaluators[evaluator.evaluator_id] = evaluator

    def get(self, evaluator_id):
        try:
            return self._evaluators[evaluator_id]
        except KeyError:
            raise UnsupportedEvaluation(f"unknown evaluator '{evaluator_id}'")

    def __contains__(self, evaluator_id):
        return evaluator_id in self._evaluators
