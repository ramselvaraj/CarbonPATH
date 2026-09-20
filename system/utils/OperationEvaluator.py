"""Operation evaluator contract, input adapter contract, and registry.

An operation evaluator estimates one ATLAS operation at an assigned endpoint.
Every evaluator returns only its compute latency and dynamic energy. The
executor owns operation identity, placement, tensor residency, tensor movement,
totals, and reporting.

An evaluator never reads the raw ATLAS artifact. It receives a typed input view
built by its paired operation input adapter, which reads the operation's
read-only ATLAS source view. This keeps model-specific ATLAS requirements local
to the model that needs them.

Evaluators are replaceable. A new GEMM or non-GEMM model is a new evaluator
registration plus its input adapter, not a new branch in the executor.
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


class OperationInputAdapter:
    """Builds one evaluator's typed input from an ATLAS operation.

    Implementations read only the facts their evaluator needs, through
    ``operation.source`` and the executor-provided tensor access plan.
    """

    input_adapter_id = ""
    operation_type = ""

    def build_input(
        self, operation, access_plan
    ):  # pragma: no cover - interface
        raise NotImplementedError


class OperationEvaluator:
    """Base contract for one operation's compute estimate."""

    evaluator_id = ""
    operation_type = ""
    input_adapter_id = ""

    def evaluate(
        self, evaluator_input, placement, context
    ):  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class EvaluatorBinding:
    """An evaluator paired with the input adapter that feeds it."""

    evaluator: OperationEvaluator
    input_adapter: OperationInputAdapter

    @property
    def evaluator_id(self):
        return self.evaluator.evaluator_id

    @property
    def input_adapter_id(self):
        return self.input_adapter.input_adapter_id

    @property
    def operation_type(self):
        return self.evaluator.operation_type

    def estimate(self, operation, access_plan, placement, context):
        evaluator_input = self.input_adapter.build_input(operation, access_plan)
        return self.evaluator.evaluate(evaluator_input, placement, context)


class EvaluatorRegistry:
    """Maps evaluation-profile evaluator IDs to evaluator bindings."""

    def __init__(self):
        self._bindings = {}

    def register(self, evaluator, input_adapter):
        if not isinstance(evaluator, OperationEvaluator):
            raise ValueError("registered evaluator must be an OperationEvaluator")
        if not isinstance(input_adapter, OperationInputAdapter):
            raise ValueError(
                "registered input adapter must be an OperationInputAdapter"
            )
        if not evaluator.evaluator_id:
            raise ValueError("registered evaluator must declare an evaluator_id")
        if not input_adapter.input_adapter_id:
            raise ValueError(
                "registered input adapter must declare an input_adapter_id"
            )
        if evaluator.input_adapter_id != input_adapter.input_adapter_id:
            raise ValueError(
                f"evaluator '{evaluator.evaluator_id}' requires input adapter "
                f"'{evaluator.input_adapter_id}', got "
                f"'{input_adapter.input_adapter_id}'"
            )
        if evaluator.operation_type != input_adapter.operation_type:
            raise ValueError(
                f"evaluator '{evaluator.evaluator_id}' operation type "
                f"'{evaluator.operation_type}' does not match input adapter "
                f"operation type '{input_adapter.operation_type}'"
            )
        if evaluator.evaluator_id in self._bindings:
            raise ValueError(
                f"duplicate evaluator id '{evaluator.evaluator_id}'"
            )
        self._bindings[evaluator.evaluator_id] = EvaluatorBinding(
            evaluator, input_adapter
        )

    def get(self, evaluator_id):
        try:
            return self._bindings[evaluator_id]
        except KeyError:
            raise UnsupportedEvaluation(f"unknown evaluator '{evaluator_id}'")

    def __contains__(self, evaluator_id):
        return evaluator_id in self._bindings
