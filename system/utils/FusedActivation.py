"""Fused activation: a ReLU that runs on the systolic arrays of its GEMM.

Fusion is a property of one GEMM -> ReLU handoff. The movement policy
``fused_sa_activation_v1`` says that the ReLU runs where the GEMM output is, so
the tensor does not move. The executor reads ``fuses_consumer`` and makes the
GEMM the producer of the next handoff.

``sa_fused_relu_v1`` is the evaluator of the fused ReLU. An evaluation profile
names it with the key ``relu@sa``. Its latency and energy come from the
evaluator settings. The default of each is zero, and the systolic array gets
no area or power for the ReLU hardware: these are assumed rates, not hardware
characterizations.
"""

from __future__ import annotations

from dataclasses import dataclass

from system.utils.OperationEvaluator import (
    OperationEstimate,
    OperationEvaluator,
    OperationInputAdapter,
)
from system.utils.TensorMovement import (
    MOVEMENT_POLICIES,
    DistributedMovementPlan,
    _endpoint_value,
    _product,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.WorkloadMapping import (
    FUSABLE_OPERATIONS,
    FUSED_ACTIVATION_POLICY,
)


#: The evaluation-profile key of the evaluator for a ReLU on a systolic array.
FUSED_RELU_EVALUATOR_KEY = "relu@sa"

_SETTINGS = ("latency_ns", "energy_per_element_pj")


class FusedSaActivationService:
    """Movement policy of a handoff whose consumer is fused with its GEMM."""

    policy_id = FUSED_ACTIVATION_POLICY
    requires_prepared_operations = False
    fuses_consumer = True

    def __init__(self, transfer_estimator=None):
        pass

    def plan(self, request) -> DistributedMovementPlan:
        producer = request.producer_operation
        consumer = request.consumer_operation
        if (
            producer.operation_type != "gemm"
            or consumer.operation_type not in FUSABLE_OPERATIONS
        ):
            raise UnsupportedEvaluation(
                f"movement policy '{self.policy_id}' fuses a ReLU with the GEMM "
                f"directly before it, not {producer.operation_type}->"
                f"{consumer.operation_type}"
            )
        if (
            request.consumer_assignment.endpoint_kind != "sa"
            or request.consumer_assignment.endpoint_ids
            != request.producer_assignment.endpoint_ids
        ):
            raise UnsupportedEvaluation(
                f"fused operation '{consumer.operation_id}' must use the "
                f"systolic arrays of '{producer.operation_id}'"
            )
        endpoint = _endpoint_value(request.producer_assignment.endpoint_ids)
        return DistributedMovementPlan(
            method="fused",
            source_endpoint=endpoint,
            destination_endpoint=endpoint,
            logical_bytes=_product(request.tensor_shape),
            physical_bytes=0,
            charged_latency_ns=0.0,
            charged_energy_pj=0.0,
            transfer_slices=(),
        )


@dataclass(frozen=True)
class SaFusedReluInput:
    operation_id: str
    element_count: int


class SaFusedReluInputAdapter(OperationInputAdapter):
    """Input adapter for the fused ReLU model."""

    input_adapter_id = "sa_fused_relu_input_v1"
    operation_type = "relu"

    def build_input(self, operation, access_plan):
        if operation.operation_type != "relu":
            raise UnsupportedEvaluation(
                f"SaFusedReluInputAdapter cannot read "
                f"'{operation.operation_type}' operation "
                f"'{operation.operation_id}'"
            )
        return SaFusedReluInput(
            operation_id=operation.operation_id,
            element_count=operation.element_count,
        )


class SaFusedReluEvaluator(OperationEvaluator):
    """Operation evaluator for a ReLU fused with the GEMM on its SAs.

    Compute latency is the ``latency_ns`` setting. Dynamic energy is the
    element count times the ``energy_per_element_pj`` setting.
    """

    evaluator_id = "sa_fused_relu_v1"
    operation_type = "relu"
    input_adapter_id = "sa_fused_relu_input_v1"

    @staticmethod
    def _setting(settings, name):
        value = settings.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise UnsupportedEvaluation(
                f"fused ReLU setting '{name}' must be a non-negative number"
            )
        return float(value)

    def evaluate(self, evaluator_input, assignment, context) -> OperationEstimate:
        if assignment.endpoint_kind != "sa":
            raise UnsupportedEvaluation(
                f"{evaluator_input.operation_id}: the fused ReLU evaluator "
                "requires assignment on systolic arrays"
            )
        settings = context.evaluator_settings or {}
        unknown = sorted(set(settings) - set(_SETTINGS))
        if unknown:
            raise UnsupportedEvaluation(
                "unsupported fused ReLU evaluator setting(s): " + ", ".join(unknown)
            )
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=self._setting(settings, "latency_ns"),
            dynamic_energy_pj=evaluator_input.element_count
            * self._setting(settings, "energy_per_element_pj"),
        )


def register_fused_activation(registry=None):
    """Register the movement policy, and the evaluator when given a registry.

    The movement policy registration is idempotent.
    """
    MOVEMENT_POLICIES.setdefault(
        FusedSaActivationService.policy_id, FusedSaActivationService
    )
    if registry is not None:
        registry.register(SaFusedReluEvaluator(), SaFusedReluInputAdapter())
