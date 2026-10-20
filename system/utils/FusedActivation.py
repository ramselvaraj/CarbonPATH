"""Fused activation: an activation that runs on the systolic arrays of its GEMM.

Fusion is a property of one GEMM -> activation handoff. The movement policy
``fused_sa_activation_v1`` says that the activation runs where the GEMM output
is, so the tensor does not move. The executor reads ``fuses_consumer`` and
makes the GEMM the producer of the next handoff.

Each fusable activation has a fused evaluator, ``sa_fused_<type>_v1``. An
evaluation profile names it with the key ``<type>@sa``, for example
``relu@sa``. Its latency and energy come from the evaluator settings:

* ``latency_ns``              latency of one fused operation; default ``0.0``.
* ``cycles_per_group``        SA cycles for each group of elements, one group
                              being as many elements as the SA is wide; default
                              ``0.0``.
* ``energy_per_element_pj``   energy for each tensor element; default ``0.0``.

The elements are shared equally by the SAs of the GEMM. The latency of the
group part is the number of groups for one SA, times ``cycles_per_group``,
divided by the clock of that SA. With several SAs the slowest one counts.

The systolic array gets no area or power for the activation hardware. The
settings are assumed rates, not hardware characterizations.
"""

from __future__ import annotations

import math
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


def fused_evaluator_key(operation_type):
    """The evaluation-profile key of the evaluator for a fused activation."""
    return f"{operation_type}@sa"


def fusion_types(profile):
    """The activation types that the profile can evaluate when fused."""
    return tuple(
        operation_type
        for operation_type in FUSABLE_OPERATIONS
        if fused_evaluator_key(operation_type) in profile.evaluators
    )

_SETTINGS = ("latency_ns", "cycles_per_group", "energy_per_element_pj")


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
                f"movement policy '{self.policy_id}' fuses an elementwise "
                f"activation with the GEMM directly before it, not "
                f"{producer.operation_type}->"
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
    """Input adapter for the fused activation model."""

    def __init__(self, operation_type="relu"):
        self.operation_type = operation_type
        self.input_adapter_id = f"sa_fused_{operation_type}_input_v1"

    def build_input(self, operation, access_plan):
        if operation.operation_type != self.operation_type:
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
    """Operation evaluator for an activation fused with the GEMM on its SAs.

    Compute latency is the ``latency_ns`` setting plus the group latency: the
    groups of SA width that one SA handles, times the ``cycles_per_group``
    setting, divided by the SA clock. Dynamic energy is the element count times
    the ``energy_per_element_pj`` setting.
    """

    def __init__(self, operation_type="relu"):
        self.operation_type = operation_type
        self.evaluator_id = f"sa_fused_{operation_type}_v1"
        self.input_adapter_id = f"sa_fused_{operation_type}_input_v1"

    @staticmethod
    def _setting(settings, name):
        value = settings.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise UnsupportedEvaluation(
                f"fused activation setting '{name}' must be a non-negative number"
            )
        return float(value)

    def evaluate(self, evaluator_input, assignment, context) -> OperationEstimate:
        if assignment.endpoint_kind != "sa":
            raise UnsupportedEvaluation(
                f"{evaluator_input.operation_id}: the fused activation evaluator "
                "requires assignment on systolic arrays"
            )
        settings = context.evaluator_settings or {}
        unknown = sorted(set(settings) - set(_SETTINGS))
        if unknown:
            raise UnsupportedEvaluation(
                "unsupported fused activation evaluator setting(s): "
                + ", ".join(unknown)
            )
        sas = [context.system.core_dict[sa_id] for sa_id in assignment.endpoint_ids]
        elements_per_sa = math.ceil(evaluator_input.element_count / len(sas))
        group_latency_ns = max(
            math.ceil(elements_per_sa / sa.width) / sa.frequency * 1e9 for sa in sas
        )
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=self._setting(settings, "latency_ns")
            + self._setting(settings, "cycles_per_group") * group_latency_ns,
            dynamic_energy_pj=evaluator_input.element_count
            * self._setting(settings, "energy_per_element_pj"),
        )


def register_fused_activation(registry=None):
    """Register the movement policy, and the evaluators when given a registry.

    The movement policy registration is idempotent.
    """
    MOVEMENT_POLICIES.setdefault(
        FusedSaActivationService.policy_id, FusedSaActivationService
    )
    if registry is not None:
        for operation_type in FUSABLE_OPERATIONS:
            registry.register(
                SaFusedReluEvaluator(operation_type),
                SaFusedReluInputAdapter(operation_type),
            )
