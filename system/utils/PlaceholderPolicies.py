"""Placeholder annealing policies, evaluators, and registration.

DELETE-ME FILE. Everything here exists to exercise and check modular annealing:

* ``cfg/profiles/atlas_modular_v1.json`` is the honest reference profile.
* ``placeholder_pessimistic_transfer_v0`` multiplies transfer cost.
* ``placeholder_always_dram_spill_v0`` charges a DRAM spill penalty.
* ``placeholder_slow_gemm_10x_v0`` multiplies GEMM latency.

These are deliberately bad candidates, not hardware characterizations. They let
a check prove the annealer scores a known bad policy badly and does not select
it. To remove them: delete this file, ``cfg/profiles/placeholders/``, the import
and single ``register_placeholder_policies()`` call in ``main.py``, and the
placeholder-dependent tests.
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
    MovementRecord,
    TensorMovementService,
)
from system.utils.TransferEstimator import (
    TRANSFER_COST_MODELS,
    TransferEstimate,
    TransferEstimator,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


class PessimisticTransferEstimator(TransferEstimator):
    """Bad transfer model: inflate every route cost by a fixed factor."""

    model_id = "placeholder_pessimistic_transfer_v0"
    latency_factor = 10.0
    energy_factor = 10.0

    def estimate(self, request):
        base = super().estimate(request)
        return TransferEstimate(
            tensor_id=base.tensor_id,
            byte_count=base.byte_count,
            latency_ns=base.latency_ns * self.latency_factor,
            energy_pj=base.energy_pj * self.energy_factor,
            path=base.path,
        )


class AlwaysDramSpillMovement(TensorMovementService):
    """Bad movement policy: charge a DRAM spill penalty on every boundary."""

    policy_id = "placeholder_always_dram_spill_v0"
    latency_penalty_ns_per_element = 1.0
    energy_penalty_pj_per_element = 1.0

    def move(
        self,
        tensor_id,
        element_count,
        source_endpoint,
        destination_endpoint,
        system,
        transfer_model=None,
    ):
        record = super().move(
            tensor_id=tensor_id,
            element_count=element_count,
            source_endpoint=source_endpoint,
            destination_endpoint=destination_endpoint,
            system=system,
            transfer_model=transfer_model,
        )
        return MovementRecord(
            tensor_id=record.tensor_id,
            byte_count=record.byte_count,
            source_endpoint=record.source_endpoint,
            destination_endpoint=record.destination_endpoint,
            method="placeholder_dram_spill",
            latency_ns=(
                record.latency_ns
                + element_count * self.latency_penalty_ns_per_element
            ),
            energy_pj=(
                record.energy_pj
                + element_count * self.energy_penalty_pj_per_element
            ),
        )


@dataclass(frozen=True)
class PlaceholderSlowGemmInput:
    operation_id: str
    m: int
    k: int
    n: int
    activation_from_dram: bool
    output_to_dram: bool


class PlaceholderSlowGemmInputAdapter(OperationInputAdapter):
    input_adapter_id = "placeholder_slow_gemm_input_v0"
    operation_type = "gemm"

    def build_input(self, operation, access_plan):
        if operation.operation_type != "gemm":
            raise UnsupportedEvaluation(
                f"PlaceholderSlowGemmInputAdapter cannot read "
                f"'{operation.operation_type}' operation '{operation.operation_id}'"
            )
        if operation.gemm_shape is None:
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: GEMM dimensions are missing"
            )
        m, k, n = operation.gemm_shape
        return PlaceholderSlowGemmInput(
            operation_id=operation.operation_id,
            m=m,
            k=k,
            n=n,
            activation_from_dram=access_plan.activation_from_dram,
            output_to_dram=access_plan.output_to_dram,
        )


class SlowGemmEvaluator(OperationEvaluator):
    """Bad GEMM evaluator: wrap the legacy GEMM path and multiply latency."""

    evaluator_id = "placeholder_slow_gemm_10x_v0"
    operation_type = "gemm"
    input_adapter_id = "placeholder_slow_gemm_input_v0"
    latency_factor = 10.0

    def evaluate(self, evaluator_input, placement, context) -> OperationEstimate:
        from main import simulate_single_gemm

        result = simulate_single_gemm(
            context.cache,
            context.architecture,
            (evaluator_input.m, evaluator_input.k, evaluator_input.n),
            activation_from_dram=evaluator_input.activation_from_dram,
            output_to_dram=evaluator_input.output_to_dram,
        )
        dynamic_energy_pj = (
            result["dram_interconnect_energy_pj"] + result["sram_energy_pj"]
        )
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=result["latency_ns"] * self.latency_factor,
            dynamic_energy_pj=dynamic_energy_pj,
        )


def register_placeholder_policies(registry=None):
    """Register placeholder policies and (optionally) evaluators. Idempotent."""
    MOVEMENT_POLICIES.setdefault(
        AlwaysDramSpillMovement.policy_id, AlwaysDramSpillMovement
    )
    TRANSFER_COST_MODELS.setdefault(
        PessimisticTransferEstimator.model_id, PessimisticTransferEstimator
    )
    if registry is not None and SlowGemmEvaluator.evaluator_id not in registry:
        registry.register(SlowGemmEvaluator(), PlaceholderSlowGemmInputAdapter())
