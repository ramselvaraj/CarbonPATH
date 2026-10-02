"""Tensor movement service for ATLAS graph evaluation.

The service answers one question: a tensor was produced here and the next
operation needs it there, so what happens in between? It records tensor
residency, resolves routes, and produces movement plans. It does not estimate
compute and does not know which operation produced the tensor.

First scope owns intermediate activation handoffs only. Weights and the graph's
external inputs and outputs remain the legacy GEMM wrapper's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from system.utils.TransferEstimator import (
    ResolvedRoute,
    TransferEstimator,
    TransferRequest,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.IntermediateMemoryPolicy import build_boundary_mapping, plan_boundary


@dataclass(frozen=True)
class TensorResidency:
    tensor_id: str
    endpoint_ids: tuple[int, ...]
    endpoint_kind: str
    element_count: int


@dataclass(frozen=True)
class MovementRecord:
    tensor_id: str
    byte_count: int
    source_endpoint: int
    destination_endpoint: int
    method: str
    latency_ns: float
    energy_pj: float

    @property
    def charged_latency_ns(self):
        return self.latency_ns

    @property
    def charged_energy_pj(self):
        return self.energy_pj


class TensorMovementService:
    """Plans and charges one movement between a producer and consumer endpoint."""

    policy_id = "activation_boundary_v1"

    def __init__(self, transfer_estimator=None):
        self.transfer_estimator = transfer_estimator or TransferEstimator()

    def move(
        self,
        tensor_id,
        element_count,
        source_endpoint,
        destination_endpoint,
        system,
        transfer_model=None,
    ) -> MovementRecord:
        transfer_model = transfer_model or {}
        source_endpoints = (
            (source_endpoint,) if isinstance(source_endpoint, int) else tuple(source_endpoint)
        )
        destination_endpoints = (
            (destination_endpoint,)
            if isinstance(destination_endpoint, int)
            else tuple(destination_endpoint)
        )
        if source_endpoints == destination_endpoints:
            return MovementRecord(
                tensor_id=tensor_id,
                byte_count=element_count,
                source_endpoint=(
                    source_endpoints[0] if len(source_endpoints) == 1 else source_endpoints
                ),
                destination_endpoint=(
                    destination_endpoints[0]
                    if len(destination_endpoints) == 1
                    else destination_endpoints
                ),
                method="retain",
                latency_ns=0.0,
                energy_pj=0.0,
            )

        if len(source_endpoints) != 1 or len(destination_endpoints) != 1:
            raise UnsupportedEvaluation(
                "movement between different endpoint groups requires an explicit "
                "distribution plan"
            )
        source_endpoint = source_endpoints[0]
        destination_endpoint = destination_endpoints[0]

        if source_endpoint not in system.interconnect_dict:
            raise UnsupportedEvaluation(
                f"endpoint {source_endpoint} has no resolved interconnect"
            )

        path, reciprocal, energy = system.get_shortest_path(
            source_endpoint, destination_endpoint
        )
        if not path:
            raise UnsupportedEvaluation(
                f"no tensor movement route from endpoint {source_endpoint} "
                f"to endpoint {destination_endpoint}"
            )

        route = ResolvedRoute(
            source_chiplet_id=source_endpoint,
            destination_chiplet_id=destination_endpoint,
            path=tuple(path),
            path_bandwidth_reciprocal_ns_per_byte=reciprocal,
            path_energy_pj_per_bit=energy,
            setup_latency_ns=transfer_model.get("setup_latency_ns", 0.0),
            hop_latency_ns=transfer_model.get("hop_latency_ns", 0.0),
        )
        estimate = self.transfer_estimator.estimate(
            TransferRequest(
                tensor_id=tensor_id,
                element_count=element_count,
                route=route,
            )
        )
        return MovementRecord(
            tensor_id=tensor_id,
            byte_count=estimate.byte_count,
            source_endpoint=source_endpoint,
            destination_endpoint=destination_endpoint,
            method="route",
            latency_ns=estimate.latency_ns,
            energy_pj=estimate.energy_pj,
        )


class GemmIntermediateMemoryService:
    """Apply one original intermediate-memory behavior between mapped GEMMs."""

    requires_prepared_operations = True

    def __init__(self, policy_id, transfer_estimator=None):
        self.policy_id = policy_id
        self.intermediate_policy = policy_id.removesuffix("_v1")

    def move_between(
        self,
        boundary_index,
        producer_operation,
        consumer_operation,
        producer_prepared,
        consumer_prepared,
    ):
        if producer_operation.operation_type != "gemm" or consumer_operation.operation_type != "gemm":
            raise UnsupportedEvaluation(
                f"movement policy '{self.policy_id}' requires consecutive GEMM operations"
            )
        if producer_prepared is None or consumer_prepared is None:
            raise UnsupportedEvaluation(
                f"movement policy '{self.policy_id}' requires prepared GEMM mappings"
            )
        producer_scheduler, producer_system = producer_prepared
        consumer_scheduler, consumer_system = consumer_prepared
        mapping = build_boundary_mapping(
            producer_scheduler,
            producer_system,
            consumer_scheduler,
            consumer_system,
        )
        return plan_boundary(
            boundary_index=boundary_index,
            intermediate_bytes=mapping.intermediate_bytes,
            transfers=mapping.transfers,
            cores=mapping.cores,
            policy=self.intermediate_policy,
            mapping_valid=mapping.valid,
            mapping_error=mapping.error,
        )


def _gemm_memory_policy(policy_id):
    return lambda transfer_estimator=None: GemmIntermediateMemoryService(
        policy_id, transfer_estimator
    )


MOVEMENT_POLICIES = {
    TensorMovementService.policy_id: TensorMovementService,
    **{
        f"{policy}_v1": _gemm_memory_policy(f"{policy}_v1")
        for policy in ("cold_dram", "ideal_on_chip", "local_sram", "direct_forward")
    },
}


def build_movement_service(policy_id, transfer_estimator=None):
    """Resolve an evaluation profile's tensor movement policy ID."""
    try:
        policy_class = MOVEMENT_POLICIES[policy_id]
    except KeyError:
        raise UnsupportedEvaluation(f"unknown tensor movement policy '{policy_id}'")
    return policy_class(transfer_estimator)
