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


@dataclass(frozen=True)
class TensorResidency:
    tensor_id: str
    endpoint_id: int
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
        if source_endpoint == destination_endpoint:
            return MovementRecord(
                tensor_id=tensor_id,
                byte_count=element_count,
                source_endpoint=source_endpoint,
                destination_endpoint=destination_endpoint,
                method="retain",
                latency_ns=0.0,
                energy_pj=0.0,
            )

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


MOVEMENT_POLICIES = {
    TensorMovementService.policy_id: TensorMovementService,
}


def build_movement_service(policy_id, transfer_estimator=None):
    """Resolve an evaluation profile's tensor movement policy ID."""
    try:
        policy_class = MOVEMENT_POLICIES[policy_id]
    except KeyError:
        raise UnsupportedEvaluation(f"unknown tensor movement policy '{policy_id}'")
    return policy_class(transfer_estimator)
