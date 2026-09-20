"""Reusable inter-chiplet transfer estimator.

The transfer estimator receives an already-resolved route and a tensor payload
and returns movement latency and energy. It does not choose routes, inspect
topology, or know which operation produced the payload.

All tensors are assumed int8, so one element is one byte.
"""

from __future__ import annotations

from dataclasses import dataclass

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


@dataclass(frozen=True)
class ResolvedRoute:
    source_chiplet_id: int
    destination_chiplet_id: int
    path: tuple[int, ...] | None = None
    path_bandwidth_reciprocal_ns_per_byte: float = 0.0
    path_energy_pj_per_bit: float = 0.0
    setup_latency_ns: float = 0.0
    hop_latency_ns: float = 0.0

    @property
    def hop_count(self):
        if not self.path:
            return 0
        return max(0, len(self.path) - 1)

    @property
    def is_local(self):
        return self.source_chiplet_id == self.destination_chiplet_id or not self.path


@dataclass(frozen=True)
class TransferRequest:
    tensor_id: str
    element_count: int
    route: ResolvedRoute


@dataclass(frozen=True)
class TransferEstimate:
    tensor_id: str
    byte_count: int
    latency_ns: float
    energy_pj: float
    path: tuple[int, ...] | None


def _validate_route(route):
    if not isinstance(route, ResolvedRoute):
        raise ValueError("transfer route must be a ResolvedRoute")
    for field in (
        "path_bandwidth_reciprocal_ns_per_byte",
        "path_energy_pj_per_bit",
        "setup_latency_ns",
        "hop_latency_ns",
    ):
        value = getattr(route, field)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"route.{field} must be a non-negative number")


class TransferEstimator:
    """Estimate the latency and energy of moving one int8 tensor over a route."""

    model_id = "route_transfer_v1"

    def estimate(self, request: TransferRequest) -> TransferEstimate:
        if not isinstance(request.tensor_id, str) or not request.tensor_id:
            raise ValueError("transfer request must have a non-empty tensor_id")
        if (
            isinstance(request.element_count, bool)
            or not isinstance(request.element_count, int)
            or request.element_count <= 0
        ):
            raise ValueError("transfer request element_count must be a positive integer")
        _validate_route(request.route)

        byte_count = request.element_count
        route = request.route

        if route.is_local:
            return TransferEstimate(
                tensor_id=request.tensor_id,
                byte_count=byte_count,
                latency_ns=0.0,
                energy_pj=0.0,
                path=route.path,
            )

        latency_ns = (
            route.setup_latency_ns
            + route.hop_count * route.hop_latency_ns
            + byte_count * route.path_bandwidth_reciprocal_ns_per_byte
        )
        energy_pj = byte_count * 8 * route.path_energy_pj_per_bit
        return TransferEstimate(
            tensor_id=request.tensor_id,
            byte_count=byte_count,
            latency_ns=latency_ns,
            energy_pj=energy_pj,
            path=route.path,
        )


TRANSFER_COST_MODELS = {TransferEstimator.model_id: TransferEstimator}


def build_transfer_cost_model(model_id):
    """Resolve an evaluation profile's transfer cost model ID."""
    try:
        model_class = TRANSFER_COST_MODELS[model_id]
    except KeyError:
        raise UnsupportedEvaluation(f"unknown transfer cost model '{model_id}'")
    return model_class()
