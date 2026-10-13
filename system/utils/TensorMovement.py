"""Tensor movement service for ATLAS graph evaluation.

The service answers one question: a tensor was produced here and the next
operation needs it there, so what happens in between? It records tensor
residency, resolves routes, and produces movement plans. It does not estimate
compute and does not know which operation produced the tensor.

Every movement policy exposes the same call shape used by the executor::

    movement_policy.plan(request) -> MovementPlan

``MovementRequest`` carries the facts needed to plan one adjacent graph handoff.
The executor never inspects GEMM tiles or branches on GEMM/ReLU; that stays
behind the movement seam.

Three real adapters live here:

* ``activation_boundary_v1``: the original single-endpoint behavior.
* the mapped GEMM intermediate-memory adapters (``cold_dram_v1``,
  ``local_sram_v1``, ``direct_forward_v1``).
* ``distributed_activation_boundary_v1``: SA-group/FPGA behavior that owns
  region coverage, replication, route resolution, and link-load aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class MovementRequest:
    """Facts needed to plan one adjacent graph handoff."""

    handoff_index: int
    tensor_id: str
    tensor_shape: tuple
    producer_operation: object
    consumer_operation: object
    producer_assignment: object
    consumer_assignment: object
    producer_prepared: object
    consumer_prepared: object
    system: object
    transfer_model: dict


@dataclass(frozen=True)
class RoutedTensorSlice:
    """One physical transfer of a rectangular region between two endpoints."""

    tensor_id: str
    row_offset: int
    row_count: int
    feature_offset: int
    feature_count: int
    source_endpoint: int
    destination_endpoint: int
    byte_count: int
    route: tuple
    setup_latency_ns: float
    hop_latency_ns: float
    serialization_latency_ns: float
    slice_latency_ns: float
    slice_energy_pj: float


@dataclass(frozen=True)
class DistributedMovementPlan:
    """Aggregate movement values used by the executor and reporting.

    ``charged_latency_ns`` is the critical (maximum) slice completion estimate;
    ``charged_energy_pj`` is the sum of every physical slice's energy. Logical
    bytes describe the tensor once; physical bytes include replication and may
    therefore exceed logical bytes.
    """

    method: str
    source_endpoint: object
    destination_endpoint: object
    logical_bytes: int
    physical_bytes: int
    charged_latency_ns: float
    charged_energy_pj: float
    transfer_slices: tuple = ()
    uses_dram: bool = False

    @property
    def latency_ns(self):
        return self.charged_latency_ns

    @property
    def energy_pj(self):
        return self.charged_energy_pj

    @property
    def byte_count(self):
        return self.physical_bytes


def _product(shape):
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def _endpoint_value(endpoint_ids):
    endpoint_ids = tuple(
        (endpoint_ids,) if isinstance(endpoint_ids, int) else endpoint_ids
    )
    return endpoint_ids[0] if len(endpoint_ids) == 1 else endpoint_ids


def _single_endpoint(assignment):
    endpoint_ids = tuple(assignment.endpoint_ids)
    if len(endpoint_ids) != 1:
        raise UnsupportedEvaluation(
            "distributed movement requires a single FPGA endpoint"
        )
    return endpoint_ids[0]


def _route_links(route):
    return [(route[index], route[index + 1]) for index in range(len(route) - 1)]


def _link_key(left, right):
    return (left, right) if left <= right else (right, left)


class TensorMovementService:
    """Plans and charges one movement between a producer and consumer endpoint."""

    policy_id = "activation_boundary_v1"
    requires_prepared_operations = False

    def __init__(self, transfer_estimator=None):
        self.transfer_estimator = transfer_estimator or TransferEstimator()

    def plan(self, request) -> DistributedMovementPlan:
        element_count = _product(request.tensor_shape)
        record = self.move(
            tensor_id=request.tensor_id,
            element_count=element_count,
            source_endpoint=request.producer_assignment.endpoint_ids,
            destination_endpoint=request.consumer_assignment.endpoint_ids,
            system=request.system,
            transfer_model=request.transfer_model,
        )
        slices = ()
        physical_bytes = 0
        if record.method != "retain":
            physical_bytes = record.byte_count
            slices = (
                self._single_slice(request, record, element_count),
            )
        return DistributedMovementPlan(
            method=record.method,
            source_endpoint=record.source_endpoint,
            destination_endpoint=record.destination_endpoint,
            logical_bytes=element_count,
            physical_bytes=physical_bytes,
            charged_latency_ns=record.charged_latency_ns,
            charged_energy_pj=record.charged_energy_pj,
            transfer_slices=slices,
        )

    @staticmethod
    def _single_slice(request, record, element_count):
        setup_latency_ns = float(request.transfer_model.get("setup_latency_ns", 0.0))
        hop_latency_ns = float(request.transfer_model.get("hop_latency_ns", 0.0))
        path, _, _ = request.system.get_shortest_path(
            record.source_endpoint, record.destination_endpoint
        )
        path = tuple(path or ())
        hop_count = max(0, len(path) - 1)
        serialization_latency_ns = max(
            0.0,
            record.latency_ns - setup_latency_ns - hop_count * hop_latency_ns,
        )
        return RoutedTensorSlice(
            tensor_id=record.tensor_id,
            row_offset=0,
            row_count=request.tensor_shape[0] if len(request.tensor_shape) > 1 else 1,
            feature_offset=0,
            feature_count=(
                request.tensor_shape[1] if len(request.tensor_shape) > 1 else request.tensor_shape[0]
            ),
            source_endpoint=record.source_endpoint,
            destination_endpoint=record.destination_endpoint,
            byte_count=element_count,
            route=path,
            setup_latency_ns=setup_latency_ns,
            hop_latency_ns=hop_latency_ns,
            serialization_latency_ns=serialization_latency_ns,
            slice_latency_ns=record.latency_ns,
            slice_energy_pj=record.energy_pj,
        )

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

    def plan(self, request):
        return self.move_between(
            boundary_index=request.handoff_index,
            producer_operation=request.producer_operation,
            consumer_operation=request.consumer_operation,
            producer_prepared=request.producer_prepared,
            consumer_prepared=request.consumer_prepared,
        )

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


class DistributedActivationMovementService:
    """SA-group/FPGA movement for alternating GEMM/ReLU ATLAS graphs.

    GEMM output ownership is derived from the prepared scheduler's tiles. The
    same input region needed by two different SAs is replicated; regions
    repeated by several output columns on one SA are counted once.
    """

    policy_id = "distributed_activation_boundary_v1"
    requires_prepared_operations = True

    _SUPPORTED_PAIRS = {
        ("gemm", "relu"): "_plan_gemm_to_fpga",
        ("relu", "gemm"): "_plan_fpga_to_gemm",
    }

    def __init__(self, transfer_estimator=None):
        self.transfer_estimator = transfer_estimator or TransferEstimator()

    def plan(self, request) -> DistributedMovementPlan:
        producer_type = request.producer_operation.operation_type
        consumer_type = request.consumer_operation.operation_type

        if (
            producer_type == "relu"
            and consumer_type == "relu"
            and request.producer_assignment.endpoint_ids
            == request.consumer_assignment.endpoint_ids
        ):
            return self._retained_plan(request)

        handler_name = self._SUPPORTED_PAIRS.get((producer_type, consumer_type))
        if handler_name is None:
            raise UnsupportedEvaluation(
                "movement policy 'distributed_activation_boundary_v1' supports "
                f"gemm->relu, relu->gemm, and same-FPGA handoffs, not "
                f"{producer_type}->{consumer_type}"
            )
        return getattr(self, handler_name)(request)

    def _retained_plan(self, request):
        return DistributedMovementPlan(
            method="retain",
            source_endpoint=_endpoint_value(request.producer_assignment.endpoint_ids),
            destination_endpoint=_endpoint_value(request.consumer_assignment.endpoint_ids),
            logical_bytes=_product(request.tensor_shape),
            physical_bytes=0,
            charged_latency_ns=0.0,
            charged_energy_pj=0.0,
            transfer_slices=(),
        )

    def _plan_gemm_to_fpga(self, request):
        if request.producer_prepared is None:
            raise UnsupportedEvaluation(
                "distributed movement requires the prepared GEMM mapping"
            )
        scheduler, _ = request.producer_prepared
        fpga = _single_endpoint(request.consumer_assignment)
        rows, features = request.tensor_shape
        ownership = self._output_ownership(scheduler)
        coverage = sum((m1 - m0) * (n1 - n0) for (m0, m1, n0, n1) in ownership)
        if coverage != rows * features:
            raise UnsupportedEvaluation(
                "GEMM output regions do not cover the activation tensor exactly once"
            )
        drafts = [
            self._draft_slice(request, owner, fpga, m0, m1, n0, n1)
            for (m0, m1, n0, n1), owner in sorted(ownership.items())
        ]
        return self._aggregate(
            request,
            drafts,
            logical_bytes=rows * features,
            source=_endpoint_value(request.producer_assignment.endpoint_ids),
            destination=fpga,
        )

    def _plan_fpga_to_gemm(self, request):
        if request.consumer_prepared is None:
            raise UnsupportedEvaluation(
                "distributed movement requires the prepared GEMM mapping"
            )
        scheduler, _ = request.consumer_prepared
        fpga = _single_endpoint(request.producer_assignment)
        rows, features = request.tensor_shape
        requirements = {}
        for core in scheduler.systolic_arrays:
            for workload in core.workloads:
                key = (
                    core.id,
                    workload.m_offset,
                    workload.m_offset + workload.m,
                    workload.k_offset,
                    workload.k_offset + workload.k,
                )
                requirements.setdefault(key, None)
        drafts = []
        for sa_id, m0, m1, k0, k1 in sorted(requirements):
            if m1 > rows or k1 > features:
                raise UnsupportedEvaluation(
                    "consumer GEMM input region exceeds the producer tensor"
                )
            drafts.append(self._draft_slice(request, fpga, sa_id, m0, m1, k0, k1))
        return self._aggregate(
            request,
            drafts,
            logical_bytes=rows * features,
            source=fpga,
            destination=_endpoint_value(request.consumer_assignment.endpoint_ids),
        )

    @staticmethod
    def _output_ownership(scheduler):
        reduction_owner = (
            max(scheduler.systolic_arrays).id if scheduler.systolic_arrays else None
        )
        regions = {}
        for core in scheduler.systolic_arrays:
            for workload in core.workloads:
                region = (
                    workload.m_offset,
                    workload.m_offset + workload.m,
                    workload.n_offset,
                    workload.n_offset + workload.n,
                )
                owner = reduction_owner if scheduler.splitting_k else core.id
                previous_owner = regions.setdefault(region, owner)
                if previous_owner != owner:
                    raise UnsupportedEvaluation(
                        "an output region has multiple owners"
                    )
        return regions

    def _draft_slice(self, request, source, destination, row0, row1, feature0, feature1):
        byte_count = (row1 - row0) * (feature1 - feature0)
        setup_latency_ns = float(request.transfer_model.get("setup_latency_ns", 0.0))
        hop_latency_ns = float(request.transfer_model.get("hop_latency_ns", 0.0))
        if source == destination:
            return RoutedTensorSlice(
                tensor_id=request.tensor_id,
                row_offset=row0,
                row_count=row1 - row0,
                feature_offset=feature0,
                feature_count=feature1 - feature0,
                source_endpoint=source,
                destination_endpoint=destination,
                byte_count=byte_count,
                route=(),
                setup_latency_ns=0.0,
                hop_latency_ns=0.0,
                serialization_latency_ns=0.0,
                slice_latency_ns=0.0,
                slice_energy_pj=0.0,
            )
        path, _, _ = request.system.get_shortest_path(source, destination)
        if not path:
            raise UnsupportedEvaluation(
                f"no tensor movement route from endpoint {source} to endpoint "
                f"{destination}"
            )
        return RoutedTensorSlice(
            tensor_id=request.tensor_id,
            row_offset=row0,
            row_count=row1 - row0,
            feature_offset=feature0,
            feature_count=feature1 - feature0,
            source_endpoint=source,
            destination_endpoint=destination,
            byte_count=byte_count,
            route=tuple(path),
            setup_latency_ns=setup_latency_ns,
            hop_latency_ns=hop_latency_ns,
            serialization_latency_ns=0.0,
            slice_latency_ns=0.0,
            slice_energy_pj=0.0,
        )

    @staticmethod
    def _aggregate(request, drafts, logical_bytes, source, destination):
        link_load_ns = {}
        for draft in drafts:
            for left, right in _route_links(draft.route):
                reciprocal, _ = request.system.link_cost(left, right)
                key = _link_key(left, right)
                link_load_ns[key] = (
                    link_load_ns.get(key, 0.0) + draft.byte_count * reciprocal
                )

        finalized = []
        charged_latency_ns = 0.0
        charged_energy_pj = 0.0
        physical_bytes = 0
        for draft in drafts:
            links = _route_links(draft.route)
            hop_count = len(links)
            serialization_latency_ns = sum(
                link_load_ns[_link_key(*link)] for link in links
            )
            slice_energy_pj = (
                draft.byte_count
                * 8
                * sum(request.system.link_cost(*link)[1] for link in links)
            )
            slice_latency_ns = (
                draft.setup_latency_ns
                + hop_count * draft.hop_latency_ns
                + serialization_latency_ns
            )
            finalized.append(
                replace(
                    draft,
                    serialization_latency_ns=serialization_latency_ns,
                    slice_latency_ns=slice_latency_ns,
                    slice_energy_pj=slice_energy_pj,
                )
            )
            charged_latency_ns = max(charged_latency_ns, slice_latency_ns)
            charged_energy_pj += slice_energy_pj
            physical_bytes += draft.byte_count

        return DistributedMovementPlan(
            method="distributed_route",
            source_endpoint=source,
            destination_endpoint=destination,
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
            charged_latency_ns=charged_latency_ns,
            charged_energy_pj=charged_energy_pj,
            transfer_slices=tuple(finalized),
        )


def _gemm_memory_policy(policy_id):
    return lambda transfer_estimator=None: GemmIntermediateMemoryService(
        policy_id, transfer_estimator
    )


MOVEMENT_POLICIES = {
    TensorMovementService.policy_id: TensorMovementService,
    DistributedActivationMovementService.policy_id: DistributedActivationMovementService,
    **{
        f"{policy}_v1": _gemm_memory_policy(f"{policy}_v1")
        for policy in ("cold_dram", "local_sram", "direct_forward")
    },
}


def build_movement_service(policy_id, transfer_estimator=None):
    """Resolve an evaluation profile's tensor movement policy ID."""
    try:
        policy_class = MOVEMENT_POLICIES[policy_id]
    except KeyError:
        raise UnsupportedEvaluation(f"unknown tensor movement policy '{policy_id}'")
    return policy_class(transfer_estimator)
