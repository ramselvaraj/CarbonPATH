from __future__ import annotations

from dataclasses import dataclass
import math


INTERMEDIATE_POLICIES = (
    "cold_dram",
    "ideal_on_chip",
    "local_sram",
    "direct_forward",
)


@dataclass(frozen=True)
class CoreMemory:
    core_id: int
    capacity_bytes: int
    dram_bandwidth_bytes_per_ns: float
    dram_energy_pj_per_bit: float


@dataclass(frozen=True)
class TransferSlice:
    byte_count: int
    producer_core: int
    consumer_core: int
    route: tuple[int, ...] = ()
    path_bandwidth_reciprocal_ns_per_byte: float = 0
    path_energy_pj_per_bit: float = 0


@dataclass(frozen=True)
class BoundaryPlan:
    boundary_index: int
    requested_policy: str
    selected_method: str
    intermediate_bytes: int
    retained_bytes: int
    forwarded_bytes: int
    dram_spilled_bytes: int
    dram_traffic_bytes: int
    latency_ns: float
    energy_pj: float
    producer_cores: tuple[int, ...]
    consumer_cores: tuple[int, ...]
    routes: tuple[tuple[int, ...], ...]
    fallback_reason: str = ""

    @property
    def uses_dram(self):
        return self.dram_spilled_bytes > 0


@dataclass(frozen=True)
class BoundaryMapping:
    intermediate_bytes: int
    transfers: tuple[TransferSlice, ...]
    cores: dict[int, CoreMemory]
    valid: bool = True
    error: str = ""


def _cold_plan(boundary_index, intermediate_bytes, transfers, cores, policy, reason=""):
    writes_by_core = {}
    reads_by_core = {}
    energy_pj = 0
    for transfer in transfers:
        producer = cores[transfer.producer_core]
        consumer = cores[transfer.consumer_core]
        writes_by_core[producer.core_id] = (
            writes_by_core.get(producer.core_id, 0) + transfer.byte_count
        )
        reads_by_core[consumer.core_id] = (
            reads_by_core.get(consumer.core_id, 0) + transfer.byte_count
        )
        energy_pj += transfer.byte_count * 8 * (
            producer.dram_energy_pj_per_bit + consumer.dram_energy_pj_per_bit
        )

    write_latency = max(
        (
            byte_count / cores[core_id].dram_bandwidth_bytes_per_ns
            for core_id, byte_count in writes_by_core.items()
        ),
        default=0,
    )
    read_latency = max(
        (
            byte_count / cores[core_id].dram_bandwidth_bytes_per_ns
            for core_id, byte_count in reads_by_core.items()
        ),
        default=0,
    )
    return BoundaryPlan(
        boundary_index=boundary_index,
        requested_policy=policy,
        selected_method="cold_dram",
        intermediate_bytes=intermediate_bytes,
        retained_bytes=0,
        forwarded_bytes=0,
        dram_spilled_bytes=intermediate_bytes,
        dram_traffic_bytes=intermediate_bytes * 2,
        latency_ns=write_latency + read_latency,
        energy_pj=energy_pj,
        producer_cores=tuple(sorted(writes_by_core)),
        consumer_cores=tuple(sorted(reads_by_core)),
        routes=(),
        fallback_reason=reason,
    )


def _local_plan(boundary_index, intermediate_bytes, transfers, cores, policy):
    required_by_core = {}
    for transfer in transfers:
        if transfer.producer_core != transfer.consumer_core:
            return None, "producer and consumer mappings differ"
        required_by_core[transfer.consumer_core] = (
            required_by_core.get(transfer.consumer_core, 0) + transfer.byte_count
        )

    for core_id, required_bytes in required_by_core.items():
        if required_bytes > cores[core_id].capacity_bytes:
            return None, f"core {core_id} SRAM capacity is insufficient"

    return BoundaryPlan(
        boundary_index=boundary_index,
        requested_policy=policy,
        selected_method="local_sram",
        intermediate_bytes=intermediate_bytes,
        retained_bytes=intermediate_bytes,
        forwarded_bytes=0,
        dram_spilled_bytes=0,
        dram_traffic_bytes=0,
        latency_ns=0,
        energy_pj=0,
        producer_cores=tuple(sorted(required_by_core)),
        consumer_cores=tuple(sorted(required_by_core)),
        routes=(),
    ), ""


def _forward_plan(boundary_index, intermediate_bytes, transfers, cores, policy):
    required_by_core = {}
    forwarded_bytes = 0
    retained_bytes = 0
    latency_ns = 0
    energy_pj = 0
    routes = []

    for transfer in transfers:
        required_by_core[transfer.consumer_core] = (
            required_by_core.get(transfer.consumer_core, 0) + transfer.byte_count
        )
        if transfer.producer_core == transfer.consumer_core:
            retained_bytes += transfer.byte_count
            continue
        if (
            not transfer.route
            or transfer.route[0] != transfer.producer_core
            or transfer.route[-1] != transfer.consumer_core
            or not math.isfinite(transfer.path_bandwidth_reciprocal_ns_per_byte)
            or transfer.path_bandwidth_reciprocal_ns_per_byte <= 0
            or not math.isfinite(transfer.path_energy_pj_per_bit)
            or transfer.path_energy_pj_per_bit < 0
        ):
            return None, "no forwarding route is available"
        forwarded_bytes += transfer.byte_count
        latency_ns += (
            transfer.byte_count * transfer.path_bandwidth_reciprocal_ns_per_byte
        )
        energy_pj += (
            transfer.byte_count * 8 * transfer.path_energy_pj_per_bit
        )
        routes.append(transfer.route)

    for core_id, required_bytes in required_by_core.items():
        if required_bytes > cores[core_id].capacity_bytes:
            return None, f"core {core_id} SRAM capacity is insufficient"

    return BoundaryPlan(
        boundary_index=boundary_index,
        requested_policy=policy,
        selected_method="direct_forward" if forwarded_bytes else "local_sram",
        intermediate_bytes=intermediate_bytes,
        retained_bytes=retained_bytes,
        forwarded_bytes=forwarded_bytes,
        dram_spilled_bytes=0,
        dram_traffic_bytes=0,
        latency_ns=latency_ns,
        energy_pj=energy_pj,
        producer_cores=tuple(sorted({transfer.producer_core for transfer in transfers})),
        consumer_cores=tuple(sorted(required_by_core)),
        routes=tuple(dict.fromkeys(routes)),
    ), ""


def plan_boundary(
    boundary_index,
    intermediate_bytes,
    transfers,
    cores,
    policy,
    mapping_valid=True,
    mapping_error="",
):
    if policy not in INTERMEDIATE_POLICIES:
        raise ValueError(f"Unknown intermediate-memory policy: {policy}")
    if sum(transfer.byte_count for transfer in transfers) != intermediate_bytes:
        raise ValueError("Transfer slices must exactly cover the intermediate")

    cold = _cold_plan(
        boundary_index,
        intermediate_bytes,
        transfers,
        cores,
        policy,
    )
    if policy == "cold_dram":
        return cold
    if policy == "ideal_on_chip":
        return BoundaryPlan(
            boundary_index=boundary_index,
            requested_policy=policy,
            selected_method="ideal_on_chip",
            intermediate_bytes=intermediate_bytes,
            retained_bytes=intermediate_bytes,
            forwarded_bytes=0,
            dram_spilled_bytes=0,
            dram_traffic_bytes=0,
            latency_ns=0,
            energy_pj=0,
            producer_cores=tuple(sorted({t.producer_core for t in transfers})),
            consumer_cores=tuple(sorted({t.consumer_core for t in transfers})),
            routes=(),
        )

    if not mapping_valid:
        return BoundaryPlan(
            **{
                **cold.__dict__,
                "fallback_reason": mapping_error or "tile mapping is unsupported",
            }
        )

    local, local_reason = _local_plan(
        boundary_index, intermediate_bytes, transfers, cores, policy
    )
    if policy == "local_sram":
        if local is not None:
            return local
        return _cold_plan(
            boundary_index,
            intermediate_bytes,
            transfers,
            cores,
            policy,
            local_reason,
        )

    forwarded, forward_reason = _forward_plan(
        boundary_index, intermediate_bytes, transfers, cores, policy
    )
    if forwarded is not None:
        return forwarded
    return _cold_plan(
        boundary_index,
        intermediate_bytes,
        transfers,
        cores,
        policy,
        forward_reason,
    )


def _region_intersection(left, right):
    m_start = max(left[0], right[0])
    m_stop = min(left[1], right[1])
    inner_start = max(left[2], right[2])
    inner_stop = min(left[3], right[3])
    if m_start >= m_stop or inner_start >= inner_stop:
        return None
    return m_start, m_stop, inner_start, inner_stop


def build_boundary_mapping(producer_scheduler, producer_system, consumer_scheduler, consumer_system):
    intermediate_bytes = producer_scheduler.M * producer_scheduler.N
    cores = {}
    for system in (producer_system, consumer_system):
        for core in system.core_dict.values():
            bandwidth_bytes_per_ns = core.dram_bandwidth * core.frequency / 1e9
            cores[core.id] = CoreMemory(
                core_id=core.id,
                capacity_bytes=core.buffer_size * 1024,
                dram_bandwidth_bytes_per_ns=bandwidth_bytes_per_ns,
                dram_energy_pj_per_bit=core.dram_energy_scale,
            )

    output_regions = {}
    reduction_owner = max(producer_scheduler.systolic_arrays).id
    for core in producer_scheduler.systolic_arrays:
        for workload in core.workloads:
            region = (
                workload.m_offset,
                workload.m_offset + workload.m,
                workload.n_offset,
                workload.n_offset + workload.n,
            )
            owner = reduction_owner if producer_scheduler.splitting_k else core.id
            previous_owner = output_regions.setdefault(region, owner)
            if previous_owner != owner:
                return _unsupported_mapping(
                    intermediate_bytes,
                    cores,
                    "an output region has multiple owners",
                )

    input_regions = set()
    for core in consumer_scheduler.systolic_arrays:
        for workload in core.workloads:
            input_regions.add(
                (
                    workload.m_offset,
                    workload.m_offset + workload.m,
                    workload.k_offset,
                    workload.k_offset + workload.k,
                    core.id,
                )
            )

    transfers = []
    for input_region in sorted(input_regions):
        region = input_region[:4]
        consumer_core = input_region[4]
        covered_bytes = 0
        for output_region, producer_core in output_regions.items():
            intersection = _region_intersection(output_region, region)
            if intersection is None:
                continue
            byte_count = (
                intersection[1] - intersection[0]
            ) * (
                intersection[3] - intersection[2]
            )
            covered_bytes += byte_count
            route = ()
            reciprocal = 0
            energy = 0
            if producer_core != consumer_core:
                route_result = producer_system.get_shortest_path(
                    producer_core, consumer_core
                )
                if route_result is not None:
                    route_path, reciprocal, energy = route_result
                    route = tuple(route_path or ())
            transfers.append(
                TransferSlice(
                    byte_count=byte_count,
                    producer_core=producer_core,
                    consumer_core=consumer_core,
                    route=route,
                    path_bandwidth_reciprocal_ns_per_byte=reciprocal,
                    path_energy_pj_per_bit=energy,
                )
            )
        expected_bytes = (region[1] - region[0]) * (region[3] - region[2])
        if covered_bytes != expected_bytes:
            return _unsupported_mapping(
                intermediate_bytes,
                cores,
                "producer output does not cover a consumer input region",
            )

    transfer_bytes = sum(transfer.byte_count for transfer in transfers)
    if transfer_bytes != intermediate_bytes:
        return _unsupported_mapping(
            intermediate_bytes,
            cores,
            "consumer mapping duplicates or omits intermediate data",
        )
    return BoundaryMapping(
        intermediate_bytes=intermediate_bytes,
        transfers=tuple(transfers),
        cores=cores,
    )


def _unsupported_mapping(intermediate_bytes, cores, error):
    core_ids = sorted(cores)
    producer = core_ids[0]
    consumer = core_ids[-1]
    return BoundaryMapping(
        intermediate_bytes=intermediate_bytes,
        transfers=(
            TransferSlice(
                byte_count=intermediate_bytes,
                producer_core=producer,
                consumer_core=consumer,
            ),
        ),
        cores=cores,
        valid=False,
        error=error,
    )
