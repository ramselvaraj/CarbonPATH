"""Load of the constant parameters of an FPGA kernel from DRAM.

An FPGA chiplet has no memory channel of its own. The package attaches DRAM to
systolic-array chiplets, so the parameters of a kernel (for example the scale,
bias, and table of LayerNorm) travel DRAM -> systolic array -> link(s) -> FPGA.

The cost follows the existing transfer model. The DRAM read is the first stage
of the route, and its reciprocal bandwidth adds to those of the links, as the
reciprocal weights of the links add to each other in the route search. The
route setup and hop latencies come from the architecture's ``transfer_model``.

The load can start before the kernel does. The executor passes the time that the
FPGA is idle before the operation (``prefetch_slack_ns``): the compute time of
the previous operation when it runs on another endpoint. That time hides the
load. The exposed latency is ``max(0, raw latency - slack)``. The energy is paid
in full.
"""

from __future__ import annotations

from dataclasses import dataclass

from system.utils.TransferEstimator import (
    ResolvedRoute,
    TransferEstimator,
    TransferRequest,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation

#: Where the parameters of an FPGA kernel come from.
PARAMETER_SOURCES = ("dram", "bitstream")


@dataclass(frozen=True)
class ParameterLoad:
    """One parameter load: the raw cost, and the part that the kernel waits for."""

    byte_count: int
    source_endpoint: int
    destination_endpoint: int
    raw_latency_ns: float
    exposed_latency_ns: float
    energy_pj: float

    @property
    def hidden_latency_ns(self):
        return self.raw_latency_ns - self.exposed_latency_ns


def combine_parameter_loads(loads, slack_ns=0.0, shared_dram=True):
    """One load that stands for the loads of several FPGA chiplets.

    The chiplets of a split operation each load their own copy of the
    constants. With ``shared_dram`` the loads share the DRAM, so they run one
    after the other: the raw latencies add. Without it each load has its own
    path and they overlap: the longest load counts. The idle time before the
    operation hides the result once. The bytes and the energy add.
    """
    loads = tuple(loads)
    latencies = [load.raw_latency_ns for load in loads]
    raw_latency_ns = sum(latencies) if shared_dram else max(latencies)
    sources = {load.source_endpoint for load in loads}
    return ParameterLoad(
        byte_count=sum(load.byte_count for load in loads),
        source_endpoint=sources.pop() if len(sources) == 1 else tuple(sorted(sources)),
        destination_endpoint=tuple(load.destination_endpoint for load in loads),
        raw_latency_ns=raw_latency_ns,
        exposed_latency_ns=max(0.0, raw_latency_ns - max(0.0, slack_ns)),
        energy_pj=sum(load.energy_pj for load in loads),
    )


def _memory_endpoints(system, architecture):
    """The systolic arrays that have at least one memory channel."""
    channels = architecture.get("pkg", {}).get("mem_pkg_conn", {})
    return [
        core
        for core in system.core_dict.values()
        if channels.get(f"Chiplet_{core.id + 1}", 0) > 0
    ]


def plan_parameter_load(
    system,
    architecture,
    fpga_endpoint,
    byte_count,
    operation_id,
    slack_ns=0.0,
    transfer_estimator=None,
):
    """Plan the DRAM load of ``byte_count`` bytes into one FPGA endpoint.

    The source is the memory-attached systolic array with the cheapest route to
    the FPGA.
    """
    if byte_count <= 0:
        raise UnsupportedEvaluation(
            f"{operation_id}: a parameter load needs a positive byte count"
        )
    transfer_model = architecture.get("transfer_model", {}) or {}
    estimator = transfer_estimator or TransferEstimator()

    best = None
    for core in _memory_endpoints(system, architecture):
        if core.dram_bandwidth <= 0:
            continue
        path, path_reciprocal, path_energy = system.get_shortest_path(
            core.id, fpga_endpoint
        )
        if not path:
            continue
        route = ResolvedRoute(
            source_chiplet_id=core.id,
            destination_chiplet_id=fpga_endpoint,
            path=tuple(path),
            path_bandwidth_reciprocal_ns_per_byte=(
                1.0 / core.dram_bandwidth + path_reciprocal
            ),
            path_energy_pj_per_bit=core.dram_energy_scale + path_energy,
            setup_latency_ns=transfer_model.get("setup_latency_ns", 0.0),
            hop_latency_ns=transfer_model.get("hop_latency_ns", 0.0),
        )
        estimate = estimator.estimate(
            TransferRequest(
                tensor_id=f"{operation_id}_parameters",
                element_count=byte_count,
                route=route,
            )
        )
        if best is None or estimate.latency_ns < best[1].latency_ns:
            best = (core, estimate)

    if best is None:
        raise UnsupportedEvaluation(
            f"{operation_id}: no memory-attached systolic array has a route to "
            f"FPGA endpoint {fpga_endpoint}"
        )
    core, estimate = best
    return ParameterLoad(
        byte_count=estimate.byte_count,
        source_endpoint=core.id,
        destination_endpoint=fpga_endpoint,
        raw_latency_ns=estimate.latency_ns,
        exposed_latency_ns=max(0.0, estimate.latency_ns - max(0.0, slack_ns)),
        energy_pj=estimate.energy_pj,
    )
