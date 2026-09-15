"""Non-GEMM compute estimators and the FPGA stage composer.

V0 supports a single non-GEMM operation: the element-wise ReLU activation.

Responsibilities:

* :class:`ReluComputeEstimator` estimates only FPGA ReLU execution.
* :class:`TransferEstimator` (separate module) estimates data movement.
* :class:`ReluStageComposer` combines input transfer, compute, and output
  transfer into one serialized stage estimate.
* :class:`NonGemmEstimator` is the public facade and operation dispatcher.

The estimator never sees chiplet topology, routes, bandwidths, or SCALE-Sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from system.utils.FpgaChiplet import FpgaChiplet
from system.utils.TransferEstimator import (
    ResolvedRoute,
    TransferEstimator,
    TransferRequest,
)


@dataclass(frozen=True)
class ReluComputeEstimate:
    operation_id: str
    element_count: int
    parallel_lanes: int
    compute_cycles: int
    compute_latency_ns: float
    compute_energy_pj: float | None
    feasible: bool
    infeasibility_reason: str | None = None


@dataclass(frozen=True)
class ReluStageEstimate:
    operation_id: str
    input_transfer: object
    compute: ReluComputeEstimate
    output_transfer: object
    total_latency_ns: float
    transfer_energy_pj: float
    compute_energy_pj: float | None


class OperationComputeEstimator:
    """Base contract for a single operation's FPGA compute estimate."""

    def estimate(self, operation, fpga):  # pragma: no cover - interface
        raise NotImplementedError


class ReluComputeEstimator(OperationComputeEstimator):
    """Derive ReLU parallelism from FPGA resources and estimate execution."""

    def _parallel_lanes(self, fpga: FpgaChiplet):
        implementation = fpga.relu_implementation
        limits = [fpga.clbs // implementation.clbs_per_lane]
        if implementation.brams_per_lane > 0:
            limits.append(fpga.brams // implementation.brams_per_lane)
        if implementation.dsps_per_lane > 0:
            limits.append(fpga.dsps // implementation.dsps_per_lane)
        lanes = min(limits)
        if implementation.max_parallel_lanes is not None:
            lanes = min(lanes, implementation.max_parallel_lanes)
        return lanes

    def estimate(self, operation, fpga: FpgaChiplet) -> ReluComputeEstimate:
        if operation.operation_type != "relu":
            raise ValueError(
                f"ReluComputeEstimator cannot estimate '{operation.operation_type}'"
            )
        if not isinstance(fpga, FpgaChiplet):
            raise ValueError("ReluComputeEstimator requires an FpgaChiplet")

        if fpga.frequency_hz <= 0:
            return ReluComputeEstimate(
                operation_id=operation.operation_id,
                element_count=operation.element_count,
                parallel_lanes=0,
                compute_cycles=0,
                compute_latency_ns=0.0,
                compute_energy_pj=None,
                feasible=False,
                infeasibility_reason="fpga frequency must be positive",
            )

        parallel_lanes = self._parallel_lanes(fpga)
        if parallel_lanes <= 0:
            return ReluComputeEstimate(
                operation_id=operation.operation_id,
                element_count=operation.element_count,
                parallel_lanes=0,
                compute_cycles=0,
                compute_latency_ns=0.0,
                compute_energy_pj=None,
                feasible=False,
                infeasibility_reason="resource-derived parallel lane count is zero",
            )

        compute_cycles = math.ceil(operation.element_count / parallel_lanes)
        compute_latency_ns = compute_cycles / fpga.frequency_hz * 1e9

        energy_per_element = fpga.relu_implementation.energy_per_element_pj
        compute_energy_pj = (
            None
            if energy_per_element is None
            else operation.element_count * energy_per_element
        )

        return ReluComputeEstimate(
            operation_id=operation.operation_id,
            element_count=operation.element_count,
            parallel_lanes=parallel_lanes,
            compute_cycles=compute_cycles,
            compute_latency_ns=compute_latency_ns,
            compute_energy_pj=compute_energy_pj,
            feasible=True,
        )


class ReluStageComposer:
    """Compose input transfer, ReLU compute, and output transfer."""

    def __init__(self, transfer_estimator=None):
        self.transfer_estimator = transfer_estimator or TransferEstimator()

    def compose(
        self,
        operation,
        fpga: FpgaChiplet,
        input_route: ResolvedRoute,
        output_route: ResolvedRoute,
        compute_estimator=None,
    ) -> ReluStageEstimate:
        compute_estimator = compute_estimator or ReluComputeEstimator()
        compute = compute_estimator.estimate(operation, fpga)

        input_transfer = self.transfer_estimator.estimate(
            TransferRequest(
                tensor_id=operation.input_tensor_id,
                element_count=operation.element_count,
                route=input_route,
            )
        )
        output_transfer = self.transfer_estimator.estimate(
            TransferRequest(
                tensor_id=operation.output_tensor_id,
                element_count=operation.element_count,
                route=output_route,
            )
        )

        transfer_energy_pj = input_transfer.energy_pj + output_transfer.energy_pj
        total_latency_ns = (
            input_transfer.latency_ns
            + compute.compute_latency_ns
            + output_transfer.latency_ns
        )

        return ReluStageEstimate(
            operation_id=operation.operation_id,
            input_transfer=input_transfer,
            compute=compute,
            output_transfer=output_transfer,
            total_latency_ns=total_latency_ns,
            transfer_energy_pj=transfer_energy_pj,
            compute_energy_pj=compute.compute_energy_pj,
        )


class NonGemmEstimator:
    """Public facade dispatching non-GEMM operations to their estimators."""

    def __init__(self, transfer_estimator=None):
        self._compute_estimators = {"relu": ReluComputeEstimator()}
        self.stage_composer = ReluStageComposer(transfer_estimator)

    def estimate_operation(self, operation, fpga: FpgaChiplet):
        estimator = self._compute_estimators.get(operation.operation_type)
        if estimator is None:
            raise ValueError(
                f"Unsupported non-GEMM operation: {operation.operation_type}"
            )
        return estimator.estimate(operation, fpga)

    def estimate_stage(
        self,
        operation,
        fpga: FpgaChiplet,
        input_route: ResolvedRoute,
        output_route: ResolvedRoute,
    ) -> ReluStageEstimate:
        if operation.operation_type not in self._compute_estimators:
            raise ValueError(
                f"Unsupported non-GEMM operation: {operation.operation_type}"
            )
        return self.stage_composer.compose(
            operation,
            fpga,
            input_route,
            output_route,
            compute_estimator=self._compute_estimators[operation.operation_type],
        )
