"""Non-GEMM compute estimators, input adapters, and operation evaluators.

The FPGA ReLU compute model is the only real non-GEMM model in this scope. Two
placeholder evaluators demonstrate the extension seam:

* ``placeholder_hls_relu_v0`` reads a real ATLAS attribute (``reuse_factor``)
  through its input adapter.
* ``placeholder_fpga_softmax_v0`` reads the confirmed ATLAS Softmax facts.

Both placeholders are extension demonstrators, not hardware characterizations.

Responsibilities:

* :class:`ReluComputeEstimator` estimates only FPGA ReLU execution.
* :class:`TransferEstimator` (separate module) estimates data movement.
* :class:`ReluStageComposer` combines input transfer, compute, and output
  transfer into one serialized stage estimate (legacy helper).
* The evaluators return only compute latency and dynamic energy.

The estimators never see chiplet topology, routes, bandwidths, or SCALE-Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from system.utils.FpgaChiplet import FpgaChiplet
from system.utils.OperationEvaluator import (
    OperationEstimate,
    OperationEvaluator,
    OperationInputAdapter,
)
from system.utils.TransferEstimator import (
    ResolvedRoute,
    TransferEstimator,
    TransferRequest,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


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

    def estimate(self, element_count, fpga, operation_id="relu"):
        raise NotImplementedError  # pragma: no cover - interface


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

    def estimate(self, element_count, fpga: FpgaChiplet, operation_id="relu"):
        if not isinstance(fpga, FpgaChiplet):
            raise ValueError("ReluComputeEstimator requires an FpgaChiplet")

        if fpga.frequency_hz <= 0:
            return ReluComputeEstimate(
                operation_id=operation_id,
                element_count=element_count,
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
                operation_id=operation_id,
                element_count=element_count,
                parallel_lanes=0,
                compute_cycles=0,
                compute_latency_ns=0.0,
                compute_energy_pj=None,
                feasible=False,
                infeasibility_reason="resource-derived parallel lane count is zero",
            )

        compute_cycles = math.ceil(element_count / parallel_lanes)
        compute_latency_ns = compute_cycles / fpga.frequency_hz * 1e9

        energy_per_element = fpga.relu_implementation.energy_per_element_pj
        compute_energy_pj = (
            None
            if energy_per_element is None
            else element_count * energy_per_element
        )

        return ReluComputeEstimate(
            operation_id=operation_id,
            element_count=element_count,
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
        compute = compute_estimator.estimate(
            operation.element_count, fpga, operation.operation_id
        )

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


@dataclass(frozen=True)
class LegacyReluInput:
    operation_id: str
    element_count: int


class LegacyReluInputAdapter(OperationInputAdapter):
    """Input adapter for the legacy resource-derived FPGA ReLU model."""

    input_adapter_id = "legacy_fpga_relu_input_v1"
    operation_type = "relu"

    def build_input(self, operation, access_plan):
        if operation.operation_type != "relu":
            raise UnsupportedEvaluation(
                f"LegacyReluInputAdapter cannot read "
                f"'{operation.operation_type}' operation "
                f"'{operation.operation_id}'"
            )
        return LegacyReluInput(
            operation_id=operation.operation_id,
            element_count=operation.element_count,
        )


class FpgaReluEvaluator(OperationEvaluator):
    """Operation evaluator for a ReLU placed on an FPGA endpoint.

    Wraps the resource-derived ReLU compute estimator. It returns only compute
    latency and dynamic energy; tensor movement is owned by the tensor movement
    service and the executor owns placement and residency.
    """

    evaluator_id = "legacy_fpga_relu_v1"
    operation_type = "relu"
    input_adapter_id = "legacy_fpga_relu_input_v1"

    def __init__(self, compute_estimator=None):
        self.compute_estimator = compute_estimator or ReluComputeEstimator()

    def evaluate(self, evaluator_input, placement, context) -> OperationEstimate:
        fpga = context.system.fpga_chiplet_dict.get(placement.endpoint_id)
        if fpga is None:
            raise UnsupportedEvaluation(
                f"no FPGA endpoint {placement.endpoint_id} for "
                f"'{evaluator_input.operation_id}'"
            )
        estimate = self.compute_estimator.estimate(
            evaluator_input.element_count, fpga, evaluator_input.operation_id
        )
        if not estimate.feasible:
            raise UnsupportedEvaluation(
                f"{evaluator_input.operation_id}: {estimate.infeasibility_reason}"
            )
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=estimate.compute_latency_ns,
            dynamic_energy_pj=estimate.compute_energy_pj or 0.0,
        )


@dataclass(frozen=True)
class PlaceholderHlsReluInput:
    operation_id: str
    element_count: int
    reuse_factor: int


class PlaceholderHlsReluInputAdapter(OperationInputAdapter):
    """Reads a real ATLAS ReLU attribute to prove the source-input seam."""

    input_adapter_id = "placeholder_hls_relu_input_v0"
    operation_type = "relu"

    def build_input(self, operation, access_plan):
        if operation.operation_type != "relu":
            raise UnsupportedEvaluation(
                f"PlaceholderHlsReluInputAdapter cannot read "
                f"'{operation.operation_type}' operation "
                f"'{operation.operation_id}'"
            )
        reuse_factor = operation.source.attribute("reuse_factor")
        if (
            isinstance(reuse_factor, bool)
            or not isinstance(reuse_factor, int)
            or reuse_factor <= 0
        ):
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: ATLAS attribute 'reuse_factor' "
                f"must be a positive integer"
            )
        return PlaceholderHlsReluInput(
            operation_id=operation.operation_id,
            element_count=operation.element_count,
            reuse_factor=reuse_factor,
        )


class PlaceholderHlsReluEvaluator(OperationEvaluator):
    """Placeholder HLS-aware ReLU model; not a hardware characterization.

    Demonstrates that a model can require an additional ATLAS fact without any
    change to the graph adapter or the executor.
    """

    evaluator_id = "placeholder_hls_relu_v0"
    operation_type = "relu"
    input_adapter_id = "placeholder_hls_relu_input_v0"

    def __init__(self, compute_estimator=None):
        self.compute_estimator = compute_estimator or ReluComputeEstimator()

    def evaluate(self, evaluator_input, placement, context) -> OperationEstimate:
        fpga = context.system.fpga_chiplet_dict.get(placement.endpoint_id)
        if fpga is None:
            raise UnsupportedEvaluation(
                f"no FPGA endpoint {placement.endpoint_id} for "
                f"'{evaluator_input.operation_id}'"
            )
        estimate = self.compute_estimator.estimate(
            evaluator_input.element_count, fpga, evaluator_input.operation_id
        )
        if not estimate.feasible:
            raise UnsupportedEvaluation(
                f"{evaluator_input.operation_id}: {estimate.infeasibility_reason}"
            )
        compute_cycles = estimate.compute_cycles * evaluator_input.reuse_factor
        compute_latency_ns = compute_cycles / fpga.frequency_hz * 1e9
        dynamic_energy_pj = evaluator_input.element_count * 1.0
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=compute_latency_ns,
            dynamic_energy_pj=dynamic_energy_pj,
        )


@dataclass(frozen=True)
class PlaceholderSoftmaxInput:
    operation_id: str
    element_count: int
    axis: int | None


class PlaceholderSoftmaxInputAdapter(OperationInputAdapter):
    """Input adapter for the placeholder Softmax model."""

    input_adapter_id = "placeholder_fpga_softmax_input_v0"
    operation_type = "softmax"

    def build_input(self, operation, access_plan):
        if operation.operation_type != "softmax":
            raise UnsupportedEvaluation(
                f"PlaceholderSoftmaxInputAdapter cannot read "
                f"'{operation.operation_type}' operation "
                f"'{operation.operation_id}'"
            )
        axis = operation.source.attribute("axis", None)
        if axis is not None and (
            isinstance(axis, bool) or not isinstance(axis, int)
        ):
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: ATLAS attribute 'axis' must be an "
                f"integer when present"
            )
        return PlaceholderSoftmaxInput(
            operation_id=operation.operation_id,
            element_count=operation.element_count,
            axis=axis,
        )


class PlaceholderSoftmaxEvaluator(OperationEvaluator):
    """Placeholder Softmax model; not a hardware characterization."""

    evaluator_id = "placeholder_fpga_softmax_v0"
    operation_type = "softmax"
    input_adapter_id = "placeholder_fpga_softmax_input_v0"

    def evaluate(self, evaluator_input, placement, context) -> OperationEstimate:
        fpga = context.system.fpga_chiplet_dict.get(placement.endpoint_id)
        if fpga is None:
            raise UnsupportedEvaluation(
                f"no FPGA endpoint {placement.endpoint_id} for "
                f"'{evaluator_input.operation_id}'"
            )
        if fpga.frequency_hz <= 0:
            raise UnsupportedEvaluation(
                f"{evaluator_input.operation_id}: fpga frequency must be positive"
            )
        compute_cycles = evaluator_input.element_count
        compute_latency_ns = compute_cycles / fpga.frequency_hz * 1e9
        dynamic_energy_pj = evaluator_input.element_count * 2.0
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=compute_latency_ns,
            dynamic_energy_pj=dynamic_energy_pj,
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
        return estimator.estimate(
            operation.element_count, fpga, operation.operation_id
        )

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
