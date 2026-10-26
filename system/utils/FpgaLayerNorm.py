"""LayerNorm on an FPGA endpoint.

rule4ml has no predictor for LayerNormalization, so this evaluator derives the
latency from the loop structure of the hls4ml kernel (``nnet_layernorm.h``)
instead of predicting it.

The kernel normalizes ``seq_len`` tokens of ``dim`` features each. The loop over
tokens is pipelined with an initiation interval of ``reuse_factor``, and every
loop inside one token is unrolled. One token is therefore a chain of dependent
operations:

* a sum over ``dim`` values: an adder tree of ``ceil(log2(dim))`` levels,
* the mean (multiply by 1/dim), the difference to the mean, and its square: 3,
* a second sum over ``dim`` values: ``ceil(log2(dim))`` levels,
* the variance (multiply by 1/dim), the table address (multiply, add), the
  clamp of the address, and the table read: 5,
* the result ``diff * inverse_std * scale + bias``: 3.

The model counts one cycle for each dependent operation. This is a first-order
schedule. It is not validated: no synthesis report of the kernel is available,
and HLS may chain narrow operations into one cycle or split a wide one.

Cycles: ``(seq_len - 1) * reuse_factor + depth``, with
``depth = 2 * ceil(log2(dim)) + 11``.

Dynamic energy: ``elements x fpga.layernorm_implementation.energy_per_element_pj``.

Input buffer. Like every ``io_parallel`` kernel, LayerNorm reads its whole input
at once, so the FPGA holds it in its input buffer (``rule4ml_input_buffer``).
The read delay of the buffer adds to the latency, and its BRAM blocks count
against the FPGA.

Parameters. The scale, the bias, and the rsqrt table are constants of the
kernel. Each value is 8 bits. They are stored in BRAM, wired to this kernel:
``ceil(bits / 18,432)`` blocks count against the FPGA. With
``fpga.parameter_source == "dram"`` (not the default) they are loaded over the route
DRAM -> systolic array -> FPGA before the kernel starts (``ParameterLoad``).
The previous operation hides the part of that load that fits in its compute
time. With ``"bitstream"`` (the default) they are part of the configuration and cost nothing
per inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from system.utils.ParameterLoad import plan_parameter_load
from system.utils.OperationEvaluator import (
    OperationEstimate,
    OperationEvaluator,
    OperationInputAdapter,
)
from system.utils.Rule4MlEvaluators import check_resources_fit
from system.utils.Rule4MlInputBuffer import (
    VALUE_BITS,
    input_buffer,
    minimum_blocks,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation

#: Table entries when the ATLAS node does not give a table size (hls4ml default).
_DEFAULT_TABLE_SIZE = 1024

#: Dependent operations of one token that do not depend on ``dim``.
_FIXED_DEPTH = 3 + 5 + 3


def layernorm_depth_cycles(dim):
    """Cycles for one token to pass through the kernel."""
    levels = math.ceil(math.log2(dim)) if dim > 1 else 0
    return 2 * levels + _FIXED_DEPTH


def layernorm_cycles(seq_len, dim, reuse_factor):
    """Cycles to normalize ``seq_len`` tokens of ``dim`` features."""
    return (seq_len - 1) * reuse_factor + layernorm_depth_cycles(dim)


@dataclass(frozen=True)
class LayerNormInput:
    operation_id: str
    element_count: int
    seq_len: int
    dim: int
    reuse_factor: int
    parameter_count: int


class LayerNormInputAdapter(OperationInputAdapter):
    """Reads the shape and reuse factor of a LayerNorm operation."""

    input_adapter_id = "fpga_layernorm_input_v1"
    operation_type = "layernorm"

    def build_input(self, operation, access_plan):
        if operation.operation_type != "layernorm":
            raise UnsupportedEvaluation(
                f"LayerNormInputAdapter cannot read "
                f"'{operation.operation_type}' operation "
                f"'{operation.operation_id}'"
            )
        shape = tuple(operation.output_tensor.shape)
        if len(shape) == 1:
            seq_len, dim = 1, shape[0]
        elif len(shape) == 2:
            seq_len, dim = shape
        else:
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: layernorm tensors must be rank 1 or 2"
            )
        axis = operation.source.attribute("axis", None)
        if axis is not None and axis not in (-1, len(shape) - 1, len(shape)):
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: layernorm must normalize the last axis"
            )
        reuse_factor = operation.source.attribute("reuse_factor", 1)
        if (
            isinstance(reuse_factor, bool)
            or not isinstance(reuse_factor, int)
            or reuse_factor <= 0
        ):
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: ATLAS attribute 'reuse_factor' "
                f"must be a positive integer"
            )
        table_size = operation.source.attribute("table_size", _DEFAULT_TABLE_SIZE)
        if isinstance(table_size, bool) or not isinstance(table_size, int) or table_size <= 0:
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: ATLAS attribute 'table_size' "
                f"must be a positive integer"
            )
        weight_count = len(operation.source.weights)
        # A LayerNorm has a scale and a bias of one value per feature.
        scale_and_bias = (
            sum(operation.source.weight(index).element_count for index in range(weight_count))
            if weight_count
            else 2 * dim
        )
        return LayerNormInput(
            operation_id=operation.operation_id,
            element_count=operation.element_count,
            seq_len=seq_len,
            dim=dim,
            reuse_factor=reuse_factor,
            parameter_count=scale_and_bias + table_size,
        )


class LayerNormEvaluator(OperationEvaluator):
    """Structural LayerNorm model of the hls4ml kernel."""

    evaluator_id = "fpga_layernorm_v1"
    operation_type = "layernorm"
    input_adapter_id = "fpga_layernorm_input_v1"

    def evaluate(self, evaluator_input, assignment, context) -> OperationEstimate:
        fpga = context.system.fpga_chiplet_dict.get(assignment.endpoint_id)
        if fpga is None:
            raise UnsupportedEvaluation(
                f"no FPGA endpoint {assignment.endpoint_id} for "
                f"'{evaluator_input.operation_id}'"
            )
        if fpga.frequency_hz <= 0:
            raise UnsupportedEvaluation(
                f"{evaluator_input.operation_id}: fpga frequency must be positive"
            )
        buffer = input_buffer(
            evaluator_input.element_count, VALUE_BITS, fpga.input_buffer_blocks
        )
        parameter_bytes = evaluator_input.parameter_count * VALUE_BITS // 8
        parameter_blocks = minimum_blocks(evaluator_input.parameter_count * VALUE_BITS)
        check_resources_fit(
            evaluator_input.operation_id,
            fpga,
            {"BRAM": buffer.blocks + parameter_blocks},
        )

        parameter_load = None
        if fpga.parameter_source == "dram":
            parameter_load = plan_parameter_load(
                context.system,
                context.architecture,
                assignment.endpoint_id,
                parameter_bytes,
                evaluator_input.operation_id,
                slack_ns=context.prefetch_slack_ns,
            )
        cycles = layernorm_cycles(
            evaluator_input.seq_len,
            evaluator_input.dim,
            evaluator_input.reuse_factor,
        ) + buffer.read_cycles
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=cycles / fpga.frequency_hz * 1e9,
            dynamic_energy_pj=evaluator_input.element_count
            * fpga.layernorm_implementation.energy_per_element_pj,
            parameter_load=parameter_load,
        )


def register_layernorm_evaluators(registry):
    """Register the LayerNorm evaluator into ``registry``."""
    if registry is None:
        raise ValueError("register_layernorm_evaluators requires an evaluator registry")
    registry.register(LayerNormEvaluator(), LayerNormInputAdapter())
    return registry
