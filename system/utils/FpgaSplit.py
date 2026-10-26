"""Split of one FPGA operation over several FPGA chiplets.

A systolic-array GEMM can run on a set of SAs: the scheduler cuts the GEMM into
tiles and gives each SA its tiles. An FPGA operation (ReLU, Softmax, LayerNorm)
can run on a set of FPGA chiplets in the same way. The cut is by rows:

* Row ``i`` of the tensor depends on row ``i`` of the input only. The mean and
  the variance of LayerNorm and the sum of Softmax run over the features of one
  row. So the chiplets need no data from each other, and nothing is reduced.
* The rows divide in contiguous blocks that differ by at most one row. The
  chiplets sort by endpoint ID; the first ones get the extra row.
* Each chiplet runs the kernel on its own rows, at the same time as the others.
  The latency of the operation is the latency of the slowest chiplet. The
  dynamic energy is the sum.
* Each chiplet needs all the constants of the kernel (scale, bias, tables), so
  every chiplet stores and loads its own copy. The BRAM, the resource check, and
  the input buffer are for each chiplet, with the rows of that chiplet. Each
  chiplet reports its own share (``chiplet_use``) for the per-chiplet check.

The tensor movement service uses the same row blocks to route the rows of a
GEMM output to the chiplet that owns them, and the rows back to the SAs.
"""

from __future__ import annotations

from dataclasses import replace

from system.utils.OperationEvaluator import OperationEstimate
from system.utils.ParameterLoad import combine_parameter_loads
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


def fpga_row_ranges(operation_id, rows, endpoint_ids):
    """Map each FPGA endpoint to its ``(first row, end row)`` block."""
    endpoint_ids = tuple(sorted(endpoint_ids))
    if not endpoint_ids:
        raise UnsupportedEvaluation(f"{operation_id}: no FPGA endpoint")
    if len(endpoint_ids) > rows:
        raise UnsupportedEvaluation(
            f"{operation_id}: {len(endpoint_ids)} FPGA chiplets cannot share "
            f"{rows} row(s); each chiplet needs at least one row"
        )
    base, extra = divmod(rows, len(endpoint_ids))
    ranges = {}
    start = 0
    for position, endpoint_id in enumerate(endpoint_ids):
        count = base + (1 if position < extra else 0)
        ranges[endpoint_id] = (start, start + count)
        start += count
    return ranges


def estimate_row_split(estimate_one, operation, access_plan, assignment, context):
    """Estimate one FPGA operation that is split by rows over its chiplets.

    ``estimate_one(operation, access_plan, assignment, context)`` estimates an
    operation on a single chiplet. It checks the resources of that chiplet.
    """
    from system.utils.OperationAssignment import EndpointAssignment

    ranges = fpga_row_ranges(
        operation.operation_id, operation.output_tensor.shape[0], assignment.endpoint_ids
    )
    estimates = []
    for endpoint_id, (first, end) in ranges.items():
        estimates.append(
            estimate_one(
                operation.with_rows(end - first),
                access_plan,
                EndpointAssignment((endpoint_id,), "fpga"),
                context,
            )
        )
    loads = [item.parameter_load for item in estimates if item.parameter_load]
    first_estimate = estimates[0]
    return replace(
        first_estimate,
        compute_latency_ns=max(item.compute_latency_ns for item in estimates),
        dynamic_energy_pj=sum(item.dynamic_energy_pj for item in estimates),
        parameter_load=(
            combine_parameter_loads(loads, context.prefetch_slack_ns)
            if loads
            else None
        ),
        chiplet_use=tuple(use for item in estimates for use in item.chiplet_use),
    )
