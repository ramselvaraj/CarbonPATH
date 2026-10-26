"""Split of one FPGA operation over several FPGA chiplets.

A systolic-array GEMM can run on a set of SAs: the scheduler cuts the GEMM into
tiles and gives each SA its tiles. An FPGA operation (ReLU, Softmax, LayerNorm)
can run on a set of FPGA chiplets in the same way. The cut is by rows:

* Row ``i`` of the tensor depends on row ``i`` of the input only. The mean and
  the variance of LayerNorm and the sum of Softmax run over the features of one
  row. So the chiplets need no data from each other, and nothing is reduced.
* The split policy (``FpgaSplitPolicy.py``) owns the three decisions below. The
  default ``row_split_v1`` is described here. The rows divide in contiguous blocks that differ by at most one row. The
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
from system.utils.FpgaSplitPolicy import build_split_policy


def fpga_row_ranges(operation_id, rows, endpoint_ids, split_policy=None):
    """Map each FPGA endpoint to its ``(first row, end row)`` block."""
    return (split_policy or build_split_policy()).partition(
        operation_id, rows, endpoint_ids
    )


def estimate_row_split(estimate_one, operation, access_plan, assignment, context):
    """Estimate one FPGA operation that is split over its chiplets.

    ``estimate_one(operation, access_plan, assignment, context)`` estimates an
    operation on a single chiplet. It checks the resources of that chiplet. The
    split policy of the context divides the rows and combines the results.
    """
    from system.utils.OperationAssignment import EndpointAssignment

    policy = context.split_policy or build_split_policy()
    ranges = fpga_row_ranges(
        operation.operation_id,
        operation.output_tensor.shape[0],
        assignment.endpoint_ids,
        policy,
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
    latency_ns, energy_pj = policy.combine(estimates)
    return replace(
        estimates[0],
        compute_latency_ns=latency_ns,
        dynamic_energy_pj=energy_pj,
        parameter_load=(
            policy.combine_loads(loads, context.prefetch_slack_ns) if loads else None
        ),
        chiplet_use=tuple(use for item in estimates for use in item.chiplet_use),
    )
