"""Policy for the split of one FPGA operation over several FPGA chiplets.

The evaluation profile names the policy in ``fpga_split_policy``. A policy owns
three decisions:

* ``partition``: which rows each chiplet runs.
* ``combine``: how the results of the chiplets make one result.
* ``combine_loads``: how the constant loads of the chiplets make one load.

``row_split_v1`` is the default. It cuts the tensor in contiguous row blocks
that differ by at most one row, takes the slowest chiplet as the latency and the
sum as the energy, and runs the constant loads one after the other over the
shared DRAM. ``row_split_parallel_loads_v1`` differs in one point: each chiplet
loads its constants over its own DRAM path, so the loads overlap and the
latency is the longest load.

The tensor movement service reads the same ``partition``, so the rows that a
chiplet runs are the rows that move to it.
"""

from __future__ import annotations

from system.utils.ParameterLoad import combine_parameter_loads
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation

DEFAULT_SPLIT_POLICY = "row_split_v1"


class RowSplitPolicy:
    policy_id = "row_split_v1"
    # True when the constant loads of the chiplets share one DRAM path.
    shared_dram = True

    def partition(self, operation_id, rows, endpoint_ids):
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

    def combine(self, estimates):
        """The ``(latency, energy)`` of chiplets that run at the same time."""
        return (
            max(item.compute_latency_ns for item in estimates),
            sum(item.dynamic_energy_pj for item in estimates),
        )

    def combine_loads(self, loads, slack_ns):
        return combine_parameter_loads(loads, slack_ns, shared_dram=self.shared_dram)


class RowSplitParallelLoadsPolicy(RowSplitPolicy):
    policy_id = "row_split_parallel_loads_v1"
    shared_dram = False


SPLIT_POLICIES = {
    policy.policy_id: policy
    for policy in (RowSplitPolicy, RowSplitParallelLoadsPolicy)
}


def build_split_policy(policy_id=None):
    """Resolve an evaluation profile's FPGA split policy ID."""
    policy_id = policy_id or DEFAULT_SPLIT_POLICY
    try:
        return SPLIT_POLICIES[policy_id]()
    except KeyError:
        raise UnsupportedEvaluation(f"unknown FPGA split policy '{policy_id}'")
