"""Resource check for each FPGA chiplet over all operations that run on it.

``check_resources_fit`` rejects one operation that needs more than its chiplet
offers. This module rejects a design point whose operations, taken together,
need more than one chiplet offers.

The rules follow from how an FPGA chiplet is built, with no partial
reconfiguration: the bitstream holds the kernels of all operations mapped to the
chiplet at the same time.

* The kernels add up. Each operation has its own kernel, and the bitstream wires
  every LUT, FF, DSP, and BRAM block to one kernel. Two operations of the same
  type and shape are two kernels.
* The input buffer counts once. Operations do not run at the same time, so they
  share the buffer. It must hold the largest input, so its size is the largest
  buffer of the operations. When the chiplet fixes the number of blocks, every
  operation uses that number.
* The total BRAM of the chiplet is the BRAM of the kernels plus the buffer.
* An operation split by rows puts one share on each chiplet. Each chiplet counts
  only its own share.

An operation that reports no use is not counted. The caller decides which
evaluators report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from system.utils.Rule4MlBoards import fpga_resource_vector
from system.utils.UnsupportedEvaluation import ResourceExceeded

RESOURCE_LABELS = ("BRAM", "DSP", "FF", "LUT")


@dataclass(frozen=True)
class ChipletUse:
    """What the kernel of one operation occupies on one FPGA chiplet."""

    endpoint_id: int
    # Predicted kernel resources by label (BRAM, DSP, FF, LUT).
    kernel: dict = field(default_factory=dict)
    # BRAM blocks of the input buffer this operation needs.
    buffer_blocks: int = 0


def chiplet_totals(uses):
    """Total resources of each chiplet: ``{endpoint_id: {label: count}}``.

    ``uses`` is a sequence of ``(operation_id, ChipletUse)``.
    """
    totals = {}
    buffers = {}
    for _, use in uses:
        total = totals.setdefault(use.endpoint_id, dict.fromkeys(RESOURCE_LABELS, 0.0))
        for label, count in use.kernel.items():
            total[label] = total.get(label, 0.0) + count
        buffers[use.endpoint_id] = max(
            buffers.get(use.endpoint_id, 0), use.buffer_blocks
        )
    for endpoint_id, blocks in buffers.items():
        totals[endpoint_id]["BRAM"] += blocks
    return totals


def check_chiplet_resources(system, uses):
    """Raise ``ResourceExceeded`` when a chiplet needs more than it offers."""
    uses = tuple(uses)
    totals = chiplet_totals(uses)
    for endpoint_id, total in sorted(totals.items()):
        fpga = system.fpga_chiplet_dict[endpoint_id]
        available = fpga_resource_vector(fpga)
        exceeded = [
            f"{label} {total[label]:.0f} > {available[label.lower()]}"
            for label in RESOURCE_LABELS
            if label.lower() in available and total[label] > available[label.lower()]
        ]
        if exceeded:
            operations = sorted(
                {
                    operation_id
                    for operation_id, use in uses
                    if use.endpoint_id == endpoint_id
                }
            )
            raise ResourceExceeded(
                f"FPGA endpoint {endpoint_id}: its operations "
                f"({', '.join(operations)}) need more than the chiplet offers "
                f"({', '.join(exceeded)})"
            )
    return totals
