"""Input buffer of an io_parallel rule4ml kernel.

An io_parallel hls4ml kernel reads every input value in one clock cycle, and
rule4ml predicts the kernel only. The registers or RAM that hold the input
while it arrives are outside the synthesized kernel, so rule4ml's resources do
not include them. This module charges them.

The buffer is held in block RAM. A buffer of ``blocks`` blocks reads
``blocks x BRAM_BLOCK_READ_BITS`` bits in one cycle, so the kernel can start
only after ``read_cycles`` cycles. More blocks start the kernel sooner and use
more of the BRAM of the FPGA.

The block geometry is that of one rule4ml BRAM unit. The board catalog counts
BRAM_18K units (zcu102: 1824 = 912 BRAM36 x 2), so a unit holds 18 Kbit and
reads 36 bits per cycle (simple dual-port mode).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from system.utils.UnsupportedEvaluation import (
    ResourceExceeded,
    UnsupportedEvaluation,
)

#: Every value in the model is 8-bit.
VALUE_BITS = 8
BRAM_BLOCK_BITS = 18 * 1024
BRAM_BLOCK_READ_BITS = 36

_AP_FIXED = re.compile(r"^\s*ap_fixed\s*<\s*(\d+)\s*,")


@dataclass(frozen=True)
class InputBuffer:
    bits: int
    blocks: int
    read_cycles: int


def precision_bits(precision):
    """Return the bit width W of an ``ap_fixed<W, I>`` precision string."""
    match = _AP_FIXED.match(precision) if isinstance(precision, str) else None
    if match is None:
        raise UnsupportedEvaluation(
            f"cannot read a bit width from precision '{precision}'; "
            "expected ap_fixed<W, I>"
        )
    return int(match.group(1))


def minimum_blocks(bits):
    """Fewest BRAM blocks that hold ``bits`` bits."""
    return max(1, math.ceil(bits / BRAM_BLOCK_BITS))


def input_buffer(element_count, bits_per_value, blocks=None):
    """Size the input buffer of one operation.

    ``blocks`` is the number of BRAM blocks of the buffer, all read in parallel.
    ``None`` selects
    the fewest blocks that hold the input, which is the cheapest buffer and the
    slowest to read.
    """
    if element_count <= 0 or bits_per_value <= 0:
        raise UnsupportedEvaluation(
            "the input buffer needs a positive element count and bit width"
        )
    bits = element_count * bits_per_value
    least = minimum_blocks(bits)
    if blocks is None:
        blocks = least
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
        raise UnsupportedEvaluation("input_buffer_blocks must be a positive integer")
    if blocks < least:
        raise ResourceExceeded(
            f"the input buffer of {blocks} block(s) cannot hold {bits} bits; "
            f"it needs at least {least}"
        )
    read_cycles = math.ceil(bits / (blocks * BRAM_BLOCK_READ_BITS))
    return InputBuffer(bits=bits, blocks=blocks, read_cycles=read_cycles)
