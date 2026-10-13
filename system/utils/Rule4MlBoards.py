"""rule4ml board catalog and FPGA-endpoint to board mapping.

rule4ml ships a fixed set of characterized FPGA boards, each described by a
resource capacity table (LUT, FF, BRAM, DSP). A CarbonPATH architecture instead
describes its FPGA endpoint with a different resource vocabulary (CLB, BRAM,
DSP). This module is the single seam that maps one CarbonPATH FPGA endpoint to
the nearest rule4ml board, so a rule4ml operation evaluator can reuse rule4ml's
trained predictors without asking the architecture to name a board.

The catalog is read from the installed rule4ml package when it is importable, so
it never drifts from the estimator. An embedded copy keeps the mapping usable in
tests and on machines where rule4ml is not installed.
"""

from __future__ import annotations

import math

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


LUTS_PER_CLB = 8
FFS_PER_LUT = 2

_RESOURCE_DIMENSIONS = ("lut", "ff", "bram", "dsp")

# Mirrors rule4ml/parsers/supported_boards.json. Used only when rule4ml cannot
# be imported; the installed package is always preferred.
_FALLBACK_BOARDS = {
    "pynq-z2": {
        "part": "xc7z020clg400-1",
        "max_bram": 280,
        "max_dsp": 220,
        "max_ff": 106400,
        "max_lut": 53200,
        "max_uram": 0,
    },
    "zcu102": {
        "part": "xczu9eg-ffvb1156-2-e",
        "max_bram": 1824,
        "max_dsp": 2520,
        "max_ff": 548160,
        "max_lut": 274080,
        "max_uram": 0,
    },
    "alveo-u200": {
        "part": "xcu200-fsgd2104-2-e",
        "max_bram": 4320,
        "max_dsp": 6840,
        "max_ff": 2364480,
        "max_lut": 1182240,
        "max_uram": 960,
    },
    "alveo-u250": {
        "part": "xcu250-figd2104-2L-e",
        "max_bram": 5376,
        "max_dsp": 12288,
        "max_ff": 3456000,
        "max_lut": 1728000,
        "max_uram": 1280,
    },
}


def _embedded_boards():
    return {name: dict(caps) for name, caps in _FALLBACK_BOARDS.items()}


def load_boards():
    """Return the rule4ml board catalog, preferring the installed package."""
    try:
        from rule4ml.parsers.data_parser import boards_data
    except Exception:
        return _embedded_boards()
    if not isinstance(boards_data, dict) or not boards_data:
        return _embedded_boards()
    return boards_data


def fpga_resource_vector(fpga):
    """Resource vector for one FpgaChiplet, filling LUT/FF when absent.

    ``clbs`` are expanded with :data:`LUTS_PER_CLB`, and LUTs with
    :data:`FFS_PER_LUT`, only when the architecture did not declare ``luts`` or
    ``ffs`` directly.
    """
    luts = getattr(fpga, "luts", None)
    if luts is None:
        luts = fpga.clbs * LUTS_PER_CLB
    ffs = getattr(fpga, "ffs", None)
    if ffs is None:
        ffs = luts * FFS_PER_LUT
    return {
        "lut": luts,
        "ff": ffs,
        "bram": fpga.brams,
        "dsp": fpga.dsps,
    }


def nearest_board(fpga, boards=None):
    """Return the rule4ml board whose capacity best matches ``fpga``.

    The distance is summed squared log-ratio over the resource dimensions the
    architecture declares. Log space makes the match scale-invariant and
    symmetric, so an endpoint is not dragged toward a larger board merely
    because its ratio to every capacity is below one.
    """
    boards = boards if boards is not None else load_boards()
    vector = fpga_resource_vector(fpga)

    best_name = None
    best_score = None
    for name, capacities in boards.items():
        score = 0.0
        matched = 0
        for dimension in _RESOURCE_DIMENSIONS:
            capacity = capacities.get(f"max_{dimension}")
            available = vector.get(dimension, 0)
            if not capacity or not available or available <= 0:
                continue
            score += math.log(available / capacity) ** 2
            matched += 1
        if matched == 0:
            continue
        if best_score is None or score < best_score:
            best_name = name
            best_score = score

    if best_name is None:
        raise UnsupportedEvaluation(
            "no rule4ml board is available to match this FPGA endpoint"
        )
    return best_name
