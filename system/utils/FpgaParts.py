"""FPGA part table: what a chiplet that names ``fpga_part`` is made of.

The table is ``cfg/fpga_parts.json``. A part gives the resources, the process,
the DSP generation, and which estimators know it. One part table is the single
source of a chiplet's resources, so a search move cannot make them disagree.

* ``part_chiplet_fields(part)`` returns the chiplet fields that follow from a
  part, like ``Rule4MlBoards.board_chiplet_fields`` does for a board.
* ``fpga_summary(architecture)`` returns the readable block that the design
  point file holds for the FPGA chiplets.

Numbers the table does not have are ``None`` and are said to be absent. Area and
power then come from the LUT-ratio proxy of ``Rule4MlBoards`` and are labelled
``proxy``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from system.utils.Rule4MlBoards import (
    FFS_PER_LUT,
    LUTS_PER_CLB,
    REFERENCE_AREA_MM2,
    REFERENCE_LUTS,
    REFERENCE_POWER_W,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation

PARTS_PATH = Path(__file__).resolve().parents[2] / "cfg" / "fpga_parts.json"
#: A rule4ml BRAM unit is 18 Kb; a part's BRAM is counted in 36 Kb tiles.
BRAM18_PER_BRAM36 = 2
#: The technology nodes (nm) of the carbon model, in chiplet/carbon_model/tech_scaling.py.
CARBON_TECH_NODES = (7, 10, 14, 22, 28)


@lru_cache(maxsize=1)
def load_parts():
    with PARTS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def part_names():
    return tuple(load_parts()["parts"])


def fpga_part(name):
    """The table entry of one part."""
    parts = load_parts()["parts"]
    if name not in parts:
        raise UnsupportedEvaluation(
            f"unknown FPGA part '{name}'; the parts are {', '.join(parts)}"
        )
    return parts[name]


def default_utilization():
    return dict(load_parts()["default_utilization"])


def part_for_board(board):
    """The part of a rule4ml board, or None."""
    for name, entry in load_parts()["parts"].items():
        if entry.get("rule4ml_board") == board:
            return name
    return None


def carbon_tech_node(process_nm):
    """The carbon-model node nearest the real process: 16 nm is not modelled, so it is 14."""
    return min(CARBON_TECH_NODES, key=lambda node: (abs(node - process_nm), node))


def part_chiplet_fields(name):
    """The FPGA chiplet fields that follow from one part.

    Area and power are the LUT-ratio proxy unless the part gives a sourced
    ``die_area_mm2`` or ``static_power_w``.
    """
    part = fpga_part(name)
    scale = part["lut"] / REFERENCE_LUTS
    return {
        "fpga_part": name,
        "clbs": part["lut"] // LUTS_PER_CLB,
        "brams": part["bram36"] * BRAM18_PER_BRAM36,
        "dsps": part["dsp"],
        "luts": part["lut"],
        "ffs": part["ff"],
        "uram": part["uram"],
        "process_nm": part["process_nm"],
        "tech_node": str(carbon_tech_node(part["process_nm"])),
        "area": part["die_area_mm2"] or round(REFERENCE_AREA_MM2 * scale, 4),
        "power": part["static_power_w"] or round(REFERENCE_POWER_W * scale, 4),
        "max_utilization": default_utilization(),
    }


def neurigami_part_for(fpga):
    """The neurigami part that estimates ``fpga``, or raise naming what is missing."""
    name = getattr(fpga, "fpga_part", None)
    if name is None and getattr(fpga, "board", None):
        name = part_for_board(fpga.board)
    if name is None:
        raise UnsupportedEvaluation(
            f"FPGA chiplet {fpga.name} names no fpga_part (or a board with a known part), "
            "so neurigami cannot tell which part to estimate"
        )
    if not fpga_part(name)["neurigami"]:
        known = ", ".join(n for n, e in load_parts()["parts"].items() if e["neurigami"])
        raise UnsupportedEvaluation(
            f"neurigami has no calibration for part '{name}' (FPGA chiplet {fpga.name}); "
            f"it knows {known}"
        )
    return name


def fpga_summary(architecture):
    """One readable entry for each FPGA chiplet of an architecture dictionary.

    The summary holds no scores. It restates the chiplet's own fields, and says
    where the area and power came from.
    """
    summary = {}
    for key, chiplet in architecture.items():
        if not isinstance(chiplet, dict) or chiplet.get("chiplet_type") != "fpga":
            continue
        name = chiplet.get("fpga_part")
        part = fpga_part(name) if name in load_parts()["parts"] else None
        luts = chiplet.get("luts") or chiplet["clbs"] * LUTS_PER_CLB
        entry = {
            "fpga_part": name,
            "board": chiplet.get("board"),
            "luts": luts,
            "ffs": chiplet.get("ffs") or luts * FFS_PER_LUT,
            "bram_18k": chiplet["brams"],
            "dsps": chiplet["dsps"],
            "uram": chiplet.get("uram"),
            "frequency_mhz": chiplet["frequency_hz"] / 1e6,
            "process_nm": chiplet.get("process_nm") or (part and part["process_nm"]),
            "tech_node": chiplet.get("tech_node"),
            "max_utilization": chiplet.get("max_utilization"),
            "area_mm2": chiplet["area"],
            "power_w": chiplet["power"],
            "area_power_source": (
                "sourced" if part and part["die_area_mm2"] and part["static_power_w"] else "proxy"
            ),
            "max_clock_hz": part and part["max_clock_hz"],
            "clock_checked": bool(part and part["max_clock_hz"]),
        }
        if part is not None:
            entry["dsp_generation"] = part["dsp_generation"]
            entry["family"] = part["family"]
        entry["notes"] = _notes(entry, part)
        summary[key] = entry
    return summary


def _notes(entry, part):
    notes = []
    if part is not None and not part["max_clock_hz"]:
        notes.append("no sourced max clock for this part: the clock is a target, not checked")
    if entry["area_power_source"] == "proxy":
        notes.append("area and power are the LUT-ratio proxy, not sourced")
    process, node = entry["process_nm"], entry["tech_node"]
    if process and node and int(node) != int(process):
        notes.append(f"process is {process} nm; the carbon model uses its {node} nm node")
    return notes
