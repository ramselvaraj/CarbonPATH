"""Architecture and profile moves for modular ATLAS annealing.

The modular placement policy requires exactly one systolic-array endpoint and
exactly one FPGA endpoint, so these moves never add or delete chiplets: they
mutate the values of the two fixed endpoints, the package selection, the
transfer model, or the evaluation profile. Every proposed architecture is
checked to still hold the one-SA/one-FPGA invariant.

This module is separate from ``chiplet/n_utils`` legacy mutations, which assume
SA-only chiplet sets and would break the modular placement policy.
"""

from __future__ import annotations

import copy
import random

from chiplet.n_utils import get_area_power, get_sram_area_energy
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


MOVE_TYPES = (
    "sa_array",
    "sa_tech_node",
    "sa_sram",
    "fpga_clbs",
    "fpga_brams",
    "fpga_dsps",
    "fpga_frequency",
    "fpga_relu",
    "mem_type",
    "protocol",
    "transfer_model",
    "profile",
)

_FPGA_FIELDS = {"fpga_clbs": "clbs", "fpga_brams": "brams", "fpga_dsps": "dsps"}


def _is_fpga_chiplet(chiplet):
    return str(chiplet.get("chiplet_type", "systolic_array")).lower() == "fpga"


def validate_atlas_architecture(architecture):
    """Raise unless the architecture has exactly one SA and one FPGA endpoint."""
    chiplets = {
        key: value
        for key, value in architecture.items()
        if key.startswith("Chiplet_")
    }
    sa = [key for key, value in chiplets.items() if not _is_fpga_chiplet(value)]
    fpga = [key for key, value in chiplets.items() if _is_fpga_chiplet(value)]
    if len(sa) != 1 or len(fpga) != 1:
        raise UnsupportedEvaluation(
            "modular annealing requires exactly one systolic-array and one "
            f"FPGA endpoint, found {len(sa)} SA and {len(fpga)} FPGA"
        )
    return sa[0], fpga[0]


def _refresh_sa_area_power(chiplet):
    tech_node = str(chiplet["tech_node"])
    array = chiplet["sys_array_size"]
    buffer_size = int(chiplet["sram_buf"])
    logic_area, power = get_area_power(array, tech_node)
    sram_area, _ = get_sram_area_energy(buffer_size, tech_node)
    chiplet["area"] = logic_area + sram_area
    chiplet["power"] = power


def _valid_sram_sizes(params, array):
    options = params.get("sram_buf_sizes", {}).get(array)
    if not options:
        raise UnsupportedEvaluation(
            f"search space has no SRAM buffer sizes for array '{array}'"
        )
    return options


def mutate_atlas_design_point(
    architecture, profile, params, candidate_profiles=(), rng=None
):
    """Return ``(architecture, profile, move_name)`` for one modular move.

    Mutates either the architecture or the profile. The two-endpoint invariant
    is validated before returning; an invalid proposal raises
    :class:`UnsupportedEvaluation` so the caller can record it as invalid.
    """
    rng = rng or random
    candidate_profiles = tuple(candidate_profiles)
    candidate_moves = list(MOVE_TYPES)
    if not candidate_profiles:
        candidate_moves.remove("profile")

    move = rng.choice(candidate_moves)
    moved = copy.deepcopy(architecture)
    new_profile = profile
    sa_key, fpga_key = validate_atlas_architecture(moved)
    sa = moved[sa_key]
    fpga = moved[fpga_key]

    if move == "sa_array":
        array = rng.choice(params["sys_array"])
        sa["sys_array_size"] = array
        sa["sram_buf"] = rng.choice(_valid_sram_sizes(params, array))
        _refresh_sa_area_power(sa)
    elif move == "sa_tech_node":
        sa["tech_node"] = rng.choice(params["tech_nodes"])
        _refresh_sa_area_power(sa)
    elif move == "sa_sram":
        sa["sram_buf"] = rng.choice(_valid_sram_sizes(params, sa["sys_array_size"]))
        _refresh_sa_area_power(sa)
    elif move in _FPGA_FIELDS:
        fpga[_FPGA_FIELDS[move]] = rng.choice(params["fpga"][_FPGA_FIELDS[move]])
    elif move == "fpga_frequency":
        fpga["frequency_hz"] = rng.choice(params["fpga"]["frequency_hz"])
    elif move == "fpga_relu":
        fpga["relu_implementation"] = copy.deepcopy(
            rng.choice(params["fpga"]["relu_implementation"])
        )
    elif move == "mem_type":
        moved["pkg"]["mem_pkg_conn"]["mem_type"] = rng.choice(
            params["pkg"]["mem_pkg_architecture"]
        )
    elif move == "protocol":
        moved["pkg"]["protocol_2.5d"] = rng.choice(params["pkg"]["protocol"])
    elif move == "transfer_model":
        moved["transfer_model"] = copy.deepcopy(
            rng.choice(params["transfer_model"])
        )
    elif move == "profile":
        alternatives = [
            candidate
            for candidate in candidate_profiles
            if candidate.fingerprint() != profile.fingerprint()
            or candidate.name != profile.name
        ]
        new_profile = rng.choice(alternatives or candidate_profiles)

    validate_atlas_architecture(moved)
    return moved, new_profile, move


def candidate_profile_paths(search_space):
    paths = search_space.get("profiles") or []
    if not isinstance(paths, list) or not paths:
        raise UnsupportedEvaluation(
            "atlas search space must name at least one evaluation profile"
        )
    return paths
