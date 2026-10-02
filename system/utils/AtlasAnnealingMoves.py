"""Architecture and profile moves for modular ATLAS annealing.

Value moves operate on any valid one-to-six-SA architecture. Chiplet-count and
package-topology moves are separate mutation families because they must rebuild
the package as one atomic proposal.

This module is separate from ``chiplet/n_utils`` top-level mutation control.
"""

from __future__ import annotations

import copy
import math
import random

from chiplet.n_utils import get_area_power, get_sram_area_energy
from system.utils.ArchitectureModel import (
    regenerate_memory_channels,
    rebuild_package_for_endpoints,
    supported_package_types,
    validate_modular_architecture,
)
from system.utils.ArchitectureIdentity import canonicalize_chiplet_labels
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


MOVE_TYPES = (
    "sa_count",
    "sa_array",
    "sa_tech_node",
    "sa_sram",
    "gemm_dataflow",
    "gemm_split_k",
    "gemm_assignment_order",
    "fpga_clbs",
    "fpga_brams",
    "fpga_dsps",
    "fpga_frequency",
    "fpga_relu",
    "mem_type",
    "interconnect",
    "protocol",
    "transfer_model",
    "profile",
)

_FPGA_FIELDS = {"fpga_clbs": "clbs", "fpga_brams": "brams", "fpga_dsps": "dsps"}


def sequential_gemm_search_space(input_parameters):
    """Build the modular search domain used for an original GEMM workload.

    The original annealer chose mapping versus hardware with equal probability.
    A mapping proposal then chose one of three fields; a hardware proposal chose
    one of seven fields.  Integer weights of seven and three preserve those
    probabilities in the generic modular proposer.
    """
    if not isinstance(input_parameters, dict):
        raise UnsupportedEvaluation("sequential GEMM search parameters must be an object")
    search_space = copy.deepcopy(input_parameters)
    maximum = search_space.get("max_chiplet")
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise UnsupportedEvaluation("sequential GEMM search requires max_chiplet")
    search_space["max_sa_chiplets"] = maximum
    search_space["single_sa_ddr_only"] = True
    search_space["preserve_package_type_on_interconnect"] = True
    search_space["regenerate_memory_on_sa_count"] = True
    mapping_moves = {
        "gemm_dataflow",
        "gemm_split_k",
        "gemm_assignment_order",
    }
    hardware_moves = {
        "sa_count",
        "sa_array",
        "sa_tech_node",
        "sa_sram",
        "mem_type",
        "interconnect",
        "protocol",
    }
    search_space["move_weights"] = {
        move: 7.0 if move in mapping_moves else 3.0 if move in hardware_moves else 0.0
        for move in MOVE_TYPES
    }
    return search_space


def _select_move(rng, candidate_moves, params):
    weights = params.get("move_weights")
    if weights is None:
        return rng.choice(candidate_moves)
    if not isinstance(weights, dict):
        raise UnsupportedEvaluation("move_weights must be an object")

    unknown = sorted(set(weights) - set(MOVE_TYPES))
    if unknown:
        raise UnsupportedEvaluation(
            "unknown move weight: " + ", ".join(unknown)
        )
    for move_name, value in weights.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise UnsupportedEvaluation(
                f"move weight for '{move_name}' must be a finite non-negative number"
            )

    enabled = []
    enabled_weights = []
    for move_name in candidate_moves:
        value = weights.get(move_name, 0.0)
        if value > 0:
            enabled.append(move_name)
            enabled_weights.append(float(value))
    if not enabled:
        raise UnsupportedEvaluation(
            "search space has no enabled move available for this design point"
        )
    return rng.choices(enabled, weights=enabled_weights, k=1)[0]


def validate_atlas_architecture(architecture):
    """Compatibility alias for callers not yet moved to the architecture layer."""
    return validate_modular_architecture(architecture)


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


def _choose_alternative(rng, options, current, label):
    alternatives = [option for option in options if option != current]
    if not alternatives:
        raise UnsupportedEvaluation(f"no alternative value is available for {label}")
    return rng.choice(alternatives)


def _mutate_sa_count(architecture, params, rng):
    moved = copy.deepcopy(architecture)
    endpoints = validate_atlas_architecture(moved)
    maximum = int(params.get("max_sa_chiplets", params.get("max_chiplet", 1)))
    can_add = len(endpoints.sa_keys) < min(maximum, 6)
    can_delete = len(endpoints.sa_keys) > 1
    operations = [
        operation
        for operation, allowed in (("add", can_add), ("delete", can_delete))
        if allowed
    ]
    if not operations:
        raise UnsupportedEvaluation("no SA-count move is available")
    operation = rng.choice(operations)
    if operation == "add":
        used_ids = [int(key.split("_", 1)[1]) for key in moved if key.startswith("Chiplet_")]
        key = f"Chiplet_{max(used_ids, default=0) + 1}"
        array = rng.choice(params["sys_array"])
        tech_node = rng.choice(params["tech_nodes"])
        sram = rng.choice(_valid_sram_sizes(params, array))
        moved[key] = {
            "tech_node": tech_node,
            "sys_array_size": array,
            "sram_buf": sram,
        }
        _refresh_sa_area_power(moved[key])
    else:
        del moved[rng.choice(endpoints.sa_keys)]

    preferred_package_type = None
    if params.get("regenerate_memory_on_sa_count"):
        sa_count = len(
            [
                key
                for key, chiplet in moved.items()
                if key.startswith("Chiplet_")
                and str(chiplet.get("chiplet_type", "systolic_array")).lower()
                != "fpga"
            ]
        )
        memory_options = params["pkg"]["mem_pkg_architecture"]
        if params.get("single_sa_ddr_only") and sa_count == 1:
            memory_options = [
                option for option in memory_options if "ddr" in option.lower()
            ]
        moved["pkg"]["mem_pkg_conn"]["mem_type"] = rng.choice(memory_options)
        package_types = supported_package_types(moved, params)
        if package_types:
            preferred_package_type = rng.choice(package_types)
    return canonicalize_chiplet_labels(
        rebuild_package_for_endpoints(
            moved,
            params,
            rng,
            preferred_package_type=preferred_package_type,
        )
    )


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
    if "max_sa_chiplets" not in params and "max_chiplet" not in params:
        candidate_moves.remove("sa_count")

    endpoints = validate_atlas_architecture(architecture)
    if not endpoints.fpga_keys:
        candidate_moves = [
            candidate for candidate in candidate_moves if not candidate.startswith("fpga_")
        ]

    move = _select_move(rng, candidate_moves, params)
    if move == "sa_count":
        moved = _mutate_sa_count(architecture, params, rng)
        return moved, profile, move
    moved = copy.deepcopy(architecture)
    new_profile = profile
    sa_key = rng.choice(endpoints.sa_keys)
    sa = moved[sa_key]

    if move == "sa_array":
        array = _choose_alternative(
            rng, params["sys_array"], sa["sys_array_size"], "SA array size"
        )
        sa["sys_array_size"] = array
        sa["sram_buf"] = rng.choice(_valid_sram_sizes(params, array))
        _refresh_sa_area_power(sa)
    elif move == "sa_tech_node":
        sa["tech_node"] = _choose_alternative(
            rng, params["tech_nodes"], sa["tech_node"], "SA technology node"
        )
        _refresh_sa_area_power(sa)
    elif move == "sa_sram":
        sa["sram_buf"] = _choose_alternative(
            rng,
            _valid_sram_sizes(params, sa["sys_array_size"]),
            sa["sram_buf"],
            "SA SRAM size",
        )
        _refresh_sa_area_power(sa)
    elif move == "gemm_dataflow":
        current = profile.evaluator_settings_for("gemm").get(
            "dataflow", moved["WL_mapping"]["mapping"]["dataflow"][0]
        )
        alternatives = [value for value in ("ws", "os", "is") if value != current]
        new_profile = profile.with_evaluator_setting(
            "gemm", "dataflow", rng.choice(alternatives)
        )
    elif move == "gemm_split_k":
        current = profile.evaluator_settings_for("gemm").get(
            "split_k", bool(moved["WL_mapping"]["mapping"]["if_splitting_k"])
        )
        new_profile = profile.with_evaluator_setting("gemm", "split_k", not current)
    elif move == "gemm_assignment_order":
        mapping = moved["WL_mapping"]["mapping"]
        current = profile.evaluator_settings_for("gemm").get(
            "assignment_order",
            "ascending"
            if mapping["assign_workload_in_ascending_order"]
            else "descending",
        )
        new_profile = profile.with_evaluator_setting(
            "gemm",
            "assignment_order",
            "descending" if current == "ascending" else "ascending",
        )
    elif move in _FPGA_FIELDS:
        fpga_key = rng.choice(endpoints.fpga_keys)
        fpga = moved[fpga_key]
        field = _FPGA_FIELDS[move]
        fpga[field] = _choose_alternative(
            rng, params["fpga"][field], fpga[field], f"FPGA {field}"
        )
    elif move == "fpga_frequency":
        fpga_key = rng.choice(endpoints.fpga_keys)
        fpga = moved[fpga_key]
        fpga["frequency_hz"] = _choose_alternative(
            rng,
            params["fpga"]["frequency_hz"],
            fpga["frequency_hz"],
            "FPGA frequency",
        )
    elif move == "fpga_relu":
        fpga_key = rng.choice(endpoints.fpga_keys)
        fpga = moved[fpga_key]
        fpga["relu_implementation"] = copy.deepcopy(
            _choose_alternative(
                rng,
                params["fpga"]["relu_implementation"],
                fpga["relu_implementation"],
                "FPGA ReLU implementation",
            )
        )
    elif move == "mem_type":
        memory = moved["pkg"]["mem_pkg_conn"]
        memory_options = params["pkg"]["mem_pkg_architecture"]
        if params.get("single_sa_ddr_only") and len(endpoints.sa_keys) == 1:
            memory_options = [
                option for option in memory_options if "ddr" in option.lower()
            ]
        memory_type = _choose_alternative(
            rng,
            memory_options,
            memory["mem_type"],
            "memory technology",
        )
        moved["pkg"]["mem_pkg_conn"] = regenerate_memory_channels(
            moved, memory_type
        )
    elif move == "interconnect":
        package_options = params.get("pkg", {}).get("inter_pkg_architecture", [])
        if endpoints.fpga_keys:
            package_options = [
                option for option in package_options if option.startswith("2.5d_")
            ]
        else:
            package_options = [option for option in package_options if option != "2d_na"]
        current_connections = moved["pkg"].get("inter_pkg_conn", [])
        current_types = {
            connection.get("connection_type")
            for connection in current_connections
            if isinstance(connection, dict)
        } if isinstance(current_connections, list) else set()
        current_package_type = moved["pkg"].get("HI_pkg_type")
        preserve_package_type = params.get(
            "preserve_package_type_on_interconnect", False
        )
        if preserve_package_type:
            if current_package_type == "2.5d":
                package_options = [
                    option for option in package_options if option.startswith("2.5d_")
                ]
            elif current_package_type == "3d":
                package_options = [
                    option for option in package_options if option.startswith("3d_")
                ]
            elif current_package_type not in ("2.5d_3d",):
                package_options = []
            topology_alternatives = []
        else:
            topology_alternatives = [
                value
                for value in supported_package_types(moved, params)
                if value != current_package_type
            ]
        if topology_alternatives:
            moved = rebuild_package_for_endpoints(
                moved,
                params,
                rng,
                preferred_package_type=rng.choice(topology_alternatives),
            )
        else:
            alternatives = [
                option for option in package_options if option not in current_types
            ]
            if not alternatives:
                raise UnsupportedEvaluation("no alternative interconnect is available")
            moved = rebuild_package_for_endpoints(
                moved,
                params,
                rng,
                preferred_connection_type=rng.choice(alternatives),
            )
    elif move == "protocol":
        package = moved["pkg"]
        connections = package.get("inter_pkg_conn", [])
        connection_types = {
            connection.get("connection_type")
            for connection in connections
            if isinstance(connection, dict)
        } if isinstance(connections, list) else set()
        protocol_options = params["pkg"]["protocol"]
        if any(value and value.startswith("3d_") for value in connection_types):
            compatible = [value for value in protocol_options if value == "ucie_3d"]
            field = "protocol_3d"
        elif "2.5d_rdl" in connection_types:
            compatible = [value for value in protocol_options if value == "ucie_std"]
            field = "protocol_2.5d"
        else:
            compatible = [
                value for value in protocol_options if value in ("ucie_adv", "aib", "bow")
            ]
            field = "protocol_2.5d"
        package[field] = _choose_alternative(
            rng, compatible, package[field], f"{field} protocol"
        )
    elif move == "transfer_model":
        moved["transfer_model"] = copy.deepcopy(
            _choose_alternative(
                rng,
                params["transfer_model"],
                moved.get("transfer_model"),
                "transfer model",
            )
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
