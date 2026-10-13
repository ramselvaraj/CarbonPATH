"""Architecture and workload-mapping moves for modular ATLAS annealing.

One move changes one parameter of the design point: either the architecture or
the workload mapping. The evaluation profile is not part of the design point.

Value moves operate on any valid one-to-six-SA architecture. Chiplet-count and
package-topology moves are separate mutation families because they must rebuild
the package as one atomic proposal.

A parameter with one option cannot move. The search selects a different move
in its place, unless the search space sets ``skip_moves_without_alternative``
to false.

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
from system.utils.ArchitectureIdentity import (
    canonical_chiplet_labels,
    canonicalize_chiplet_labels,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.WorkloadMapping import (
    FUSED_ACTIVATION_POLICY,
    GEMM_DATAFLOWS,
    GEMM_TILE_ORDERS,
    movement_policy_options,
)


MOVE_TYPES = (
    "sa_count",
    "sa_array",
    "sa_tech_node",
    "sa_sram",
    "gemm_dataflow",
    "gemm_split_k",
    "gemm_assignment_order",
    "gemm_merge_tiles",
    "gemm_sa_set",
    "handoff_movement",
    "relu_fusion",
    "fpga_count",
    "fpga_clbs",
    "fpga_brams",
    "fpga_dsps",
    "fpga_frequency",
    "fpga_relu",
    "mem_type",
    "interconnect",
    "protocol",
    "transfer_model",
)

_FPGA_FIELDS = {"fpga_clbs": "clbs", "fpga_brams": "brams", "fpga_dsps": "dsps"}
_GEMM_SETTING_MOVES = {
    "gemm_dataflow": "dataflow",
    "gemm_split_k": "split_k",
    "gemm_assignment_order": "tile_order",
    "gemm_merge_tiles": "merge_tiles",
}
_GEMM_MAPPING_OPTIONS = {
    "dataflow": GEMM_DATAFLOWS,
    "split_k": (True, False),
    "tile_order": GEMM_TILE_ORDERS,
    # Tile merge is off, as in original CarbonPATH: one option.
    "merge_tiles": (False,),
}


class NoAlternative(UnsupportedEvaluation):
    """The selected parameter has no other option for this design point."""


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
    # The original annealer used one GEMM mapping for the whole workload and
    # counted a move with no alternative as one attempt.
    search_space["gemm_mapping"] = {"scope": "shared"}
    search_space["skip_moves_without_alternative"] = False
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
        raise NoAlternative(f"no alternative value is available for {label}")
    return rng.choice(alternatives)


def _gemm_mapping_space(params):
    space = params.get("gemm_mapping", {})
    if not isinstance(space, dict):
        raise UnsupportedEvaluation("search space gemm_mapping must be an object")
    scope = space.get("scope", "per_gemm")
    if scope not in ("per_gemm", "shared"):
        raise UnsupportedEvaluation("gemm_mapping scope must be per_gemm or shared")
    return space, scope


def _mutate_gemm_setting(mapping, field, params, rng):
    """Change one GEMM setting of one GEMM, or of every GEMM in shared scope."""
    space, scope = _gemm_mapping_space(params)
    gemm_indexes = mapping.gemm_indexes
    if not gemm_indexes:
        raise NoAlternative("the workload has no GEMM operation")
    options = space.get(field, _GEMM_MAPPING_OPTIONS[field])
    index = gemm_indexes[0] if scope == "shared" else rng.choice(gemm_indexes)
    current = getattr(mapping.operations[index], field)
    alternatives = [option for option in options if option != current]
    if not alternatives:
        raise NoAlternative(f"no alternative value is available for GEMM {field}")
    value = alternatives[0] if len(alternatives) == 1 else rng.choice(alternatives)
    if scope == "shared":
        return mapping.with_gemms(**{field: value})
    return mapping.with_operation(index, **{field: value})


def _mutate_gemm_sa_set(mapping, sa_ids, rng):
    """Add one SA to, or remove one SA from, the SA set of one GEMM."""
    gemm_indexes = mapping.gemm_indexes
    if not gemm_indexes or len(sa_ids) < 2:
        raise NoAlternative("no alternative SA set is available")
    index = rng.choice(gemm_indexes)
    current = mapping.operations[index].endpoint_ids
    sa_id = rng.choice(
        [sa_id for sa_id in sa_ids if len(current) > 1 or sa_id not in current]
    )
    if sa_id in current:
        moved = tuple(value for value in current if value != sa_id)
    else:
        moved = current + (sa_id,)
    return mapping.with_operation(index, endpoint_ids=moved)


def _mutate_handoff_movement(mapping, params, rng):
    """Select a different movement policy for one handoff."""
    candidates = []
    for handoff_index, current in enumerate(mapping.handoffs):
        options = movement_policy_options(
            params, mapping.handoff_kind(handoff_index)
        )
        if any(option != current for option in options):
            candidates.append((handoff_index, options))
    if not candidates:
        raise NoAlternative("no alternative movement policy is available")
    handoff_index, options = rng.choice(candidates)
    return mapping.with_handoff(
        handoff_index,
        _choose_alternative(
            rng, options, mapping.handoffs[handoff_index], "movement policy"
        ),
    )


def _fusion_candidates(mapping, endpoints, fusion_types):
    """The operations whose fusion can change.

    Each one of a type in ``fusion_types`` can be fused. A fused one can go
    back only to an FPGA.
    """
    return [
        index
        for index in mapping.fusable_indexes
        if mapping.operations[index].operation_type in fusion_types
        and (not mapping.is_fused(index) or endpoints.fpga_ids)
    ]


def _handoff_policy(params, handoff_kind, current, default_movement_policy):
    """The movement policy of a handoff whose kind changed.

    The current policy stays when the search space lists it for the new kind.
    Otherwise the first listed policy is used, and the default policy when the
    search space lists none.
    """
    options = [
        option
        for option in movement_policy_options(params, handoff_kind)
        if option != FUSED_ACTIVATION_POLICY
    ]
    if options:
        return current if current in options else options[0]
    if default_movement_policy is None:
        raise NoAlternative(
            f"no movement policy is available for a {handoff_kind} handoff"
        )
    return default_movement_policy


def _mutate_relu_fusion(
    mapping, endpoints, default_movement_policy, fusion_types, params, rng
):
    """Fuse one activation with the GEMM before it, or move a fused one to the FPGA.

    The endpoint of the activation and the handoffs on its two sides change in
    one move, because each of them is invalid without the others.
    """
    candidates = _fusion_candidates(mapping, endpoints, fusion_types)
    if not candidates:
        raise NoAlternative("no activation can change its fusion")
    index = rng.choice(candidates)
    last = index == len(mapping.operations) - 1
    if mapping.is_fused(index):
        endpoint_kind = "fpga"
        endpoint_ids = endpoints.fpga_ids
        incoming_policy = _handoff_policy(
            params,
            "sa_to_fpga",
            mapping.handoffs[index - 1],
            default_movement_policy,
        )
    else:
        endpoint_kind = "sa"
        endpoint_ids = mapping.operations[index - 1].endpoint_ids
        incoming_policy = FUSED_ACTIVATION_POLICY
    outgoing_policy = None
    if not last:
        outgoing_policy = _handoff_policy(
            params,
            f"{endpoint_kind}_to_{mapping.operations[index + 1].endpoint_kind}",
            mapping.handoffs[index],
            default_movement_policy,
        )
    return mapping.with_fusion_change(
        index, endpoint_kind, endpoint_ids, incoming_policy, outgoing_policy
    )


def _mutate_fpga_count(endpoints, params, rng):
    """Select a different FPGA count from the search space list.

    The list has one option now, so this move has no alternative. A second
    option needs the code that adds and removes an FPGA endpoint.
    """
    current = len(endpoints.fpga_keys)
    options = params.get("fpga", {}).get("count", [current])
    _choose_alternative(rng, options, current, "FPGA count")
    raise UnsupportedEvaluation(
        "the search space lists more than one FPGA count, but the FPGA-count "
        "move supports one option now"
    )


def _endpoint_id(key):
    return int(key.split("_", 1)[1]) - 1


def _mutate_sa_count(architecture, mapping, params, rng):
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
        raise NoAlternative("no SA-count move is available")
    operation = rng.choice(operations)
    added_keys = ()
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
        added_keys = (key,)
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
    rebuilt = rebuild_package_for_endpoints(
        moved,
        params,
        rng,
        preferred_package_type=preferred_package_type,
    )
    labels = canonical_chiplet_labels(rebuilt)
    moved_mapping = mapping.with_endpoints_relabeled(
        {
            _endpoint_id(old): _endpoint_id(new)
            for old, new in labels.items()
            if old not in added_keys
        },
        added_sa_ids=tuple(_endpoint_id(labels[key]) for key in added_keys),
    )
    return canonicalize_chiplet_labels(rebuilt), moved_mapping


def mutate_atlas_design_point(
    architecture,
    mapping,
    params,
    rng=None,
    default_movement_policy=None,
    fusion_types=(),
):
    """Return ``(architecture, workload mapping, move_name)`` for one move.

    Changes either the architecture or the workload mapping. The endpoint
    invariant is validated before returning; an invalid proposal raises
    :class:`UnsupportedEvaluation` so the caller can record it as invalid.

    ``fusion_types`` lists the activation types that the fusion move can
    change: the types for which the evaluation profile has a fused evaluator.
    ``default_movement_policy`` is the movement policy of a handoff kind that
    the search space does not list.
    """
    rng = rng or random
    candidate_moves = list(MOVE_TYPES)
    if "max_sa_chiplets" not in params and "max_chiplet" not in params:
        candidate_moves.remove("sa_count")

    endpoints = validate_atlas_architecture(architecture)
    if not endpoints.fpga_keys:
        candidate_moves = [
            candidate
            for candidate in candidate_moves
            if not candidate.startswith("fpga_") or candidate == "fpga_count"
        ]

    if not _fusion_candidates(mapping, endpoints, fusion_types):
        candidate_moves.remove("relu_fusion")

    skip = params.get("skip_moves_without_alternative", True)
    while True:
        move = _select_move(rng, candidate_moves, params)
        try:
            moved, moved_mapping = _apply_move(
                move,
                architecture,
                mapping,
                endpoints,
                params,
                rng,
                default_movement_policy,
                fusion_types,
            )
        except NoAlternative:
            if not skip:
                raise
            candidate_moves.remove(move)
            continue
        return moved, moved_mapping, move


def _apply_move(
    move,
    architecture,
    mapping,
    endpoints,
    params,
    rng,
    default_movement_policy=None,
    fusion_types=(),
):
    if move == "sa_count":
        return _mutate_sa_count(architecture, mapping, params, rng)
    if move == "fpga_count":
        _mutate_fpga_count(endpoints, params, rng)
    if move == "gemm_sa_set":
        return architecture, _mutate_gemm_sa_set(mapping, endpoints.sa_ids, rng)
    if move == "handoff_movement":
        return architecture, _mutate_handoff_movement(mapping, params, rng)
    if move == "relu_fusion":
        return architecture, _mutate_relu_fusion(
            mapping, endpoints, default_movement_policy, fusion_types, params, rng
        )

    # Every move below draws the SA first, also the moves that do not use it.
    # This keeps the random sequence of the original annealer.
    sa_key = rng.choice(endpoints.sa_keys)
    if move in _GEMM_SETTING_MOVES:
        return architecture, _mutate_gemm_setting(
            mapping, _GEMM_SETTING_MOVES[move], params, rng
        )
    moved = copy.deepcopy(architecture)
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
                raise NoAlternative("no alternative interconnect is available")
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

    validate_atlas_architecture(moved)
    return moved, mapping
