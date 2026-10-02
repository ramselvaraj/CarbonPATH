"""Canonical endpoint and package validation for modular CarbonPATH.

This module describes which compute endpoints an architecture contains.  It is
shared by construction, placement, and mutation; it has no annealing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


@dataclass(frozen=True)
class EndpointGroup:
    """A stable, non-empty collection of endpoints of the same kind."""

    endpoint_kind: str
    endpoint_ids: tuple[int, ...]

    def __post_init__(self):
        if not self.endpoint_ids:
            raise ValueError("endpoint group cannot be empty")
        if len(set(self.endpoint_ids)) != len(self.endpoint_ids):
            raise ValueError("endpoint group cannot contain duplicate endpoints")


@dataclass(frozen=True)
class ArchitectureEndpoints:
    """The SA and FPGA chiplets declared by one architecture."""

    sa_keys: tuple[str, ...]
    fpga_keys: tuple[str, ...]

    @property
    def sa_ids(self):
        return tuple(_chiplet_id(key) for key in self.sa_keys)

    @property
    def fpga_ids(self):
        return tuple(_chiplet_id(key) for key in self.fpga_keys)

    @property
    def sa_group(self):
        return EndpointGroup("sa", self.sa_ids)

    def require_single_sa_fpga(self):
        """Return the two keys required by the first-generation ATLAS moves."""
        if len(self.sa_keys) != 1 or len(self.fpga_keys) != 1:
            raise UnsupportedEvaluation(
                "current modular mutations require exactly one systolic-array "
                f"and one FPGA endpoint, found {len(self.sa_keys)} SA and "
                f"{len(self.fpga_keys)} FPGA"
            )
        return self.sa_keys[0], self.fpga_keys[0]


def _chiplet_id(key):
    try:
        return int(key.split("_", 1)[1]) - 1
    except (IndexError, ValueError):
        raise UnsupportedEvaluation(f"invalid chiplet key '{key}'")


def _is_fpga_chiplet(chiplet):
    return str(chiplet.get("chiplet_type", "systolic_array")).lower() == "fpga"


def _reachable(start, adjacency):
    seen = set()
    pending = [start]
    while pending:
        endpoint = pending.pop()
        if endpoint in seen:
            continue
        seen.add(endpoint)
        pending.extend(adjacency.get(endpoint, ()))
    return seen


def validate_modular_architecture(architecture):
    """Validate endpoint membership and return its stable endpoint groups.

    Detailed technology characterization remains the responsibility of
    ``ChipletSystem``.  This boundary checks the invariants needed before that
    characterization can safely run: supported SA count, memory membership,
    and package connectivity.
    """
    if not isinstance(architecture, dict):
        raise UnsupportedEvaluation("architecture must be an object")

    chiplets = {
        key: value
        for key, value in architecture.items()
        if isinstance(key, str) and key.startswith("Chiplet_")
    }
    for key in chiplets:
        _chiplet_id(key)
    ordered_keys = tuple(sorted(chiplets, key=_chiplet_id))
    if len({_chiplet_id(key) for key in ordered_keys}) != len(ordered_keys):
        raise UnsupportedEvaluation("chiplet identifiers must be unique")

    sa_keys = tuple(
        key for key in ordered_keys if not _is_fpga_chiplet(chiplets[key])
    )
    fpga_keys = tuple(
        key for key in ordered_keys if _is_fpga_chiplet(chiplets[key])
    )
    if not sa_keys:
        raise UnsupportedEvaluation(
            "modular architecture requires at least one systolic-array endpoint"
        )
    if len(sa_keys) > 6:
        raise UnsupportedEvaluation(
            f"modular architecture supports at most six systolic arrays, found {len(sa_keys)}"
        )

    package = architecture.get("pkg")
    if not isinstance(package, dict):
        raise UnsupportedEvaluation("modular architecture requires a package object")
    memory = package.get("mem_pkg_conn")
    if not isinstance(memory, dict):
        raise UnsupportedEvaluation("package requires a memory connection object")

    declared = set(ordered_keys)
    memory_endpoints = {
        key for key in memory if isinstance(key, str) and key.startswith("Chiplet_")
    }
    missing_memory = sorted(declared - memory_endpoints, key=_chiplet_id)
    unknown_memory = sorted(memory_endpoints - declared, key=_chiplet_id)
    if missing_memory:
        raise UnsupportedEvaluation(
            "package has no memory allocation for " + ", ".join(missing_memory)
        )
    if unknown_memory:
        raise UnsupportedEvaluation(
            "package memory references missing " + ", ".join(unknown_memory)
        )

    package_type = str(package.get("HI_pkg_type", "")).lower()
    connections = package.get("inter_pkg_conn")
    if package_type in ("2d", "2d_na"):
        if len(declared) != 1:
            raise UnsupportedEvaluation(
                "a 2D package can contain only one compute endpoint"
            )
    else:
        if not isinstance(connections, list):
            raise UnsupportedEvaluation(
                "multi-endpoint package requires an interconnect list"
            )
        adjacency = {key: set() for key in declared}
        unknown_connections = set()
        for connection in connections:
            if not isinstance(connection, dict):
                raise UnsupportedEvaluation("package connections must be objects")
            source = connection.get("from")
            destination = connection.get("to")
            for endpoint in (source, destination):
                if endpoint != "na" and endpoint not in declared:
                    unknown_connections.add(endpoint)
            if source in declared and destination in declared:
                adjacency[source].add(destination)
                adjacency[destination].add(source)
        if unknown_connections:
            raise UnsupportedEvaluation(
                "modular architecture package references missing "
                + ", ".join(sorted(unknown_connections))
            )
        if declared and _reachable(ordered_keys[0], adjacency) != declared:
            raise UnsupportedEvaluation(
                "package interconnect must connect every compute endpoint"
            )

    return ArchitectureEndpoints(sa_keys=sa_keys, fpga_keys=fpga_keys)


def _package_options(params):
    package = params.get("pkg", {})
    return (
        package.get("inter_pkg_architecture")
        or params.get("inter_pkg_arch")
        or []
    )


def _protocol_options(params):
    package = params.get("pkg", {})
    return package.get("protocol") or params.get("protocol_arch") or []


def _memory_channels(architecture, endpoints, memory_type):
    sa_keys = endpoints.sa_keys
    total = 16 if "hbm" in memory_type.lower() else 8
    total = max(total, len(sa_keys))
    channels = {key: 1 for key in sa_keys}
    remaining = total - len(sa_keys)
    ordered = sorted(
        sa_keys,
        key=lambda key: (-float(architecture[key].get("area", 0)), _chiplet_id(key)),
    )
    for index in range(remaining):
        channels[ordered[index % len(ordered)]] += 1
    return {
        "mem_type": memory_type,
        **channels,
        **{key: 0 for key in endpoints.fpga_keys},
    }


def _compatible_protocol(connection_type, protocol_options, rng=None):
    if connection_type == "2.5d_rdl":
        return "ucie_std" if "ucie_std" in protocol_options else "na"
    if connection_type.startswith("2.5d_"):
        options = [
            option
            for option in protocol_options
            if option in ("ucie_adv", "aib", "bow")
        ]
        return (rng.choice(options) if rng is not None else options[0]) if options else "na"
    if connection_type.startswith("3d_"):
        return "ucie_3d" if "ucie_3d" in protocol_options else "na"
    return "na"


def supported_package_types(architecture, params):
    """Return package topologies that can be built for these endpoints."""
    endpoints = validate_modular_architecture_membership(architecture)
    endpoint_count = len(endpoints.sa_keys) + len(endpoints.fpga_keys)
    if endpoint_count == 1:
        return ("2d",)
    if endpoints.fpga_keys:
        return ("2.5d",)

    links = _package_options(params)
    has_25d = any(option.startswith("2.5d_") for option in links)
    has_3d = any(option.startswith("3d_") for option in links)
    inferred = []
    if has_25d:
        inferred.append("2.5d")
    if has_3d:
        inferred.append("3d")
    if endpoint_count >= 3 and has_25d and has_3d:
        inferred.append("2.5d_3d")

    configured = params.get("pkg", {}).get("package_types")
    if configured:
        inferred = [value for value in configured if value in inferred]
    return tuple(inferred)


def _select_link(available, current, preferred, rng, label):
    if preferred is not None:
        if preferred not in available:
            raise UnsupportedEvaluation(
                f"{label} interconnect '{preferred}' is outside the search space"
            )
        return preferred
    if current in available:
        return current
    if not available:
        raise UnsupportedEvaluation(f"no {label} interconnect is available")
    return rng.choice(available)


def rebuild_package_for_endpoints(
    architecture,
    params,
    rng,
    preferred_connection_type=None,
    preferred_package_type=None,
):
    """Return a copy with memory and links rebuilt for its declared endpoints.

    Structural mutation callers provide hardware choices only. This module owns
    the dependent package membership, link construction, and compatible
    protocol selection needed to make that hardware evaluable.
    """
    rebuilt = deepcopy(architecture)
    endpoints = validate_modular_architecture_membership(rebuilt)
    all_keys = tuple(sorted(endpoints.sa_keys + endpoints.fpga_keys, key=_chiplet_id))
    current_package = rebuilt.get("pkg", {})
    memory_type = current_package.get("mem_pkg_conn", {}).get("mem_type", "ddr5")

    if len(all_keys) == 1:
        package_type = "2d"
        connections = "2d_na"
        protocol_3d = "na"
        protocol_25d = "na"
    else:
        available_links = [option for option in _package_options(params) if option != "2d_na"]
        available_25d = [option for option in available_links if option.startswith("2.5d_")]
        available_3d = [option for option in available_links if option.startswith("3d_")]
        current_connections = current_package.get("inter_pkg_conn", [])
        current_25d = next(
            (
                connection.get("connection_type")
                for connection in current_connections
                if isinstance(connection, dict)
                and str(connection.get("connection_type", "")).startswith("2.5d_")
            ),
            None,
        ) if isinstance(current_connections, list) else None
        current_3d = next(
            (
                connection.get("connection_type")
                for connection in current_connections
                if isinstance(connection, dict)
                and str(connection.get("connection_type", "")).startswith("3d_")
            ),
            None,
        ) if isinstance(current_connections, list) else None

        package_types = supported_package_types(rebuilt, params)
        if preferred_package_type is not None and preferred_package_type not in package_types:
            raise UnsupportedEvaluation(
                f"package type '{preferred_package_type}' is outside the search space"
            )
        current_type = current_package.get("HI_pkg_type")
        package_type = preferred_package_type or (
            current_type if current_type in package_types else rng.choice(package_types)
        )

        preferred_25d = (
            preferred_connection_type
            if str(preferred_connection_type).startswith("2.5d_")
            else None
        )
        preferred_3d = (
            preferred_connection_type
            if str(preferred_connection_type).startswith("3d_")
            else None
        )

        if package_type == "3d":
            connection_type = _select_link(
                available_3d, current_3d, preferred_3d, rng, "3D"
            )
            ordered = sorted(
                all_keys,
                key=lambda key: (-float(rebuilt[key].get("area", 0)), _chiplet_id(key)),
            )
            connections = []
            for index, source in enumerate(ordered):
                destination = ordered[index + 1] if index + 1 < len(ordered) else "na"
                location = (
                    "stack0_base"
                    if index == 0
                    else "stack0_top"
                    if index == len(ordered) - 1
                    else f"stack0_middle{index}"
                )
                connections.append(
                    {
                        "from": source,
                        "to": destination,
                        "connection_type": connection_type,
                        "loc": location,
                    }
                )
            protocol_3d = _compatible_protocol(
                connection_type, _protocol_options(params), rng
            )
            protocol_25d = "na"
        elif package_type == "2.5d":
            connection_type = _select_link(
                available_25d, current_25d, preferred_25d, rng, "2.5D"
            )
            connections = [
                {
                    "from": source,
                    "to": destination,
                    "connection_type": connection_type,
                    "loc": "2.5d_chiplet",
                }
                for source, destination in zip(all_keys, all_keys[1:])
            ]
            protocol_3d = "na"
            protocol_25d = _compatible_protocol(
                connection_type, _protocol_options(params), rng
            )
        else:
            connection_25d = _select_link(
                available_25d, current_25d, preferred_25d, rng, "2.5D"
            )
            connection_3d = _select_link(
                available_3d, current_3d, preferred_3d, rng, "3D"
            )
            ordered = sorted(
                all_keys,
                key=lambda key: (-float(rebuilt[key].get("area", 0)), _chiplet_id(key)),
            )
            stack_size = rng.choice(tuple(range(2, len(ordered))))
            stack = ordered[:stack_size]
            beside_stack = ordered[stack_size:]
            connections = []
            for index, source in enumerate(stack):
                destination = stack[index + 1] if index + 1 < len(stack) else "na"
                location = (
                    "stack0_base"
                    if index == 0
                    else "stack0_top"
                    if index == len(stack) - 1
                    else f"stack0_middle{index}"
                )
                connections.append(
                    {
                        "from": source,
                        "to": destination,
                        "connection_type": connection_3d,
                        "loc": location,
                    }
                )
            connections.extend(
                {
                    "from": stack[0],
                    "to": endpoint,
                    "connection_type": connection_25d,
                    "loc": "2.5d_chiplet",
                }
                for endpoint in beside_stack
            )
            protocol_3d = _compatible_protocol(
                connection_3d, _protocol_options(params), rng
            )
            protocol_25d = _compatible_protocol(
                connection_25d, _protocol_options(params), rng
            )

    rebuilt["pkg"] = {
        "HI_pkg_type": package_type,
        "inter_pkg_conn": connections,
        "protocol_3d": protocol_3d,
        "protocol_2.5d": protocol_25d,
        "mem_pkg_conn": _memory_channels(rebuilt, endpoints, memory_type),
    }
    validate_modular_architecture(rebuilt)
    return rebuilt


def validate_modular_architecture_membership(architecture):
    """Return endpoints before package links are regenerated."""
    chiplets = {
        key: value
        for key, value in architecture.items()
        if isinstance(key, str) and key.startswith("Chiplet_")
    }
    ordered = tuple(sorted(chiplets, key=_chiplet_id))
    sa_keys = tuple(key for key in ordered if not _is_fpga_chiplet(chiplets[key]))
    fpga_keys = tuple(key for key in ordered if _is_fpga_chiplet(chiplets[key]))
    if not sa_keys or len(sa_keys) > 6:
        raise UnsupportedEvaluation(
            f"modular architecture requires one through six systolic arrays, found {len(sa_keys)}"
        )
    return ArchitectureEndpoints(sa_keys=sa_keys, fpga_keys=fpga_keys)
