import hashlib
import json
from copy import deepcopy


def _canonicalize(value, key=None):
    if isinstance(value, dict):
        return {
            item_key: _canonicalize(value[item_key], item_key)
            for item_key in sorted(value)
        }
    if isinstance(value, list):
        items = [_canonicalize(item) for item in value]
        if key == "inter_pkg_conn":
            return sorted(
                items,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":")
                ),
            )
        return items
    return value


def canonicalize_architecture(architecture):
    return _canonicalize(architecture)


def architecture_fingerprint(architecture):
    canonical = canonicalize_architecture(architecture)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()[:12]


def canonicalize_chiplet_labels(architecture):
    """Normalize arbitrary Chiplet_N names while preserving stack roles."""
    result = deepcopy(architecture)
    chiplets = sorted(
        (key for key in result if key.startswith("Chiplet_")),
        key=lambda key: int(key.split("_", 1)[1]),
    )
    if not chiplets:
        return result

    package = result.get("pkg", {})
    memory = package.get("mem_pkg_conn", {})
    connections = package.get("inter_pkg_conn", [])
    connection_dicts = [
        connection for connection in connections if isinstance(connection, dict)
    ] if isinstance(connections, list) else []
    outgoing = {
        connection.get("from"): connection.get("to")
        for connection in connection_dicts
        if connection.get("to") != "na"
    }
    incoming = {
        connection.get("to")
        for connection in connection_dicts
        if connection.get("to") != "na"
    }
    starts = [chiplet for chiplet in chiplets if chiplet not in incoming]
    if len(starts) == 1:
        ordered = []
        current = starts[0]
        while current in chiplets and current not in ordered:
            ordered.append(current)
            current = outgoing.get(current)
        ordered.extend(chiplet for chiplet in chiplets if chiplet not in ordered)
    else:
        def chiplet_key(chiplet):
            hardware = result[chiplet]
            chiplet_type = str(
                hardware.get("chiplet_type", "systolic_array")
            ).lower()
            identity = {
                "hardware": hardware,
                "memory_allocation": memory.get(chiplet),
            }
            return (
                1 if chiplet_type == "fpga" else 0,
                json.dumps(
                    _canonicalize(identity), sort_keys=True, separators=(",", ":")
                ),
            )

        ordered = sorted(chiplets, key=chiplet_key)
    labels = {old: f"Chiplet_{index}" for index, old in enumerate(ordered, start=1)}

    normalized = {
        labels.get(key, key): value
        for key, value in result.items()
        if not key.startswith("Chiplet_")
    }
    for old in ordered:
        normalized[labels[old]] = result[old]

    package = deepcopy(result.get("pkg", {}))
    package["inter_pkg_conn"] = [
        (
            {
                **connection,
                "from": labels.get(connection.get("from"), connection.get("from")),
                "to": labels.get(connection.get("to"), connection.get("to")),
            }
            if isinstance(connection, dict)
            else connection
        )
        for connection in package.get("inter_pkg_conn", [])
    ] if isinstance(package.get("inter_pkg_conn", []), list) else package.get("inter_pkg_conn")
    package["mem_pkg_conn"] = {
        (labels.get(key, key)): value for key, value in memory.items()
    }
    normalized["pkg"] = package
    return normalized


def canonical_architecture_fingerprint(architecture):
    canonical = canonicalize_chiplet_labels(architecture)
    serialized = json.dumps(
        _canonicalize(canonical), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()[:12]


def physical_architecture_fingerprint(architecture):
    """Fingerprint hardware and packaging while excluding workload mapping."""
    physical = deepcopy(architecture)
    physical.pop("WL_mapping", None)
    return canonical_architecture_fingerprint(physical)


def effective_gemm_mapping(architecture, profile=None):
    """Return the legacy GEMM decisions after modular profile overrides."""
    mapping = architecture.get("WL_mapping", {}).get("mapping", {})
    dataflow = mapping.get("dataflow")
    if isinstance(dataflow, list):
        dataflow = dataflow[0] if dataflow else None
    result = {
        "dataflow": dataflow,
        "split_k": bool(mapping.get("if_splitting_k", 0)),
        "assignment_order": (
            "ascending"
            if mapping.get("assign_workload_in_ascending_order", 0)
            else "descending"
        ),
    }
    if profile is None:
        return result
    if hasattr(profile, "canonical_dict"):
        profile = profile.canonical_dict()
    gemm = profile.get("evaluators", {}).get("gemm", {})
    settings = gemm.get("settings", {}) if isinstance(gemm, dict) else {}
    for key in result:
        if key in settings:
            result[key] = settings[key]
    result["split_k"] = bool(result["split_k"])
    return result


def compatible_result_fingerprint(architecture, profile=None):
    """Identify a result by physical design and effective legacy GEMM mapping."""
    value = {
        "physical_architecture": physical_architecture_fingerprint(architecture),
        "gemm_mapping": effective_gemm_mapping(architecture, profile),
    }
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()[:12]
