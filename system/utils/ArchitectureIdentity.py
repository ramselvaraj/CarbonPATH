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
    connections = package.get("inter_pkg_conn", [])
    outgoing = {
        connection.get("from"): connection.get("to")
        for connection in connections
        if connection.get("to") != "na"
    }
    incoming = {
        connection.get("to")
        for connection in connections
        if connection.get("to") != "na"
    }
    starts = [chiplet for chiplet in chiplets if chiplet not in incoming]
    ordered = []
    current = starts[0] if len(starts) == 1 else chiplets[0]
    while current in chiplets and current not in ordered:
        ordered.append(current)
        current = outgoing.get(current)
    ordered.extend(chiplet for chiplet in chiplets if chiplet not in ordered)
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
        {
            **connection,
            "from": labels.get(connection.get("from"), connection.get("from")),
            "to": labels.get(connection.get("to"), connection.get("to")),
        }
        for connection in package.get("inter_pkg_conn", [])
    ]
    memory = package.get("mem_pkg_conn", {})
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
