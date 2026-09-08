from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from system.utils.IntermediateMemoryPolicy import INTERMEDIATE_POLICIES


def canonical_fingerprint(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()[:12]


@dataclass(frozen=True)
class LinearLayer:
    name: str
    batch_size: int
    in_features: int
    out_features: int

    @property
    def gemm_shape(self):
        return self.batch_size, self.in_features, self.out_features


@dataclass(frozen=True)
class LinearNetwork:
    name: str
    dtype: str
    batch_size: int
    input_features: int
    layers: tuple[LinearLayer, ...]
    intermediate_policy: str

    def to_workload_sequence(self):
        return {
            "id": self.name,
            "name": self.name,
            "gemms": [
                {"name": layer.name, "shape": layer.gemm_shape}
                for layer in self.layers
            ],
        }

    def canonical_dict(self):
        return {
            "schema_version": 1,
            "name": self.name,
            "dtype": self.dtype,
            "input": {
                "batch_size": self.batch_size,
                "features": self.input_features,
            },
            "layers": [
                {
                    "name": layer.name,
                    "op": "linear",
                    "out_features": layer.out_features,
                }
                for layer in self.layers
            ],
            "memory": {"intermediate_policy": self.intermediate_policy},
        }

    def fingerprint(self, search_space=None):
        network_definition = self.canonical_dict()
        network_definition.pop("memory")
        payload = {"network": network_definition}
        if search_space is not None:
            payload["search_space"] = search_space
        return canonical_fingerprint(payload)


def _positive_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _reject_unknown_fields(value, allowed, field):
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported {field} field(s): {', '.join(unknown)}")


def parse_network_entry(entry):
    if not isinstance(entry, dict):
        raise ValueError("Network definition must be an object")
    _reject_unknown_fields(
        entry,
        {"schema_version", "name", "dtype", "input", "layers", "memory"},
        "network",
    )
    if entry.get("schema_version") != 1:
        raise ValueError("Only network schema_version 1 is supported")

    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Network name must be a non-empty string")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise ValueError(
            "Network name must use only letters, numbers, dots, hyphens, and underscores"
        )
    dtype = entry.get("dtype")
    if dtype != "int8":
        raise ValueError("Only int8 networks are currently supported")

    network_input = entry.get("input")
    if not isinstance(network_input, dict):
        raise ValueError("Network input must be an object")
    _reject_unknown_fields(network_input, {"batch_size", "features"}, "input")
    batch_size = _positive_integer(
        network_input.get("batch_size"), "input.batch_size"
    )
    input_features = _positive_integer(
        network_input.get("features"), "input.features"
    )

    raw_layers = entry.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("Network must contain at least one linear layer")
    layers = []
    names = set()
    current_features = input_features
    for index, raw_layer in enumerate(raw_layers, start=1):
        if not isinstance(raw_layer, dict):
            raise ValueError(f"Layer {index} must be an object")
        _reject_unknown_fields(
            raw_layer,
            {"name", "op", "out_features"},
            f"layer {index}",
        )
        layer_name = raw_layer.get("name")
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError(f"Layer {index} must have a non-empty name")
        if layer_name in names:
            raise ValueError(f"Layer names must be unique: {layer_name}")
        if raw_layer.get("op") != "linear":
            raise ValueError(
                f"Layer '{layer_name}' uses unsupported operation "
                f"'{raw_layer.get('op')}'"
            )
        out_features = _positive_integer(
            raw_layer.get("out_features"),
            f"layers[{index - 1}].out_features",
        )
        layers.append(
            LinearLayer(
                name=layer_name,
                batch_size=batch_size,
                in_features=current_features,
                out_features=out_features,
            )
        )
        names.add(layer_name)
        current_features = out_features

    memory = entry.get("memory", {})
    if not isinstance(memory, dict):
        raise ValueError("Network memory configuration must be an object")
    _reject_unknown_fields(memory, {"intermediate_policy"}, "memory")
    policy = memory.get("intermediate_policy", "direct_forward")
    if policy not in INTERMEDIATE_POLICIES:
        raise ValueError(f"Unknown intermediate-memory policy: {policy}")

    return LinearNetwork(
        name=name,
        dtype=dtype,
        batch_size=batch_size,
        input_features=input_features,
        layers=tuple(layers),
        intermediate_policy=policy,
    )


def load_network(path):
    with Path(path).open(encoding="utf-8") as file:
        return parse_network_entry(json.load(file))
