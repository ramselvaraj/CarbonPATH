"""Normalized parser-workload frontend for CarbonPATH.

CarbonPATH does not read raw ATLAS graph dumps. An external parser converts an
ATLAS graph into the normalized envelope accepted here, and CarbonPATH consumes
that envelope directly.

Schema ``atlas-normalized-v0`` supports a single sequential chain of ``gemm``
and ``relu`` operations with explicit tensor IDs. See
``docs/atlas_fpga_relu_v0.md`` for the full contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

from system.utils.IntermediateMemoryPolicy import INTERMEDIATE_POLICIES


ATLAS_WORKLOAD_FORMAT = "atlas-normalized-v0"
ATLAS_SUPPORTED_DTYPE = "int8"
SUPPORTED_OPERATION_TYPES = ("gemm", "relu")


def canonical_fingerprint(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()[:12]


def _positive_integer(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _shape(value, field, arity=None):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of positive integers")
    if arity is not None and len(value) != arity:
        raise ValueError(f"{field} must have exactly {arity} dimensions")
    result = tuple(
        _positive_integer(dimension, f"{field}[{index}]")
        for index, dimension in enumerate(value)
    )
    return result


def _product(shape):
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def _reject_unknown_fields(value, allowed, field):
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported {field} field(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class GemmOperation:
    operation_id: str
    operation_type: str
    input_tensor_id: str
    weight_tensor_id: str
    output_tensor_id: str
    input_shape: tuple[int, int]
    weight_shape: tuple[int, int]
    output_shape: tuple[int, int]
    m: int
    k: int
    n: int

    @property
    def gemm_shape(self):
        return self.m, self.k, self.n

    def canonical_dict(self):
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "input_tensor_id": self.input_tensor_id,
            "weight_tensor_id": self.weight_tensor_id,
            "output_tensor_id": self.output_tensor_id,
            "input_shape": list(self.input_shape),
            "weight_shape": list(self.weight_shape),
            "output_shape": list(self.output_shape),
            "m": self.m,
            "k": self.k,
            "n": self.n,
        }


@dataclass(frozen=True)
class ReluOperation:
    operation_id: str
    operation_type: str
    input_tensor_id: str
    output_tensor_id: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    element_count: int

    def canonical_dict(self):
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "input_tensor_id": self.input_tensor_id,
            "output_tensor_id": self.output_tensor_id,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "element_count": self.element_count,
        }


@dataclass(frozen=True)
class AtlasWorkload:
    name: str
    operations: tuple
    intermediate_policy: str = "direct_forward"

    @property
    def dtype(self):
        return ATLAS_SUPPORTED_DTYPE

    def canonical_dict(self):
        return {
            "format": ATLAS_WORKLOAD_FORMAT,
            "name": self.name,
            "dtype": ATLAS_SUPPORTED_DTYPE,
            "operations": [op.canonical_dict() for op in self.operations],
        }

    def fingerprint(self, search_space=None):
        definition = self.canonical_dict()
        definition.pop("dtype", None)
        payload = {"network": definition}
        if search_space is not None:
            payload["search_space"] = search_space
        return canonical_fingerprint(payload)


def _parse_gemm(raw, index, names, produced):
    allowed = {
        "operation_id",
        "operation_type",
        "input_tensor_id",
        "weight_tensor_id",
        "output_tensor_id",
        "input_shape",
        "weight_shape",
        "output_shape",
        "m",
        "k",
        "n",
    }
    _reject_unknown_fields(raw, allowed, f"operation {index}")

    operation_id = raw.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise ValueError(f"operation {index} must have a non-empty operation_id")
    if operation_id in names:
        raise ValueError(f"operation_id must be unique: {operation_id}")

    input_tensor_id = raw.get("input_tensor_id")
    weight_tensor_id = raw.get("weight_tensor_id")
    output_tensor_id = raw.get("output_tensor_id")
    for field, value in (
        ("input_tensor_id", input_tensor_id),
        ("weight_tensor_id", weight_tensor_id),
        ("output_tensor_id", output_tensor_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{operation_id} must have a non-empty {field}")

    input_shape = _shape(raw.get("input_shape"), f"{operation_id}.input_shape", 2)
    weight_shape = _shape(raw.get("weight_shape"), f"{operation_id}.weight_shape", 2)
    output_shape = _shape(
        raw.get("output_shape"), f"{operation_id}.output_shape", 2
    )
    m = _positive_integer(raw.get("m"), f"{operation_id}.m")
    k = _positive_integer(raw.get("k"), f"{operation_id}.k")
    n = _positive_integer(raw.get("n"), f"{operation_id}.n")

    if input_shape != (m, k):
        raise ValueError(
            f"{operation_id} input_shape {input_shape} must equal [M, K] = [{m}, {k}]"
        )
    if weight_shape != (k, n):
        raise ValueError(
            f"{operation_id} weight_shape {weight_shape} must equal [K, N] = [{k}, {n}]"
        )
    if output_shape != (m, n):
        raise ValueError(
            f"{operation_id} output_shape {output_shape} must equal [M, N] = [{m}, {n}]"
        )

    return GemmOperation(
        operation_id=operation_id,
        operation_type="gemm",
        input_tensor_id=input_tensor_id,
        weight_tensor_id=weight_tensor_id,
        output_tensor_id=output_tensor_id,
        input_shape=input_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        m=m,
        k=k,
        n=n,
    )


def _parse_relu(raw, index, names, produced):
    allowed = {
        "operation_id",
        "operation_type",
        "input_tensor_id",
        "output_tensor_id",
        "input_shape",
        "output_shape",
        "element_count",
    }
    _reject_unknown_fields(raw, allowed, f"operation {index}")

    operation_id = raw.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise ValueError(f"operation {index} must have a non-empty operation_id")
    if operation_id in names:
        raise ValueError(f"operation_id must be unique: {operation_id}")

    input_tensor_id = raw.get("input_tensor_id")
    output_tensor_id = raw.get("output_tensor_id")
    for field, value in (
        ("input_tensor_id", input_tensor_id),
        ("output_tensor_id", output_tensor_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{operation_id} must have a non-empty {field}")

    input_shape = _shape(raw.get("input_shape"), f"{operation_id}.input_shape")
    output_shape = _shape(raw.get("output_shape"), f"{operation_id}.output_shape")
    if input_shape != output_shape:
        raise ValueError(
            f"{operation_id} relu input_shape {input_shape} must equal "
            f"output_shape {output_shape}"
        )

    element_count = _positive_integer(
        raw.get("element_count"), f"{operation_id}.element_count"
    )
    if element_count != _product(input_shape):
        raise ValueError(
            f"{operation_id} element_count {element_count} must equal "
            f"the shape product {_product(input_shape)}"
        )

    return ReluOperation(
        operation_id=operation_id,
        operation_type="relu",
        input_tensor_id=input_tensor_id,
        output_tensor_id=output_tensor_id,
        input_shape=input_shape,
        output_shape=output_shape,
        element_count=element_count,
    )


def parse_atlas_workload_entry(entry):
    """Parse a normalized parser-workload envelope into an :class:`AtlasWorkload`."""
    if not isinstance(entry, dict):
        raise ValueError("ATLAS workload must be an object")
    _reject_unknown_fields(
        entry,
        {"format", "name", "dtype", "operations", "memory"},
        "ATLAS workload",
    )
    if entry.get("format") != ATLAS_WORKLOAD_FORMAT:
        raise ValueError(
            f"Only workload format '{ATLAS_WORKLOAD_FORMAT}' is supported"
        )

    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("ATLAS workload name must be a non-empty string")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise ValueError(
            "ATLAS workload name must use only letters, numbers, dots, "
            "hyphens, and underscores"
        )

    dtype = entry.get("dtype", ATLAS_SUPPORTED_DTYPE)
    if dtype != ATLAS_SUPPORTED_DTYPE:
        raise ValueError("Only int8 ATLAS workloads are currently supported")

    raw_operations = entry.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("ATLAS workload must contain at least one operation")

    memory = entry.get("memory", {})
    if not isinstance(memory, dict):
        raise ValueError("ATLAS workload memory configuration must be an object")
    _reject_unknown_fields(memory, {"intermediate_policy"}, "memory")
    policy = memory.get("intermediate_policy", "direct_forward")
    if policy not in INTERMEDIATE_POLICIES:
        raise ValueError(f"Unknown intermediate-memory policy: {policy}")

    operations = []
    names = set()
    produced = set()
    previous_output = None
    for index, raw in enumerate(raw_operations, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"operation {index} must be an object")
        operation_type = raw.get("operation_type")
        if operation_type not in SUPPORTED_OPERATION_TYPES:
            raise ValueError(
                f"operation {index} uses unsupported operation_type "
                f"'{operation_type}'"
            )

        if index > 1:
            expected = previous_output
            if raw.get("input_tensor_id") != expected:
                raise ValueError(
                    f"operation {raw.get('operation_id')} input tensor "
                    f"'{raw.get('input_tensor_id')}' must consume the previous "
                    f"output '{expected}'"
                )

        if operation_type == "gemm":
            operation = _parse_gemm(raw, index, names, produced)
        else:
            operation = _parse_relu(raw, index, names, produced)

        names.add(operation.operation_id)
        if operation.output_tensor_id in produced:
            raise ValueError(
                f"tensor '{operation.output_tensor_id}' is produced more than once"
            )
        produced.add(operation.output_tensor_id)
        operations.append(operation)
        previous_output = operation.output_tensor_id

    return AtlasWorkload(
        name=name,
        operations=tuple(operations),
        intermediate_policy=policy,
    )


def load_atlas_workload(path):
    with Path(path).open(encoding="utf-8") as file:
        return parse_atlas_workload_entry(json.load(file))
