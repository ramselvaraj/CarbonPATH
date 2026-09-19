"""Direct ATLAS graph-dump front end for CarbonPATH.

ATLAS emits a top-level JSON array of lowered hls4ml nodes: each node carries
``name``, ``class``, ``inputs``, ``output``, ``weights``, and ``attrs``. This
adapter reads that artifact directly. CarbonPATH does not own or accept a
normalized ATLAS workload format.

First scope is one linear chain of constant-weight ``Gemm`` nodes and ``relu``
``Activation`` nodes. Every other node or graph shape is reported as an
unsupported evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


SUPPORTED_OPERATION_TYPES = ("gemm", "relu")
_GENERIC_STEMS = {"graph_dump", "graph", "dump"}


@dataclass(frozen=True)
class GemmOperation:
    operation_id: str
    input_tensor_id: str
    output_tensor_id: str
    m: int
    k: int
    n: int
    operation_type: str = "gemm"

    @property
    def gemm_shape(self):
        return self.m, self.k, self.n

    @property
    def element_count(self):
        return self.m * self.n


@dataclass(frozen=True)
class ReluOperation:
    operation_id: str
    input_tensor_id: str
    output_tensor_id: str
    element_count: int
    operation_type: str = "relu"


@dataclass(frozen=True)
class AtlasGraph:
    name: str
    operations: tuple

    @property
    def dtype(self):
        return "int8"

    def canonical_dict(self):
        return {
            "name": self.name,
            "operations": [
                {
                    "operation_id": operation.operation_id,
                    "operation_type": operation.operation_type,
                    "input_tensor_id": operation.input_tensor_id,
                    "output_tensor_id": operation.output_tensor_id,
                    "element_count": operation.element_count,
                }
                for operation in self.operations
            ],
        }

    def fingerprint(self):
        serialized = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("ascii")).hexdigest()[:12]


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedEvaluation(f"{field} must be a positive integer")
    return value


def _shape(value, field):
    if not isinstance(value, list) or not value:
        raise UnsupportedEvaluation(f"{field} must be a non-empty shape list")
    return tuple(_positive_int(dimension, f"{field}") for dimension in value)


def _product(shape):
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def _reference_name(node):
    """Name other nodes use to consume this node's output.

    ATLAS renames lowered ``Dense`` layers to ``gemm_<name>`` while consumers
    still reference the original layer name, so strip the prefix for Gemm nodes.
    """
    name = node["name"]
    if node.get("class") == "Gemm" and name.startswith("gemm_"):
        return name[len("gemm_"):]
    return name


def parse_gemm_node(node):
    if node.get("class") != "Gemm":
        raise UnsupportedEvaluation("expected a Gemm node")
    operation_id = node.get("name")
    if not isinstance(operation_id, str) or not operation_id:
        raise UnsupportedEvaluation("Gemm node must have a name")

    attrs = node.get("attrs")
    if not isinstance(attrs, dict) or attrs.get("weights_in_core") is not True:
        raise UnsupportedEvaluation(
            f"{operation_id}: only constant-weight GEMM is supported"
        )

    inputs = node.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise UnsupportedEvaluation(
            f"{operation_id}: GEMM requires exactly one activation input"
        )
    weights = node.get("weights")
    if not isinstance(weights, list) or not weights:
        raise UnsupportedEvaluation(f"{operation_id}: GEMM requires a weight tensor")
    weight_shape = _shape(weights[0].get("shape"), f"{operation_id}: weight shape")
    if len(weight_shape) != 2:
        raise UnsupportedEvaluation(
            f"{operation_id}: GEMM requires a rank-2 weight tensor"
        )

    input_shape = _shape(inputs[0].get("shape"), f"{operation_id}: input shape")
    output = node.get("output")
    if not isinstance(output, dict):
        raise UnsupportedEvaluation(f"{operation_id}: GEMM requires an output shape")
    output_shape = _shape(output.get("shape"), f"{operation_id}: output shape")

    if len(input_shape) == len(output_shape) == 1:
        m, k = 1, input_shape[0]
        n = output_shape[0]
    elif len(input_shape) == len(output_shape) == 2:
        m, k = input_shape
        if output_shape[0] != m:
            raise UnsupportedEvaluation(
                f"{operation_id}: input and output M dimensions must match"
            )
        n = output_shape[1]
    else:
        raise UnsupportedEvaluation(
            f"{operation_id}: activation tensors must both be rank 1 or rank 2"
        )

    return GemmOperation(
        operation_id=operation_id,
        input_tensor_id=inputs[0].get("name"),
        output_tensor_id=operation_id,
        m=m,
        k=k,
        n=n,
    )


def parse_relu_node(node):
    if node.get("class") != "Activation":
        raise UnsupportedEvaluation("expected an Activation node")
    operation_id = node.get("name")
    if not isinstance(operation_id, str) or not operation_id:
        raise UnsupportedEvaluation("Activation node must have a name")
    attrs = node.get("attrs")
    if not isinstance(attrs, dict) or attrs.get("activation") != "relu":
        raise UnsupportedEvaluation(f"{operation_id}: only relu activation is supported")

    inputs = node.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise UnsupportedEvaluation(
            f"{operation_id}: relu requires exactly one activation input"
        )
    input_shape = _shape(inputs[0].get("shape"), f"{operation_id}: input shape")
    output = node.get("output")
    if not isinstance(output, dict):
        raise UnsupportedEvaluation(f"{operation_id}: relu requires an output shape")
    output_shape = _shape(output.get("shape"), f"{operation_id}: output shape")
    if input_shape != output_shape:
        raise UnsupportedEvaluation(
            f"{operation_id}: relu input and output shapes must match"
        )

    return ReluOperation(
        operation_id=operation_id,
        input_tensor_id=inputs[0].get("name"),
        output_tensor_id=operation_id,
        element_count=_product(input_shape),
    )


def _parse_operation(node):
    if not isinstance(node, dict):
        raise UnsupportedEvaluation("graph node must be an object")
    node_class = node.get("class")
    if node_class == "Gemm":
        return parse_gemm_node(node)
    if node_class == "Activation":
        return parse_relu_node(node)
    raise UnsupportedEvaluation(
        f"unsupported ATLAS node class '{node_class}' ({node.get('name')})"
    )


def parse_atlas_graph(nodes, name):
    if not isinstance(nodes, list):
        raise UnsupportedEvaluation("ATLAS graph dump must be a top-level array")
    if not isinstance(name, str) or not name:
        raise UnsupportedEvaluation("ATLAS graph name must be a non-empty string")

    operations = []
    pending = None
    for node in nodes:
        if isinstance(node, dict) and node.get("class") == "Input":
            continue
        operation = _parse_operation(node)
        if pending is not None and operation.input_tensor_id != pending:
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: only a linear ATLAS chain is supported "
                f"(expected input '{pending}', got '{operation.input_tensor_id}')"
            )
        operations.append(operation)
        pending = _reference_name(node)

    if not operations:
        raise UnsupportedEvaluation("ATLAS graph produced no supported operations")

    return AtlasGraph(name=name, operations=tuple(operations))


def _sanitize_name(value):
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value)
    )
    return cleaned or "atlas_graph"


def default_graph_name(path):
    path = Path(path)
    if path.stem.lower() in _GENERIC_STEMS:
        return _sanitize_name(path.parent.name)
    return _sanitize_name(path.stem)


def load_atlas_graph(path, name=None):
    with Path(path).open(encoding="utf-8") as file:
        nodes = json.load(file)
    return parse_atlas_graph(nodes, name or default_graph_name(path))
