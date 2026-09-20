"""Direct ATLAS graph adapter.

ATLAS emits a graph-only artifact: a top-level JSON array of lowered nodes.
This module is the only code in CarbonPATH that reads that JSON. It turns
recognized nodes into one immutable, typed operation view and keeps the original
node available through a read-only source view for evaluator input adapters.

Graph-level structure (input nodes, linear-chain order, tensor shapes) is
validated here. Model-specific source requirements (for example ``reuse_factor``)
are validated later by the selected evaluator's input adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


SUPPORTED_OPERATION_TYPES = ("gemm", "relu", "softmax")
_GENERIC_STEMS = {"graph_dump", "graph", "dump"}
_MISSING = object()
_EMPTY_MAPPING = MappingProxyType({})


def _deep_freeze(value):
    """Return an immutable copy of parsed JSON data."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _positive_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise UnsupportedEvaluation(f"{field} must be a positive integer")
    return value


def _shape(value, field):
    if not isinstance(value, (list, tuple)) or not value:
        raise UnsupportedEvaluation(f"{field} must be a non-empty shape list")
    return tuple(_positive_int(dimension, field) for dimension in value)


def _product(shape):
    total = 1
    for dimension in shape:
        total *= dimension
    return total


@dataclass(frozen=True)
class TensorSpec:
    """One ATLAS tensor: identifier, shape, and source precision."""

    tensor_id: str
    shape: tuple
    precision: str | None = None

    @property
    def element_count(self):
        return _product(self.shape)


def _tensor_spec(entry, default_id=None):
    if not isinstance(entry, Mapping):
        raise UnsupportedEvaluation("ATLAS tensor must be an object")
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        name = default_id
    if not isinstance(name, str) or not name:
        raise UnsupportedEvaluation("ATLAS tensor must have a name")
    raw_shape = entry.get("shape")
    shape = () if raw_shape is None else _shape(raw_shape, f"{name}: tensor shape")
    precision = entry.get("precision")
    if precision is not None and not isinstance(precision, str):
        precision = str(precision)
    return TensorSpec(tensor_id=name, shape=shape, precision=precision)


class AtlasNodeView:
    """Read-only view over one recognized raw ATLAS node.

    The view is the only path an evaluator input adapter uses to read
    node-specific facts. It never exposes mutable JSON dictionaries.
    """

    def __init__(self, node):
        self._node = _deep_freeze(node)

    @property
    def name(self):
        return self._node.get("name")

    @property
    def node_class(self):
        return self._node.get("class")

    @property
    def index(self):
        return self._node.get("index")

    @property
    def attrs(self):
        attrs = self._node.get("attrs")
        return attrs if isinstance(attrs, Mapping) else _EMPTY_MAPPING

    def attribute(self, name, default=_MISSING):
        if name in self.attrs:
            return self.attrs[name]
        if default is not _MISSING:
            return default
        raise UnsupportedEvaluation(
            f"{self.name}: required ATLAS attribute '{name}' is missing"
        )

    @property
    def inputs(self):
        return tuple(self._node.get("inputs") or ())

    @property
    def weights(self):
        return tuple(self._node.get("weights") or ())

    def input_tensor(self, index=0):
        inputs = self.inputs
        if index < 0 or index >= len(inputs):
            raise UnsupportedEvaluation(
                f"{self.name}: missing input tensor {index}"
            )
        return _tensor_spec(inputs[index])

    def output_tensor(self, default_id=None):
        output = self._node.get("output")
        if not isinstance(output, Mapping):
            raise UnsupportedEvaluation(f"{self.name}: missing output tensor")
        return _tensor_spec(output, default_id=default_id)

    def weight(self, index=0):
        weights = self.weights
        if index < 0 or index >= len(weights):
            raise UnsupportedEvaluation(
                f"{self.name}: missing weight tensor {index}"
            )
        return _tensor_spec(weights[index])

    @property
    def reference_name(self):
        """Name other nodes use to consume this node's output.

        ATLAS renames lowered ``Dense`` layers to ``gemm_<name>`` while
        consumers still reference the original layer name, so strip the prefix
        for Gemm nodes.
        """
        name = self.name
        if self.node_class == "Gemm" and isinstance(name, str) and name.startswith("gemm_"):
            return name[len("gemm_"):]
        return name


@dataclass(frozen=True)
class AtlasOperation:
    """One immutable CarbonPATH view of a recognized ATLAS graph node."""

    operation_id: str
    operation_type: str
    input_tensor: TensorSpec
    output_tensor: TensorSpec
    source: AtlasNodeView
    gemm_shape: tuple | None = None

    @property
    def input_tensor_id(self):
        return self.input_tensor.tensor_id

    @property
    def output_tensor_id(self):
        return self.output_tensor.tensor_id

    @property
    def element_count(self):
        return self.output_tensor.element_count


@dataclass(frozen=True)
class AtlasGraph:
    name: str
    operations: tuple
    canonical_json: str

    @property
    def dtype(self):
        return "int8"

    def fingerprint(self):
        """Fingerprint the complete raw ATLAS artifact.

        The hash covers the full node array, so any change to any raw field
        changes the fingerprint even if this CarbonPATH version does not yet
        read that field.
        """
        serialized = self.canonical_json.encode("ascii")
        return hashlib.sha256(serialized).hexdigest()[:12]


def _require_name(node_view):
    name = node_view.name
    if not isinstance(name, str) or not name:
        raise UnsupportedEvaluation("ATLAS node must have a name")
    return name


def parse_gemm_node(node_view):
    operation_id = _require_name(node_view)
    if node_view.attrs.get("weights_in_core") is not True:
        raise UnsupportedEvaluation(
            f"{operation_id}: only constant-weight GEMM is supported"
        )

    if len(node_view.inputs) != 1:
        raise UnsupportedEvaluation(
            f"{operation_id}: GEMM requires exactly one activation input"
        )
    if not node_view.weights:
        raise UnsupportedEvaluation(f"{operation_id}: GEMM requires a weight tensor")

    weight = node_view.weight(0)
    if len(weight.shape) != 2:
        raise UnsupportedEvaluation(
            f"{operation_id}: GEMM requires a rank-2 weight tensor"
        )

    input_tensor = node_view.input_tensor(0)
    output_tensor = node_view.output_tensor(default_id=operation_id)

    if len(input_tensor.shape) == len(output_tensor.shape) == 1:
        m, k = 1, input_tensor.shape[0]
        n = output_tensor.shape[0]
    elif len(input_tensor.shape) == len(output_tensor.shape) == 2:
        m, k = input_tensor.shape
        if output_tensor.shape[0] != m:
            raise UnsupportedEvaluation(
                f"{operation_id}: input and output M dimensions must match"
            )
        n = output_tensor.shape[1]
    else:
        raise UnsupportedEvaluation(
            f"{operation_id}: activation tensors must both be rank 1 or rank 2"
        )

    return AtlasOperation(
        operation_id=operation_id,
        operation_type="gemm",
        input_tensor=input_tensor,
        output_tensor=TensorSpec(
            tensor_id=operation_id,
            shape=output_tensor.shape,
            precision=output_tensor.precision,
        ),
        source=node_view,
        gemm_shape=(m, k, n),
    )


def parse_relu_node(node_view):
    operation_id = _require_name(node_view)
    if node_view.attrs.get("activation") != "relu":
        raise UnsupportedEvaluation(
            f"{operation_id}: only relu activation is supported"
        )
    return _parse_elementwise_node(node_view, operation_id, "relu")


def parse_softmax_node(node_view):
    operation_id = _require_name(node_view)
    return _parse_elementwise_node(node_view, operation_id, "softmax")


def _parse_elementwise_node(node_view, operation_id, operation_type):
    if len(node_view.inputs) != 1:
        raise UnsupportedEvaluation(
            f"{operation_id}: {operation_type} requires exactly one activation input"
        )
    input_tensor = node_view.input_tensor(0)
    output_tensor = node_view.output_tensor(default_id=operation_id)
    if input_tensor.shape != output_tensor.shape:
        raise UnsupportedEvaluation(
            f"{operation_id}: {operation_type} input and output shapes must match"
        )
    return AtlasOperation(
        operation_id=operation_id,
        operation_type=operation_type,
        input_tensor=input_tensor,
        output_tensor=TensorSpec(
            tensor_id=operation_id,
            shape=output_tensor.shape,
            precision=output_tensor.precision,
        ),
        source=node_view,
    )


def _parse_operation(node_view):
    node_class = node_view.node_class
    if node_class == "Gemm":
        return parse_gemm_node(node_view)
    if node_class == "Activation":
        return parse_relu_node(node_view)
    if node_class == "Softmax":
        return parse_softmax_node(node_view)
    raise UnsupportedEvaluation(
        f"unsupported ATLAS node class '{node_class}' ({node_view.name})"
    )


def parse_atlas_graph(nodes, name):
    if not isinstance(nodes, list):
        raise UnsupportedEvaluation("ATLAS graph dump must be a top-level array")
    if not isinstance(name, str) or not name:
        raise UnsupportedEvaluation("ATLAS graph name must be a non-empty string")

    operations = []
    pending = None
    for node in nodes:
        if not isinstance(node, dict):
            raise UnsupportedEvaluation("graph node must be an object")
        node_view = AtlasNodeView(node)
        if node_view.node_class == "Input":
            continue
        operation = _parse_operation(node_view)
        if pending is not None and operation.input_tensor.tensor_id != pending:
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: only a linear ATLAS chain is supported "
                f"(expected input '{pending}', got '{operation.input_tensor.tensor_id}')"
            )
        operations.append(operation)
        pending = node_view.reference_name

    if not operations:
        raise UnsupportedEvaluation("ATLAS graph produced no supported operations")

    canonical_json = json.dumps(
        nodes, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return AtlasGraph(
        name=name,
        operations=tuple(operations),
        canonical_json=canonical_json,
    )


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
