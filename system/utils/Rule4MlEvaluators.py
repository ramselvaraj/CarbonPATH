"""rule4ml operation evaluators, input adapters, and registration.

A rule4ml evaluator estimates one FPGA activation operation (ReLU or Softmax)
with rule4ml's trained latency/resource predictors instead of CarbonPATH's
resource-derived legacy model. It is a drop-in operation evaluator: an
evaluation profile names it, the executor and movement policies are unchanged.

rule4ml and its heavy dependencies (TensorFlow, torch) are imported lazily,
inside the prediction path. Every profile that does not name a rule4ml evaluator
therefore loads and runs without rule4ml installed. The loose coupling also lets
the unit tests stub the prediction call.

Evaluator settings, read from the profile's per-operation ``settings`` object:

* ``board``                   rule4ml board; default is the nearest board.
* ``strategy``                ``latency`` or ``resource`` (default ``latency``).
* ``precision``               HLS precision; default from the ATLAS tensor.
* ``reuse_factor``            HLS reuse factor; default from the ATLAS attribute.
* ``hls4ml_version``          default ``1.1.0``.
* ``vivado_version``          default ``2024.2``.
* ``latency_metric``          ``cycles`` (default) or ``interval``.
* ``energy_per_element_pj``   dynamic energy per element; default ``0.0``.
"""

from __future__ import annotations

import re

from dataclasses import dataclass

from system.utils.OperationEvaluator import (
    OperationEstimate,
    OperationEvaluator,
    OperationInputAdapter,
)
from system.utils.Rule4MlBoards import nearest_board
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


_SUPPORTED_SETTINGS = {
    "board",
    "strategy",
    "precision",
    "reuse_factor",
    "hls4ml_version",
    "vivado_version",
    "latency_metric",
    "energy_per_element_pj",
}
_DEFAULT_PRECISION = "ap_fixed<16, 6>"
_DEFAULT_HLS4ML_VERSION = "1.1.0"
_DEFAULT_VIVADO_VERSION = "2024.2"
_RESOURCE_LABELS = ("BRAM", "DSP", "FF", "LUT")


_ATLAS_FIXED = re.compile(r"^\s*u?fixed\s*<\s*(\d+)\s*,\s*(-?\d+)\s*[,>]")


def rule4ml_precision(precision):
    """Return an ``ap_fixed<a, b>`` string for an ATLAS or rule4ml precision.

    ATLAS writes ``fixed<W,I,RND,WRAP,0>``; rule4ml accepts only
    ``ap_fixed<W, I>``. Unknown formats fall back to the default precision.
    """
    if not isinstance(precision, str):
        return _DEFAULT_PRECISION
    if precision.strip().startswith("ap_fixed<"):
        return precision.strip()
    match = _ATLAS_FIXED.match(precision)
    if match is None:
        return _DEFAULT_PRECISION
    return f"ap_fixed<{match.group(1)}, {match.group(2)}>"

_WRAPPER = None
_KERAS_MODELS = {}
_PREDICTIONS = {}


def _wrapper():
    global _WRAPPER
    if _WRAPPER is None:
        try:
            from rule4ml.models.wrappers import MultiModelWrapper
        except ImportError as error:
            raise UnsupportedEvaluation(
                "the rule4ml evaluator requires the 'rule4ml' package (with "
                "TensorFlow and torch) in the CarbonPATH environment"
            ) from error
        wrapper = MultiModelWrapper()
        wrapper.load_default_models()
        _WRAPPER = wrapper
    return _WRAPPER


def _keras_model(activation, shape):
    key = (activation, tuple(shape))
    model = _KERAS_MODELS.get(key)
    if model is not None:
        return model
    import keras

    name = f"{activation}_{'x'.join(str(dimension) for dimension in shape)}"
    inputs = keras.Input(shape=tuple(shape), name=f"{name}_input")
    outputs = keras.layers.Activation(activation, name=name)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    _KERAS_MODELS[key] = model
    return model


def predict_activation(
    activation,
    shape,
    board,
    strategy,
    precision,
    reuse_factor,
    hls4ml_version,
    vivado_version,
):
    """Return ``{"cycles", "interval", "resources"}`` for one activation op.

    Results are cached per (activation, shape, board, settings) tuple so that an
    annealing campaign pays rule4ml inference once per distinct operation, not
    once per proposal.
    """
    key = (
        activation,
        tuple(shape),
        board,
        strategy,
        precision,
        reuse_factor,
        hls4ml_version,
        vivado_version,
    )
    cached = _PREDICTIONS.get(key)
    if cached is not None:
        return cached

    frame = _wrapper().predict(
        _keras_model(activation, shape),
        hls_configs=[
            {
                "board": board,
                "model": {
                    "strategy": strategy,
                    "precision": precision,
                    "reuse_factor": reuse_factor,
                },
            }
        ],
        hls4ml_versions=[hls4ml_version],
        vivado_versions=[vivado_version],
    )
    if frame is None or frame.empty:
        raise UnsupportedEvaluation(
            f"rule4ml produced no prediction for {activation} on board '{board}'"
        )
    row = frame.iloc[0]
    if "CYCLES" not in row:
        raise UnsupportedEvaluation(
            f"rule4ml prediction for {activation} is missing CYCLES"
        )
    result = {
        "cycles": float(row["CYCLES"]),
        "interval": float(row["INTERVAL"]) if "INTERVAL" in row else None,
        "resources": {
            label: float(row[label])
            for label in _RESOURCE_LABELS
            if label in row
        },
    }
    _PREDICTIONS[key] = result
    return result


@dataclass(frozen=True)
class Rule4MlActivationInput:
    operation_id: str
    element_count: int
    shape: tuple
    precision: str | None
    reuse_factor: int | None


class Rule4MlActivationInputAdapter(OperationInputAdapter):
    """Input adapter shared by the rule4ml activation evaluators."""

    operation_type = ""
    input_adapter_id = ""

    def build_input(self, operation, access_plan):
        if operation.operation_type != self.operation_type:
            raise UnsupportedEvaluation(
                f"{type(self).__name__} cannot read "
                f"'{operation.operation_type}' operation "
                f"'{operation.operation_id}'"
            )
        reuse_factor = operation.source.attribute("reuse_factor", None)
        if reuse_factor is not None and (
            isinstance(reuse_factor, bool)
            or not isinstance(reuse_factor, int)
            or reuse_factor <= 0
        ):
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: ATLAS attribute 'reuse_factor' "
                f"must be a positive integer"
            )
        return Rule4MlActivationInput(
            operation_id=operation.operation_id,
            element_count=operation.element_count,
            shape=tuple(operation.output_tensor.shape),
            precision=operation.output_tensor.precision,
            reuse_factor=reuse_factor,
        )


class Rule4MlActivationEvaluator(OperationEvaluator):
    """rule4ml-backed evaluator shared by the activation operation types."""

    operation_type = ""
    evaluator_id = ""
    input_adapter_id = ""
    activation = ""

    def evaluate(self, evaluator_input, assignment, context) -> OperationEstimate:
        fpga = context.system.fpga_chiplet_dict.get(assignment.endpoint_id)
        if fpga is None:
            raise UnsupportedEvaluation(
                f"no FPGA endpoint {assignment.endpoint_id} for "
                f"'{evaluator_input.operation_id}'"
            )
        if fpga.frequency_hz <= 0:
            raise UnsupportedEvaluation(
                f"{evaluator_input.operation_id}: fpga frequency must be positive"
            )

        settings = dict(context.evaluator_settings or {})
        unknown = sorted(set(settings) - _SUPPORTED_SETTINGS)
        if unknown:
            raise UnsupportedEvaluation(
                "unsupported rule4ml evaluator setting(s): " + ", ".join(unknown)
            )
        strategy = settings.get("strategy", "latency")
        if strategy not in ("latency", "resource"):
            raise UnsupportedEvaluation(
                "rule4ml strategy must be 'latency' or 'resource'"
            )
        latency_metric = settings.get("latency_metric", "cycles")
        if latency_metric not in ("cycles", "interval"):
            raise UnsupportedEvaluation(
                "rule4ml latency_metric must be 'cycles' or 'interval'"
            )

        precision = rule4ml_precision(
            settings.get("precision") or evaluator_input.precision
        )
        reuse_factor = settings.get("reuse_factor")
        if reuse_factor is None:
            reuse_factor = evaluator_input.reuse_factor or 1
        if (
            isinstance(reuse_factor, bool)
            or not isinstance(reuse_factor, int)
            or reuse_factor <= 0
        ):
            raise UnsupportedEvaluation(
                "rule4ml reuse_factor must be a positive integer"
            )

        board = settings.get("board") or nearest_board(fpga)
        prediction = predict_activation(
            activation=self.activation,
            shape=evaluator_input.shape,
            board=board,
            strategy=strategy,
            precision=precision,
            reuse_factor=reuse_factor,
            hls4ml_version=settings.get("hls4ml_version", _DEFAULT_HLS4ML_VERSION),
            vivado_version=settings.get("vivado_version", _DEFAULT_VIVADO_VERSION),
        )
        cycles = prediction[latency_metric]
        if cycles is None:
            raise UnsupportedEvaluation(
                f"rule4ml has no {latency_metric} estimate for "
                f"'{evaluator_input.operation_id}'"
            )
        compute_latency_ns = cycles / fpga.frequency_hz * 1e9
        dynamic_energy_pj = evaluator_input.element_count * float(
            settings.get("energy_per_element_pj", 0.0)
        )
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=compute_latency_ns,
            dynamic_energy_pj=dynamic_energy_pj,
        )


class Rule4MlReluInputAdapter(Rule4MlActivationInputAdapter):
    operation_type = "relu"
    input_adapter_id = "rule4ml_relu_input_v1"


class Rule4MlReluEvaluator(Rule4MlActivationEvaluator):
    operation_type = "relu"
    evaluator_id = "rule4ml_fpga_relu_v1"
    input_adapter_id = "rule4ml_relu_input_v1"
    activation = "relu"


class Rule4MlSoftmaxInputAdapter(Rule4MlActivationInputAdapter):
    operation_type = "softmax"
    input_adapter_id = "rule4ml_softmax_input_v1"


class Rule4MlSoftmaxEvaluator(Rule4MlActivationEvaluator):
    operation_type = "softmax"
    evaluator_id = "rule4ml_fpga_softmax_v1"
    input_adapter_id = "rule4ml_softmax_input_v1"
    activation = "softmax"


def register_rule4ml_evaluators(registry):
    """Register the rule4ml activation evaluators into ``registry``."""
    if registry is None:
        raise ValueError("register_rule4ml_evaluators requires an evaluator registry")
    registry.register(Rule4MlReluEvaluator(), Rule4MlReluInputAdapter())
    registry.register(Rule4MlSoftmaxEvaluator(), Rule4MlSoftmaxInputAdapter())
    # Imported here because the hybrid module imports this one.
    from system.utils.Rule4MlHybridEvaluators import (
        register_rule4ml_hybrid_evaluators,
    )

    register_rule4ml_hybrid_evaluators(registry)
    return registry
