"""Hybrid rule4ml activation evaluators.

Latency comes from rule4ml's predicted cycles. Dynamic energy follows one
rule for both operations: the work in the input (its element count) times the
energy per element of the implementation in the chosen architecture.

* ReLU energy:    ``elements x fpga.relu_implementation.energy_per_element_pj``.
  A ``null`` architecture value uses the ``null_energy_per_element_pj`` setting.
* Softmax energy: ``elements x fpga.softmax_implementation.energy_per_element_pj``.

Settings (ReLU only):

* ``latency_floor``               ``"legacy_lanes"`` (default) takes the larger of
                                  rule4ml's latency and the resource-derived lane
                                  latency, so a rule4ml prediction of zero cycles
                                  is not free; ``"none"`` uses rule4ml alone.
* ``null_energy_per_element_pj``  energy used when the architecture's ReLU energy
                                  is ``null``; default 0.1.

All other settings are the rule4ml evaluator's own and are passed through.
"""

import dataclasses

from system.utils.NonGemmEstimator import ReluComputeEstimator
from system.utils.OperationEvaluator import OperationEstimate
from system.utils.Rule4MlEvaluators import (
    Rule4MlActivationEvaluator,
    Rule4MlReluInputAdapter,
    Rule4MlSoftmaxInputAdapter,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation

_RELU_SETTINGS = frozenset({"latency_floor", "null_energy_per_element_pj"})
_LATENCY_FLOORS = ("none", "legacy_lanes")
_DEFAULT_LATENCY_FLOOR = "legacy_lanes"
_DEFAULT_NULL_RELU_ENERGY_PJ = 0.1


def _rule4ml_estimate(evaluator, evaluator_input, assignment, context, own_names):
    """Run the rule4ml evaluator with this evaluator's own settings removed."""
    settings = dict(context.evaluator_settings or {})
    own = {name: settings.pop(name) for name in list(settings) if name in own_names}
    stripped = dataclasses.replace(context, evaluator_settings=settings)
    estimate = Rule4MlActivationEvaluator.evaluate(
        evaluator, evaluator_input, assignment, stripped
    )
    fpga = context.system.fpga_chiplet_dict[assignment.endpoint_id]
    return estimate, own, fpga


def _nonnegative_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise UnsupportedEvaluation(f"{name} must be a non-negative number")
    return float(value)


class Rule4MlHybridReluEvaluator(Rule4MlActivationEvaluator):
    operation_type = "relu"
    evaluator_id = "rule4ml_hybrid_relu_v1"
    input_adapter_id = "rule4ml_relu_input_v1"
    activation = "relu"

    def evaluate(self, evaluator_input, assignment, context) -> OperationEstimate:
        estimate, own, fpga = _rule4ml_estimate(
            self, evaluator_input, assignment, context, _RELU_SETTINGS
        )
        floor = own.get("latency_floor", _DEFAULT_LATENCY_FLOOR)
        if floor not in _LATENCY_FLOORS:
            raise UnsupportedEvaluation(
                f"latency_floor must be one of {', '.join(_LATENCY_FLOORS)}"
            )
        null_energy = _nonnegative_number(
            "null_energy_per_element_pj",
            own.get("null_energy_per_element_pj", _DEFAULT_NULL_RELU_ENERGY_PJ),
        )

        latency_ns = estimate.compute_latency_ns
        if floor == "legacy_lanes":
            legacy = ReluComputeEstimator().estimate(
                evaluator_input.element_count, fpga, evaluator_input.operation_id
            )
            if not legacy.feasible:
                raise UnsupportedEvaluation(
                    f"{evaluator_input.operation_id}: {legacy.infeasibility_reason}"
                )
            latency_ns = max(latency_ns, legacy.compute_latency_ns)

        per_element = fpga.relu_implementation.energy_per_element_pj
        if per_element is None:
            per_element = null_energy
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=latency_ns,
            dynamic_energy_pj=evaluator_input.element_count * per_element,
        )


class Rule4MlHybridSoftmaxEvaluator(Rule4MlActivationEvaluator):
    operation_type = "softmax"
    evaluator_id = "rule4ml_hybrid_softmax_v1"
    input_adapter_id = "rule4ml_softmax_input_v1"
    activation = "softmax"

    def evaluate(self, evaluator_input, assignment, context) -> OperationEstimate:
        estimate, _, fpga = _rule4ml_estimate(
            self, evaluator_input, assignment, context, frozenset()
        )
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=estimate.compute_latency_ns,
            dynamic_energy_pj=evaluator_input.element_count
            * fpga.softmax_implementation.energy_per_element_pj,
        )


def register_rule4ml_hybrid_evaluators(registry):
    """Register the hybrid rule4ml activation evaluators into ``registry``."""
    if registry is None:
        raise ValueError(
            "register_rule4ml_hybrid_evaluators requires an evaluator registry"
        )
    registry.register(Rule4MlHybridReluEvaluator(), Rule4MlReluInputAdapter())
    registry.register(Rule4MlHybridSoftmaxEvaluator(), Rule4MlSoftmaxInputAdapter())
    return registry
