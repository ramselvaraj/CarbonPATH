import dataclasses
import unittest

from system.utils.AtlasGraphAdapter import parse_atlas_graph
from system.utils.FpgaChiplet import (
    FpgaChiplet,
    FpgaReluImplementation,
    FpgaSoftmaxImplementation,
)
from system.utils.OperationEvaluator import EvaluatorRegistry
from system.utils.OperationAssignment import EndpointAssignment
from system.utils.Rule4MlEvaluators import (
    Rule4MlSoftmaxInputAdapter,
    register_rule4ml_evaluators,
)
from system.utils.Rule4MlHybridEvaluators import (
    Rule4MlHybridReluEvaluator,
    Rule4MlHybridSoftmaxEvaluator,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from tests.test_rule4ml_evaluator import (
    _context,
    _fpga,
    _patched_prediction,
    _relu_view,
)

FREQUENCY_HZ = 300000000
# 32 elements x 8 bits = 256 bits in one 18 Kbit block read at 36 bits per cycle.
READ_CYCLES = 8


def _softmax_view():
    nodes = [
        {
            "name": "input_layer",
            "class": "Input",
            "inputs": [],
            "output": {"name": "input_layer", "shape": [4, 8]},
        },
        {
            "name": "softmax_1",
            "class": "Softmax",
            "inputs": [{"name": "input_layer", "shape": [4, 8]}],
            "output": {"name": "softmax_1", "shape": [4, 8]},
            "attrs": {},
        },
    ]
    operation = parse_atlas_graph(nodes, "graph").operations[0]
    return Rule4MlSoftmaxInputAdapter().build_input(operation, None)


def _fpga_with(relu_energy=None, relu_lanes=None, clbs_per_lane=5, **kwargs):
    fpga = _fpga(frequency_hz=FREQUENCY_HZ)
    return dataclasses.replace(
        fpga,
        relu_implementation=FpgaReluImplementation(
            clbs_per_lane=clbs_per_lane,
            max_parallel_lanes=relu_lanes,
            energy_per_element_pj=relu_energy,
        ),
        **kwargs,
    )


def _relu(fpga, settings=None, cycles=100.0):
    with _patched_prediction(cycles=cycles):
        return Rule4MlHybridReluEvaluator().evaluate(
            _relu_view(), EndpointAssignment(1, "fpga"), _context(fpga, settings)
        )


class HybridReluTests(unittest.TestCase):
    def test_energy_comes_from_the_architecture_relu_option(self):
        estimate = _relu(_fpga_with(relu_energy=0.5))
        self.assertEqual(estimate.evaluator_id, "rule4ml_hybrid_relu_v1")
        self.assertAlmostEqual(estimate.dynamic_energy_pj, 32 * 0.5)

    def test_null_energy_uses_the_setting_default(self):
        self.assertAlmostEqual(_relu(_fpga_with()).dynamic_energy_pj, 32 * 0.1)

    def test_null_energy_setting_is_overridable(self):
        estimate = _relu(_fpga_with(), {"null_energy_per_element_pj": 0.25})
        self.assertAlmostEqual(estimate.dynamic_energy_pj, 32 * 0.25)

    def test_rule4ml_latency_wins_when_it_is_larger(self):
        estimate = _relu(_fpga_with(relu_lanes=128), cycles=100.0)
        self.assertAlmostEqual(estimate.compute_latency_ns, (100.0 + READ_CYCLES) / FREQUENCY_HZ * 1e9)

    def test_floor_replaces_a_zero_cycle_prediction(self):
        # 32 elements / 4 lanes = 8 cycles from the architecture. Eight blocks
        # read the input in one cycle, so the floor is the larger term.
        estimate = _relu(
            _fpga_with(relu_lanes=4, input_buffer_blocks=8), cycles=0.0
        )
        self.assertAlmostEqual(estimate.compute_latency_ns, 8 / FREQUENCY_HZ * 1e9)

    def test_floor_none_keeps_the_zero_cycle_prediction(self):
        estimate = _relu(_fpga_with(relu_lanes=4), {"latency_floor": "none"}, cycles=0.0)
        self.assertAlmostEqual(
            estimate.compute_latency_ns, READ_CYCLES / FREQUENCY_HZ * 1e9
        )

    def test_more_lanes_lower_the_floor(self):
        slow = _relu(_fpga_with(relu_lanes=2), cycles=0.0).compute_latency_ns
        fast = _relu(_fpga_with(relu_lanes=16), cycles=0.0).compute_latency_ns
        self.assertGreater(slow, fast)

    def test_invalid_floor_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            _relu(_fpga_with(), {"latency_floor": "bogus"})

    def test_negative_null_energy_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            _relu(_fpga_with(), {"null_energy_per_element_pj": -1})

    def test_rule4ml_settings_still_pass_through(self):
        with _patched_prediction(cycles=100.0, interval=10.0):
            estimate = Rule4MlHybridReluEvaluator().evaluate(
                _relu_view(),
                EndpointAssignment(1, "fpga"),
                _context(
                    _fpga_with(relu_lanes=128),
                    {"latency_metric": "interval", "latency_floor": "none"},
                ),
            )
        self.assertAlmostEqual(estimate.compute_latency_ns, (10.0 + READ_CYCLES) / FREQUENCY_HZ * 1e9)

    def test_unknown_setting_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            _relu(_fpga_with(), {"not_a_setting": 1})


class HybridSoftmaxTests(unittest.TestCase):
    def _softmax(self, fpga, settings=None, cycles=100.0):
        with _patched_prediction(cycles=cycles):
            return Rule4MlHybridSoftmaxEvaluator().evaluate(
                _softmax_view(), EndpointAssignment(1, "fpga"), _context(fpga, settings)
            )

    def test_latency_is_rule4ml_and_energy_is_per_element(self):
        estimate = self._softmax(_fpga_with())
        self.assertEqual(estimate.evaluator_id, "rule4ml_hybrid_softmax_v1")
        self.assertAlmostEqual(estimate.compute_latency_ns, (100.0 + READ_CYCLES) / FREQUENCY_HZ * 1e9)
        # The default matches the placeholder softmax: 2.0 pJ per element.
        self.assertAlmostEqual(estimate.dynamic_energy_pj, 32 * 2.0)

    def test_energy_follows_the_architecture_softmax_option(self):
        fpga = _fpga_with(softmax_implementation=FpgaSoftmaxImplementation(3.5))
        self.assertAlmostEqual(self._softmax(fpga).dynamic_energy_pj, 32 * 3.5)

    def test_unknown_setting_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            self._softmax(_fpga_with(), {"latency_floor": "none"})


class SoftmaxImplementationSchemaTests(unittest.TestCase):
    @staticmethod
    def _fpga_dict(**extra):
        value = {
            "chiplet_type": "fpga",
            "tech_node": "7",
            "area": 10.0,
            "power": 2.0,
            "frequency_hz": 400000000,
            "clbs": 8000,
            "brams": 100,
            "dsps": 250,
            "relu_implementation": {"clbs_per_lane": 2},
        }
        value.update(extra)
        return value

    def test_default_matches_the_placeholder(self):
        chiplet = FpgaChiplet.from_dict("Chiplet_1", self._fpga_dict())
        self.assertEqual(chiplet.softmax_implementation.energy_per_element_pj, 2.0)

    def test_value_is_read_from_the_architecture(self):
        chiplet = FpgaChiplet.from_dict(
            "Chiplet_1",
            self._fpga_dict(softmax_implementation={"energy_per_element_pj": 1.25}),
        )
        self.assertEqual(chiplet.softmax_implementation.energy_per_element_pj, 1.25)

    def test_invalid_values_are_rejected(self):
        for bad in ({"energy_per_element_pj": -1}, {"unknown": 1}, "x"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    FpgaChiplet.from_dict(
                        "Chiplet_1", self._fpga_dict(softmax_implementation=bad)
                    )


class InputBufferSchemaTests(unittest.TestCase):
    def test_default_is_sized_per_operation(self):
        chiplet = FpgaChiplet.from_dict(
            "Chiplet_1", SoftmaxImplementationSchemaTests._fpga_dict()
        )
        self.assertIsNone(chiplet.input_buffer_blocks)

    def test_value_is_read_from_the_architecture(self):
        chiplet = FpgaChiplet.from_dict(
            "Chiplet_1",
            SoftmaxImplementationSchemaTests._fpga_dict(input_buffer_blocks=4),
        )
        self.assertEqual(chiplet.input_buffer_blocks, 4)

    def test_invalid_values_are_rejected(self):
        for bad in (0, -1, 1.5, True, "4"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    FpgaChiplet.from_dict(
                        "Chiplet_1",
                        SoftmaxImplementationSchemaTests._fpga_dict(
                            input_buffer_blocks=bad
                        ),
                    )


class HybridRegistrationTests(unittest.TestCase):
    def test_registers_both_hybrid_evaluators(self):
        registry = EvaluatorRegistry()
        register_rule4ml_evaluators(registry)
        self.assertIn("rule4ml_hybrid_relu_v1", registry)
        self.assertIn("rule4ml_hybrid_softmax_v1", registry)
        self.assertIn("rule4ml_fpga_relu_v1", registry)


if __name__ == "__main__":
    unittest.main()
