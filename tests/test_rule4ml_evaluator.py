import unittest
from types import SimpleNamespace
from unittest import mock

from system.utils.AtlasGraphAdapter import parse_atlas_graph
from system.utils.FpgaChiplet import FpgaChiplet, FpgaReluImplementation
from system.utils.OperationEvaluator import EvaluationContext, EvaluatorRegistry
from system.utils.OperationAssignment import EndpointAssignment
from system.utils.Rule4MlBoards import fpga_resource_vector, nearest_board
from system.utils.Rule4MlEvaluators import (
    Rule4MlActivationInput,
    Rule4MlReluEvaluator,
    Rule4MlReluInputAdapter,
    Rule4MlSoftmaxEvaluator,
    register_rule4ml_evaluators,
)
from system.utils.UnsupportedEvaluation import (
    ResourceExceeded,
    UnsupportedEvaluation,
)


BOARDS = {
    "pynq-z2": {
        "max_bram": 280,
        "max_dsp": 220,
        "max_ff": 106400,
        "max_lut": 53200,
    },
    "zcu102": {
        "max_bram": 1824,
        "max_dsp": 2520,
        "max_ff": 548160,
        "max_lut": 274080,
    },
    "alveo-u200": {
        "max_bram": 4320,
        "max_dsp": 6840,
        "max_ff": 2364480,
        "max_lut": 1182240,
    },
    "alveo-u250": {
        "max_bram": 5376,
        "max_dsp": 12288,
        "max_ff": 3456000,
        "max_lut": 1728000,
    },
}


# 32 elements x 8 bits = 256 bits in one 18 Kbit block read at 36 bits per cycle.
READ_CYCLES = 8


def _fpga(
    clbs=10000,
    brams=200,
    dsps=500,
    luts=None,
    ffs=None,
    frequency_hz=300000000,
    input_buffer_blocks=None,
):
    return FpgaChiplet(
        id=1,
        name="Chiplet_2",
        frequency_hz=frequency_hz,
        clbs=clbs,
        brams=brams,
        dsps=dsps,
        area=10.0,
        power=2.0,
        node=7,
        relu_implementation=FpgaReluImplementation(clbs_per_lane=5),
        luts=luts,
        ffs=ffs,
        input_buffer_blocks=input_buffer_blocks,
    )


def _relu_operation(reuse_factor=None):
    attrs = {"activation": "relu"}
    if reuse_factor is not None:
        attrs["reuse_factor"] = reuse_factor
    nodes = [
        {
            "name": "input_layer",
            "class": "Input",
            "inputs": [],
            "output": {"name": "input_layer", "shape": [4, 8]},
        },
        {
            "name": "relu_1",
            "class": "Activation",
            "inputs": [{"name": "input_layer", "shape": [4, 8]}],
            "output": {"name": "relu_1", "shape": [4, 8]},
            "attrs": attrs,
        },
    ]
    return parse_atlas_graph(nodes, "graph").operations[0]


def _relu_view(reuse_factor=None):
    return Rule4MlReluInputAdapter().build_input(
        _relu_operation(reuse_factor=reuse_factor), None
    )


def _context(fpga, settings=None):
    return EvaluationContext(
        cache=None,
        architecture={},
        system=SimpleNamespace(fpga_chiplet_dict={1: fpga}),
        evaluator_settings=settings or {},
    )


def _patched_prediction(cycles=100.0, interval=10.0):
    return mock.patch(
        "system.utils.Rule4MlEvaluators.predict_activation",
        return_value={
            "cycles": cycles,
            "interval": interval,
            "resources": {"LUT": 10.0},
        },
    )


class BoardMappingTests(unittest.TestCase):
    def test_small_architecture_maps_to_pynq(self):
        self.assertEqual(nearest_board(_fpga(), boards=BOARDS), "pynq-z2")

    def test_mid_architecture_maps_to_zcu102(self):
        fpga = _fpga(clbs=10000, brams=900, dsps=1500, luts=500000, ffs=1000000)
        self.assertEqual(nearest_board(fpga, boards=BOARDS), "zcu102")

    def test_large_architecture_maps_to_u250(self):
        fpga = _fpga(clbs=10000, brams=5000, dsps=10000, luts=1600000, ffs=3200000)
        self.assertEqual(nearest_board(fpga, boards=BOARDS), "alveo-u250")

    def test_clbs_are_expanded_when_luts_absent(self):
        vector = fpga_resource_vector(_fpga(clbs=10000))
        self.assertEqual(vector["lut"], 80000)
        self.assertEqual(vector["ff"], 160000)

    def test_explicit_luts_and_ffs_win(self):
        vector = fpga_resource_vector(_fpga(clbs=10000, luts=1234, ffs=5678))
        self.assertEqual(vector["lut"], 1234)
        self.assertEqual(vector["ff"], 5678)


class FpgaChipletSchemaTests(unittest.TestCase):
    def _payload(self, **extra):
        payload = {
            "chiplet_type": "fpga",
            "tech_node": "7",
            "area": 10.0,
            "power": 2.0,
            "frequency_hz": 300000000,
            "clbs": 10000,
            "brams": 200,
            "dsps": 500,
            "relu_implementation": {"clbs_per_lane": 5},
        }
        payload.update(extra)
        return payload

    def test_from_dict_accepts_luts_and_ffs(self):
        chiplet = FpgaChiplet.from_dict(
            "Chiplet_2", self._payload(luts=90000, ffs=180000)
        )
        self.assertEqual(chiplet.luts, 90000)
        self.assertEqual(chiplet.ffs, 180000)

    def test_from_dict_defaults_luts_to_none(self):
        chiplet = FpgaChiplet.from_dict("Chiplet_2", self._payload())
        self.assertIsNone(chiplet.luts)
        self.assertIsNone(chiplet.ffs)


class Rule4MlReluEvaluatorTests(unittest.TestCase):
    def test_input_adapter_reads_shape_precision_and_reuse(self):
        view = _relu_view(reuse_factor=4)
        self.assertEqual(view.shape, (4, 8))
        self.assertEqual(view.element_count, 32)
        self.assertEqual(view.reuse_factor, 4)
        self.assertIsNone(view.precision)

    def test_latency_and_energy_from_rule4ml(self):
        with _patched_prediction(cycles=100.0) as predict:
            estimate = Rule4MlReluEvaluator().evaluate(
                _relu_view(),
                EndpointAssignment(1, "fpga"),
                _context(_fpga(), {"energy_per_element_pj": 0.5}),
            )
        self.assertEqual(estimate.evaluator_id, "rule4ml_fpga_relu_v1")
        self.assertAlmostEqual(estimate.compute_latency_ns, (100.0 + READ_CYCLES) / 300000000 * 1e9)
        self.assertAlmostEqual(estimate.dynamic_energy_pj, 32 * 0.5)
        _, kwargs = predict.call_args
        self.assertEqual(kwargs["board"], "pynq-z2")

    def test_board_setting_overrides_mapping(self):
        with _patched_prediction() as predict:
            Rule4MlReluEvaluator().evaluate(
                _relu_view(),
                EndpointAssignment(1, "fpga"),
                _context(_fpga(), {"board": "alveo-u250"}),
            )
        _, kwargs = predict.call_args
        self.assertEqual(kwargs["board"], "alveo-u250")

    def test_interval_metric_is_selectable(self):
        with _patched_prediction(cycles=100.0, interval=10.0):
            estimate = Rule4MlReluEvaluator().evaluate(
                _relu_view(),
                EndpointAssignment(1, "fpga"),
                _context(_fpga(), {"latency_metric": "interval"}),
            )
        self.assertAlmostEqual(estimate.compute_latency_ns, (10.0 + READ_CYCLES) / 300000000 * 1e9)

    def test_prediction_above_fpga_resources_is_rejected(self):
        # _fpga() has 10000 CLBs = 80000 LUTs and 200 BRAM.
        with mock.patch(
            "system.utils.Rule4MlEvaluators.predict_activation",
            return_value={
                "cycles": 1.0,
                "interval": 1.0,
                "resources": {"LUT": 80001.0, "BRAM": 10.0},
            },
        ):
            with self.assertRaisesRegex(ResourceExceeded, "LUT 80001 > 80000"):
                Rule4MlReluEvaluator().evaluate(
                    _relu_view(),
                    EndpointAssignment(1, "fpga"),
                    _context(_fpga()),
                )

    def test_prediction_at_fpga_resources_is_accepted(self):
        with mock.patch(
            "system.utils.Rule4MlEvaluators.predict_activation",
            return_value={
                "cycles": 1.0,
                "interval": 1.0,
                "resources": {"LUT": 80000.0, "BRAM": 199.0, "DSP": 500.0},
            },
        ):
            Rule4MlReluEvaluator().evaluate(
                _relu_view(),
                EndpointAssignment(1, "fpga"),
                _context(_fpga()),
            )

    def test_input_buffer_bram_counts_against_the_fpga(self):
        # The prediction needs 200 BRAM and the input buffer needs one more.
        with mock.patch(
            "system.utils.Rule4MlEvaluators.predict_activation",
            return_value={
                "cycles": 1.0,
                "interval": 1.0,
                "resources": {"BRAM": 200.0},
            },
        ):
            with self.assertRaisesRegex(ResourceExceeded, "BRAM 201 > 200"):
                Rule4MlReluEvaluator().evaluate(
                    _relu_view(),
                    EndpointAssignment(1, "fpga"),
                    _context(_fpga()),
                )

    def test_more_input_buffer_blocks_start_the_kernel_sooner(self):
        def latency(blocks):
            with _patched_prediction(cycles=0.0):
                return Rule4MlReluEvaluator().evaluate(
                    _relu_view(),
                    EndpointAssignment(1, "fpga"),
                    _context(_fpga(input_buffer_blocks=blocks)),
                ).compute_latency_ns

        self.assertAlmostEqual(latency(1), 8 / 300000000 * 1e9)
        self.assertAlmostEqual(latency(8), 1 / 300000000 * 1e9)

    def test_fpga_buffer_too_small_for_the_input_is_rejected(self):
        # 8192 elements x 8 bits = 65536 bits need four 18 Kbit blocks.
        view = Rule4MlActivationInput(
            operation_id="relu_1",
            element_count=8192,
            shape=(8192,),
            precision=None,
            reuse_factor=None,
        )
        with _patched_prediction(cycles=1.0):
            with self.assertRaisesRegex(ResourceExceeded, "needs at least 4"):
                Rule4MlReluEvaluator().evaluate(
                    view,
                    EndpointAssignment(1, "fpga"),
                    _context(_fpga(input_buffer_blocks=2)),
                )

    def test_unknown_setting_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            Rule4MlReluEvaluator().evaluate(
                _relu_view(),
                EndpointAssignment(1, "fpga"),
                _context(_fpga(), {"not_a_setting": 1}),
            )

    def test_missing_endpoint_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            Rule4MlReluEvaluator().evaluate(
                _relu_view(),
                EndpointAssignment(7, "fpga"),
                _context(_fpga()),
            )

    def test_reuse_factor_comes_from_atlas_attribute(self):
        with _patched_prediction() as predict:
            Rule4MlReluEvaluator().evaluate(
                _relu_view(reuse_factor=8),
                EndpointAssignment(1, "fpga"),
                _context(_fpga()),
            )
        _, kwargs = predict.call_args
        self.assertEqual(kwargs["reuse_factor"], 8)


class RegistrationTests(unittest.TestCase):
    def test_registers_both_activation_evaluators(self):
        registry = EvaluatorRegistry()
        register_rule4ml_evaluators(registry)
        self.assertIn("rule4ml_fpga_relu_v1", registry)
        self.assertIn("rule4ml_fpga_softmax_v1", registry)
        self.assertEqual(
            registry.get("rule4ml_fpga_relu_v1").operation_type, "relu"
        )
        self.assertEqual(
            registry.get("rule4ml_fpga_softmax_v1").operation_type, "softmax"
        )

    def test_requires_a_registry(self):
        with self.assertRaises(ValueError):
            register_rule4ml_evaluators(None)


if __name__ == "__main__":
    unittest.main()


class Rule4MlPrecisionTests(unittest.TestCase):
    def test_every_value_is_8_bit_fixed_point(self):
        from system.utils.Rule4MlEvaluators import rule4ml_precision

        # A declared 8-bit width keeps its integer bits.
        self.assertEqual(rule4ml_precision("fixed<8,2,RND,WRAP,0>"), "ap_fixed<8, 2>")
        self.assertEqual(rule4ml_precision("ufixed<8,2,RND,WRAP,0>"), "ap_fixed<8, 2>")
        self.assertEqual(rule4ml_precision("ap_fixed<8, 4>"), "ap_fixed<8, 4>")
        self.assertEqual(rule4ml_precision("int8"), "ap_fixed<8, 8>")
        # Any other width, or none, is quantized to the 8-bit default.
        for precision in ("fixed<9,3,RND,WRAP,0>", "ap_fixed<16, 6>", "?", None):
            self.assertEqual(rule4ml_precision(precision), "ap_fixed<8, 3>")
