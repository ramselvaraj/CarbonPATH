import os
import unittest
from types import SimpleNamespace
from unittest import mock

from system.utils.AtlasGraphAdapter import parse_atlas_graph
from system.utils.FpgaChiplet import FpgaChiplet, FpgaReluImplementation
from system.utils.NeurigamiEvaluators import (
    neurigami_evaluator,
    neurigami_input_adapter,
    register_neurigami_evaluators,
)
from system.utils.OperationEvaluator import EvaluationContext, EvaluatorRegistry
from system.utils.UnsupportedEvaluation import ResourceExceeded, UnsupportedEvaluation

# 300 MHz -> 3.333 ns; 8 bits x 32 elements = 256 bits -> one block read in 8 cycles.
READ_CYCLES = 8


def _fpga(brams=200, dsps=500, clbs=10000, frequency_hz=300000000, fpga_part="xczu7ev", board=None,
          max_utilization=None, luts=None, ffs=None, calibration_part=None):
    return FpgaChiplet(
        id=1, name="Chiplet_2", frequency_hz=frequency_hz, clbs=clbs, brams=brams,
        dsps=dsps, area=10.0, power=2.0, node=7,
        relu_implementation=FpgaReluImplementation(clbs_per_lane=5),
        fpga_part=fpga_part, board=board, max_utilization=max_utilization,
        luts=luts, ffs=ffs, calibration_part=calibration_part,
    )


def _operation(operation_type="relu", shape=(4, 8), attrs=None):
    nodes = [
        {"name": "input_layer", "class": "Input", "inputs": [],
         "output": {"name": "input_layer", "shape": list(shape)}},
        {"name": f"{operation_type}_1", "class": "Activation",
         "inputs": [{"name": "input_layer", "shape": list(shape)}],
         "output": {"name": f"{operation_type}_1", "shape": list(shape)},
         "attrs": {"activation": operation_type, **(attrs or {})}},
    ]
    return parse_atlas_graph(nodes, "graph").operations[0]


def _context(fpga, settings=None):
    return EvaluationContext(
        cache=None, architecture={},
        system=SimpleNamespace(fpga_chiplet_dict={1: fpga}),
        evaluator_settings=settings or {},
    )


def _result(**over):
    base = dict(elements=32, rows=1, lut=5.0, ff=10.0, bram36=0.0, dsp=0.0, uram=0.0,
                interval=32, first_latency=33, clock_ns=3.33, part="xczu7ev")
    base.update(over)
    return base


def _patched(**over):
    return mock.patch(
        "system.utils.NeurigamiRunner.estimate_single_layer", return_value=_result(**over)
    )


def _evaluate(fpga=None, settings=None, operation=None, op="relu"):
    fpga = fpga or _fpga()
    operation = operation or _operation()
    adapter = neurigami_input_adapter(op)
    evaluator = neurigami_evaluator(op)
    view = adapter.build_input(operation, None)
    return evaluator.evaluate(view, SimpleNamespace(endpoint_id=1), _context(fpga, settings))


class NeurigamiEvaluatorTest(unittest.TestCase):
    def test_the_single_layer_is_the_size_of_the_operation(self):
        with _patched() as run:
            _evaluate()
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["op"], "relu")
        self.assertEqual(kwargs["elements"], 32)          # 4 x 8 as one layer
        self.assertAlmostEqual(kwargs["clock_ns"], 1e9 / 300000000)

    def test_latency_is_first_latency_plus_buffer_read_at_the_fpga_clock(self):
        with _patched():
            estimate = _evaluate()
        self.assertAlmostEqual(estimate.compute_latency_ns, (33 + READ_CYCLES) * 1e9 / 300000000)

    def test_interval_metric_uses_the_interval(self):
        with _patched():
            estimate = _evaluate(settings={"latency_metric": "interval"})
        self.assertAlmostEqual(estimate.compute_latency_ns, (32 + READ_CYCLES) * 1e9 / 300000000)

    def test_resources_are_reported_with_bram36_as_two_units(self):
        with _patched(bram36=3.0, lut=40.0, ff=80.0, dsp=2.0):
            estimate = _evaluate()
        use = estimate.chiplet_use[0]
        self.assertEqual(use.kernel, {"LUT": 40.0, "FF": 80.0, "BRAM": 6.0, "DSP": 2.0})
        self.assertEqual(use.buffer_blocks, 1)

    def test_a_design_larger_than_the_chiplet_is_rejected(self):
        with _patched(dsp=900.0), self.assertRaises(ResourceExceeded):
            _evaluate(fpga=_fpga(dsps=500))

    def test_uram_is_unsupported(self):
        with _patched(uram=1.0), self.assertRaises(UnsupportedEvaluation):
            _evaluate()

    def test_unknown_setting_and_metric_are_rejected(self):
        with _patched():
            with self.assertRaises(UnsupportedEvaluation):
                _evaluate(settings={"strategy": "latency"})
            with self.assertRaises(UnsupportedEvaluation):
                _evaluate(settings={"latency_metric": "cycles"})

    def test_energy_is_per_element(self):
        with _patched():
            estimate = _evaluate(settings={"energy_per_element_pj": 0.5})
        self.assertEqual(estimate.dynamic_energy_pj, 16.0)

    def test_leaky_slope_and_threshold_come_from_activ_param(self):
        for atlas_name, op, default in (("leakyrelu", "leaky_relu", 0.01), ("thresholdedrelu", "thresholded_relu", 1.0)):
            for attrs, expected in (({"activ_param": 0.125}, 0.125), ({}, default)):
                node_class = {"leaky_relu": "ParametrizedActivation", "thresholded_relu": "ParametrizedActivation"}[op]
                nodes = [
                    {"name": "input_layer", "class": "Input", "inputs": [], "output": {"name": "input_layer", "shape": [4, 8]}},
                    {"name": "a", "class": node_class, "inputs": [{"name": "input_layer", "shape": [4, 8]}],
                     "output": {"name": "a", "shape": [4, 8]}, "attrs": {"activation": atlas_name, **attrs}},
                ]
                operation = parse_atlas_graph(nodes, "graph").operations[0]
                with _patched() as run:
                    _evaluate(operation=operation, op=op)
                self.assertEqual(run.call_args.kwargs["param"], expected)
                self.assertEqual(run.call_args.kwargs["op"], op)

    def test_atlas_hard_and_clipped_activations_are_parsed_and_estimated(self):
        cases = (
            ("HardActivation", {"activation": "hard_sigmoid", "slope": 0.2, "shift": 0.5}, "hard_sigmoid", None),
            ("HardActivation", {"activation": "hard_tanh", "slope": 0.5, "shift": 0.5}, "hard_tanh", None),
            ("ParametrizedActivation", {"activation": "clippedrelu", "activ_param": 3.0}, "clipped_relu", 3.0),
            ("ParametrizedActivation", {"activation": "clippedrelu"}, "clipped_relu", 6.0),
        )
        for node_class, attrs, op, param in cases:
            with self.subTest(op=op, attrs=attrs):
                nodes = [
                    {"name": "input_layer", "class": "Input", "inputs": [], "output": {"name": "input_layer", "shape": [4, 8]}},
                    {"name": "a", "class": node_class, "inputs": [{"name": "input_layer", "shape": [4, 8]}],
                     "output": {"name": "a", "shape": [4, 8]}, "attrs": attrs},
                ]
                operation = parse_atlas_graph(nodes, "graph").operations[0]
                self.assertEqual(operation.operation_type, op)
                with _patched() as run:
                    _evaluate(operation=operation, op=op)
                self.assertEqual(run.call_args.kwargs["op"], op)
                self.assertEqual(run.call_args.kwargs["param"], param)

    def test_an_unknown_hard_activation_is_refused(self):
        nodes = [
            {"name": "input_layer", "class": "Input", "inputs": [], "output": {"name": "input_layer", "shape": [4, 8]}},
            {"name": "a", "class": "HardActivation", "inputs": [{"name": "input_layer", "shape": [4, 8]}],
             "output": {"name": "a", "shape": [4, 8]}, "attrs": {"activation": "hard_swish"}},
        ]
        with self.assertRaisesRegex(UnsupportedEvaluation, "unsupported activation"):
            parse_atlas_graph(nodes, "graph")

    def test_softmax_is_built_for_one_row_and_the_rest_stream_through(self):
        nodes = [
            {"name": "input_layer", "class": "Input", "inputs": [], "output": {"name": "input_layer", "shape": [8, 16]}},
            {"name": "sm", "class": "Softmax", "inputs": [{"name": "input_layer", "shape": [8, 16]}],
             "output": {"name": "sm", "shape": [8, 16]}},
        ]
        operation = parse_atlas_graph(nodes, "graph").operations[0]
        with _patched(rows=8, interval=1, first_latency=38) as run:
            estimate = _evaluate(operation=operation, op="softmax")
        self.assertEqual(run.call_args.kwargs["vector"], 16)
        self.assertEqual(run.call_args.kwargs["elements"], 128)
        self.assertAlmostEqual(estimate.compute_latency_ns, (38 + 7 * 1 + 29) * 1e9 / 300000000)   # 1024 bits at 36 bits a cycle: 29 read cycles

    def test_the_estimate_is_for_the_chiplets_own_part(self):
        with _patched() as run:
            _evaluate(fpga=_fpga(fpga_part="xc7z020"))
        self.assertEqual(run.call_args.kwargs["part"], "xc7z020")
        with _patched() as run:                      # the evaluator setting still overrides
            _evaluate(fpga=_fpga(fpga_part="xc7z020"), settings={"part": "xck26"})
        self.assertEqual(run.call_args.kwargs["part"], "xck26")

    def test_a_board_chiplet_uses_its_boards_part_and_an_uncalibrated_one_uses_the_nearest(self):
        with _patched() as run:
            _evaluate(fpga=_fpga(fpga_part=None, board="pynq-z2"))
        self.assertEqual(run.call_args.kwargs["part"], "xc7z020")
        with _patched() as run:                      # zcu102 is xczu9eg: not calibrated, so the nearest one
            _evaluate(fpga=_fpga(fpga_part=None, board="zcu102"))
        self.assertEqual(run.call_args.kwargs["part"], "xczu3eg")   # nearest to the test chiplet's counts

    def test_a_free_spec_uses_the_nearest_part_or_its_calibration_part(self):
        with _patched() as run:                      # the size of xc7z020
            _evaluate(fpga=_fpga(fpga_part=None, luts=53200, ffs=106400, clbs=6650, brams=280, dsps=220))
        self.assertEqual(run.call_args.kwargs["part"], "xc7z020")
        with _patched() as run:
            _evaluate(fpga=_fpga(fpga_part=None, calibration_part="xck26"))
        self.assertEqual(run.call_args.kwargs["part"], "xck26")

    def test_the_usable_share_shrinks_the_budget(self):
        with _patched() as run:
            _evaluate(fpga=_fpga(max_utilization={"lut": 0.5, "dsp": 0.5}))
        budget = run.call_args.kwargs["budget"]
        self.assertEqual(budget["dsp"], 250)
        self.assertEqual(budget["lut"], 10000 * 8 * 0.5)

    def test_the_chiplet_is_the_budget(self):
        with _patched() as run:
            _evaluate(fpga=_fpga(brams=200, dsps=500, clbs=10000))
        budget = run.call_args.kwargs["budget"]
        self.assertEqual(budget["dsp"], 500)
        self.assertEqual(budget["lut"], 10000 * 8)
        self.assertEqual(budget["bram36"], (200 - 1) / 2)       # the input buffer's block comes off first

    def test_no_design_that_fits_the_budget_is_a_resource_rejection(self):
        from system.utils import NeurigamiRunner
        NeurigamiRunner._MEMORY.clear()
        with mock.patch("system.utils.NeurigamiRunner.subprocess.run") as run, \
                mock.patch("system.utils.NeurigamiRunner.locate", return_value=("h", "p")):
            run.return_value = SimpleNamespace(returncode=0, stdout='{"no_fit": "LUT needs 9 > limit 1"}\n', stderr="")
            with self.assertRaises(ResourceExceeded):
                NeurigamiRunner.estimate_single_layer("softmax", 16, 4.0, cache_dir=None)
        NeurigamiRunner._MEMORY.clear()

    def test_the_adapter_rejects_another_operation_type(self):
        with self.assertRaises(UnsupportedEvaluation):
            neurigami_input_adapter("relu").build_input(_operation("softmax"), None)

    def test_every_supported_operation_is_registered(self):
        registry = register_neurigami_evaluators(EvaluatorRegistry())
        for op in ("relu", "leaky_relu", "thresholded_relu", "sigmoid", "tanh", "softmax",
                   "clipped_relu", "hard_sigmoid", "hard_tanh"):
            self.assertIn(f"neurigami_fpga_{op}_v1", registry)
        with self.assertRaises(ValueError):
            register_neurigami_evaluators(None)


@unittest.skipUnless(os.environ.get("NEURIGAMI_HOME"), "set NEURIGAMI_HOME (and NEURIGAMI_PYTHON) to run")
class NeurigamiRealBuildTest(unittest.TestCase):
    def _run(self, op, elements, **kwargs):
        import tempfile
        from system.utils import NeurigamiRunner
        with tempfile.TemporaryDirectory() as cache:
            return NeurigamiRunner.estimate_single_layer(op, elements, 4.0, cache_dir=cache, **kwargs)

    def test_a_real_single_relu_build(self):
        result = self._run("relu", 64)
        self.assertEqual(result["interval"], 64)
        self.assertEqual(result["first_latency"], 65)
        self.assertEqual(result["dsp"], 0)

    def test_every_activation_builds(self):
        for op in ("leaky_relu", "thresholded_relu", "sigmoid", "tanh", "hard_sigmoid", "hard_tanh", "clipped_relu"):
            with self.subTest(op=op):
                result = self._run(op, 128, param=0.125)
                self.assertEqual(result["interval"], 128)
                self.assertGreater(result["bram36"], 0)          # the 256-entry table

    def test_a_real_softmax_and_a_budget_that_cannot_hold_it(self):
        result = self._run("softmax", 128, vector=16)
        self.assertEqual(result["rows"], 8)
        with self.assertRaises(ResourceExceeded):
            self._run("softmax", 128, vector=16, budget={"lut": 100, "ff": 100, "bram36": 1, "dsp": 0})


if __name__ == "__main__":
    unittest.main()
