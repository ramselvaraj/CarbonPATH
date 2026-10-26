import math
import unittest
from types import SimpleNamespace

from system.utils.AtlasGraphAdapter import FPGA_OPERATIONS, parse_atlas_graph
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.FpgaChiplet import (
    FpgaChiplet,
    FpgaLayerNormImplementation,
    FpgaReluImplementation,
)
from system.utils.FpgaLayerNorm import (
    LayerNormEvaluator,
    LayerNormInputAdapter,
    layernorm_cycles,
    layernorm_depth_cycles,
    register_layernorm_evaluators,
)
from system.utils.OperationAssignment import EndpointAssignment
from system.utils.OperationEvaluator import EvaluationContext, EvaluatorRegistry
from system.utils.UnsupportedEvaluation import (
    ResourceExceeded,
    UnsupportedEvaluation,
)

FREQUENCY_HZ = 300000000


def _fpga(energy=None, brams=200, parameter_source="dram", input_buffer_blocks=None):
    kwargs = {}
    if energy is not None:
        kwargs["layernorm_implementation"] = FpgaLayerNormImplementation(energy)
    return FpgaChiplet(
        id=1,
        name="Chiplet_2",
        frequency_hz=FREQUENCY_HZ,
        clbs=10000,
        brams=brams,
        dsps=500,
        area=10.0,
        power=2.0,
        node=7,
        relu_implementation=FpgaReluImplementation(clbs_per_lane=5),
        parameter_source=parameter_source,
        input_buffer_blocks=input_buffer_blocks,
        **kwargs,
    )


def _operation(shape=(8, 64), attrs=None, with_weights=True):
    shape = list(shape)
    nodes = [
        {
            "name": "input_layer",
            "class": "Input",
            "inputs": [],
            "output": {"name": "input_layer", "shape": shape},
        },
        {
            "name": "ln_1",
            "class": "LayerNormalization",
            "inputs": [{"name": "input_layer", "shape": shape}],
            "output": {"name": "ln_1", "shape": shape},
            "weights": (
                [
                    {"name": "scale", "shape": [shape[-1]]},
                    {"name": "bias", "shape": [shape[-1]]},
                ]
                if with_weights
                else []
            ),
            "attrs": attrs or {},
        },
    ]
    return parse_atlas_graph(nodes, "graph").operations[0]


# Route of the stub: DRAM 64 bytes per cycle, one link at 0.5 ns per byte.
# A byte then costs 1/64 + 0.5 = 0.515625 ns and 4 + 1 = 5 pJ per bit.
ARCHITECTURE = {
    "pkg": {"mem_pkg_conn": {"mem_type": "hbm2", "Chiplet_1": 4, "Chiplet_2": 0}},
    "transfer_model": {"setup_latency_ns": 10.0, "hop_latency_ns": 2.0},
}
# 64 scale + 64 bias + 1024 table entries, one byte each.
PARAMETER_BYTES = 64 + 64 + 1024
RAW_LOAD_NS = 10.0 + 2.0 + PARAMETER_BYTES * 0.515625


def _system(fpga, channels=True):
    core = SimpleNamespace(id=0, dram_bandwidth=64, dram_energy_scale=4.0)
    return SimpleNamespace(
        core_dict={0: core} if channels else {},
        fpga_chiplet_dict={1: fpga},
        get_shortest_path=lambda source, destination: (
            [source, destination],
            0.5,
            1.0,
        ),
    )


def _context(fpga, slack_ns=0.0, channels=True):
    return EvaluationContext(
        cache=None,
        architecture=ARCHITECTURE,
        system=_system(fpga, channels),
        evaluator_settings={},
        prefetch_slack_ns=slack_ns,
    )


class LayerNormCycleTests(unittest.TestCase):
    def test_depth_counts_two_adder_trees_and_eleven_fixed_operations(self):
        self.assertEqual(layernorm_depth_cycles(64), 2 * 6 + 11)
        self.assertEqual(layernorm_depth_cycles(768), 2 * 10 + 11)

    def test_a_single_feature_has_no_adder_tree(self):
        self.assertEqual(layernorm_depth_cycles(1), 11)

    def test_tokens_follow_at_the_initiation_interval(self):
        self.assertEqual(layernorm_cycles(1, 64, 1), 23)
        self.assertEqual(layernorm_cycles(8, 64, 1), 23 + 7)
        self.assertEqual(layernorm_cycles(8, 64, 4), 23 + 7 * 4)


class LayerNormParserTests(unittest.TestCase):
    def test_node_class_is_recognized(self):
        operation = _operation()
        self.assertEqual(operation.operation_type, "layernorm")
        self.assertEqual(operation.element_count, 8 * 64)

    def test_layernorm_runs_on_the_fpga(self):
        self.assertIn("layernorm", FPGA_OPERATIONS)

    def test_shapes_must_match(self):
        nodes = [
            {
                "name": "input_layer",
                "class": "Input",
                "inputs": [],
                "output": {"name": "input_layer", "shape": [8, 64]},
            },
            {
                "name": "ln_1",
                "class": "LayerNormalization",
                "inputs": [{"name": "input_layer", "shape": [8, 64]}],
                "output": {"name": "ln_1", "shape": [8, 32]},
            },
        ]
        with self.assertRaises(UnsupportedEvaluation):
            parse_atlas_graph(nodes, "graph")


class LayerNormInputAdapterTests(unittest.TestCase):
    def _build(self, **kwargs):
        return LayerNormInputAdapter().build_input(_operation(**kwargs), None)

    def test_reads_tokens_and_features(self):
        view = self._build(shape=(8, 64))
        self.assertEqual((view.seq_len, view.dim, view.reuse_factor), (8, 64, 1))

    def test_rank_one_is_one_token(self):
        view = self._build(shape=(64,))
        self.assertEqual((view.seq_len, view.dim), (1, 64))

    def test_reuse_factor_comes_from_the_attribute(self):
        self.assertEqual(self._build(attrs={"reuse_factor": 4}).reuse_factor, 4)

    def test_invalid_reuse_factor_is_unsupported(self):
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(UnsupportedEvaluation):
                self._build(attrs={"reuse_factor": bad})

    def test_normalizing_a_leading_axis_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            self._build(attrs={"axis": 0})

    def test_last_axis_is_accepted_in_either_indexing(self):
        for axis in (-1, 1, 2):
            self._build(attrs={"axis": axis})

    def test_rank_three_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            self._build(shape=(2, 8, 64))


class LayerNormEvaluatorTests(unittest.TestCase):
    def _evaluate(
        self,
        fpga,
        shape=(8, 64),
        attrs=None,
        endpoint=1,
        slack_ns=0.0,
        channels=True,
        with_weights=True,
    ):
        view = LayerNormInputAdapter().build_input(
            _operation(shape, attrs, with_weights), None
        )
        return LayerNormEvaluator().evaluate(
            view,
            EndpointAssignment(endpoint, "fpga"),
            _context(fpga, slack_ns, channels),
        )

    def test_latency_is_the_kernel_plus_the_input_buffer_read(self):
        # 8 x 64 values x 8 bits = 4096 bits: one block, ceil(4096 / 36) cycles.
        estimate = self._evaluate(_fpga())
        self.assertEqual(estimate.evaluator_id, "fpga_layernorm_v1")
        cycles = layernorm_cycles(8, 64, 1) + math.ceil(4096 / 36)
        self.assertAlmostEqual(estimate.compute_latency_ns, cycles / FREQUENCY_HZ * 1e9)

    def test_more_input_buffer_blocks_shorten_the_start(self):
        slow = self._evaluate(_fpga(input_buffer_blocks=1)).compute_latency_ns
        fast = self._evaluate(_fpga(input_buffer_blocks=8)).compute_latency_ns
        self.assertGreater(slow, fast)

    def test_energy_is_per_element(self):
        estimate = self._evaluate(_fpga(energy=0.5))
        self.assertAlmostEqual(estimate.dynamic_energy_pj, 8 * 64 * 0.5)

    def test_more_tokens_take_longer(self):
        short = self._evaluate(_fpga(), shape=(2, 64)).compute_latency_ns
        long = self._evaluate(_fpga(), shape=(16, 64)).compute_latency_ns
        self.assertGreater(long, short)

    def test_missing_endpoint_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            self._evaluate(_fpga(), endpoint=7)


class ParameterLoadTests(unittest.TestCase):
    _evaluate = LayerNormEvaluatorTests._evaluate

    def test_parameters_load_over_dram_and_the_link(self):
        load = self._evaluate(_fpga()).parameter_load
        self.assertEqual(load.byte_count, PARAMETER_BYTES)
        self.assertEqual((load.source_endpoint, load.destination_endpoint), (0, 1))
        self.assertAlmostEqual(load.raw_latency_ns, RAW_LOAD_NS)
        self.assertAlmostEqual(load.energy_pj, PARAMETER_BYTES * 8 * 5.0)

    def test_without_slack_the_whole_load_is_exposed(self):
        load = self._evaluate(_fpga()).parameter_load
        self.assertAlmostEqual(load.exposed_latency_ns, RAW_LOAD_NS)

    def test_slack_hides_part_of_the_load(self):
        load = self._evaluate(_fpga(), slack_ns=400.0).parameter_load
        self.assertAlmostEqual(load.exposed_latency_ns, RAW_LOAD_NS - 400.0)
        self.assertAlmostEqual(load.hidden_latency_ns, 400.0)

    def test_enough_slack_hides_the_whole_load_but_not_its_energy(self):
        load = self._evaluate(_fpga(), slack_ns=10_000.0).parameter_load
        self.assertEqual(load.exposed_latency_ns, 0.0)
        self.assertAlmostEqual(load.energy_pj, PARAMETER_BYTES * 8 * 5.0)

    def test_bitstream_parameters_have_no_load(self):
        estimate = self._evaluate(_fpga(parameter_source="bitstream"))
        self.assertIsNone(estimate.parameter_load)

    def test_no_memory_attached_array_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            self._evaluate(_fpga(), channels=False)

    def test_without_listed_weights_scale_and_bias_have_one_value_per_feature(self):
        load = self._evaluate(_fpga(), with_weights=False).parameter_load
        self.assertEqual(load.byte_count, PARAMETER_BYTES)

    def test_the_table_size_attribute_sets_the_table(self):
        load = self._evaluate(_fpga(), attrs={"table_size": 256}).parameter_load
        self.assertEqual(load.byte_count, 64 + 64 + 256)

    def test_parameters_and_buffer_must_fit_in_the_bram_of_the_fpga(self):
        # One block for the 4096-bit input buffer and one for 9216 bits of parameters.
        self._evaluate(_fpga(brams=2))
        with self.assertRaisesRegex(ResourceExceeded, "BRAM 2 > 1"):
            self._evaluate(_fpga(brams=1))


class LayerNormSchemaTests(unittest.TestCase):
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

    def test_default_energy(self):
        chiplet = FpgaChiplet.from_dict("Chiplet_1", self._fpga_dict())
        self.assertEqual(chiplet.layernorm_implementation.energy_per_element_pj, 2.0)

    def test_value_is_read_from_the_architecture(self):
        chiplet = FpgaChiplet.from_dict(
            "Chiplet_1",
            self._fpga_dict(layernorm_implementation={"energy_per_element_pj": 1.25}),
        )
        self.assertEqual(chiplet.layernorm_implementation.energy_per_element_pj, 1.25)

    def test_parameter_source_defaults_to_bitstream(self):
        chiplet = FpgaChiplet.from_dict("Chiplet_1", self._fpga_dict())
        self.assertEqual(chiplet.parameter_source, "bitstream")

    def test_parameter_source_is_read_from_the_architecture(self):
        chiplet = FpgaChiplet.from_dict(
            "Chiplet_1", self._fpga_dict(parameter_source="bitstream")
        )
        self.assertEqual(chiplet.parameter_source, "bitstream")

    def test_unknown_parameter_source_is_rejected(self):
        with self.assertRaises(ValueError):
            FpgaChiplet.from_dict(
                "Chiplet_1", self._fpga_dict(parameter_source="flash")
            )

    def test_invalid_values_are_rejected(self):
        for bad in ({"energy_per_element_pj": -1}, {"unknown": 1}, "x"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    FpgaChiplet.from_dict(
                        "Chiplet_1", self._fpga_dict(layernorm_implementation=bad)
                    )


class LayerNormRegistrationTests(unittest.TestCase):
    def test_registers_the_evaluator(self):
        registry = EvaluatorRegistry()
        register_layernorm_evaluators(registry)
        self.assertIn("fpga_layernorm_v1", registry)

    def test_layernorm_profiles_name_the_evaluator(self):
        for path in (
            "cfg/profiles/atlas_modular_layernorm_v1.json",
            "cfg/profiles/rule4ml_hybrid_layernorm_v1.json",
        ):
            with self.subTest(path=path):
                profile = load_evaluation_profile(path)
                self.assertEqual(
                    profile.evaluator_id_for("layernorm"), "fpga_layernorm_v1"
                )

    def test_existing_profiles_are_unchanged(self):
        # Their fingerprints are pinned, so they do not name a LayerNorm evaluator.
        profile = load_evaluation_profile("cfg/profiles/atlas_modular_v1.json")
        with self.assertRaises(UnsupportedEvaluation):
            profile.evaluator_id_for("layernorm")


if __name__ == "__main__":
    unittest.main()
