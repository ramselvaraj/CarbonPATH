import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from main import (
    TensorAccessPlan,
    build_default_evaluator_registry,
    evaluate_atlas_graph,
)
from network import evaluate_atlas_network, evaluate_network
from system.utils.AtlasGraphAdapter import (
    AtlasOperation,
    load_atlas_graph,
    parse_atlas_graph,
)
from system.utils.EvaluationProfile import (
    DEFAULT_PROFILE_PATH,
    load_evaluation_profile,
)
from system.utils.NonGemmEstimator import (
    LegacyReluInputAdapter,
    PlaceholderHlsReluInputAdapter,
    PlaceholderSoftmaxInputAdapter,
)
from system.utils.OperationEvaluator import EvaluatorRegistry
from system.utils.OperationPlacement import (
    FixedSingleSaSingleFpgaPlacement,
    build_placement_policy,
)
from system.utils.SimulationCache import SimulationCache
from system.utils.Simulator import Simulator
from system.utils.TensorMovement import (
    TensorMovementService,
    build_movement_service,
)
from system.utils.TransferEstimator import (
    TransferEstimator,
    build_transfer_cost_model,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


FIXTURE = Path("cfg/examples/atlas/dense_relu_funnel.graph_dump.json")
SOFTMAX_FIXTURE = Path("cfg/examples/atlas/dense_softmax.graph_dump.json")
MIXED_FIXTURE = Path("cfg/examples/atlas/dense_relu_softmax.graph_dump.json")
ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
PLACEHOLDER_PROFILE = Path("cfg/profiles/placeholder_fpga_ops_v0.json")

EXPECTED_SHAPES = [(128, 128, 1024), (128, 1024, 512), (128, 512, 256), (128, 256, 64)]
FUNNEL_FINGERPRINT = "810bd39b27e0"
SOFTMAX_FINGERPRINT = "88cd5accb4ca"
MIXED_FIXTURE_FINGERPRINT = "abf1849faedc"
LEGACY_PROFILE_FINGERPRINT = "8f477eb548ff"
PLACEHOLDER_PROFILE_FINGERPRINT = "269e1faefe2b"

GOLDEN = {
    "latency_ns": 133648.7947773606,
    "baseline_energy_pj": 416817178.7118933,
    "compute_energy_pj": 105521570.86836326,
    "movement_energy_pj": 1835008.0,
    "total_energy_pj": 524173757.5802566,
}

CACHE_COLUMNS = [
    "core_size",
    "data_flow",
    "bandwidth",
    "buffer_size",
    "M",
    "K",
    "N",
    "latency",
]


def _load(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _gemm_node(**overrides):
    node = {
        "name": "gemm_dense",
        "class": "Gemm",
        "inputs": [{"name": "input_layer", "shape": [128, 128]}],
        "output": {"shape": [128, 256]},
        "weights": [{"name": "w", "shape": [256, 128]}],
        "attrs": {"weights_in_core": True},
    }
    node.update(overrides)
    return node


def _relu_node(**overrides):
    node = {
        "name": "dense_relu",
        "class": "Activation",
        "inputs": [{"name": "dense", "shape": [128, 256]}],
        "output": {"shape": [128, 256]},
        "weights": [],
        "attrs": {"activation": "relu"},
    }
    node.update(overrides)
    return node


class AtlasGraphAdapterTests(unittest.TestCase):
    def test_loads_direct_atlas_dump(self):
        graph = load_atlas_graph(FIXTURE)
        types = [operation.operation_type for operation in graph.operations]
        self.assertEqual(types, ["gemm", "relu"] * 3 + ["gemm"])
        self.assertEqual(graph.dtype, "int8")

    def test_operations_share_one_type(self):
        graph = load_atlas_graph(FIXTURE)
        self.assertTrue(
            all(isinstance(operation, AtlasOperation) for operation in graph.operations)
        )

    def test_gemm_shapes_come_from_atlas(self):
        graph = load_atlas_graph(FIXTURE)
        shapes = [
            operation.gemm_shape
            for operation in graph.operations
            if operation.operation_type == "gemm"
        ]
        self.assertEqual(shapes, EXPECTED_SHAPES)

    def test_relu_element_counts_come_from_atlas(self):
        graph = load_atlas_graph(FIXTURE)
        counts = [
            operation.element_count
            for operation in graph.operations
            if operation.operation_type == "relu"
        ]
        self.assertEqual(counts, [128 * 1024, 128 * 512, 128 * 256])

    def test_softmax_fixture_loads(self):
        graph = load_atlas_graph(SOFTMAX_FIXTURE)
        types = [operation.operation_type for operation in graph.operations]
        self.assertEqual(types, ["gemm", "softmax", "gemm"])
        softmax = graph.operations[1]
        self.assertIsNone(softmax.gemm_shape)
        self.assertEqual(softmax.element_count, 8 * 16)
        self.assertEqual(softmax.source.attribute("axis"), -1)

    def test_unbatched_rank_one_gemm_becomes_m1(self):
        node = _gemm_node(
            inputs=[{"name": "x", "shape": [64]}],
            output={"shape": [32]},
            weights=[{"name": "w", "shape": [32, 64]}],
        )
        operation = parse_atlas_graph([node], "unbatched").operations[0]
        self.assertIsInstance(operation, AtlasOperation)
        self.assertEqual(operation.gemm_shape, (1, 64, 32))

    def test_relu_parses_without_carbonpath_envelope(self):
        operation = parse_atlas_graph([_relu_node()], "relu_only").operations[0]
        self.assertIsInstance(operation, AtlasOperation)
        self.assertEqual(operation.element_count, 128 * 256)

    def test_source_view_exposes_reuse_factor(self):
        graph = load_atlas_graph(FIXTURE)
        relu = next(
            operation
            for operation in graph.operations
            if operation.operation_type == "relu"
        )
        self.assertEqual(relu.source.attribute("reuse_factor"), 1)

    def test_source_view_is_immutable(self):
        operation = parse_atlas_graph([_relu_node()], "relu_only").operations[0]
        with self.assertRaises(TypeError):
            operation.source.attrs["activation"] = "gelu"
        self.assertIsInstance(operation.source.weights, tuple)

    def test_missing_reuse_factor_is_not_a_graph_error(self):
        operation = parse_atlas_graph(
            [_relu_node(attrs={"activation": "relu"})], "relu_only"
        ).operations[0]
        self.assertIsNone(operation.source.attribute("reuse_factor", None))

    def test_unsupported_node_class_is_rejected(self):
        nodes = [_gemm_node(), {"name": "pool", "class": "Pooling", "attrs": {}}]
        with self.assertRaises(UnsupportedEvaluation):
            parse_atlas_graph(nodes, "bad")

    def test_non_linear_chain_is_rejected(self):
        nodes = [
            _gemm_node(),
            _relu_node(inputs=[{"name": "someone_else", "shape": [128, 256]}]),
        ]
        with self.assertRaises(UnsupportedEvaluation):
            parse_atlas_graph(nodes, "bad")

    def test_dynamic_weight_gemm_is_rejected(self):
        with self.assertRaises(UnsupportedEvaluation):
            parse_atlas_graph([_gemm_node(attrs={"weights_in_core": False})], "bad")

    def test_fingerprint_is_stable(self):
        graph = load_atlas_graph(FIXTURE)
        self.assertEqual(graph.fingerprint(), load_atlas_graph(FIXTURE).fingerprint())

    def test_fingerprint_covers_full_artifact(self):
        baseline = parse_atlas_graph([_relu_node()], "x").fingerprint()
        changed = parse_atlas_graph(
            [_relu_node(attrs={"activation": "relu", "table_size": 2048})], "x"
        ).fingerprint()
        self.assertNotEqual(baseline, changed)

    def test_mixed_fixture_loads(self):
        graph = load_atlas_graph(MIXED_FIXTURE)
        types = [operation.operation_type for operation in graph.operations]
        self.assertEqual(types, ["gemm", "relu", "gemm", "softmax", "gemm"])
        gemm_shapes = [
            operation.gemm_shape
            for operation in graph.operations
            if operation.operation_type == "gemm"
        ]
        self.assertEqual(gemm_shapes, [(8, 16, 16), (8, 16, 32), (8, 32, 8)])

    def test_fixture_fingerprints_are_pinned(self):
        self.assertEqual(load_atlas_graph(FIXTURE).fingerprint(), FUNNEL_FINGERPRINT)
        self.assertEqual(
            load_atlas_graph(SOFTMAX_FIXTURE).fingerprint(), SOFTMAX_FINGERPRINT
        )
        self.assertEqual(
            load_atlas_graph(MIXED_FIXTURE).fingerprint(), MIXED_FIXTURE_FINGERPRINT
        )


class EvaluationProfileTests(unittest.TestCase):
    def test_default_profile_maps_operation_types(self):
        profile = load_evaluation_profile()
        self.assertEqual(profile.evaluator_id_for("gemm"), "legacy_scale_sim_gemm_v1")
        self.assertEqual(profile.evaluator_id_for("relu"), "legacy_fpga_relu_v1")
        self.assertTrue(DEFAULT_PROFILE_PATH.exists())

    def test_placeholder_profile_maps_placeholder_models(self):
        profile = load_evaluation_profile(PLACEHOLDER_PROFILE)
        self.assertEqual(profile.evaluator_id_for("relu"), "placeholder_hls_relu_v0")
        self.assertEqual(
            profile.evaluator_id_for("softmax"), "placeholder_fpga_softmax_v0"
        )

    def test_profile_fingerprints_are_pinned(self):
        self.assertEqual(
            load_evaluation_profile().fingerprint(), LEGACY_PROFILE_FINGERPRINT
        )
        self.assertEqual(
            load_evaluation_profile(PLACEHOLDER_PROFILE).fingerprint(),
            PLACEHOLDER_PROFILE_FINGERPRINT,
        )

    def test_unknown_operation_type_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            load_evaluation_profile().evaluator_id_for("layernorm")


class InputAdapterTests(unittest.TestCase):
    def setUp(self):
        self.access_plan = TensorAccessPlan(
            activation_from_dram=True, output_to_dram=False
        )

    def test_legacy_relu_adapter_needs_only_element_count(self):
        operation = parse_atlas_graph(
            [_relu_node(attrs={"activation": "relu"})], "relu_only"
        ).operations[0]
        evaluator_input = LegacyReluInputAdapter().build_input(
            operation, self.access_plan
        )
        self.assertEqual(evaluator_input.element_count, 128 * 256)

    def test_hls_relu_adapter_reads_reuse_factor(self):
        operation = parse_atlas_graph(
            [_relu_node(attrs={"activation": "relu", "reuse_factor": 4})],
            "relu_only",
        ).operations[0]
        evaluator_input = PlaceholderHlsReluInputAdapter().build_input(
            operation, self.access_plan
        )
        self.assertEqual(evaluator_input.reuse_factor, 4)

    def test_hls_relu_adapter_rejects_missing_reuse_factor(self):
        operation = parse_atlas_graph(
            [_relu_node(attrs={"activation": "relu"})], "relu_only"
        ).operations[0]
        with self.assertRaises(UnsupportedEvaluation):
            PlaceholderHlsReluInputAdapter().build_input(operation, self.access_plan)

    def test_hls_relu_adapter_rejects_wrong_operation_type(self):
        operation = parse_atlas_graph([_gemm_node()], "gemm_only").operations[0]
        with self.assertRaises(UnsupportedEvaluation):
            PlaceholderHlsReluInputAdapter().build_input(operation, self.access_plan)

    def test_softmax_adapter_reads_axis(self):
        graph = load_atlas_graph(SOFTMAX_FIXTURE)
        softmax = graph.operations[1]
        evaluator_input = PlaceholderSoftmaxInputAdapter().build_input(
            softmax, self.access_plan
        )
        self.assertEqual(evaluator_input.axis, -1)
        self.assertEqual(evaluator_input.element_count, 8 * 16)


class RegistryTests(unittest.TestCase):
    def test_built_in_evaluators_resolve(self):
        registry = build_default_evaluator_registry()
        for evaluator_id in (
            "legacy_scale_sim_gemm_v1",
            "legacy_fpga_relu_v1",
            "placeholder_hls_relu_v0",
            "placeholder_fpga_softmax_v0",
        ):
            self.assertIn(evaluator_id, registry)
            self.assertEqual(registry.get(evaluator_id).evaluator_id, evaluator_id)

    def test_unknown_evaluator_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            build_default_evaluator_registry().get("nope")

    def test_duplicate_evaluator_is_rejected(self):
        registry = EvaluatorRegistry()
        binding = build_default_evaluator_registry().get("legacy_fpga_relu_v1")
        registry.register(binding.evaluator, binding.input_adapter)
        with self.assertRaises(ValueError):
            registry.register(binding.evaluator, binding.input_adapter)

    def test_placement_policy_resolves_by_id(self):
        system = SimpleNamespace(core_dict={0: object()}, fpga_chiplet_dict={1: object()})
        policy = build_placement_policy("fixed_single_sa_single_fpga_v1", system)
        self.assertEqual(policy.endpoint_for("gemm").endpoint_id, 0)
        self.assertEqual(policy.endpoint_for("relu").endpoint_kind, "fpga")
        self.assertEqual(policy.endpoint_for("softmax").endpoint_kind, "fpga")

    def test_unknown_placement_policy_is_unsupported(self):
        system = SimpleNamespace(core_dict={0: object()}, fpga_chiplet_dict={1: object()})
        with self.assertRaises(UnsupportedEvaluation):
            build_placement_policy("nope", system)

    def test_movement_policy_resolves_by_id(self):
        service = build_movement_service(
            "activation_boundary_v1", TransferEstimator()
        )
        self.assertIsInstance(service, TensorMovementService)

    def test_unknown_movement_policy_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            build_movement_service("nope", TransferEstimator())

    def test_transfer_cost_model_resolves_by_id(self):
        model = build_transfer_cost_model("route_transfer_v1")
        self.assertIsInstance(model, TransferEstimator)

    def test_unknown_transfer_cost_model_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            build_transfer_cost_model("nope")


class PlacementTests(unittest.TestCase):
    def test_fixed_policy_places_gemm_and_relu(self):
        system = SimpleNamespace(core_dict={0: object()}, fpga_chiplet_dict={1: object()})
        policy = FixedSingleSaSingleFpgaPlacement(system)
        self.assertEqual(policy.endpoint_for("gemm").endpoint_id, 0)
        self.assertEqual(policy.endpoint_for("relu").endpoint_kind, "fpga")

    def test_multiple_sa_endpoints_are_unsupported(self):
        system = SimpleNamespace(
            core_dict={0: object(), 1: object()}, fpga_chiplet_dict={2: object()}
        )
        with self.assertRaises(UnsupportedEvaluation):
            FixedSingleSaSingleFpgaPlacement(system)


class TensorMovementServiceTests(unittest.TestCase):
    def test_same_endpoint_is_retained(self):
        record = TensorMovementService().move("t", 512, 0, 0, system=None)
        self.assertEqual(record.method, "retain")
        self.assertEqual(record.latency_ns, 0.0)

    def test_missing_route_is_unsupported(self):
        system = SimpleNamespace(interconnect_dict={0: [], 1: []})
        system.get_shortest_path = lambda source, destination: (None, 0, 0)
        with self.assertRaises(UnsupportedEvaluation):
            TensorMovementService().move("t", 512, 0, 1, system=system)


class AtlasEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        self.graph = load_atlas_graph(FIXTURE)

    def _cache(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache_path = Path(directory.name) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        return SimulationCache(cache_path, simulator_dir=directory.name)

    def _evaluate(self, graph=None, profile=None):
        cache = self._cache()
        original = Simulator.simulate_single_core
        with patch.object(
            Simulator,
            "simulate_single_core",
            autospec=True,
            side_effect=original,
        ) as simulate_core:
            evaluation = evaluate_atlas_network(
                graph or self.graph, self.architecture, cache, profile=profile
            )
        return evaluation, simulate_core

    def _assert_close(self, value, expected):
        self.assertLessEqual(abs(value - expected), abs(expected) * 1e-9 + 1e-6)

    def test_operation_counts(self):
        evaluation, _ = self._evaluate()
        self.assertEqual(evaluation.summary["gemm_count"], 4)
        self.assertEqual(evaluation.summary["relu_count"], 3)
        self.assertEqual(evaluation.summary["operation_count"], 7)
        self.assertEqual(evaluation.summary["operation_counts"], {"gemm": 4, "relu": 3})

    def test_first_gemm_output_does_not_write_dram_before_relu(self):
        cache = self._cache()
        with patch("main.simulate_single_gemm") as simulate:
            simulate.return_value = {
                "latency_ns": 1.0,
                "dram_interconnect_energy_pj": 2.0,
                "sram_energy_pj": 3.0,
            }
            evaluate_atlas_graph(cache, self.architecture, self.graph)
        first = simulate.call_args_list[0].kwargs
        last = simulate.call_args_list[-1].kwargs
        self.assertTrue(first["activation_from_dram"])
        self.assertFalse(first["output_to_dram"])
        self.assertFalse(last["activation_from_dram"])
        self.assertTrue(last["output_to_dram"])

    def test_relu_owns_sa_to_fpga_and_gemm_owns_fpga_to_sa(self):
        evaluation, _ = self._evaluate()
        layers = evaluation.layers
        expand_relu = layers[layers["layer_name"] == "expand_relu"].iloc[0]
        self.assertEqual(expand_relu["movement_method"], "route")
        self.assertEqual(expand_relu["movement_source_endpoint"], 0)
        self.assertEqual(expand_relu["movement_destination_endpoint"], 1)
        contract1 = layers[layers["layer_name"] == "gemm_contract1"].iloc[0]
        self.assertEqual(contract1["movement_source_endpoint"], 1)
        self.assertEqual(contract1["movement_destination_endpoint"], 0)

    def test_every_operation_has_an_evaluator(self):
        evaluation, _ = self._evaluate()
        self.assertTrue(evaluation.layers["evaluator_id"].notna().all())

    def test_total_latency_is_sum_of_layer_latencies(self):
        evaluation, _ = self._evaluate()
        expected = evaluation.layers["latency_ns"].sum()
        self.assertAlmostEqual(evaluation.summary["latency_ns"], expected)

    def test_total_energy_is_additive(self):
        evaluation, _ = self._evaluate()
        summary = evaluation.summary
        expected = (
            summary["baseline_energy_pj"]
            + summary["compute_energy_pj"]
            + summary["movement_energy_pj"]
        )
        self.assertAlmostEqual(summary["total_energy_pj"], expected)
        self.assertGreater(summary["movement_energy_pj"], 0)

    def test_profile_is_recorded(self):
        evaluation, _ = self._evaluate()
        self.assertEqual(evaluation.summary["evaluation_profile"], "legacy_sa_fpga_v1")
        self.assertTrue(evaluation.summary["evaluation_profile_fingerprint"])

    def test_legacy_result_is_pinned(self):
        evaluation, _ = self._evaluate()
        summary = evaluation.summary
        self._assert_close(summary["latency_ns"], GOLDEN["latency_ns"])
        self._assert_close(summary["baseline_energy_pj"], GOLDEN["baseline_energy_pj"])
        self._assert_close(summary["compute_energy_pj"], GOLDEN["compute_energy_pj"])
        self._assert_close(summary["movement_energy_pj"], GOLDEN["movement_energy_pj"])
        self._assert_close(summary["total_energy_pj"], GOLDEN["total_energy_pj"])

    def test_placeholder_profile_swaps_relu_model_without_graph_change(self):
        profile = load_evaluation_profile(PLACEHOLDER_PROFILE)
        evaluation, _ = self._evaluate(profile=profile)
        self.assertEqual(evaluation.summary["operation_counts"], {"gemm": 4, "relu": 3})
        relu_rows = evaluation.layers[
            evaluation.layers["operation_type"] == "relu"
        ]
        self.assertTrue(
            (relu_rows["evaluator_id"] == "placeholder_hls_relu_v0").all()
        )

    def test_softmax_graph_evaluates_on_fpga(self):
        graph = load_atlas_graph(SOFTMAX_FIXTURE)
        profile = load_evaluation_profile(PLACEHOLDER_PROFILE)
        evaluation, _ = self._evaluate(graph=graph, profile=profile)
        summary = evaluation.summary
        self.assertEqual(summary["operation_counts"], {"gemm": 2, "softmax": 1})
        softmax = evaluation.layers[
            evaluation.layers["operation_type"] == "softmax"
        ].iloc[0]
        self.assertEqual(softmax["endpoint_kind"], "fpga")
        self.assertEqual(softmax["evaluator_id"], "placeholder_fpga_softmax_v0")
        self.assertEqual(softmax["movement_source_endpoint"], 0)
        self.assertEqual(softmax["movement_destination_endpoint"], 1)

    def test_mixed_workload_evaluates_with_placeholder_profile(self):
        graph = load_atlas_graph(MIXED_FIXTURE)
        profile = load_evaluation_profile(PLACEHOLDER_PROFILE)
        evaluation, _ = self._evaluate(graph=graph, profile=profile)
        summary = evaluation.summary
        self.assertEqual(
            summary["operation_counts"], {"gemm": 3, "relu": 1, "softmax": 1}
        )
        layers = evaluation.layers.set_index("layer_name")
        self.assertEqual(layers.loc["stem_relu", "endpoint_kind"], "fpga")
        self.assertEqual(layers.loc["softmax", "endpoint_kind"], "fpga")
        self.assertEqual(layers.loc["softmax", "evaluator_id"], "placeholder_fpga_softmax_v0")
        for name in ("stem_relu", "gemm_expand", "softmax", "gemm_head"):
            self.assertEqual(layers.loc[name, "movement_method"], "route")
        self.assertTrue(pd.isna(layers.loc["gemm_stem", "movement_method"]))
        self.assertAlmostEqual(
            summary["movement_energy_pj"],
            layers["movement_energy_pj"].sum(),
        )

    def test_legacy_network_rejects_evaluation_profile(self):
        with self.assertRaises(ValueError):
            evaluate_network(SimpleNamespace(), {}, None, profile=object())


if __name__ == "__main__":
    unittest.main()
