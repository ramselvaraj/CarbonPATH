import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from main import evaluate_atlas_graph
from network import evaluate_atlas_network
from system.utils.AtlasGraphAdapter import (
    GemmOperation,
    ReluOperation,
    load_atlas_graph,
    parse_atlas_graph,
)
from system.utils.EvaluationProfile import (
    DEFAULT_PROFILE_PATH,
    load_evaluation_profile,
)
from system.utils.OperationPlacement import FixedSingleSaSingleFpgaPlacement
from system.utils.SimulationCache import SimulationCache
from system.utils.Simulator import Simulator
from system.utils.TensorMovement import TensorMovementService
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


FIXTURE = Path("cfg/examples/atlas/dense_relu_funnel.graph_dump.json")
ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")

EXPECTED_SHAPES = [(128, 128, 1024), (128, 1024, 512), (128, 512, 256), (128, 256, 64)]

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

    def test_unbatched_rank_one_gemm_becomes_m1(self):
        node = _gemm_node(
            inputs=[{"name": "x", "shape": [64]}],
            output={"shape": [32]},
            weights=[{"name": "w", "shape": [32, 64]}],
        )
        operation = parse_atlas_graph([node], "unbatched").operations[0]
        self.assertIsInstance(operation, GemmOperation)
        self.assertEqual(operation.gemm_shape, (1, 64, 32))

    def test_relu_parses_without_carbonpath_envelope(self):
        operation = parse_atlas_graph([_relu_node()], "relu_only").operations[0]
        self.assertIsInstance(operation, ReluOperation)
        self.assertEqual(operation.element_count, 128 * 256)

    def test_unsupported_node_class_is_rejected(self):
        nodes = [_gemm_node(), {"name": "softmax", "class": "Softmax", "attrs": {}}]
        with self.assertRaises(UnsupportedEvaluation):
            parse_atlas_graph(nodes, "bad")

    def test_non_linear_chain_is_rejected(self):
        nodes = [_gemm_node(), _relu_node(inputs=[{"name": "someone_else", "shape": [128, 256]}])]
        with self.assertRaises(UnsupportedEvaluation):
            parse_atlas_graph(nodes, "bad")

    def test_dynamic_weight_gemm_is_rejected(self):
        with self.assertRaises(UnsupportedEvaluation):
            parse_atlas_graph([_gemm_node(attrs={"weights_in_core": False})], "bad")

    def test_fingerprint_is_stable(self):
        graph = load_atlas_graph(FIXTURE)
        self.assertEqual(graph.fingerprint(), load_atlas_graph(FIXTURE).fingerprint())


class EvaluationProfileTests(unittest.TestCase):
    def test_default_profile_maps_operation_types(self):
        profile = load_evaluation_profile()
        self.assertEqual(profile.evaluator_id_for("gemm"), "legacy_scale_sim_gemm_v1")
        self.assertEqual(profile.evaluator_id_for("relu"), "legacy_fpga_relu_v1")
        self.assertTrue(DEFAULT_PROFILE_PATH.exists())

    def test_unknown_operation_type_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            load_evaluation_profile().evaluator_id_for("softmax")


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

    def _evaluate(self):
        cache = self._cache()
        original = Simulator.simulate_single_core
        with patch.object(
            Simulator,
            "simulate_single_core",
            autospec=True,
            side_effect=original,
        ) as simulate_core:
            evaluation = evaluate_atlas_network(self.graph, self.architecture, cache)
        return evaluation, simulate_core

    def test_operation_counts(self):
        evaluation, _ = self._evaluate()
        self.assertEqual(evaluation.summary["gemm_count"], 4)
        self.assertEqual(evaluation.summary["relu_count"], 3)
        self.assertEqual(evaluation.summary["operation_count"], 7)

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


if __name__ == "__main__":
    unittest.main()
