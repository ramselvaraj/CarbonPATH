import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from main import build_default_evaluator_registry, evaluate_atlas_graph
from system.utils import Rule4MlEvaluators
from system.utils.AtlasAnnealingMoves import mutate_atlas_design_point
from system.utils.AtlasGraphAdapter import (
    FPGA_OPERATIONS,
    FUSABLE_ACTIVATIONS,
    SIMPLE_ACTIVATIONS,
    load_atlas_graph,
    parse_atlas_graph,
)
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.FusedActivation import fusion_types
from system.utils.OperationAssignment import build_assignment_policy
from system.utils.SimulationCache import SimulationCache
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.WorkloadMapping import (
    FUSED_ACTIVATION_POLICY,
    initial_workload_mapping,
)


GRAPH = Path("cfg/examples/atlas/dense_activations.graph_dump.json")
RULE4ML_GRAPH = Path("cfg/examples/atlas/dense_rule4ml_activations.graph_dump.json")
RULE4ML_OPS_PROFILE = Path("cfg/profiles/rule4ml_fpga_ops_v3.json")
DUMMY_PROFILE = Path("cfg/profiles/rule4ml_fused_dummy_v0.json")
ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
PROFILE = Path("cfg/profiles/atlas_fused_activations_v1.json")
RULE4ML_PROFILE = Path("cfg/profiles/rule4ml_fused_activations_v1.json")
SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space.json")
TYPES = [
    "gemm",
    "leaky_relu",
    "gemm",
    "thresholded_relu",
    "gemm",
    "sigmoid",
    "gemm",
    "tanh",
    "gemm",
]
PREDICTION = {"cycles": 10.0, "interval": 2.0, "resources": {}}

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
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _activation_node(node_class, activation):
    return {
        "name": "act",
        "class": node_class,
        "inputs": [{"name": "input_layer", "shape": [4, 8]}],
        "output": {"shape": [4, 8]},
        "weights": [],
        "attrs": {"activation": activation},
    }


class ActivationTypeTests(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        self.graph = load_atlas_graph(GRAPH)
        self.profile = load_evaluation_profile(PROFILE)
        self.search_space = _load(SEARCH_SPACE)
        self.mapping = initial_workload_mapping(
            self.graph, self.architecture, self.profile, self.search_space
        )
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        cache_path = Path(self._directory.name) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        self.cache = SimulationCache(cache_path, simulator_dir=self._directory.name)

    def _evaluate(self, mapping, profile=None):
        with patch.object(
            Rule4MlEvaluators, "predict_activation", return_value=PREDICTION
        ) as predict:
            evaluation = evaluate_atlas_graph(
                self.cache,
                self.architecture,
                self.graph,
                profile=profile or self.profile,
                mapping=mapping,
            )
        return evaluation, predict

    def test_graph_has_each_activation_type(self):
        types = [operation.operation_type for operation in self.graph.operations]
        self.assertEqual(types, TYPES)

    def test_parser_reads_each_supported_activation(self):
        cases = {
            ("Activation", "relu"): "relu",
            ("Activation", "sigmoid"): "sigmoid",
            ("Activation", "tanh"): "tanh",
            ("ParametrizedActivation", "leakyrelu"): "leaky_relu",
            ("ParametrizedActivation", "thresholdedrelu"): "thresholded_relu",
        }
        for (node_class, activation), expected in cases.items():
            with self.subTest(activation=activation):
                graph = parse_atlas_graph(
                    [_activation_node(node_class, activation)], "one"
                )
                self.assertEqual(graph.operations[0].operation_type, expected)

    def test_parser_rejects_an_unknown_activation(self):
        for node_class, activation in (
            ("Activation", "relu6"),
            ("ParametrizedActivation", "elu"),
            ("ParametrizedActivation", "relu"),
        ):
            with self.subTest(activation=activation):
                with self.assertRaisesRegex(
                    UnsupportedEvaluation, "unsupported activation"
                ):
                    parse_atlas_graph(
                        [_activation_node(node_class, activation)], "one"
                    )

    def test_each_activation_starts_on_the_fpga(self):
        for operation in self.mapping.operations:
            expected = "sa" if operation.is_gemm else "fpga"
            self.assertEqual(operation.endpoint_kind, expected)
        system = self._evaluate(self.mapping)[0].system
        policy = build_assignment_policy("all_sas_single_fpga_v1", system)
        for operation_type in FPGA_OPERATIONS:
            self.assertEqual(policy.endpoint_for(operation_type).endpoint_kind, "fpga")

    def test_unfused_graph_uses_the_evaluator_of_each_type(self):
        evaluation, predict = self._evaluate(self.mapping)

        self.assertEqual(
            [result.evaluator_id for result in evaluation.results],
            [
                "legacy_scale_sim_gemm_v1",
                "legacy_fpga_leaky_relu_v1",
                "legacy_scale_sim_gemm_v1",
                "legacy_fpga_thresholded_relu_v1",
                "legacy_scale_sim_gemm_v1",
                "rule4ml_fpga_sigmoid_v1",
                "legacy_scale_sim_gemm_v1",
                "rule4ml_fpga_tanh_v1",
                "legacy_scale_sim_gemm_v1",
            ],
        )
        self.assertEqual(
            [call.kwargs["activation"] for call in predict.call_args_list],
            ["sigmoid", "tanh"],
        )
        for result in evaluation.results[1:]:
            self.assertEqual(result.movement.method, "distributed_route")

    def test_profile_selects_the_activation_types_that_can_be_fused(self):
        self.assertEqual(self.mapping.fusable_indexes, (1, 3, 5, 7))
        self.assertEqual(fusion_types(self.profile), SIMPLE_ACTIVATIONS)
        self.assertEqual(
            fusion_types(load_evaluation_profile(DUMMY_PROFILE)),
            ("relu", "sigmoid", "tanh"),
        )
        self.assertNotIn("softmax", FUSABLE_ACTIVATIONS)

    def test_softmax_cannot_be_fused(self):
        graph = load_atlas_graph(RULE4ML_GRAPH)
        profile = load_evaluation_profile(DUMMY_PROFILE)
        mapping = initial_workload_mapping(
            graph, self.architecture, profile, self.search_space
        )
        self.assertEqual(mapping.fusable_indexes, (1, 3, 5))
        with self.assertRaisesRegex(UnsupportedEvaluation, "cannot be fused"):
            mapping.with_fusion_change(
                7, "sa", (0,), FUSED_ACTIVATION_POLICY, "local_sram_v1"
            )

    def test_fused_cost_is_proportional_to_the_element_count(self):
        graph = load_atlas_graph(RULE4ML_GRAPH)
        profile = load_evaluation_profile(DUMMY_PROFILE)
        mapping = initial_workload_mapping(
            graph, self.architecture, profile, self.search_space
        )
        for index in (1, 3, 5):
            mapping = mapping.with_fusion_change(
                index, "sa", (0,), FUSED_ACTIVATION_POLICY, "local_sram_v1"
            )
        with patch.object(
            Rule4MlEvaluators, "predict_activation", return_value=PREDICTION
        ) as predict:
            evaluation = evaluate_atlas_graph(
                self.cache, self.architecture, graph, profile=profile, mapping=mapping
            )

        rates = {1: (0.01, 0.05), 3: (0.1, 0.5), 5: (0.1, 0.5)}
        for index, (latency_rate, energy_rate) in rates.items():
            result = evaluation.results[index]
            self.assertEqual(
                result.evaluator_id, f"sa_fused_{result.operation_type}_v1"
            )
            self.assertEqual(result.movement.method, "fused")
            self.assertAlmostEqual(
                result.compute_latency_ns, latency_rate * result.element_count
            )
            self.assertAlmostEqual(
                result.compute_energy_pj, energy_rate * result.element_count
            )
        self.assertEqual(evaluation.results[7].endpoint_kind, "fpga")
        self.assertEqual(
            [call.kwargs["activation"] for call in predict.call_args_list],
            ["softmax"],
        )

    def test_fused_simple_activations_have_no_cost(self):
        mapping = self.mapping
        for index in (1, 3):
            mapping = mapping.with_fusion_change(
                index, "sa", (0,), FUSED_ACTIVATION_POLICY, "local_sram_v1"
            )
        unfused, _ = self._evaluate(self.mapping)
        fused, _ = self._evaluate(mapping)

        self.assertEqual(fused.results[1].evaluator_id, "sa_fused_leaky_relu_v1")
        self.assertEqual(
            fused.results[3].evaluator_id, "sa_fused_thresholded_relu_v1"
        )
        for index in (1, 3):
            self.assertEqual(fused.results[index].movement.method, "fused")
            self.assertEqual(fused.results[index].compute_latency_ns, 0.0)
        self.assertEqual(fused.results[5].endpoint_kind, "fpga")
        self.assertLess(fused.latency_ns, unfused.latency_ns)

    def test_fusion_move_changes_only_the_types_of_the_profile(self):
        params = dict(self.search_space)
        params["move_weights"] = {"relu_fusion": 1.0}
        for seed in range(10):
            _, moved, _ = mutate_atlas_design_point(
                self.architecture,
                self.mapping,
                params,
                rng=random.Random(seed),
                default_movement_policy=self.profile.movement_policy,
                fusion_types=("thresholded_relu",),
            )
            self.assertTrue(moved.is_fused(3))
            self.assertFalse(moved.is_fused(1))

    def test_rule4ml_profile_uses_the_relu_prediction_for_simple_types(self):
        profile = load_evaluation_profile(RULE4ML_PROFILE)

        evaluation, predict = self._evaluate(self.mapping, profile=profile)

        self.assertEqual(
            [result.evaluator_id for result in evaluation.results[1::2]],
            [
                "rule4ml_hybrid_leaky_relu_v1",
                "rule4ml_hybrid_thresholded_relu_v1",
                "rule4ml_fpga_sigmoid_v1",
                "rule4ml_fpga_tanh_v1",
            ],
        )
        self.assertEqual(
            [call.kwargs["activation"] for call in predict.call_args_list],
            ["relu", "relu", "sigmoid", "tanh"],
        )

    def test_rule4ml_graph_uses_a_rule4ml_evaluator_for_each_activation(self):
        graph = load_atlas_graph(RULE4ML_GRAPH)
        profile = load_evaluation_profile(RULE4ML_OPS_PROFILE)
        activations = ["relu", "sigmoid", "tanh", "softmax"]
        self.assertEqual(
            [operation.operation_type for operation in graph.operations[1::2]],
            activations,
        )

        with patch.object(
            Rule4MlEvaluators, "predict_activation", return_value=PREDICTION
        ) as predict:
            evaluation = evaluate_atlas_graph(
                self.cache, self.architecture, graph, profile=profile
            )

        self.assertEqual(
            [result.evaluator_id for result in evaluation.results[1::2]],
            [f"rule4ml_fpga_{activation}_v1" for activation in activations],
        )
        self.assertEqual(
            [call.kwargs["activation"] for call in predict.call_args_list],
            activations,
        )

    def test_registry_has_an_evaluator_for_each_type(self):
        registry = build_default_evaluator_registry()
        for operation_type in SIMPLE_ACTIVATIONS:
            for evaluator_id in (
                f"legacy_fpga_{operation_type}_v1",
                f"sa_fused_{operation_type}_v1",
                f"rule4ml_fpga_{operation_type}_v1",
                f"rule4ml_hybrid_{operation_type}_v1",
            ):
                with self.subTest(evaluator_id=evaluator_id):
                    self.assertEqual(
                        registry.get(evaluator_id).operation_type, operation_type
                    )
        for operation_type in ("sigmoid", "tanh"):
            self.assertEqual(
                registry.get(f"rule4ml_fpga_{operation_type}_v1").operation_type,
                operation_type,
            )


if __name__ == "__main__":
    unittest.main()
