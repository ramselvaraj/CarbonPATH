import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from main import evaluate_atlas_graph
from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.SimulationCache import SimulationCache
from system.utils.WorkloadMapping import initial_workload_mapping

GRAPH = Path("cfg/examples/atlas/dense_layernorm.graph_dump.json")
PROFILE = Path("cfg/profiles/atlas_modular_layernorm_v1.json")
ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space.json")
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


class LayerNormGraphTests(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        # The default source is the bitstream; these tests cover the DRAM load.
        self.architecture["Chiplet_2"]["parameter_source"] = "dram"
        self.graph = load_atlas_graph(GRAPH)
        self.profile = load_evaluation_profile(PROFILE)
        self.mapping = initial_workload_mapping(
            self.graph, self.architecture, self.profile, _load(SEARCH_SPACE)
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache_path = Path(directory.name) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        self.cache = SimulationCache(cache_path, simulator_dir=directory.name)

    def test_graph_is_a_gemm_then_a_layernorm(self):
        self.assertEqual(
            [operation.operation_type for operation in self.graph.operations],
            ["gemm", "layernorm"],
        )

    def test_layernorm_starts_on_the_fpga(self):
        kinds = [operation.endpoint_kind for operation in self.mapping.operations]
        self.assertEqual(kinds, ["sa", "fpga"])

    def _evaluate(self):
        return evaluate_atlas_graph(
            self.cache,
            self.architecture,
            self.graph,
            profile=self.profile,
            mapping=self.mapping,
        )

    def test_graph_evaluates_through_the_layernorm_evaluator(self):
        evaluation = self._evaluate()
        self.assertEqual(
            [result.evaluator_id for result in evaluation.results],
            ["legacy_scale_sim_gemm_v1", "fpga_layernorm_v1"],
        )
        self.assertGreater(evaluation.results[1].compute_latency_ns, 0.0)

    def test_the_gemm_hides_part_of_the_parameter_load(self):
        evaluation = self._evaluate()
        gemm, layernorm = evaluation.results
        load = layernorm.parameter_load
        self.assertIsNotNone(load)
        # The GEMM runs on the systolic array while the FPGA is idle.
        self.assertAlmostEqual(
            load.exposed_latency_ns,
            max(0.0, load.raw_latency_ns - gemm.compute_latency_ns),
        )
        self.assertEqual(layernorm.parameter_load_latency_ns, load.exposed_latency_ns)

    def test_exposed_load_and_its_energy_are_charged(self):
        evaluation = self._evaluate()
        layernorm = evaluation.results[1]
        movement = layernorm.movement
        expected = sum(
            result.compute_latency_ns for result in evaluation.results
        ) + movement.charged_latency_ns + layernorm.parameter_load_latency_ns
        self.assertAlmostEqual(evaluation.latency_ns, expected)
        self.assertGreaterEqual(
            evaluation.movement_energy_pj,
            movement.charged_energy_pj + layernorm.parameter_load.energy_pj,
        )

    def test_bitstream_default_has_no_parameter_load(self):
        architecture = _load(ARCHITECTURE)
        evaluation = evaluate_atlas_graph(
            self.cache,
            architecture,
            self.graph,
            profile=self.profile,
            mapping=self.mapping,
        )
        layernorm = evaluation.results[1]
        self.assertIsNone(layernorm.parameter_load)
        self.assertEqual(layernorm.parameter_load_latency_ns, 0.0)
        self.assertEqual(layernorm.parameter_load_energy_pj, 0.0)

    def test_the_first_operation_has_no_slack(self):
        architecture = self.architecture
        graph = self.graph
        # A graph that starts with the LayerNorm cannot hide its load.
        from system.utils.AtlasGraphAdapter import parse_atlas_graph

        nodes = [
            node
            for node in _load(GRAPH)
            if node["class"] in ("Input", "LayerNormalization")
        ]
        nodes[1]["inputs"] = [
            {"name": "input_layer", "shape": [8, 16], "precision": "fixed<8,3,RND,WRAP,0>"}
        ]
        solo = parse_atlas_graph(nodes, "solo")
        mapping = initial_workload_mapping(
            solo, architecture, self.profile, _load(SEARCH_SPACE)
        )
        evaluation = evaluate_atlas_graph(
            self.cache, architecture, solo, profile=self.profile, mapping=mapping
        )
        load = evaluation.results[0].parameter_load
        self.assertAlmostEqual(load.exposed_latency_ns, load.raw_latency_ns)
        del graph


if __name__ == "__main__":
    unittest.main()
