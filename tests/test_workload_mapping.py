import tempfile
import unittest
from pathlib import Path

import pandas as pd

from main import evaluate_atlas_graph
from script.validate_intermediate_memory_capacity import (
    build_remote_validation_architecture,
)
from system.utils.AtlasGraphAdapter import gemm_sequence_to_atlas_graph
from system.utils.EvaluationProfile import parse_evaluation_profile
from system.utils.SimulationCache import SimulationCache
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.WorkloadMapping import initial_workload_mapping


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
SHAPES = ((128, 256, 512), (128, 512, 64), (128, 64, 32))


def _graph():
    return gemm_sequence_to_atlas_graph(
        {
            "name": "three_gemm_mapping",
            "gemms": [
                {"name": f"stage_{index}", "shape": list(shape)}
                for index, shape in enumerate(SHAPES, start=1)
            ],
        }
    )


def _profile(policy="direct_forward"):
    return parse_evaluation_profile(
        {
            "profile": f"mapping_{policy}_v1",
            "version": 1,
            "evaluators": {"gemm": "legacy_scale_sim_gemm_v1"},
            "assignment_policy": "all_sas_single_fpga_v1",
            "movement_policy": f"{policy}_v1",
            "transfer_model": "route_transfer_v1",
        }
    )


class WorkloadMappingTests(unittest.TestCase):
    def setUp(self):
        self.architecture = build_remote_validation_architecture()
        self.graph = _graph()
        self.profile = _profile()
        self.mapping = initial_workload_mapping(
            self.graph, self.architecture, self.profile
        )
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        cache_path = Path(self._directory.name) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        self.cache = SimulationCache(cache_path, simulator_dir=self._directory.name)

    def _evaluate(self, mapping=None):
        return evaluate_atlas_graph(
            self.cache,
            self.architecture,
            self.graph,
            profile=self.profile,
            mapping=mapping,
        )

    def test_first_mapping_uses_all_sas_and_the_profile_settings(self):
        mapping_settings = self.architecture["WL_mapping"]["mapping"]
        for operation in self.mapping.operations:
            self.assertEqual(operation.endpoint_ids, (0, 1))
            self.assertEqual(operation.dataflow, mapping_settings["dataflow"][0])
            self.assertEqual(operation.split_k, True)
        self.assertEqual(self.mapping.handoffs, ("direct_forward_v1",) * 2)

    def test_first_mapping_takes_a_listed_policy_when_the_profile_has_none(self):
        search_space = {"movement_policy": {"sa_to_sa": ["local_sram_v1"]}}
        mapping = initial_workload_mapping(
            self.graph, self.architecture, self.profile, search_space
        )
        self.assertEqual(mapping.handoffs, ("local_sram_v1",) * 2)

    def test_first_mapping_scores_the_same_as_no_mapping(self):
        without = self._evaluate()
        with_mapping = self._evaluate(self.mapping)

        self.assertAlmostEqual(with_mapping.latency_ns, without.latency_ns)
        self.assertAlmostEqual(
            with_mapping.compute_energy_pj, without.compute_energy_pj
        )
        self.assertAlmostEqual(
            with_mapping.movement_energy_pj, without.movement_energy_pj
        )
        self.assertEqual(
            [result.tile_mappings for result in with_mapping.results],
            [result.tile_mappings for result in without.results],
        )

    def test_a_gemm_runs_only_on_its_sa_set(self):
        mapping = self.mapping.with_operation(0, endpoint_ids=(1,))

        evaluation = self._evaluate(mapping)

        self.assertEqual(evaluation.results[0].endpoint_ids, (1,))
        self.assertEqual(
            {tile["core_id"] for tile in evaluation.results[0].tile_mappings}, {1}
        )
        self.assertEqual(
            {tile["core_id"] for tile in evaluation.results[1].tile_mappings},
            {0, 1},
        )
        covered = sum(
            tile["m"] * tile["k"] * tile["n"]
            for tile in evaluation.results[0].tile_mappings
        )
        self.assertEqual(covered, SHAPES[0][0] * SHAPES[0][1] * SHAPES[0][2])
        self.assertNotAlmostEqual(
            evaluation.latency_ns, self._evaluate(self.mapping).latency_ns
        )

    def test_each_gemm_uses_its_own_settings(self):
        mapping = self.mapping.with_operation(1, split_k=False, dataflow="os")

        evaluation = self._evaluate(mapping)
        baseline = self._evaluate(self.mapping)

        self.assertEqual(
            evaluation.results[0].tile_mappings, baseline.results[0].tile_mappings
        )
        self.assertNotEqual(
            evaluation.results[1].tile_mappings, baseline.results[1].tile_mappings
        )

    def test_each_handoff_uses_its_own_movement_policy(self):
        mapping = self.mapping.with_handoff(0, "cold_dram_v1")

        evaluation = self._evaluate(mapping)

        first, second = (result.movement for result in evaluation.results[1:])
        self.assertEqual(first.requested_policy, "cold_dram")
        self.assertTrue(first.uses_dram)
        self.assertEqual(second.requested_policy, "direct_forward")
        self.assertFalse(second.uses_dram)

    def test_a_change_gives_a_new_fingerprint(self):
        changed = (
            self.mapping.with_operation(0, endpoint_ids=(0,)),
            self.mapping.with_operation(0, dataflow="os"),
            self.mapping.with_operation(2, split_k=False),
            self.mapping.with_handoff(1, "cold_dram_v1"),
        )
        fingerprints = {mapping.fingerprint() for mapping in changed}
        self.assertEqual(len(fingerprints), len(changed))
        self.assertNotIn(self.mapping.fingerprint(), fingerprints)

    def test_a_mapping_for_a_different_graph_is_rejected(self):
        shorter = gemm_sequence_to_atlas_graph(
            {"name": "one", "gemms": [{"name": "only", "shape": list(SHAPES[0])}]}
        )
        with self.assertRaisesRegex(UnsupportedEvaluation, "does not match"):
            evaluate_atlas_graph(
                self.cache,
                self.architecture,
                shorter,
                profile=self.profile,
                mapping=self.mapping,
            )

    def test_a_missing_endpoint_is_rejected(self):
        mapping = self.mapping.with_operation(0, endpoint_ids=(0, 5))
        with self.assertRaisesRegex(UnsupportedEvaluation, "missing endpoint"):
            self._evaluate(mapping)

    def test_an_empty_sa_set_is_rejected(self):
        with self.assertRaisesRegex(UnsupportedEvaluation, "no endpoint"):
            self.mapping.with_operation(0, endpoint_ids=())

    def test_relabel_drops_a_removed_sa_and_adds_a_new_one(self):
        mapping = self.mapping.with_operation(0, endpoint_ids=(0,))

        relabeled = mapping.with_endpoints_relabeled({0: 1}, added_sa_ids=(0,))

        self.assertEqual(relabeled.operations[0].endpoint_ids, (0, 1))
        self.assertEqual(relabeled.operations[1].endpoint_ids, (0, 1))
        with self.assertRaisesRegex(UnsupportedEvaluation, "no endpoint"):
            mapping.with_endpoints_relabeled({1: 0})


if __name__ == "__main__":
    unittest.main()
