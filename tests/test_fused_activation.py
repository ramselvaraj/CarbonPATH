import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import main
from main import evaluate_atlas_design_point, evaluate_atlas_graph
from system.utils.AtlasAnnealingMoves import mutate_atlas_design_point
from system.utils.AtlasGraphAdapter import (
    gemm_sequence_to_atlas_graph,
    load_atlas_graph,
)
from system.utils.AtlasObjective import RawWeightedSumObjective
from system.utils.DesignPointFile import design_point_document, parse_design_point
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.SimulationCache import SimulationCache
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.WorkloadMapping import (
    FUSED_ACTIVATION_POLICY,
    initial_workload_mapping,
)


GRAPH = Path("cfg/examples/atlas/dense_relu_funnel.graph_dump.json")
MIXED_GRAPH = Path("cfg/examples/atlas/dense_relu_softmax.graph_dump.json")
ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
SA_ONLY_ARCHITECTURE = Path("cfg/examples/sa_only_architecture.json")
PROFILE = Path("cfg/profiles/atlas_fused_relu_v1.json")
UNFUSED_PROFILE = Path("cfg/profiles/atlas_multi_sa_fpga_v2.json")
SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space.json")
SHAPES = [(128, 128, 1024), (128, 1024, 512), (128, 512, 256), (128, 256, 64)]
RELU_INDEXES = (1, 3, 5)

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


def _fuse_all(mapping, outgoing_policy):
    for index in RELU_INDEXES:
        mapping = mapping.with_fusion_change(
            index,
            "sa",
            mapping.operations[index - 1].endpoint_ids,
            FUSED_ACTIVATION_POLICY,
            outgoing_policy,
        )
    return mapping


class FusedActivationTests(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        self.sa_only = _load(SA_ONLY_ARCHITECTURE)
        self.graph = load_atlas_graph(GRAPH)
        self.profile = load_evaluation_profile(PROFILE)
        self.search_space = _load(SEARCH_SPACE)
        self.unfused = initial_workload_mapping(
            self.graph, self.architecture, self.profile, self.search_space
        )
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        cache_path = Path(self._directory.name) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        self.cache = SimulationCache(cache_path, simulator_dir=self._directory.name)

    def _evaluate(self, mapping, architecture=None, graph=None, profile=None):
        return evaluate_atlas_graph(
            self.cache,
            architecture or self.architecture,
            graph or self.graph,
            profile=profile or self.profile,
            mapping=mapping,
        )

    def test_first_mapping_keeps_the_relu_on_the_fpga(self):
        self.assertFalse(any(self.unfused.is_fused(i) for i in RELU_INDEXES))
        self.assertEqual(self.unfused.fusable_indexes, RELU_INDEXES)
        self.assertEqual(self.unfused.operations[1].endpoint_kind, "fpga")

    def test_first_mapping_without_an_fpga_fuses_each_relu(self):
        mapping = initial_workload_mapping(
            self.graph, self.sa_only, self.profile, self.search_space
        )

        for index in RELU_INDEXES:
            self.assertTrue(mapping.is_fused(index))
            self.assertEqual(mapping.operations[index].endpoint_kind, "sa")
            self.assertEqual(mapping.handoff_kind(index - 1), "sa_fused")
            self.assertIn(
                mapping.handoffs[index],
                self.search_space["movement_policy"]["sa_to_sa"],
            )

    def test_softmax_without_an_fpga_is_unsupported(self):
        with self.assertRaisesRegex(UnsupportedEvaluation, "requires an FPGA"):
            initial_workload_mapping(
                load_atlas_graph(MIXED_GRAPH),
                self.sa_only,
                self.profile,
                self.search_space,
            )

    def test_fused_relu_has_no_movement_and_no_compute_cost(self):
        evaluation = self._evaluate(_fuse_all(self.unfused, "local_sram_v1"))

        for index in RELU_INDEXES:
            result = evaluation.results[index]
            self.assertEqual(result.evaluator_id, "sa_fused_relu_v1")
            self.assertEqual(result.endpoint_kind, "sa")
            self.assertEqual(result.endpoint_ids, (0,))
            self.assertEqual(result.movement.method, "fused")
            self.assertEqual(result.movement.charged_latency_ns, 0.0)
            self.assertEqual(result.compute_latency_ns, 0.0)
            self.assertEqual(result.compute_energy_pj, 0.0)

    def test_fusion_is_faster_than_the_fpga_path(self):
        unfused = self._evaluate(self.unfused)
        fused = self._evaluate(_fuse_all(self.unfused, "local_sram_v1"))

        self.assertLess(fused.latency_ns, unfused.latency_ns)
        self.assertLess(fused.movement_energy_pj, unfused.movement_energy_pj)

    def test_fused_graph_scores_the_same_as_the_gemm_chain(self):
        gemm_graph = gemm_sequence_to_atlas_graph(
            {
                "name": "funnel_gemms",
                "gemms": [
                    {"name": f"gemm_{index}", "shape": list(shape)}
                    for index, shape in enumerate(SHAPES)
                ],
            }
        )
        for policy in ("cold_dram_v1", "local_sram_v1", "direct_forward_v1"):
            with self.subTest(policy=policy):
                gemm_mapping = initial_workload_mapping(
                    gemm_graph,
                    self.architecture,
                    self.profile,
                    {"movement_policy": {"sa_to_sa": [policy]}},
                )
                chain = self._evaluate(gemm_mapping, graph=gemm_graph)
                fused = self._evaluate(_fuse_all(self.unfused, policy))

                self.assertAlmostEqual(fused.latency_ns, chain.latency_ns)
                self.assertAlmostEqual(
                    fused.compute_energy_pj, chain.compute_energy_pj
                )
                self.assertAlmostEqual(
                    fused.movement_energy_pj, chain.movement_energy_pj
                )

    def test_gemm_before_a_fused_relu_reads_the_next_handoff(self):
        simulated = {
            "latency_ns": 1.0,
            "dram_interconnect_energy_pj": 2.0,
            "sram_energy_pj": 3.0,
        }
        for policy, uses_dram in (("cold_dram_v1", True), ("local_sram_v1", False)):
            with self.subTest(policy=policy):
                with patch.object(
                    main, "simulate_single_gemm", return_value=simulated
                ) as simulate:
                    self._evaluate(_fuse_all(self.unfused, policy))
                calls = [call.kwargs for call in simulate.call_args_list]
                self.assertEqual(len(calls), 4)
                self.assertEqual(
                    [call["output_to_dram"] for call in calls],
                    [uses_dram, uses_dram, uses_dram, True],
                )
                self.assertEqual(
                    [call["activation_from_dram"] for call in calls],
                    [True, uses_dram, uses_dram, uses_dram],
                )

    def test_fused_relu_settings_add_latency_and_energy(self):
        profile = self.profile.with_evaluator_setting(
            "relu@sa", "latency_ns", 7.0
        ).with_evaluator_setting("relu@sa", "energy_per_element_pj", 0.5)
        mapping = _fuse_all(self.unfused, "local_sram_v1")

        free = self._evaluate(mapping)
        charged = self._evaluate(mapping, profile=profile)

        elements = sum(free.results[index].element_count for index in RELU_INDEXES)
        self.assertAlmostEqual(charged.latency_ns, free.latency_ns + 3 * 7.0)
        self.assertAlmostEqual(
            charged.compute_energy_pj, free.compute_energy_pj + 0.5 * elements
        )

    def test_profile_without_a_fused_evaluator_is_unsupported(self):
        with self.assertRaisesRegex(UnsupportedEvaluation, "relu@sa"):
            self._evaluate(
                _fuse_all(self.unfused, "local_sram_v1"),
                profile=load_evaluation_profile(UNFUSED_PROFILE),
            )

    def test_fused_graph_runs_on_an_architecture_without_an_fpga(self):
        mapping = initial_workload_mapping(
            self.graph, self.sa_only, self.profile, self.search_space
        )
        objective = RawWeightedSumObjective()

        with_fpga, _ = evaluate_atlas_design_point(
            self.cache,
            self.architecture,
            self.graph,
            self.profile,
            objective,
            mapping=self.unfused,
        )
        sa_only, _ = evaluate_atlas_design_point(
            self.cache,
            self.sa_only,
            self.graph,
            self.profile,
            objective,
            mapping=mapping,
        )

        for metric in (
            "latency_ns",
            "total_energy_pj",
            "power_w",
            "area_mm2",
            "cost_usd",
            "embodied_carbon_kg",
            "operational_carbon_kg",
        ):
            with self.subTest(metric=metric):
                self.assertLess(sa_only.metric(metric), with_fpga.metric(metric))

    def test_mapping_rejects_a_fused_relu_on_other_arrays(self):
        fused = _fuse_all(self.unfused, "local_sram_v1")
        with self.assertRaisesRegex(UnsupportedEvaluation, "must use the systolic"):
            fused.with_fusion_change(
                1, "sa", (5,), FUSED_ACTIVATION_POLICY, "local_sram_v1"
            )
        with self.assertRaisesRegex(UnsupportedEvaluation, "not fused"):
            self.unfused.with_fusion_change(
                1, "sa", (0,), "local_sram_v1", "local_sram_v1"
            )
        with self.assertRaisesRegex(UnsupportedEvaluation, "cannot be fused"):
            self.unfused.with_handoff(1, FUSED_ACTIVATION_POLICY)

    def test_fused_relu_follows_the_sa_set_of_its_gemm(self):
        architecture = copy.deepcopy(self.architecture)
        architecture["Chiplet_3"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"]["Chiplet_3"] = 4
        architecture["pkg"]["inter_pkg_conn"].append(
            {
                "from": "Chiplet_1",
                "to": "Chiplet_3",
                "connection_type": "2.5d_emib",
                "loc": "2.5d_chiplet",
            }
        )
        mapping = _fuse_all(
            initial_workload_mapping(
                self.graph, architecture, self.profile, self.search_space
            ),
            "local_sram_v1",
        )
        self.assertEqual(mapping.operations[1].endpoint_ids, (0, 2))

        moved = mapping.with_operation(0, endpoint_ids=(2,))
        self.assertEqual(moved.operations[1].endpoint_ids, (2,))

        relabeled = mapping.with_endpoints_relabeled({0: 0, 1: 1})
        self.assertEqual(relabeled.operations[0].endpoint_ids, (0,))
        self.assertEqual(relabeled.operations[1].endpoint_ids, (0,))

    def test_design_point_file_keeps_the_fusion(self):
        mapping = _fuse_all(self.unfused, "local_sram_v1")
        document = design_point_document(
            self.architecture, mapping, self.graph, self.profile
        )

        loaded = parse_design_point(json.loads(json.dumps(document)))

        self.assertEqual(loaded.mapping, mapping)
        self.assertEqual(loaded.mapping.fingerprint(), mapping.fingerprint())

    def _fusion_params(self):
        params = copy.deepcopy(self.search_space)
        params["move_weights"] = {"relu_fusion": 1.0}
        return params

    def test_fusion_move_changes_the_endpoint_and_both_handoffs(self):
        fused, mapping, move = mutate_atlas_design_point(
            self.architecture,
            self.unfused,
            self._fusion_params(),
            rng=random.Random(0),
            default_movement_policy=self.profile.movement_policy,
            fusion_types=("relu",),
        )
        self.assertEqual(move, "relu_fusion")
        self.assertEqual(fused, self.architecture)
        index = next(i for i in RELU_INDEXES if mapping.is_fused(i))
        self.assertEqual(mapping.operations[index].endpoint_kind, "sa")
        self.assertEqual(mapping.handoffs[index], "direct_forward_v1")
        self._evaluate(mapping)

        all_fused = _fuse_all(self.unfused, "local_sram_v1")
        _, back, _ = mutate_atlas_design_point(
            self.architecture,
            all_fused,
            self._fusion_params(),
            rng=random.Random(0),
            default_movement_policy=self.profile.movement_policy,
            fusion_types=("relu",),
        )
        index = next(i for i in RELU_INDEXES if not back.is_fused(i))
        self.assertEqual(back.operations[index].endpoint_kind, "fpga")
        self.assertEqual(back.operations[index].endpoint_ids, (1,))
        self.assertEqual(back.handoffs[index - 1], self.profile.movement_policy)
        self.assertEqual(back.handoffs[index], self.profile.movement_policy)
        self._evaluate(back)

    def test_fusion_move_is_off_without_a_fused_evaluator(self):
        with self.assertRaisesRegex(UnsupportedEvaluation, "enabled move"):
            mutate_atlas_design_point(
                self.architecture,
                self.unfused,
                self._fusion_params(),
                rng=random.Random(0),
            )

    def test_fused_relu_cannot_leave_an_architecture_without_an_fpga(self):
        mapping = initial_workload_mapping(
            self.graph, self.sa_only, self.profile, self.search_space
        )
        with self.assertRaisesRegex(UnsupportedEvaluation, "enabled move"):
            mutate_atlas_design_point(
                self.sa_only,
                mapping,
                self._fusion_params(),
                rng=random.Random(0),
                default_movement_policy=self.profile.movement_policy,
                fusion_types=("relu",),
            )

    def test_handoff_move_does_not_change_a_fused_handoff(self):
        mapping = _fuse_all(self.unfused, "local_sram_v1")
        params = copy.deepcopy(self.search_space)
        params["move_weights"] = {"handoff_movement": 1.0}

        for seed in range(20):
            _, moved, _ = mutate_atlas_design_point(
                self.architecture, mapping, params, rng=random.Random(seed)
            )
            for index in RELU_INDEXES:
                self.assertTrue(moved.is_fused(index))


if __name__ == "__main__":
    unittest.main()
