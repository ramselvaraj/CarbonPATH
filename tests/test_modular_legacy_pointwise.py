import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from main import evaluate_atlas_graph, simulate_latency_energy
from script.validate_intermediate_memory_capacity import (
    build_remote_validation_architecture,
    build_two_gemm_validation_case,
)
from system.utils.AtlasGraphAdapter import gemm_sequence_to_atlas_graph
from system.utils.ArchitectureIdentity import architecture_fingerprint
from system.utils.EvaluationProfile import parse_evaluation_profile
from system.utils.IntermediateMemoryPolicy import INTERMEDIATE_POLICIES
from system.utils.SimulationCache import SimulationCache


ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
REFERENCE = Path("tests/fixtures/original_carbonpath_two_gemm_pointwise.json")
MULTI_SA_REFERENCE = Path("tests/fixtures/original_carbonpath_two_sa_pointwise.json")
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


def _architecture():
    with ARCHITECTURE.open(encoding="utf-8") as file:
        return json.load(file)


def _sequential_gemm_inputs():
    shapes = ((8, 16, 32), (8, 32, 16))
    workload = {
        "name": "two_gemm_pointwise_fixture",
        "gemms": [
            {"name": f"stage_{index}", "shape": list(shape)}
            for index, shape in enumerate(shapes, start=1)
        ],
    }
    return gemm_sequence_to_atlas_graph(workload), workload


def _atlas_graph_for_workload(workload):
    return gemm_sequence_to_atlas_graph(workload)


def _profile(policy):
    return parse_evaluation_profile(
        {
            "profile": f"pointwise_{policy}_v1",
            "version": 1,
            "evaluators": {"gemm": "legacy_scale_sim_gemm_v1"},
            "placement_policy": "all_sas_single_fpga_v1",
            "movement_policy": f"{policy}_v1",
            "transfer_model": "route_transfer_v1",
        }
    )


def _mapping_fingerprint(gemm_metrics):
    mappings = [gemm["tile_mappings"] for gemm in gemm_metrics]
    payload = json.dumps(mappings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:12]


class ModularLegacyPointwiseTests(unittest.TestCase):
    def test_two_gemm_metrics_and_mapping_match_for_every_original_policy(self):
        architecture = _architecture()
        graph, workload = _sequential_gemm_inputs()
        with REFERENCE.open(encoding="utf-8") as file:
            reference = json.load(file)
        self.assertEqual(
            architecture_fingerprint(architecture),
            reference["architecture_fingerprint"],
        )
        self.assertEqual(graph.fingerprint(), reference["atlas_graph_fingerprint"])
        self.assertEqual(workload, reference["workload"])

        for policy in INTERMEDIATE_POLICIES:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                cache_path = Path(directory) / "cache.csv"
                pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
                cache = SimulationCache(cache_path, simulator_dir=directory)

                legacy = simulate_latency_energy(
                    cache,
                    architecture,
                    workload,
                    intermediate_policy=policy,
                )
                modular = evaluate_atlas_graph(
                    cache,
                    architecture,
                    graph,
                    profile=_profile(policy),
                )

                legacy_latency, legacy_communication, legacy_sram = legacy[:3]
                legacy_gemms, legacy_boundaries = legacy[3:]
                expected = reference["policies"][policy]
                self.assertAlmostEqual(legacy_latency, expected["latency_ns"])
                self.assertAlmostEqual(
                    legacy_communication, expected["communication_energy_pj"]
                )
                self.assertAlmostEqual(legacy_sram, expected["sram_energy_pj"])
                self.assertEqual(
                    [boundary.selected_method for boundary in legacy_boundaries],
                    expected["boundary_methods"],
                )
                self.assertEqual(
                    [gemm["tile_mappings"] for gemm in legacy_gemms],
                    reference["tile_mappings"],
                )
                self.assertAlmostEqual(modular.latency_ns, legacy_latency)
                self.assertAlmostEqual(
                    modular.compute_energy_pj + modular.movement_energy_pj,
                    legacy_communication + legacy_sram,
                )
                self.assertEqual(
                    [result.tile_mappings for result in modular.results],
                    [tuple(gemm["tile_mappings"]) for gemm in legacy_gemms],
                )
                self.assertEqual(
                    [result.movement.selected_method for result in modular.results[1:]],
                    [boundary.selected_method for boundary in legacy_boundaries],
                )
                self.assertEqual(
                    [result.movement.uses_dram for result in modular.results[1:]],
                    [boundary.uses_dram for boundary in legacy_boundaries],
                )

    def test_two_sa_split_k_mapping_and_forwarding_match_original_flow(self):
        architecture = build_remote_validation_architecture()
        workload = build_two_gemm_validation_case()
        graph = _atlas_graph_for_workload(workload)
        with MULTI_SA_REFERENCE.open(encoding="utf-8") as file:
            reference = json.load(file)
        self.assertEqual(
            architecture_fingerprint(architecture),
            reference["architecture_fingerprint"],
        )
        self.assertEqual(graph.fingerprint(), reference["atlas_graph_fingerprint"])

        for policy in INTERMEDIATE_POLICIES:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                cache_path = Path(directory) / "cache.csv"
                pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
                cache = SimulationCache(cache_path, simulator_dir=directory)

                legacy = simulate_latency_energy(
                    cache,
                    architecture,
                    workload,
                    intermediate_policy=policy,
                )
                modular = evaluate_atlas_graph(
                    cache,
                    architecture,
                    graph,
                    profile=_profile(policy),
                )

                legacy_latency, legacy_communication, legacy_sram = legacy[:3]
                legacy_gemms, legacy_boundaries = legacy[3:]
                expected = reference["policies"][policy]
                legacy_boundary = legacy_boundaries[0]
                self.assertAlmostEqual(legacy_latency, expected["latency_ns"])
                self.assertAlmostEqual(
                    legacy_communication, expected["communication_energy_pj"]
                )
                self.assertAlmostEqual(legacy_sram, expected["sram_energy_pj"])
                self.assertEqual(
                    _mapping_fingerprint(legacy_gemms),
                    reference["tile_mapping_fingerprint"],
                )
                self.assertEqual(
                    [len(gemm["tile_mappings"]) for gemm in legacy_gemms],
                    reference["tiles_per_gemm"],
                )
                for field in (
                    "selected_method",
                    "retained_bytes",
                    "forwarded_bytes",
                    "dram_spilled_bytes",
                ):
                    self.assertEqual(getattr(legacy_boundary, field), expected[field])
                self.assertAlmostEqual(modular.latency_ns, legacy_latency)
                self.assertAlmostEqual(
                    modular.compute_energy_pj + modular.movement_energy_pj,
                    legacy_communication + legacy_sram,
                )
                self.assertEqual(
                    [result.tile_mappings for result in modular.results],
                    [tuple(gemm["tile_mappings"]) for gemm in legacy_gemms],
                )
                self.assertEqual(
                    [result.movement for result in modular.results[1:]],
                    legacy_boundaries,
                )
                used_cores = {
                    tile["core_id"]
                    for gemm in legacy_gemms
                    for tile in gemm["tile_mappings"]
                }
                self.assertEqual(used_cores, {0, 1})

                if policy == "direct_forward":
                    boundary = modular.results[1].movement
                    self.assertEqual(boundary.selected_method, "direct_forward")
                    self.assertGreater(boundary.forwarded_bytes, 0)
                    self.assertEqual(boundary.dram_spilled_bytes, 0)


if __name__ == "__main__":
    unittest.main()
