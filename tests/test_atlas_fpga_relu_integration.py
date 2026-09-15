import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from system.utils.AtlasWorkload import load_atlas_workload
from system.utils.SimulationCache import SimulationCache
from system.utils.Simulator import Simulator
from network import evaluate_atlas_network


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

ARCHITECTURE_PATH = Path("cfg/examples/sa_fpga_architecture.json")
WORKLOAD_PATH = Path("cfg/examples/atlas_gemm_relu_gemm.json")


def _load(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


class AtlasFpgaReluIntegrationTests(unittest.TestCase):
    def _evaluate(self):
        architecture = _load(ARCHITECTURE_PATH)
        workload = load_atlas_workload(WORKLOAD_PATH)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache_path = Path(directory.name) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        cache = SimulationCache(cache_path, simulator_dir=directory.name)
        original = Simulator.simulate_single_core
        with patch.object(
            Simulator,
            "simulate_single_core",
            autospec=True,
            side_effect=original,
        ) as simulate_core:
            evaluation = evaluate_atlas_network(workload, architecture, cache)
        return evaluation, simulate_core

    def test_mixed_sequence_has_two_gemms_and_one_relu(self):
        evaluation, _ = self._evaluate()
        self.assertEqual(evaluation.summary["gemm_count"], 2)
        self.assertEqual(evaluation.summary["relu_count"], 1)
        self.assertEqual(evaluation.summary["operation_count"], 3)

    def test_relu_uses_resource_derived_lanes(self):
        evaluation, _ = self._evaluate()
        relu = evaluation.layers[evaluation.layers["operation_type"] == "relu"].iloc[0]
        self.assertEqual(relu["parallel_lanes"], 64)
        self.assertEqual(relu["compute_cycles"], 8)
        self.assertTrue(relu["resource_feasible"])

    def test_relu_stage_is_serialized_transfer_compute_transfer(self):
        evaluation, _ = self._evaluate()
        relu = evaluation.layers[evaluation.layers["operation_type"] == "relu"].iloc[0]
        expected = (
            relu["input_transfer_latency_ns"]
            + relu["compute_latency_ns"]
            + relu["output_transfer_latency_ns"]
        )
        self.assertAlmostEqual(relu["latency_ns"], expected)

    def test_relu_owns_both_transfer_legs(self):
        evaluation, _ = self._evaluate()
        relu = evaluation.layers[evaluation.layers["operation_type"] == "relu"].iloc[0]
        self.assertEqual(relu["input_transfer_bytes"], 512)
        self.assertEqual(relu["output_transfer_bytes"], 512)
        self.assertGreater(relu["communication_energy_pj"], 0)

    def test_total_latency_is_sum_of_operations(self):
        evaluation, _ = self._evaluate()
        expected = evaluation.layers["latency_ns"].sum()
        self.assertAlmostEqual(evaluation.summary["latency_ns"], expected)

    def test_relu_does_not_invoke_scale_sim(self):
        # Both GEMMs share one shape, so the cache deduplicates to one run.
        _, simulate_core = self._evaluate()
        self.assertEqual(simulate_core.call_count, 1)

    def test_gemm_output_write_is_suppressed_before_relu(self):
        # gemm_1 feeds the ReLU, so it must not write its output to DRAM.
        # Its communication energy is activation + weight load only.
        evaluation, _ = self._evaluate()
        gemm_1 = evaluation.layers.iloc[0]
        expected_words = 8 * 64 + 64 * 64
        expected_energy = expected_words * 8 * 5  # hbm2 pJ/bit
        self.assertAlmostEqual(gemm_1["communication_energy_pj"], expected_energy)


if __name__ == "__main__":
    unittest.main()
