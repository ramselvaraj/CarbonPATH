import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from network import (
    _calibration_metadata,
    _load_network_calibration,
    compare_memory_policies,
    evaluate_network,
    write_network_comparison,
    write_network_evaluation,
)
from main import (
    WORKLOAD_CONFIGS,
    calibration_identity,
    parse_workload_entry,
    simulate_latency_energy,
)
from script.validate_intermediate_memory_capacity import (
    CACHE_COLUMNS,
    build_validation_architecture,
)
from system.utils.NetworkWorkload import parse_network_entry
from system.utils.SimulationCache import SimulationCache


def make_network():
    return parse_network_entry(
        {
            "schema_version": 1,
            "name": "two_layer_mlp",
            "dtype": "int8",
            "input": {"batch_size": 128, "features": 128},
            "layers": [
                {"name": "fc1", "op": "linear", "out_features": 128},
                {"name": "fc2", "op": "linear", "out_features": 64},
            ],
            "memory": {"intermediate_policy": "local_sram"},
        }
    )


class NetworkEvaluationTests(unittest.TestCase):
    def make_cache(self, directory):
        cache_path = Path(directory) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        return SimulationCache(cache_path, simulator_dir=directory)

    def test_evaluation_returns_long_form_layer_boundary_and_mapping_results(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = evaluate_network(
                make_network(),
                build_validation_architecture(),
                self.make_cache(directory),
                intermediate_policy="local_sram",
            )

            output_dir = Path(directory) / "results"
            write_network_evaluation(evaluation, output_dir)

            self.assertEqual(evaluation.summary["network"], "two_layer_mlp")
            self.assertEqual(evaluation.summary["requested_policy"], "local_sram")
            self.assertEqual(evaluation.layers["layer_name"].tolist(), ["fc1", "fc2"])
            self.assertEqual(len(evaluation.boundaries), 1)
            self.assertEqual(
                evaluation.boundaries.iloc[0]["selected_method"], "local_sram"
            )
            self.assertTrue(len(evaluation.mapping) > 0)
            self.assertEqual(set(evaluation.mapping["layer_name"]), {"fc1", "fc2"})
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "layers.csv").exists())
            self.assertTrue((output_dir / "boundaries.csv").exists())
            self.assertTrue((output_dir / "mapping.csv").exists())
            self.assertTrue((output_dir / "report.md").exists())
            with (output_dir / "summary.json").open() as file:
                self.assertEqual(json.load(file)["network"], "two_layer_mlp")

    def test_one_layer_network_writes_readable_empty_boundaries(self):
        network = parse_network_entry(
            {
                "schema_version": 1,
                "name": "classifier",
                "dtype": "int8",
                "input": {"batch_size": 128, "features": 128},
                "layers": [
                    {"name": "head", "op": "linear", "out_features": 10}
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            evaluation = evaluate_network(
                network,
                build_validation_architecture(),
                self.make_cache(directory),
            )
            output_dir = Path(directory) / "results"
            write_network_evaluation(evaluation, output_dir)

            boundaries = pd.read_csv(output_dir / "boundaries.csv")

        self.assertTrue(boundaries.empty)
        self.assertIn("selected_method", boundaries.columns)

    def test_comparison_runs_all_explicit_policies_on_one_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            comparison, evaluations = compare_memory_policies(
                make_network(),
                build_validation_architecture(),
                self.make_cache(directory),
            )
            output_dir = Path(directory) / "comparison"
            write_network_comparison(comparison, evaluations, output_dir)

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "policy_comparison.csv").exists())
            self.assertTrue((output_dir / "report.md").exists())

        self.assertEqual(
            comparison["requested_policy"].tolist(),
            ["cold_dram", "ideal_on_chip", "local_sram", "direct_forward"],
        )
        self.assertEqual(len(evaluations), 4)
        cold_latency = comparison.loc[
            comparison["requested_policy"] == "cold_dram", "latency_ns"
        ].iloc[0]
        local_latency = comparison.loc[
            comparison["requested_policy"] == "local_sram", "latency_ns"
        ].iloc[0]
        self.assertLess(local_latency, cold_latency)

    def test_evaluation_rejects_removed_auto_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Unknown intermediate-memory policy"):
                evaluate_network(
                    make_network(),
                    build_validation_architecture(),
                    self.make_cache(directory),
                    intermediate_policy="auto",
                )

    def test_calibration_identity_tracks_intermediate_policy(self):
        workload = make_network().to_workload_sequence()

        self.assertNotEqual(
            calibration_identity({}, workload, "cold_dram"),
            calibration_identity({}, workload, "direct_forward"),
        )

    def test_network_calibration_rejects_a_different_search_space(self):
        network = make_network()
        search_space = {"sys_array": ["64x64"]}
        cost_averages = {
            "_calibration_identity": calibration_identity(
                search_space,
                network.to_workload_sequence(),
                network.intermediate_policy,
            ),
            "avg_energy": 1.0,
            "avg_area": 1.0,
            "avg_dollar_cost": 1.0,
            "avg_latency": 1.0,
            "avg_embCarbon": 1.0,
            "avg_opeCarbon": 1.0,
        }
        for metric in (
            "energy",
            "latency",
            "area",
            "cost",
            "embCarbon",
            "opeCarbon",
        ):
            cost_averages.update(
                {
                    f"{metric}_min": 1.0,
                    f"{metric}_max": 2.0,
                    f"{metric}_stddev": 0.5,
                    f"{metric}_mean": 1.5,
                    f"{metric}_median": 1.5,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps(
                    _calibration_metadata(network, search_space, cost_averages)
                )
            )

            self.assertEqual(
                _load_network_calibration(path, network, search_space),
                cost_averages,
            )
            with self.assertRaisesRegex(ValueError, "search_space_fingerprint"):
                _load_network_calibration(
                    path,
                    network,
                    {"sys_array": ["128x128"]},
                )

    def test_network_calibration_rejects_incomplete_metrics(self):
        network = make_network()
        search_space = {"sys_array": ["64x64"]}
        incomplete = {
            "_calibration_identity": calibration_identity(
                search_space,
                network.to_workload_sequence(),
                network.intermediate_policy,
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps(_calibration_metadata(network, search_space, incomplete))
            )

            with self.assertRaisesRegex(ValueError, "missing required metrics"):
                _load_network_calibration(path, network, search_space)

    def test_compiled_network_preserves_workload_8_numerical_results(self):
        network = parse_network_entry(
            {
                "schema_version": 1,
                "name": "workload_8_parity",
                "dtype": "int8",
                "input": {"batch_size": 128, "features": 128},
                "layers": [
                    {
                        "name": f"block_{index}",
                        "op": "linear",
                        "out_features": 128,
                    }
                    for index in range(1, 5)
                ],
            }
        )
        architecture = build_validation_architecture()

        with tempfile.TemporaryDirectory() as directory:
            cache = self.make_cache(directory)
            evaluation = evaluate_network(
                network,
                architecture,
                cache,
                intermediate_policy="cold_dram",
            )
            legacy = simulate_latency_energy(
                cache,
                architecture,
                parse_workload_entry(8, WORKLOAD_CONFIGS[8]),
                intermediate_policy="cold_dram",
            )

        self.assertAlmostEqual(evaluation.summary["latency_ns"], legacy[0])
        self.assertAlmostEqual(
            evaluation.summary["communication_energy_pj"], legacy[1]
        )
        self.assertAlmostEqual(evaluation.summary["sram_energy_pj"], legacy[2])


if __name__ == "__main__":
    unittest.main()
