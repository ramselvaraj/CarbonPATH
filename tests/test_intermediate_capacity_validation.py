import tempfile
import unittest
from pathlib import Path

import pandas as pd

from chiplet.n_disagg import _validate_protocols
from script.validate_intermediate_memory_capacity import (
    CACHE_COLUMNS,
    build_mixed_boundary_validation_case,
    build_policy_validation_architectures,
    build_remote_validation_architecture,
    build_two_gemm_validation_case,
    build_validation_architecture,
    build_validation_cases,
    run_capacity_validation,
    run_policy_architecture_validation,
    run_two_gemm_mapping_validation,
)
from system.utils.SimulationCache import SimulationCache


class IntermediateCapacityValidationTests(unittest.TestCase):
    def test_validation_cases_straddle_the_supported_sram_capacity(self):
        architecture = build_validation_architecture()
        cases = build_validation_cases()

        self.assertEqual(architecture["Chiplet_1"]["sram_buf"], 256)
        self.assertEqual(cases["fits"]["intermediate_bytes"], 256 * 1024)
        self.assertEqual(
            cases["overflows"]["intermediate_bytes"],
            256 * 1024 + 512,
        )

    def test_live_validation_matches_expected_fit_and_fallback_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
            cache = SimulationCache(cache_path, simulator_dir=directory)

            results = run_capacity_validation(cache)

        indexed = results.set_index(["case", "requested_policy"])
        fit_local = indexed.loc[("fits", "local_sram")]
        overflow_local = indexed.loc[("overflows", "local_sram")]
        fit_ideal = indexed.loc[("fits", "ideal_on_chip")]
        overflow_cold = indexed.loc[("overflows", "cold_dram")]

        self.assertEqual(fit_local["actual_selected_method"], "local_sram")
        self.assertEqual(fit_local["retained_bytes"], 256 * 1024)
        self.assertAlmostEqual(
            fit_local["actual_latency_ns"], fit_ideal["actual_latency_ns"]
        )

        self.assertEqual(overflow_local["actual_selected_method"], "cold_dram")
        self.assertEqual(overflow_local["dram_spilled_bytes"], 256 * 1024 + 512)
        self.assertEqual(overflow_local["dram_traffic_bytes"], 2 * (256 * 1024 + 512))
        self.assertIn("capacity", overflow_local["fallback_reason"])
        self.assertAlmostEqual(
            overflow_local["actual_latency_ns"],
            overflow_cold["actual_latency_ns"],
        )

        self.assertTrue(results["validation_passed"].all())
        self.assertLess(results["latency_error_percent"].max(), 1e-9)
        self.assertLess(results["communication_energy_error_percent"].max(), 1e-9)
        for row in results.itertuples():
            self.assertEqual(row.retained_bytes, row.expected_retained_bytes)
            self.assertEqual(row.forwarded_bytes, row.expected_forwarded_bytes)
            self.assertEqual(row.dram_spilled_bytes, row.expected_dram_spilled_bytes)
            self.assertEqual(row.dram_traffic_bytes, row.expected_dram_traffic_bytes)
            if row.case == "overflows" and row.requested_policy in (
                "local_sram",
                "direct_forward",
            ):
                self.assertIn("capacity", row.fallback_reason)

    def test_workload_7_same_and_remote_core_validation(self):
        workload = build_two_gemm_validation_case()
        remote_architecture = build_remote_validation_architecture()

        self.assertEqual(workload["gemms"][0]["shape"], (128, 256, 512))
        self.assertEqual(workload["gemms"][1]["shape"], (128, 512, 64))
        self.assertIn("Chiplet_2", remote_architecture)
        self.assertEqual(
            remote_architecture["WL_mapping"]["mapping"]["if_splitting_k"], 1
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
            cache = SimulationCache(cache_path, simulator_dir=directory)

            results = run_two_gemm_mapping_validation(cache)

        indexed = results.set_index("case")
        same_core = indexed.loc["same_core"]
        remote_core = indexed.loc["remote_core"]

        self.assertEqual(same_core["selected_method"], "local_sram")
        self.assertEqual(same_core["retained_bytes"], 65536)
        self.assertEqual(same_core["forwarded_bytes"], 0)
        self.assertEqual(same_core["routes"], "")

        self.assertEqual(remote_core["selected_method"], "direct_forward")
        self.assertEqual(remote_core["retained_bytes"], 32768)
        self.assertEqual(remote_core["forwarded_bytes"], 32768)
        self.assertEqual(remote_core["routes"], "0->1")

        self.assertTrue(results["default_matches_direct"].all())
        self.assertTrue(results["validation_passed"].all())

    def test_policy_matrix_distinguishes_all_three_policy_totals(self):
        workload = build_mixed_boundary_validation_case()
        architectures = build_policy_validation_architectures()

        self.assertEqual(len(workload["gemms"]), 3)
        self.assertEqual(
            set(architectures),
            {
                "2.5d_rdl_ucie",
                "2.5d_emib_bow",
                "2.5d_active_ucie_adv",
                "3d_hybrid_bond",
            },
        )
        for architecture in architectures.values():
            package = architecture["pkg"]
            self.assertTrue(
                _validate_protocols(
                    package["protocol_3d"],
                    package["protocol_2.5d"],
                    package["inter_pkg_conn"],
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
            cache = SimulationCache(cache_path, simulator_dir=directory)

            results = run_policy_architecture_validation(cache)

        self.assertEqual(len(results), 12)
        self.assertTrue(results["all_policy_metrics_distinct"].all())
        self.assertTrue(results["accounting_matches"].all())
        self.assertTrue(results["forwarding_oracle_matches"].all())
        self.assertTrue(results["default_matches_direct"].all())
        self.assertTrue(results["validation_passed"].all())

        for _, architecture_rows in results.groupby("architecture"):
            indexed = architecture_rows.set_index("policy")
            self.assertEqual(
                indexed.loc["cold_dram", "boundary_methods"],
                "cold_dram;cold_dram",
            )
            self.assertEqual(
                indexed.loc["local_sram", "boundary_methods"],
                "cold_dram;local_sram",
            )
            self.assertEqual(
                indexed.loc["direct_forward", "boundary_methods"],
                "direct_forward;local_sram",
            )
            self.assertEqual(indexed.loc["cold_dram", "dram_spilled_bytes"], 5120)
            self.assertEqual(indexed.loc["local_sram", "dram_spilled_bytes"], 4096)
            self.assertEqual(indexed.loc["direct_forward", "dram_spilled_bytes"], 0)
            self.assertEqual(indexed.loc["direct_forward", "forwarded_bytes"], 2048)

        direct_rows = results[results["policy"] == "direct_forward"]
        self.assertEqual(direct_rows["forwarding_latency_ns"].nunique(), 4)
        self.assertEqual(direct_rows["forwarding_energy_pj"].nunique(), 4)
        self.assertIn("boundary_1_latency_ns", results.columns)
        self.assertIn("boundary_2_latency_ns", results.columns)
        self.assertTrue((direct_rows["boundary_2_latency_ns"] == 0).all())
        active_direct = direct_rows.set_index("architecture").loc[
            "2.5d_active_ucie_adv"
        ]
        self.assertAlmostEqual(
            active_direct["boundary_1_latency_ns"], 14.794964521535569
        )


if __name__ == "__main__":
    unittest.main()
