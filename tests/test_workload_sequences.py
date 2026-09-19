import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from main import (
    WORKLOAD_CONFIGS,
    add_per_gemm_metrics,
    parse_workload_entry,
    simulate_latency_energy,
)
from system.utils.ChipletSystem import DRAM_pj_per_bit
from system.utils.SimulationCache import SimulationCache
from system.utils.Simulator import Simulator


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


def make_test_architecture():
    return {
        "Chiplet_1": {
            "tech_node": "7",
            "sys_array_size": "128x128",
            "sram_buf": 2048,
            "area": 0.5,
            "power": 1.0,
        },
        "pkg": {
            "HI_pkg_type": "2d",
            "inter_pkg_conn": "2d_na",
            "protocol_3d": "na",
            "protocol_2.5d": "na",
            "mem_pkg_conn": {"mem_type": "ddr5", "Chiplet_1": 2},
        },
        "WL_mapping": {
            "mapping": {
                "assign_workload_in_ascending_order": 1,
                "chiplet_data_sharing_enabled": 0,
                "dataflow": ["ws"],
                "if_splitting_k": 0,
                "merge_tiles": 0,
                "static_tiling": 0,
            }
        },
    }


class WorkloadSequenceTests(unittest.TestCase):
    def test_legacy_gemm_is_normalized_to_a_sequence(self):
        workload = parse_workload_entry(1, [512, 768, 3072])

        self.assertEqual(workload["id"], 1)
        self.assertEqual(workload["name"], "workload_1")
        self.assertEqual(
            workload["gemms"],
            [{"name": "gemm_1", "shape": (512, 768, 3072)}],
        )

    def test_existing_workload_configs_remain_available(self):
        for workload_id in range(1, 7):
            with self.subTest(workload_id=workload_id):
                self.assertEqual(len(WORKLOAD_CONFIGS[workload_id]), 3)
        self.assertEqual(WORKLOAD_CONFIGS[7]["name"], "two_gemm_demo")
        self.assertEqual(len(WORKLOAD_CONFIGS[7]["gemms"]), 2)
        self.assertEqual(WORKLOAD_CONFIGS[10]["name"], "projection_head_demo")
        self.assertEqual(len(WORKLOAD_CONFIGS[10]["gemms"]), 2)
        self.assertEqual(WORKLOAD_CONFIGS[11]["name"], "ffn_up_down_512")
        self.assertEqual(len(WORKLOAD_CONFIGS[11]["gemms"]), 2)

    def test_workload_11_is_a_dimension_compatible_ffn_chain(self):
        workload = parse_workload_entry(11, WORKLOAD_CONFIGS[11])

        self.assertEqual(
            workload["gemms"],
            [
                {"name": "expand", "shape": (512, 768, 3072)},
                {"name": "contract", "shape": (512, 3072, 768)},
            ],
        )
        self.assertEqual(
            sum(m * k * n for m, k, n in (gemm["shape"] for gemm in workload["gemms"])),
            2_415_919_104,
        )
        self.assertEqual(512 * 3072, 1_572_864)

    def test_workload_10_is_a_projection_head_sequence(self):
        workload = parse_workload_entry(10, WORKLOAD_CONFIGS[10])

        self.assertEqual(
            workload["gemms"],
            [
                {"name": "projection", "shape": (128, 256, 512)},
                {"name": "head", "shape": (128, 512, 256)},
            ],
        )
        self.assertEqual(
            sum(m * k * n for m, k, n in (gemm["shape"] for gemm in workload["gemms"])),
            33_554_432,
        )

    def test_chained_gemms_require_compatible_output_and_input_shapes(self):
        workload = parse_workload_entry(
            7,
            {
                "name": "two_gemm_demo",
                "gemms": [
                    {"name": "projection", "shape": [128, 256, 512]},
                    {"name": "classifier", "shape": [128, 512, 64]},
                ],
            },
        )

        self.assertEqual(
            workload["gemms"],
            [
                {"name": "projection", "shape": (128, 256, 512)},
                {"name": "classifier", "shape": (128, 512, 64)},
            ],
        )

        incompatible_entries = [
            {
                "gemms": [
                    {"name": "first", "shape": [128, 256, 512]},
                    {"name": "second", "shape": [64, 512, 64]},
                ]
            },
            {
                "gemms": [
                    {"name": "first", "shape": [128, 256, 512]},
                    {"name": "second", "shape": [128, 256, 64]},
                ]
            },
        ]
        for entry in incompatible_entries:
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    parse_workload_entry(7, entry)

    def test_sequence_objects_require_at_least_two_gemms(self):
        invalid_entries = [
            {"gemms": []},
            {"gemms": [{"name": "only", "shape": [128, 256, 512]}]},
        ]

        for entry in invalid_entries:
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, "at least two GEMMs"):
                    parse_workload_entry(7, entry)

    def test_four_identical_gemms_scale_metrics_by_four(self):
        single = parse_workload_entry(1, WORKLOAD_CONFIGS[8]["gemms"][0]["shape"])
        four = parse_workload_entry(8, WORKLOAD_CONFIGS[8])

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
            cache = SimulationCache(
                cache_path,
                simulator_dir=directory,
            )
            original_simulate = Simulator.simulate_single_core
            with patch.object(
                Simulator,
                "simulate_single_core",
                autospec=True,
                side_effect=original_simulate,
            ) as simulate_core:
                single_metrics = simulate_latency_energy(
                    cache, make_test_architecture(), single
                )
                four_metrics = simulate_latency_energy(
                    cache,
                    make_test_architecture(),
                    four,
                    intermediate_policy="cold_dram",
                )
                default_metrics = simulate_latency_energy(
                    cache, make_test_architecture(), four
                )
                direct_metrics = simulate_latency_energy(
                    cache,
                    make_test_architecture(),
                    four,
                    intermediate_policy="direct_forward",
                )
                ideal_metrics = simulate_latency_energy(
                    cache,
                    make_test_architecture(),
                    four,
                    intermediate_policy="ideal_on_chip",
                )

        single_latency, single_communication, single_sram, single_gemms, single_boundaries = single_metrics
        four_latency, four_communication, four_sram, four_gemms, four_boundaries = four_metrics
        self.assertEqual(single_boundaries, [])
        self.assertEqual(default_metrics, direct_metrics)
        self.assertEqual(len(four_boundaries), 3)
        self.assertEqual(len(four_gemms), 4)
        self.assertEqual(
            [metric["name"] for metric in four_gemms],
            ["block_1", "block_2", "block_3", "block_4"],
        )
        self.assertEqual(
            [metric["shape"] for metric in four_gemms],
            [(128, 128, 128)] * 4,
        )
        expected_dram_words = 128 * 128 + 128 * 128 + 128 * 128
        expected_communication = expected_dram_words * 8 * DRAM_pj_per_bit["ddr5"]
        expected_ideal_words = 6 * 128 * 128
        expected_ideal_communication = (
            expected_ideal_words * 8 * DRAM_pj_per_bit["ddr5"]
        )
        self.assertAlmostEqual(single_communication, expected_communication)
        self.assertAlmostEqual(four_latency, single_latency * 4)
        self.assertAlmostEqual(four_communication, single_communication * 4)
        self.assertAlmostEqual(four_sram, single_sram * 4)
        single_total_energy = single_communication + single_sram + single_latency * 1000
        four_total_energy = four_communication + four_sram + four_latency * 1000
        self.assertAlmostEqual(four_total_energy, single_total_energy * 4)
        ideal_latency, ideal_communication, _, _, ideal_boundaries = ideal_metrics
        self.assertLess(ideal_latency, four_latency)
        self.assertLess(ideal_communication, four_communication)
        self.assertAlmostEqual(ideal_communication, expected_ideal_communication)
        self.assertEqual(
            [boundary.retained_bytes for boundary in ideal_boundaries],
            [128 * 128] * 3,
        )
        self.assertTrue(all(not boundary.uses_dram for boundary in ideal_boundaries))
        for metric in four_gemms:
            self.assertEqual(
                {
                    "latency_ns": metric["latency_ns"],
                    "dram_interconnect_energy_pj": metric[
                        "dram_interconnect_energy_pj"
                    ],
                    "sram_energy_pj": metric["sram_energy_pj"],
                },
                {
                    "latency_ns": single_gemms[0]["latency_ns"],
                    "dram_interconnect_energy_pj": single_gemms[0][
                        "dram_interconnect_energy_pj"
                    ],
                    "sram_energy_pj": single_gemms[0]["sram_energy_pj"],
                },
            )
        self.assertEqual(simulate_core.call_count, 1)
        self.assertEqual(len(cache.cache_df), 1)

    @patch("main.plan_boundary")
    @patch("main.build_boundary_mapping")
    @patch("main.prepare_single_gemm")
    @patch("main.simulate_single_gemm")
    def test_sequence_runs_gemms_in_order_and_sums_metrics(
        self,
        simulate_single,
        prepare_single,
        build_mapping,
        plan_boundary,
    ):
        workload = parse_workload_entry(
            7,
            {
                "name": "two_gemm_demo",
                "gemms": [
                    {"name": "first", "shape": [128, 256, 512]},
                    {"name": "second", "shape": [128, 512, 64]},
                ],
            },
        )
        simulate_single.side_effect = [
            {
                "latency_ns": 10.0,
                "dram_interconnect_energy_pj": 20.0,
                "sram_energy_pj": 30.0,
            },
            {
                "latency_ns": 40.0,
                "dram_interconnect_energy_pj": 50.0,
                "sram_energy_pj": 60.0,
            },
        ]
        prepare_single.side_effect = [(object(), object()), (object(), object())]
        build_mapping.return_value = type(
            "Mapping",
            (),
            {
                "intermediate_bytes": 128 * 512,
                "transfers": (),
                "cores": {},
                "valid": True,
                "error": "",
            },
        )()
        plan_boundary.return_value = type(
            "Plan",
            (),
            {
                "uses_dram": True,
                "selected_method": "cold_dram",
            },
        )()

        latency, communication_energy, sram_energy, gemm_metrics, boundaries = (
            simulate_latency_energy(object(), {"architecture": "candidate"}, workload)
        )

        self.assertEqual(latency, 50.0)
        self.assertEqual(communication_energy, 70.0)
        self.assertEqual(sram_energy, 90.0)
        self.assertEqual([metric["name"] for metric in gemm_metrics], ["first", "second"])
        self.assertEqual(
            [metric["shape"] for metric in gemm_metrics],
            [(128, 256, 512), (128, 512, 64)],
        )
        self.assertEqual(
            [call.args[2] for call in simulate_single.call_args_list],
            [(128, 256, 512), (128, 512, 64)],
        )
        self.assertEqual(boundaries, [plan_boundary.return_value])

    def test_per_gemm_total_energy_includes_compute_memory_and_sram(self):
        output = {}
        add_per_gemm_metrics(
            output,
            [
                {
                    "name": "projection",
                    "shape": (128, 256, 512),
                    "latency_ns": 10,
                    "dram_interconnect_energy_pj": 20,
                    "sram_energy_pj": 30,
                }
            ],
            power=2,
        )

        self.assertEqual(output["gemm_1_name"], "projection")
        self.assertEqual(output["gemm_1_shape"], "128x256x512")
        self.assertEqual(output["gemm_1_total_energy_pj"], 20 + 30 + 2 * 10 * 1000)


if __name__ == "__main__":
    unittest.main()
