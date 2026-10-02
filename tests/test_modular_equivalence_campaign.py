import json
import tempfile
import unittest
from pathlib import Path

from script.validate_modular_equivalence_campaign import compare_campaigns


def architecture(array="64x64", dataflow="ws"):
    return {
        "Chiplet_1": {"sys_array_size": array, "tech_node": "7", "sram_buf": 256},
        "pkg": {"HI_pkg_type": "2d", "mem_pkg_conn": {"mem_type": "ddr5"}},
        "WL_mapping": {
            "mapping": {
                "dataflow": [dataflow],
                "if_splitting_k": 0,
                "assign_workload_in_ascending_order": 0,
            }
        },
    }


def profile(dataflow):
    return {
        "profile": "atlas_modular_v1",
        "version": 1,
        "evaluators": {
            "gemm": {
                "id": "legacy_scale_sim_gemm_v1",
                "settings": {"dataflow": dataflow},
            }
        },
        "placement_policy": "all_sas_v1",
        "movement_policy": "direct_forward_v1",
        "transfer_model": "route_transfer_v1",
    }


def write_campaign(root, designs, costs, *, modular=False, duplicate_seeds=False):
    root.mkdir()
    manifest = {
        "workloads": [7],
        "runs_per_workload": len(designs),
        "schedule_name": "same_schedule",
        "workload_config_sha256": {"7": "same_workload"},
        "evaluation_flow": "modular" if modular else "original",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for index, (design, cost) in enumerate(zip(designs, costs), start=1):
        run = root / "runs" / "workload_7" / f"run_{index:02d}"
        run.mkdir(parents=True)
        (run / "best_architecture.json").write_text(
            json.dumps(design), encoding="utf-8"
        )
        (run / "result.json").write_text(
            json.dumps({
                "workload_id": 7,
                "run": index,
                "initial_seed": 1 if duplicate_seeds else 1000 + index,
                "search_seed": 2 if duplicate_seeds else 2000 + index,
                "verified_best_cost": cost,
            }),
            encoding="utf-8",
        )
        if modular:
            (run / "best_profile.json").write_text(
                json.dumps(profile(design["WL_mapping"]["mapping"]["dataflow"][0])),
                encoding="utf-8",
            )


class ModularEquivalenceCampaignTests(unittest.TestCase):
    def test_sixteen_of_twenty_matching_runs_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = architecture()
            other = architecture(array="128x128")
            write_campaign(root / "original", [target] * 12 + [other] * 8, [10.0] * 20)
            write_campaign(
                root / "modular", [target] * 16 + [other] * 4, [10.0] * 20,
                modular=True,
            )

            report = compare_campaigns(root / "original", root / "modular")

            self.assertTrue(report["passed"])
            self.assertEqual(report["workloads"][0]["matching_runs"], 16)

    def test_same_hardware_and_score_allows_mapping_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = architecture(dataflow="ws")
            mapping_variant = architecture(dataflow="os")
            write_campaign(root / "original", [target] * 20, [10.0] * 20)
            write_campaign(
                root / "modular", [mapping_variant] * 16 + [architecture("128x128")] * 4,
                [10.005] * 16 + [10.0] * 4,
                modular=True,
            )

            report = compare_campaigns(root / "original", root / "modular")

            self.assertTrue(report["passed"])
            self.assertEqual(report["workloads"][0]["equivalent_score_runs"], 16)

    def test_fifteen_of_twenty_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = architecture()
            other = architecture(array="128x128")
            write_campaign(root / "original", [target] * 20, [10.0] * 20)
            write_campaign(
                root / "modular", [target] * 15 + [other] * 5, [10.0] * 20,
                modular=True,
            )

            report = compare_campaigns(root / "original", root / "modular")

            self.assertFalse(report["passed"])
            self.assertEqual(report["workloads"][0]["matching_runs"], 15)

    def test_duplicate_seeds_are_rejected_as_non_independent_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = architecture()
            write_campaign(
                root / "original", [target] * 20, [10.0] * 20,
                duplicate_seeds=True,
            )
            write_campaign(root / "modular", [target] * 20, [10.0] * 20, modular=True)

            with self.assertRaisesRegex(ValueError, "independent"):
                compare_campaigns(root / "original", root / "modular")


if __name__ == "__main__":
    unittest.main()
