import json
import unittest
from pathlib import Path

from system.utils.AtlasAnnealingMoves import sequential_gemm_search_space


SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space.json")
ORIGINAL_SPACE = Path("cfg/parameters/input.json")


class ModularSearchSpaceTests(unittest.TestCase):
    def test_production_space_contains_every_original_hardware_choice(self):
        with SEARCH_SPACE.open(encoding="utf-8") as file:
            search_space = json.load(file)

        package = search_space["pkg"]
        self.assertGreaterEqual(search_space["max_sa_chiplets"], 6)
        self.assertTrue(
            {
                "2d_na",
                "2.5d_emib",
                "2.5d_rdl",
                "3d_tsv",
                "3d_u_bump",
                "3d_hyb_bond",
            }.issubset(package["inter_pkg_architecture"])
        )
        self.assertTrue(
            {"ucie_std", "ucie_adv", "aib", "bow", "ucie_3d"}.issubset(
                package["protocol"]
            )
        )
        self.assertTrue(
            {"ddr4", "ddr5", "hbm2", "hbm3"}.issubset(
                package["mem_pkg_architecture"]
            )
        )

    def test_sequential_gemm_space_preserves_original_move_family_odds(self):
        with ORIGINAL_SPACE.open(encoding="utf-8") as file:
            original = json.load(file)

        search_space = sequential_gemm_search_space(original)
        weights = search_space["move_weights"]

        self.assertEqual(search_space["max_sa_chiplets"], original["max_chiplet"])
        self.assertTrue(search_space["single_sa_ddr_only"])
        self.assertTrue(search_space["preserve_package_type_on_interconnect"])
        self.assertTrue(search_space["regenerate_memory_on_sa_count"])
        self.assertEqual(
            {name for name, weight in weights.items() if weight > 0},
            {
                "sa_count",
                "sa_array",
                "sa_tech_node",
                "sa_sram",
                "gemm_dataflow",
                "gemm_split_k",
                "gemm_assignment_order",
                "mem_type",
                "interconnect",
                "protocol",
            },
        )
        mapping_weight = sum(
            weights[name]
            for name in (
                "gemm_dataflow",
                "gemm_split_k",
                "gemm_assignment_order",
            )
        )
        hardware_weight = sum(weights.values()) - mapping_weight
        self.assertEqual(mapping_weight, hardware_weight)
        self.assertEqual(
            weights["gemm_dataflow"], weights["gemm_split_k"]
        )
        self.assertEqual(weights["sa_count"], weights["protocol"])


if __name__ == "__main__":
    unittest.main()
