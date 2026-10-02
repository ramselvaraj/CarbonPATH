import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from chiplet.n_utils import get_area_power, get_sram_area_energy
from main import calculate_cost, evaluate_atlas_design_point, simulate_latency_energy
from script.validate_intermediate_memory_capacity import build_validation_architecture
from system.utils.ArchitectureIdentity import architecture_fingerprint
from system.utils.AtlasObjective import build_atlas_objective
from system.utils.SimulationCache import SimulationCache
from tests.test_modular_legacy_pointwise import (
    CACHE_COLUMNS,
    _atlas_graph_for_workload,
    _profile,
)


WORKLOAD = {
    "name": "reduced_space_two_gemm",
    "gemms": [
        {"name": "expand", "shape": [8, 16, 32]},
        {"name": "contract", "shape": [8, 32, 16]},
    ],
}


def _calibration():
    calibration = {}
    average_names = {
        "energy": "avg_energy",
        "latency": "avg_latency",
        "area": "avg_area",
        "cost": "avg_dollar_cost",
        "embCarbon": "avg_embCarbon",
        "opeCarbon": "avg_opeCarbon",
    }
    for metric, average_name in average_names.items():
        calibration[average_name] = 1.0
        calibration[f"{metric}_min"] = 0.1
        calibration[f"{metric}_max"] = 1_000_000_000_000.0
        calibration[f"{metric}_mean"] = 1.0
        calibration[f"{metric}_stddev"] = 1.0
        calibration[f"{metric}_median"] = 1.0
    return calibration


def _reduced_architectures():
    base = build_validation_architecture()
    for array in ("64x64", "128x128"):
        for tech_node in ("7", "14"):
            for memory_type in ("ddr4", "ddr5"):
                architecture = copy.deepcopy(base)
                sram = 256 if array == "64x64" else 1024
                logic_area, power = get_area_power(array, tech_node)
                sram_area, _ = get_sram_area_energy(sram, tech_node)
                architecture["Chiplet_1"].update(
                    {
                        "sys_array_size": array,
                        "tech_node": tech_node,
                        "sram_buf": sram,
                        "area": logic_area + sram_area,
                        "power": power,
                    }
                )
                architecture["pkg"]["mem_pkg_conn"]["mem_type"] = memory_type
                yield architecture


class ReducedSpaceEquivalenceTests(unittest.TestCase):
    def test_original_and_modular_flows_score_and_rank_every_design_identically(self):
        graph = _atlas_graph_for_workload(WORKLOAD)
        calibration = _calibration()
        objective = build_atlas_objective(
            objective_id="t1",
            config={"default": "t1"},
            calibration=calibration,
            normalization_mode="min_median",
        )
        original_results = []
        modular_results = []

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
            cache = SimulationCache(cache_path, simulator_dir=directory)

            for architecture in _reduced_architectures():
                identity = architecture_fingerprint(architecture)
                simulation = simulate_latency_energy(
                    cache,
                    architecture,
                    WORKLOAD,
                    intermediate_policy="direct_forward",
                )
                original_score, _, original_raw = calculate_cost(
                    profile_name="t1",
                    cost_avgerage=calibration,
                    system_dict=architecture,
                    cache=cache,
                    workload_sequence=WORKLOAD,
                    intermediate_policy="direct_forward",
                    simulation_result=simulation,
                )
                design_point, modular_score = evaluate_atlas_design_point(
                    cache,
                    architecture,
                    graph,
                    _profile("direct_forward"),
                    objective,
                )

                self.assertAlmostEqual(design_point.latency_ns, original_raw["latency"])
                self.assertAlmostEqual(design_point.total_energy_pj, original_raw["energy"])
                self.assertAlmostEqual(design_point.area_mm2, original_raw["area"])
                self.assertAlmostEqual(design_point.cost_usd, original_raw["dollar"])
                self.assertAlmostEqual(
                    design_point.embodied_carbon_kg, original_raw["embCarbon"]
                )
                self.assertAlmostEqual(
                    design_point.operational_carbon_kg, original_raw["opeCarbon"]
                )
                self.assertAlmostEqual(modular_score, original_score)
                original_results.append((original_score, identity))
                modular_results.append((modular_score, identity))

        self.assertEqual(
            [identity for _, identity in sorted(original_results)],
            [identity for _, identity in sorted(modular_results)],
        )
        self.assertEqual(
            min(original_results)[1],
            min(modular_results)[1],
        )


if __name__ == "__main__":
    unittest.main()
