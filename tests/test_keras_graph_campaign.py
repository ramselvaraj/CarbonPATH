import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from script.run_optimizer_experiments import run_search
from system.utils.AtlasGraphAdapter import gemm_sequence_to_atlas_graph


class KerasGraphCampaignTests(unittest.TestCase):
    def test_run_search_uses_supplied_graph_with_calibrated_modular_flow(self):
        workload = {
            "name": "single_gemm",
            "gemms": [{"name": "projection", "shape": [128, 256, 512]}],
        }
        graph = gemm_sequence_to_atlas_graph(workload)
        architecture = {"sentinel": "architecture"}
        trace = pd.DataFrame(
            [{"move_accepted": True, "new_cost": 1.0, "best_cost": 1.0}]
        )
        architecture_trace = pd.DataFrame([{"iteration_id": 1}])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache.csv"
            cache.write_text("model_version\n", encoding="utf-8")
            calibration = root / "calibration.json"
            calibration.write_text("{}", encoding="utf-8")

            with (
                patch(
                    "script.run_optimizer_experiments.build_atlas_objective",
                    return_value="calibrated-objective",
                ),
                patch(
                    "script.run_optimizer_experiments.sim_annealing",
                    return_value=(
                        1.0,
                        architecture,
                        trace,
                        architecture_trace,
                    ),
                ) as anneal,
            ):
                run_search(
                    label="run_01",
                    output_dir=root / "runs",
                    workload=workload,
                    workload_id=9,
                    initial_architecture=architecture,
                    search_seed=13000,
                    search_space="cfg/parameters/input.json",
                    calibration_path=calibration,
                    base_cache=cache,
                    intermediate_policy="direct_forward",
                    atlas_graph=graph,
                )

        arguments = anneal.call_args.kwargs
        self.assertIs(arguments["atlas_graph"], graph)
        self.assertIsNone(arguments["workload_sequence"])
        self.assertEqual(arguments["atlas_objective"], "calibrated-objective")
        self.assertEqual(
            arguments["atlas_profile"].movement_policy,
            "direct_forward_v1",
        )
        self.assertEqual(arguments["atlas_search_space"]["max_sa_chiplets"], 6)


if __name__ == "__main__":
    unittest.main()
