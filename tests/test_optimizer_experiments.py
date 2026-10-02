import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from script.run_optimizer_experiments import (
    REDUCED_SEARCH_SPACE,
    build_reduced_candidates,
    persist_best_profile,
    wilson_interval,
)
from script.run_full_space_convergence import SCHEDULES, planned_move_count
from system.utils.ArchitectureIdentity import architecture_fingerprint


class OptimizerExperimentTests(unittest.TestCase):
    def test_modular_search_persists_its_best_evaluation_profile(self):
        trace = pd.DataFrame({"best_cost": [1.0]})
        trace.attrs["best_profile"] = {
            "profile": "atlas_modular_v1",
            "version": 1,
            "evaluators": {"gemm": "legacy_scale_sim_gemm_v1"},
            "placement_policy": "all_sas_single_fpga_v1",
            "movement_policy": "direct_forward_v1",
            "transfer_model": "route_transfer_v1",
        }
        trace.attrs["best_profile_fingerprint"] = "abc123"

        with tempfile.TemporaryDirectory() as directory:
            result = persist_best_profile(trace, Path(directory))
            stored = json.loads(
                (Path(directory) / "best_profile.json").read_text(encoding="utf-8")
            )

        self.assertEqual(stored, trace.attrs["best_profile"])
        self.assertEqual(result["best_profile_fingerprint"], "abc123")

    def test_reduced_space_contains_96_unique_candidates(self):
        candidates = build_reduced_candidates(REDUCED_SEARCH_SPACE)

        self.assertEqual(len(candidates), 96)
        self.assertEqual(
            len({architecture_fingerprint(candidate) for candidate in candidates}),
            96,
        )

    def test_wilson_interval_contains_observed_proportion(self):
        lower, upper = wilson_interval(15, 20)

        self.assertLess(lower, 0.75)
        self.assertGreater(upper, 0.75)

    def test_convergence_schedules_have_expected_move_budgets(self):
        self.assertEqual(planned_move_count(SCHEDULES["current_45"]), 45)
        self.assertEqual(planned_move_count(SCHEDULES["moderate_600"]), 600)
        self.assertEqual(planned_move_count(SCHEDULES["thorough_600"]), 600)
        self.assertEqual(planned_move_count(SCHEDULES["thorough_1200"]), 1200)
        self.assertEqual(planned_move_count(SCHEDULES["thorough_2400"]), 2400)
        self.assertEqual(planned_move_count(SCHEDULES["thorough_4800"]), 4800)
        self.assertEqual(planned_move_count(SCHEDULES["slow_320_5632"]), 5632)
        self.assertEqual(planned_move_count(SCHEDULES["slow_640_5824"]), 5824)
        self.assertEqual(
            planned_move_count(SCHEDULES["intensive_320_11264"]), 11264
        )
        self.assertEqual(
            planned_move_count(SCHEDULES["intensive_640_11648"]), 11648
        )
        self.assertEqual(
            planned_move_count(SCHEDULES["requested_4000_75650"]), 75650
        )


if __name__ == "__main__":
    unittest.main()
