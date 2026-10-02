import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from system.utils.Calibration import summarize_design_points


def _point(energy, area, cost, latency, embodied, operational):
    return SimpleNamespace(
        total_energy_pj=energy,
        area_mm2=area,
        cost_usd=cost,
        latency_ns=latency,
        embodied_carbon_kg=embodied,
        operational_carbon_kg=operational,
    )


class ModularCalibrationTests(unittest.TestCase):
    def test_summarizes_modular_design_points_for_existing_objectives(self):
        calibration = summarize_design_points(
            [
                _point(10, 2, 3, 4, 5, 6),
                _point(30, 4, 7, 8, 9, 10),
            ],
            calibration_identity="fixed-search-and-workload",
            model_version=3,
        )

        self.assertEqual(calibration["_calibration_samples"], 2)
        self.assertEqual(
            calibration["_calibration_identity"], "fixed-search-and-workload"
        )
        self.assertEqual(calibration["avg_energy"], 20.0)
        self.assertEqual(calibration["avg_area"], 3.0)
        self.assertEqual(calibration["avg_dollar_cost"], 5.0)
        self.assertEqual(calibration["avg_latency"], 6.0)
        self.assertEqual(calibration["energy_min"], 10)
        self.assertEqual(calibration["energy_max"], 30)
        self.assertEqual(calibration["energy_median"], 20.0)
        self.assertAlmostEqual(calibration["energy_stddev"], math.sqrt(200))

    def test_requires_at_least_one_design_point(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_design_points([], "identity", 3)

    def test_calibration_samples_are_evaluated_through_modular_flow(self):
        points = [
            _point(10, 2, 3, 4, 5, 6),
            _point(30, 4, 7, 8, 9, 10),
        ]
        workload = {
            "name": "tiny",
            "gemms": [{"name": "only", "shape": (1, 2, 3)}],
        }
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"
            with (
                patch("main.gen_initial_arch", side_effect=[{"sample": 1}, {"sample": 2}]),
                patch(
                    "main.evaluate_atlas_design_point",
                    side_effect=[(points[0], 0.0), (points[1], 0.0)],
                ) as evaluate,
                patch(
                    "main.simulate_latency_energy",
                    side_effect=AssertionError("old evaluator must not be used"),
                ),
            ):
                calibration = main.get_modular_calib_cost_avg(
                    calibration_iterations=2,
                    config_path={"max_chiplet": 1},
                    cache=object(),
                    calibration_file_path=calibration_path,
                    workload_sequence=workload,
                    intermediate_policy="direct_forward",
                    graph=object(),
                    profile=object(),
                )

        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(calibration["avg_energy"], 20.0)
        self.assertEqual(calibration["_calibration_samples"], 2)

    def test_calibration_command_uses_modular_calibration(self):
        workload = {
            "name": "tiny",
            "gemms": [{"name": "only", "shape": (1, 2, 3)}],
        }
        expected = {"avg_energy": 12.0}
        with (
            patch("main.SimulationCache"),
            patch("main.get_modular_calib_cost_avg", return_value=expected) as modular,
            patch(
                "main.get_calib_cost_avg",
                side_effect=AssertionError("old calibration must not be used"),
            ),
        ):
            result = main.run_calibration(
                wl_idx=1,
                workload_sequence=workload,
                cache_file="unused.csv",
                run_name="unused",
                cost_profile="t1",
                calibration_iterations=2,
            )

        self.assertIs(result, expected)
        call = modular.call_args.kwargs
        self.assertEqual(call["graph"].operations[0].gemm_shape, (1, 2, 3))
        self.assertEqual(call["profile"].movement_policy, "direct_forward_v1")


if __name__ == "__main__":
    unittest.main()
