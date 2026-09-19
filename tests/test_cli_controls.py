import math
import unittest

from main import (
    CLI_COOLING_RATE,
    CLI_FREEZING_TEMP,
    CLI_INITIAL_TEMP,
    CLI_MAX_MOVE_PER_TEMP_STEP,
    calibration_is_current,
)


def planned_moves(initial_temp, freezing_temp, cooling_rate, moves_per_step):
    levels = math.ceil(
        math.log(freezing_temp / initial_temp) / math.log(cooling_rate)
    )
    return levels * moves_per_step


class CliControlTests(unittest.TestCase):
    def test_cli_schedule_has_fifty_move_attempts(self):
        self.assertEqual(
            planned_moves(
                CLI_INITIAL_TEMP,
                CLI_FREEZING_TEMP,
                CLI_COOLING_RATE,
                CLI_MAX_MOVE_PER_TEMP_STEP,
            ),
            50,
        )

    def test_baseline_schedule_has_expected_move_budget(self):
        self.assertEqual(planned_moves(4000, 1e-3, 0.99, 50), 75650)

    def test_calibration_sample_count_is_part_of_reuse_check(self):
        calibration = {"_calibration_identity": "identity", "_calibration_samples": 10}

        self.assertTrue(calibration_is_current(calibration, "identity", 10))
        self.assertFalse(calibration_is_current(calibration, "identity", 10000))
        self.assertFalse(calibration_is_current(calibration, "other", 10))


if __name__ == "__main__":
    unittest.main()
