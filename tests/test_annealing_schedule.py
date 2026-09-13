import math
import unittest

from system.utils.AnnealingSchedule import (
    AdaptiveScheduleConfig,
    AdaptiveTemperatureController,
)


def row(diff, accepted, valid=True, changed=True):
    return {
        "cost_diff": diff,
        "move_accepted": accepted,
        "proposal_valid": valid,
        "proposal_changed": changed,
        "new_cost": 1 if valid else None,
    }


class AdaptiveScheduleTests(unittest.TestCase):
    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            AdaptiveScheduleConfig(max_total_moves=0).validate()
        with self.assertRaises(ValueError):
            AdaptiveScheduleConfig(target_acceptance_start=0.01).validate()
        with self.assertRaises(ValueError):
            AdaptiveScheduleConfig(fallback_cooling_rate=1).validate()

    def test_target_acceptance_curve(self):
        controller = AdaptiveTemperatureController(12, AdaptiveScheduleConfig())
        self.assertAlmostEqual(controller.target_acceptance(0), 0.8)
        self.assertAlmostEqual(controller.target_acceptance(37825), 0.2)
        self.assertAlmostEqual(controller.target_acceptance(75650), 0.05)

    def test_counts_only_changed_uphill_proposals(self):
        controller = AdaptiveTemperatureController(12, AdaptiveScheduleConfig())
        rows = (
            row(2, True),
            row(2, False, changed=False),
            row(0, True),
            row(-2, True),
            row(2, True, valid=False),
        )
        controller.next_temperature(
            level_index=0,
            temperature=12,
            rows=rows,
            attempted_moves=5,
            current_cost=0,
            best_cost_before=0,
            best_cost_after=-2,
        )
        self.assertEqual(controller.history[-1].evidence_count, 1)
        self.assertIsNone(controller.history[-1].observed_acceptance)

    def test_high_acceptance_cools_and_low_acceptance_heats(self):
        config = AdaptiveScheduleConfig(minimum_uphill_proposals=5)
        controller = AdaptiveTemperatureController(12, config)
        high = tuple(row(2, True) for _ in range(10))
        low = tuple(row(2, False) for _ in range(10))
        cooled = controller.next_temperature(
            level_index=0, temperature=12, rows=high, attempted_moves=100,
            current_cost=0, best_cost_before=0, best_cost_after=0,
        )
        heated = controller.next_temperature(
            level_index=1, temperature=cooled, rows=low, attempted_moves=200,
            current_cost=0, best_cost_before=0, best_cost_after=0,
        )
        self.assertLess(cooled, 12)
        self.assertGreater(heated, cooled)

    def test_reheat_is_bounded_and_disabled_late(self):
        config = AdaptiveScheduleConfig(
            stagnation_moves=10,
            maximum_reheats=1,
            minimum_uphill_proposals=1,
        )
        controller = AdaptiveTemperatureController(2, config)
        rows = tuple(row(2, False) for _ in range(10))
        temperature = controller.next_temperature(
            level_index=0, temperature=2, rows=rows, attempted_moves=10,
            current_cost=0, best_cost_before=0, best_cost_after=0,
        )
        self.assertEqual(controller.history[-1].action, "reheat")
        controller.next_temperature(
            level_index=1, temperature=temperature, rows=rows, attempted_moves=20,
            current_cost=0, best_cost_before=0, best_cost_after=0,
        )
        self.assertEqual(controller.reheat_count, 1)
        controller.next_temperature(
            level_index=2, temperature=temperature, rows=rows, attempted_moves=75650,
            current_cost=0, best_cost_before=0, best_cost_after=0,
        )
        self.assertEqual(controller.reheat_count, 1)

    def test_temperature_is_finite_and_bounded(self):
        controller = AdaptiveTemperatureController(12, AdaptiveScheduleConfig())
        with self.assertRaises(ValueError):
            controller.next_temperature(
                level_index=0, temperature=math.nan, rows=(), attempted_moves=0,
                current_cost=0, best_cost_before=0, best_cost_after=0,
            )
        result = controller.next_temperature(
            level_index=0, temperature=12, rows=(), attempted_moves=0,
            current_cost=0, best_cost_before=0, best_cost_after=0,
        )
        self.assertGreaterEqual(result, 1e-3)
        self.assertLessEqual(result, 48)


if __name__ == "__main__":
    unittest.main()
