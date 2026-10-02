import random
import unittest

from system.utils.AnnealingSchedule import (
    AdaptiveScheduleConfig,
    AdaptiveTemperatureController,
)
from system.utils.GenericAnnealer import anneal
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


class GenericAnnealerTests(unittest.TestCase):
    def test_tracks_current_separately_from_best(self):
        proposals = iter([1, 3, 2])

        result = anneal(
            initial_design=0,
            propose=lambda _current, _rng: (next(proposals), "step"),
            evaluate=lambda design: ({"value": design}, float(design)),
            identity=str,
            rng=random.Random(4),
            initial_temperature=100.0,
            freezing_temperature=1.0,
            moves_per_temperature=3,
            cooling_rate=0.01,
        )

        self.assertEqual(result.best_design, 0)
        self.assertEqual(result.best_cost, 0.0)
        self.assertNotEqual(result.current_design, result.best_design)
        self.assertEqual(len(result.trace), 3)

    def test_invalid_and_unchanged_proposals_are_recorded_not_evaluated(self):
        calls = []
        proposals = iter(("invalid", 0, -1))

        def propose(_current, _rng):
            candidate = next(proposals)
            if candidate == "invalid":
                raise UnsupportedEvaluation("not feasible")
            return candidate, "step"

        def evaluate(design):
            calls.append(design)
            return {"value": design}, float(design)

        result = anneal(
            initial_design=0,
            propose=propose,
            evaluate=evaluate,
            identity=str,
            rng=random.Random(1),
            initial_temperature=2.0,
            freezing_temperature=1.0,
            moves_per_temperature=3,
            cooling_rate=0.25,
        )

        self.assertEqual(calls, [0, -1])
        self.assertEqual(result.best_design, -1)
        self.assertEqual(
            [(row["proposal_valid"], row["proposal_changed"]) for row in result.trace],
            [(False, False), (True, False), (True, True)],
        )

    def test_fixed_seed_replays_the_same_trace(self):
        def run(seed):
            def propose(current, rng):
                return current + rng.choice((-2, -1, 1, 2)), "step"

            return anneal(
                initial_design=5,
                propose=propose,
                evaluate=lambda design: (design, abs(float(design))),
                identity=str,
                rng=random.Random(seed),
                initial_temperature=4.0,
                freezing_temperature=0.5,
                moves_per_temperature=2,
                cooling_rate=0.5,
            )

        first = run(77)
        second = run(77)
        self.assertEqual(first.best_design, second.best_design)
        self.assertEqual(first.trace, second.trace)

    def test_equal_best_cost_uses_canonical_identity_as_tie_breaker(self):
        result = anneal(
            initial_design="z",
            propose=lambda _current, _rng: ("a", "rename"),
            evaluate=lambda design: (design, 1.0),
            identity=str,
            rng=random.Random(1),
            initial_temperature=2.0,
            freezing_temperature=1.0,
            moves_per_temperature=1,
            cooling_rate=0.25,
        )

        self.assertEqual(result.best_design, "a")

    def test_adaptive_schedule_stops_at_the_exact_move_budget(self):
        controller = AdaptiveTemperatureController(
            2.0,
            AdaptiveScheduleConfig(max_total_moves=5),
        )
        levels = []
        result = anneal(
            initial_design=0,
            propose=lambda current, _rng: (current - 1, "step"),
            evaluate=lambda design: (design, float(design)),
            identity=str,
            rng=random.Random(1),
            initial_temperature=2.0,
            freezing_temperature=1.0,
            moves_per_temperature=3,
            cooling_rate=0.5,
            temperature_controller=controller,
            max_total_moves=5,
            level_callback=lambda decision, rows: levels.append((decision, rows)),
        )

        self.assertEqual(len(result.trace), 5)
        self.assertEqual([len(rows) for _, rows in levels], [3, 2])


if __name__ == "__main__":
    unittest.main()
