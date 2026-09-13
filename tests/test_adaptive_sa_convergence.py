import json
import tempfile
import unittest
from pathlib import Path

from script.run_adaptive_sa_convergence import (
    gate,
    initial_temperature,
    relative_gap,
)


def result(fingerprint, cost, run=1):
    return {
        "run": run,
        "canonical_best_fingerprint": fingerprint,
        "verified_best_cost": cost,
    }


class AdaptiveCampaignTests(unittest.TestCase):
    def test_even_reference_delta_count_uses_statistical_median(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory)
            run = reference / "runs" / "run_01"
            run.mkdir(parents=True)
            (run / "search_trace.csv").write_text(
                "cost_diff\n1\n2\n3\n4\n", encoding="utf-8"
            )
            self.assertAlmostEqual(initial_temperature(reference), 2.5 / 0.2231435513142097)

    def test_negative_objective_gap_is_signed_relative_to_best(self):
        self.assertAlmostEqual(relative_gap(-5.1, -5.2), 0.019230769230769232)

    def test_gate_requires_quality_and_modal_agreement(self):
        pilot = [result("a", -5.0, run) for run in range(1, 4)]
        passed, _ = gate(pilot, 2, all_results=pilot)
        self.assertTrue(passed)

        formal = [result("a", -5.0, run) for run in range(1, 9)]
        formal.extend(result("b", -5.2, run) for run in range(9, 11))
        passed, detail = gate(formal, 8, pilot_fingerprint="a", all_results=pilot + formal)
        self.assertFalse(passed)
        self.assertIn("modal_quality_gap", detail)

    def test_gate_rejects_missing_verified_results(self):
        passed, detail = gate([{"canonical_best_fingerprint": "a"}], 1)
        self.assertFalse(passed)
        self.assertEqual(detail, "missing verified results")


if __name__ == "__main__":
    unittest.main()
