import fcntl
import json
import tempfile
import unittest
from pathlib import Path

from script.run_baseline_campaign import (
    DEFAULT_WORKLOADS,
    run_dir,
    run_lock_available,
    result_is_valid,
    SCHEDULE,
    planned_move_count,
    task_list,
)


class BaselineCampaignTests(unittest.TestCase):
    def test_requested_schedule_has_expected_move_budget(self):
        self.assertEqual(planned_move_count(SCHEDULE), 75650)

    def test_tasks_are_round_robin_by_run(self):
        self.assertEqual(
            task_list(DEFAULT_WORKLOADS, 2),
            [(7, 1), (9, 1), (10, 1), (7, 2), (9, 2), (10, 2)],
        )

    def test_active_run_lock_is_not_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            lock_path = root / "run.lock"
            with lock_path.open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.assertFalse(run_lock_available(root))

    def test_metadata_validation_rejects_wrong_workload_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = run_dir(output, 7, 1)
            root.mkdir(parents=True)
            for name in (
                "search_trace.csv",
                "architecture_trace.csv",
                "best_architecture.json",
                "initial_architecture.json",
            ):
                (root / name).touch()
            result = {
                "workload_id": 7,
                "run": 1,
                "initial_seed": 12000,
                "search_seed": 13000,
                "schedule": "requested_4000_75650",
                "best_cost": -1.0,
                "verified_best_cost": -1.0,
                "best_fingerprint": "best",
                "canonical_best_fingerprint": "canonical",
                "trace_fingerprint": "trace",
                "attempted_moves": 75650,
                "search_space_sha256": "space",
                "calibration_sha256": "calibration",
                "workload_config_sha256": "wrong",
            }
            (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
            manifest = {
                "schedule_name": "requested_4000_75650",
                "planned_moves": 75650,
                "schedule": {"max_move_per_temp_step": 50},
                "seed_panel": {"initial_seed_base": 12000, "search_seed_base": 13000},
                "search_space_sha256": "space",
                "calibrations": {"7": {"sha256": "calibration"}},
                "workload_config_sha256": {"7": "expected"},
            }
            self.assertFalse(result_is_valid(root / "result.json", manifest, deep=False))


if __name__ == "__main__":
    unittest.main()
