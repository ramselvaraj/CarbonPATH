import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from script.run_baseline_campaign import (
    DEFAULT_WORKLOADS,
    make_progress_callback,
    run_dir,
    run_lock_available,
    result_is_valid,
    require_clean_worktree,
    sha256,
    SCHEDULE,
    planned_move_count,
    parse_atlas_graph_specs,
    task_list,
)


class BaselineCampaignTests(unittest.TestCase):
    def test_atlas_graph_specs_require_every_selected_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / "graph.json"
            graph.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing 10"):
                parse_atlas_graph_specs([f"9={graph}"], (9, 10))

            self.assertEqual(
                parse_atlas_graph_specs([f"9={graph}"], (9,)),
                {9: graph.resolve()},
            )

    def test_progress_file_reports_attempted_and_remaining_moves(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            callback = make_progress_callback(
                path, workload_id=7, run=1, planned_moves=100
            )

            self.assertEqual(json.loads(path.read_text())["moves_remaining"], 100)
            callback(None, tuple({} for _ in range(50)))
            halfway = json.loads(path.read_text())
            self.assertEqual(halfway["attempted_moves"], 50)
            self.assertEqual(halfway["moves_remaining"], 50)
            callback(None, tuple({} for _ in range(50)))
            complete = json.loads(path.read_text())
            self.assertEqual(complete["attempted_moves"], 100)
            self.assertEqual(complete["moves_remaining"], 0)
            self.assertEqual(complete["status"], "search_complete")

    def test_campaign_preparation_rejects_a_dirty_worktree(self):
        completed = CompletedProcess(
            args=["git", "status", "--porcelain"],
            returncode=0,
            stdout=" M main.py\n",
            stderr="",
        )
        with patch(
            "script.run_baseline_campaign.subprocess.run", return_value=completed
        ):
            with self.assertRaisesRegex(RuntimeError, "clean git worktree"):
                require_clean_worktree()

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

    def test_modular_result_requires_a_valid_matching_best_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = run_dir(Path(directory), 7, 1)
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
                "best_cost": 1.0,
                "verified_best_cost": 1.0,
                "best_fingerprint": "best",
                "canonical_best_fingerprint": "canonical",
                "trace_fingerprint": "trace",
                "attempted_moves": 75650,
                "invalid_proposal_rate": 0.0,
                "late_improvement": False,
                "last_improvement_move": 0,
                "latency": 1.0,
                "energy": 1.0,
                "area": 1.0,
                "dollar": 1.0,
                "embCarbon": 1.0,
                "opeCarbon": 1.0,
                "accepted_moves": 1,
                "acceptance_rate": 0.1,
                "runtime_seconds": 1.0,
                "wall_seconds_with_evaluation": 1.0,
                "simulator_calls": 1,
                "git_commit": "commit",
                "calibration_identity": "identity",
                "search_trace_sha256": "unused",
                "architecture_trace_sha256": "unused",
                "best_architecture_sha256": "unused",
                "initial_architecture_sha256": "unused",
                "search_space_sha256": "space",
                "calibration_sha256": "calibration",
                "workload_config_sha256": "workload",
            }
            manifest = {
                "evaluation_flow": "modular",
                "schedule_name": "requested_4000_75650",
                "planned_moves": 75650,
                "schedule": {"max_move_per_temp_step": 50},
                "seed_panel": {"initial_seed_base": 12000, "search_seed_base": 13000},
                "search_space_sha256": "space",
                "calibrations": {
                    "7": {"sha256": "calibration", "identity": "identity"}
                },
                "workload_config_sha256": {"7": "workload"},
                "git_commit": "commit",
            }
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")

            self.assertFalse(
                result_is_valid(
                    result_path, manifest, deep=False, verify_artifacts=False
                )
            )

            profile = {
                "profile": "atlas_modular_v1",
                "version": 1,
                "evaluators": {"gemm": "legacy_scale_sim_gemm_v1"},
                "placement_policy": "all_sas_v1",
                "movement_policy": "direct_forward_v1",
                "transfer_model": "route_transfer_v1",
            }
            profile_path = root / "best_profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            from system.utils.EvaluationProfile import parse_evaluation_profile

            result["best_profile_fingerprint"] = parse_evaluation_profile(
                profile
            ).fingerprint()
            result["best_profile_sha256"] = sha256(profile_path)
            result_path.write_text(json.dumps(result), encoding="utf-8")

            self.assertTrue(
                result_is_valid(
                    result_path, manifest, deep=False, verify_artifacts=False
                )
            )


if __name__ == "__main__":
    unittest.main()
