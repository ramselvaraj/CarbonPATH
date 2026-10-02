import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import main
from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.AtlasObjective import build_atlas_objective
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.SimulationCache import SimulationCache


ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
GRAPH = Path("cfg/examples/atlas/dense_relu_funnel.graph_dump.json")
PROFILE_DIR = Path("cfg/profiles/placeholders")
REFERENCE_PROFILE = Path("cfg/profiles/atlas_modular_v1.json")
BAD_PROFILE_PATHS = (
    PROFILE_DIR / "ph_slow_gemm_v1.json",
    PROFILE_DIR / "ph_dram_always_v1.json",
    PROFILE_DIR / "ph_pessimistic_transfer_v1.json",
)

CACHE_COLUMNS = [
    "core_size",
    "data_flow",
    "bandwidth",
    "buffer_size",
    "M",
    "K",
    "N",
    "latency",
]

SIMULATED_GEMM = {
    "latency_ns": 1000.0,
    "dram_interconnect_energy_pj": 200.0,
    "sram_energy_pj": 100.0,
}


def _load(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _restricted_search_space(architecture):
    """Single-value architecture knobs so only the profile can change cost."""
    fpga = architecture["Chiplet_2"]
    sa = architecture["Chiplet_1"]
    return {
        "sys_array": [sa["sys_array_size"]],
        "tech_nodes": [str(sa["tech_node"])],
        "sram_buf_sizes": {sa["sys_array_size"]: [sa["sram_buf"]]},
        "fpga": {
            "clbs": [fpga["clbs"]],
            "brams": [fpga["brams"]],
            "dsps": [fpga["dsps"]],
            "frequency_hz": [fpga["frequency_hz"]],
            "relu_implementation": [dict(fpga["relu_implementation"])],
        },
        "pkg": {
            "protocol": [architecture["pkg"]["protocol_2.5d"]],
            "mem_pkg_architecture": [
                architecture["pkg"]["mem_pkg_conn"]["mem_type"]
            ],
        },
        "transfer_model": [dict(architecture["transfer_model"])],
    }


class AtlasAnnealingTests(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        self.graph = load_atlas_graph(GRAPH)
        self.reference = load_evaluation_profile(REFERENCE_PROFILE)
        self.bad_profiles = [
            load_evaluation_profile(path) for path in BAD_PROFILE_PATHS
        ]
        self.candidate_profiles = [self.reference, *self.bad_profiles]
        self.objective = build_atlas_objective()

    def _cache(self, directory):
        cache_path = Path(directory) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        return SimulationCache(cache_path, simulator_dir=str(Path(directory) / "sim"))

    def _reference_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("main.simulate_single_gemm") as simulate:
                simulate.return_value = dict(SIMULATED_GEMM)
                _, cost = main.evaluate_atlas_design_point(
                    self._cache(directory),
                    self.architecture,
                    self.graph,
                    self.reference,
                    self.objective,
                )
        return cost

    def _run(self, seed):
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_file, index=False)
            with patch("main.simulate_single_gemm") as simulate:
                simulate.return_value = dict(SIMULATED_GEMM)
                return main.sim_annealing(
                    wl_idx=None,
                    workload_sequence=None,
                    cache_file=str(cache_file),
                    run_name=str(Path(directory) / "sim"),
                    cost_profile="t1",
                    initial_temp=2.0,
                    freezing_temp=1e-3,
                    max_move_per_temp_step=3,
                    cooling_rate=0.5,
                    calibration_iterations=1,
                    random_seed=seed,
                    initial_architecture=self.architecture,
                    atlas_graph=self.graph,
                    atlas_profile=self.reference,
                    atlas_search_space=_restricted_search_space(self.architecture),
                    atlas_objective=self.objective,
                    candidate_profiles=self.candidate_profiles,
                )

    def test_annealing_selects_the_good_profile(self):
        reference_cost = self._reference_cost()
        best_cost, _, trace, _ = self._run(seed=2026)
        self.assertEqual(trace.attrs["atlas"], True)
        self.assertEqual(trace.attrs["best_profile"]["profile"], "atlas_modular_v1")
        self.assertAlmostEqual(best_cost, reference_cost, places=6)

    def test_every_bad_profile_candidate_scores_worse(self):
        reference_cost = self._reference_cost()
        _, _, trace, _ = self._run(seed=99)
        bad_names = {profile.name for profile in self.bad_profiles}
        bad_rows = trace[trace["profile_name"].isin(bad_names)]
        bad_rows = bad_rows[bad_rows["new_cost"].notna()]
        self.assertGreater(len(bad_rows), 0, "expected at least one bad proposal")
        self.assertTrue((bad_rows["new_cost"] > reference_cost).all())

    def test_annealing_is_seeded_deterministic(self):
        first = self._run(seed=5)
        second = self._run(seed=5)
        self.assertEqual(first[0], second[0])
        self.assertEqual(
            first[2]["new_cost"].fillna(-1).tolist(),
            second[2]["new_cost"].fillna(-1).tolist(),
        )

    def test_trace_records_profile_identity(self):
        _, _, trace, _ = self._run(seed=11)
        self.assertIn("profile_name", trace.columns)
        self.assertIn("profile_fingerprint", trace.columns)
        self.assertEqual(
            set(trace.attrs["candidate_profiles"]),
            {"atlas_modular_v1", "ph_slow_gemm_v1", "ph_dram_always_v1",
             "ph_pessimistic_transfer_v1"},
        )

    def test_sequential_gemm_entry_uses_the_modular_annealer(self):
        workload = {
            "name": "two_gemm",
            "gemms": [
                {"name": "first", "shape": (8, 16, 32)},
                {"name": "second", "shape": (8, 32, 4)},
            ],
        }
        expected = (1.0, self.architecture, pd.DataFrame(), pd.DataFrame())
        calibration = {
            "avg_energy": 1.0,
            "avg_area": 1.0,
            "avg_dollar_cost": 1.0,
            "avg_latency": 1.0,
            "avg_embCarbon": 1.0,
            "avg_opeCarbon": 1.0,
        }
        for metric in ("energy", "area", "cost", "latency", "embCarbon", "opeCarbon"):
            calibration.update(
                {
                    f"{metric}_min": 1.0,
                    f"{metric}_max": 2.0,
                    f"{metric}_mean": 1.5,
                    f"{metric}_stddev": 0.5,
                    f"{metric}_median": 1.5,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_file, index=False)
            with (
                patch("main.get_modular_calib_cost_avg", return_value=calibration) as calibrate,
                patch("main.get_calib_cost_avg", side_effect=AssertionError("old calibration")),
                patch("main._run_modular_annealing", return_value=expected) as modular,
            ):
                result = main.sim_annealing(
                    wl_idx=7,
                    workload_sequence=workload,
                    cache_file=str(cache_file),
                    run_name=str(Path(directory) / "sim"),
                    cost_profile="t1",
                    initial_temp=2.0,
                    freezing_temp=1e-3,
                    max_move_per_temp_step=1,
                    cooling_rate=0.5,
                    calibration_iterations=2,
                    initial_architecture=self.architecture,
                )

        self.assertIs(result, expected)
        calibrate.assert_called_once()
        call = modular.call_args.kwargs
        self.assertEqual(
            [operation.gemm_shape for operation in call["graph"].operations],
            [(8, 16, 32), (8, 32, 4)],
        )
        self.assertEqual(call["profile"].movement_policy, "direct_forward_v1")
        self.assertEqual(call["search_space"]["max_sa_chiplets"], 6)

    def test_explicit_calibration_is_loaded_without_regeneration(self):
        workload = {
            "name": "two_gemm",
            "gemms": [
                {"name": "first", "shape": (8, 16, 32)},
                {"name": "second", "shape": (8, 32, 4)},
            ],
        }
        calibration = {
            "_calibration_identity": main.calibration_identity(
                "cfg/parameters/input.json", workload, "direct_forward"
            ),
            "_calibration_samples": 10,
            "avg_energy": 1.0,
            "avg_area": 1.0,
            "avg_dollar_cost": 1.0,
            "avg_latency": 1.0,
            "avg_embCarbon": 1.0,
            "avg_opeCarbon": 1.0,
        }
        for metric in ("energy", "area", "cost", "latency", "embCarbon", "opeCarbon"):
            calibration.update(
                {
                    f"{metric}_min": 1.0,
                    f"{metric}_max": 2.0,
                    f"{metric}_mean": 1.5,
                    f"{metric}_stddev": 0.5,
                    f"{metric}_median": 1.5,
                }
            )
        expected = (1.0, self.architecture, pd.DataFrame(), pd.DataFrame())

        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "prepared-calibration.json"
            calibration_path.write_text(
                json.dumps(calibration, indent=2), encoding="utf-8"
            )
            original_bytes = calibration_path.read_bytes()
            cache_file = Path(directory) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_file, index=False)
            with (
                patch(
                    "main.get_modular_calib_cost_avg",
                    side_effect=AssertionError("prepared calibration must be immutable"),
                ),
                patch("main._run_modular_annealing", return_value=expected) as modular,
            ):
                result = main.sim_annealing(
                    wl_idx=7,
                    workload_sequence=workload,
                    cache_file=str(cache_file),
                    run_name=str(Path(directory) / "sim"),
                    cost_profile="t1",
                    initial_temp=2.0,
                    freezing_temp=1e-3,
                    max_move_per_temp_step=1,
                    cooling_rate=0.5,
                    calibration_iterations=1,
                    calibration_file_path=str(calibration_path),
                    initial_architecture=self.architecture,
                )

            self.assertIs(result, expected)
            self.assertEqual(calibration_path.read_bytes(), original_bytes)
            self.assertEqual(modular.call_args.kwargs["objective"].objective_id, "t1")


if __name__ == "__main__":
    unittest.main()
