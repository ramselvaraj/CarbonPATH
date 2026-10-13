import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import main
from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.AtlasObjective import build_atlas_objective
from system.utils.DesignPointFile import parse_design_point
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.GEMMWorkload import GEMMWorkload
from system.utils.SimulationCache import SimulationCache


ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
GRAPH = Path("cfg/examples/atlas/dense_relu_funnel.graph_dump.json")
PROFILE_DIR = Path("cfg/profiles/placeholders")
REFERENCE_PROFILE = Path("cfg/profiles/atlas_modular_v1.json")
MULTI_SA_PROFILE = Path("cfg/profiles/atlas_multi_sa_fpga_v1.json")
FULL_SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space.json")

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


def _sa_count(architecture):
    return sum(
        1
        for key, chiplet in architecture.items()
        if key.startswith("Chiplet_")
        and str(chiplet.get("chiplet_type", "systolic_array")).lower() != "fpga"
    )


def _sa_only_search_space():
    with FULL_SEARCH_SPACE.open(encoding="utf-8") as file:
        search_space = json.load(file)
    search_space["move_weights"] = {
        name: (1.0 if name == "sa_count" else 0.0)
        for name in search_space["move_weights"]
    }
    return search_space


def _fake_prepare(cache, arch_dict, shape):
    return main.build_scheduler_system(GEMMWorkload(*shape), arch_dict)


def _restricted_search_space(architecture):
    """Single-value architecture knobs so only the workload mapping can move."""
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

    def _run(self, seed, **overrides):
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
                    initial_architecture=overrides.pop(
                        "initial_architecture", self.architecture
                    ),
                    atlas_graph=self.graph,
                    atlas_profile=self.reference,
                    atlas_search_space=_restricted_search_space(self.architecture),
                    atlas_objective=self.objective,
                    **overrides,
                )

    def test_annealing_returns_one_design_point_that_can_start_a_run(self):
        _, best_architecture, trace, _ = self._run(seed=11)
        document = trace.attrs["best_design_point"]
        loaded = parse_design_point(document)

        self.assertEqual(loaded.architecture.keys(), best_architecture.keys())
        self.assertEqual(
            loaded.mapping.fingerprint(),
            trace.attrs["best_workload_mapping_fingerprint"],
        )
        self.assertEqual(document["workload"]["fingerprint"], self.graph.fingerprint())
        self.assertEqual(document["evaluated_with"]["profile"], "atlas_modular_v1")

        # A run that starts from the file does not build a first mapping.
        with patch(
            "main.initial_workload_mapping",
            side_effect=AssertionError("the file gives the first mapping"),
        ):
            _, _, restarted, _ = self._run(
                seed=11,
                initial_architecture=loaded.architecture,
                initial_mapping=loaded.mapping,
            )
        self.assertIn("best_design_point", restarted.attrs)

    def test_annealing_keeps_the_evaluation_profile_constant(self):
        reference_cost = self._reference_cost()
        best_cost, _, trace, _ = self._run(seed=2026)
        self.assertEqual(trace.attrs["atlas"], True)
        self.assertEqual(trace.attrs["best_profile"]["profile"], "atlas_modular_v1")
        self.assertEqual(set(trace["profile_name"].dropna()), {"atlas_modular_v1"})
        self.assertEqual(
            trace["profile_fingerprint"].dropna().nunique(), 1
        )
        self.assertAlmostEqual(best_cost, reference_cost, places=6)

    def test_annealing_is_seeded_deterministic(self):
        first = self._run(seed=5)
        second = self._run(seed=5)
        self.assertEqual(first[0], second[0])
        self.assertEqual(
            first[2]["new_cost"].fillna(-1).tolist(),
            second[2]["new_cost"].fillna(-1).tolist(),
        )

    def test_trace_records_the_workload_mapping(self):
        _, _, trace, _ = self._run(seed=11)
        self.assertIn("profile_name", trace.columns)
        self.assertIn("profile_fingerprint", trace.columns)
        self.assertIn("mapping_fingerprint", trace.columns)
        self.assertGreater(trace["mapping_fingerprint"].dropna().nunique(), 1)
        best_mapping = trace.attrs["best_workload_mapping"]
        self.assertEqual(
            [operation["operation"] for operation in best_mapping["operations"]],
            [operation.operation_id for operation in self.graph.operations],
        )
        self.assertEqual(
            len(best_mapping["handoffs"]), len(self.graph.operations) - 1
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

    def test_multi_sa_candidates_evaluate_and_are_traced(self):
        multi_profile = load_evaluation_profile(MULTI_SA_PROFILE)
        search_space = _sa_only_search_space()
        objective = build_atlas_objective()
        evaluated = []
        real_evaluate = main.evaluate_atlas_design_point

        def spy(
            cache, architecture, graph, profile, objective, registry=None, mapping=None
        ):
            result = real_evaluate(
                cache,
                architecture,
                graph,
                profile,
                objective,
                registry=registry,
                mapping=mapping,
            )
            evaluated.append(architecture)
            return result

        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_file, index=False)
            with (
                patch("main.evaluate_atlas_design_point", side_effect=spy),
                patch("main.prepare_single_gemm", side_effect=_fake_prepare),
                patch("main.simulate_single_gemm", return_value=dict(SIMULATED_GEMM)),
            ):
                _, _, trace, _ = main.sim_annealing(
                    wl_idx=None,
                    workload_sequence=None,
                    cache_file=str(cache_file),
                    run_name=str(Path(directory) / "sim"),
                    cost_profile="t1",
                    initial_temp=4.0,
                    freezing_temp=0.5,
                    max_move_per_temp_step=6,
                    cooling_rate=0.5,
                    calibration_iterations=1,
                    random_seed=7,
                    initial_architecture=self.architecture,
                    atlas_graph=self.graph,
                    atlas_profile=multi_profile,
                    atlas_search_space=search_space,
                    atlas_objective=objective,
                )

        counts = {_sa_count(architecture) for architecture in evaluated}
        self.assertIn(1, counts)
        self.assertTrue(any(count > 1 for count in counts), counts)
        self.assertTrue(bool((trace["proposal_valid"]).any()))


if __name__ == "__main__":
    unittest.main()
