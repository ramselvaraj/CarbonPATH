import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from script.analyze_architecture_similarity import (
    architecture_features,
    decorate_distances,
    feature_distance,
    field_agreement,
    load_run_set,
    profile_agreement,
    render_report,
    within_tolerance_count,
)


def make_arch(n_chiplets=2, sram=256, mem=(5, 5), ascending=0):
    architecture = {}
    for index in range(1, n_chiplets + 1):
        architecture[f"Chiplet_{index}"] = {
            "tech_node": "7",
            "sys_array_size": "64x64",
            "sram_buf": sram if index == 1 else 256,
        }
    connections = [
        {
            "from": f"Chiplet_{index}",
            "to": f"Chiplet_{index + 1}",
            "connection_type": "3d_hyb_bond",
            "loc": f"stack0_{index}",
        }
        for index in range(1, n_chiplets)
    ]
    connections.append(
        {
            "from": f"Chiplet_{n_chiplets}",
            "to": "na",
            "connection_type": "3d_hyb_bond",
            "loc": "stack0_top",
        }
    )
    architecture["pkg"] = {
        "HI_pkg_type": "3d",
        "inter_pkg_conn": connections,
        "protocol_3d": "ucie_3d",
        "protocol_2.5d": "na",
        "mem_pkg_conn": {
            "mem_type": "hbm3",
            **{f"Chiplet_{index}": mem[index - 1] for index in range(1, n_chiplets + 1)},
        },
    }
    architecture["WL_mapping"] = {
        "mapping": {
            "dataflow": ["os"],
            "assign_workload_in_ascending_order": ascending,
            "static_tiling": 0,
            "merge_tiles": 0,
            "if_splitting_k": 0,
            "chiplet_data_sharing_enabled": 0,
        }
    }
    return architecture


def write_run(directory, objective, architecture, moves=100, profile=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "best_arch_run.json").write_text(
        json.dumps(architecture), encoding="utf-8"
    )
    pd.DataFrame(
        {"SA_run_loop": list(range(1, moves + 1)), "best_cost_after": [objective] * moves}
    ).to_csv(directory / "sa_metrics_run.csv", index=False)
    if profile is not None:
        (directory / "best_profile.json").write_text(
            json.dumps(profile), encoding="utf-8"
        )


def make_atlas_arch(clbs=10000, setup=5.0, freq=300000000):
    return {
        "Chiplet_1": {
            "tech_node": "7",
            "sys_array_size": "64x64",
            "sram_buf": 256,
        },
        "Chiplet_2": {
            "chiplet_type": "fpga",
            "tech_node": "7",
            "area": 10.0,
            "power": 2.0,
            "frequency_hz": freq,
            "clbs": clbs,
            "brams": 200,
            "dsps": 500,
            "relu_implementation": {
                "clbs_per_lane": 5,
                "brams_per_lane": 0,
                "dsps_per_lane": 0,
                "max_parallel_lanes": 64,
                "energy_per_element_pj": None,
            },
        },
        "pkg": {
            "HI_pkg_type": "2.5d",
            "inter_pkg_conn": [
                {
                    "from": "Chiplet_1",
                    "to": "Chiplet_2",
                    "connection_type": "2.5d_emib",
                    "loc": "2.5d_chiplet",
                },
                {
                    "from": "Chiplet_2",
                    "to": "Chiplet_1",
                    "connection_type": "2.5d_emib",
                    "loc": "2.5d_chiplet",
                },
            ],
            "protocol_3d": "na",
            "protocol_2.5d": "ucie_std",
            "mem_pkg_conn": {"mem_type": "hbm2", "Chiplet_1": 4, "Chiplet_2": 0},
        },
        "WL_mapping": {
            "mapping": {
                "dataflow": ["ws"],
                "assign_workload_in_ascending_order": 0,
                "static_tiling": 0,
                "merge_tiles": 0,
                "if_splitting_k": 0,
                "chiplet_data_sharing_enabled": 0,
            }
        },
        "transfer_model": {"setup_latency_ns": setup, "hop_latency_ns": 1.0},
    }


ATLAS_PROFILE = {
    "profile": "atlas_modular_v1",
    "version": 1,
    "evaluators": {
        "gemm": "legacy_scale_sim_gemm_v1",
        "relu": "legacy_fpga_relu_v1",
        "softmax": "placeholder_fpga_softmax_v0",
    },
    "placement_policy": "fixed_single_sa_single_fpga_v1",
    "movement_policy": "activation_boundary_v1",
    "transfer_model": "route_transfer_v1",
}


class ArchitectureSimilarityTests(unittest.TestCase):
    def test_features_flatten_canonical_fields(self):
        features = architecture_features(make_arch(n_chiplets=3, mem=(5, 6, 5)))

        self.assertEqual(features["n_chiplets"], 3)
        self.assertEqual(features["chip1.tech_node"], "7")
        self.assertEqual(features["chip3.sram_buf"], 256)
        self.assertEqual(features["pkg.mem_type"], "hbm3")
        self.assertEqual(features["pkg.mem_chip2"], 6)
        self.assertEqual(features["wl.dataflow"], ["os"])

    def test_identical_architectures_have_zero_distance(self):
        self.assertEqual(
            feature_distance(
                architecture_features(make_arch()), architecture_features(make_arch())
            ),
            0.0,
        )

    def test_one_field_change_is_one_over_key_count(self):
        left = architecture_features(make_arch(sram=256))
        right = architecture_features(make_arch(sram=768))
        keys = set(left) | set(right)

        self.assertAlmostEqual(feature_distance(left, right), 1 / len(keys))

    def test_agreement_reports_modal_share(self):
        records = [
            _record("a", make_arch(sram=256)),
            _record("b", make_arch(sram=256)),
            _record("c", make_arch(sram=768)),
        ]
        agreement = field_agreement(records).set_index("field")

        self.assertAlmostEqual(agreement.loc["chip1.sram_buf", "agreement"], 2 / 3)
        self.assertEqual(agreement.loc["chip1.sram_buf", "distribution"], "256=2; 768=1")

    def test_best_run_has_zero_distance_to_best(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            write_run(runs / "run01", 36.7, make_arch(sram=256))
            write_run(runs / "run02", 36.8, make_arch(sram=256))
            write_run(runs / "run03", 40.0, make_arch(sram=768))

            run_set = load_run_set("demo", str(runs / "run*"))
            decorate_distances(run_set)
            by_run = {record.run: record for record in run_set.records}

            self.assertEqual(by_run["run01"].distance_to_best, 0.0)
            self.assertGreater(by_run["run03"].distance_to_best, 0.0)

    def test_report_and_tables_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            write_run(runs / "run01", 36.7, make_arch(sram=256))
            write_run(runs / "run02", 36.8, make_arch(sram=256))
            write_run(runs / "run03", 40.0, make_arch(sram=768))
            output = root / "out"

            run_set = load_run_set("demo", str(runs / "run*"))
            self.assertEqual(len(run_set.records), 3)
            report = render_report(6, [run_set], output)

            self.assertIn("Workload 6", report)
            self.assertIn("Distance summary", report)
            self.assertTrue((output / "report.md").exists())
            self.assertTrue((output / "runs_demo.csv").exists())
            self.assertTrue((output / "distance_matrix_demo.csv").exists())
            self.assertTrue((output / "field_agreement_demo.csv").exists())

    def test_features_extract_fpga_and_transfer_fields(self):
        features = architecture_features(make_atlas_arch(clbs=8000, setup=2.0))

        self.assertEqual(features["chip1.chiplet_type"], "systolic_array")
        self.assertEqual(features["chip2.chiplet_type"], "fpga")
        self.assertEqual(features["chip2.clbs"], 8000)
        self.assertEqual(features["chip2.brams"], 200)
        self.assertEqual(features["chip2.frequency_hz"], 300000000)
        self.assertEqual(
            features["chip2.relu_implementation.max_parallel_lanes"], 64
        )
        self.assertEqual(features["transfer.setup_latency_ns"], 2.0)
        self.assertEqual(features["transfer.hop_latency_ns"], 1.0)

    def test_atlas_feature_change_is_visible_to_distance(self):
        left = architecture_features(make_atlas_arch(clbs=8000))
        right = architecture_features(make_atlas_arch(clbs=16000))
        self.assertGreater(feature_distance(left, right), 0.0)

    def test_load_run_set_reads_best_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            write_run(runs / "run01", 3.0, make_atlas_arch(), profile=ATLAS_PROFILE)
            write_run(runs / "run02", 3.1, make_atlas_arch(), profile=ATLAS_PROFILE)

            run_set = load_run_set("atlas", str(runs / "run*"))
            self.assertEqual(len(run_set.records), 2)
            for record in run_set.records:
                self.assertEqual(record.profile_name, "atlas_modular_v1")
                self.assertEqual(len(record.profile_fingerprint), 12)

    def test_legacy_run_without_profile_has_none(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            write_run(runs / "run01", 3.0, make_arch())
            run_set = load_run_set("legacy", str(runs / "run*"))
            self.assertIsNone(run_set.records[0].profile_name)

    def test_within_tolerance_count_positive_and_negative(self):
        positive = [
            _record("a", make_arch(), objective=100.0),
            _record("b", make_arch(), objective=100.05),
            _record("c", make_arch(), objective=101.0),
        ]
        self.assertEqual(within_tolerance_count(positive), 2)

        negative = [
            _record("a", make_arch(), objective=-2.0314255921830546),
            _record("b", make_arch(), objective=-2.0314255921830546),
            _record("c", make_arch(), objective=-1.5),
        ]
        self.assertEqual(within_tolerance_count(negative), 2)

    def test_profile_agreement_reports_members(self):
        records = [
            _record("a", make_arch(), profile_name="p1"),
            _record("b", make_arch(), profile_name="p1"),
            _record("c", make_arch(), profile_name="p2"),
        ]
        agreement = profile_agreement(records)
        self.assertEqual(list(agreement["profile"]), ["p1", "p2"])
        self.assertEqual(int(agreement.iloc[0]["runs"]), 2)

    def test_report_includes_profile_section_and_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            write_run(runs / "run01", 3.0, make_atlas_arch(), profile=ATLAS_PROFILE)
            write_run(runs / "run02", 3.0, make_atlas_arch(), profile=ATLAS_PROFILE)
            write_run(runs / "run03", 4.0, make_atlas_arch(clbs=16000), profile=ATLAS_PROFILE)
            output = root / "out"

            run_set = load_run_set("atlas", str(runs / "run*"))
            report = render_report("atlas_funnel", [run_set], output)

            self.assertIn("Within 0.1% of best", report)
            self.assertIn("Evaluation profile agreement", report)
            self.assertIn("atlas_modular_v1", report)
            self.assertTrue((output / "profiles_atlas.csv").exists())


def _record(run, architecture, objective=1.0, profile_name=None):
    from script.analyze_architecture_similarity import RunRecord

    return RunRecord(
        set_label="demo",
        run=run,
        objective=objective,
        moves=100,
        fingerprint="fingerprint",
        features=architecture_features(architecture),
        profile_name=profile_name,
    )


if __name__ == "__main__":
    unittest.main()
