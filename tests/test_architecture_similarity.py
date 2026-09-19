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
    render_report,
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


def write_run(directory, objective, architecture, moves=100):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "best_arch_run.json").write_text(
        json.dumps(architecture), encoding="utf-8"
    )
    pd.DataFrame(
        {"SA_run_loop": list(range(1, moves + 1)), "best_cost_after": [objective] * moves}
    ).to_csv(directory / "sa_metrics_run.csv", index=False)


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


def _record(run, architecture):
    from script.analyze_architecture_similarity import RunRecord

    return RunRecord(
        set_label="demo",
        run=run,
        objective=1.0,
        moves=100,
        fingerprint="fingerprint",
        features=architecture_features(architecture),
    )


if __name__ == "__main__":
    unittest.main()
