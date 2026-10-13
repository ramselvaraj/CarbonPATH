import copy
import json
import tempfile
import unittest
from pathlib import Path

from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.DesignPointFile import (
    check_design_point_workload,
    design_point_document,
    latest_result_file,
    parse_design_point,
    profile_differs,
    read_design_point,
    write_design_point,
)
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.WorkloadMapping import initial_workload_mapping


ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
GRAPH = Path("cfg/examples/atlas/dense_relu_funnel.graph_dump.json")
OTHER_GRAPH = Path("cfg/examples/atlas/dense_softmax.graph_dump.json")
PROFILE = Path("cfg/profiles/atlas_multi_sa_fpga_v1.json")
OTHER_PROFILE = Path("cfg/profiles/atlas_modular_v1.json")


class DesignPointFileTests(unittest.TestCase):
    def setUp(self):
        with ARCHITECTURE.open(encoding="utf-8") as file:
            self.architecture = json.load(file)
        self.graph = load_atlas_graph(GRAPH)
        self.profile = load_evaluation_profile(PROFILE)
        first = initial_workload_mapping(self.graph, self.architecture, self.profile)
        # Make the two GEMMs differ, so the file must keep each one.
        self.mapping = first.with_operation(
            first.gemm_indexes[1], dataflow="os", split_k=True
        )
        self.document = design_point_document(
            self.architecture, self.mapping, self.graph, self.profile
        )

    def test_file_has_the_architecture_and_the_workload_mapping_sections(self):
        self.assertEqual(
            set(self.document),
            {
                "format",
                "version",
                "workload",
                "evaluated_with",
                "architecture",
                "workload_mapping",
            },
        )
        self.assertNotIn("WL_mapping", self.document["architecture"])
        self.assertIn("Chiplet_1", self.document["architecture"])
        self.assertIn("transfer_model", self.document["architecture"])
        self.assertEqual(
            self.document["workload"],
            {"graph": self.graph.name, "fingerprint": self.graph.fingerprint()},
        )
        self.assertEqual(
            self.document["evaluated_with"],
            {
                "profile": self.profile.name,
                "fingerprint": self.profile.fingerprint(),
            },
        )

    def test_operations_name_chiplets_and_keep_gemm_settings(self):
        operations = self.document["workload_mapping"]["operations"]
        self.assertEqual(
            [entry["operation"] for entry in operations],
            [operation.operation_id for operation in self.graph.operations],
        )
        gemm, relu, second_gemm = operations[:3]
        self.assertEqual(gemm["endpoints"], ["Chiplet_1"])
        self.assertEqual(relu["endpoints"], ["Chiplet_2"])
        self.assertEqual(
            set(gemm),
            {
                "operation",
                "type",
                "endpoints",
                "dataflow",
                "split_k",
                "tile_order",
                "merge_tiles",
            },
        )
        self.assertEqual(set(relu), {"operation", "type", "endpoints"})
        self.assertEqual(second_gemm["dataflow"], "os")
        self.assertIs(gemm["merge_tiles"], False)
        self.assertEqual(
            len(self.document["workload_mapping"]["handoffs"]), len(operations) - 1
        )

    def test_a_written_file_reads_back_to_the_same_design_point(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "best_design_point.json"
            write_design_point(path, self.document)
            loaded = read_design_point(path)

        self.assertEqual(loaded.mapping, self.mapping)
        self.assertEqual(loaded.mapping.fingerprint(), self.mapping.fingerprint())
        hardware = {
            key: value
            for key, value in loaded.architecture.items()
            if key != "WL_mapping"
        }
        self.assertEqual(hardware, self.document["architecture"])
        # The scheduler block is built again from the first GEMM.
        first_gemm = self.mapping.operations[0]
        block = loaded.architecture["WL_mapping"]["mapping"]
        self.assertEqual(block["dataflow"], [first_gemm.dataflow])
        self.assertEqual(block["if_splitting_k"], int(first_gemm.split_k))
        self.assertEqual(block["merge_tiles"], 0)

    def test_an_original_architecture_file_has_no_workload_mapping(self):
        loaded = parse_design_point(copy.deepcopy(self.architecture))

        self.assertEqual(loaded.architecture, self.architecture)
        self.assertIsNone(loaded.mapping)
        check_design_point_workload(loaded, self.graph)
        self.assertFalse(profile_differs(loaded, self.profile))

    def test_a_file_for_a_different_workload_is_rejected(self):
        loaded = parse_design_point(self.document)
        check_design_point_workload(loaded, self.graph)

        with self.assertRaisesRegex(UnsupportedEvaluation, "is for workload"):
            check_design_point_workload(loaded, load_atlas_graph(OTHER_GRAPH))

    def test_a_different_profile_is_reported_but_not_rejected(self):
        loaded = parse_design_point(self.document)

        self.assertFalse(profile_differs(loaded, self.profile))
        self.assertTrue(
            profile_differs(loaded, load_evaluation_profile(OTHER_PROFILE))
        )

    def test_a_chiplet_that_is_not_in_the_architecture_is_rejected(self):
        document = copy.deepcopy(self.document)
        document["workload_mapping"]["operations"][0]["endpoints"] = ["Chiplet_9"]

        with self.assertRaisesRegex(UnsupportedEvaluation, "not in the architecture"):
            parse_design_point(document)

    def test_an_operation_on_both_endpoint_kinds_is_rejected(self):
        document = copy.deepcopy(self.document)
        document["workload_mapping"]["operations"][0]["endpoints"] = [
            "Chiplet_1",
            "Chiplet_2",
        ]

        with self.assertRaisesRegex(UnsupportedEvaluation, "mixes"):
            parse_design_point(document)

    def test_a_handoff_that_does_not_follow_the_operations_is_rejected(self):
        document = copy.deepcopy(self.document)
        document["workload_mapping"]["handoffs"][0]["to"] = "elsewhere"

        with self.assertRaisesRegex(UnsupportedEvaluation, "handoff 1"):
            parse_design_point(document)

        document = copy.deepcopy(self.document)
        document["workload_mapping"]["handoffs"].pop()
        with self.assertRaisesRegex(UnsupportedEvaluation, "one handoff"):
            parse_design_point(document)

    def test_an_unknown_version_is_rejected(self):
        document = {**self.document, "version": 99}

        with self.assertRaisesRegex(UnsupportedEvaluation, "version"):
            parse_design_point(document)

    def test_latest_result_file_prefers_a_design_point(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self.assertIsNone(latest_result_file(run_dir))
            (run_dir / "best_arch_run_1.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                latest_result_file(run_dir).name, "best_arch_run_1.json"
            )
            (run_dir / "best_design_point_run_1.json").write_text(
                "{}", encoding="utf-8"
            )
            (run_dir / "best_design_point_run_2.json").write_text(
                "{}", encoding="utf-8"
            )
            self.assertEqual(
                latest_result_file(run_dir).name, "best_design_point_run_2.json"
            )


if __name__ == "__main__":
    unittest.main()
