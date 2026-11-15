import copy
import json
import random
import unittest
from pathlib import Path

from system.utils.AtlasAnnealingMoves import mutate_atlas_design_point
from system.utils.FpgaChiplet import FpgaChiplet
from system.utils.FpgaNewChiplet import new_fpga_chiplet
from system.utils.FpgaParts import (
    carbon_tech_node,
    fpga_summary,
    load_parts,
    neurigami_part_for,
    part_chiplet_fields,
    part_for_board,
)
from system.utils.Rule4MlBoards import (
    board_chiplet_fields,
    fpga_capacity_vector,
    fpga_resource_vector,
    load_boards,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.WorkloadMapping import initial_workload_mapping
from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.EvaluationProfile import load_evaluation_profile

ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture_xczu7ev.json")
BOARD_ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space_parts.json")
GRAPH = Path("cfg/examples/atlas/dense_relu_funnel.graph_dump.json")
PROFILE = Path("cfg/profiles/atlas_fused_relu_v1.json")


def _load(path):
    return json.loads(Path(path).read_text())


def _chiplet(**over):
    value = _load(ARCHITECTURE)["Chiplet_2"]
    value.update(over)
    return value


class PartTableTest(unittest.TestCase):
    def test_every_rule4ml_board_has_a_part_with_the_same_resources(self):
        boards = load_boards()
        for board, caps in boards.items():
            part = part_for_board(board)
            self.assertIsNotNone(part, board)
            fields = part_chiplet_fields(part)
            self.assertEqual(fields["luts"], caps["max_lut"], board)
            self.assertEqual(fields["ffs"], caps["max_ff"], board)
            self.assertEqual(fields["brams"], caps["max_bram"], board)     # 18 Kb units
            self.assertEqual(fields["dsps"], caps["max_dsp"], board)

    def test_the_part_of_pynq_z2_reproduces_the_board_chiplet(self):
        board = board_chiplet_fields("pynq-z2")
        part = part_chiplet_fields("xc7z020")
        for field in ("clbs", "brams", "dsps", "luts", "ffs", "area", "power"):
            self.assertAlmostEqual(part[field], board[field], places=3, msg=field)

    def test_the_neurigami_parts_are_the_ones_neurigami_calibrates(self):
        known = {n for n, e in load_parts()["parts"].items() if e["neurigami"]}
        self.assertEqual(known, {"xczu7ev", "xck26", "xczu3eg", "xczu5eg", "xc7z020", "xcu55c"})

    def test_16_nm_maps_to_the_nearest_carbon_node(self):
        self.assertEqual(carbon_tech_node(16), 14)
        self.assertEqual(carbon_tech_node(28), 28)

    def test_an_unknown_part_is_refused(self):
        with self.assertRaises(UnsupportedEvaluation):
            part_chiplet_fields("xc0nothing")


class PartChipletTest(unittest.TestCase):
    def test_the_example_chiplet_loads(self):
        chiplet = FpgaChiplet.from_dict("Chiplet_2", _chiplet())
        self.assertEqual(chiplet.fpga_part, "xczu7ev")
        self.assertEqual(chiplet.uram, 96)

    def test_counts_that_disagree_with_the_part_are_refused(self):
        for field, value in (("dsps", 1), ("brams", 10), ("clbs", 5), ("luts", 7), ("uram", 3)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "has"):
                FpgaChiplet.from_dict("Chiplet_2", _chiplet(**{field: value}))

    def test_a_board_that_is_not_the_parts_board_is_refused(self):
        with self.assertRaisesRegex(ValueError, "zcu102"):       # the board check or the part check, both refuse
            FpgaChiplet.from_dict("Chiplet_2", _chiplet(board="zcu102"))

    def test_a_bad_part_or_cap_is_refused(self):
        with self.assertRaises(ValueError):
            FpgaChiplet.from_dict("Chiplet_2", _chiplet(fpga_part="nope"))
        with self.assertRaises(ValueError):
            FpgaChiplet.from_dict("Chiplet_2", _chiplet(max_utilization={"lut": 1.5}))
        with self.assertRaises(ValueError):
            FpgaChiplet.from_dict("Chiplet_2", _chiplet(max_utilization={"gpu": 0.5}))

    def test_a_board_chiplet_still_loads_and_has_no_cap(self):
        chiplet = FpgaChiplet.from_dict("Chiplet_2", _load(BOARD_ARCHITECTURE)["Chiplet_2"])
        self.assertIsNone(chiplet.max_utilization)
        self.assertEqual(fpga_resource_vector(chiplet), fpga_capacity_vector(chiplet))

    def test_the_cap_scales_what_a_design_may_use_but_not_the_capacity(self):
        chiplet = FpgaChiplet.from_dict("Chiplet_2", _chiplet())
        usable, capacity = fpga_resource_vector(chiplet), fpga_capacity_vector(chiplet)
        self.assertEqual(capacity["lut"], 230400)
        self.assertAlmostEqual(usable["lut"], 230400 * 0.85)
        self.assertAlmostEqual(usable["dsp"], 1728 * 0.95)

    def test_neurigami_part_lookup(self):
        chiplet = FpgaChiplet.from_dict("Chiplet_2", _chiplet())
        self.assertEqual(neurigami_part_for(chiplet), "xczu7ev")
        pynq = FpgaChiplet.from_dict("Chiplet_2", _load(BOARD_ARCHITECTURE)["Chiplet_2"])
        self.assertEqual(neurigami_part_for(pynq), "xc7z020")


CUSTOM = Path("cfg/examples/sa_fpga_architecture_custom.json")


class FreeSpecTest(unittest.TestCase):
    def test_a_free_spec_loads_without_a_part_and_keeps_its_own_counts(self):
        chiplet = FpgaChiplet.from_dict("Chiplet_2", _load(CUSTOM)["Chiplet_2"])
        self.assertIsNone(chiplet.fpga_part)
        self.assertEqual((chiplet.luts, chiplet.dsps), (100000, 700))

    def test_its_calibration_is_the_nearest_part_unless_set(self):
        value = _load(CUSTOM)["Chiplet_2"]
        chiplet = FpgaChiplet.from_dict("Chiplet_2", value)
        self.assertEqual(neurigami_part_for(chiplet), "xck26")
        value["calibration_part"] = "xc7z020"
        self.assertEqual(neurigami_part_for(FpgaChiplet.from_dict("Chiplet_2", value)), "xc7z020")

    def test_a_calibration_part_neurigami_does_not_know_is_refused(self):
        value = _load(CUSTOM)["Chiplet_2"]
        for bad in ("xczu9eg", "nope"):
            value["calibration_part"] = bad
            with self.assertRaisesRegex(ValueError, "calibration_part"):
                FpgaChiplet.from_dict("Chiplet_2", value)

    def test_the_summary_says_custom_and_records_the_limitation(self):
        entry = fpga_summary(_load(CUSTOM))["Chiplet_2"]
        self.assertEqual(entry["spec_kind"], "custom")
        self.assertIsNone(entry["fpga_part"])
        self.assertEqual(entry["neurigami_calibration_part"], "xck26")
        self.assertIn("LIMITATION", " ".join(entry["notes"]))
        own = fpga_summary(_load(ARCHITECTURE))["Chiplet_2"]
        self.assertEqual(own["spec_kind"], "part")
        self.assertNotIn("LIMITATION", " ".join(own["notes"]))


class PartMoveTest(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        self.params = _load(SEARCH_SPACE)
        graph = load_atlas_graph(GRAPH)
        self.mapping = initial_workload_mapping(
            graph, self.architecture, load_evaluation_profile(PROFILE),
            _load("cfg/experiments/atlas_modular_search_space.json"),
        )

    def test_the_part_move_swaps_the_whole_part(self):
        self.params["move_weights"] = {"fpga_part": 1.0}
        seen = set()
        for seed in range(20):
            moved, _, move = mutate_atlas_design_point(
                self.architecture, self.mapping, self.params, rng=random.Random(seed)
            )
            fpga = moved["Chiplet_2"]
            self.assertEqual(move, "fpga_part")
            self.assertIn(fpga["fpga_part"], self.params["fpga"]["parts"])
            self.assertNotEqual(fpga["fpga_part"], "xczu7ev")
            for name, value in part_chiplet_fields(fpga["fpga_part"]).items():
                if name != "max_utilization":
                    self.assertEqual(fpga[name], value, name)
            FpgaChiplet.from_dict("Chiplet_2", fpga)        # still a valid chiplet
            seen.add(fpga["fpga_part"])
        self.assertEqual(seen, {"xc7z020", "xck26"})

    def test_a_cap_the_architecture_set_survives_a_part_move(self):
        self.architecture["Chiplet_2"]["max_utilization"] = {"lut": 0.6}
        self.params["move_weights"] = {"fpga_part": 1.0}
        moved, _, _ = mutate_atlas_design_point(
            self.architecture, self.mapping, self.params, rng=random.Random(0)
        )
        self.assertEqual(moved["Chiplet_2"]["max_utilization"], {"lut": 0.6})

    def test_a_chiplet_outside_the_search_space_parts_is_refused(self):
        self.params["fpga"]["parts"] = ["xc7z020", "xck26"]
        with self.assertRaisesRegex(UnsupportedEvaluation, "search space parts"):
            mutate_atlas_design_point(self.architecture, self.mapping, self.params, rng=random.Random(0))

    def test_without_a_parts_list_the_move_is_not_offered(self):
        params = _load("cfg/experiments/atlas_modular_search_space.json")
        params["move_weights"] = {"fpga_part": 1.0}
        with self.assertRaises(UnsupportedEvaluation):
            mutate_atlas_design_point(_load(BOARD_ARCHITECTURE), self.mapping, params, rng=random.Random(0))

    def test_a_new_chiplet_is_drawn_from_the_parts(self):
        params = copy.deepcopy(self.params)
        params["fpga"]["new_chiplet"] = "from_search_space_v1"
        chiplet = new_fpga_chiplet(self.architecture, ["Chiplet_2"], params, random.Random(1))
        self.assertIn(chiplet["fpga_part"], params["fpga"]["parts"])
        FpgaChiplet.from_dict("Chiplet_3", chiplet)


class SummaryTest(unittest.TestCase):
    def test_the_summary_states_the_chiplet_and_what_is_not_sourced(self):
        entry = fpga_summary(_load(ARCHITECTURE))["Chiplet_2"]
        self.assertEqual(entry["fpga_part"], "xczu7ev")
        self.assertEqual(entry["luts"], 230400)
        self.assertEqual(entry["frequency_mhz"], 300.0)
        self.assertEqual(entry["dsp_generation"], "DSP48E2")
        self.assertEqual(entry["area_power_source"], "proxy")
        self.assertFalse(entry["clock_checked"])
        notes = " ".join(entry["notes"])
        self.assertIn("not checked", notes)
        self.assertIn("proxy", notes)
        self.assertIn("16 nm", notes)                    # the carbon model uses its 14 nm node

    def test_a_board_chiplet_is_summarised_too_and_an_sa_only_design_has_no_summary(self):
        self.assertIn("Chiplet_2", fpga_summary(_load(BOARD_ARCHITECTURE)))
        self.assertEqual(fpga_summary(_load("cfg/examples/sa_only_architecture.json")), {})

    def test_the_design_point_file_holds_the_summary(self):
        from system.utils.DesignPointFile import design_point_document, parse_design_point
        architecture = _load(ARCHITECTURE)
        graph = load_atlas_graph(GRAPH)
        profile = load_evaluation_profile(PROFILE)
        mapping = initial_workload_mapping(
            graph, architecture, profile, _load("cfg/experiments/atlas_modular_search_space.json")
        )
        document = design_point_document(architecture, mapping, graph, profile)
        self.assertEqual(document["fpga_chiplets"]["Chiplet_2"]["fpga_part"], "xczu7ev")
        loaded = parse_design_point(json.loads(json.dumps(document)))     # the summary is ignored on read
        self.assertEqual(loaded.architecture["Chiplet_2"]["fpga_part"], "xczu7ev")


if __name__ == "__main__":
    unittest.main()
