import copy
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

from system.utils.AtlasAnnealingMoves import (
    MOVE_TYPES,
    mutate_atlas_design_point,
    sequential_gemm_search_space,
    validate_atlas_architecture,
)
from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.WorkloadMapping import initial_workload_mapping
from system.utils.ChipletSystem import ChipletSystem
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space_placeholders.json")
GRAPH = Path("cfg/examples/atlas/dense_relu_funnel.graph_dump.json")
PROFILE = Path("cfg/profiles/atlas_multi_sa_fpga_v1.json")
# These parameters have one option for the reference design point.
ONE_OPTION_MOVES = (
    "gemm_merge_tiles",
    "gemm_sa_set",
    "handoff_movement",
    "fpga_count",
    "transfer_model",
)


def _load(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


class _MoveRng:
    def __init__(self, move_name, preferred=None):
        self.move_name = move_name
        self.preferred = preferred
        self.selected_move = False

    def choice(self, values):
        if not self.selected_move and self.move_name in values:
            self.selected_move = True
            return self.move_name
        if self.preferred in values:
            return self.preferred
        return values[0]

    def choices(self, values, weights, k):
        del weights
        return [self.choice(values) for _ in range(k)]


class _GemmGraph:
    """A chain of GEMM operations for an architecture without an FPGA."""

    def __init__(self, count):
        self.operations = tuple(
            SimpleNamespace(operation_id=f"gemm_{index}", operation_type="gemm")
            for index in range(count)
        )


class AtlasAnnealingMoveTests(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        self.search_space = _load(SEARCH_SPACE)
        self.graph = load_atlas_graph(GRAPH)
        self.profile = load_evaluation_profile(PROFILE)

    def _mapping(self, architecture, graph=None):
        has_fpga = any(
            isinstance(chiplet, dict) and chiplet.get("chiplet_type") == "fpga"
            for chiplet in architecture.values()
        )
        if not has_fpga and graph is None:
            graph = _GemmGraph(2)
        return initial_workload_mapping(
            graph or self.graph, architecture, self.profile, self.search_space
        )

    def test_explicit_move_weights_select_the_only_enabled_family(self):
        params = copy.deepcopy(self.search_space)
        params["move_weights"] = {
            move_name: 1.0 if move_name == "sa_array" else 0.0
            for move_name in MOVE_TYPES
        }

        for seed in range(20):
            moved, mapping, move = mutate_atlas_design_point(
                self.architecture,
                self._mapping(self.architecture),
                params,
                rng=random.Random(seed),
            )

            self.assertEqual(move, "sa_array")
            self.assertNotEqual(moved, self.architecture)
            self.assertEqual(mapping, self._mapping(self.architecture))

    def test_unknown_move_weight_is_rejected(self):
        params = copy.deepcopy(self.search_space)
        params["move_weights"] = {"invented_move": 1.0}

        with self.assertRaisesRegex(UnsupportedEvaluation, "unknown move"):
            mutate_atlas_design_point(
                self.architecture,
                self._mapping(self.architecture),
                params,
                rng=random.Random(1),
            )

    def test_weighted_moves_that_are_unavailable_do_not_become_candidates(self):
        architecture = copy.deepcopy(self.architecture)
        del architecture["Chiplet_2"]
        architecture["pkg"] = {
            "HI_pkg_type": "2d",
            "inter_pkg_conn": "2d_na",
            "protocol_3d": "na",
            "protocol_2.5d": "na",
            "mem_pkg_conn": {"mem_type": "ddr5", "Chiplet_1": 8},
        }
        params = copy.deepcopy(self.search_space)
        params["move_weights"] = {"fpga_clbs": 1.0}

        with self.assertRaisesRegex(UnsupportedEvaluation, "enabled move"):
            mutate_atlas_design_point(
                architecture,
                self._mapping(architecture),
                params,
                rng=random.Random(1),
            )

    def test_invalid_move_weight_is_rejected_even_when_family_is_unavailable(self):
        architecture = copy.deepcopy(self.architecture)
        del architecture["Chiplet_2"]
        architecture["pkg"] = {
            "HI_pkg_type": "2d",
            "inter_pkg_conn": "2d_na",
            "protocol_3d": "na",
            "protocol_2.5d": "na",
            "mem_pkg_conn": {"mem_type": "ddr5", "Chiplet_1": 8},
        }
        params = copy.deepcopy(self.search_space)
        params["move_weights"] = {"fpga_clbs": -1.0, "sa_array": 1.0}

        with self.assertRaisesRegex(UnsupportedEvaluation, "finite non-negative"):
            mutate_atlas_design_point(
                architecture,
                self._mapping(architecture),
                params,
                rng=random.Random(1),
            )

    def test_every_move_with_an_alternative_is_reachable_from_reference_design(self):
        start = self._mapping(self.architecture)
        for move_name in MOVE_TYPES:
            if move_name in ONE_OPTION_MOVES:
                continue
            with self.subTest(move_name=move_name):
                params = copy.deepcopy(self.search_space)
                params["move_weights"] = {
                    candidate: 1.0 if candidate == move_name else 0.0
                    for candidate in MOVE_TYPES
                }
                moved, mapping, selected = mutate_atlas_design_point(
                    self.architecture,
                    start,
                    params,
                    rng=_MoveRng(move_name),
                    default_movement_policy=self.profile.movement_policy,
                    fusion_types=("relu",),
                )

                self.assertEqual(selected, move_name)
                self.assertTrue(moved != self.architecture or mapping != start)

    def test_a_parameter_with_one_option_is_never_selected(self):
        start = self._mapping(self.architecture)
        for move_name in ONE_OPTION_MOVES:
            with self.subTest(move_name=move_name):
                params = copy.deepcopy(self.search_space)
                params["move_weights"] = {move_name: 1.0, "sa_array": 1.0}
                for seed in range(20):
                    _, _, selected = mutate_atlas_design_point(
                        self.architecture, start, params, rng=random.Random(seed)
                    )
                    self.assertEqual(selected, "sa_array")

                params["move_weights"] = {move_name: 1.0}
                with self.assertRaisesRegex(UnsupportedEvaluation, "enabled move"):
                    mutate_atlas_design_point(
                        self.architecture, start, params, rng=random.Random(1)
                    )

    def test_one_option_move_is_an_invalid_proposal_when_skipping_is_off(self):
        params = copy.deepcopy(self.search_space)
        params["move_weights"] = {"transfer_model": 1.0}
        params["skip_moves_without_alternative"] = False

        with self.assertRaisesRegex(UnsupportedEvaluation, "no alternative"):
            mutate_atlas_design_point(
                self.architecture,
                self._mapping(self.architecture),
                params,
                rng=random.Random(1),
            )

    def test_a_second_fpga_count_option_is_reported_as_unsupported(self):
        params = copy.deepcopy(self.search_space)
        params["fpga"]["count"] = [1, 2]
        params["move_weights"] = {"fpga_count": 1.0}

        with self.assertRaisesRegex(UnsupportedEvaluation, "one option now"):
            mutate_atlas_design_point(
                self.architecture,
                self._mapping(self.architecture),
                params,
                rng=random.Random(1),
            )

    def test_many_moves_preserve_supported_sa_count_and_single_fpga(self):
        rng = random.Random(2024)
        architecture = self.architecture
        mapping = self._mapping(architecture)
        valid_moves = 0
        for _ in range(400):
            try:
                architecture, mapping, move = mutate_atlas_design_point(
                    architecture,
                    mapping,
                    self.search_space,
                    rng=rng,
                )
            except UnsupportedEvaluation:
                continue
            valid_moves += 1
            self.assertIn(move, MOVE_TYPES)
            endpoints = validate_atlas_architecture(architecture)
            self.assertGreaterEqual(len(endpoints.sa_keys), 1)
            self.assertLessEqual(len(endpoints.sa_keys), 6)
            self.assertEqual(len(endpoints.fpga_keys), 1)
            self.assertTrue(
                architecture[endpoints.fpga_keys[0]]["chiplet_type"] == "fpga"
            )
            self.assertTrue(
                all("chiplet_type" not in architecture[key] for key in endpoints.sa_keys)
            )
            for operation in mapping.operations:
                valid_ids = endpoints.sa_ids if operation.is_gemm else endpoints.fpga_ids
                self.assertTrue(set(operation.endpoint_ids) <= set(valid_ids))
        self.assertGreaterEqual(valid_moves, 300)

    def test_mutation_does_not_mutate_input_architecture(self):
        original = json.dumps(self.architecture, sort_keys=True)
        rng = random.Random(7)
        mutate_atlas_design_point(
            self.architecture,
            self._mapping(self.architecture),
            self.search_space,
            rng=rng,
        )
        self.assertEqual(json.dumps(self.architecture, sort_keys=True), original)

    def test_reference_architecture_validates(self):
        sa_key, fpga_key = (
            validate_atlas_architecture(self.architecture).require_single_sa_fpga()
        )
        self.assertEqual(sa_key, "Chiplet_1")
        self.assertEqual(fpga_key, "Chiplet_2")

    def test_sa_only_architecture_is_valid(self):
        sa_only = {
            key: value
            for key, value in self.architecture.items()
            if not key.startswith("Chiplet_2")
        }
        sa_only["pkg"] = {
            "HI_pkg_type": "2d",
            "inter_pkg_conn": "2d_na",
            "protocol_3d": "na",
            "protocol_2.5d": "na",
            "mem_pkg_conn": {"mem_type": "hbm2", "Chiplet_1": 4},
        }
        endpoints = validate_atlas_architecture(sa_only)

        self.assertEqual(endpoints.sa_keys, ("Chiplet_1",))
        self.assertEqual(endpoints.fpga_keys, ())

    def test_dangling_package_endpoint_is_rejected(self):
        dangling = {
            key: value
            for key, value in self.architecture.items()
            if not key.startswith("Chiplet_2")
        }

        with self.assertRaisesRegex(UnsupportedEvaluation, "missing Chiplet_2"):
            validate_atlas_architecture(dangling)

    def test_multiple_sa_chiplets_are_valid(self):
        architecture = _load(ARCHITECTURE)
        architecture["Chiplet_3"] = dict(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"]["Chiplet_3"] = 4
        architecture["pkg"]["inter_pkg_conn"].append(
            {
                "from": "Chiplet_2",
                "to": "Chiplet_3",
                "connection_type": "2.5d_emib",
                "loc": "2.5d_chiplet",
            }
        )

        endpoints = validate_atlas_architecture(architecture)

        self.assertEqual(endpoints.sa_keys, ("Chiplet_1", "Chiplet_3"))
        self.assertEqual(endpoints.fpga_keys, ("Chiplet_2",))

    def test_architecture_without_sa_is_rejected(self):
        fpga_only = {
            key: value
            for key, value in self.architecture.items()
            if not key.startswith("Chiplet_1")
        }

        with self.assertRaisesRegex(UnsupportedEvaluation, "at least one"):
            validate_atlas_architecture(fpga_only)

    def test_disconnected_chiplet_is_rejected(self):
        architecture = _load(ARCHITECTURE)
        architecture["Chiplet_3"] = dict(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"]["Chiplet_3"] = 4

        with self.assertRaisesRegex(UnsupportedEvaluation, "connect every"):
            validate_atlas_architecture(architecture)

    def test_more_than_six_sa_chiplets_are_rejected(self):
        architecture = _load(ARCHITECTURE)
        sa = architecture.pop("Chiplet_1")
        fpga = architecture.pop("Chiplet_2")
        architecture["Chiplet_8"] = fpga
        architecture["pkg"]["mem_pkg_conn"] = {"mem_type": "hbm2"}
        architecture["pkg"]["inter_pkg_conn"] = []
        for index in range(1, 8):
            key = f"Chiplet_{index}"
            architecture[key] = dict(sa)
            architecture["pkg"]["mem_pkg_conn"][key] = 1

        with self.assertRaisesRegex(UnsupportedEvaluation, "at most six"):
            validate_atlas_architecture(architecture)

    def test_gemm_setting_moves_change_one_gemm_of_the_mapping(self):
        start = self._mapping(self.architecture)
        for move_name, field in (
            ("gemm_dataflow", "dataflow"),
            ("gemm_split_k", "split_k"),
            ("gemm_assignment_order", "tile_order"),
        ):
            with self.subTest(move_name=move_name):
                architecture, mapping, actual_move = mutate_atlas_design_point(
                    self.architecture,
                    start,
                    self.search_space,
                    rng=_MoveRng(move_name),
                )

                self.assertEqual(actual_move, move_name)
                self.assertEqual(architecture, self.architecture)
                changed = [
                    index
                    for index in start.gemm_indexes
                    if getattr(mapping.operations[index], field)
                    != getattr(start.operations[index], field)
                ]
                self.assertEqual(len(changed), 1)
                self.assertGreater(len(start.gemm_indexes), 1)

    def test_shared_scope_changes_every_gemm_together(self):
        params = sequential_gemm_search_space(
            {**self.search_space, "max_chiplet": 6}
        )
        start = self._mapping(self.architecture)

        _, mapping, move = mutate_atlas_design_point(
            self.architecture, start, params, rng=_MoveRng("gemm_dataflow")
        )

        self.assertEqual(move, "gemm_dataflow")
        dataflows = {mapping.operations[index].dataflow for index in mapping.gemm_indexes}
        self.assertEqual(len(dataflows), 1)
        self.assertNotEqual(dataflows, {start.operations[0].dataflow})

    def test_a_listed_gemm_option_limits_the_move(self):
        params = copy.deepcopy(self.search_space)
        params["gemm_mapping"] = {"dataflow": ["ws", "os"]}
        mapping = self._mapping(self.architecture)

        for seed in range(30):
            _, mapping, _ = mutate_atlas_design_point(
                self.architecture,
                mapping,
                {**params, "move_weights": {"gemm_dataflow": 1.0}},
                rng=random.Random(seed),
            )
            for index in mapping.gemm_indexes:
                self.assertIn(mapping.operations[index].dataflow, ("ws", "os"))

    def test_sa_set_move_changes_the_sa_set_of_one_gemm(self):
        architecture = self._two_sa_architecture()
        mapping = self._mapping(architecture)
        params = {**self.search_space, "move_weights": {"gemm_sa_set": 1.0}}
        seen = set()

        for seed in range(40):
            before = mapping
            moved, mapping, move = mutate_atlas_design_point(
                architecture, mapping, params, rng=random.Random(seed)
            )
            self.assertEqual(move, "gemm_sa_set")
            self.assertEqual(moved, architecture)
            changed = [
                index
                for index in mapping.gemm_indexes
                if mapping.operations[index].endpoint_ids
                != before.operations[index].endpoint_ids
            ]
            self.assertEqual(len(changed), 1)
            for index in mapping.gemm_indexes:
                sa_ids = mapping.operations[index].endpoint_ids
                self.assertTrue(sa_ids)
                self.assertTrue(set(sa_ids) <= {0, 2})
                seen.add(sa_ids)
        self.assertEqual(seen, {(0,), (2,), (0, 2)})

    def test_handoff_move_selects_a_listed_policy_for_one_handoff(self):
        architecture = self._sa_only_architecture()
        graph = _GemmGraph(3)
        mapping = self._mapping(architecture, graph)
        options = self.search_space["movement_policy"]["sa_to_sa"]
        self.assertEqual(mapping.handoffs, (options[0], options[0]))
        params = {**self.search_space, "move_weights": {"handoff_movement": 1.0}}

        moved, moved_mapping, move = mutate_atlas_design_point(
            architecture, mapping, params, rng=random.Random(3)
        )

        self.assertEqual(move, "handoff_movement")
        self.assertEqual(moved, architecture)
        changed = [
            index
            for index, policy in enumerate(moved_mapping.handoffs)
            if policy != mapping.handoffs[index]
        ]
        self.assertEqual(len(changed), 1)
        self.assertIn(moved_mapping.handoffs[changed[0]], options)

    def test_sa_count_move_keeps_the_mapping_on_existing_endpoints(self):
        architecture = self._two_sa_architecture()
        mapping = self._mapping(architecture)
        first_gemm = mapping.gemm_indexes[0]
        mapping = mapping.with_operation(first_gemm, endpoint_ids=(0,))
        params = {**self.search_space, "move_weights": {"sa_count": 1.0}}

        moved, moved_mapping, _ = mutate_atlas_design_point(
            architecture, mapping, params, rng=_MoveRng("sa_count", preferred="add")
        )

        endpoints = validate_atlas_architecture(moved)
        self.assertEqual(len(endpoints.sa_keys), 3)
        for operation in moved_mapping.operations:
            valid_ids = endpoints.sa_ids if operation.is_gemm else endpoints.fpga_ids
            self.assertTrue(set(operation.endpoint_ids) <= set(valid_ids))
        # The new SA joins every GEMM; the first GEMM keeps its smaller set.
        self.assertEqual(len(moved_mapping.operations[first_gemm].endpoint_ids), 2)
        self.assertEqual(
            len(moved_mapping.operations[mapping.gemm_indexes[1]].endpoint_ids), 3
        )

    def _two_sa_architecture(self):
        architecture = _load(ARCHITECTURE)
        architecture["Chiplet_3"] = dict(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"]["Chiplet_3"] = 4
        architecture["pkg"]["inter_pkg_conn"].append(
            {
                "from": "Chiplet_2",
                "to": "Chiplet_3",
                "connection_type": "2.5d_emib",
                "loc": "2.5d_chiplet",
            }
        )
        return architecture

    def _sa_only_architecture(self):
        architecture = copy.deepcopy(self.architecture)
        del architecture["Chiplet_2"]
        architecture["pkg"] = {
            "HI_pkg_type": "2d",
            "inter_pkg_conn": "2d_na",
            "protocol_3d": "na",
            "protocol_2.5d": "na",
            "mem_pkg_conn": {"mem_type": "ddr5", "Chiplet_1": 8},
        }
        return architecture

    def test_sa_value_move_supports_a_multi_sa_architecture(self):
        architecture = _load(ARCHITECTURE)
        architecture["Chiplet_3"] = dict(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"]["Chiplet_3"] = 4
        architecture["pkg"]["inter_pkg_conn"].append(
            {
                "from": "Chiplet_2",
                "to": "Chiplet_3",
                "connection_type": "2.5d_emib",
                "loc": "2.5d_chiplet",
            }
        )

        moved, _, move = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            self.search_space,
            rng=_MoveRng("sa_array"),
        )

        self.assertEqual(move, "sa_array")
        self.assertEqual(
            validate_atlas_architecture(moved).sa_keys,
            ("Chiplet_1", "Chiplet_3"),
        )

    def test_sa_count_move_adds_an_sa_and_rebuilds_a_valid_package(self):
        params = copy.deepcopy(self.search_space)
        params["max_chiplet"] = 6
        params["pkg"]["inter_pkg_architecture"] = ["2.5d_emib"]
        original_fpga = copy.deepcopy(self.architecture["Chiplet_2"])

        moved, mapping, move = mutate_atlas_design_point(
            self.architecture,
            self._mapping(self.architecture),
            params,
            rng=_MoveRng("sa_count"),
        )

        endpoints = validate_atlas_architecture(moved)
        self.assertEqual(move, "sa_count")
        self.assertEqual(len(endpoints.sa_keys), 2)
        self.assertEqual(len(endpoints.fpga_keys), 1)
        self.assertEqual(moved[endpoints.fpga_keys[0]], original_fpga)
        for operation in mapping.operations:
            self.assertEqual(
                operation.endpoint_ids,
                endpoints.sa_ids if operation.is_gemm else endpoints.fpga_ids,
            )
        self.assertEqual(
            set(moved["pkg"]["mem_pkg_conn"]) - {"mem_type"},
            set(endpoints.sa_keys) | set(endpoints.fpga_keys),
        )
        self.assertEqual(len(ChipletSystem(arch_dict=moved).core_dict), 2)

    def test_sa_count_move_deletes_only_an_sa(self):
        architecture = _load(ARCHITECTURE)
        architecture["Chiplet_3"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"]["Chiplet_3"] = 4
        architecture["pkg"]["inter_pkg_conn"].append(
            {
                "from": "Chiplet_2",
                "to": "Chiplet_3",
                "connection_type": "2.5d_emib",
                "loc": "2.5d_chiplet",
            }
        )
        params = copy.deepcopy(self.search_space)
        params["max_sa_chiplets"] = 2
        params["pkg"]["inter_pkg_architecture"] = ["2.5d_emib"]
        original_fpga = copy.deepcopy(architecture["Chiplet_2"])

        moved, _, move = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("sa_count"),
        )

        endpoints = validate_atlas_architecture(moved)
        self.assertEqual(move, "sa_count")
        self.assertEqual(len(endpoints.sa_keys), 1)
        self.assertEqual(len(endpoints.fpga_keys), 1)
        self.assertEqual(moved[endpoints.fpga_keys[0]], original_fpga)

    def test_interconnect_move_rebuilds_links_and_compatible_protocol(self):
        params = copy.deepcopy(self.search_space)
        params["pkg"]["inter_pkg_architecture"] = ["2.5d_emib", "2.5d_rdl"]

        moved, _, move = mutate_atlas_design_point(
            self.architecture,
            self._mapping(self.architecture),
            params,
            rng=_MoveRng("interconnect"),
        )

        self.assertEqual(move, "interconnect")
        self.assertEqual(moved["pkg"]["HI_pkg_type"], "2.5d")
        self.assertEqual(
            {connection["connection_type"] for connection in moved["pkg"]["inter_pkg_conn"]},
            {"2.5d_rdl"},
        )
        self.assertEqual(moved["pkg"]["protocol_2.5d"], "ucie_std")
        ChipletSystem(arch_dict=moved)

    def test_sa_only_interconnect_move_can_build_a_3d_stack(self):
        architecture = _load(ARCHITECTURE)
        architecture["Chiplet_2"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"]["Chiplet_2"] = 4
        params = copy.deepcopy(self.search_space)
        params["pkg"]["inter_pkg_architecture"] = [
            "2.5d_emib",
            "3d_hyb_bond",
        ]
        params["pkg"]["protocol"] = ["ucie_adv", "ucie_3d"]

        moved, _, move = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("interconnect"),
        )

        self.assertEqual(move, "interconnect")
        self.assertEqual(moved["pkg"]["HI_pkg_type"], "3d")
        self.assertEqual(moved["pkg"]["protocol_3d"], "ucie_3d")
        self.assertEqual(moved["pkg"]["protocol_2.5d"], "na")
        ChipletSystem(arch_dict=moved)

    def test_sa_only_interconnect_move_can_build_a_hybrid_package(self):
        architecture = _load(ARCHITECTURE)
        del architecture["Chiplet_2"]
        architecture["Chiplet_2"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["Chiplet_3"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["pkg"] = {
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
                    "to": "Chiplet_3",
                    "connection_type": "2.5d_emib",
                    "loc": "2.5d_chiplet",
                },
            ],
            "protocol_3d": "na",
            "protocol_2.5d": "ucie_adv",
            "mem_pkg_conn": {
                "mem_type": "hbm2",
                "Chiplet_1": 6,
                "Chiplet_2": 5,
                "Chiplet_3": 5,
            },
        }
        params = copy.deepcopy(self.search_space)
        params["pkg"]["package_types"] = ["2.5d", "3d", "2.5d_3d"]
        params["pkg"]["inter_pkg_architecture"] = [
            "2.5d_emib",
            "3d_hyb_bond",
        ]
        params["pkg"]["protocol"] = ["ucie_adv", "ucie_3d"]

        moved, _, move = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("interconnect", preferred="2.5d_3d"),
        )

        self.assertEqual(move, "interconnect")
        self.assertEqual(moved["pkg"]["HI_pkg_type"], "2.5d_3d")
        locations = {connection["loc"] for connection in moved["pkg"]["inter_pkg_conn"]}
        self.assertIn("stack0_base", locations)
        self.assertIn("stack0_top", locations)
        self.assertIn("2.5d_chiplet", locations)
        self.assertEqual(moved["pkg"]["protocol_3d"], "ucie_3d")
        self.assertEqual(moved["pkg"]["protocol_2.5d"], "ucie_adv")
        self.assertEqual(len(ChipletSystem(arch_dict=moved).core_dict), 3)

    def test_repeated_sa_count_moves_keep_endpoint_labels_compact(self):
        params = copy.deepcopy(self.search_space)
        params["max_sa_chiplets"] = 2
        params["move_weights"] = {
            move_name: 1.0 if move_name == "sa_count" else 0.0
            for move_name in MOVE_TYPES
        }
        architecture = self.architecture

        for _ in range(20):
            architecture, _, _ = mutate_atlas_design_point(
                architecture,
                self._mapping(architecture),
                params,
                rng=_MoveRng("sa_count", preferred="add"),
            )
            architecture, _, _ = mutate_atlas_design_point(
                architecture,
                self._mapping(architecture),
                params,
                rng=_MoveRng("sa_count", preferred="delete"),
            )

            endpoints = validate_atlas_architecture(architecture)
            endpoint_keys = endpoints.sa_keys + endpoints.fpga_keys
            self.assertEqual(
                {int(key.split("_", 1)[1]) for key in endpoint_keys},
                set(range(1, len(endpoint_keys) + 1)),
            )

    def test_protocol_move_only_selects_a_link_compatible_protocol(self):
        moved, _, move = mutate_atlas_design_point(
            self.architecture,
            self._mapping(self.architecture),
            self.search_space,
            rng=_MoveRng("protocol"),
        )

        self.assertEqual(move, "protocol")
        self.assertIn(moved["pkg"]["protocol_2.5d"], ("ucie_adv", "aib", "bow"))
        ChipletSystem(arch_dict=moved)

    def test_sequential_one_sa_memory_move_stays_on_ddr(self):
        architecture = copy.deepcopy(self.architecture)
        del architecture["Chiplet_2"]
        architecture["pkg"] = {
            "HI_pkg_type": "2d",
            "inter_pkg_conn": "2d_na",
            "protocol_3d": "na",
            "protocol_2.5d": "na",
            "mem_pkg_conn": {"mem_type": "ddr4", "Chiplet_1": 8},
        }
        params = sequential_gemm_search_space(_load("cfg/parameters/input.json"))
        params["move_weights"] = {"mem_type": 1.0}

        moved, _, move = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("mem_type", preferred="hbm3"),
        )

        self.assertEqual(move, "mem_type")
        self.assertEqual(moved["pkg"]["mem_pkg_conn"]["mem_type"], "ddr5")

    def test_sequential_memory_move_regenerates_hbm_channel_allocation(self):
        architecture = copy.deepcopy(self.architecture)
        architecture["Chiplet_2"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"] = {
            "mem_type": "ddr5",
            "Chiplet_1": 4,
            "Chiplet_2": 4,
        }
        params = sequential_gemm_search_space(_load("cfg/parameters/input.json"))
        params["move_weights"] = {"mem_type": 1.0}

        moved, _, move = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("mem_type", preferred="hbm3"),
        )

        self.assertEqual(move, "mem_type")
        self.assertEqual(
            moved["pkg"]["mem_pkg_conn"],
            {"mem_type": "hbm3", "Chiplet_1": 8, "Chiplet_2": 8},
        )

    def test_memory_move_preserves_legacy_area_weighted_channel_distribution(self):
        architecture = copy.deepcopy(self.architecture)
        architecture["Chiplet_2"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["Chiplet_1"]["area"] = 9.0
        architecture["Chiplet_2"]["area"] = 1.0
        architecture["pkg"]["mem_pkg_conn"] = {
            "mem_type": "ddr5",
            "Chiplet_1": 7,
            "Chiplet_2": 1,
        }
        params = sequential_gemm_search_space(_load("cfg/parameters/input.json"))
        params["move_weights"] = {"mem_type": 1.0}

        moved, _, _ = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("mem_type", preferred="hbm3"),
        )

        self.assertEqual(
            moved["pkg"]["mem_pkg_conn"],
            {"mem_type": "hbm3", "Chiplet_1": 14, "Chiplet_2": 2},
        )

    def test_sequential_memory_move_regenerates_ddr_channel_allocation(self):
        architecture = copy.deepcopy(self.architecture)
        architecture["Chiplet_2"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"] = {
            "mem_type": "hbm3",
            "Chiplet_1": 8,
            "Chiplet_2": 8,
        }
        params = sequential_gemm_search_space(_load("cfg/parameters/input.json"))
        params["move_weights"] = {"mem_type": 1.0}

        moved, _, _ = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("mem_type", preferred="ddr5"),
        )

        self.assertEqual(
            moved["pkg"]["mem_pkg_conn"],
            {"mem_type": "ddr5", "Chiplet_1": 4, "Chiplet_2": 4},
        )

    def test_sequential_interconnect_move_preserves_package_topology(self):
        architecture = copy.deepcopy(self.architecture)
        architecture["Chiplet_2"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"]["Chiplet_2"] = 4
        params = sequential_gemm_search_space(_load("cfg/parameters/input.json"))
        params["move_weights"] = {"interconnect": 1.0}

        moved, _, move = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("interconnect", preferred="3d_tsv"),
        )

        self.assertEqual(move, "interconnect")
        self.assertEqual(moved["pkg"]["HI_pkg_type"], "2.5d")
        self.assertEqual(
            {link["connection_type"] for link in moved["pkg"]["inter_pkg_conn"]},
            {"2.5d_rdl"},
        )

    def test_sequential_interconnect_move_can_regenerate_each_valid_protocol(self):
        architecture = copy.deepcopy(self.architecture)
        architecture["Chiplet_2"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"]["Chiplet_2"] = 4
        for link in architecture["pkg"]["inter_pkg_conn"]:
            link["connection_type"] = "2.5d_rdl"
        architecture["pkg"]["protocol_2.5d"] = "ucie_std"
        params = sequential_gemm_search_space(_load("cfg/parameters/input.json"))
        params["move_weights"] = {"interconnect": 1.0}

        moved, _, _ = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("interconnect", preferred="aib"),
        )

        self.assertEqual(moved["pkg"]["protocol_2.5d"], "aib")

    def test_sequential_sa_count_move_regenerates_single_sa_memory_and_package(self):
        architecture = copy.deepcopy(self.architecture)
        architecture["Chiplet_2"] = copy.deepcopy(architecture["Chiplet_1"])
        architecture["pkg"]["mem_pkg_conn"].update(
            {"mem_type": "hbm2", "Chiplet_2": 4}
        )
        params = sequential_gemm_search_space(_load("cfg/parameters/input.json"))
        params["max_sa_chiplets"] = 2
        params["move_weights"] = {"sa_count": 1.0}

        moved, _, move = mutate_atlas_design_point(
            architecture,
            self._mapping(architecture),
            params,
            rng=_MoveRng("sa_count", preferred="delete"),
        )

        self.assertEqual(move, "sa_count")
        self.assertEqual(len(validate_atlas_architecture(moved).sa_keys), 1)
        self.assertEqual(moved["pkg"]["HI_pkg_type"], "2d")
        self.assertIn(moved["pkg"]["mem_pkg_conn"]["mem_type"], ("ddr4", "ddr5"))


if __name__ == "__main__":
    unittest.main()
