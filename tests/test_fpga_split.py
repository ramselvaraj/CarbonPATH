import copy
import json
import random
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from main import evaluate_atlas_graph
from system.utils.ChipletSystem import ChipletSystem
from system.utils.AtlasAnnealingMoves import (
    mutate_atlas_design_point,
    validate_atlas_architecture,
)
from system.utils.AtlasGraphAdapter import load_atlas_graph, parse_atlas_graph
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.FpgaActivationMovement import FpgaActivationMovementService
from system.utils.FpgaChiplet import FpgaChiplet, FpgaReluImplementation
from system.utils.FpgaLayerNorm import (
    LayerNormEvaluator,
    LayerNormInputAdapter,
    register_layernorm_evaluators,
)
from system.utils.FpgaSplit import fpga_row_ranges
from system.utils.OperationAssignment import EndpointAssignment
from system.utils.OperationEvaluator import EvaluationContext, EvaluatorRegistry
from system.utils.ParameterLoad import ParameterLoad, combine_parameter_loads
from system.utils.SimulationCache import SimulationCache
from system.utils.TensorMovement import MovementRequest
from system.utils.UnsupportedEvaluation import (
    ResourceExceeded,
    UnsupportedEvaluation,
)
from system.utils.WorkloadMapping import initial_workload_mapping

FREQUENCY_HZ = 300000000
GRAPH = Path("cfg/examples/atlas/dense_layernorm.graph_dump.json")
PROFILE = Path("cfg/profiles/atlas_multi_fpga_layernorm_v1.json")
ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space_placeholders.json")
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


def _load(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _fpga(fpga_id, brams=200):
    return FpgaChiplet(
        id=fpga_id,
        name=f"Chiplet_{fpga_id + 1}",
        frequency_hz=FREQUENCY_HZ,
        clbs=10000,
        brams=brams,
        dsps=500,
        area=10.0,
        power=2.0,
        node=7,
        relu_implementation=FpgaReluImplementation(clbs_per_lane=5),
        parameter_source="dram",
    )


def _layernorm(shape):
    shape = list(shape)
    nodes = [
        {
            "name": "input_layer",
            "class": "Input",
            "inputs": [],
            "output": {"name": "input_layer", "shape": shape},
        },
        {
            "name": "ln_1",
            "class": "LayerNormalization",
            "inputs": [{"name": "input_layer", "shape": shape}],
            "output": {"name": "ln_1", "shape": shape},
            "weights": [
                {"name": "scale", "shape": [shape[-1]]},
                {"name": "bias", "shape": [shape[-1]]},
            ],
            "attrs": {},
        },
    ]
    return parse_atlas_graph(nodes, "graph").operations[0]


STUB_ARCHITECTURE = {
    "pkg": {
        "mem_pkg_conn": {"mem_type": "hbm2", "Chiplet_1": 4, "Chiplet_2": 0, "Chiplet_3": 0}
    },
    "transfer_model": {"setup_latency_ns": 10.0, "hop_latency_ns": 2.0},
}


def _stub_context(fpgas, slack_ns=0.0):
    core = SimpleNamespace(id=0, dram_bandwidth=64, dram_energy_scale=4.0)
    system = SimpleNamespace(
        core_dict={0: core},
        fpga_chiplet_dict={fpga.id: fpga for fpga in fpgas},
        get_shortest_path=lambda source, destination: (
            [source, destination],
            0.5,
            1.0,
        ),
    )
    return EvaluationContext(
        cache=None,
        architecture=STUB_ARCHITECTURE,
        system=system,
        evaluator_settings={},
        prefetch_slack_ns=slack_ns,
    )


def _binding():
    registry = EvaluatorRegistry()
    register_layernorm_evaluators(registry)
    return registry.get("fpga_layernorm_v1")


class RowRangeTests(unittest.TestCase):
    def test_rows_divide_in_blocks_that_differ_by_one_row(self):
        self.assertEqual(
            fpga_row_ranges("op", 8, (3, 1, 2)),
            {1: (0, 3), 2: (3, 6), 3: (6, 8)},
        )

    def test_one_chiplet_gets_every_row(self):
        self.assertEqual(fpga_row_ranges("op", 8, (4,)), {4: (0, 8)})

    def test_every_chiplet_needs_a_row(self):
        with self.assertRaisesRegex(UnsupportedEvaluation, "at least one row"):
            fpga_row_ranges("op", 1, (1, 2))

    def test_blocks_cover_the_rows_once(self):
        for rows in range(2, 40):
            for count in range(1, min(rows, 6) + 1):
                ranges = fpga_row_ranges("op", rows, tuple(range(count)))
                covered = [
                    row for first, end in ranges.values() for row in range(first, end)
                ]
                self.assertEqual(covered, list(range(rows)))


class RowSliceTests(unittest.TestCase):
    def test_a_row_slice_changes_the_tensors_and_keeps_the_weights(self):
        operation = _layernorm((8, 64))
        part = operation.with_rows(3)
        self.assertEqual(part.input_tensor.shape, (3, 64))
        self.assertEqual(part.output_tensor.shape, (3, 64))
        self.assertEqual(part.element_count, 3 * 64)
        self.assertEqual(part.source.input_tensor(0).shape, (3, 64))
        self.assertEqual(part.source.weight(0).shape, (64,))
        # The operation itself does not change.
        self.assertEqual(operation.output_tensor.shape, (8, 64))

    def test_a_rank_one_tensor_cannot_split(self):
        with self.assertRaises(UnsupportedEvaluation):
            _layernorm((64,)).with_rows(1)


class SplitEstimateTests(unittest.TestCase):
    def _single(self, shape, fpga=None, slack_ns=0.0):
        fpga = fpga or _fpga(1)
        view = LayerNormInputAdapter().build_input(_layernorm(shape), None)
        return LayerNormEvaluator().evaluate(
            view, EndpointAssignment(1, "fpga"), _stub_context([fpga], slack_ns)
        )

    def _split(self, shape, fpgas, slack_ns=0.0):
        return _binding().estimate(
            _layernorm(shape),
            None,
            EndpointAssignment(tuple(fpga.id for fpga in fpgas), "fpga"),
            _stub_context(fpgas, slack_ns),
        )

    def test_latency_is_that_of_the_slowest_chiplet(self):
        estimate = self._split((8, 64), [_fpga(1), _fpga(2)])
        half = self._single((4, 64))
        self.assertAlmostEqual(estimate.compute_latency_ns, half.compute_latency_ns)

    def test_uneven_rows_are_limited_by_the_larger_block(self):
        estimate = self._split((9, 64), [_fpga(1), _fpga(2)])
        larger = self._single((5, 64))
        self.assertAlmostEqual(estimate.compute_latency_ns, larger.compute_latency_ns)

    def test_a_split_is_faster_than_one_chiplet(self):
        whole = self._single((16, 64))
        split = self._split((16, 64), [_fpga(1), _fpga(2)])
        self.assertLess(split.compute_latency_ns, whole.compute_latency_ns)

    def test_dynamic_energy_is_the_same_as_for_one_chiplet(self):
        whole = self._single((8, 64))
        split = self._split((8, 64), [_fpga(1), _fpga(2)])
        self.assertAlmostEqual(split.dynamic_energy_pj, whole.dynamic_energy_pj)

    def test_each_chiplet_loads_its_own_copy_of_the_constants(self):
        whole = self._single((8, 64)).parameter_load
        split = self._split((8, 64), [_fpga(1), _fpga(2)]).parameter_load
        self.assertEqual(split.byte_count, 2 * whole.byte_count)
        self.assertAlmostEqual(split.energy_pj, 2 * whole.energy_pj)
        # The loads share the DRAM, so they run one after the other.
        self.assertAlmostEqual(split.raw_latency_ns, 2 * whole.raw_latency_ns)
        self.assertEqual(split.destination_endpoint, (1, 2))
        self.assertEqual(split.source_endpoint, 0)

    def test_the_idle_time_hides_the_summed_load_once(self):
        slack = 500.0
        split = self._split((8, 64), [_fpga(1), _fpga(2)], slack_ns=slack).parameter_load
        self.assertAlmostEqual(
            split.exposed_latency_ns, max(0.0, split.raw_latency_ns - slack)
        )
        self.assertGreater(split.exposed_latency_ns, 0.0)

    def test_every_chiplet_has_its_own_resource_check(self):
        # The input buffer (1 block) and the constants (1 block) need 2 BRAM.
        self._split((8, 64), [_fpga(1, brams=2), _fpga(2, brams=2)])
        with self.assertRaises(ResourceExceeded):
            self._split((8, 64), [_fpga(1, brams=2), _fpga(2, brams=1)])

    def test_the_rows_must_cover_every_chiplet(self):
        with self.assertRaisesRegex(UnsupportedEvaluation, "at least one row"):
            self._split((1, 64), [_fpga(1), _fpga(2)])

    def test_parameter_loads_combine_by_summing(self):
        loads = [
            ParameterLoad(10, 0, 1, 5.0, 1.0, 3.0),
            ParameterLoad(10, 0, 2, 7.0, 2.0, 4.0),
        ]
        combined = combine_parameter_loads(loads, slack_ns=4.0)
        self.assertEqual(combined.byte_count, 20)
        self.assertEqual(combined.raw_latency_ns, 12.0)
        self.assertEqual(combined.exposed_latency_ns, 8.0)
        self.assertEqual(combined.energy_pj, 7.0)


@dataclass
class _Workload:
    m: int
    k: int
    n: int
    m_offset: int = 0
    k_offset: int = 0
    n_offset: int = 0


class _Core:
    def __init__(self, core_id, workloads):
        self.id = core_id
        self.workloads = list(workloads)

    def __lt__(self, other):
        return self.id < other.id


class _System:
    """Every pair of endpoints has a link of 2 ns per byte and 1 pJ per bit."""

    def link_cost(self, source, destination):
        return 2.0, 1.0

    def get_shortest_path(self, source, destination):
        if source == destination:
            return None, 0, 0
        return [source, destination], 2.0, 1.0


def _request(producer, consumer, shape, producer_ids, consumer_ids, **prepared):
    def assignment(ids, kind):
        return SimpleNamespace(endpoint_ids=tuple(ids), endpoint_kind=kind)

    kinds = {"gemm": "sa"}
    return MovementRequest(
        handoff_index=1,
        tensor_id="t",
        tensor_shape=shape,
        producer_operation=SimpleNamespace(operation_id="p", operation_type=producer),
        consumer_operation=SimpleNamespace(operation_id="c", operation_type=consumer),
        producer_assignment=assignment(producer_ids, kinds.get(producer, "fpga")),
        consumer_assignment=assignment(consumer_ids, kinds.get(consumer, "fpga")),
        producer_prepared=prepared.get("producer_prepared"),
        consumer_prepared=prepared.get("consumer_prepared"),
        system=_System(),
        transfer_model={"setup_latency_ns": 0.0, "hop_latency_ns": 0.0},
    )


def _scheduler(*cores):
    return SimpleNamespace(systolic_arrays=list(cores), splitting_k=False)


class SplitMovementTests(unittest.TestCase):
    def setUp(self):
        self.service = FpgaActivationMovementService()

    def _slices(self, plan):
        return sorted(
            (
                s.source_endpoint,
                s.destination_endpoint,
                s.row_offset,
                s.row_count,
                s.feature_offset,
                s.feature_count,
            )
            for s in plan.transfer_slices
        )

    def test_gemm_rows_go_to_the_chiplet_that_owns_them(self):
        scheduler = _scheduler(_Core(0, [_Workload(m=8, k=4, n=16)]))
        plan = self.service.plan(
            _request(
                "gemm", "layernorm", (8, 16), [0], [1, 2],
                producer_prepared=(scheduler, None),
            )
        )
        self.assertEqual(
            self._slices(plan), [(0, 1, 0, 4, 0, 16), (0, 2, 4, 4, 0, 16)]
        )
        self.assertEqual(plan.logical_bytes, 128)
        self.assertEqual(plan.physical_bytes, 128)
        self.assertEqual(plan.destination_endpoint, (1, 2))

    def test_a_gemm_region_that_crosses_two_blocks_splits(self):
        # Two SAs hold rows 0-5 and 5-8; the chiplets own rows 0-4 and 4-8.
        scheduler = _scheduler(
            _Core(0, [_Workload(m=5, k=4, n=16)]),
            _Core(3, [_Workload(m=3, k=4, n=16, m_offset=5)]),
        )
        plan = self.service.plan(
            _request(
                "gemm", "layernorm", (8, 16), [0, 3], [1, 2],
                producer_prepared=(scheduler, None),
            )
        )
        self.assertEqual(
            self._slices(plan),
            [
                (0, 1, 0, 4, 0, 16),
                (0, 2, 4, 1, 0, 16),
                (3, 2, 5, 3, 0, 16),
            ],
        )
        self.assertEqual(plan.physical_bytes, plan.logical_bytes)

    def test_the_latency_is_the_slowest_slice_and_the_energy_is_the_sum(self):
        scheduler = _scheduler(_Core(0, [_Workload(m=8, k=4, n=16)]))
        plan = self.service.plan(
            _request(
                "gemm", "layernorm", (8, 16), [0], [1, 2],
                producer_prepared=(scheduler, None),
            )
        )
        # Slice 0->1 and 0->2 use different links, 64 bytes each.
        self.assertAlmostEqual(plan.charged_latency_ns, 64 * 2.0)
        self.assertAlmostEqual(plan.charged_energy_pj, 2 * 64 * 8 * 1.0)

    def test_rows_return_to_the_sa_from_the_chiplets_that_hold_them(self):
        scheduler = _scheduler(_Core(0, [_Workload(m=8, k=16, n=4)]))
        plan = self.service.plan(
            _request(
                "layernorm", "gemm", (8, 16), [1, 2], [0],
                consumer_prepared=(scheduler, None),
            )
        )
        self.assertEqual(
            self._slices(plan), [(1, 0, 0, 4, 0, 16), (2, 0, 4, 4, 0, 16)]
        )
        self.assertEqual(plan.source_endpoint, (1, 2))

    def test_the_same_set_of_chiplets_keeps_the_tensor(self):
        plan = self.service.plan(
            _request("layernorm", "relu", (8, 16), [1, 2], [1, 2])
        )
        self.assertEqual(plan.method, "retain")
        self.assertEqual(plan.charged_latency_ns, 0.0)

    def test_a_different_set_moves_only_the_rows_that_change_chiplet(self):
        plan = self.service.plan(
            _request("layernorm", "relu", (8, 16), [1], [1, 2])
        )
        # Chiplet 1 keeps rows 0-4; chiplet 2 reads rows 4-8 from chiplet 1.
        self.assertEqual(
            self._slices(plan), [(1, 1, 0, 4, 0, 16), (1, 2, 4, 4, 0, 16)]
        )
        self.assertEqual(plan.charged_latency_ns, 64 * 2.0)
        self.assertEqual(plan.charged_energy_pj, 64 * 8 * 1.0)

    def test_moving_to_one_chiplet_gathers_the_rows(self):
        plan = self.service.plan(
            _request("layernorm", "relu", (8, 16), [1, 2], [2])
        )
        self.assertEqual(
            self._slices(plan), [(1, 2, 0, 4, 0, 16), (2, 2, 4, 4, 0, 16)]
        )


def _two_fpga_architecture():
    architecture = _load(ARCHITECTURE)
    architecture["Chiplet_3"] = copy.deepcopy(architecture["Chiplet_2"])
    architecture["pkg"]["mem_pkg_conn"]["Chiplet_3"] = 0
    for source, destination in (("Chiplet_1", "Chiplet_3"), ("Chiplet_3", "Chiplet_1")):
        architecture["pkg"]["inter_pkg_conn"].append(
            {
                "from": source,
                "to": destination,
                "connection_type": "2.5d_emib",
                "loc": "2.5d_chiplet",
            }
        )
    return architecture


class _MoveRng:
    """Picks the named move first, then the first value of every other choice."""

    def __init__(self, move_name):
        self.move_name = move_name
        self.selected_move = False

    def choice(self, values):
        if not self.selected_move and self.move_name in values:
            self.selected_move = True
            return self.move_name
        return values[0]

    def choices(self, values, weights, k):
        del weights
        return [self.choice(values) for _ in range(k)]


class SplitMoveTests(unittest.TestCase):
    def setUp(self):
        self.search_space = _load(SEARCH_SPACE)
        self.search_space["pkg"]["inter_pkg_architecture"] = ["2.5d_emib"]
        self.graph = load_atlas_graph(GRAPH)
        self.profile = load_evaluation_profile(PROFILE)

    def _mapping(self, architecture):
        return initial_workload_mapping(
            self.graph, architecture, self.profile, self.search_space
        )

    def _layernorm_index(self, mapping):
        return next(
            index
            for index, operation in enumerate(mapping.operations)
            if operation.operation_type == "layernorm"
        )

    def test_the_mapping_starts_on_every_fpga_chiplet(self):
        architecture = _two_fpga_architecture()
        mapping = self._mapping(architecture)
        endpoints = validate_atlas_architecture(architecture)
        index = self._layernorm_index(mapping)
        self.assertEqual(mapping.operations[index].endpoint_ids, endpoints.fpga_ids)

    def test_fpga_set_removes_a_chiplet_from_the_operation(self):
        architecture = _two_fpga_architecture()
        mapping = self._mapping(architecture)
        params = {**self.search_space, "move_weights": {"fpga_set": 1.0}}

        moved, moved_mapping, move = mutate_atlas_design_point(
            architecture, mapping, params, rng=_MoveRng("fpga_set")
        )

        self.assertEqual(move, "fpga_set")
        self.assertEqual(moved, architecture)
        index = self._layernorm_index(mapping)
        self.assertEqual(len(moved_mapping.operations[index].endpoint_ids), 1)
        for other, operation in enumerate(moved_mapping.operations):
            if other != index:
                self.assertEqual(operation, mapping.operations[other])

        again, final, _ = mutate_atlas_design_point(
            moved, moved_mapping, params, rng=_MoveRng("fpga_set")
        )
        self.assertEqual(len(final.operations[index].endpoint_ids), 2)

    def test_fpga_set_is_not_a_candidate_with_one_chiplet(self):
        architecture = _load(ARCHITECTURE)
        params = {**self.search_space, "move_weights": {"fpga_set": 1.0}}
        with self.assertRaisesRegex(UnsupportedEvaluation, "enabled move"):
            mutate_atlas_design_point(
                architecture,
                self._mapping(architecture),
                params,
                rng=random.Random(1),
            )

    def test_fpga_count_adds_a_chiplet_that_joins_no_operation(self):
        architecture = _load(ARCHITECTURE)
        mapping = self._mapping(architecture)
        params = {**self.search_space, "move_weights": {"fpga_count": 1.0}}
        params["fpga"] = {**params["fpga"], "count": [1, 2]}
        original_fpga = copy.deepcopy(architecture["Chiplet_2"])

        moved, moved_mapping, move = mutate_atlas_design_point(
            architecture, mapping, params, rng=_MoveRng("fpga_count")
        )

        endpoints = validate_atlas_architecture(moved)
        self.assertEqual(move, "fpga_count")
        self.assertEqual(len(endpoints.fpga_keys), 2)
        self.assertEqual(len(endpoints.sa_keys), 1)
        self.assertEqual(moved[endpoints.fpga_keys[0]], original_fpga)
        self.assertEqual(moved[endpoints.fpga_keys[1]], original_fpga)
        self.assertEqual(
            moved["pkg"]["mem_pkg_conn"][endpoints.fpga_keys[1]], 0
        )
        # The operations keep their chiplets; the new one is idle until fpga_set.
        self.assertEqual(moved_mapping, mapping)
        new_id = endpoints.fpga_ids[1]
        for operation in moved_mapping.operations:
            self.assertNotIn(new_id, operation.endpoint_ids)
        self.assertEqual(len(ChipletSystem(arch_dict=moved).fpga_chiplet_dict), 2)

    def test_fpga_count_removes_a_chiplet_from_its_operations(self):
        architecture = _two_fpga_architecture()
        mapping = self._mapping(architecture)
        params = {**self.search_space, "move_weights": {"fpga_count": 1.0}}
        params["fpga"] = {**params["fpga"], "count": [1, 2]}

        moved, moved_mapping, _ = mutate_atlas_design_point(
            architecture, mapping, params, rng=_MoveRng("fpga_count")
        )

        endpoints = validate_atlas_architecture(moved)
        self.assertEqual(len(endpoints.fpga_keys), 1)
        index = self._layernorm_index(moved_mapping)
        self.assertEqual(moved_mapping.operations[index].endpoint_ids, endpoints.fpga_ids)

    def test_the_last_chiplet_of_an_operation_cannot_go(self):
        architecture = _load(ARCHITECTURE)
        mapping = self._mapping(architecture)
        params = {**self.search_space, "move_weights": {"fpga_count": 1.0}}
        params["fpga"] = {**params["fpga"], "count": [0, 1]}

        with self.assertRaisesRegex(UnsupportedEvaluation, "no endpoint"):
            mutate_atlas_design_point(
                architecture, mapping, params, rng=_MoveRng("fpga_count")
            )

    def test_fpga_count_builds_a_chiplet_from_the_search_space(self):
        architecture = _load(ARCHITECTURE)
        del architecture["Chiplet_2"]
        architecture["pkg"] = {
            "HI_pkg_type": "2d",
            "inter_pkg_conn": "2d_na",
            "protocol_3d": "na",
            "protocol_2.5d": "na",
            "mem_pkg_conn": {"mem_type": "ddr5", "Chiplet_1": 8},
        }
        graph = SimpleNamespace(
            operations=(
                SimpleNamespace(operation_id="g", operation_type="gemm"),
            )
        )
        mapping = initial_workload_mapping(
            graph, architecture, self.profile, self.search_space
        )
        params = {**self.search_space, "move_weights": {"fpga_count": 1.0}}
        params["fpga"] = {**params["fpga"], "count": [0, 1]}

        moved, _, _ = mutate_atlas_design_point(
            architecture, mapping, params, rng=_MoveRng("fpga_count")
        )

        endpoints = validate_atlas_architecture(moved)
        self.assertEqual(len(endpoints.fpga_keys), 1)
        self.assertEqual(moved[endpoints.fpga_keys[0]]["chiplet_type"], "fpga")
        self.assertIn(moved[endpoints.fpga_keys[0]]["board"], params["fpga"]["boards"])
        self.assertEqual(len(ChipletSystem(arch_dict=moved).fpga_chiplet_dict), 1)

    def test_a_bad_count_list_is_unsupported(self):
        architecture = _load(ARCHITECTURE)
        params = {**self.search_space, "move_weights": {"fpga_count": 1.0}}
        params["fpga"] = {**params["fpga"], "count": [1, -2]}
        with self.assertRaisesRegex(UnsupportedEvaluation, "fpga.count"):
            mutate_atlas_design_point(
                architecture,
                self._mapping(architecture),
                params,
                rng=random.Random(1),
            )


class SplitGraphEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.graph = load_atlas_graph(GRAPH)
        self.profile = load_evaluation_profile(PROFILE)
        self.search_space = _load(SEARCH_SPACE)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache_path = Path(directory.name) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        self.cache = SimulationCache(cache_path, simulator_dir=directory.name)

    def _evaluate(self, architecture, fpga_ids=None):
        architecture["Chiplet_2"]["parameter_source"] = "dram"
        mapping = initial_workload_mapping(
            self.graph, architecture, self.profile, self.search_space
        )
        if fpga_ids is not None:
            index = next(
                i
                for i, operation in enumerate(mapping.operations)
                if operation.operation_type == "layernorm"
            )
            mapping = mapping.with_operation(index, endpoint_ids=tuple(fpga_ids))
        return evaluate_atlas_graph(
            self.cache, architecture, self.graph, profile=self.profile, mapping=mapping
        )

    def test_a_layernorm_on_two_chiplets_runs_and_is_faster(self):
        architecture = _two_fpga_architecture()
        architecture["Chiplet_3"]["parameter_source"] = "dram"
        single = self._evaluate(copy.deepcopy(architecture), fpga_ids=(1,))
        split = self._evaluate(copy.deepcopy(architecture), fpga_ids=(1, 2))

        single_ln = single.results[1]
        split_ln = split.results[1]
        self.assertEqual(split_ln.endpoint_ids, (1, 2))
        self.assertLess(split_ln.compute_latency_ns, single_ln.compute_latency_ns)
        self.assertEqual(len(split_ln.movement.transfer_slices), 2)
        self.assertEqual(
            split_ln.parameter_load.byte_count, 2 * single_ln.parameter_load.byte_count
        )
        self.assertAlmostEqual(
            split_ln.parameter_load.energy_pj, 2 * single_ln.parameter_load.energy_pj
        )

    def test_the_movement_energy_of_the_split_is_charged(self):
        architecture = _two_fpga_architecture()
        split = self._evaluate(architecture, fpga_ids=(1, 2))
        movement = split.results[1].movement
        self.assertEqual(movement.physical_bytes, movement.logical_bytes)
        self.assertGreater(movement.charged_energy_pj, 0.0)


if __name__ == "__main__":
    unittest.main()
