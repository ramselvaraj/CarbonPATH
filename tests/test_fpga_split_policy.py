import copy
import random
import unittest
from dataclasses import replace
from types import SimpleNamespace

from system.utils.EvaluationProfile import (
    load_evaluation_profile,
    parse_evaluation_profile,
)
from system.utils.FpgaActivationMovement import FpgaActivationMovementService
from system.utils.FpgaNewChiplet import new_fpga_chiplet
from system.utils.FpgaSplit import fpga_row_ranges
from system.utils.FpgaSplitPolicy import (
    DEFAULT_SPLIT_POLICY,
    RowSplitPolicy,
    build_split_policy,
)
from system.utils.ParameterLoad import ParameterLoad, combine_parameter_loads
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from tests.test_fpga_split import (
    ARCHITECTURE,
    SEARCH_SPACE,
    SplitEstimateTests,
    _binding,
    _fpga,
    _load,
    _request,
    _scheduler,
    _Core,
    _Workload,
    _stub_context,
    _layernorm,
)
from system.utils.OperationAssignment import EndpointAssignment

PROFILE_FILE = "cfg/profiles/atlas_multi_fpga_layernorm_v1.json"


class FrontLoadedPolicy(RowSplitPolicy):
    """Test policy: the first chiplet takes every row but one per other chiplet."""

    policy_id = "front_loaded_test"

    def partition(self, operation_id, rows, endpoint_ids):
        endpoint_ids = tuple(sorted(endpoint_ids))
        ranges = {}
        start = 0
        for position, endpoint_id in enumerate(endpoint_ids):
            count = rows - (len(endpoint_ids) - 1) if position == 0 else 1
            ranges[endpoint_id] = (start, start + count)
            start += count
        return ranges


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_evaluation_profile(PROFILE_FILE)

    def test_a_profile_without_the_field_gets_the_default(self):
        self.assertEqual(self.profile.fpga_split_policy, DEFAULT_SPLIT_POLICY)
        self.assertNotIn("fpga_split_policy", self.profile.canonical_dict())

    def test_the_fingerprint_changes_only_for_another_policy(self):
        same = self.profile.with_fpga_split_policy(DEFAULT_SPLIT_POLICY)
        other = self.profile.with_fpga_split_policy("row_split_parallel_loads_v1")
        self.assertEqual(same.fingerprint(), self.profile.fingerprint())
        self.assertNotEqual(other.fingerprint(), self.profile.fingerprint())
        self.assertEqual(
            other.canonical_dict()["fpga_split_policy"], "row_split_parallel_loads_v1"
        )

    def test_the_field_is_parsed_from_a_profile_file(self):
        entry = copy.deepcopy(self.profile.canonical_dict())
        entry["fpga_split_policy"] = "row_split_parallel_loads_v1"
        self.assertEqual(
            parse_evaluation_profile(entry).fpga_split_policy,
            "row_split_parallel_loads_v1",
        )
        entry["fpga_split_policy"] = 3
        with self.assertRaisesRegex(UnsupportedEvaluation, "fpga_split_policy"):
            parse_evaluation_profile(entry)

    def test_other_changes_keep_the_split_policy(self):
        profile = self.profile.with_fpga_split_policy("row_split_parallel_loads_v1")
        self.assertEqual(
            profile.with_movement_policy("distributed_activation_boundary_v2")
            .with_evaluator_setting("layernorm", "x", 1)
            .fpga_split_policy,
            "row_split_parallel_loads_v1",
        )

    def test_an_unknown_policy_is_unsupported(self):
        with self.assertRaisesRegex(UnsupportedEvaluation, "split policy"):
            build_split_policy("no_such_policy")
        with self.assertRaisesRegex(UnsupportedEvaluation, "fpga_split_policy"):
            self.profile.with_fpga_split_policy("")


class SplitPolicyTests(unittest.TestCase):
    def test_the_default_partition_is_the_row_split(self):
        self.assertEqual(
            build_split_policy().partition("op", 9, (2, 1)),
            fpga_row_ranges("op", 9, (2, 1)),
        )

    def test_parallel_loads_take_the_longest_load(self):
        loads = [
            ParameterLoad(10, 0, 1, 5.0, 1.0, 3.0),
            ParameterLoad(10, 0, 2, 7.0, 2.0, 4.0),
        ]
        combined = combine_parameter_loads(loads, slack_ns=4.0, shared_dram=False)
        self.assertEqual(combined.raw_latency_ns, 7.0)
        self.assertEqual(combined.exposed_latency_ns, 3.0)
        self.assertEqual(combined.byte_count, 20)
        self.assertEqual(combined.energy_pj, 7.0)


class SplitEstimatePolicyTests(unittest.TestCase):
    def _split(self, shape, policy_id=None, policy=None):
        fpgas = [_fpga(1), _fpga(2)]
        context = _stub_context(fpgas)
        context.split_policy = policy or (
            build_split_policy(policy_id) if policy_id else None
        )
        return _binding().estimate(
            _layernorm(shape), None, EndpointAssignment((1, 2), "fpga"), context
        )

    def test_parallel_loads_overlap_but_cost_the_same_energy(self):
        shared = self._split((8, 64)).parameter_load
        parallel = self._split((8, 64), "row_split_parallel_loads_v1").parameter_load
        self.assertAlmostEqual(parallel.raw_latency_ns, shared.raw_latency_ns / 2)
        self.assertEqual(parallel.byte_count, shared.byte_count)
        self.assertAlmostEqual(parallel.energy_pj, shared.energy_pj)

    def test_the_default_policy_gives_the_same_estimate_as_no_policy(self):
        self.assertEqual(
            self._split((8, 64)), self._split((8, 64), DEFAULT_SPLIT_POLICY)
        )

    def test_a_policy_decides_the_rows_of_each_chiplet(self):
        single = SplitEstimateTests()._single
        estimate = self._split((9, 64), policy=FrontLoadedPolicy())
        # The first chiplet runs 8 rows and is the slowest.
        self.assertAlmostEqual(
            estimate.compute_latency_ns, single((8, 64)).compute_latency_ns
        )


class SplitMovementPolicyTests(unittest.TestCase):
    def test_the_movement_uses_the_partition_of_the_policy(self):
        scheduler = _scheduler(_Core(0, [_Workload(m=9, k=4, n=16)]))
        request = _request(
            "gemm", "layernorm", (9, 16), [0], [1, 2],
            producer_prepared=(scheduler, None),
        )
        plan = FpgaActivationMovementService().plan(
            replace(request, split_policy=FrontLoadedPolicy())
        )
        self.assertEqual(
            sorted((s.destination_endpoint, s.row_offset, s.row_count)
                   for s in plan.transfer_slices),
            [(1, 0, 8), (2, 8, 1)],
        )


class NewChipletPolicyTests(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        self.params = _load(SEARCH_SPACE)

    def test_the_default_copies_an_existing_chiplet(self):
        chiplet = new_fpga_chiplet(
            self.architecture, ["Chiplet_2"], self.params, random.Random(1)
        )
        self.assertEqual(chiplet, self.architecture["Chiplet_2"])

    def test_from_search_space_builds_a_fresh_chiplet(self):
        params = copy.deepcopy(self.params)
        params["fpga"]["new_chiplet"] = "from_search_space_v1"
        params["fpga"]["boards"] = ["zcu102"]
        chiplet = new_fpga_chiplet(
            self.architecture, ["Chiplet_2"], params, random.Random(1)
        )
        self.assertEqual(chiplet["chiplet_type"], "fpga")
        self.assertEqual(chiplet["board"], "zcu102")

    def test_an_unknown_policy_is_unsupported(self):
        params = copy.deepcopy(self.params)
        params["fpga"]["new_chiplet"] = "no_such_policy"
        with self.assertRaisesRegex(UnsupportedEvaluation, "new FPGA chiplet policy"):
            new_fpga_chiplet(
                self.architecture, ["Chiplet_2"], params, random.Random(1)
            )


if __name__ == "__main__":
    unittest.main()
