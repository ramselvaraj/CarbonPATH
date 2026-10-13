import unittest

from system.utils.FpgaActivationMovement import (
    FpgaActivationMovementService,
    register_fpga_activation_movement,
)
from system.utils.TensorMovement import MOVEMENT_POLICIES, build_movement_service
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from tests.test_tensor_movement import (
    FakeCore,
    FakeScheduler,
    FakeSystem,
    FakeWorkload,
    _assignment,
    _request,
)


class FpgaActivationMovementTests(unittest.TestCase):
    def setUp(self):
        self.service = FpgaActivationMovementService()

    def test_policy_resolves_by_profile_id(self):
        register_fpga_activation_movement()
        self.assertIn("distributed_activation_boundary_v2", MOVEMENT_POLICIES)
        self.assertIsInstance(
            build_movement_service("distributed_activation_boundary_v2"),
            FpgaActivationMovementService,
        )

    def test_v1_policy_still_rejects_gemm_to_softmax(self):
        register_fpga_activation_movement()
        v1 = build_movement_service("distributed_activation_boundary_v1")
        with self.assertRaises(UnsupportedEvaluation):
            v1.plan(
                _request(
                    "gemm", "softmax", (8, 8),
                    _assignment(0, kind="sa"), _assignment(1, kind="fpga"),
                    FakeSystem({}),
                )
            )

    def test_softmax_matches_relu_for_gemm_to_fpga(self):
        scheduler = FakeScheduler(
            [FakeCore(0, [FakeWorkload(8, 8, 8)]), FakeCore(1, [FakeWorkload(8, 8, 8, n_offset=8)])]
        )
        system = FakeSystem(
            {(0, 2): (1.0, 0.5), (1, 2): (1.0, 0.5)},
            routes={(0, 2): [0, 2], (1, 2): [1, 2]},
        )
        plans = {}
        for activation in ("relu", "softmax"):
            plans[activation] = self.service.plan(
                _request(
                    "gemm", activation, (8, 16),
                    _assignment(0, 1, kind="sa"), _assignment(2, kind="fpga"),
                    system, producer_prepared=(scheduler, system),
                )
            )
        self.assertEqual(plans["softmax"], plans["relu"])
        self.assertEqual(plans["softmax"].logical_bytes, 128)
        self.assertEqual(sum(s.byte_count for s in plans["softmax"].transfer_slices), 128)

    def test_softmax_to_gemm_replicates_per_destination_sa(self):
        region = FakeWorkload(8, 8, 8)
        scheduler = FakeScheduler([FakeCore(0, [region]), FakeCore(1, [region])])
        system = FakeSystem(
            {(2, 0): (1.0, 0.5), (2, 1): (1.0, 0.5)},
            routes={(2, 0): [2, 0], (2, 1): [2, 1]},
        )
        plan = self.service.plan(
            _request(
                "softmax", "gemm", (8, 8),
                _assignment(2, kind="fpga"), _assignment(0, 1, kind="sa"),
                system, consumer_prepared=(scheduler, system),
            )
        )
        self.assertEqual(plan.logical_bytes, 64)
        self.assertEqual(plan.physical_bytes, 128)

    def test_same_fpga_activation_handoffs_are_retained(self):
        for producer, consumer in (
            ("relu", "relu"), ("relu", "softmax"),
            ("softmax", "relu"), ("softmax", "softmax"),
        ):
            with self.subTest(pair=(producer, consumer)):
                plan = self.service.plan(
                    _request(
                        producer, consumer, (8, 8),
                        _assignment(1, kind="fpga"), _assignment(1, kind="fpga"),
                        FakeSystem({}),
                    )
                )
                self.assertEqual(plan.method, "retain")
                self.assertEqual(plan.physical_bytes, 0)
                self.assertEqual(plan.charged_latency_ns, 0.0)

    def test_unsupported_pairs_are_rejected(self):
        for producer, consumer in (("gemm", "gemm"), ("softmax", "softmax")):
            with self.subTest(pair=(producer, consumer)):
                with self.assertRaises(UnsupportedEvaluation):
                    self.service.plan(
                        _request(
                            producer, consumer, (8, 8),
                            _assignment(0, kind="sa"), _assignment(1, kind="fpga"),
                            FakeSystem({}),
                        )
                    )
