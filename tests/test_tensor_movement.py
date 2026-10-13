import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from system.utils.TensorMovement import (
    DistributedActivationMovementService,
    MovementRequest,
    RoutedTensorSlice,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


@dataclass
class FakeWorkload:
    m: int
    k: int
    n: int
    m_offset: int = 0
    k_offset: int = 0
    n_offset: int = 0


class FakeCore:
    def __init__(self, core_id, workloads):
        self.id = core_id
        self.workloads = list(workloads)

    def __lt__(self, other):
        return self.id < other.id


class FakeScheduler:
    def __init__(self, cores, splitting_k=False):
        self.systolic_arrays = list(cores)
        self.splitting_k = splitting_k


class FakeSystem:
    """Topology double with explicit routes and directed link costs."""

    def __init__(self, links, routes=None):
        self.links = {tuple(sorted(key)): value for key, value in links.items()}
        self.routes = routes or {}

    def link_cost(self, source, destination):
        key = tuple(sorted((source, destination)))
        if key not in self.links:
            raise UnsupportedEvaluation(
                f"no physical link between endpoints {source} and {destination}"
            )
        return self.links[key]

    def get_shortest_path(self, source, destination):
        if source == destination:
            return None, 0, 0
        path = self.routes.get((source, destination))
        if not path:
            return None, 0, 0
        reciprocal = sum(
            self.link_cost(a, b)[0] for a, b in zip(path, path[1:])
        )
        energy = sum(self.link_cost(a, b)[1] for a, b in zip(path, path[1:]))
        return list(path), reciprocal, energy


def _assignment(*endpoint_ids, kind):
    return SimpleNamespace(
        endpoint_ids=tuple(endpoint_ids),
        endpoint_kind=kind,
        endpoint_id=endpoint_ids[0] if len(endpoint_ids) == 1 else None,
    )


def _operation(operation_type):
    return SimpleNamespace(operation_type=operation_type)


def _request(
    producer_type,
    consumer_type,
    tensor_shape,
    producer_assignment,
    consumer_assignment,
    system,
    producer_prepared=None,
    consumer_prepared=None,
):
    return MovementRequest(
        handoff_index=1,
        tensor_id="t",
        tensor_shape=tensor_shape,
        producer_operation=_operation(producer_type),
        consumer_operation=_operation(consumer_type),
        producer_assignment=producer_assignment,
        consumer_assignment=consumer_assignment,
        producer_prepared=producer_prepared,
        consumer_prepared=consumer_prepared,
        system=system,
        transfer_model={"setup_latency_ns": 0.0, "hop_latency_ns": 0.0},
    )


def _draft(source, destination, byte_count, route, setup=0.0, hop=0.0):
    return RoutedTensorSlice(
        tensor_id="t",
        row_offset=0,
        row_count=1,
        feature_offset=0,
        feature_count=byte_count,
        source_endpoint=source,
        destination_endpoint=destination,
        byte_count=byte_count,
        route=tuple(route),
        setup_latency_ns=setup,
        hop_latency_ns=hop,
        serialization_latency_ns=0.0,
        slice_latency_ns=0.0,
        slice_energy_pj=0.0,
    )


class DistributedMovementPolicyTests(unittest.TestCase):
    def setUp(self):
        self.service = DistributedActivationMovementService()

    def test_same_fpga_relu_handoff_is_retained(self):
        request = _request(
            "relu",
            "relu",
            (8, 8),
            _assignment(1, kind="fpga"),
            _assignment(1, kind="fpga"),
            FakeSystem({}),
        )

        plan = self.service.plan(request)

        self.assertEqual(plan.method, "retain")
        self.assertEqual(plan.charged_latency_ns, 0.0)
        self.assertEqual(plan.charged_energy_pj, 0.0)
        self.assertEqual(plan.logical_bytes, 64)
        self.assertEqual(plan.physical_bytes, 0)
        self.assertEqual(plan.transfer_slices, ())

    def test_unsupported_pairs_are_rejected(self):
        for producer, consumer in (
            ("gemm", "gemm"),
            ("softmax", "relu"),
            ("relu", "softmax"),
            ("gemm", "softmax"),
        ):
            with self.subTest(pair=(producer, consumer)):
                with self.assertRaises(UnsupportedEvaluation):
                    self.service.plan(
                        _request(
                            producer,
                            consumer,
                            (8, 8),
                            _assignment(0, kind="sa"),
                            _assignment(1, kind="fpga"),
                            FakeSystem({}),
                        )
                    )

    def test_gemm_output_ownership_covers_the_tensor_once(self):
        scheduler = FakeScheduler(
            [
                FakeCore(
                    0,
                    [
                        FakeWorkload(64, 64, 64, m_offset=0, n_offset=0),
                        FakeWorkload(64, 64, 64, m_offset=0, n_offset=64),
                    ],
                ),
                FakeCore(
                    1,
                    [
                        FakeWorkload(64, 64, 64, m_offset=64, n_offset=0),
                        FakeWorkload(64, 64, 64, m_offset=64, n_offset=64),
                    ],
                ),
            ]
        )
        system = FakeSystem(
            {(0, 2): (1.0, 0.5), (1, 2): (1.0, 0.5)},
            routes={(0, 2): [0, 2], (1, 2): [1, 2]},
        )
        request = _request(
            "gemm",
            "relu",
            (128, 128),
            _assignment(0, 1, kind="sa"),
            _assignment(2, kind="fpga"),
            system,
            producer_prepared=(scheduler, system),
        )

        plan = self.service.plan(request)

        self.assertEqual(plan.method, "distributed_route")
        self.assertEqual(plan.logical_bytes, 16384)
        self.assertEqual(plan.physical_bytes, 16384)
        self.assertEqual(len(plan.transfer_slices), 4)
        self.assertEqual(sum(s.byte_count for s in plan.transfer_slices), 16384)
        self.assertEqual(plan.source_endpoint, (0, 1))
        self.assertEqual(plan.destination_endpoint, 2)

    def test_replicates_a_shared_region_across_destination_sas(self):
        region = FakeWorkload(128, 128, 64, m_offset=0, k_offset=0, n_offset=0)
        scheduler = FakeScheduler([FakeCore(0, [region]), FakeCore(1, [region])])
        system = FakeSystem(
            {(2, 0): (1.0, 0.5), (2, 1): (1.0, 0.5)},
            routes={(2, 0): [2, 0], (2, 1): [2, 1]},
        )
        request = _request(
            "relu",
            "gemm",
            (128, 128),
            _assignment(2, kind="fpga"),
            _assignment(0, 1, kind="sa"),
            system,
            consumer_prepared=(scheduler, system),
        )

        plan = self.service.plan(request)

        self.assertEqual(plan.logical_bytes, 16384)
        self.assertEqual(plan.physical_bytes, 32768)
        self.assertEqual(len(plan.transfer_slices), 2)
        self.assertEqual({s.destination_endpoint for s in plan.transfer_slices}, {0, 1})

    def test_same_sa_duplicate_regions_are_charged_once(self):
        scheduler = FakeScheduler(
            [
                FakeCore(
                    0,
                    [
                        FakeWorkload(128, 128, 64, m_offset=0, k_offset=0, n_offset=0),
                        FakeWorkload(128, 128, 64, m_offset=0, k_offset=0, n_offset=64),
                    ],
                )
            ]
        )
        system = FakeSystem({(2, 0): (1.0, 0.5)}, routes={(2, 0): [2, 0]})
        request = _request(
            "relu",
            "gemm",
            (128, 128),
            _assignment(2, kind="fpga"),
            _assignment(0, kind="sa"),
            system,
            consumer_prepared=(scheduler, system),
        )

        plan = self.service.plan(request)

        self.assertEqual(len(plan.transfer_slices), 1)
        self.assertEqual(plan.physical_bytes, 16384)

    def test_split_k_transfer_comes_from_the_reduction_owner(self):
        scheduler = FakeScheduler(
            [
                FakeCore(
                    0,
                    [FakeWorkload(128, 64, 128, m_offset=0, n_offset=0, k_offset=0)],
                ),
                FakeCore(
                    1,
                    [FakeWorkload(128, 64, 128, m_offset=0, n_offset=0, k_offset=64)],
                ),
            ],
            splitting_k=True,
        )
        system = FakeSystem(
            {(0, 2): (1.0, 0.5), (1, 2): (1.0, 0.5)},
            routes={(0, 2): [0, 2], (1, 2): [1, 2]},
        )
        request = _request(
            "gemm",
            "relu",
            (128, 128),
            _assignment(0, 1, kind="sa"),
            _assignment(2, kind="fpga"),
            system,
            producer_prepared=(scheduler, system),
        )

        plan = self.service.plan(request)

        self.assertEqual(len(plan.transfer_slices), 1)
        self.assertEqual(plan.transfer_slices[0].source_endpoint, 1)
        self.assertEqual(plan.physical_bytes, 16384)

    def test_conflicting_region_ownership_is_rejected(self):
        region = FakeWorkload(64, 64, 64, m_offset=0, n_offset=0)
        scheduler = FakeScheduler([FakeCore(0, [region]), FakeCore(1, [region])])
        system = FakeSystem({(0, 2): (1.0, 0.5)}, routes={(0, 2): [0, 2]})
        request = _request(
            "gemm",
            "relu",
            (128, 128),
            _assignment(0, 1, kind="sa"),
            _assignment(2, kind="fpga"),
            system,
            producer_prepared=(scheduler, system),
        )

        with self.assertRaises(UnsupportedEvaluation):
            self.service.plan(request)

    def test_incomplete_output_coverage_is_rejected(self):
        scheduler = FakeScheduler(
            [FakeCore(0, [FakeWorkload(64, 64, 64, m_offset=0, n_offset=0)])]
        )
        system = FakeSystem({(0, 2): (1.0, 0.5)}, routes={(0, 2): [0, 2]})
        request = _request(
            "gemm",
            "relu",
            (128, 128),
            _assignment(0, kind="sa"),
            _assignment(2, kind="fpga"),
            system,
            producer_prepared=(scheduler, system),
        )

        with self.assertRaises(UnsupportedEvaluation):
            self.service.plan(request)

    def test_missing_prepared_artifact_is_rejected(self):
        request = _request(
            "gemm",
            "relu",
            (128, 128),
            _assignment(0, kind="sa"),
            _assignment(2, kind="fpga"),
            FakeSystem({}),
        )

        with self.assertRaises(UnsupportedEvaluation):
            self.service.plan(request)


class RoutedBatchCostingTests(unittest.TestCase):
    def test_shared_link_serializes_while_disjoint_links_stay_parallel(self):
        shared = FakeSystem(
            {
                (0, 3): (2.0, 0.5),
                (1, 3): (2.0, 0.5),
                (2, 3): (2.0, 0.5),
            }
        )
        shared_plan = DistributedActivationMovementService._aggregate(
            SimpleNamespace(system=shared),
            [_draft(0, 2, 10, (0, 3, 2)), _draft(1, 2, 10, (1, 3, 2))],
            logical_bytes=20,
            source=(0, 1),
            destination=2,
        )

        disjoint = FakeSystem({(0, 2): (2.0, 0.5), (1, 3): (2.0, 0.5)})
        disjoint_plan = DistributedActivationMovementService._aggregate(
            SimpleNamespace(system=disjoint),
            [_draft(0, 2, 10, (0, 2)), _draft(1, 3, 10, (1, 3))],
            logical_bytes=20,
            source=(0, 1),
            destination=(2, 3),
        )

        self.assertAlmostEqual(shared_plan.charged_latency_ns, 60.0)
        self.assertAlmostEqual(shared_plan.charged_energy_pj, 160.0)
        self.assertAlmostEqual(disjoint_plan.charged_latency_ns, 20.0)
        self.assertAlmostEqual(disjoint_plan.charged_energy_pj, 80.0)

    def test_aggregate_decomposes_slice_latency(self):
        system = FakeSystem({(0, 1): (2.0, 0.5)}, routes={(0, 1): [0, 1]})
        plan = DistributedActivationMovementService._aggregate(
            SimpleNamespace(system=system),
            [_draft(0, 1, 10, (0, 1), setup=5.0, hop=1.0)],
            logical_bytes=10,
            source=0,
            destination=1,
        )

        tensor_slice = plan.transfer_slices[0]
        self.assertEqual(tensor_slice.setup_latency_ns, 5.0)
        self.assertEqual(tensor_slice.hop_latency_ns, 1.0)
        self.assertAlmostEqual(tensor_slice.serialization_latency_ns, 20.0)
        self.assertAlmostEqual(tensor_slice.slice_latency_ns, 26.0)
        self.assertAlmostEqual(plan.charged_latency_ns, 26.0)

    def test_missing_edge_is_unsupported(self):
        system = FakeSystem({}, routes={(0, 1): [0, 1]})
        with self.assertRaises(UnsupportedEvaluation):
            DistributedActivationMovementService._aggregate(
                SimpleNamespace(system=system),
                [_draft(0, 1, 10, (0, 1))],
                logical_bytes=10,
                source=0,
                destination=1,
            )


if __name__ == "__main__":
    unittest.main()
