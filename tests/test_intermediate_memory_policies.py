import unittest
from types import SimpleNamespace

from system.utils.IntermediateMemoryPolicy import (
    CoreMemory,
    TransferSlice,
    build_boundary_mapping,
    plan_boundary,
)


INTERMEDIATE_BYTES = 128 * 128


def core(core_id, capacity=64 * 1024, dram_bandwidth=100, dram_energy=5):
    return CoreMemory(
        core_id=core_id,
        capacity_bytes=capacity,
        dram_bandwidth_bytes_per_ns=dram_bandwidth,
        dram_energy_pj_per_bit=dram_energy,
    )


class FakeCore:
    def __init__(self, core_id, compute_power=1):
        self.id = core_id
        self.compute_power = compute_power
        self.buffer_size = 64
        self.dram_bandwidth = 100
        self.frequency = 10**9
        self.dram_energy_scale = 5
        self.workloads = []

    def __lt__(self, other):
        return self.compute_power < other.compute_power


class FakeSystem:
    def __init__(self, cores, route=None):
        self.core_dict = {item.id: item for item in cores}
        self.route = route

    def get_shortest_path(self, source, destination):
        if self.route is None:
            return None, 0, 0
        return self.route


def tile(m, k, n, core_id, m_offset=0, k_offset=0, n_offset=0):
    return SimpleNamespace(
        m=m,
        k=k,
        n=n,
        m_offset=m_offset,
        k_offset=k_offset,
        n_offset=n_offset,
        assigned_SA=SimpleNamespace(id=core_id),
    )


def scheduler(m, k, n, tiles, cores, splitting_k=False):
    for item in cores:
        item.workloads = [
            workload for workload in tiles if workload.assigned_SA.id == item.id
        ]
    return SimpleNamespace(
        M=m,
        K=k,
        N=n,
        workloads=tiles,
        systolic_arrays=cores,
        splitting_k=splitting_k,
    )


class IntermediateMemoryPolicyTests(unittest.TestCase):
    def test_cold_dram_charges_one_write_and_one_read(self):
        plan = plan_boundary(
            boundary_index=1,
            intermediate_bytes=INTERMEDIATE_BYTES,
            transfers=[TransferSlice(INTERMEDIATE_BYTES, 0, 0)],
            cores={0: core(0)},
            policy="cold_dram",
        )

        self.assertEqual(plan.selected_method, "cold_dram")
        self.assertEqual(plan.dram_spilled_bytes, INTERMEDIATE_BYTES)
        self.assertEqual(plan.dram_traffic_bytes, INTERMEDIATE_BYTES * 2)
        self.assertEqual(plan.retained_bytes, 0)
        self.assertEqual(plan.forwarded_bytes, 0)
        self.assertAlmostEqual(plan.latency_ns, INTERMEDIATE_BYTES * 2 / 100)
        self.assertEqual(
            plan.energy_pj,
            INTERMEDIATE_BYTES * 2 * 8 * 5,
        )

    def test_ideal_on_chip_removes_internal_transfer_cost(self):
        plan = plan_boundary(
            boundary_index=1,
            intermediate_bytes=INTERMEDIATE_BYTES,
            transfers=[TransferSlice(INTERMEDIATE_BYTES, 0, 1)],
            cores={0: core(0, capacity=1), 1: core(1, capacity=1)},
            policy="ideal_on_chip",
        )

        self.assertEqual(plan.selected_method, "ideal_on_chip")
        self.assertEqual(plan.retained_bytes, INTERMEDIATE_BYTES)
        self.assertEqual(plan.dram_traffic_bytes, 0)
        self.assertEqual(plan.latency_ns, 0)
        self.assertEqual(plan.energy_pj, 0)

    def test_local_sram_retains_same_core_intermediate_when_it_fits(self):
        plan = plan_boundary(
            boundary_index=1,
            intermediate_bytes=INTERMEDIATE_BYTES,
            transfers=[TransferSlice(INTERMEDIATE_BYTES, 0, 0)],
            cores={0: core(0)},
            policy="local_sram",
        )

        self.assertEqual(plan.selected_method, "local_sram")
        self.assertEqual(plan.retained_bytes, INTERMEDIATE_BYTES)
        self.assertEqual(plan.dram_traffic_bytes, 0)

    def test_local_sram_falls_back_when_capacity_is_insufficient(self):
        plan = plan_boundary(
            boundary_index=1,
            intermediate_bytes=INTERMEDIATE_BYTES,
            transfers=[TransferSlice(INTERMEDIATE_BYTES, 0, 0)],
            cores={0: core(0, capacity=INTERMEDIATE_BYTES - 1)},
            policy="local_sram",
        )

        self.assertEqual(plan.selected_method, "cold_dram")
        self.assertEqual(plan.dram_spilled_bytes, INTERMEDIATE_BYTES)
        self.assertIn("capacity", plan.fallback_reason)

    def test_direct_forward_uses_route_instead_of_dram(self):
        plan = plan_boundary(
            boundary_index=1,
            intermediate_bytes=INTERMEDIATE_BYTES,
            transfers=[
                TransferSlice(
                    INTERMEDIATE_BYTES,
                    0,
                    1,
                    route=(0, 1),
                    path_bandwidth_reciprocal_ns_per_byte=0.01,
                    path_energy_pj_per_bit=0.25,
                )
            ],
            cores={0: core(0), 1: core(1)},
            policy="direct_forward",
        )

        self.assertEqual(plan.selected_method, "direct_forward")
        self.assertEqual(plan.forwarded_bytes, INTERMEDIATE_BYTES)
        self.assertEqual(plan.dram_traffic_bytes, 0)
        self.assertAlmostEqual(plan.latency_ns, INTERMEDIATE_BYTES * 0.01)
        self.assertEqual(plan.energy_pj, INTERMEDIATE_BYTES * 8 * 0.25)
        self.assertEqual(plan.routes, ((0, 1),))

    def test_direct_forward_falls_back_for_disconnected_route(self):
        plan = plan_boundary(
            boundary_index=1,
            intermediate_bytes=INTERMEDIATE_BYTES,
            transfers=[
                TransferSlice(
                    INTERMEDIATE_BYTES,
                    0,
                    1,
                    route=(1,),
                    path_bandwidth_reciprocal_ns_per_byte=float("inf"),
                    path_energy_pj_per_bit=0.25,
                )
            ],
            cores={0: core(0), 1: core(1)},
            policy="direct_forward",
        )

        self.assertEqual(plan.selected_method, "cold_dram")
        self.assertEqual(plan.dram_spilled_bytes, INTERMEDIATE_BYTES)
        self.assertIn("route", plan.fallback_reason)

    def test_one_invalid_slice_falls_back_for_whole_boundary(self):
        half = INTERMEDIATE_BYTES // 2
        plan = plan_boundary(
            boundary_index=1,
            intermediate_bytes=INTERMEDIATE_BYTES,
            transfers=[
                TransferSlice(
                    half,
                    0,
                    1,
                    route=(0, 1),
                    path_bandwidth_reciprocal_ns_per_byte=0.01,
                    path_energy_pj_per_bit=0.25,
                ),
                TransferSlice(half, 0, 2),
            ],
            cores={0: core(0), 1: core(1), 2: core(2)},
            policy="direct_forward",
        )

        self.assertEqual(plan.selected_method, "cold_dram")
        self.assertEqual(plan.forwarded_bytes, 0)
        self.assertEqual(plan.dram_spilled_bytes, INTERMEDIATE_BYTES)

    def test_rejects_removed_auto_policy(self):
        with self.assertRaisesRegex(ValueError, "Unknown intermediate-memory policy"):
            plan_boundary(
                1,
                INTERMEDIATE_BYTES,
                [TransferSlice(INTERMEDIATE_BYTES, 0, 0)],
                {0: core(0)},
                "auto",
            )

    def test_rejects_unaccounted_intermediate_bytes(self):
        with self.assertRaisesRegex(ValueError, "cover the intermediate"):
            plan_boundary(
                boundary_index=1,
                intermediate_bytes=INTERMEDIATE_BYTES,
                transfers=[TransferSlice(INTERMEDIATE_BYTES - 1, 0, 0)],
                cores={0: core(0)},
                policy="cold_dram",
            )


class BoundaryMappingTests(unittest.TestCase):
    def test_maps_same_core_output_to_consumer_activation(self):
        producer_core = FakeCore(0)
        consumer_core = FakeCore(0)
        producer = scheduler(4, 8, 4, [tile(4, 8, 4, 0)], [producer_core])
        consumer = scheduler(4, 4, 2, [tile(4, 4, 2, 0)], [consumer_core])

        mapping = build_boundary_mapping(
            producer,
            FakeSystem([producer_core]),
            consumer,
            FakeSystem([consumer_core]),
        )

        self.assertTrue(mapping.valid)
        self.assertEqual(mapping.intermediate_bytes, 16)
        self.assertEqual(mapping.transfers, (TransferSlice(16, 0, 0),))

    def test_maps_remote_slice_with_existing_route(self):
        core_0 = FakeCore(0)
        core_1 = FakeCore(1)
        producer = scheduler(4, 8, 4, [tile(4, 8, 4, 0)], [core_0])
        consumer = scheduler(4, 4, 2, [tile(4, 4, 2, 1)], [core_1])

        mapping = build_boundary_mapping(
            producer,
            FakeSystem([core_0, core_1], route=([0, 1], 0.02, 0.5)),
            consumer,
            FakeSystem([core_0, core_1], route=([0, 1], 0.02, 0.5)),
        )

        self.assertTrue(mapping.valid)
        self.assertEqual(mapping.transfers[0].route, (0, 1))
        self.assertEqual(
            mapping.transfers[0].path_bandwidth_reciprocal_ns_per_byte,
            0.02,
        )
        self.assertEqual(mapping.transfers[0].path_energy_pj_per_bit, 0.5)

    def test_rejects_incomplete_intermediate_coverage(self):
        producer_core = FakeCore(0)
        consumer_core = FakeCore(0)
        producer = scheduler(4, 8, 4, [tile(2, 8, 4, 0)], [producer_core])
        consumer = scheduler(4, 4, 2, [tile(4, 4, 2, 0)], [consumer_core])

        mapping = build_boundary_mapping(
            producer,
            FakeSystem([producer_core]),
            consumer,
            FakeSystem([consumer_core]),
        )

        self.assertFalse(mapping.valid)
        self.assertIn("cover", mapping.error)

    def test_split_k_uses_final_reduction_owner(self):
        core_0 = FakeCore(0, compute_power=1)
        core_1 = FakeCore(1, compute_power=2)
        producer = scheduler(
            4,
            8,
            4,
            [
                tile(4, 4, 4, 0, k_offset=0),
                tile(4, 4, 4, 1, k_offset=4),
            ],
            [core_0, core_1],
            splitting_k=True,
        )
        consumer_core = FakeCore(1, compute_power=2)
        consumer = scheduler(4, 4, 2, [tile(4, 4, 2, 1)], [consumer_core])

        mapping = build_boundary_mapping(
            producer,
            FakeSystem([core_0, core_1]),
            consumer,
            FakeSystem([consumer_core]),
        )

        self.assertTrue(mapping.valid)
        self.assertEqual(mapping.transfers, (TransferSlice(16, 1, 1),))


if __name__ == "__main__":
    unittest.main()
