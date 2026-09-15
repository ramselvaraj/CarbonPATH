import unittest

from system.utils.TransferEstimator import (
    ResolvedRoute,
    TransferEstimator,
    TransferRequest,
)


class TransferEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.estimator = TransferEstimator()

    def _estimate(self, route, element_count=512):
        return self.estimator.estimate(
            TransferRequest(
                tensor_id="tensor",
                element_count=element_count,
                route=route,
            )
        )

    def test_int8_bytes_equal_element_count(self):
        route = ResolvedRoute(source_chiplet_id=0, destination_chiplet_id=1)
        estimate = self._estimate(route)
        self.assertEqual(estimate.byte_count, 512)

    def test_setup_hop_and_serialization_latency(self):
        route = ResolvedRoute(
            source_chiplet_id=0,
            destination_chiplet_id=1,
            path=(0, 1),
            path_bandwidth_reciprocal_ns_per_byte=0.02,
            path_energy_pj_per_bit=0.5,
            setup_latency_ns=5.0,
            hop_latency_ns=1.0,
        )
        estimate = self._estimate(route)
        expected_latency = 5.0 + 1 * 1.0 + 512 * 0.02
        self.assertAlmostEqual(estimate.latency_ns, expected_latency)

    def test_energy_converts_bytes_to_bits(self):
        route = ResolvedRoute(
            source_chiplet_id=0,
            destination_chiplet_id=1,
            path=(0, 1),
            path_bandwidth_reciprocal_ns_per_byte=0.02,
            path_energy_pj_per_bit=0.5,
        )
        estimate = self._estimate(route)
        self.assertAlmostEqual(estimate.energy_pj, 512 * 8 * 0.5)

    def test_local_route_has_zero_cost(self):
        route = ResolvedRoute(source_chiplet_id=3, destination_chiplet_id=3)
        estimate = self._estimate(route)
        self.assertEqual(estimate.latency_ns, 0)
        self.assertEqual(estimate.energy_pj, 0)

    def test_none_path_is_treated_as_local(self):
        route = ResolvedRoute(
            source_chiplet_id=0,
            destination_chiplet_id=1,
            path=None,
        )
        estimate = self._estimate(route)
        self.assertEqual(estimate.latency_ns, 0)
        self.assertEqual(estimate.energy_pj, 0)

    def test_invalid_element_count_rejected(self):
        route = ResolvedRoute(source_chiplet_id=0, destination_chiplet_id=1)
        with self.assertRaises(ValueError):
            self._estimate(route, element_count=0)

    def test_negative_route_parameter_rejected(self):
        route = ResolvedRoute(
            source_chiplet_id=0,
            destination_chiplet_id=1,
            path=(0, 1),
            path_bandwidth_reciprocal_ns_per_byte=-1.0,
        )
        with self.assertRaises(ValueError):
            self._estimate(route)


if __name__ == "__main__":
    unittest.main()
