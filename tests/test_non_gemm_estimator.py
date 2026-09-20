import unittest
from types import SimpleNamespace

from system.utils.FpgaChiplet import FpgaChiplet, FpgaReluImplementation
from system.utils.NonGemmEstimator import (
    FpgaReluEvaluator,
    LegacyReluInput,
    NonGemmEstimator,
    ReluComputeEstimator,
    ReluStageComposer,
)
from system.utils.OperationEvaluator import EvaluationContext
from system.utils.OperationPlacement import EndpointPlacement
from system.utils.TransferEstimator import ResolvedRoute
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


def _relu_operation(element_count=512):
    return SimpleNamespace(
        operation_id="relu_1",
        operation_type="relu",
        element_count=element_count,
        input_tensor_id="relu_1:input",
        output_tensor_id="relu_1:output",
    )


def _relu_input(element_count=512):
    return LegacyReluInput(operation_id="relu_1", element_count=element_count)


def _fpga(
    clbs=10000,
    brams=200,
    dsps=500,
    clbs_per_lane=5,
    brams_per_lane=0,
    dsps_per_lane=0,
    max_parallel_lanes=64,
    energy_per_element_pj=None,
    frequency_hz=300000000,
):
    return FpgaChiplet(
        id=1,
        name="Chiplet_2",
        frequency_hz=frequency_hz,
        clbs=clbs,
        brams=brams,
        dsps=dsps,
        area=10.0,
        power=2.0,
        node=7,
        relu_implementation=FpgaReluImplementation(
            clbs_per_lane=clbs_per_lane,
            brams_per_lane=brams_per_lane,
            dsps_per_lane=dsps_per_lane,
            max_parallel_lanes=max_parallel_lanes,
            energy_per_element_pj=energy_per_element_pj,
        ),
    )


class ReluComputeEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.estimator = ReluComputeEstimator()
        self.operation = _relu_operation()

    def _estimate(self, fpga):
        return self.estimator.estimate(
            self.operation.element_count, fpga, self.operation.operation_id
        )

    def test_clb_limited_lanes(self):
        estimate = self._estimate(_fpga(clbs=320))
        self.assertEqual(estimate.parallel_lanes, 64)

    def test_max_parallel_lanes_cap(self):
        estimate = self._estimate(_fpga(clbs=10000, max_parallel_lanes=16))
        self.assertEqual(estimate.parallel_lanes, 16)

    def test_bram_limited_lanes(self):
        estimate = self._estimate(
            _fpga(clbs=10000, brams=8, brams_per_lane=1, max_parallel_lanes=None)
        )
        self.assertEqual(estimate.parallel_lanes, 8)

    def test_dsp_limited_lanes(self):
        estimate = self._estimate(
            _fpga(clbs=10000, dsps=4, dsps_per_lane=1, max_parallel_lanes=None)
        )
        self.assertEqual(estimate.parallel_lanes, 4)

    def test_zero_lanes_is_infeasible(self):
        estimate = self._estimate(_fpga(clbs=4))
        self.assertFalse(estimate.feasible)
        self.assertEqual(estimate.parallel_lanes, 0)

    def test_512_elements_64_lanes_is_8_cycles(self):
        estimate = self._estimate(_fpga())
        self.assertEqual(estimate.compute_cycles, 8)

    def test_latency_uses_frequency(self):
        estimate = self._estimate(_fpga())
        self.assertAlmostEqual(estimate.compute_latency_ns, 8 / 300000000 * 1e9)

    def test_configured_energy_is_calculated(self):
        estimate = self._estimate(_fpga(energy_per_element_pj=0.5))
        self.assertAlmostEqual(estimate.compute_energy_pj, 512 * 0.5)

    def test_missing_energy_is_null(self):
        estimate = self._estimate(_fpga())
        self.assertIsNone(estimate.compute_energy_pj)


class ReluStageComposerTests(unittest.TestCase):
    def test_serialized_stage_total_latency(self):
        composer = ReluStageComposer()
        operation = _relu_operation()
        input_route = ResolvedRoute(
            source_chiplet_id=0,
            destination_chiplet_id=1,
            path=(0, 1),
            path_bandwidth_reciprocal_ns_per_byte=0.02,
            path_energy_pj_per_bit=0.5,
            setup_latency_ns=5.0,
            hop_latency_ns=1.0,
        )
        output_route = ResolvedRoute(
            source_chiplet_id=1,
            destination_chiplet_id=0,
            path=(1, 0),
            path_bandwidth_reciprocal_ns_per_byte=0.02,
            path_energy_pj_per_bit=0.5,
            setup_latency_ns=5.0,
            hop_latency_ns=1.0,
        )

        stage = composer.compose(operation, _fpga(), input_route, output_route)

        compute_latency = stage.compute.compute_latency_ns
        expected = stage.input_transfer.latency_ns + compute_latency + stage.output_transfer.latency_ns
        self.assertAlmostEqual(stage.total_latency_ns, expected)

    def test_stage_owns_both_transfer_energies(self):
        composer = ReluStageComposer()
        operation = _relu_operation()
        route_in = ResolvedRoute(
            source_chiplet_id=0,
            destination_chiplet_id=1,
            path=(0, 1),
            path_energy_pj_per_bit=0.5,
        )
        route_out = ResolvedRoute(
            source_chiplet_id=1,
            destination_chiplet_id=0,
            path=(1, 0),
            path_energy_pj_per_bit=0.25,
        )
        stage = composer.compose(operation, _fpga(), route_in, route_out)
        expected = 512 * 8 * 0.5 + 512 * 8 * 0.25
        self.assertAlmostEqual(stage.transfer_energy_pj, expected)


class FpgaReluEvaluatorTests(unittest.TestCase):
    def _context(self, fpga):
        return EvaluationContext(
            cache=None,
            architecture={},
            system=SimpleNamespace(fpga_chiplet_dict={1: fpga}),
        )

    def test_returns_compute_latency_and_dynamic_energy(self):
        estimate = FpgaReluEvaluator().evaluate(
            _relu_input(),
            EndpointPlacement(1, "fpga"),
            self._context(_fpga(energy_per_element_pj=0.5)),
        )
        self.assertEqual(estimate.operation_id, "relu_1")
        self.assertEqual(estimate.evaluator_id, "legacy_fpga_relu_v1")
        self.assertAlmostEqual(estimate.compute_latency_ns, 8 / 300000000 * 1e9)
        self.assertAlmostEqual(estimate.dynamic_energy_pj, 512 * 0.5)

    def test_missing_endpoint_is_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            FpgaReluEvaluator().evaluate(
                _relu_input(),
                EndpointPlacement(7, "fpga"),
                self._context(_fpga()),
            )

    def test_infeasible_resources_are_unsupported(self):
        with self.assertRaises(UnsupportedEvaluation):
            FpgaReluEvaluator().evaluate(
                _relu_input(),
                EndpointPlacement(1, "fpga"),
                self._context(_fpga(clbs=4)),
            )


class NonGemmEstimatorFacadeTests(unittest.TestCase):
    def test_dispatches_relu(self):
        estimator = NonGemmEstimator()
        estimate = estimator.estimate_operation(_relu_operation(), _fpga())
        self.assertTrue(estimate.feasible)
        self.assertEqual(estimate.compute_cycles, 8)

    def test_rejects_unknown_operation(self):
        estimator = NonGemmEstimator()

        class Fake:
            operation_type = "layernorm"
            operation_id = "x"

        with self.assertRaises(ValueError):
            estimator.estimate_operation(Fake(), _fpga())


if __name__ == "__main__":
    unittest.main()
