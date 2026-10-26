import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from main import evaluate_atlas_graph
from system.utils.AtlasGraphAdapter import parse_atlas_graph
from system.utils.ChipletResourceCheck import (
    ChipletUse,
    check_chiplet_resources,
    chiplet_totals,
)
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.FpgaChiplet import FpgaChiplet, FpgaReluImplementation
from system.utils.OperationAssignment import EndpointAssignment
from system.utils.OperationEvaluator import EvaluationContext, EvaluatorRegistry
from system.utils.Rule4MlEvaluators import register_rule4ml_evaluators
from system.utils.Rule4MlHybridEvaluators import register_rule4ml_hybrid_evaluators
from system.utils.UnsupportedEvaluation import ResourceExceeded
from system.utils.WorkloadMapping import initial_workload_mapping

ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
PROFILE = Path("cfg/profiles/rule4ml_fpga_ops_v3.json")


def _fpga(fpga_id, brams=280, luts=53200):
    return FpgaChiplet(
        id=fpga_id,
        name=f"Chiplet_{fpga_id + 1}",
        frequency_hz=300000000,
        clbs=6650,
        brams=brams,
        dsps=220,
        area=6.65,
        power=1.33,
        node=7,
        relu_implementation=FpgaReluImplementation(clbs_per_lane=5),
        luts=luts,
        ffs=106400,
    )


def _system(*fpgas):
    return SimpleNamespace(fpga_chiplet_dict={fpga.id: fpga for fpga in fpgas})


def _use(endpoint_id, bram=0.0, lut=0.0, buffer_blocks=0):
    return ChipletUse(endpoint_id, {"BRAM": bram, "LUT": lut}, buffer_blocks)


class ChipletTotalsTests(unittest.TestCase):
    def test_kernels_add_up(self):
        totals = chiplet_totals([("a", _use(1, bram=10, lut=100)), ("b", _use(1, bram=5, lut=50))])
        self.assertEqual(totals[1]["BRAM"], 15)
        self.assertEqual(totals[1]["LUT"], 150)

    def test_the_input_buffer_counts_once_as_the_largest(self):
        totals = chiplet_totals(
            [
                ("a", _use(1, bram=10, buffer_blocks=3)),
                ("b", _use(1, bram=5, buffer_blocks=8)),
                ("c", _use(1, bram=1, buffer_blocks=2)),
            ]
        )
        self.assertEqual(totals[1]["BRAM"], 10 + 5 + 1 + 8)

    def test_chiplets_do_not_add_to_each_other(self):
        totals = chiplet_totals(
            [("a", _use(1, bram=10, buffer_blocks=1)), ("b", _use(2, bram=7, buffer_blocks=4))]
        )
        self.assertEqual(totals[1]["BRAM"], 11)
        self.assertEqual(totals[2]["BRAM"], 11)

    def test_a_missing_label_counts_as_zero(self):
        totals = chiplet_totals([("a", ChipletUse(1, {"LUT": 5}, 2))])
        self.assertEqual(totals[1], {"BRAM": 2.0, "DSP": 0.0, "FF": 0.0, "LUT": 5.0})


class ChipletCheckTests(unittest.TestCase):
    def test_kernels_that_fit_alone_but_not_together_are_rejected(self):
        system = _system(_fpga(1, brams=100))
        uses = [("tanh_1", _use(1, bram=60)), ("sigmoid_1", _use(1, bram=60))]
        check_chiplet_resources(system, uses[:1])
        check_chiplet_resources(system, uses[1:])
        with self.assertRaises(ResourceExceeded) as raised:
            check_chiplet_resources(system, uses)
        message = str(raised.exception)
        self.assertIn("endpoint 1", message)
        self.assertIn("BRAM 120 > 100", message)
        self.assertIn("sigmoid_1", message)
        self.assertIn("tanh_1", message)

    def test_the_buffer_is_part_of_the_bram_total(self):
        system = _system(_fpga(1, brams=100))
        with self.assertRaises(ResourceExceeded):
            check_chiplet_resources(system, [("a", _use(1, bram=99, buffer_blocks=2))])

    def test_a_total_at_the_limit_fits(self):
        system = _system(_fpga(1, brams=100))
        check_chiplet_resources(system, [("a", _use(1, bram=98, buffer_blocks=2))])

    def test_every_resource_is_checked(self):
        system = _system(_fpga(1, luts=1000))
        with self.assertRaisesRegex(ResourceExceeded, "LUT 1200 > 1000"):
            check_chiplet_resources(
                system, [("a", _use(1, lut=600)), ("b", _use(1, lut=600))]
            )

    def test_operations_on_other_chiplets_do_not_count(self):
        system = _system(_fpga(1, brams=100), _fpga(2, brams=100))
        check_chiplet_resources(
            system, [("a", _use(1, bram=60)), ("b", _use(2, bram=60))]
        )


def _node(name, node_class, source, activation=None):
    node = {
        "name": name,
        "class": node_class,
        "inputs": [{"name": source, "shape": [8, 16]}],
        "output": {"name": name, "shape": [8, 16]},
    }
    if activation:
        node["attrs"] = {"activation": activation}
    return node


def _graph():
    nodes = [
        {
            "name": "input_layer",
            "class": "Input",
            "inputs": [],
            "output": {"name": "input_layer", "shape": [8, 16]},
        },
        _node("tanh_1", "Activation", "input_layer", "tanh"),
        _node("sigmoid_1", "Activation", "tanh_1", "sigmoid"),
        _node("softmax_1", "Softmax", "sigmoid_1", "softmax"),
    ]
    return parse_atlas_graph(nodes, "tanh_sigmoid_softmax")


def _prediction(bram=95.0):
    return mock.patch(
        "system.utils.Rule4MlEvaluators.predict_activation",
        return_value={
            "cycles": 100.0,
            "interval": 10.0,
            "resources": {"BRAM": bram, "LUT": 1000.0, "FF": 500.0, "DSP": 2.0},
        },
    )


class EvaluatorReportTests(unittest.TestCase):
    def _estimate(self, evaluator_id, registrar, operation_index):
        registry = EvaluatorRegistry()
        registrar(registry)
        context = EvaluationContext(
            cache=None,
            architecture={},
            system=_system(_fpga(1)),
            evaluator_settings={},
        )
        operation = _graph().operations[operation_index]
        with _prediction():
            return registry.get(evaluator_id).estimate(
                operation, None, EndpointAssignment(1, "fpga"), context
            )

    def test_tanh_sigmoid_and_softmax_report_their_kernel_and_buffer(self):
        for evaluator_id, index in (
            ("rule4ml_fpga_tanh_v1", 0),
            ("rule4ml_fpga_sigmoid_v1", 1),
            ("rule4ml_fpga_softmax_v1", 2),
        ):
            with self.subTest(evaluator_id=evaluator_id):
                estimate = self._estimate(
                    evaluator_id, register_rule4ml_evaluators, index
                )
                (use,) = estimate.chiplet_use
                self.assertEqual(use.endpoint_id, 1)
                self.assertEqual(use.kernel["BRAM"], 95.0)
                self.assertEqual(use.kernel["LUT"], 1000.0)
                # 8 x 16 values x 8 bits = 1024 bits: one block.
                self.assertEqual(use.buffer_blocks, 1)

    def test_the_hybrid_softmax_keeps_the_report(self):
        estimate = self._estimate(
            "rule4ml_hybrid_softmax_v1", register_rule4ml_hybrid_evaluators, 2
        )
        self.assertEqual(estimate.chiplet_use[0].kernel["BRAM"], 95.0)


def _two_fpga_architecture():
    with ARCHITECTURE.open(encoding="utf-8") as file:
        architecture = json.load(file)
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


class GraphResourceCheckTests(unittest.TestCase):
    """Three kernels of 95 BRAM each: any one fits a 280-BRAM chiplet."""

    def setUp(self):
        self.architecture = _two_fpga_architecture()
        self.graph = _graph()
        self.profile = load_evaluation_profile(PROFILE)

    def _mapping(self, placement):
        mapping = initial_workload_mapping(self.graph, self.architecture, self.profile)
        for index, operation in enumerate(mapping.operations):
            mapping = mapping.with_operation(
                index, endpoint_ids=placement[operation.operation_id]
            )
        return mapping

    def _evaluate(self, placement):
        with _prediction():
            return evaluate_atlas_graph(
                None,
                self.architecture,
                self.graph,
                profile=self.profile,
                mapping=self._mapping(placement),
            )

    def test_three_kernels_on_one_chiplet_do_not_fit(self):
        # 3 x 95 BRAM + 1 buffer block = 286 > 280.
        with self.assertRaisesRegex(ResourceExceeded, "BRAM 286 > 280"):
            self._evaluate({"tanh_1": (1,), "sigmoid_1": (1,), "softmax_1": (1,)})

    def test_a_second_chiplet_takes_the_third_kernel(self):
        evaluation = self._evaluate(
            {"tanh_1": (1,), "sigmoid_1": (1,), "softmax_1": (2,)}
        )
        totals = evaluation.chiplet_resources
        # The two kernels on chiplet 1 share one input buffer block.
        self.assertEqual(totals[1]["BRAM"], 95 * 2 + 1)
        self.assertEqual(totals[2]["BRAM"], 95 + 1)

    def test_the_shares_of_a_split_operation_count_on_their_own_chiplets(self):
        evaluation = self._evaluate(
            {"tanh_1": (1, 2), "sigmoid_1": (1,), "softmax_1": (2,)}
        )
        totals = evaluation.chiplet_resources
        self.assertEqual(totals[1]["BRAM"], 95 * 2 + 1)
        self.assertEqual(totals[2]["BRAM"], 95 * 2 + 1)

    def test_a_split_does_not_hide_an_overfull_chiplet(self):
        with self.assertRaises(ResourceExceeded):
            self._evaluate(
                {"tanh_1": (1, 2), "sigmoid_1": (1,), "softmax_1": (1,)}
            )


if __name__ == "__main__":
    unittest.main()
