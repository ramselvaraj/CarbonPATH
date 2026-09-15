import unittest

from system.utils.AtlasWorkload import (
    GemmOperation,
    ReluOperation,
    parse_atlas_workload_entry,
)


def _gemm(**overrides):
    base = {
        "operation_id": "gemm_1",
        "operation_type": "gemm",
        "input_tensor_id": "input",
        "weight_tensor_id": "gemm_1:weights",
        "output_tensor_id": "gemm_1:output",
        "input_shape": [8, 64],
        "weight_shape": [64, 64],
        "output_shape": [8, 64],
        "m": 8,
        "k": 64,
        "n": 64,
    }
    base.update(overrides)
    return base


def _relu(**overrides):
    base = {
        "operation_id": "relu_1",
        "operation_type": "relu",
        "input_tensor_id": "gemm_1:output",
        "output_tensor_id": "relu_1:output",
        "input_shape": [8, 64],
        "output_shape": [8, 64],
        "element_count": 512,
    }
    base.update(overrides)
    return base


def _envelope(operations):
    return {
        "format": "atlas-normalized-v0",
        "name": "gemm_relu_gemm",
        "operations": operations,
    }


class AtlasWorkloadTests(unittest.TestCase):
    def test_parses_gemm_relu_gemm_chain(self):
        workload = parse_atlas_workload_entry(
            _envelope(
                [
                    _gemm(),
                    _relu(),
                    _gemm(
                        operation_id="gemm_2",
                        input_tensor_id="relu_1:output",
                        weight_tensor_id="gemm_2:weights",
                        output_tensor_id="gemm_2:output",
                    ),
                ]
            )
        )

        self.assertEqual(workload.name, "gemm_relu_gemm")
        self.assertEqual(workload.dtype, "int8")
        self.assertIsInstance(workload.operations[0], GemmOperation)
        self.assertIsInstance(workload.operations[1], ReluOperation)
        self.assertEqual(workload.operations[1].element_count, 512)

    def test_gemm_shape_product_validation(self):
        with self.assertRaises(ValueError):
            parse_atlas_workload_entry(_envelope([_gemm(m=4)]))
        with self.assertRaises(ValueError):
            parse_atlas_workload_entry(_envelope([_gemm(weight_shape=[8, 8])]))

    def test_relu_element_count_must_match_shape(self):
        with self.assertRaises(ValueError):
            parse_atlas_workload_entry(_envelope([_relu(element_count=500)]))

    def test_relu_shape_must_be_preserving(self):
        with self.assertRaises(ValueError):
            parse_atlas_workload_entry(_envelope([_relu(output_shape=[8, 32])]))

    def test_chain_must_consume_previous_output(self):
        with self.assertRaises(ValueError):
            parse_atlas_workload_entry(
                _envelope([_gemm(), _relu(input_tensor_id="someone_else")])
            )

    def test_duplicate_operation_ids_rejected(self):
        with self.assertRaises(ValueError):
            parse_atlas_workload_entry(
                _envelope([_gemm(), _relu(operation_id="gemm_1")])
            )

    def test_unsupported_operation_type_rejected(self):
        with self.assertRaises(ValueError):
            parse_atlas_workload_entry(
                _envelope([_relu(operation_type="softmax")])
            )

    def test_unsupported_format_rejected(self):
        entry = _envelope([_gemm()])
        entry["format"] = "atlas-normalized-v1"
        with self.assertRaises(ValueError):
            parse_atlas_workload_entry(entry)

    def test_non_int8_rejected(self):
        entry = _envelope([_gemm()])
        entry["dtype"] = "fp16"
        with self.assertRaises(ValueError):
            parse_atlas_workload_entry(entry)

    def test_fingerprint_is_stable(self):
        workload = parse_atlas_workload_entry(_envelope([_gemm()]))
        self.assertEqual(workload.fingerprint(), workload.fingerprint())


if __name__ == "__main__":
    unittest.main()
