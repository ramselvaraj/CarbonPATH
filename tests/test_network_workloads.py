import unittest

from system.utils.NetworkWorkload import parse_network_entry


class NetworkWorkloadTests(unittest.TestCase):
    def test_compiles_linear_layers_from_input_and_output_features(self):
        network = parse_network_entry(
            {
                "schema_version": 1,
                "name": "mlp_example",
                "dtype": "int8",
                "input": {"batch_size": 128, "features": 768},
                "layers": [
                    {"name": "fc1", "op": "linear", "out_features": 3072},
                    {"name": "fc2", "op": "linear", "out_features": 768},
                    {
                        "name": "classifier",
                        "op": "linear",
                        "out_features": 1000,
                    },
                ],
                "memory": {"intermediate_policy": "local_sram"},
            }
        )

        self.assertEqual(network.name, "mlp_example")
        self.assertEqual(network.dtype, "int8")
        self.assertEqual(network.intermediate_policy, "local_sram")
        self.assertEqual(
            network.to_workload_sequence()["gemms"],
            [
                {"name": "fc1", "shape": (128, 768, 3072)},
                {"name": "fc2", "shape": (128, 3072, 768)},
                {"name": "classifier", "shape": (128, 768, 1000)},
            ],
        )

    def test_four_layer_network_compiles_to_workload_8_shapes(self):
        network = parse_network_entry(
            {
                "schema_version": 1,
                "name": "four_gemm_network",
                "dtype": "int8",
                "input": {"batch_size": 128, "features": 128},
                "layers": [
                    {"name": f"block_{index}", "op": "linear", "out_features": 128}
                    for index in range(1, 5)
                ],
            }
        )

        self.assertEqual(
            [layer["shape"] for layer in network.to_workload_sequence()["gemms"]],
            [(128, 128, 128)] * 4,
        )
        self.assertEqual(network.intermediate_policy, "direct_forward")

    def test_rejects_unsupported_or_invalid_network_fields(self):
        invalid_networks = [
            {
                "schema_version": 2,
                "name": "bad_schema",
                "dtype": "int8",
                "input": {"batch_size": 1, "features": 8},
                "layers": [{"name": "fc", "op": "linear", "out_features": 8}],
            },
            {
                "schema_version": 1,
                "name": "bad_dtype",
                "dtype": "fp16",
                "input": {"batch_size": 1, "features": 8},
                "layers": [{"name": "fc", "op": "linear", "out_features": 8}],
            },
            {
                "schema_version": 1,
                "name": "bad_op",
                "dtype": "int8",
                "input": {"batch_size": 1, "features": 8},
                "layers": [{"name": "relu", "op": "relu", "out_features": 8}],
            },
            {
                "schema_version": 1,
                "name": "duplicate_names",
                "dtype": "int8",
                "input": {"batch_size": 1, "features": 8},
                "layers": [
                    {"name": "fc", "op": "linear", "out_features": 8},
                    {"name": "fc", "op": "linear", "out_features": 8},
                ],
            },
            {
                "schema_version": 1,
                "name": "invalid_dimension",
                "dtype": "int8",
                "input": {"batch_size": 0, "features": 8},
                "layers": [{"name": "fc", "op": "linear", "out_features": 8}],
            },
            {
                "schema_version": 1,
                "name": "../../outside",
                "dtype": "int8",
                "input": {"batch_size": 1, "features": 8},
                "layers": [{"name": "fc", "op": "linear", "out_features": 8}],
            },
            {
                "schema_version": 1,
                "name": "C:outside",
                "dtype": "int8",
                "input": {"batch_size": 1, "features": 8},
                "layers": [{"name": "fc", "op": "linear", "out_features": 8}],
            },
        ]

        for network in invalid_networks:
            with self.subTest(network=network["name"]):
                with self.assertRaises(ValueError):
                    parse_network_entry(network)

        unsupported_activation = {
            "schema_version": 1,
            "name": "unsupported_activation",
            "dtype": "int8",
            "input": {"batch_size": 8, "features": 16},
            "layers": [
                {
                    "name": "fc",
                    "op": "linear",
                    "out_features": 32,
                    "activation": "relu",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "Unsupported layer 1 field"):
            parse_network_entry(unsupported_activation)

    def test_network_fingerprint_tracks_topology_not_policy(self):
        base = {
            "schema_version": 1,
            "name": "fingerprint_network",
            "dtype": "int8",
            "input": {"batch_size": 8, "features": 16},
            "layers": [{"name": "fc", "op": "linear", "out_features": 32}],
        }
        direct = parse_network_entry(base)
        local = parse_network_entry(
            {**base, "memory": {"intermediate_policy": "local_sram"}}
        )

        self.assertEqual(direct.fingerprint(), local.fingerprint())
        self.assertNotEqual(
            direct.fingerprint({"sys_array": ["64x64"]}),
            direct.fingerprint({"sys_array": ["128x128"]}),
        )


if __name__ == "__main__":
    unittest.main()
