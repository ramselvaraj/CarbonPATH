import unittest

from system.utils.ArchitectureIdentity import (
    architecture_fingerprint,
    canonical_architecture_fingerprint,
    compatible_result_fingerprint,
    effective_gemm_mapping,
    physical_architecture_fingerprint,
)


class ArchitectureIdentityTests(unittest.TestCase):
    def test_fingerprint_ignores_dictionary_and_interconnect_order(self):
        first = {
            "Chiplet_1": {"tech_node": "7", "sys_array_size": "64x64"},
            "pkg": {
                "inter_pkg_conn": [
                    {"from": "Chiplet_2", "to": "Chiplet_1"},
                    {"from": "Chiplet_1", "to": "Chiplet_2"},
                ],
                "HI_pkg_type": "2.5d",
            },
        }
        second = {
            "pkg": {
                "HI_pkg_type": "2.5d",
                "inter_pkg_conn": list(reversed(first["pkg"]["inter_pkg_conn"])),
            },
            "Chiplet_1": {"sys_array_size": "64x64", "tech_node": "7"},
        }

        self.assertEqual(
            architecture_fingerprint(first), architecture_fingerprint(second)
        )

    def test_canonical_fingerprint_accepts_scalar_interconnect_marker(self):
        architecture = {
            "Chiplet_1": {"area": 1},
            "pkg": {"inter_pkg_conn": "2d_na", "mem_pkg_conn": {}},
        }
        self.assertIsInstance(canonical_architecture_fingerprint(architecture), str)

    def test_fingerprint_changes_with_architecture(self):
        first = {"Chiplet_1": {"sys_array_size": "64x64"}}
        second = {"Chiplet_1": {"sys_array_size": "128x128"}}

        self.assertNotEqual(
            architecture_fingerprint(first), architecture_fingerprint(second)
        )

    def test_canonical_fingerprint_ignores_chiplet_labels(self):
        first = {
            "Chiplet_1": {"sys_array_size": "128x128"},
            "Chiplet_2": {"sys_array_size": "64x64"},
            "pkg": {
                "inter_pkg_conn": [
                    {"from": "Chiplet_1", "to": "Chiplet_2"},
                    {"from": "Chiplet_2", "to": "na"},
                ],
                "mem_pkg_conn": {"Chiplet_1": 12, "Chiplet_2": 4},
            },
        }
        second = {
            "Chiplet_1": {"sys_array_size": "64x64"},
            "Chiplet_2": {"sys_array_size": "128x128"},
            "pkg": {
                "inter_pkg_conn": [
                    {"from": "Chiplet_2", "to": "Chiplet_1"},
                    {"from": "Chiplet_1", "to": "na"},
                ],
                "mem_pkg_conn": {"Chiplet_1": 4, "Chiplet_2": 12},
            },
        }
        self.assertEqual(
            canonical_architecture_fingerprint(first),
            canonical_architecture_fingerprint(second),
        )

    def test_canonical_fingerprint_preserves_memory_allocation(self):
        first = {"Chiplet_1": {}, "pkg": {"mem_pkg_conn": {"Chiplet_1": 4}}}
        second = {"Chiplet_1": {}, "pkg": {"mem_pkg_conn": {"Chiplet_1": 8}}}
        self.assertNotEqual(
            canonical_architecture_fingerprint(first),
            canonical_architecture_fingerprint(second),
        )

    def test_canonical_fingerprint_ignores_labels_in_symmetric_25d_package(self):
        first = {
            "Chiplet_1": {"sys_array_size": "128x128", "tech_node": "7"},
            "Chiplet_2": {"sys_array_size": "64x64", "tech_node": "10"},
            "pkg": {
                "HI_pkg_type": "2.5d",
                "inter_pkg_conn": [
                    {"from": "Chiplet_1", "to": "Chiplet_2", "loc": "2.5d_chiplet"},
                    {"from": "Chiplet_2", "to": "Chiplet_1", "loc": "2.5d_chiplet"},
                ],
                "mem_pkg_conn": {
                    "mem_type": "hbm3",
                    "Chiplet_1": 12,
                    "Chiplet_2": 4,
                },
            },
        }
        relabeled = {
            "Chiplet_1": first["Chiplet_2"],
            "Chiplet_2": first["Chiplet_1"],
            "pkg": {
                "HI_pkg_type": "2.5d",
                "inter_pkg_conn": [
                    {"from": "Chiplet_2", "to": "Chiplet_1", "loc": "2.5d_chiplet"},
                    {"from": "Chiplet_1", "to": "Chiplet_2", "loc": "2.5d_chiplet"},
                ],
                "mem_pkg_conn": {
                    "mem_type": "hbm3",
                    "Chiplet_1": 4,
                    "Chiplet_2": 12,
                },
            },
        }

        self.assertEqual(
            canonical_architecture_fingerprint(first),
            canonical_architecture_fingerprint(relabeled),
        )

    def test_physical_fingerprint_ignores_workload_mapping_only(self):
        first = {
            "Chiplet_1": {"sys_array_size": "64x64"},
            "WL_mapping": {"mapping": {"dataflow": ["ws"]}},
        }
        second = {
            "Chiplet_1": {"sys_array_size": "64x64"},
            "WL_mapping": {"mapping": {"dataflow": ["os"]}},
        }
        changed_hardware = {
            "Chiplet_1": {"sys_array_size": "128x128"},
            "WL_mapping": {"mapping": {"dataflow": ["ws"]}},
        }

        self.assertEqual(
            physical_architecture_fingerprint(first),
            physical_architecture_fingerprint(second),
        )
        self.assertNotEqual(
            physical_architecture_fingerprint(first),
            physical_architecture_fingerprint(changed_hardware),
        )

    def test_profile_gemm_settings_define_effective_mapping(self):
        architecture = {
            "Chiplet_1": {"sys_array_size": "64x64"},
            "WL_mapping": {
                "mapping": {
                    "dataflow": ["ws"],
                    "if_splitting_k": 0,
                    "assign_workload_in_ascending_order": 0,
                }
            },
        }
        profile = {
            "evaluators": {
                "gemm": {
                    "id": "legacy_scale_sim_gemm_v1",
                    "settings": {
                        "dataflow": "os",
                        "split_k": True,
                        "assignment_order": "ascending",
                    },
                }
            }
        }

        self.assertEqual(
            effective_gemm_mapping(architecture, profile),
            {
                "dataflow": "os",
                "split_k": True,
                "assignment_order": "ascending",
            },
        )

    def test_compatible_result_matches_legacy_and_modular_mapping_storage(self):
        legacy = {
            "Chiplet_1": {"sys_array_size": "64x64"},
            "WL_mapping": {
                "mapping": {
                    "dataflow": ["os"],
                    "if_splitting_k": 1,
                    "assign_workload_in_ascending_order": 1,
                }
            },
        }
        modular = {
            "Chiplet_1": {"sys_array_size": "64x64"},
            "WL_mapping": {
                "mapping": {
                    "dataflow": ["ws"],
                    "if_splitting_k": 0,
                    "assign_workload_in_ascending_order": 0,
                }
            },
        }
        profile = {
            "evaluators": {
                "gemm": {
                    "id": "legacy_scale_sim_gemm_v1",
                    "settings": {
                        "dataflow": "os",
                        "split_k": True,
                        "assignment_order": "ascending",
                    },
                }
            }
        }

        self.assertEqual(
            compatible_result_fingerprint(legacy),
            compatible_result_fingerprint(modular, profile),
        )


if __name__ == "__main__":
    unittest.main()
