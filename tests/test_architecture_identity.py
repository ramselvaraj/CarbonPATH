import unittest

from system.utils.ArchitectureIdentity import (
    architecture_fingerprint,
    canonical_architecture_fingerprint,
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


if __name__ == "__main__":
    unittest.main()
