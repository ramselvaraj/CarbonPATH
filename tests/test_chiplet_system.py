import copy
import json
import unittest
from pathlib import Path

from system.utils.ChipletSystem import ChipletSystem
from system.utils.GEMMWorkload import GEMMWorkload
from main import build_scheduler_system, collect_tile_mappings


ROOT = Path(__file__).resolve().parents[1]
BASE_ARCHITECTURE = ROOT / "cfg/examples/sa_fpga_architecture.json"


def _base_architecture():
    with BASE_ARCHITECTURE.open(encoding="utf-8") as file:
        return json.load(file)


def _multi_sa_architecture(sa_count, include_fpga=True):
    base = _base_architecture()
    sa_template = base["Chiplet_1"]
    fpga_template = base["Chiplet_2"]
    architecture = {
        "pkg": {
            "HI_pkg_type": "2.5d",
            "inter_pkg_conn": [],
            "protocol_3d": "na",
            "protocol_2.5d": "ucie_std",
            "mem_pkg_conn": {"mem_type": "hbm2"},
        },
        "WL_mapping": copy.deepcopy(base["WL_mapping"]),
    }
    endpoint_keys = []
    for index in range(1, sa_count + 1):
        key = f"Chiplet_{index}"
        chiplet = copy.deepcopy(sa_template)
        chiplet["tech_node"] = "7" if index % 2 else "10"
        chiplet["sys_array_size"] = "64x64" if index % 2 else "128x128"
        chiplet["sram_buf"] = 256 if index % 2 else 512
        chiplet["area"] = 2.0
        architecture[key] = chiplet
        architecture["pkg"]["mem_pkg_conn"][key] = index
        endpoint_keys.append(key)
    if include_fpga:
        key = f"Chiplet_{sa_count + 1}"
        architecture[key] = copy.deepcopy(fpga_template)
        architecture["pkg"]["mem_pkg_conn"][key] = 0
        endpoint_keys.append(key)
    for source, destination in zip(endpoint_keys, endpoint_keys[1:]):
        architecture["pkg"]["inter_pkg_conn"].append(
            {
                "from": source,
                "to": destination,
                "connection_type": "2.5d_emib",
                "loc": "2.5d_chiplet",
            }
        )
    return architecture


class ChipletSystemTests(unittest.TestCase):
    def test_scheduler_supports_sparse_core_ids_from_non_contiguous_labels(self):
        architecture = _multi_sa_architecture(2, include_fpga=False)
        architecture["Chiplet_3"] = architecture.pop("Chiplet_2")
        memory = architecture["pkg"]["mem_pkg_conn"]
        memory["Chiplet_3"] = memory.pop("Chiplet_2")
        for connection in architecture["pkg"]["inter_pkg_conn"]:
            if connection["from"] == "Chiplet_2":
                connection["from"] = "Chiplet_3"
            if connection["to"] == "Chiplet_2":
                connection["to"] = "Chiplet_3"

        scheduler, system = build_scheduler_system(
            GEMMWorkload(512, 512, 512), architecture
        )

        self.assertEqual(set(system.core_dict), {0, 2})
        scheduler.system_modeling(system)

    def test_constructs_every_supported_sa_count(self):
        for sa_count in range(1, 7):
            with self.subTest(sa_count=sa_count):
                system = ChipletSystem(arch_dict=_multi_sa_architecture(sa_count))
                self.assertEqual(len(system.core_dict), sa_count)
                self.assertEqual(
                    system.sa_endpoint_group.endpoint_ids, tuple(range(sa_count))
                )
                self.assertEqual(len(system.fpga_chiplet_dict), 1)

    def test_preserves_heterogeneous_sa_characterization(self):
        system = ChipletSystem(arch_dict=_multi_sa_architecture(2))

        self.assertEqual(system.core_dict[0].width, 64)
        self.assertEqual(system.core_dict[0].node, 7)
        self.assertEqual(system.core_dict[0].buffer_size, 256)
        self.assertEqual(system.core_dict[1].width, 128)
        self.assertEqual(system.core_dict[1].node, 10)
        self.assertEqual(system.core_dict[1].buffer_size, 512)

    def test_builds_routes_across_sa_group_and_fpga(self):
        system = ChipletSystem(arch_dict=_multi_sa_architecture(3))

        path, reciprocal_bandwidth, energy = system.get_shortest_path(0, 3)

        self.assertEqual(path, [0, 1, 2, 3])
        self.assertGreater(reciprocal_bandwidth, 0)
        self.assertGreater(energy, 0)

    def test_supports_sa_only_multi_chiplet_system(self):
        system = ChipletSystem(
            arch_dict=_multi_sa_architecture(4, include_fpga=False)
        )

        self.assertEqual(len(system.systolic_arrays), 4)
        self.assertEqual(system.fpga_chiplets, [])

    def test_multi_sa_scheduler_covers_each_gemm_point_once(self):
        shape = (512, 512, 512)
        for split_k in (False, True):
            with self.subTest(split_k=split_k):
                architecture = _multi_sa_architecture(3, include_fpga=False)
                architecture["WL_mapping"]["mapping"]["if_splitting_k"] = int(
                    split_k
                )

                scheduler, _ = build_scheduler_system(
                    GEMMWorkload(*shape), architecture
                )
                mappings = collect_tile_mappings(scheduler)
                tile_boxes = {
                    (
                        tile["m_offset"],
                        tile["k_offset"],
                        tile["n_offset"],
                        tile["m"],
                        tile["k"],
                        tile["n"],
                    )
                    for tile in mappings
                }

                self.assertEqual(len(tile_boxes), len(mappings))
                self.assertEqual(
                    sum(tile["m"] * tile["k"] * tile["n"] for tile in mappings),
                    shape[0] * shape[1] * shape[2],
                )
                self.assertEqual({tile["core_id"] for tile in mappings}, {0, 1, 2})

    def test_scheduler_applies_every_supported_dataflow(self):
        for dataflow in ("ws", "os", "is"):
            with self.subTest(dataflow=dataflow):
                architecture = _multi_sa_architecture(2, include_fpga=False)
                architecture["WL_mapping"]["mapping"]["dataflow"] = [dataflow]

                scheduler, _ = build_scheduler_system(
                    GEMMWorkload(512, 512, 512), architecture
                )

                self.assertTrue(
                    all(core.data_flow == dataflow for core in scheduler.systolic_arrays)
                )

    def test_split_k_reports_reduction_costs(self):
        architecture = _multi_sa_architecture(3, include_fpga=False)
        architecture["WL_mapping"]["mapping"]["if_splitting_k"] = 1
        scheduler, system = build_scheduler_system(
            GEMMWorkload(512, 512, 512), architecture
        )

        scheduler.system_modeling(system)

        details = scheduler.last_modeling_details
        self.assertGreater(details["reduction_latency_ns"], 0)
        self.assertGreater(details["reduction_transfer_latency_ns"], 0)
        self.assertGreater(details["reduction_communication_energy_pj"], 0)

    def test_non_split_mapping_reports_no_reduction_cost(self):
        architecture = _multi_sa_architecture(3, include_fpga=False)
        architecture["WL_mapping"]["mapping"]["if_splitting_k"] = 0
        scheduler, system = build_scheduler_system(
            GEMMWorkload(512, 512, 512), architecture
        )

        scheduler.system_modeling(system)

        details = scheduler.last_modeling_details
        self.assertEqual(details["reduction_latency_ns"], 0)
        self.assertEqual(details["reduction_transfer_latency_ns"], 0)
        self.assertEqual(details["reduction_communication_energy_pj"], 0)


if __name__ == "__main__":
    unittest.main()
