import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from scalesim.scale_sim import scalesim

from system.utils.GEMMWorkload import GEMMWorkload
from system.utils.Scheduler import Scheduler
from system.utils.SimulationCache import SIMULATION_MODEL_VERSION, SimulationCache
from system.utils.Simulator import Simulator
from system.utils.SystolicArray import SystolicArray


def make_core():
    return SystolicArray(
        width=64,
        height=64,
        buffer_size=256,
        id=0,
        power=1.0,
        area=1.0,
        sram_energy_scale=1.0,
        dram_energy_scale=2.0,
        bandwidth=1,
        frequency=1e9,
    )


class SimulationCacheTests(unittest.TestCase):
    def test_legacy_cache_entries_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.csv"
            pd.DataFrame(
                [
                    {
                        "core_size": 64,
                        "data_flow": "ws",
                        "bandwidth": 1,
                        "buffer_size": 256,
                        "M": 32,
                        "K": 32,
                        "N": 32,
                        "latency": 999,
                    }
                ]
            ).to_csv(cache_path, index=False)
            cache = SimulationCache(cache_path)
            core = make_core()
            core.assign_workload(GEMMWorkload(32, 32, 32))

            def fake_simulation(_simulator, simulated_core):
                simulated_core.cycle_per_layer = [10]
                return 10

            with patch.object(Simulator, "simulate_single_core", fake_simulation):
                cache.all_cores_simulation_with_cache([core])

            self.assertEqual(core.total_cycle, 10)
            self.assertIn(SIMULATION_MODEL_VERSION, cache.cache_df.index.levels[0])

    def test_duplicate_cache_misses_are_charged_for_every_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.csv"
            pd.DataFrame(
                columns=[
                    "core_size",
                    "data_flow",
                    "bandwidth",
                    "buffer_size",
                    "M",
                    "K",
                    "N",
                    "latency",
                ]
            ).to_csv(cache_path, index=False)
            cache = SimulationCache(cache_path)
            core = make_core()
            core.assign_workload(GEMMWorkload(32, 32, 32))
            core.assign_workload(GEMMWorkload(32, 32, 32))

            def fake_simulation(_simulator, simulated_core):
                simulated_core.cycle_per_layer = [10] * len(simulated_core.workloads)
                return sum(simulated_core.cycle_per_layer)

            with patch.object(Simulator, "simulate_single_core", fake_simulation):
                latency_by_core = cache.all_cores_simulation_with_cache([core])

            self.assertEqual(core.cycle_per_layer, [10, 10])
            self.assertEqual(core.total_cycle, 20)
            self.assertEqual(latency_by_core, {0: 20.0})


class SchedulerMemoryTests(unittest.TestCase):
    def test_dram_write_energy_uses_output_word_count(self):
        core = make_core()
        workload = GEMMWorkload(2, 3, 5)
        mapping = {
            "chiplet_data_sharing_enabled": 0,
            "if_splitting_k": 0,
            "dataflow": ["ws"],
            "assign_workload_in_ascending_order": 0,
            "merge_tiles": 0,
        }
        scheduler = Scheduler(workload, core, mapping_dict=mapping)
        core.assign_workload(GEMMWorkload(2, 3, 5))
        system = SimpleNamespace(core_dict={0: core})

        latency_ns, energy_pj = scheduler.system_modeling(system)

        # Reads: 2*3 + 3*5 = 21 words. Writes: 2*5 = 10 words.
        self.assertEqual(latency_ns, 31.0)
        self.assertEqual(energy_pj, (21 + 10) * 8 * core.dram_energy_scale)

    def test_memory_phase_controls_only_remove_replaced_dram_traffic(self):
        core = make_core()
        workload = GEMMWorkload(2, 3, 5)
        mapping = {
            "chiplet_data_sharing_enabled": 0,
            "if_splitting_k": 0,
            "dataflow": ["ws"],
            "assign_workload_in_ascending_order": 0,
            "merge_tiles": 0,
        }
        scheduler = Scheduler(workload, core, mapping_dict=mapping)
        core.assign_workload(GEMMWorkload(2, 3, 5))
        system = SimpleNamespace(core_dict={0: core})

        latency_ns, energy_pj = scheduler.system_modeling(
            system,
            activation_from_dram=False,
            output_to_dram=False,
        )

        # Only the 3*5 weight matrix remains a DRAM transfer.
        self.assertAlmostEqual(latency_ns, 15.0)
        self.assertAlmostEqual(energy_pj, 15 * 8 * core.dram_energy_scale)


class SimulatorConfigurationTests(unittest.TestCase):
    def test_generated_configuration_is_accepted_by_scalesim(self):
        core = make_core()
        core.assign_workload(GEMMWorkload(32, 32, 32))

        with tempfile.TemporaryDirectory() as directory:
            simulator = Simulator(core, home_dir=directory)
            workload_path, _ = simulator.write_core_workload(core)
            config_path, _ = simulator.write_config_file(core)
            layout_path, _ = simulator.write_core_layout(core)

            configured = scalesim(
                config=config_path,
                topology=workload_path,
                layout=layout_path,
                input_type_gemm=True,
                verbose=False,
                save_disk_space=True,
            )

        self.assertTrue(configured.config.valid_conf_flag)


if __name__ == "__main__":
    unittest.main()
