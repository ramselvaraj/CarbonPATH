from system.utils.Simulator import Simulator
from system.utils.SystolicArray import SystolicArray
from system.utils.GEMMWorkload import GEMMWorkload
from config import print_info

import pandas as pd
import numpy as np


from pathlib import Path


SIMULATION_MODEL_VERSION = 2

# Find project root by walking up until we see the sentinel file/folder, e.g. `.git` or `cfg`
def find_repo_root(start: Path, sentinel: str = "cfg") -> Path:
    for parent in [start, *start.parents]:
        if (parent / sentinel).exists():
            return parent
    raise FileNotFoundError(f"Could not locate '{sentinel}' upward from {start}")



class SimulationCache:
    def __init__(self, path = None, fast_test = False, simulator_dir = None):
        if path == None:
            repo_root = find_repo_root(Path.cwd())
            self.dir = repo_root / "cfg" / "static_cache" / "static_cache.csv"
        else:
            self.dir = path
        self.dtype_dict = {
            'core_size': int,
            'M':int,
            'K':int,
            'N':int,
            'buffer_size':int,
            'bandwidth': int,
            'data_flow': str,
            'latency': int
        }

        if simulator_dir is None:
            self.simulator_dir = "simulation"
        else:
            self.simulator_dir = simulator_dir

        cache = pd.read_csv(self.dir, dtype=self.dtype_dict)
        if 'model_version' not in cache:
            cache['model_version'] = 1
        cache['model_version'] = cache['model_version'].astype(int)
        self.index_cols = ['model_version', 'core_size', 'data_flow', 'bandwidth', 'buffer_size', 'M', 'K', 'N']
        cache.set_index(self.index_cols, inplace=True)

        # deduplicate
        # mean_latency = cache.groupby(level=self.index_cols)['latency'].mean()
        deduped = cache[~cache.index.duplicated(keep='last')].copy() # always keep the latest results
        # deduped['latency'] = mean_latency.astype(np.int32)
        dups_remaining = deduped.index.duplicated().sum()
        assert dups_remaining == 0, f"[ERROR] Duplicates are still remaining"

        # deduped cache initialization
        self.cache_df = deduped
        self.fast_test = fast_test
    
    def __single_core_latency_lookup(self, core: SystolicArray):
        """
        latency lookup for all WLs of input core.
        1. Return the uncached workloads for scalesim simulation
        2. Read the cached cycles, assign 0 for uncached GEMM shape 
        """
        # default gemm sizes are pre cached for every size
        # cached_gemm wll never be None/empty 
        assert len(core.workloads) > 0, "No workloads attached to this core"

        # Make tuple keys for each workload
        lookup_keys = [
            (SIMULATION_MODEL_VERSION, core.width, core.data_flow, core.dram_bandwidth, core.buffer_size, wl.m, wl.k, wl.n)
            for wl in core.workloads
        ]
        
        res_df = self.cache_df.reindex(lookup_keys)

        # Reset index so columns are back
        res_df = res_df.reset_index()

        # Fill missing latency
        res_df['latency'] = res_df['latency'].fillna(0).astype(int)
        # If all were misses, create a dummy row to avoid empty errors
        if res_df['latency'].empty:
            res_df = pd.DataFrame({'latency': [0]})

        core.cycle_per_layer = res_df['latency'].tolist()
        # lookup cached results
        core.total_cycle = sum(core.cycle_per_layer)

        # Identify which workloads need simulation
        to_sim_mask = (res_df['latency'] == 0)
        
        if not to_sim_mask.any(): # all Masks are false, all hit, no simulation required
            return core.total_cycle
        
        wl_to_simulate = [(i, wl) for i, (wl, need) in enumerate(zip(core.workloads, to_sim_mask)) if need]

        self.__single_core_simulation_with_cache(core, wl_to_simulate)
        core.total_cycle = sum(core.cycle_per_layer)
        return core.total_cycle
    

    def __single_core_simulation_with_cache(self, core: SystolicArray, wl_to_simulate: list[(int, GEMMWorkload)]):
        """
        launch simulation for un-simulated GEMMs
        """
        buffer_core = SystolicArray.init_from_other_instance(core)

        # only accept the unique GEMMWorkload, drop the duplicated GEMM  
        unique_workload = dict()
        for idx, wl in wl_to_simulate:
            key = wl.shape()
            if key not in unique_workload:
                unique_workload[key] = []
            unique_workload[key].append(idx)


        buffer_core.workloads = [GEMMWorkload(key[0], key[1], key[2]) for key in unique_workload.keys()]
        # launch simulation for the buffer core
        simulator = Simulator(buffer_core,home_dir=self.simulator_dir,fast_test=self.fast_test, mode="USER")
    
        shape = []
        for wl in buffer_core.workloads:
            shape.append(wl.shape())
        print(f"[INFO] Running simulator .......")
        if print_info:
            print(f"[Simulation] Launch Simulation for Core[{buffer_core.id}] {buffer_core.width} x {buffer_core.height}")
            print(
            f"    GEMM: {shape} dataflow:{buffer_core.data_flow} mode:{simulator.mode}\n"
            f"    SRAM:{buffer_core.buffer_size}KB DRAM BW:{buffer_core.dram_bandwidth} bytes/cycle\n"
            f"-------\n"
            )
        total_latency = simulator.simulate_single_core(buffer_core)
        
        if not self.fast_test:
            self.__update_cache(buffer_core)
        count = 0
        for indices, cycle in zip(unique_workload.values(), buffer_core.cycle_per_layer):
            assert cycle is not None and cycle > 0, f"[ERROR] Cycle is {cycle} at buffer_core for GEMM {wl}"
            for idx in indices:
                count += 1
                core.cycle_per_layer[idx] = int(cycle)

        core.simulation_reporter = buffer_core.simulation_reporter
        assert count == sum([len(idx) for idx in unique_workload.values()]), \
            f"[ERROR] num of cycle assignment should equal to num of assigned GEMMs, {len(wl_to_simulate)} GEMMs assigned, but only {count} cycle assignment performed"

        return sum(core.cycle_per_layer)

    def __update_cache(self, core: SystolicArray) -> None:
        """
        Update the cache with newly updated results
        """
        wl_shapes = [(wl.m, wl.k, wl.n) for wl in core.workloads]
        holder = pd.DataFrame(wl_shapes, columns=['M', 'K', 'N'])
        holder['core_size'] = [core.width] * len(wl_shapes)
        holder['data_flow'] = [core.data_flow] * len(wl_shapes)
        holder['latency'] = core.cycle_per_layer
        holder['buffer_size'] = [core.buffer_size] * len(wl_shapes)
        holder['bandwidth'] = [core.dram_bandwidth] * len(wl_shapes)
        holder['model_version'] = [SIMULATION_MODEL_VERSION] * len(wl_shapes)

        holder = holder.drop_duplicates()
        
        no_simulate_mask = holder['latency'] == 0
        if no_simulate_mask.any():
            no_simulate_df = holder[no_simulate_mask]
            shape = list(zip(no_simulate_df['M'], no_simulate_df['K'], no_simulate_df['N']))
            raise AssertionError(f"[ERROR] Found 0-cycle GEMMs: {shape}. Please run simulation for these GEMMs")


        # Align to Index
        holder.set_index(self.cache_df.index.names, inplace=True)

        if print_info:
            print(f"[INFO] Existing cache size: {self.cache_df.shape}")
            new_rows = holder[~holder.index.isin(self.cache_df.index)]
            print(f"[INFO] New rows being added: {new_rows.shape}")
            print(new_rows)

        # Update the cache
        self.cache_df = pd.concat([self.cache_df, holder])
        self.cache_df = self.cache_df[~self.cache_df.index.duplicated(keep='last')]

        if print_info:
            print(f"[INFO] Cache size after update: {self.cache_df.shape}")


    def all_cores_simulation_with_cache(self, cores: list[SystolicArray]):
        """
        Lookup cache for all in put systolic cores
        """
        latency = dict()
        # print(f"All Core simulation with Cache\n")
        for core in cores:
            if len(core.workloads) > 0:
                cycles = self.__single_core_latency_lookup(core)
                nano_sec = cycles / core.frequency * 10**9
                latency[core.id] = nano_sec
                # print(f"{core}\n    total {cycles} cycles {nano_sec:.3f} ns\n")

            else:
                latency[core.id] = 0
        return latency


    def dump_cache(self):
        """
        Dump the cache to csv file
        """
        if self.fast_test:
            return
        if print_info:
            print(f"Updating Cache :{self.dir}")
        self.cache_df.reset_index(inplace=True)
        self.cache_df = self.cache_df.astype(self.dtype_dict)
        self.cache_df.to_csv(self.dir, index=False)
        
