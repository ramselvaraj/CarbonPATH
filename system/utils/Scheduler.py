'''
Scheduler to perform workload partition based on given criteria
'''
from __future__ import annotations

from system.utils.GEMMWorkload import GEMMWorkload
from system.utils.SystolicArray import SystolicArray
from system.utils.ChipletSystem import ChipletSystem
import json
import numpy as np
from pathlib import Path
from config import  print_info


WORD_SIZE = 1 # assume int8 data format

DRAM_TRANSFER = 0 # words transferred between chiplets and DRAMs
CHIP2CHIP_TRANSFER = 0 # words transferred between chiplets


with open('cfg/parameters/sram_energy.json', 'r') as f:
    sram_data = json.load(f)
SRAM_ENERGY_SCALE = sram_data.get("SRAM_ENERGY_SCALE", {})
SRAM_ENERGY_LOOKUP = sram_data.get("SRAM_ENERGY_LOOKUP",{})
SRAM_ENERGY_LOOKUP = {int(k): v for k, v in SRAM_ENERGY_LOOKUP.items()}


def _rounding_with_correction(total_tiles: int, ratio: list[float]) -> list[int]:
    raw_allocation = total_tiles * ratio
    int_allocation = np.floor(raw_allocation).astype(int)
    remaining = total_tiles - sum(int_allocation)
    residuals = raw_allocation - int_allocation
    indices = np.argsort(residuals)[::-1]
    for i in range(remaining):
        int_allocation[indices[i]] += 1

    return int_allocation


class Scheduler:
    def __init__(self, workloads: list[GEMMWorkload] | GEMMWorkload, 
                 cores: list[SystolicArray] | SystolicArray, 
                 specs_json: str = None, mapping_dict: dict = None):
        
        assert specs_json != None or mapping_dict != None, f"[ERROR]: Spec Json File and Mapping specs cannot be both None!!!\n"

        if isinstance(workloads, GEMMWorkload):
            self.M = workloads.m
            self.K = workloads.k
            self.N = workloads.n
            workloads = [workloads]
        if isinstance(cores, SystolicArray):
            cores = [cores]
        if mapping_dict:
            mapping = mapping_dict
        else:
            json_path = Path(specs_json)
            with json_path.open("r", encoding='utf-8') as fp:
                mapping:dict = json.load(fp)['WL_mapping']["mapping"]
        


        # Accept mapping configurations
        self.data_sharing = True if mapping["chiplet_data_sharing_enabled"] == 1 else False
        self.splitting_k = True if mapping["if_splitting_k"] == 1 else False
        self.data_flow = mapping["dataflow"][0]
        self.ascending_assign = True if mapping["assign_workload_in_ascending_order"] == 1 else False
        # self.is_static_tiling = True if mapping["static_tiling"] == 1 else False
        self.merge_tiles = True if mapping['merge_tiles'] == 1 else False        

        self.systolic_arrays: list[SystolicArray] = sorted(cores, reverse= self.ascending_assign)
        # update core dataflow
        for core in self.systolic_arrays:
            core.data_flow = self.data_flow

        # Sort the workloads and core in nature order.
        self.workloads: list[GEMMWorkload] = sorted(workloads)

        self.num_cores = len(self.systolic_arrays)
        self.tile_m_min = max(SystolicArray.iter_get_width(self.systolic_arrays))
        self.tile_n_min = max(SystolicArray.iter_get_height(self.systolic_arrays))


    def _assign_workload_to_core(self, systolic_array: SystolicArray, 
                                 workload: GEMMWorkload, force = False) -> bool:
        assert not workload.assigned, f'Current GEMM Workload is already assigned'
        assert workload.assigned_SA is None, f' Current GEMM workload is previously assigned to Core[{workload.assigned_SA.id}]'
        if not systolic_array.is_full() or force:
            workload.assigned = True
            workload.assigned_SA = systolic_array
            systolic_array.assign_workload(workload)
            return True
        else:
            return False
    
    def iter_workloads(self):
        for wl in self.workloads:
            yield wl

    def iter_cores(self):
        for sa in self.systolic_arrays:
            yield sa
    
    @staticmethod
    def tiles_to_cores(tiles: list[GEMMWorkload]) -> set[SystolicArray]:
        core_set = set()
        for tile in tiles:
            core_set.add(tile.assigned_SA)

        return core_set
    

    def __partition_wl(self, wl: GEMMWorkload, partition_K = True) -> np.ndarray:
        m_min = self.tile_m_min
        n_min = self.tile_n_min
        k_min = self.tile_m_min if partition_K else wl.k
        # Guard against len(arrays) > wl.k

        # k_min = (wl.k // len(self.systolic_arrays)) if partition_K else wl.k

        def build_sizes(total, base):
            if total <= base:
                return [total]
            q, r = divmod(total, base)

            total = q * base + r
            return [base] * (q - 1) + ([base + r] if r else [base])
        
        m_sizes = build_sizes(wl.m, m_min)
        k_sizes = build_sizes(wl.k, k_min)
        n_sizes = build_sizes(wl.n, n_min)

        m_tiles = len(m_sizes)
        k_tiles = len(k_sizes)
        n_tiles = len(n_sizes)

        workloads = np.empty((m_tiles, k_tiles, n_tiles), dtype=GEMMWorkload)
        for mi in range(m_tiles):
            m_offset = sum(m_sizes[:mi])
            for ki in range(k_tiles):
                k_offset = sum(k_sizes[:ki])
                for ni in range(n_tiles):
                    n_offset = sum(n_sizes[:ni])
                    workloads[mi, ki, ni] = GEMMWorkload(m_sizes[mi], k_sizes[ki], n_sizes[ni],
                                                         m_offset, n_offset, k_offset)

        return workloads

    def static_workload_scheduling(self):
        """
        Statically partition both M, K and N dimensions
        and allocate workloads to each core.
        M, N are tiled based on minimum tile size.
        K is tiled according to number of chiplets.
        """
        if not self.workloads or not self.systolic_arrays:
            raise ValueError('Current Scheduler has no valid workloads list and systolic array cores')
        
        workload_to_assign: list[GEMMWorkload] = self.workloads # shallow copy
        compute_power = np.array(list(SystolicArray.iter_compute_power(self.systolic_arrays)))
        compute_power_ratio = compute_power / np.sum(compute_power)

        self.partitioned_tiles = None

        while workload_to_assign:
            current_wl = workload_to_assign.pop()
            self.partitioned_tiles = self.__partition_wl(current_wl, partition_K=self.splitting_k)
            flat_tiles = self.partitioned_tiles.ravel()
            tile_allocation = _rounding_with_correction(len(flat_tiles), compute_power_ratio)
            cursor = 0
            for idx, num in enumerate(tile_allocation):
                self.systolic_arrays[idx].increase_tile_capacity(num)
                for _ in range(num):
                    mi, ki, ni = np.unravel_index(cursor, self.partitioned_tiles.shape)
                    self._assign_workload_to_core(self.systolic_arrays[idx], 
                                                 self.partitioned_tiles[mi, ki, ni], 
                                                 force=True)
                    cursor += 1
        
        if self.merge_tiles:
            for core in self.systolic_arrays:
                merged_workloads = Scheduler._merge_tiles(core.workloads)
                assert(merged_workloads != None), f"Merged Tiles returned None"
                core.workloads = merged_workloads

        assert(all(w.assigned for core in self.systolic_arrays for w in core.workloads )),f"[ERROR]: Statically 3D Schedule finished with unassigned workloads"

    def __calculate_dram_latency_energy(
        self,
        core: SystolicArray,
        activation_from_dram=True,
        output_to_dram=True,
    ):
        if len(core.workloads) > 0:
            local_activation_words = (
                sum(wl.m * wl.k for wl in core.workloads)
                if activation_from_dram
                else 0
            )
            local_weight_words = sum(wl.k * wl.n for wl in core.workloads)
            local_load_words = local_activation_words + local_weight_words
            dram_load_cycles =  local_load_words / core.dram_bandwidth # latency in cycle
            dram_load_ns = dram_load_cycles / core.frequency * 10**9 # latency in ns
            #energy calculation
            dram_load_energy = local_load_words * core.dram_energy_scale * WORD_SIZE * 8 # in pj
            
            # calculate the DRAM write latency and energy
            if not self.splitting_k and output_to_dram:
                # no splitting k, every core has to write back the result to DRAM
                local_write_words = sum(wl.m * wl.n for wl in core.workloads)
                dram_write_cycles = local_write_words / core.dram_bandwidth # latency in cycles
                dram_write_ns = dram_write_cycles / core.frequency * 10**9 # latency in ns
                dram_write_energy = local_write_words * core.dram_energy_scale * WORD_SIZE * 8 # in pj
            else:
                dram_write_ns = 0
                dram_write_energy = 0
          

            return dram_load_ns + dram_write_ns, dram_load_energy + dram_write_energy
        else:
            return 0, 0
        
    def __get_hops_in_path(self, system:ChipletSystem, src_id, dst_id):
        if src_id == dst_id:
            return 0
        path, effective_bw_reciprocal, effective_energy_scale = system.get_shortest_path(src_id, dst_id)
        for id in path:
            core = system.core_dict[id]
            if core.location == "chiplet":
                print(f"Current core is a 2.5D chiplet")
            else:
                print(f"Current core is a 3D chipelt location {core.location}")
        hops = max(0, len(path) - 2)
        return hops


    def __calculate_interconnect_latency_energy(self, system: ChipletSystem, 
                                                src_core: SystolicArray, 
                                                dst_core: SystolicArray) -> tuple[float, float]:
        #modeling the interconnect overhead from src core to dest core
        global CHIP2CHIP_TRANSFER
        if len(src_core.workloads) > 0:
            path, effective_bw_reciprocal, effective_energy_scale = system.get_shortest_path(src_core.id, dst_core.id)
            words_transfer = sum(wl.m * wl.n for wl in src_core.workloads)

            CHIP2CHIP_TRANSFER += words_transfer

            bytes_transfer = words_transfer * WORD_SIZE


            latency_ns = bytes_transfer * effective_bw_reciprocal # latency in ns
            actual_bw = 1 / effective_bw_reciprocal if effective_bw_reciprocal != 0 else float('inf')
            energy_pj = bytes_transfer * 8 * effective_energy_scale

            assert latency_ns != float('inf'), print(f"Latency_ns is INF: effective_bw_reciprocal {effective_bw_reciprocal}, effective_energy_scale {effective_energy_scale}, path {path}")


            return latency_ns, energy_pj
        else:
            return 0,0

    def __compute_reduction_latency_ns(self, cores: list[SystolicArray], max_core: SystolicArray) -> int:
        output_tile_group = dict()
        # aggregate all output tiles and their corresponding input tiles
        for core in cores:
            for wl in core.workloads:
                tile_key = (wl.m, wl.n, wl.m_offset, wl.n_offset)
                output_tile_group.setdefault(tile_key, []).append(wl)
        # count reductions for each output tile
        total_reduction_macs = 0
        for key, tile_group in output_tile_group.items():
            num_tiles = len(list(tile_group))

            if num_tiles > 1:
                # reduce whenever 2 or more input tiles are contributing to current output tiles
                m, n = key[0], key[1] 
                reduction_ops = m * n * (num_tiles - 1)
                total_reduction_macs += reduction_ops

        # compute the reduction latency
        cycles = int(total_reduction_macs / (max_core.width * max_core.height)) + 1
        latency_ns = cycles / max_core.frequency * 10**9
        return latency_ns

    def _merge_tiles(workloads: list[GEMMWorkload]) -> list[GEMMWorkload]:
        """
        merge input tiles to bigger workloads
        1. Align workloads based on K dimension
        2. Horizontal Merge: same m_offset and m but n is adjacent
        3. Vertical Merge: same n_offset and n but m is adjacent
        offset is the starting coordinates of current GEMMWorkload
        """
        if not workloads:
            return []
        if len(workloads) == 1:
            return workloads
        
        # sort input tiles
        buckets = dict()
        for w in workloads:
            buckets.setdefault((w.k, w.k_offset), []).append(w)

        merged: list[GEMMWorkload] = []

        # process each bucket
        for bucket in buckets.values():
            bucket.sort(key=lambda w: (w.m_offset, w.n_offset))
            changed = True

            while changed: # terminate until no more merged tile found
                changed = False
                new_bucket = []
                while bucket:
                    base = bucket.pop(0)
                    i = 0
                    while i < len(bucket):
                        w = bucket[i]
                        m_adjacent = w.m_offset == base.m_offset + base.m
                        n_adjacent = w.n_offset == base.n_offset + base.n
                        k_align = w.k == base.k and w.k_offset == base.k_offset
                        m_align = w.m == base.m and w.m_offset == base.m_offset
                        n_align = w.n == base.n and w.n_offset == base.n_offset

                        horizontal = (m_align and k_align and n_adjacent)
                        vertical = (n_align and k_align and m_adjacent)

                        if horizontal:
                            base.n += w.n
                            bucket.pop(i)
                            changed = True
                            continue
                        elif vertical:
                            base.m += w.m
                            bucket.pop(i)
                            changed = True
                            continue
                        i += 1
                    new_bucket.append(base)
                bucket = new_bucket # re-scan the merged tiles
            merged.extend(bucket)
        if len(merged) > 0:
            return merged
        else:
            return workloads

    def _get_sram_energy_per_core(self, core:SystolicArray):
        
        reporter = core.simulation_reporter
        assert reporter is not None, f"Reporter is None Core[{core.id}]"

        df = reporter.details_df    
        # cycles SRAM read from DRAM
        dram_ifmap_start_cycle = df['DRAM IFMAP Start Cycle'].iloc[0]
        dram_ifmap_stop_cycle = df['DRAM IFMAP Stop Cycle'].iloc[0]
        ifmap_cycles = int(dram_ifmap_stop_cycle - dram_ifmap_start_cycle) + 1 
        bytes_to_ifmap = ifmap_cycles * core.dram_bandwidth

        dram_filter_start_cycle = df['DRAM Filter Start Cycle'].iloc[0]
        dram_filter_stop_cycle = df['DRAM Filter Stop Cycle'].iloc[0]
        filter_cycles = int(dram_filter_stop_cycle - dram_filter_start_cycle) + 1
        bytes_to_filter = filter_cycles * core.dram_bandwidth


        dram_ofmap_start_cycle = df['DRAM OFMAP Start Cycle'].iloc[0]
        dram_ofmap_stop_cycle = df['DRAM OFMAP Stop Cycle'].iloc[0] 
        ofmap_cycles = int(dram_ofmap_stop_cycle - dram_ofmap_start_cycle) + 1
        bytes_to_ofmap = ofmap_cycles * core.dram_bandwidth
        
        # SRAM write and read energy, assume #write bits = #read bits
        ifmap_energy = core.sram_energy_scale * bytes_to_ifmap * 2 * WORD_SIZE
        filter_energy = core.sram_energy_scale * bytes_to_filter * 2 * WORD_SIZE
        ofmap_energy = core.sram_energy_scale * bytes_to_ofmap * 2 * WORD_SIZE
        

        return sum([ifmap_energy, filter_energy, ofmap_energy])

    def _get_sram_energy_analytical(self,core):
        total_bytes = sum(wl.m * wl.k * wl.n for wl in core.iter_get_workloads())
        energy_factor = SRAM_ENERGY_LOOKUP[core.buffer_size] * SRAM_ENERGY_SCALE[str(core.node)]
        sram_energy = total_bytes * 8 * energy_factor
        return sram_energy


    def _get_sram_energy(self):
        total_energy = 0
        for core in self.systolic_arrays:
            # sram_energy = self._get_sram_energy_per_core(core)
            sram_energy = self._get_sram_energy_analytical(core)
            total_energy += sram_energy
        return total_energy

    def __calculate_interconnect_latency_energy_with_congestion(self, system:ChipletSystem) -> tuple[float,float]:
        def __form_links(path: list):
            links = []
            for i in range(len(path) - 1):
                curr = path[i]
                next = path[i + 1]
                link = tuple(sorted((curr, next)))
                links.append(link)
            return links 
        
        
        cores: list[SystolicArray] = list(system.core_dict.values())
        max_core = max(cores)
        link_share_dict:dict = dict()
        path_dict:dict = dict()
        # form the global communication links
        # gather all the traversed links and their shared (src, dst) into a map
        # so that we know exact which path is shared for each (src, dst) pair
        for core in cores:
            if len(core.workloads) > 0 and core.id != max_core.id:
                path, effective_bw_reciprocal, effective_energy_scale = system.get_shortest_path(core.id, dst_id=max_core.id)
                path_dict[core.id] = path
                if path is not None:
                    links = __form_links(path)
                    for link in links:
                        link_share_dict.setdefault(link, []).append((core.id, max_core.id, effective_energy_scale))

        # accumulate the latency for each link
        congested_latency_dict = {}
        congested_energy_dict = {}
        for link, communications in link_share_dict.items():
            total_latency = 0
            total_energy = 0
            _, effective_bw_reciprocal, energy_scale = system.get_shortest_path(link[0], link[1])
            for comm in communications:
                src_id = comm[0]
                src_core:SystolicArray = system.core_dict[src_id]
                bytes_to_transfer = sum(wl.m * wl.n for wl in src_core.workloads) * WORD_SIZE
                latency_ns = bytes_to_transfer * effective_bw_reciprocal
                energy = bytes_to_transfer * 8 * energy_scale
                total_latency += latency_ns
                total_energy += energy

            congested_latency_dict[link] = total_latency # TODO: replace this with actual latency accumulation
            congested_energy_dict[link] = total_energy # TODO: replace this with actual energy accumulation
        
        # get the final end-to-end latency
        final_path_latency:dict = dict()
        for src_id, path in path_dict.items():
            path_total_latency = 0
            path_links = __form_links(path)
            for link in path_links:
                path_total_latency += congested_latency_dict.get(link, 0)
            final_path_latency[(src_id, path[-1])] = path_total_latency
        
        final_total_energy = sum(list(congested_energy_dict.values()))


        max_ltency = max(final_path_latency.values()) if final_path_latency.values() else 0
        return max_ltency, final_total_energy

    def __calculate_interconnect_latency_energy_with_separate_controller(self, system:ChipletSystem) -> tuple[float, float]:
        if system.links is None:
            return 0, 0 #no interconnet for 2D 
        
        def __form_links(path: list):
            links = []
            for i in range(len(path) - 1):
                curr = path[i]
                next = path[i + 1]
                link = tuple(sorted((curr, next)))
                links.append(link)
            return links 
        cores = list(system.core_dict.values())
        max_core = max(cores)
        # extrac all the links to the path
        link_to_dest = set()
        for link in system.links:
            for core in link:
                if max_core.id == core:
                    link_to_dest.add(link)
        latency_by_link_to_dest = dict()
        energy_by_link_to_dest = dict()


        for core in cores:
            core:SystolicArray
            if len(core.workloads) >= 0 and core.id != max_core.id:
                path, effective_bw_reciprocal, effective_energy_scale = system.get_shortest_path(core.id, max_core.id)
                links_in_path = __form_links(path)
                print_links_in_path = [(i+1, j+1) for i, j in links_in_path]
                transfer_bytes = sum(wl.m * wl.n for wl in core.workloads) * WORD_SIZE
                latency_ns = transfer_bytes * effective_bw_reciprocal
                energy_pj = transfer_bytes * effective_energy_scale
                 
                for l in links_in_path:
                    if l in link_to_dest:
                        latency_by_link_to_dest.setdefault(l, []).append(tuple([core.id, max_core.id, latency_ns]))
                        energy_by_link_to_dest.setdefault(l, []).append(energy_pj)
        
        print(f"Communication by link_to_dst with latency: {latency_by_link_to_dest}") if print_info else None
        print(f"Eenergy by link_to_dst with energy: {energy_by_link_to_dest}") if print_info else None
        # sum latency by the link to the dst
        total_latency_by_link = []
        for key, val in latency_by_link_to_dest.items():
            latency_per_link = 0
            for tup in val:
                latency_per_link += tup[-1]
            total_latency_by_link.append(latency_per_link)
            print_key_var = tuple(i+1 for i in key)

        final_total_latency = max(total_latency_by_link)
        total_energy = sum(energy for sublist in energy_by_link_to_dest.values() for energy in sublist)

        return final_total_latency, total_energy

    def system_modeling(
        self,
        system: ChipletSystem,
        activation_from_dram=True,
        output_to_dram=True,
    ) -> tuple[float, float]:
        #top-level function to launch system modeling, return the modeled total latency and energy 
        # construct latency timeline for every core: Phase1: from dram_load to finish compute
        cores: list[SystolicArray] = list (system.core_dict.values())
        core_time_stamp = np.zeros((len(cores))) # dram_load latency + compute latency
        core_energy_stamp = np.zeros((len(cores))) # dram load energy + transferred energy

        max_core = max(cores)
        for core in cores:
            dram_latency_ns, dram_energy_pj = self.__calculate_dram_latency_energy(
                core,
                activation_from_dram=activation_from_dram,
                output_to_dram=output_to_dram,
            )
            core_time_stamp[core.id] += dram_latency_ns
            core_time_stamp[core.id] += core.total_cycle / core.frequency * 10**9 # accu total compute latency in ns
            core_energy_stamp[core.id] += dram_energy_pj
            
        compute_time = max(core_time_stamp)
        
        if self.splitting_k:
            total_interconnect_latency, total_interconnect_energy = self.__calculate_interconnect_latency_energy_with_separate_controller(system)
            reduction_latency = self.__compute_reduction_latency_ns(cores, max_core)

        else:
            total_interconnect_latency, total_interconnect_energy = 0, 0 
            reduction_latency = 0 # reduction should not happen

            
        current_time = compute_time + total_interconnect_latency # synchronize every core for final reduction
        # accumulate the reduction latency
        current_time += reduction_latency
        if self.splitting_k and output_to_dram:
            # write back to DRAM from max_core if splitting K is enabled. 
            total_write_back_words = self.M * self.N
            write_back_latency_cycle = total_write_back_words / max_core.dram_bandwidth 
            write_back_latency_ns = write_back_latency_cycle / max_core.frequency * 10**9 # in ns
            current_time += write_back_latency_ns
            write_back_energy = total_write_back_words * WORD_SIZE * 8 * max_core.dram_energy_scale # in pj
            core_energy_stamp[max_core.id] += write_back_energy
        else:
            write_back_latency_ns = 0
            write_back_energy = 0

        total_energy = np.sum(core_energy_stamp) + total_interconnect_energy

        if print_info:
            print(f"[INFO] reduction_latency:                {reduction_latency:.3e}")
            print(f"[INFO] write_back_latency (ns):          {write_back_latency_ns:.3e}")  
            print(f"[INFO] Current Time Final:               {current_time:.3e}")
            print(f"LATENCY_END\n")  
            print(f"ENERGY_START")     
            print(f"[INFO] Write back energy (pj):           {write_back_energy:.3e}")
            print(f"[INFO] Core dram energy w WB energy(pj): {np.sum(core_energy_stamp):.3e}")
            print(f"[INFO] Comm'n energy stamp(pj):          {total_interconnect_energy:.3e}")
            print(f"[INFO] DRAM+Comm'n  Energy(pJ):          {total_energy:.3e}\n")
            print(f"ENERGY_END\n")

        return current_time, total_energy 
