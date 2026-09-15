'''
Whole chiplet system class.
Global routing, interconnect info recorded here.
'''

from system.utils.SystolicArray import SystolicArray
from system.utils.FpgaChiplet import FpgaChiplet
from chiplet.n_utils import d2d_bw_calc, calculate_memory_bandwidth, get_sram_area_energy
from config import print_info

from pathlib import Path
import json
import numpy as np




with open('cfg/parameters/freq_scale.json', 'r') as f:
    freq_scale = json.load(f)
BASE_FREQUENCY = freq_scale.get("BASE_FREQUENCY", {})
freq_scaling_factors = freq_scale.get("freq_scaling_factors",{})
freq_scaling_factors = {int(k): v for k, v in freq_scaling_factors.items()}


with open('cfg/parameters/energy_eff.json', 'r') as f:
    energy_pj_data = json.load(f)
Die2Die_pj_per_bit = energy_pj_data.get("Die2Die_pj_per_bit", {})
DRAM_pj_per_bit = energy_pj_data.get("DRAM_pj_per_bit", {})


class ChipletSystem:

    @staticmethod
    def _is_fpga_chiplet(chiplet: dict) -> bool:
        return str(chiplet.get("chiplet_type", "systolic_array")).lower() == "fpga"

    def _core_setup(self, chiplet_dict: dict, dram_bw_list: list) -> dict[int: SystolicArray]:
        core_dict = dict()
        ordered_keys = sorted(
            chiplet_dict.keys(), key=lambda key: int(key.split("_")[1])
        )
        for bandwidth_index, key in enumerate(ordered_keys):
            chiplet = chiplet_dict[key]
            tech_node = int(chiplet["tech_node"])
            size = int(chiplet["sys_array_size"].split("x")[0])
            buf = int(chiplet["sram_buf"])
            sram_area, sram_energy_scale = get_sram_area_energy(buf, tech_node)
            id = int(key.split("_")[1]) - 1
            power = float(chiplet["power"])
            area = float(chiplet["area"]) - float(sram_area)
            frequency = self.BASE_FREQUENCY * freq_scaling_factors[tech_node]
            bandwidth = int (dram_bw_list[bandwidth_index] * 10**9 / self.BASE_FREQUENCY) #bytes per cycle calibrated to base frequency
            core = SystolicArray(size, size, buf, 
                                 id, power = power, area = area, 
                                 bandwidth=bandwidth,
                                 sram_energy_scale=float(sram_energy_scale),
                                 dram_energy_scale = DRAM_pj_per_bit[self.dram_type], 
                                node = tech_node, frequency=frequency)
            core_dict[id] = core

        return core_dict

    def _fpga_setup(self, chiplet_dict: dict) -> dict:
        fpga_dict = dict()
        for key in chiplet_dict.keys():
            chiplet = FpgaChiplet.from_dict(key, chiplet_dict[key])
            fpga_dict[chiplet.id] = chiplet
        return fpga_dict
    

    def _interconnect_setup(self, pkg:dict):
        hi_type = pkg["HI_pkg_type"]
        result = dict()
        #build interconnect

        if hi_type == '2d' or hi_type == '2d_na':
        #system is in 2D, no interconnect needed
            src_id = 0
            dst_id = 0
            result.setdefault(src_id, []).append((dst_id, 0, 0))
            return result, hi_type, None
        
        protocol_25d = pkg["protocol_2.5d"]
        protocol_3d = pkg["protocol_3d"]    
        

        if protocol_25d == 'na' and protocol_3d == 'na':
            assert False, f"Both 2.5D and 3D protocol are 'na', input package:\n{pkg}"

        self.protocol = {"2.5d": protocol_25d, "3d": protocol_3d}
        self.d2d_connection = {"2.5d": None, "3d": None}
        links = set()

        # construct 3D chiplet system interconnect
        for conn in pkg["inter_pkg_conn"]:
            src = conn["from"]
            dst = conn["to"]
            d2d_connect_type = conn["connection_type"]

            src_id = int(src.split('_')[1]) - 1
            location = conn["loc"]
            if location != "2.5d_chiplet": # set the location only if this core is not 2.5D chiplet
                self.endpoint_dict[src_id].location = location.split("_")[1]
            if dst == "na":
                continue
            dst_id = int(dst.split('_')[1]) - 1
            if d2d_connect_type.lower().startswith("2.5d_"): # 2.5d connection
                protocol = protocol_25d
                is_3d_connection = False
                if self.d2d_connection["2.5d"] == None:
                    self.d2d_connection["2.5d"] = d2d_connect_type
            else:
                if self.d2d_connection["3d"] == None:
                    self.d2d_connection["3d"] = d2d_connect_type
                protocol = protocol_3d
                is_3d_connection = True
            links.add(tuple(sorted((src_id, dst_id))))
            src_area = self.endpoint_dict[src_id].area
            dst_area = self.endpoint_dict[dst_id].area

            src_core_bw = d2d_bw_calc(pkg=d2d_connect_type, protocol=protocol, area=src_area, is_3d=is_3d_connection, node=self.endpoint_dict[src_id].node)
            dst_core_bw = d2d_bw_calc(d2d_connect_type, protocol, dst_area, is_3d_connection, node=self.endpoint_dict[dst_id].node)

            bw = min(src_core_bw, dst_core_bw) * 10**9 / BASE_FREQUENCY
            print(f"[DEBUG COST - LATENCY - D2D 2] BW is {bw} will be min (src, dst) min{src_core_bw} , {dst_core_bw} = {bw} \n") if print_info else None
            assert bw != 0, f"[ERROR] BW = {bw}, Core[{src_id} area: {src_area} bw {src_core_bw}  Core[{dst_id}] area: {dst_area} bw {dst_core_bw} {d2d_connect_type} {protocol} {is_3d_connection}\n {pkg}"
            weight = 1 / bw 

            # use pj per bit as second weight for ease of later energy modeling 
            energy_weight = Die2Die_pj_per_bit.get(protocol)
            
            assert energy_weight != None, f"{protocol} System: Die2Die_pj_per_bit.get({protocol}) is None"

            result.setdefault(src_id, []).append((dst_id, weight, energy_weight))
            result.setdefault(dst_id, []).append((src_id, weight, energy_weight))
                        
        return result, hi_type, links


    def __init__(self, spec_file:str = None, arch_dict = None, BASE_FREQUENCY = 10**9):
        assert spec_file != None or arch_dict != None, f"[ERROR] Input spec_file and architecture specs cannot both be None!!! {spec_file} {arch_dict} \n"

        if arch_dict:
            solution = arch_dict
        else:
            json_path = Path(spec_file)
            with json_path.open("r", encoding='utf-8') as fp:
                solution:dict = json.load(fp)
            
        pkg = solution["pkg"]
        self.dram_type = pkg["mem_pkg_conn"]["mem_type"]
        dram_bw_list = calculate_memory_bandwidth(arch_dict)
        chiplet = {k:v for k, v in solution.items() if k.startswith("Chiplet_")}
        sa_chiplet = {
            k: v for k, v in chiplet.items() if not self._is_fpga_chiplet(v)
        }
        fpga_chiplet = {
            k: v for k, v in chiplet.items() if self._is_fpga_chiplet(v)
        }
        self.BASE_FREQUENCY = BASE_FREQUENCY
        self.core_dict = self._core_setup(sa_chiplet, dram_bw_list=dram_bw_list)
        self.fpga_chiplet_dict = self._fpga_setup(fpga_chiplet)
        self.endpoint_dict = {**self.core_dict, **self.fpga_chiplet_dict}

        self.interconnect_dict, self.interconnect_type, self.links = self._interconnect_setup(pkg)

        for core in self.core_dict.values():
            print(f"core[{core.id}]: {core.frequency}, {core.dram_bandwidth}") if print_info else None

        self._update_dram_bw_energy_from_interconnect()
        self.Die2Die_pj_per_bit = Die2Die_pj_per_bit
        self.DRAM_pj_per_bit = DRAM_pj_per_bit
        
        for core in self.core_dict.values():
            print(f"core[{core.id}]: {core.frequency}, {core.dram_bandwidth}") if print_info else None

    @property
    def systolic_arrays(self):
        return list(self.core_dict.values())

    @property
    def fpga_chiplets(self):
        return list(self.fpga_chiplet_dict.values())

        
    def get_shortest_path(self, src_id: int, dst_id: int) -> tuple[list[int], float, float]:
        if self.interconnect_dict.get(src_id) == None:
            #TODO: In what condition will fall into this case? 
            assert False, f"self.interconnect_dict.get(src_id) == None, src_id = {src_id}, {self.interconnect_dict}"
        
        if self.interconnect_type == '2d' or src_id == dst_id:
            return None, 0, 0
        
        nodes: set[int] = set(self.interconnect_dict.keys())
        dist_bw_weights = {v: float('inf') for v in nodes}
        dist_energy = {v: 0 for v in nodes}
        prev = {v: None for v in nodes}
        dist_bw_weights[src_id] = 0
        
        unvisited = set(nodes)
        
        while unvisited:
            u = min(unvisited, key=lambda v: dist_bw_weights[v])
            if dist_bw_weights[u] == float('inf') or u == dst_id:
                # either unreachable or reached dst
                break 
            # remove current visited node
            unvisited.remove(u)
            try:
                for v, bw_weights, energy_weights in self.interconnect_dict.get(u, []):
                    if v in unvisited:
                        updated_dist_bw = dist_bw_weights[u] + bw_weights # effective BW is upper bounded by the minimum BW along the visited path 
                        updated_dist_energy = dist_energy[u] + energy_weights
                        # find shorter path, update the path and record visited path
                        if updated_dist_bw < dist_bw_weights[v]:
                            dist_bw_weights[v] = updated_dist_bw
                            dist_energy[v] = updated_dist_energy
                            prev[v] = u
            except ValueError as e:
                print(f"[ERROR] Failed to unpack interconnect_dict entry for node {u}: {self.interconnect_dict.get(u)}")
                print(f"Expected each entry to be a tuple of (v, weights), but got something else.")
                raise e
        # reconstruct visited path
        if dist_bw_weights[src_id] is float('inf'):
            assert False, print(f"dist_bw_weights[{src_id}] is INF")
        
        path = []
        v = dst_id
        while v is not None:
            path.append(v)
            v = prev[v]
        path.reverse()

        return path, dist_bw_weights[dst_id], dist_energy[dst_id] 
    def __process_3d_chiplets(self, base_die:SystolicArray, top_die:SystolicArray):

        assert self.interconnect_type == "3d" or self.interconnect_type == "2.5d_3d", \
            f"[ERROR] Only call __process_3d_chiplets() for 3d / 2.5d_3d architecture, current {self.interconnect_type}"
        # key = "3d" if dram_is_3d_top  else "2.5d"
        # dram_die = top_die if dram_is_3d_top else base_die

        key = "3d"
        dram_die = base_die
        base_to_top_path, base_to_top_bw_reciprocal, base_to_top_energy = self.get_shortest_path(base_die.id, top_die.id)

        bw_list = [dram_die.dram_bandwidth]
        # print(f"Path from base to top: {base_to_top_path}")
        for idx in range(1,len(base_to_top_path)):
            curr_id = base_to_top_path[idx]
            prev_id = base_to_top_path[idx - 1]
            curr_core = self.core_dict[curr_id]
            prev_core = self.core_dict[prev_id]
            prev_bw = int(d2d_bw_calc(self.d2d_connection[key], self.protocol[key],prev_core.area, is_3d=True, node = prev_core.node))
            curr_bw = int(d2d_bw_calc(self.d2d_connection[key],self.protocol[key], curr_core.area, is_3d=True, node = curr_core.node))
            bw_list.append(min(curr_bw, prev_bw))
            curr_core.dram_bandwidth = min(bw_list)
            _, _, path_energy_scale = self.get_shortest_path(base_die.id, curr_core.id)
            curr_core.dram_energy_scale = path_energy_scale + base_die.dram_energy_scale


    def _update_dram_bw_energy_from_interconnect(self):

        if self.interconnect_type != "2.5d_3d" and self.interconnect_type != "3d":
            # only need to update BW if 3d stack is involved 
            return
        
        # search dram die
        for core in self.core_dict.values():
            if core.location == "base":
                base_die = core
            if core.location == "top":
                top_die = core


        assert base_die is not None and top_die is not None, \
            f"[ERROR] No base die / top die found in {self.interconnect_type} system"
        
        # update the bandwidth and energy for 3D chiplets
        self.__process_3d_chiplets(base_die, top_die)





    def print_core_dict(self):
        output = ""
        for sa in self.core_dict.values():
            output += sa.__str__()
        print(output)

            
