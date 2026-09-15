import random 
import json
import numpy as np
import math
import os
import copy
import re
import csv
import pandas as pd
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from chiplet.n_disagg import PackageGenerator, _validate_protocols, _determine_protocols
from system.utils.SimulationCache import SimulationCache
from config import sram_selection_mode, calibration_mode, print_info


#Takes array size and tech node and then scales it to generate final power and area values for chiplets
def get_area_power(array_size, tech_node):
    #tech_node = [7,10,14]
    current_dir = Path(__file__).resolve().parent
    scaling_path =  "cfg/parameters/scaling.json"
    base_spec_path =  "cfg/parameters/base_spec_7nm.json"
    with open(scaling_path, 'r') as scale_val:
        scaling_factor = json.load(scale_val)
    with open(base_spec_path, 'r') as f:
        base_spec_7nm = json.load(f)
    
    base_val = base_spec_7nm[array_size]
    scaled_area = base_val["area"] * scaling_factor["area"][tech_node]
    scaled_power = base_val["power"] * scaling_factor["power"][tech_node]
    
    #Adding D2D IP's area 
    scaled_area = scaled_area * 1.08 # 8% overhead for D2D IPs 
    
    return scaled_area, scaled_power

#Takes sram size and tech_node and then gives the final sram area 
def get_sram_area_energy(sram_size, tech_node):
    sram_size = str(sram_size)
    tech_node = str(tech_node)
    sram_scaling_path = "cfg/parameters/sram_scaling.json"
    sram_base_spec_path = "cfg/parameters/base_spec_sram_7nm.json"
    with open(sram_scaling_path, 'r') as scale_val:
        sram_scaling_factor = json.load(scale_val)
    with open(sram_base_spec_path, 'r') as f:
        sram_base_spec_7nm = json.load(f)
    
    sram_base_val = sram_base_spec_7nm[sram_size]
    scaled_sram_area = sram_base_val["area"] * sram_scaling_factor["area"][str(tech_node)]
    scaled_sram_energy = sram_base_val["energy"] * sram_scaling_factor["energy"][str(tech_node)]
    
    return scaled_sram_area,scaled_sram_energy
    


def read_json_input_params(json_file):
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)

        # Use .get() for safer access, providing defaults where sensible
        max_chiplet = config.get('max_chiplet', 3) # Default 3 if missing
        sys_array = config.get('sys_array', [])     # Default empty list if missing
        tech_nodes = config.get('tech_nodes', [])    # Default empty list if missing
        sram_buf_sizes = config.get('sram_buf_sizes', {}) # Default to empty dict

        # Safely access nested dictionary 'pkg'
        pkg_config = config.get('pkg', {}) # Default empty dict if 'pkg' is missing

        # Access keys within 'pkg', defaulting to empty lists
        inter_pkg_arch = pkg_config.get('inter_pkg_architecture', [])
        mem_pkg_arch = pkg_config.get('mem_pkg_architecture', [])
        protocol_arch = pkg_config.get('protocol', [])

        return max_chiplet, sys_array, tech_nodes, sram_buf_sizes, inter_pkg_arch, mem_pkg_arch, protocol_arch

    except FileNotFoundError:
        print(f"Error: File not found at '{json_file}'")
        return None, None, None, None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file}'. Check file format.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None, None, None

### Write json 
def write_solution_to_json(solution_data, output_json_file):
    if not isinstance(solution_data, dict):
        print("Error: Invalid solution data provided (must be a dictionary).")
        return False

    try:
        with open(output_json_file, 'w') as f:
            json.dump(solution_data, f, indent=4) 
        print(f"Successfully wrote architecture solution to: {output_json_file}") if print_info else None
        return True
    except IOError as e:
        print(f"Error writing JSON file {output_json_file}: {e}")
        return False
    except TypeError as e:
        print(f"Error: Solution data structure could not be serialized to JSON: {e}")
        return False



def generate_mem_channel_distribution(chiplets_dict, selected_mem_type, dont_use_random_channel_distribution=True):
    chiplet_ids, num_chiplets = list(chiplets_dict.keys()), len(chiplets_dict)
    
    selected_mem_type = selected_mem_type.lower()
    if dont_use_random_channel_distribution:
        # Use fixed values for each memory type
        if "ddr" in selected_mem_type:
            total_channels = 8
        elif "hbm" in selected_mem_type:
            total_channels = 16
        else:
            total_channels = 8
        
        total_channels = max(total_channels, num_chiplets)
    else:
        # Determine min and max channels based on memory type
        if "ddr" in selected_mem_type:
            min_ch, max_ch = 2, 8
        elif "hbm" in selected_mem_type:
            min_ch, max_ch = 8, 16
        else:
            min_ch, max_ch = 2, 8
        
        total_channels = (
            random.randint(max(min_ch, num_chiplets), max_ch)
            if max_ch >= num_chiplets else max_ch
        )
    
    channels_dist, remaining_channels = [1] * num_chiplets, total_channels - num_chiplets
    
    if remaining_channels > 0:
        chiplet_areas = [chiplets_dict[cid].get("area", 1.0) for cid in chiplet_ids]
        total_area = sum(chiplet_areas)
        if total_area > 0:
            proportions = [area / total_area for area in chiplet_areas]
            for _ in range(remaining_channels):
                ideal_alloc = [(sum(channels_dist) + 1) * p for p in proportions]
                errors = [ideal_alloc[i] - channels_dist[i] for i in range(num_chiplets)]
                channels_dist[errors.index(max(errors))] += 1
        else: # Fallback for zero area
            for i in range(remaining_channels): channels_dist[i % num_chiplets] += 1
            
    new_mem_conn = {"mem_type": selected_mem_type}
    new_mem_conn.update(dict(zip(chiplet_ids, channels_dist)))
    return new_mem_conn


def recursive_split_updated(areas, axis=0, depth=0):
    
    indent = "  " * depth
    if print_info:
        print(f"{indent}Depth {depth}, Axis: {axis}, Areas: {areas}")

    # Base Case
    if len(areas) <= 1:
        if not areas:
            return np.array((0.0, 0.0)), {}
        
        chiplet_id, area_val = areas[0]
        v = (area_val / 2) ** 0.5
        size = np.array((v + v * ((axis + 1) % 2), v + v * axis))
        
        layouts = {chiplet_id: (0, 0, size[0], size[1])}
        
        if print_info:
            print(f"{indent}Base case: {chiplet_id}, area {area_val}, size {size}")
        return size, layouts

    # Sort chiplets by area (descending)
    sorted_areas = sorted(areas, key=lambda x: x[1], reverse=True)
    
    sums = np.array([0.0, 0.0])
    blocks = [[], []]
    for chiplet in sorted_areas:
        idx = np.argmin(sums)
        blocks[idx].append(chiplet)
        sums[idx] += chiplet[1]
    
    if print_info:
        print(f"{indent}Block 0: {blocks[0]}, Sum: {sums[0]}")
        print(f"{indent}Block 1: {blocks[1]}, Sum: {sums[1]}")

    # Recursive splits
    left_size, left_layouts = recursive_split_updated(blocks[0], (axis + 1) % 2, depth + 1)
    right_size, right_layouts = recursive_split_updated(blocks[1], (axis + 1) % 2, depth + 1)

    # Combine results
    combined_size = np.zeros(2)
    combined_layouts = {}
    padding = 0.1  # optional padding

    if axis == 0:  # Vertical split (side-by-side)
        combined_size[0] = left_size[0] + right_size[0] + padding
        combined_size[1] = max(left_size[1], right_size[1])
        
        combined_layouts.update(left_layouts)
        offset_x = left_size[0] + padding
        for cid, (x, y, w, h) in right_layouts.items():
            combined_layouts[cid] = (x + offset_x, y, w, h)

    else:  # Horizontal split (stacked)
        combined_size[0] = max(left_size[0], right_size[0])
        combined_size[1] = left_size[1] + right_size[1] + padding
        
        combined_layouts.update(left_layouts)
        offset_y = left_size[1] + padding
        for cid, (x, y, w, h) in right_layouts.items():
            combined_layouts[cid] = (x, y + offset_y, w, h)

    if print_info:
        print(f"{indent}Combined size: {combined_size}")
        
    return combined_size, combined_layouts


def calc_HI_dimension(system_dict):
    pkg_config = system_dict.get('pkg', {})
    pkg_type = pkg_config.get('HI_pkg_type')
    
    print(f"\n----- Processing Package Type: {pkg_type} -----") if print_info else None

    areas_to_process = []
    
    # a) If it is 2D or 2.5D, use the original logic
    if pkg_type in ['2d', '2.5d']:
        print("[INFO] --- Detected 2D or 2.5D system. Using all chiplet areas.") if print_info else None
        chiplet_entries = {k: v for k, v in system_dict.items() if k.startswith('Chiplet_')}
        # Sort by chiplet number to maintain a consistent order
        sorted_chiplet_items = sorted(
            chiplet_entries.items(), 
            key=lambda item: int(item[0].split('_')[1])
        )
        # areas_to_process = [data['area'] for _, data in sorted_chiplet_items]
        areas_to_process = [(name, data['area']) for name, data in sorted_chiplet_items]

    # b) If it is 3D, find the base die and use its area
    elif pkg_type == '3d':
        print("[INFO] --- Detected 3D system. Finding base die area.") if print_info else None 
        inter_pkg_conn = pkg_config.get('inter_pkg_conn', [])
        base_die_name = None
        for conn in inter_pkg_conn:
            # The base die is identified by 'base' in its location
            if 'base' in conn.get('loc', ''):
                base_die_name = conn.get('from')
                break
        
        if base_die_name and base_die_name in system_dict:
            base_area = system_dict[base_die_name]['area']
            print(f"[INFO] --- Base die is '{base_die_name}' with area {base_area}.") if print_info else None
            # The layout is determined by the single largest footprint in the stack
            #areas_to_process = [base_area]
            areas_to_process = [(base_die_name, base_area)]
        else:
            print("[WARNING] --- Could not determine base die for 3D system. Defaulting to empty.") if print_info else None

    # b) If it is 2.5D_3D, find base dies of stacks and standalone chiplets
    elif pkg_type == '2.5d_3d':
        print("[INFO] --- Detected 2.5D_3D system. Identifying layout entities.") if print_info else None
        all_chiplet_keys = {k for k in system_dict if k.startswith('Chiplet_')}
        inter_pkg_conn = pkg_config.get('inter_pkg_conn', [])
        
        stacked_chiplets = set()
        layout_entities = {} # Using a dict to store entity name and area

        # First, find all 3D stacks and their base dies
        for conn in inter_pkg_conn:
            if '3d' in conn.get('connection_type', ''):
                # Add both chiplets in the 3D connection to the stacked set
                stacked_chiplets.add(conn.get('from'))
                if conn.get('to') != 'na':
                    stacked_chiplets.add(conn.get('to'))
                
                # If this connection defines a base, its 'from' chiplet is a layout entity
                if 'base' in conn.get('loc', ''):
                    base_die_name = conn.get('from')
                    if base_die_name in system_dict:
                        layout_entities[base_die_name] = system_dict[base_die_name]['area']
                        print(f"[INFO] --- Found 3D stack base: '{base_die_name}'.") if print_info else None

        # Second, identify standalone 2.5D chiplets
        standalone_chiplets = all_chiplet_keys - stacked_chiplets
        for chiplet_key in standalone_chiplets:
            layout_entities[chiplet_key] = system_dict[chiplet_key]['area']
            print(f"[INFO] --- Found standalone 2.5D chiplet: '{chiplet_key}'.") if print_info else None
            
        # Sort the entities by chiplet number for consistent processing
        sorted_entity_keys = sorted(layout_entities.keys(), key=lambda k: int(k.split('_')[1]))
        #areas_to_process = [layout_entities[key] for key in sorted_entity_keys]
        areas_to_process = [(key, layout_entities[key]) for key in sorted_entity_keys]

    else:
        print(f"[ERROR] --- Unknown or unsupported package type: {pkg_type}. Aborting.") 
        return None

    print(f"[DEBUG COST] --- Final list of areas for layout calculation: {areas_to_process}") if print_info else None
    
    # Call the recursive split function with the determined list of areas
    lay_size, lay_info = recursive_split_updated(areas_to_process)
    print(f"[DEBUG COST] --- Finished Layout Calculation --- and layout dimensions are {lay_size}") if print_info else None
    return lay_size, lay_info

   
   

class SystemCostCalculator:
    def __init__(self, wafer_diameter_mm=300):
        self.wafer_diameter = wafer_diameter_mm
        
        # --- Internal Cost Model Data ---
        #https://cset.georgetown.edu/wp-content/uploads/AI-Chips%E2%80%94What-They-Are-and-Why-They-Matter-1.pdf
        #AI Chips , Saif M Khan, Alexander Mann
        process_nodes = [250, 180, 130, 90, 65, 40, 28, 20, 14, 10, 7, 5]
        wafer_costs = [900, 1020, 1580, 1650, 1937, 2274, 2891, 3677, 3984, 5992, 9346, 16988]
        #ECO-CHIP
        defect_density_map = {7: 0.2, 10: 0.11, 14: 0.08, 20: 0.08, 28: 0.07, 65: 0.05}

        self.process_data_table = {}
        for i, node in enumerate(process_nodes):
            self.process_data_table[node] = {
                "wafer_cost": wafer_costs[i],
                "defect_density": defect_density_map.get(node)
            }

    def _dpw(self, area):
        """Calculates dies per wafer."""
        if area <= 0: return 0
        return math.pi * self.wafer_diameter * ((self.wafer_diameter / (4 * area)) - (1 / math.sqrt(2 * area)))

    def _yield_calc(self, area, defect_density):
        """Calculates the die yield."""
        if defect_density is None: return 1.0 # Assume perfect yield if no data
        # Using Murphy's yield model
        return (1 + (defect_density * 1e4) * (area * 1e-6) / 10) ** -10

    def _cost_per_die(self, area, tech_node, consider_yield=True):
        """Calculates the cost and yield for a single die (chiplet or interposer)."""
        node_data = self.process_data_table.get(int(tech_node))
        if not node_data:
            # print(f"Warning: No cost data for tech node {tech_node}. Cost will be zero.")
            return 0.0, 1.0

        dies_per_wafer = self._dpw(area)
        if dies_per_wafer <= 0: return float('inf'), 0.0

        cost_per_wafer = node_data["wafer_cost"]
        def_den_val = node_data["defect_density"]
        
        dollar_cost_full_yield = cost_per_wafer / dies_per_wafer
        yield_of_die = self._yield_calc(area=area, defect_density=def_den_val)
        
        if consider_yield and yield_of_die > 0:
            final_cost = dollar_cost_full_yield / yield_of_die
        else:
            final_cost = dollar_cost_full_yield
            
        return final_cost, yield_of_die

    def _get_bonding_yield(self, pkg, bonding_yields):
        hi_type = pkg.get("HI_pkg_type", "").lower()
        conn = pkg.get("inter_pkg_conn")

        if hi_type == "2.5d_3d":
            if isinstance(conn, list):
                conn_types = set(c.get("connection_type") for c in conn if c.get("connection_type"))
                yield_2_5d = None
                yield_3d = None

                for conn_type in conn_types:
                    if conn_type.startswith("2.5d_") and conn_type in bonding_yields:
                        yield_2_5d = bonding_yields[conn_type]
                    elif conn_type.startswith("3d_") and conn_type in bonding_yields:
                        yield_3d = bonding_yields[conn_type]

                if yield_2_5d is not None and yield_3d is not None:
                    return (yield_2_5d + yield_3d) / 2

                return None

        elif hi_type == "3d":
            if isinstance(conn, list) and conn:
                conn_type = conn[0].get("connection_type")
                return bonding_yields.get(conn_type)

        elif hi_type == "2.5d":
            if isinstance(conn, list) and conn:
                conn_type = conn[0].get("connection_type")
                return bonding_yields.get(conn_type)
            #if isinstance(conn, str):
            #    return bonding_yields.get(conn)
        elif hi_type == "2d":
            return 1
        return None

    def calculate_total_cost(self, system_dict, interposer_area, interposer_tech_node): #bonding_yield):
        #Calculates the overall system cost based on chiplets, an interposer, and bonding yield.
        total_chiplet_cost = 0.0
        
        # Calculate cost for each chiplet
        for key, value in system_dict.items():
            if key.startswith('Chiplet_'):
                area = value['area']
                tech_node = value['tech_node']
                chiplet_cost, _ = self._cost_per_die(area, tech_node)
                print(f"[INFO] {key}: area = {area}, tech_node = {tech_node}, cost = {chiplet_cost}") if print_info else None
                total_chiplet_cost += chiplet_cost
        print(f"[INFO] Total chiplet cost: {total_chiplet_cost}") if print_info else None
        
        # Calculate the interposer cost using the provided parameters
        interposer_cost, _ = self._cost_per_die(interposer_area, interposer_tech_node)
        print(f"[INFO] Interposer: area = {interposer_area}, tech_node = {interposer_tech_node}, cost = {interposer_cost}") if print_info else None 
        
        # Sum of all component costs
        total_pre_bond_cost = total_chiplet_cost + interposer_cost
        print(f"[INFO] Total pre-bond cost: {total_pre_bond_cost}") if print_info else None
        
        # Apply final bonding yield
        with open("cfg/parameters/bonding_yield.json") as f:
            bonding_yields = json.load(f) 
        
        package_value = system_dict['pkg']
        bonding_yield_val = self._get_bonding_yield(package_value, bonding_yields)
        final_system_cost = total_pre_bond_cost / bonding_yield_val if bonding_yield_val > 0 else float('inf')
        # For 1million samples, the cost is multipled by 1e6
        final_system_cost = final_system_cost * 1e6 
        print(f"[INFO] Final system cost after bonding yield ({bonding_yield_val}): {final_system_cost}") if print_info else None
        
        mem_type_value = system_dict['pkg']['mem_pkg_conn']['mem_type']
        if "hbm" in mem_type_value.lower():
            final_system_cost = final_system_cost * 1.35 #35% increase for HBM memory
        else:
            final_system_cost = final_system_cost * 1.15 #15% increase for DDR memory
        return round(final_system_cost, 2)

 
def calculate_system_metrics(system_dict):
    
    # Power calculations 
    cal_power = 0
    for k in sorted(system_dict):
        if k.startswith('Chiplet_'):
            power = system_dict[k].get('power', 0)
            print(f"[INFO] {k}: power = {power}") if print_info else None
            cal_power += power
    
    # Area calculations 
    final_dim, _ = calc_HI_dimension(system_dict)
    total_interposer_area = np.prod(final_dim)
    cal_area = total_interposer_area
    
    # Cost calculations
    cost_calculator = SystemCostCalculator()
    cal_dollar_cost = cost_calculator.calculate_total_cost(
        system_dict=system_dict,
        interposer_area=total_interposer_area,
        interposer_tech_node=65
        #bonding_yield=0.99
    )
    return cal_power, cal_area, cal_dollar_cost
    
def calculate_system_normalized_metrics(power, area, energy, energy_sram, dollar, latency, embCarbon, opeCarbon, profile_name,cost_averages: dict, arch_dict: dict):
    
    with open("cfg/parameters/cost_profiles.json") as f:
        COST_PROFILES = json.load(f)
        
    profile = COST_PROFILES.get(profile_name.lower())
    if profile is None:
        raise ValueError(f"Unknown profile: {profile_name}")

    energy_coff = profile["energy_coff"]
    perf_coeff = profile["perf_coeff"]
    area_coeff = profile["area_coeff"]
    cost_coeff = profile["cost_coeff"]
    embc_coeff = profile["embc_coeff"]
    opec_coeff = profile["opec_coeff"]
    print(f"[INFO] Using profile : {profile_name} with coeff. Energy_coeff:{energy_coff}, Perf_coeff:{perf_coeff}, Area_coeff:{area_coeff}, Cost_coeff:{cost_coeff}, Embc_coeff:{embc_coeff}, Opec_coeff:{opec_coeff}\n") if print_info else None
    
    
    #################
    energy_compute = power * latency # W * ns
    energy_compute = energy_compute * 1000 # Convert to pJ
    energy_compute = energy_compute + energy_sram #SRAM+COMPUTE energy
    energy_f = energy + energy_compute
    #################
    if print_info:
        print(f"[INFO] Dram+Comm'n energy (pJ):           {energy:.3e}")
        print(f"[INFO] SRAM energy (pJ):                  {energy_sram:.3e}")
        print_var_energy_compute = power * latency * 1000 #in pJ
        print(f"[INFO] Power: {power} and latency (ns) : {latency:.3e} ") 
        print(f"[INFO] Energy_compute (pJ) =              {print_var_energy_compute:.3e}") 
        print(f"[INFO] Total Energy (pJ):                 {energy_f:.3e}") 

    raw_cost_dict = {'power': power, 'area': area, 'dollar': dollar, 'latency': latency, 'energy': energy_f, 'embCarbon': embCarbon, 'opeCarbon': opeCarbon } #TODO 1: Remove Power 
    
    
    ################
    ## Method 1 : Use average 
    if calibration_mode == 'avg':
        print(f"[INFO] Using average method for cost normalization") if print_info else None

        avg_energy = cost_averages.get('avg_energy', 1.0) 
        avg_area = cost_averages.get('avg_area', 1.0)
        avg_dollar = cost_averages.get('avg_dollar_cost', 1.0)
        avg_latency = cost_averages.get('avg_latency', 1.0)
        avg_embCarbon = cost_averages.get('avg_embCarbon', 1.0)
        avg_opeCarbon = cost_averages.get('avg_opeCarbon', 1.0)

        norm_energy = energy_f / avg_energy if avg_energy != 0 else (0 if energy == 0 else float('inf')) 
        norm_area = area / avg_area if avg_area != 0 else (0 if area == 0 else float('inf'))
        norm_dollar = dollar / avg_dollar if avg_dollar != 0 else (0 if dollar == 0 else float('inf'))
        norm_latency = latency / avg_latency if avg_latency != 0 else (0 if latency == 0 else float('inf'))
        norm_embCarbon = embCarbon / avg_embCarbon if avg_embCarbon != 0 else (0 if embCarbon == 0 else float('inf'))
        norm_opeCarbon = opeCarbon / avg_opeCarbon if avg_opeCarbon != 0 else (0 if opeCarbon == 0 else float('inf'))
    
    ################
    
    ################
    ## Method 2: Use max 
    if calibration_mode == 'max':
        print(f"[INFO] Using max method for cost normalization") if print_info else None
        max_energy = cost_averages.get('energy_max', 1.0) 
        max_area = cost_averages.get('area_max', 1.0)
        max_dollar = cost_averages.get('cost_max', 1.0)
        max_latency = cost_averages.get('latency_max', 1.0)
        max_embCarbon = cost_averages.get('embCarbon_max', 1.0) 
        max_opeCarbon = cost_averages.get('opeCarbon_max', 1.0) 
    
        norm_energy = energy_f / max_energy if max_energy != 0 else (0 if energy == 0 else float('inf')) 
        norm_area = area / max_area if max_area != 0 else (0 if area == 0 else float('inf'))
        norm_dollar = dollar / max_dollar if max_dollar != 0 else (0 if dollar == 0 else float('inf'))
        norm_latency = latency / max_latency if max_latency != 0 else (0 if latency == 0 else float('inf'))
        norm_embCarbon = embCarbon / max_embCarbon if max_embCarbon != 0 else (0 if embCarbon == 0 else float('inf')) 
        norm_opeCarbon = opeCarbon / max_opeCarbon if max_opeCarbon != 0 else (0 if opeCarbon == 0 else float('inf')) 
    
    ################
   
    ################
    ## Method 3: Use value - min / max - min
    if calibration_mode == 'max_minus_min':
        print(f"[INFO] Using max - min method for cost normalization") if print_info else None
        max_energy = cost_averages.get('energy_max', 1.0) 
        max_area = cost_averages.get('area_max', 1.0)
        max_dollar = cost_averages.get('cost_max', 1.0)
        max_latency = cost_averages.get('latency_max', 1.0)
        max_embCarbon = cost_averages.get('embCarbon_max', 1.0) 
        max_opeCarbon = cost_averages.get('opeCarbon_max', 1.0) 
        min_energy = cost_averages.get('energy_min', 1.0) 
        min_area = cost_averages.get('area_min', 1.0)
        min_dollar = cost_averages.get('cost_min', 1.0)
        min_latency = cost_averages.get('latency_min', 1.0)
        min_embCarbon = cost_averages.get('embCarbon_min', 1.0)
        min_opeCarbon = cost_averages.get('opeCarbon_min', 1.0)
    
        norm_energy = (energy_f - min_energy)/ (max_energy - min_energy) if max_energy != 0 else (0 if energy == 0 else float('inf')) 
        norm_area = (area - min_area) / (max_area - min_area) if max_area != 0 else (0 if area == 0 else float('inf'))
        norm_dollar = (dollar - min_dollar) / (max_dollar - min_dollar) if max_dollar != 0 else (0 if dollar == 0 else float('inf'))
        norm_latency = (latency - min_latency) / (max_latency - min_latency) if max_latency != 0 else (0 if latency == 0 else float('inf')) 
        norm_embCarbon = (embCarbon - min_embCarbon) / (max_embCarbon - min_embCarbon) if max_embCarbon != 0 else (0 if embCarbon == 0 else float('inf'))    
        norm_opeCarbon = (opeCarbon - min_opeCarbon) / (max_opeCarbon - min_opeCarbon) if max_opeCarbon != 0 else (0 if opeCarbon == 0 else float('inf'))    
    ################ 
    
    ################ 
    # Method 4: Use value - mean / std
    if calibration_mode == 'mean_std':
        print(f"[INFO] Using mean - std method for cost normalization") if print_info else None 
        mean_energy = cost_averages.get('energy_mean', 1.0) 
        std_energy = cost_averages.get('energy_stddev', 1.0) 
        mean_area = cost_averages.get('area_mean', 1.0)
        std_area = cost_averages.get('area_stddev', 1.0)
        mean_dollar = cost_averages.get('cost_mean', 1.0)
        std_dollar = cost_averages.get('cost_stddev', 1.0)
        mean_latency = cost_averages.get('latency_mean', 1.0)
        std_latency = cost_averages.get('latency_stddev', 1.0)
        mean_embCarbon = cost_averages.get('embCarbon_mean', 1.0)
        std_embCarbon = cost_averages.get('embCarbon_stddev', 1.0)
        mean_opeCarbon = cost_averages.get('opeCarbon_mean', 1.0)
        std_opeCarbon = cost_averages.get('opeCarbon_stddev', 1.0)
        
        norm_energy = (energy_f - mean_energy) / std_energy if std_energy != 0 else (0 if energy_f == 0 else float('inf')) 
        norm_area = (area - mean_area) / std_area if std_area != 0 else (0 if area == 0 else float('inf'))
        norm_dollar = (dollar - mean_dollar) / std_dollar if std_dollar != 0 else (0 if dollar == 0 else float('inf'))
        norm_latency = (latency - mean_latency) / std_latency if std_latency != 0 else (0 if latency == 0 else float('inf'))
        norm_embCarbon = (embCarbon - mean_embCarbon) / std_embCarbon if std_embCarbon != 0 else (0 if embCarbon == 0 else float('inf'))
        norm_opeCarbon = (opeCarbon - mean_opeCarbon) / std_opeCarbon if std_opeCarbon != 0 else (0 if opeCarbon == 0 else float('inf'))
    ################ 
    
    ################ 
    # Method 5: Use value - min / median
    if calibration_mode == 'min_median': 
        print(f"[INFO] Using value-min / median method for cost normalization")  if print_info else None
        min_energy = cost_averages.get('energy_min', 1.0) 
        min_area = cost_averages.get('area_min', 1.0)
        min_dollar = cost_averages.get('cost_min', 1.0)
        min_latency = cost_averages.get('latency_min', 1.0)
        min_embCarbon = cost_averages.get('embCarbon_min', 1.0)
        min_opeCarbon = cost_averages.get('opeCarbon_min', 1.0)
        median_energy = cost_averages.get('energy_median', 1.0) 
        median_area = cost_averages.get('area_median', 1.0)
        median_dollar = cost_averages.get('cost_median', 1.0)
        median_latency = cost_averages.get('latency_median', 1.0)
        median_embCarbon = cost_averages.get('embCarbon_median', 1.0)
        median_opeCarbon = cost_averages.get('opeCarbon_median', 1.0)

        norm_energy = (energy_f - min_energy) / median_energy  if median_energy != 0 else (0 if energy_f == 0 else float('inf')) 
        norm_area = (area - min_area) / median_area if median_area != 0 else (0 if area == 0 else float('inf'))
        norm_dollar = (dollar - min_dollar) / median_dollar if median_dollar != 0 else (0 if dollar == 0 else float('inf'))
        norm_latency = (latency - min_latency) / median_latency if median_latency != 0 else (0 if latency == 0 else float('inf'))
        norm_embCarbon = (embCarbon - min_embCarbon) / median_embCarbon if median_embCarbon != 0 else (0 if embCarbon == 0 else float('inf'))
        norm_opeCarbon = (opeCarbon - min_opeCarbon) / median_opeCarbon if median_opeCarbon != 0 else (0 if opeCarbon == 0 else float('inf'))
    ################ 
    
    
    
   
    max_energy      = cost_averages.get('energy_max', 1.0) 
    max_area        = cost_averages.get('area_max', 1.0)
    max_dollar      = cost_averages.get('cost_max', 1.0)
    max_latency     = cost_averages.get('latency_max', 1.0)
    max_embCarbon   = cost_averages.get('embCarbon_max', 1.0) 
    max_opeCarbon   = cost_averages.get('opeCarbon_max', 1.0) 
    min_energy      = cost_averages.get('energy_min', 1.0) 
    min_area        = cost_averages.get('area_min', 1.0)
    min_dollar      = cost_averages.get('cost_min', 1.0)
    min_latency     = cost_averages.get('latency_min', 1.0)
    min_embCarbon   = cost_averages.get('embCarbon_min', 1.0)
    min_opeCarbon   = cost_averages.get('opeCarbon_min', 1.0)
    energy_scale    = max_energy/min_energy
    area_scale      = max_area/min_area
    dollar_scale    = max_dollar/min_dollar
    latency_scale   = max_latency/min_latency
    embCarbon_scale = max_embCarbon/min_embCarbon
    opeCarbon_scale = max_opeCarbon/min_opeCarbon
    
      
    
    
    final_cost = (energy_coff*energy_scale * norm_energy) + (area_coeff*area_scale * norm_area) + (cost_coeff*dollar_scale * norm_dollar)  + (perf_coeff*latency_scale * norm_latency) + (embc_coeff*embCarbon_scale * norm_embCarbon) + (opec_coeff*opeCarbon_scale * norm_opeCarbon) #TODO 1: Remove Power and add Energy
    
    normalized_cost_dict = {
        'norm_energy': norm_energy, 
        'norm_area': norm_area, 
        'norm_dollar': norm_dollar,
        'norm_latency': norm_latency,
        'norm_embCarbon': norm_embCarbon,
        'norm_opeCarbon': norm_opeCarbon,
        'norm_energy_scale': energy_scale*energy_coff*norm_energy, 
        'norm_area_scale': area_scale*area_coeff*norm_area,
        'norm_dollar_scale': dollar_scale*cost_coeff*norm_dollar,
        'norm_latency_scale': latency_scale*perf_coeff*norm_latency,
        'norm_embCarbon_scale': embCarbon_scale*embc_coeff*norm_embCarbon,
        'norm_opeCarbon_scale': opeCarbon_scale*opec_coeff*norm_opeCarbon,
        'energy_scale': energy_scale,
        'area_scale': area_scale,
        'dollar_scale': dollar_scale,
        'latency_scale': latency_scale,
        'embCarbon_scale': embCarbon_scale,
        'opeCarbon_scale': opeCarbon_scale,
    }  
    
    if print_info:
        print("\n[INFO] --- Cost Calculation Debug ---")
        print(f"[INFO] Raw Costs: {raw_cost_dict}")
        print(f"[INFO] Averages Used for Normalization: {cost_averages}")
        print(f"[INFO] Normalized Cost Contributions: {normalized_cost_dict}")
        print(f"[INFO] Final Normalized Cost: {final_cost}")
        print("------------------------")
    
    return final_cost,normalized_cost_dict,raw_cost_dict


############################################

################ WL Move Starts ############################
def mutate_wl_mapping(system_config):
  updated_config = copy.deepcopy(system_config)

  try:
      mapping_details = updated_config['WL_mapping']['mapping']
  except (KeyError, TypeError):
      print("Error: Could not find 'mapping':'mapping' structure in the input data.")
      return system_config # Return original if structure is wrong

  # List of keys that can be modified
  modifiable_keys = [
      "if_splitting_k",
      "dataflow",
      "assign_workload_in_ascending_order"
  ]

  #dataflow key
  dataflow_options = ["ws", "os", "is"]

  # Randomly select one key to change
  key_to_change = random.choice(modifiable_keys)
  current_value = mapping_details[key_to_change]
  new_value = None

  # Determine the new value
  if key_to_change == "dataflow":
      current_df_value = current_value[0] # Get the string from the list
      possible_new_values = [df for df in dataflow_options if df != current_df_value]
      if possible_new_values:
          new_value = [random.choice(possible_new_values)]
      else:
          new_value = current_value
  else:
      new_value = 1 - current_value

  mapping_details[key_to_change] = new_value

  return updated_config
################ WL Move Ends ############################

################ Arch Tech Node Starts ##################
def is_stack_valid(stack_chiplet_ids, all_chiplet_props, stack_diff_size):
    
    # An empty or single-chiplet "stack" is always valid by definition.
    if not stack_chiplet_ids or len(stack_chiplet_ids) < 2:
        return True
    
    # --- Rule 1: Homogeneous Stacking ---
    # If stack_diff_size is False, all chiplets in the stack must have the
    # same systolic array size and technology node.
    if not stack_diff_size:
        # Get the properties of the base chiplet to use as the reference.
        base_chiplet_props = all_chiplet_props[stack_chiplet_ids[0]]
        base_key = (base_chiplet_props['sys_array_size'], base_chiplet_props['tech_node'])
        
        # Iterate through the rest of the chiplets in the stack.
        for i in range(1, len(stack_chiplet_ids)):
            current_props = all_chiplet_props[stack_chiplet_ids[i]]
            current_key = (current_props['sys_array_size'], current_props['tech_node'])
            # If any chiplet's properties do not match the base, the stack is invalid.
            if current_key != base_key:
                return False
    # --- Rule 2: Heterogeneous (Area-based) Stacking ---
    # If stack_diff_size is True, any chiplet placed on top of another must
    # have an area that is less than or equal to the one below it.
    else:
        # Iterate through adjacent pairs of chiplets from bottom to top.
        for i in range(len(stack_chiplet_ids) - 1):
            bottom_chiplet_area = all_chiplet_props[stack_chiplet_ids[i]]['area']
            top_chiplet_area = all_chiplet_props[stack_chiplet_ids[i+1]]['area']
            # If a top chiplet is larger than the bottom chiplet, the stack is invalid.
            if top_chiplet_area > bottom_chiplet_area:
                return False
            
    # If all checks passed, the stack is valid.
    return True

def _validate_and_apply_mutation(temp_config, stack_diff_size):
    pkg_info = temp_config.get("pkg", {})
    pkg_type = pkg_info.get("HI_pkg_type", "2d")
    
    if "3d" not in pkg_type: # No stacking rules to check
        all_chiplets = {k: v for k, v in temp_config.items() if k.startswith('Chiplet_')}
        current_mem_conn = temp_config['pkg']['mem_pkg_conn']
        temp_config['pkg']['mem_pkg_conn'] = generate_mem_channel_distribution(all_chiplets, pkg_info['mem_pkg_conn']['mem_type'])
        return temp_config

    connections = pkg_info.get("inter_pkg_conn", [])
    if not isinstance(connections, list):
        return temp_config # Not a complex package

    to_from_map = {c['from']: c['to'] for c in connections if c['to'] != 'na' and 'stack' in c['loc']}
    all_bases = {c['from'] for c in connections if 'base' in c['loc']}
    
    for base in all_bases:
        current_stack = [base]
        while current_stack[-1] in to_from_map:
            current_stack.append(to_from_map[current_stack[-1]])
        if not is_stack_valid(current_stack, temp_config, stack_diff_size):
            return None # Invalid stack found
    
    # If all stacks are valid, update memory channels and return
    all_chiplets = {k: v for k, v in temp_config.items() if k.startswith('Chiplet_')}
    temp_config['pkg']['mem_pkg_conn'] = generate_mem_channel_distribution(all_chiplets, pkg_info['mem_pkg_conn']['mem_type'])
    return temp_config

def mutate_arch_sram_buf(system_config, sram_buf_sizes):
    new_config = copy.deepcopy(system_config)
    chiplet_ids = [k for k in new_config.keys() if k.startswith('Chiplet_')]
    if not chiplet_ids:
        return new_config

    chiplet_to_mutate = random.choice(chiplet_ids)
    
    # Get current properties
    chiplet_props = new_config[chiplet_to_mutate]
    current_sys_array = chiplet_props['sys_array_size']
    current_sram_buf = chiplet_props.get('sram_buf')

    # Find possible new options
    valid_sram_options = sram_buf_sizes.get(current_sys_array, [])
    possible_new_options = [s for s in valid_sram_options if s != current_sram_buf]

    if possible_new_options:
        # Apply the mutation
        new_sram_buf = random.choice(possible_new_options)
        new_config[chiplet_to_mutate]['sram_buf'] = new_sram_buf
        
        #Updating area and power with new sram buf size
        sys_array_size = new_config[chiplet_to_mutate]['sys_array_size']
        tech_node = new_config[chiplet_to_mutate]['tech_node']
        new_sram_area,_ = get_sram_area_energy(new_sram_buf, tech_node)
        new_area, new_power = get_area_power(sys_array_size,tech_node)
        new_config[chiplet_to_mutate]['area'] = new_area+new_sram_area
        new_config[chiplet_to_mutate]['power'] = new_power

    return new_config

def mutate_arch_tech_node(system_config, params, stack_diff_size=True, max_retries=10):
    
    chiplet_ids = [k for k in system_config.keys() if k.startswith('Chiplet_')]
    random.shuffle(chiplet_ids)
    
    for chiplet_to_mutate in chiplet_ids:
        for _ in range(max_retries):
            temp_config = copy.deepcopy(system_config)
            current_tech_node = temp_config[chiplet_to_mutate]['tech_node']
            possible_new_nodes = [n for n in params['tech_nodes'] if n != current_tech_node]
            if not possible_new_nodes: continue
            
            # --- 1. Apply Mutation ---
            new_node = random.choice(possible_new_nodes)
            temp_config[chiplet_to_mutate]['tech_node'] = new_node
            
            # --- 2. Update Dependent Properties (Area & Power) ---
            sys_array_size = temp_config[chiplet_to_mutate]['sys_array_size']
            sram_size = temp_config[chiplet_to_mutate]['sram_buf']
            new_sram_area,_ = get_sram_area_energy(sram_size,new_node)
            new_area, new_power = get_area_power(sys_array_size,new_node) 
            temp_config[chiplet_to_mutate]['area'] = new_area+new_sram_area
            temp_config[chiplet_to_mutate]['power'] = new_power
            
            try:
                chiplets = {k: v for k, v in temp_config.items() if k.startswith('Chiplet_')}
                package_gen = PackageGenerator(chiplets, params['inter_pkg_arch'], params['mem_pkg_arch'], params['protocol_arch'],stack_diff_size=params['stack_diff_size'])
                # Use update mode by passing the existing package
                temp_config['pkg'] = package_gen.generate(existing_pkg=temp_config['pkg'])

                temp_config = update_dict_inter_pkg_topology(temp_config)
            
                print(f"[INFO] Chiplet Tech node update Chiplet {chiplet_to_mutate} to tech_node {new_node}") if print_info else None
                validated_config = _validate_and_apply_mutation(temp_config, stack_diff_size)
                if validated_config:
                    return validated_config
            except:
                print(f"[ERROR] Failed to generate package for chiplet {chiplet_to_mutate} with new tech node {new_node}")
                print("[ERROR] No valid mutation found for tech node")  
                continue
            
    return None # If all attempts fail, return None

def mutate_arch_chiplet_size(system_config, all_sys_array_sizes, params, stack_diff_size=True, max_retries=10):
    
    chiplet_ids = [k for k in system_config.keys() if k.startswith('Chiplet_')]
    random.shuffle(chiplet_ids)

    for chiplet_to_mutate in chiplet_ids:
        for _ in range(max_retries):
            temp_config = copy.deepcopy(system_config)
            current_size = temp_config[chiplet_to_mutate]['sys_array_size']
            possible_new_sizes = [s for s in all_sys_array_sizes if s != current_size]
            if not possible_new_sizes: continue

            # Apply mutation
            new_size = random.choice(possible_new_sizes)
            temp_config[chiplet_to_mutate]['sys_array_size'] = new_size

            # Update dependent properties
            tech_node = temp_config[chiplet_to_mutate]['tech_node']
            
            # Correctly select a valid SRAM buffer size
            sram_buf_options = params.get('sram_buf_sizes', {})
            valid_sram_sizes = sram_buf_options.get(new_size, [256])
            if sram_selection_mode == 'random':
                new_sram_buf = random.choice(valid_sram_sizes)
            else: 
                new_sram_buf = max(valid_sram_sizes) 
            
            new_sram_area,_ = get_sram_area_energy(new_sram_buf,tech_node)
            new_area, new_power = get_area_power(new_size, tech_node)
            temp_config[chiplet_to_mutate]['area'] = new_area+new_sram_area
            temp_config[chiplet_to_mutate]['power'] = new_power
            temp_config[chiplet_to_mutate]['sram_buf'] = new_sram_buf

            try:
                chiplets = {k: v for k, v in temp_config.items() if k.startswith('Chiplet_')}
                package_gen = PackageGenerator(chiplets, params['inter_pkg_arch'], params['mem_pkg_arch'], params['protocol_arch'], stack_diff_size=params['stack_diff_size'])
                
                # Regenerate the package, keeping the old type but allowing interconnects to change
                temp_config['pkg'] = package_gen.generate(
                    hi_pkg_type=temp_config['pkg']['HI_pkg_type'],
                    mem_pkg_type=temp_config['pkg']['mem_pkg_conn']['mem_type'],
                    existing_pkg=temp_config['pkg']
                )
                
                temp_config = update_dict_inter_pkg_topology(temp_config)
                

                print(f"[INFO] Chiplet Tech node update Chiplet {chiplet_to_mutate} to New size {new_size}") if print_info else None
                # Validate and return if successful
                validated_config = _validate_and_apply_mutation(temp_config, stack_diff_size)
                if validated_config:
                    return validated_config
            except: 
                print(f"[ERROR] Failed to generate package for chiplet {chiplet_to_mutate} with new size {new_size}")
                print("[ERROR] No valid mutation found for chiplet size") 
                continue

    return None

def mutate_arch_chiplet_size_cg_mode(system_config, all_sys_array_sizes, params, stack_diff_size=True, max_retries=10):
    
    chiplet_ids = [k for k in system_config.keys() if k.startswith('Chiplet_')]
    if not chiplet_ids:
        return None  # no chiplets to mutate

    for _ in range(max_retries):
        temp_config = copy.deepcopy(system_config)
        
        # Assume all chiplets currently have the same size
        current_size = temp_config[chiplet_ids[0]]['sys_array_size']
        
        # Pick a new size that is different from current
        possible_new_sizes = [s for s in all_sys_array_sizes if s != current_size]
        if not possible_new_sizes:
            return None  # no valid new size available
        

        # Apply mutation
        new_size = random.choice(possible_new_sizes)
        
        # Apply the new size to ALL chiplets
        for chiplet_id in chiplet_ids:
            temp_config[chiplet_id]['sys_array_size'] = new_size

            # Update dependent properties
            tech_node = temp_config[chiplet_id]['tech_node']
            
            # Correctly select a valid SRAM buffer size
            sram_buf_options = params.get('sram_buf_sizes', {})
            valid_sram_sizes = sram_buf_options.get(new_size, [256])
            if sram_selection_mode == 'random':
                new_sram_buf = random.choice(valid_sram_sizes)
            else: 
                new_sram_buf = max(valid_sram_sizes) 

            new_sram_area,_ = get_sram_area_energy(new_sram_buf,tech_node)
            new_area, new_power = get_area_power(new_size, tech_node)
            temp_config[chiplet_id]['area'] = new_area+new_sram_area
            temp_config[chiplet_id]['power'] = new_power
            temp_config[chiplet_id]['sram_buf'] = new_sram_buf
            
        try:
            chiplets = {k: v for k, v in temp_config.items() if k.startswith('Chiplet_')}
            package_gen = PackageGenerator(chiplets, params['inter_pkg_arch'], params['mem_pkg_arch'], params['protocol_arch'], stack_diff_size=params['stack_diff_size'])
            
            # Regenerate the package, keeping the old type but allowing interconnects to change
            temp_config['pkg'] = package_gen.generate(
                hi_pkg_type=temp_config['pkg']['HI_pkg_type'],
                mem_pkg_type=temp_config['pkg']['mem_pkg_conn']['mem_type'],
                existing_pkg=temp_config['pkg']
            )
            
            temp_config = update_dict_inter_pkg_topology(temp_config)
            

            print(f"[INFO] Chiplet Tech node update Chiplet to New size {new_size}") if print_info else None
            # Validate and return if successful
            validated_config = _validate_and_apply_mutation(temp_config, stack_diff_size)
            if validated_config:
                return validated_config
        except: 
            print(f"[ERROR] Failed to generate package for chiplet  with new size {new_size}")
            print("[ERROR] No valid mutation found for chiplet size") 
            continue

    return None

def mutate_arch_mem_pkg(system_config, all_mem_pkg_options):
    
    new_config = copy.deepcopy(system_config)
    chiplets = {k: v for k, v in new_config.items() if k.startswith('Chiplet_')}
    num_chiplets = len(chiplets)
    current_mem_type = new_config['pkg']['mem_pkg_conn']['mem_type']
    
    # Determine the valid pool of new memory types based on the rules
    if num_chiplets == 1:
        # Filter for DDR types
        valid_options = [t for t in all_mem_pkg_options if 'ddr' in t.lower()]
    else: # 2 or more chiplets
        # Filter for HBM types
        ddr_options = [t for t in all_mem_pkg_options if 'ddr' in t.lower()]
        hbm_options = [t for t in all_mem_pkg_options if 'hbm' in t.lower()]
        valid_options = ddr_options + hbm_options
    
    # Find possible new types that are different from the current one
    possible_new_types = [t for t in valid_options if t != current_mem_type]
    
    if not possible_new_types:
        return new_config # No other valid options, return original
    
    
    new_mem_type = random.choice(possible_new_types)
    
    # Use the new helper function to generate the new distribution
    new_mem_distribution = generate_mem_channel_distribution(chiplets, new_mem_type)
    
    new_config['pkg']['mem_pkg_conn'] = new_mem_distribution
    print(f"[INFO] Changing mem pkg type from {current_mem_type} to {new_mem_type}") if print_info else None 
    return new_config

def mutate_arch_inter_pkg(system_config, all_inter_pkg_options, all_protocol_options):
    
    new_config = copy.deepcopy(system_config)
    pkg = new_config['pkg']
    hi_pkg_type = pkg['HI_pkg_type']
    
    opts_2_5d = [p for p in all_inter_pkg_options if p.startswith("2.5d")]
    opts_3d = [p for p in all_inter_pkg_options if p.startswith("3d")]

    if hi_pkg_type == "2d":
        print(f"[INFO] Inter Pkg changes 2D so NO CHANGE")
        print("[ERROR] No valid mutation found for interconnect package type, since current is 2D") 
        return None # No change possible

    elif hi_pkg_type == "2.5d":
        #current_conn = pkg['inter_pkg_conn']
        current_conn = pkg['inter_pkg_conn'][0]['connection_type']
        possible_new_conns = [c for c in opts_2_5d if c != current_conn]
        #R new_conn = random.choice(possible_new_conns)
        if possible_new_conns:
            new_conn = random.choice(possible_new_conns)
            # pkg['inter_pkg_conn'] = new_conn
            # updated_protocol_is_valid = _validate_protocols(protocol_3d=pkg['protocol_3d'], protocol_2_5d=pkg['protocol_2.5d'], inter_conn= new_conn)
            # if not updated_protocol_is_valid:
            #     p3d, p2_5d = _determine_protocols(hi_pkg_type, pkg['inter_pkg_conn'], all_protocol_options)
            #     pkg['protocol_3d'] = p3d
            #     pkg['protocol_2.5d'] = p2_5d
            # print(f"[INFO] 2.5D Inter Pkg Changes to {new_conn}") if print_arch_inter_pkg_change else None
            for conn in pkg['inter_pkg_conn']:
                conn['connection_type'] = new_conn
                updated_protocol_is_valid = _validate_protocols(protocol_3d=pkg['protocol_3d'], protocol_2_5d=pkg['protocol_2.5d'], inter_conn= new_conn)
                if not updated_protocol_is_valid:
                    p3d, p2_5d = _determine_protocols(hi_pkg_type, pkg['inter_pkg_conn'], all_protocol_options)
                    pkg['protocol_3d'] = p3d
                    pkg['protocol_2.5d'] = p2_5d
                print(f"[INFO] 2.5D Inter Pkg cahgnes to {new_conn}") if print_info else None
        else:
            print("[ERROR] No valid mutation found for interconnect package type, since current is 2.5D") 
            return None
        return new_config

    elif hi_pkg_type == "3d":
        # Assumes all 3D connections should be mutated to the same new type
        current_conn_type = pkg['inter_pkg_conn'][0]['connection_type']
        possible_new_types = [t for t in opts_3d if t != current_conn_type]
        if possible_new_types:
            new_type = random.choice(possible_new_types)
            for conn in pkg['inter_pkg_conn']:
                conn['connection_type'] = new_type
                updated_protocol_is_valid = _validate_protocols(protocol_3d=pkg['protocol_3d'], protocol_2_5d=pkg['protocol_2.5d'], inter_conn= new_type)
                if not updated_protocol_is_valid:
                    p3d, p2_5d = _determine_protocols(hi_pkg_type, pkg['inter_pkg_conn'], all_protocol_options)
                    pkg['protocol_3d'] = p3d
                    pkg['protocol_2.5d'] = p2_5d
                print(f"[INFO] 3D Inter Pkg cahgnes to {new_type}") if print_info else None
        return new_config
    
    elif hi_pkg_type == "2.5d_3d":
        connections = pkg['inter_pkg_conn']
        if not connections: return new_config
        
        # Decide whether to mutate the 2.5D or 3D part of the interconnect
        mutation_target = random.choice(['2.5d', '3d'])
        
        if mutation_target == '3d':
            # Find the current 3D type and change all instances of it
            current_3d_type = next((c['connection_type'] for c in connections if c['connection_type'] in opts_3d), None)
            if current_3d_type:
                possible_new_types = [t for t in opts_3d if t != current_3d_type]
                if possible_new_types:
                    new_type = random.choice(possible_new_types)
                    for conn in connections:
                        if conn['connection_type'] in opts_3d:
                            conn['connection_type'] = new_type
                            print(f"[INFO] 2.5D-3D Inter Pkg cahgnes to {new_type}") if print_info else None
                            
        elif mutation_target == '2.5d':
            # Find the current 2.5D type and change all instances of it
            current_2_5d_type = next((c['connection_type'] for c in connections if c['connection_type'] in opts_2_5d), None)
            if current_2_5d_type:
                possible_new_types = [t for t in opts_2_5d if t != current_2_5d_type]
                if possible_new_types:
                    new_type = random.choice(possible_new_types)
                    for conn in connections:
                        if conn['connection_type'] in opts_2_5d:
                            conn['connection_type'] = new_type
                            print(f"[INFO] 2.5D-3D Inter Pkg cahgnes to {new_type}") if print_info else None
        
        # Recalculate protocols after any potential mutation
        updated_protocol_is_valid = _validate_protocols(protocol_3d=pkg['protocol_3d'], protocol_2_5d=pkg['protocol_2.5d'], inter_conn= new_type)
        if not updated_protocol_is_valid:
            p3d, p2_5d = _determine_protocols(hi_pkg_type, pkg['inter_pkg_conn'], all_protocol_options)
            pkg['protocol_3d'] = p3d
            pkg['protocol_2.5d'] = p2_5d
        
        return new_config
        
    # If no valid mutation was possible, return None
    print("[ERROR] No valid mutation found for interconnect package type")
    return None


def mutate_arch_protocol(system_config, all_protocol_options):
    
    new_config = copy.deepcopy(system_config)
    pkg = new_config.get('pkg', {})
    
    current_protocol = pkg.get('protocol_2.5d')
    
    # Define the group of mutable, advanced protocols
    advanced_protocols = [p for p in all_protocol_options if p in ["aib", "bow", "ucie_adv"]]
    
    # Only mutate if the current protocol is one of the advanced types
    if current_protocol in advanced_protocols:
        # Find other possible options within the advanced group
        possible_new_protocols = [p for p in advanced_protocols if p != current_protocol]
        
        if possible_new_protocols:
            # If other options exist, pick one and apply the mutation
            new_protocol = random.choice(possible_new_protocols)
            pkg['protocol_2.5d'] = new_protocol
            return new_config
            
    print("[ERROR] No valid mutation found for protocol type since its either na(2D or 3D) OR ucie_std (RDL) ") if print_info else None
    return None


def mutate_arch_chiplet_num(system_config, params, max_retries=10):
    
    for _ in range(max_retries):
        temp_config = copy.deepcopy(system_config)
        
        chiplets = {k: v for k, v in temp_config.items() if k.startswith('Chiplet_')}
        num_chiplets = len(chiplets)

        # Decide whether to add or delete
        can_add = num_chiplets < params['max_chiplet']
        can_delete = num_chiplets > 1
        
        if can_add and can_delete:
            operation = random.choice(['add', 'delete'])
        elif can_add:
            operation = 'add'
        elif can_delete:
            operation = 'delete'
        else:
            return None # Cannot perform any operation
        print(f"[INFO] Chiplet Num Changes, will do *** {operation} *** of chiplet") if print_info else None

        # Perform the operation
        if operation == 'add':
            new_chiplet_id = f"Chiplet_{num_chiplets + 1}"
            new_sys_array = random.choice(params['sys_array'])
            new_tech_node = random.choice(params['tech_nodes'])
            
            # Correctly select a valid SRAM buffer size
            sram_buf_options = params.get('sram_buf_sizes', {})
            valid_sram_sizes = sram_buf_options.get(new_sys_array, [256])
            if sram_selection_mode == 'random':
                new_sram_buf = random.choice(valid_sram_sizes)
            else:
                new_sram_buf = max(valid_sram_sizes) # Select the largest SRAM buffer size
            
            new_sram_area,_ = get_sram_area_energy(new_sram_buf,new_tech_node)
            new_area, new_power = get_area_power(new_sys_array, new_tech_node)
            chiplets[new_chiplet_id] = {
                "tech_node": new_tech_node,
                "sys_array_size": new_sys_array,
                "sram_buf": new_sram_buf,
                "area": new_area+new_sram_area,
                "power": new_power
            }
        else: # delete
            chiplet_to_delete = random.choice(list(chiplets.keys()))
            del chiplets[chiplet_to_delete]
            # Re-index remaining chiplets to be contiguous
            sorted_chiplets = sorted(chiplets.items(), key=lambda item: int(item[0].split('_')[1]))
            chiplets = {f"Chiplet_{i+1}": v for i, (k, v) in enumerate(sorted_chiplets)}

        try:
            package_gen = PackageGenerator(chiplets, params['inter_pkg_arch'], params['mem_pkg_arch'], params['protocol_arch'], stack_diff_size=params['stack_diff_size'])
            new_package = package_gen.generate()
            
            # Build the new config
            final_mutated_config = {}
            final_mutated_config.update(chiplets)
            final_mutated_config['pkg'] = new_package
            final_mutated_config['WL_mapping'] = temp_config['WL_mapping']
            
            final_mutated_config = update_dict_inter_pkg_topology(final_mutated_config)
            
            return final_mutated_config
        except ValueError:
            # This combination was invalid, try again
            continue
    print("[ERROR] Failed to mutate chiplet number after maximum retries.")
    return None # Failed to find a valid mutation after retries

def accept_move_func(cost_diff, temp):
    # If the new solution is better (i.e., cost is lower), accept it immediately
    if cost_diff < 0:
        return True, 1  # Accepted due to improvement (reason code 1)
    else:
        # Otherwise, accept it probabilistically based on temperature (simulated annealing)
        # The higher the cost_diff or the lower the temp, the less likely to accept
        if random.uniform(0, 1) < math.exp(-cost_diff / temp):
            return True, 2  # Accepted probabilistically (reason code 2)
        else:
            return False, 3  # Rejected (reason code 3)
        
def dump_results(sa_info_df, sim_result_df, best_arch, best_cost, file_run_name):
    import matplotlib.pyplot as plt
    from pathlib import Path
    subdir = "cfg/gen_arch"
    #out_dir = Path(__file__).resolve().parent / subdir
    #out_dir.mkdir(parents=True, exist_ok=True)
    
    rundir = f"{subdir}/{file_run_name}"
    os.makedirs(rundir, exist_ok=True) # Ensure the directory exists
    print(f"[INFO] Output directory created: {rundir}")
    
    #R file_run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_run_time = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    metric_file_name = f"sa_metrics_{file_run_name}_{file_run_time}.csv"
    arch_file_name = f"sa_arch_detail_{file_run_name}_{file_run_time}.csv"
    plot_file_name = f"cost_v_iteration_{file_run_name}_{file_run_time}.png"
    best_arch_file_name = f"best_arch_{file_run_name}_{file_run_time}.json"
    
    #Dump sim_annelaing metric info
    print(sa_info_df)
    print(f"[INFO] Dumping SA Detail csv ...... ")
    #R sa_info_df.to_csv("cfg/gen_arch/sa_detail_info.csv", index=False)
    sa_info_df.to_csv(f"{rundir}/{metric_file_name}", index=False)
    print("[INFO] Done dumping SA Detail csv")
    
    #Dump sim_annealing arch info
    print(f"[INFO] Dumping Simulation Results csv ...... ")
    print(sim_result_df.head())
    #R sim_result_df.to_csv("cfg/gen_arch/simulation_results.csv", index=False)
    sim_result_df.to_csv(f"{rundir}/{arch_file_name}", index=False)
    print("[INFO] Done dumping Simulation Results csv") 
    #R print("\nDataFrame after reordering columns (first 5 rows):")
    #R print(df.head())
    #R output_filename = 'cfg/gen_arch/simulation_results_pandas.csv'
    #R print(f"\nWriting DataFrame to '{output_filename}'...")
    #R df.to_csv(output_filename, index=False) 
    
    #Plot for sim_annealing cehck
    plt.figure(figsize=(8,5))
    plt.plot(sa_info_df['SA_run_loop'],sa_info_df['best_cost'],linestyle='-')
    plt.xlabel('SA_run_loop')
    plt.ylabel('Best cost')
    plt.title('Best Cost vs SA Run Loop')
    #R plt.savefig('cfg/gen_arch/cost_v_iteration.png',dpi=300,bbox_inches="tight")
    plt.savefig(f"{rundir}/{plot_file_name}",dpi=300,bbox_inches="tight")
    
    #R write_solution_to_json(solution_data=best_arch,output_json_file='cfg/gen_arch/best-arch.json')
    write_solution_to_json(solution_data=best_arch,output_json_file=f"{rundir}/{best_arch_file_name}")
    print(f"[INFO] Best arch cost is {best_cost}")
    
    return

def d2d_bw_calc(pkg, protocol, area, is_3d, node=7, use_node_based_rate=True):
    #Units in Gbps
        
    #D2D_pitch_pkg 
    # https://www.tomshardware.com/pc-components/cpus/intel-details-new-advanced-packaging-breakthroughs-emib-t-paves-the-way-for-hbm4-and-increased-ucie-bandwidth#:~:text=Additionally%2C%20the%20first%20generation%20of,to%20a%2045%2Dmicron%20pitch. 
    #https://www.tomshardware.com/pc-components/cpus/intel-details-new-advanced-packaging-breakthroughs-emib-t-paves-the-way-for-hbm4-and-increased-ucie-bandwidth
  
    
    with open('cfg/parameters/d2d_input.json', 'r') as f:
        data = json.load(f)
    D2D_RATES_BY_NODE = data.get("D2D_RATES_BY_NODE", {})
    eff_protocol = data.get("eff_protocol", {})
    D2D_pitch_pkg = data.get("D2D_pitch_pkg", {})  
    
    converted_rates = {}
    for rate_protocol, node_rates in D2D_RATES_BY_NODE.items():
        converted_rates[rate_protocol] = {
            int(node): rate for node, rate in node_rates.items()
        }
    D2D_RATES_BY_NODE = converted_rates

    
    #R if is_3d:
    #R     # For 3D, we assume area BW 
    #R     #num_bumps = area / (D2D_pitch_pkg[pkg] ** 2)
    #R     num_bumps = area / ((D2D_pitch_pkg[pkg]/1000) ** 2)
    #R else: 
    #R     # For 2D, we assume linear BW
    #R     length = area ** 0.5  # Assuming area is square for simplicity
    #R     #num_bumps = length / D2D_pitch_pkg[pkg]  # Number of bumps along one dimension
    #R     num_bumps = length / (D2D_pitch_pkg[pkg]/1000)  # Number of bumps along one dimension
    #R bw_Gbps = num_bumps * D2D_data_rate_per_pin[protocol] * eff_protocol[protocol]  # Bandwidth in Gbps
    #R bw_GBps = bw_Gbps / 8  # Convert Gbps to GBps
    
    # 1. Determine the data rate based on the knob
    data_rate = 0
    try:
        node_key = int("".join(filter(str.isdigit, str(node))))
        data_rate = D2D_RATES_BY_NODE[protocol][node_key]
    except (KeyError, ValueError):
        assert False, f"[D2D Error] Invalid protocol/node combination: protocol={protocol}, node={node}"
        return 0
    
    # 2. Calculate the number of connections
    try:
        pitch_mm = D2D_pitch_pkg[pkg] / 1000.0  # Convert pitch from um to mm
        if is_3d:
            num_bumps = area / (pitch_mm ** 2)
        else:
            length = area ** 0.5
            num_bumps = length / pitch_mm
    except KeyError:
        print(f"[D2D Error] Invalid package type: {pkg}")
        return 0

    # 3. Calculate total bandwidth
    bw_Gbps = num_bumps * data_rate * eff_protocol[protocol]
    bw_GBps = bw_Gbps / 8
    
    #R print(f"[D2D Debug ONLY] D2D Bandwidth Calculation: pkg={pkg}, protocol={protocol}, area={area}, is_3d={is_3d}")
    #R print(f"[D2D Debug ONLY] D2D Bandwidth Calculation: bwGBps = {bw_GBps} GB/s")
    
    print(f"[INFO] Pkg: {pkg}, Protocol: {protocol}, Area: {area}, 3D: {is_3d}") if print_info else None
    print(f"[INFO] BW = {bw_GBps:.2f} GB/s") if print_info else None
    
    return bw_GBps



def calculate_memory_bandwidth(solution_data, return_unified_bw=True):
   
    # 1. Define Bandwidth Mapping (GB/s per channel)
   
    with open('cfg/parameters/mem_bw.json', 'r') as f:
        data = json.load(f)
    bandwidth_map = data.get("bandwidth_map", {})

    # 2. Extract Data (Safely)
    try:
        mem_conn = solution_data['pkg']['mem_pkg_conn']
    except (KeyError, TypeError):
        print("[ERROR] Could not find 'pkg':'mem_pkg_conn' structure in the input data.")
        return None

    # 3. Identify Memory Type
    mem_type = mem_conn.get('mem_type', 'unknown').lower()
    
    if mem_type not in bandwidth_map:
        print(f"[ERROR] Bandwidth for mem_type '{mem_type}' is not defined.")
        return None

    # 4. Extract Channel Counts
    chiplet_channels = {
        key: value
        for key, value in mem_conn.items()
        if key.startswith("Chiplet_") and isinstance(value, (int, float))
    }

    # 5. Calculate Bandwidth for each chiplet individually
    chiplet_bandwidth_dict = {}
    for chip_id, channels in chiplet_channels.items():
        chiplet_config = solution_data.get(chip_id, {})
        # FPGA chiplets are non-GEMM endpoints and have no systolic-array
        # memory-bandwidth entry; they are excluded from the SA bandwidth list.
        if str(chiplet_config.get('chiplet_type', 'systolic_array')).lower() == 'fpga':
            continue
        # Get the systolic array size for this specific chiplet
        sys_array_size = chiplet_config.get('sys_array_size')
        if not sys_array_size:
            print(f"[ERROR] Could not find sys_array_size for {chip_id}.")
            chiplet_bandwidth_dict[chip_id] = 0
            continue
            
        # Look up the bandwidth per channel using both mem_type and sys_array_size
        bw_per_channel = bandwidth_map[mem_type].get(sys_array_size)
        if bw_per_channel is None:
            print(f"[ERROR] No bandwidth defined for combination: {mem_type} and {sys_array_size}.")
            chiplet_bandwidth_dict[chip_id] = 0
            continue
            
        # Calculate the total bandwidth for this chiplet
        if return_unified_bw:
            chiplet_bandwidth_dict[chip_id] = bw_per_channel #channels * bw_per_channel
        else:
            chiplet_bandwidth_dict[chip_id] = channels * bw_per_channel
        #print(chiplet_bandwidth_dict)


    # 6. Create a sorted list of bandwidths 
    def get_chiplet_number(chiplet_id):
        match = re.search(r'\d+$', chiplet_id)
        return int(match.group()) if match else -1

    sorted_chiplet_ids = sorted(chiplet_bandwidth_dict.keys(), key=get_chiplet_number)
    bandwidth_list = [chiplet_bandwidth_dict[chip_id] for chip_id in sorted_chiplet_ids] 

    return bandwidth_list

def find_run_time(start_time, end_time):
    run_time_sec = end_time - start_time
    minuites = int(run_time_sec // 60)
    seconds = int(run_time_sec % 60)
    print(f"Runtime : {minuites} min, {seconds} sec")
    return 

def process_iteration_wide(current_config: dict, iteration_id: int) -> dict:
    
    flat_row = {'iteration_id': iteration_id}

    def flatten_recursive(data, prefix=''):
        if isinstance(data, dict):
            for key, value in data.items():
                flatten_recursive(value, f"{prefix}{key}_")
        elif isinstance(data, list):
            flat_row[prefix[:-1]] = ",".join(map(str, data))
        else:
            flat_row[prefix[:-1]] = data
    
    flatten_recursive(current_config)
    return flat_row 

def process_arch_details_dump(all_rows_data):
    print(f"[DBG] End of final iteration $$$$$$$$$$$$$")    
    print("Creating DataFrame...")
    df = pd.DataFrame(all_rows_data)
    print("--- Reordering Columns ---")
    # Get the list of all columns from the DataFrame
    all_cols = df.columns.tolist()
    # Create lists for each category based on the prefix of the column name
    # We also sort each list alphabetically for consistent ordering within the category
    iter_cols = sorted([col for col in all_cols if col.startswith('iteration_')])
    chiplet_cols = sorted([col for col in all_cols if col.startswith('Chiplet_')])
    pkg_cols = sorted([col for col in all_cols if col.startswith('pkg_')])
    wl_cols = sorted([col for col in all_cols if col.startswith('WL_')]) # Will catch 'WL_mapping_'
    other_cols = sorted([col for col in all_cols if not (col.startswith('Chiplet_') or col.startswith('pkg_') or col.startswith('WL_'))])
    # Combine the lists into the final desired order.
    # We put 'other_cols' first to ensure 'iteration_id' is at the beginning.
    final_column_order = iter_cols + other_cols + chiplet_cols + pkg_cols + wl_cols
    print("\nFinal Column Order:")
    print(final_column_order)
    # Re-index the DataFrame with the new column order
    df = df[final_column_order]
    
    #Moved this to dump function
    #R print("\nDataFrame after reordering columns (first 5 rows):")
    #R print(df.head())
    #R output_filename = 'cfg/gen_arch/simulation_results_pandas.csv'
    #R print(f"\nWriting DataFrame to '{output_filename}'...")
    #R df.to_csv(output_filename, index=False)
    return df
    
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def find_config_diff(dict1, dict2, path=""):
    
    # Check for keys in the first dictionary
    for key in dict1:
        # Build the full path for the current key
        new_path = f"{path}.{key}" if path else key

        if key not in dict2:
            print(f"  - Key REMOVED at '{new_path}': {dict1[key]}")
            continue

        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            find_config_diff(dict1[key], dict2[key], new_path)
        elif dict1[key] != dict2[key]:
            print(f"  - Value CHANGED at '{new_path}':")
            print(f"      - OLD: {dict1[key]}")
            print(f"      - NEW: {dict2[key]}")

    for key in dict2:
        if key not in dict1:
            new_path = f"{path}.{key}" if path else key
            print(f"  + Key ADDED at '{new_path}': {dict2[key]}")
            

def find_connections(layouts, tol=1e-6):
    
    connections = {area: [] for area in layouts}
    chiplet_ids = list(layouts.keys())

    # Precompute bounding boxes
    boxes = {
        cid: {
            "x1": layouts[cid][0],
            "y1": layouts[cid][1],
            "x2": layouts[cid][0] + layouts[cid][2],
            "y2": layouts[cid][1] + layouts[cid][3],
        }
        for cid in chiplet_ids
    }

    # --- Horizontal adjacency (left-right neighbors) ---
    for cid in chiplet_ids:
        cbox = boxes[cid]
        candidates = []
        for oid in chiplet_ids:
            if oid == cid:
                continue
            obox = boxes[oid]

            # Must overlap in y
            y_overlap = (min(cbox["y2"], obox["y2"]) - max(cbox["y1"], obox["y1"])) > tol
            if not y_overlap:
                continue

            # Distance in x (must be positive → to the right)
            if obox["x1"] >= cbox["x2"] - tol:  # oid is to the right
                dist = obox["x1"] - cbox["x2"]
                candidates.append((dist, oid))
            elif cbox["x1"] >= obox["x2"] - tol:  # oid is to the left
                dist = cbox["x1"] - obox["x2"]
                candidates.append((dist, oid))

        # Pick closest neighbor(s)
        if candidates:
            min_dist = min(d for d, _ in candidates)
            for d, oid in candidates:
                if np.isclose(d, min_dist, atol=tol):
                    connections[cid].append(oid)
                    connections[oid].append(cid)

    # --- Vertical adjacency (top-bottom neighbors) ---
    for cid in chiplet_ids:
        cbox = boxes[cid]
        candidates = []
        for oid in chiplet_ids:
            if oid == cid:
                continue
            obox = boxes[oid]

            # Must overlap in x
            x_overlap = (min(cbox["x2"], obox["x2"]) - max(cbox["x1"], obox["x1"])) > tol
            if not x_overlap:
                continue

            # Distance in y (must be positive → above)
            if obox["y1"] >= cbox["y2"] - tol:  # oid is above
                dist = obox["y1"] - cbox["y2"]
                candidates.append((dist, oid))
            elif cbox["y1"] >= obox["y2"] - tol:  # oid is below
                dist = cbox["y1"] - obox["y2"]
                candidates.append((dist, oid))

        # Pick closest neighbor(s)
        if candidates:
            min_dist = min(d for d, _ in candidates)
            for d, oid in candidates:
                if np.isclose(d, min_dist, atol=tol):
                    connections[cid].append(oid)
                    connections[oid].append(cid)

    # Deduplicate connections
    connections = {k: list(set(v)) for k, v in connections.items()}
    return connections

def update_inter_pkg_connections(system_dict, connection_by_areas):
    
    hi_type = system_dict["pkg"]["HI_pkg_type"].lower()
    inter_pkg_conn = system_dict["pkg"].get("inter_pkg_conn", [])
    if isinstance(inter_pkg_conn, dict):
        inter_pkg_conn = [inter_pkg_conn]
    elif not isinstance(inter_pkg_conn, list):
        inter_pkg_conn = []

    # Early exit for pure 3D
    if hi_type == "3d":
        return system_dict  

    three_d_conn_type = None
    two_five_d_conn_type = None
    base_from_chiplets = set()

    if hi_type == "2.5d":
        # Pick any 2.5D type if it exists
        for conn in inter_pkg_conn:
            if isinstance(conn, dict) and "2.5d" in conn.get("loc", ""):
                two_five_d_conn_type = conn.get("connection_type")
                break
        if not two_five_d_conn_type:
            two_five_d_conn_type =  system_dict["pkg"].get("inter_pkg_conn", []) 

    elif hi_type == "2.5d_3d":
        for conn in inter_pkg_conn:
            if not isinstance(conn, dict):
                continue
            if conn.get("loc") == "stack0_base":
                base_from_chiplets.add(conn.get("from"))
                if conn.get("to") not in ("na", None):
                    base_from_chiplets.add(conn.get("to"))
                three_d_conn_type = conn.get("connection_type")
            elif "2.5d" in conn.get("loc", ""):
                two_five_d_conn_type = conn.get("connection_type")
        #if not two_five_d_conn_type:
        #    two_five_d_conn_type = "2.5d_rdl"

    # Case: 2.5d
    if hi_type == "2.5d":
        new_conns = []
        for chiplet_from, chiplet_tos in connection_by_areas.items():
            for chiplet_to in chiplet_tos:
                new_conns.append({
                    "from": chiplet_from,
                    "to": chiplet_to,  # group all tos here
                    "connection_type": two_five_d_conn_type,
                    "loc": "2.5d_chiplet"
                })
        system_dict["pkg"]["inter_pkg_conn"] = new_conns
        return system_dict

    # Case: 2.5d_3d
    if hi_type == "2.5d_3d":
        preserved = [
            c for c in inter_pkg_conn
            if isinstance(c, dict) and c.get("loc") != "2.5d_chiplet"
        ]

        new_conns = []
        for chiplet_from, chiplet_tos in connection_by_areas.items():
            for chiplet_to in chiplet_tos:
                if (chiplet_from in base_from_chiplets) and (chiplet_to in base_from_chiplets):
                    new_conns.append({
                        "from": chiplet_from,
                        "to": chiplet_to,
                        "connection_type": three_d_conn_type,
                        "loc": "stack0_base"
                    })
                else:
                    new_conns.append({
                        "from": chiplet_from,
                        "to": chiplet_to,
                        "connection_type": two_five_d_conn_type,
                        "loc": "2.5d_chiplet"
                    })

        system_dict["pkg"]["inter_pkg_conn"] = preserved + new_conns
        return system_dict

    return system_dict

def update_dict_inter_pkg_topology(system_dict):
    
    if print_info:
        print(f"[INFO] Original inter_pkg_conn based on mutation:")
        print(json.dumps(system_dict["pkg"]["inter_pkg_conn"], indent=2))
        print(f"[INFO] Start of final iteration $$$$$$$$$$$$$")
    
    #Update connections for 2.5d grid topology 
    _, system_info = calc_HI_dimension(system_dict)
    system_connections = find_connections(system_info)
    system_dict = update_inter_pkg_connections(system_dict, system_connections)
    
    if print_info:
        print(f"[INFO] Updated inter_pkg_conn based on topology:")
        print(json.dumps(system_dict["pkg"]["inter_pkg_conn"], indent=2))
        print(f"[DBG] system_info is :{system_info}")
        print(f"[DBG] system_connections is :{system_connections}")
        print(f"[DBG] End of final iteration $$$$$$$$$$$$$")
    
    return system_dict

def opC_from_pj(pj_value, per_second=True, years=3):
    
    print(f"[INFO] OPE_CARBON_START") if print_info else None
    print(f"[INFO] Input energy: {pj_value} pJ") if print_info else None
    
    # Conversion constants
    PJ_TO_J = 1e-12          # 1 pJ = 1e-12 J
    J_TO_KWH = 1 / 3.6e6     # 1 J = 1/3.6e6 kWh
    SECONDS_PER_YEAR = 365 * 24 * 3600

    print(f"[INFO] 1 pJ = {PJ_TO_J} J") if print_info else None
    print(f"[INFO] 1 J = {J_TO_KWH} kWh") if print_info else None

    # Convert to joules
    energy_joules = pj_value * PJ_TO_J
    print(f"[INFO] Energy in joules: {energy_joules} J") if print_info else None

    # If it’s a rate (pJ per second), multiply by total seconds
    if per_second:
        total_seconds = years * SECONDS_PER_YEAR
        print(f"[INFO] Total seconds for {years} years: {total_seconds}") if print_info else None
        energy_joules *= total_seconds
        print(f"[INFO] Total energy over {years} years: {energy_joules} J") if print_info else None

    # Convert joules to kWh
    energy_kwh = energy_joules * J_TO_KWH
    print(f"[INFO] Converted energy: {energy_kwh} kWh") if print_info else None
    
    Carbon_per_kWh = 0.700  # kg CO2 per kWh
    opeC_kgs = energy_kwh * Carbon_per_kWh * 1e6 #Kgs for 1M volume
    print(f"[INFO] Operational Carbon Footprint: {opeC_kgs} kgs CO2") if print_info else None

    return energy_kwh, opeC_kgs

def total_opC(energy_sram, energy_compute, energy_comm, lifetime_years=3):
    _, opC_sram = opC_from_pj(energy_sram, per_second=True, years=lifetime_years)
    _, opC_compute = opC_from_pj(energy_compute, per_second=True, years=lifetime_years)
    _, opC_comm = opC_from_pj(energy_comm, per_second=True, years=lifetime_years)
    print(f"[INFO] *********** ") if print_info else None
    print(f"[INFO] SRAM Operational Carbon Footprint: {opC_sram} kgs CO2") if print_info else None
    print(f"[INFO] COMPUTE Operational Carbon Footprint: {opC_compute} kgs CO2") if print_info else None
    print(f"[INFO] COMM'N Operational Carbon Footprint: {opC_comm} kgs CO2") if print_info else None
    total_opC_value = opC_sram + opC_compute + opC_comm #kgs
    print(f"[INFO] Total Operational Carbon Footprint: {total_opC_value} kgs CO2") if print_info else None
    print(f"[INFO] OPE_CARBON_END\n") if print_info else None
    return total_opC_value

Segment = Tuple[str, pd.DataFrame, Optional[str]]  # (mode, df, pkg_suffix)
def build_design_tables(spec: Dict) -> List[Segment]:
    print("[DEBUG] Starting build_design_tables") if print_info else None

    chiplet_keys = [k for k in spec.keys() if k.lower().startswith("chiplet_")]
    print(f"[DEBUG] Found chiplets: {chiplet_keys}") if print_info else None

    def _mk_row(chip: str) -> Dict:
        c = spec[chip]
        return {
            "type": "logic",
            "area": float(c.get("area", 0.0)),
            "power": float(c.get("power", 0.0)),
            "node": int(c.get("tech_node")) if c.get("tech_node") is not None else None,
        }

    base_df = pd.DataFrame({k: _mk_row(k) for k in chiplet_keys}).T
    base_df = base_df[["type", "area", "power", "node"]]
    print("[DEBUG] Base DataFrame constructed:") if print_info else None
    print(base_df) if print_info else None

    pkg = spec.get("pkg", {})
    hi_pkg_type = str(pkg.get("HI_pkg_type", "")).lower()
    interconns_raw = pkg.get("inter_pkg_conn", []) or []                        
    # Normalize to a list of dicts so loops below are safe                      
    if isinstance(interconns_raw, dict):                                        
        interconns = list(interconns_raw.values())                              
    elif isinstance(interconns_raw, list):                                      
        interconns = [e for e in interconns_raw if isinstance(e, dict)]         
    else:  # covers strings like "2d_na"                                        
        interconns = []  
    print(f"[INFO] HI_pkg_type: {hi_pkg_type}") if print_info else None
    print(f"[INFO] inter_pkg_conn entries: {len(interconns)}") if print_info else None

    suffix = {"2.5d": None, "3d": None}
    def _suffix_from(conn_type: str, head: str) -> Optional[str]:
        try:
            return conn_type.split(f"{head}_", 1)[1]
        except Exception:
            return None
    for c in interconns:
        ctype = str(c.get("connection_type", "")).lower()
        if ctype.startswith("2.5d_"):
            sfx = _suffix_from(ctype, "2.5d")
            if sfx: suffix["2.5d"] = sfx
        elif ctype.startswith("3d_"):
            sfx = _suffix_from(ctype, "3d")
            if sfx: suffix["3d"] = sfx
    print(f"[INFO] Suffix map (last-seen per mode): {suffix}") if print_info else None

    segments: List[Segment] = []

    if hi_pkg_type in {"2d", "2.5d", "3d"}:
        if hi_pkg_type == "2d":
            segments.append(("2d", base_df, "rdl"))
        elif hi_pkg_type == "2.5d":
            segments.append(("2.5d", base_df, suffix["2.5d"]))
        else:
            segments.append(("3d", base_df, suffix["3d"]))
        print(f"[INFO] Single-mode -> segments len={len(segments)}") if print_info else None
        return segments

    if hi_pkg_type == "2.5d_3d":
        members = {"2.5d": set(), "3d": set()}
        chiplet_set = set(chiplet_keys)

        for i, c in enumerate(interconns):
            ctype = str(c.get("connection_type", "")).lower()
            frm = c.get("from")
            loc = str(c.get("loc", "")).lower()
            print(f"[INFO] Conn[{i}]: type={ctype}, from={frm}, loc={loc}") if print_info else None

            # Only consider 'from' if it's a known chiplet
            if frm not in chiplet_set:
                continue

            if ctype.startswith("2.5d_"):
                # Rule: any 2.5d_* puts FROM chiplet into 2.5d group
                members["2.5d"].add(frm)

            elif ctype.startswith("3d_"):
                # Rule: 3d_* with stack0_base → 2.5d group; otherwise → 3d group
                if loc == "stack0_base":
                    members["2.5d"].add(frm)
                else:
                    members["3d"].add(frm)

        print(f"[INFO] 2.5d members (new rule): {sorted(members['2.5d'])}") if print_info else None
        print(f"[INFO] 3d members   (new rule): {sorted(members['3d'])}") if print_info else None

        # Build DFs in fixed order: 2.5d then 3d
        df_25d = base_df.loc[sorted(members["2.5d"])] if members["2.5d"] else base_df.iloc[0:0]
        df_3d  = base_df.loc[sorted(members["3d"])]  if members["3d"]  else base_df.iloc[0:0]
        segments.append(("2.5d", df_25d, suffix["2.5d"]))
        segments.append(("3d",   df_3d,  suffix["3d"]))
        print(f"[INFO] Hybrid -> segments len={len(segments)} (2.5d, 3d)") if print_info else None
        return segments

    # -------- Fallback (unchanged) --------
    print("[WARN] Unknown HI_pkg_type; emitting single '2d' segment as default") if print_info else None
    segments.append(("2d", base_df, None))
    return segments
