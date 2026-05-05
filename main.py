from ast import arg
import json 
import os
import random
import pandas as pd
import time
import csv
import itertools
import copy
import sys
import math
import statistics
import types

from chiplet.n_utils import get_area_power, read_json_input_params, calculate_system_metrics, write_solution_to_json, \
                            calculate_system_normalized_metrics,\
                            mutate_wl_mapping, mutate_arch_tech_node,\
                            mutate_arch_chiplet_size, mutate_arch_mem_pkg,\
                            mutate_arch_inter_pkg, mutate_arch_chiplet_num,\
                            mutate_arch_chiplet_size_cg_mode, \
                            mutate_arch_sram_buf, mutate_arch_protocol,\
                            accept_move_func, dump_results, calculate_memory_bandwidth,\
                            find_run_time, process_iteration_wide, process_arch_details_dump, \
                            flatten_dict, find_config_diff, get_sram_area_energy, recursive_split_updated, \
                            calc_HI_dimension, find_connections, update_inter_pkg_connections, opC_from_pj, total_opC,\
                            build_design_tables
from chiplet.n_disagg import ChipletGenerator, PackageGenerator, WLMappingGenerator
from chiplet.carbon_model.ECO_chip import find_carbon

from system.utils.GEMMWorkload import GEMMWorkload
from system.utils.Scheduler import CHIP2CHIP_TRANSFER, Scheduler
from system.utils.ChipletSystem import ChipletSystem
from system.utils.SimulationCache import SimulationCache
from config import print_info, fast_test, latency_en, \
                    sram_selection_mode


#########
#Freq config read
with open('cfg/parameters/freq_scale.json', 'r') as f:
    freq_config = json.load(f)

FREQUENCY = freq_config['BASE_FREQUENCY']
scaling_factors = {int(k): v for k, v in freq_config['freq_scaling_factors'].items()}
#########



######
## Workload 
with open("cfg/examples/workload.json") as f:
    workload = json.load(f)
GEMM_SHAPE = {int(k): v for k, v in workload.items()}
######



def build_scheduler_system(workload, arch_dict):

    system = ChipletSystem(arch_dict=arch_dict, BASE_FREQUENCY=FREQUENCY)
    scheduler = Scheduler(workload, system.core_dict.values(), mapping_dict=arch_dict['WL_mapping']['mapping'])
    scheduler.static_workload_scheduling()

    return scheduler, system



def simulate_latency_energy(cache: SimulationCache, arch_dict: dict, dbg = False):


    wl = GEMMWorkload(GEMM_M, GEMM_K, GEMM_N)


    scheduler, system = build_scheduler_system(wl, arch_dict)
    cache.all_cores_simulation_with_cache(scheduler.systolic_arrays)
    latency_ns, energy_pj = scheduler.system_modeling(system)

    sram_energy_pj = scheduler._get_sram_energy() #Use fucntion get_sram_area_energy for this 
    
    return latency_ns, energy_pj, sram_energy_pj

class SystemGenerator:
    
    def __init__(self, config_path, stack_diff_size=False):
        (self.max_chiplet, self.sys_array, self.tech_nodes, self.sram_buf_sizes, 
         self.inter_pkg_arch, self.mem_pkg_arch, self.protocol_arch) = read_json_input_params(config_path)
        self.chiplet_gen = ChipletGenerator(self.max_chiplet, self.sys_array, self.tech_nodes, self.sram_buf_sizes, get_area_power, get_sram_area_energy)
        self.wl_mapping_gen = WLMappingGenerator()
        self.stack_diff_size = stack_diff_size

    def generate_system(self, max_retries=20):
        for attempt in range(max_retries):
            try:
                # 1. Generate the chiplets. This is the base for everything.
                chiplets_dict = self.chiplet_gen.generate()
                
                # 2. Instantiate PackageGenerator and attempt to create the package.
                #    This is the step that may fail if stacking rules are not met.
                package_gen = PackageGenerator(chiplets_dict, self.inter_pkg_arch, self.mem_pkg_arch, self.protocol_arch ,stack_diff_size=self.stack_diff_size)
                package_dict = package_gen.generate()
                # 3. Generate the workload mapping.
                mapping_details = self.wl_mapping_gen.generate()
                # 4. Assemble the final dictionary.
                final_system = {}
                final_system.update(chiplets_dict)
                final_system["pkg"] = package_dict
                final_system["WL_mapping"] = {"mapping": mapping_details}
                
                #Update connections for 2.5d grid topology 
                final_system_size, final_system_info = calc_HI_dimension(final_system)
                final_connections = find_connections(final_system_info)
                final_system = update_inter_pkg_connections(final_system, final_connections)
                
                return final_system
            except ValueError as e:
                print(f"[WARNING] Attempt {attempt + 1}/{max_retries} failed to generate valid package: {e}. Retrying...")
        
        raise RuntimeError(f"[ERROR] Failed to generate a valid system after {max_retries} attempts.")

def calculate_cost(profile_name='t1',cost_avgerage=dict, system_dict=dict, cache=SimulationCache):
    print("\n[INFO] --- System Analysis Metrics ---") if print_info else None
    power, area, dollar_cost = calculate_system_metrics(
        system_dict=system_dict
    )
    
    
    print(f"[DEBUG COST] ************** LATENCY ************** ") if print_info else None
    if latency_en:
        print(f"[INFO] Working on calcuting performance ...") if print_info else None
        latency, energy_comm, energy_sram = simulate_latency_energy(cache,system_dict)#, cost_no_latency_last_iteration, latency_coeff, average_latency)
    else: 
        latency = 0
        energy_comm = 0
    
    total_operational_C_kg = total_opC(energy_sram=energy_sram, energy_comm=energy_comm, energy_compute=power*latency*1000, lifetime_years=3)
    print(f"[CARBON DEBUG] Operational Carbon for 3 years (kgs) = {total_operational_C_kg} ") if print_info else None
    
    print(f"EMB_CARBON_START\n") if print_info else None
    segments = build_design_tables(system_dict)
    if len(segments) == 1:
        mode, df, pkg = segments[0]
        # process single segment
        print(f"[EMB CARBON] mode is {mode} pkg is {pkg}")  if print_info else None 
        print(f"[EMB CARBON] df is \n{df}")  if print_info else None
        args = types.SimpleNamespace(
            design=df,  # pandas DataFrame
            pkg_type=pkg,      # or whatever applies
            lifetime= 3*365*24         # hrs 
            )
        total_embC_kg, unused_opeC_kg, unused_totalC_kg = find_carbon(args)
        print(f"[EMB CARBON] Embedded Carbon (kgs) = {total_embC_kg} ")   if print_info else None
        print(f"EMB_CARBON_END\n") if print_info else None
    else:
        total_embC_kg = 0
        for mode, df, pkg in segments:
            # process each of '2.5d' and '3d' in fixed order
            print(f"[EMB CARBON] mode is {mode} pkg is {pkg}")   if print_info else None
            print(f"[EMB CARBON] df is \n{df}")  if print_info else None
            args = types.SimpleNamespace(
                design=df,  # pandas DataFrame
                pkg_type=pkg,      # or whatever applies
                lifetime= 3*365*24         # hrs 
            )
            embC_kg, unused_opeC_kg, unused_totalC_kg = find_carbon(args)
            print(f"[EMB CARBON] Embedded Carbon (kgs) = {embC_kg} ") if print_info else None
            print(f"EMB_CARBON_END\n") if print_info else None
            total_embC_kg += embC_kg
            
        
     
    print("\n[INFO] --- System Normalized Metrics calculation ---") if print_info else None
    cost_val, norm_cost_dict, raw_cost_dict = calculate_system_normalized_metrics(
        area=area,
        power=power, 
        energy=energy_comm, 
        energy_sram=energy_sram,
        dollar=dollar_cost,
        latency=latency,
        embCarbon=total_embC_kg,
        opeCarbon=total_operational_C_kg,
        profile_name=profile_name,
        cost_averages=cost_avgerage,
        arch_dict=system_dict
    )
    return cost_val, norm_cost_dict, raw_cost_dict


def gen_initial_arch(config_path,stack_diff_size=True):
    
    system_builder = SystemGenerator(config_path=config_path,stack_diff_size=stack_diff_size)
    final_config = system_builder.generate_system()
    return final_config    

def get_calib_cost_avg(calibration_iterations, config_path, cache, calibration_file_path):
    
    # Check if the calibration file already exists.
    if os.path.exists(calibration_file_path) and calibration_iterations > 0:
        print(f"\n[INFO] --- Loading existing cost averages from {calibration_file_path} ---") if print_info else None
        with open(calibration_file_path, 'r') as f:
            cost_averages = json.load(f)
        print(f"[INFO] --- Successfully loaded averages: {cost_averages} ---") if print_info else None
        return cost_averages

    print(f"\n[INFO] --- Running new calibration for {calibration_iterations} iterations in {calibration_file_path} ---") if print_info else None
    
    # Lists to store metrics from each run
    powers, areas, costs, latency_vals = [], [], [], []
    energys = [] 
    embCarbons = []
    opeCarbons = []
    
    calib_all_rows_data =[] #Stores data of all rows 
    calib_cost_stats = {
    'energy': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'cost': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'latency': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'area': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'embCarbon': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'opeCarbon': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0}
    }
    calib_cost_values = {  # <-- NEW
    'energy': [],
    'cost': [],
    'latency': [],
    'area': [],
    'embCarbon': [],
    'opeCarbon': []
}

    for i in range(calibration_iterations):
        # Generate a new system architecture
        final_config = gen_initial_arch(config_path=config_path)
        print(f"[INFO] Initial architecture json written at cfg/gen_arch directory") if print_info else None
        
        power, area, cost = calculate_system_metrics(system_dict=final_config)
        latency, energy_comm, energy_sram = simulate_latency_energy(cache=cache, arch_dict=final_config)
        print(f"[INFO] Calibration stage, power, area, cost done. Now computing latency ....") if print_info else None 
        ope_carbon_kg = total_opC(energy_sram=energy_sram, energy_comm=energy_comm, energy_compute=power*latency*1000, lifetime_years=3)
        seg = build_design_tables(final_config)
        if len(seg) ==1:
            mode, df, pkg = seg[0]
            args = types.SimpleNamespace(
                design=df,  # pandas DataFrame
                pkg_type=pkg,      # or whatever applies
                lifetime= 3*365*24         # hrs 
            )
            emb_carbon_kg, unused_opeC_kg, unused_totalC_kg = find_carbon(args)
        else:
            emb_carbon_kg = 0
            for mode, df, pkg in seg:
                args = types.SimpleNamespace(
                    design=df,  # pandas DataFrame
                    pkg_type=pkg,      # or whatever applies
                    lifetime= 3*365*24         # hrs 
                )
                embC_kg_part, unused_opeC_kg, unused_totalC_kg = find_carbon(args)
                emb_carbon_kg += embC_kg_part
        
        areas.append(area)
        costs.append(cost)
        latency_vals.append(latency)
        embCarbons.append(emb_carbon_kg)
        opeCarbons.append(ope_carbon_kg)
        #####
        powers.append(power) # Keeping this for now, but we will use energy in the future
        energy_compute = power*latency 
        energy = energy_comm + energy_compute + energy_sram
        energys.append(energy)
        #####

        print(f"*****************************************************************************") 
        print(f"\n Calibraiton iteration is {i+1}/{calibration_iterations} ....")
        print(f"\n  Iteration {i+1}/{calibration_iterations}: Power={power}W, Area={area}mm^2, Cost=${cost}, Latency={latency}, Energy={energy} pJ, EmbCarbon={emb_carbon_kg} kgs, OpeCarbon={ope_carbon_kg} kgs")
        print(f"*****************************************************************************") if print_info else None

        
        ##########
        #Dict of current iteration cost metrics
        cost_dict = {
            "energy": energy, 
            "area": area, 
            "latency": latency,
            "dollar_cost": cost,
            "embCarbon": emb_carbon_kg,
            "opeCarbon": ope_carbon_kg
        }
        
        #Add to current arch dict 
        final_config['cost_metrics'] = cost_dict
        
        processed_dict_rows = process_iteration_wide(final_config, iteration_id=i)
        calib_all_rows_data.append(processed_dict_rows)
        
        calib_data_csv_results = process_arch_details_dump(all_rows_data=calib_all_rows_data)
        
        ##########
        
        for key, val in zip(['energy', 'cost', 'latency', 'area', 'embCarbon', 'opeCarbon'], [energy, cost, latency, area, emb_carbon_kg, ope_carbon_kg]):
            calib_cost_values[key].append(val)  # <-- NEW
            
            # Update min/max
            calib_cost_stats[key]['max'] = max(calib_cost_stats[key]['max'], val)
            calib_cost_stats[key]['min'] = min(calib_cost_stats[key]['min'], val)

            # Welford's online algorithm for mean and variance
            calib_cost_stats[key]['count'] += 1
            delta = val - calib_cost_stats[key]['mean']
            calib_cost_stats[key]['mean'] += delta / calib_cost_stats[key]['count']
            delta2 = val - calib_cost_stats[key]['mean']
            calib_cost_stats[key]['M2'] += delta * delta2
         
    for key in calib_cost_stats:
        count = calib_cost_stats[key]['count']
        if count > 1:
            variance = calib_cost_stats[key]['M2'] / (count - 1)
            calib_cost_stats[key]['stddev'] = math.sqrt(variance)
        else:
            calib_cost_stats[key]['stddev'] = 0.0 
        
        calib_cost_stats[key]['median'] = statistics.median(calib_cost_values[key]) if calib_cost_values[key] else 0.0

    # Calculate the average of each metric
    avg_power = round(sum(powers) / len(powers), 2) if powers else 0 #TODO 1: Remove Power 
    avg_energy = round(sum(energys) / len(energys), 2) if energys else 0
    avg_area = round(sum(areas) / len(areas), 2) if areas else 0
    avg_cost = round(sum(costs) / len(costs), 2) if costs else 0
    avg_latency = round(sum(latency_vals) / len(latency_vals),2) if latency_vals else 0
    avg_embCarbon = round(sum(embCarbons) / len(embCarbons),2) if embCarbons else 0
    avg_opeCarbon = round(sum(opeCarbons) / len(opeCarbons),2) if opeCarbons else 0
    
    energy_min = calib_cost_stats['energy']['min']
    energy_max = calib_cost_stats['energy']['max']
    energy_stddev = calib_cost_stats['energy']['stddev']
    energy_mean = calib_cost_stats['energy']['mean']
    energy_median = calib_cost_stats['energy']['median']
    latency_min = calib_cost_stats['latency']['min']
    latency_max = calib_cost_stats['latency']['max']
    latency_stddev = calib_cost_stats['latency']['stddev']
    latency_mean = calib_cost_stats['latency']['mean']
    latency_median = calib_cost_stats['latency']['median']
    area_min = calib_cost_stats['area']['min']
    area_max = calib_cost_stats['area']['max']
    area_stddev = calib_cost_stats['area']['stddev']
    area_mean = calib_cost_stats['area']['mean']
    area_median = calib_cost_stats['area']['median']
    cost_min = calib_cost_stats['cost']['min']
    cost_max = calib_cost_stats['cost']['max']
    cost_stddev = calib_cost_stats['cost']['stddev']
    cost_mean = calib_cost_stats['cost']['mean']
    cost_median = calib_cost_stats['cost']['median']
    embcarbon_min = calib_cost_stats['embCarbon']['min']
    embcarbon_max = calib_cost_stats['embCarbon']['max']
    embcarbon_stddev = calib_cost_stats['embCarbon']['stddev']
    embcarbon_mean = calib_cost_stats['embCarbon']['mean']
    embcarbon_median = calib_cost_stats['embCarbon']['median']
    opecarbon_min = calib_cost_stats['opeCarbon']['min']
    opecarbon_max = calib_cost_stats['opeCarbon']['max']
    opecarbon_stddev = calib_cost_stats['opeCarbon']['stddev']
    opecarbon_mean = calib_cost_stats['opeCarbon']['mean']
    opecarbon_median = calib_cost_stats['opeCarbon']['median']
    
    cost_averages = {
        "avg_energy": avg_energy,  
        "avg_area": avg_area,
        "avg_dollar_cost": avg_cost,
        "avg_latency": avg_latency,
        "avg_embCarbon": avg_embCarbon,
        "avg_opeCarbon": avg_opeCarbon,
        "energy_min": energy_min,
        "energy_max": energy_max,
        "energy_stddev": energy_stddev,
        "energy_mean": energy_mean,
        "energy_median": energy_median,
        "latency_min": latency_min,
        "latency_max": latency_max,
        "latency_stddev": latency_stddev,
        "latency_mean": latency_mean,
        "latency_median": latency_median,
        "area_min": area_min,
        "area_max": area_max,
        "area_stddev": area_stddev,
        "area_mean": area_mean,
        "area_median": area_median,
        "cost_min": cost_min,
        "cost_max": cost_max,
        "cost_stddev": cost_stddev,
        "cost_mean": cost_mean,
        "cost_median": cost_median,
        "embCarbon_min": embcarbon_min,
        "embCarbon_max": embcarbon_max,
        "embCarbon_stddev": embcarbon_stddev,
        "embCarbon_mean": embcarbon_mean,
        "embCarbon_median": embcarbon_median,
        "opeCarbon_min": opecarbon_min,
        "opeCarbon_max": opecarbon_max,
        "opeCarbon_stddev": opecarbon_stddev,
        "opeCarbon_mean": opecarbon_mean,
        "opeCarbon_median": opecarbon_median
    }

    #Dump sim_annealing arch info
    print(f"[CALIBRATION] Dumping Calibration Results csv ...... ") #if print_info else None
    print(calib_data_csv_results.head())
    csv_file_path = os.path.splitext(calibration_file_path)[0] + ".csv"
    calib_data_csv_results.to_csv(f"{csv_file_path}", index=False)
    print("[INFO] Done dumping Simulation Results csv")

    # Save the new averages to the file for future use
    with open(calibration_file_path, 'w') as f:
        json.dump(cost_averages, f, indent=4)
        
    print(f"[INFO] --- Saved new cost averages to {calibration_file_path}: {cost_averages} ---") if print_info else None
    
    return cost_averages

def run_calibration(wl_idx, cache_file, run_name, cost_profile, calibration_iterations=10000):
    
    print(f"[STANDALONE_MODE] Standalone framework mode is enabled")
    input_file_path = "cfg/parameters/input.json"
    calibration_file_path = f"cfg/calibration/calibration_{wl_idx}.json"
    print(f"[STANDALONE_MODE] Input file path is {input_file_path}")
    print(f"[STANDALONE_MODE] Calibration file path is {calibration_file_path}")

    

    max_chiplet, sys_array, tech_nodes, sram_buf_sizes, inter_pkg_arch, mem_pkg_arch, protocol_arch = read_json_input_params(input_file_path)
    # Pack all params needed for regeneration into a dict
    gen_params = {
            'max_chiplet': max_chiplet, 'sys_array': sys_array, 'tech_nodes': tech_nodes,
            'sram_buf_sizes': sram_buf_sizes,'inter_pkg_arch': inter_pkg_arch, 'mem_pkg_arch': mem_pkg_arch,
            'protocol_arch': protocol_arch, 'stack_diff_size': True
        }
    
    if print_info:
        print(f"[INFO] Max chiplet is {max_chiplet}")
        print(f"[INFO] Sys array is {sys_array}")
        print(f"[INFO] Tech node is {tech_nodes}")
        print(f"[INFO] SRAM buf sizes is {sram_buf_sizes}")
        print(f"[INFO] Inter pkg arch is {inter_pkg_arch}")
        print(f"[INFO] Mem pkg arch is {mem_pkg_arch}")
    
    

    cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)
    
    ########## CALIBRATION ############
    cost_avg = get_calib_cost_avg(
        calibration_iterations=calibration_iterations,
        config_path=input_file_path,
        cache=cache,
        calibration_file_path=calibration_file_path 
    )
    ###################################
    print(f"[CALIBRATION] Calibration is completed")
##########################################



##########################################
####### Simulated Annealing Function

def sim_annealing(wl_idx, cache_file, run_name, cost_profile, initial_temp=4000, freezing_temp=1e-3, max_move_per_temp_step=20,
                  cooling_rate=0.99, calibration_iterations=200):
    
    print(f"[STANDALONE_MODE] Standalone framework mode is enabled")
    input_file_path = "cfg/parameters/input.json"
    calibration_file_path = f"cfg/calibration/calibration_{wl_idx}.json"
    print(f"[STANDALONE_MODE] Input file path is {input_file_path}")
    print(f"[STANDALONE_MODE] Calibration file path is {calibration_file_path}")

    

    max_chiplet, sys_array, tech_nodes, sram_buf_sizes, inter_pkg_arch, mem_pkg_arch, protocol_arch = read_json_input_params(input_file_path)
    # Pack all params needed for regeneration into a dict
    gen_params = {
            'max_chiplet': max_chiplet, 'sys_array': sys_array, 'tech_nodes': tech_nodes,
            'sram_buf_sizes': sram_buf_sizes,'inter_pkg_arch': inter_pkg_arch, 'mem_pkg_arch': mem_pkg_arch,
            'protocol_arch': protocol_arch, 'stack_diff_size': True
        }
    
    if print_info:
        print(f"[INFO] Max chiplet is {max_chiplet}")
        print(f"[INFO] Sys array is {sys_array}")
        print(f"[INFO] Tech node is {tech_nodes}")
        print(f"[INFO] SRAM buf sizes is {sram_buf_sizes}")
        print(f"[INFO] Inter pkg arch is {inter_pkg_arch}")
        print(f"[INFO] Mem pkg arch is {mem_pkg_arch}")
    
    

    cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)
    
    ########## CALIBRATION ############
    cost_avg = get_calib_cost_avg(
        calibration_iterations=calibration_iterations,
        config_path=input_file_path,
        cache=cache,
        calibration_file_path=calibration_file_path 
    )
    ###################################
    
    
    # Use the new top-level function to generate the architecture
    cur_architecture = gen_initial_arch(config_path=input_file_path,stack_diff_size=True)
    
    if cur_architecture:
        #write_solution_to_json(cur_architecture,'cfg/gen_arch/initial-arch.json')
        print(f"[INFO] Initial architecture json written at cfg/gen_arch directory") if print_info else None
    else:
        print(f"[ERROR] Could not generate initial solution")
        
    
    print("\n[INFO] --- Working on calculating cost ---") if print_info else None
    
    
    cost_val, norm_cost_dict, raw_cost_dict = calculate_cost(
        profile_name=cost_profile,
        cost_avgerage=cost_avg,
        system_dict=cur_architecture,
        cache=cache
        )
    
    #Debug
    if print_info:
        print("[INFO] Cost averages is ",json.dumps(cost_avg))
        print(f"[INFO] Calculated cost value is {cost_val}")
        print(f"[INFO] Normalized cost dict is {norm_cost_dict}")
        print(f"[INFO] Raw cost dict is {raw_cost_dict}")
    
    #Assign initial cost_val as the best cost at the start of sim_annelaing 
    best_cost = cost_val
    
    temperature = initial_temp
    
    #Initialize
    SA_log_data = []
    SA_run_loop = 0
    all_rows_data = []
    
    while (temperature > freezing_temp):
        for iterations in range(max_move_per_temp_step): 
            print(f" -------------------------------------- ") #if print_info else None
            print(f"\n[DBG] Current temp is ***** {temperature} ****** and move iteration is ***** {iterations} ******") #if print_info else None

            #Generate random move to either perform WL changes or Architecture changes 
            SA_first_level_move = random.randint(0,1) # 0 for WL move, and 1 for Architecture move
            random_arch_move = None #Assigning to None initially, so we can dump csv for WL Moves, otherwise the code cribs
            
                
            if SA_first_level_move==0: #WL Move
                print(f"[INFO] 1st Level Move type is WL move") if print_info else None
                print(f"[INFO] Working on WL Mapping mutation") if print_info else None
                new_architecture = mutate_wl_mapping(cur_architecture)

            else: #Architecture Move
                print(f"[INFO] 1st Level Move type is Architecure Move") if print_info else None
                random_arch_move = random.randint(0,6) #Randomly make an Arch move

                # ** Chiplet Num **
                if random_arch_move==0:
                    print("[INFO] 2nd Level Move is - Chiplet Number") if print_info else None
                    print(f"[INFO] Working on *** Arch Chiplet Number *** mutation") if print_info else None
                    new_architecture = mutate_arch_chiplet_num(cur_architecture,gen_params)
                # ** Chiplet Size **
                if random_arch_move==1:
                    print("[INFO] 2nd Level Move is - Chiplet Size") if print_info else None
                    print("[INFO] Working on *** Arch Chiplet Size *** mutation") if print_info else None
                    new_architecture = mutate_arch_chiplet_size(cur_architecture,all_sys_array_sizes=sys_array,params=gen_params)
                # ** Chiplet Tech Node **
                if random_arch_move==2:
                    print("[INFO] 2nd Level Move is - Chiplet Tech Node") if print_info else None
                    print(f"[INFO] Working on *** Arch Chiplet Tech Node *** mutation") if print_info else None
                    new_architecture = mutate_arch_tech_node(cur_architecture,params=gen_params)
                # ** Mem pkg type **
                if random_arch_move==3:
                    print("[INFO] 2nd Level Move is - Mem Type") if print_info else None
                    print(f"[INFO] Working on *** Arch Mem Pkg *** mutation") if print_info else None
                    new_architecture = mutate_arch_mem_pkg(cur_architecture,all_mem_pkg_options=mem_pkg_arch)
                # ** Inter pkg type **
                if random_arch_move==4:
                    print("[INFO] 2nd Level Move is - Pkg Type ") if print_info else None
                    print(f"[INFO] WOkring on *** Arch Inter Pkg *** mutation") if print_info else None
                    new_architecture = mutate_arch_inter_pkg(cur_architecture,all_inter_pkg_options=inter_pkg_arch, all_protocol_options=protocol_arch)
                # ** SRAM Buffer Size **
                if random_arch_move==5:
                    print("[INFO] 2nd Level Move is - SRAM Buf Size") if print_info else None
                    print(f"[INFO] Working on *** Arch SRAM Buf Size *** mutation") if print_info else None
                    new_architecture = mutate_arch_sram_buf(cur_architecture, sram_buf_sizes=sram_buf_sizes)
                # ** Protocol Arch **
                if random_arch_move==6:
                    print("[INFO] 2nd Level Move is - Protocol Arch") if print_info else None
                    print(f"[INFO] Working on *** Arch Protocol Arch *** mutation") if print_info else None
                    new_architecture = mutate_arch_protocol(cur_architecture, all_protocol_options=protocol_arch)

            
            
            #Write the new arch json to a file 
            if new_architecture is None:
                print(f"[ERROR] Could not generate mutated new_arch solution") if print_info else None
                print(f"[WARNING] Skipping current iteration and moving to next since architecture is NONE") if print_info else None
                
                print(f"[INFO] Writing log entry for arch=NONE case ...") if print_info else None
                SA_run_loop += 1 #Increment loop variable as an iteration attemt was made
                
                ####### Dump details #######
                processed_row = process_iteration_wide(new_architecture, iteration_id=SA_run_loop)
                all_rows_data.append(processed_row)
                ############################
                
                log_entry = {
                'temperature': temperature,
                'inner_iter': iterations,
                '1L_move': SA_first_level_move,
                '2L_move': random_arch_move,
                'SA_run_loop': SA_run_loop,
                'best_cost': best_cost,
                'new_cost': None,
                'cost_diff': None,
                'move_accepted': None,
                'move_type': None
                }
                log_entry.update({k: None for k in norm_cost_dict}) #Assign all values to be None from norm_cost_dict
                log_entry.update({k: None for k in raw_cost_dict}) #Assign all values to be None from raw_cost_dict
                SA_log_data.append(log_entry)
                continue
            else:
            
                print("\n[INFO] --- Calculating new cost ---") if print_info else None
                new_cost_val, norm_cost_dict, raw_cost_dict = calculate_cost(
                                                    profile_name=cost_profile,
                                                    cost_avgerage=cost_avg,
                                                    system_dict=new_architecture,
                                                    cache=cache
                )
                print(f"\n[INFO] The new cost is {new_cost_val} and current cost is {best_cost}") if print_info else None
            
                #Calcualte the cost delta
                cost_diff = new_cost_val - best_cost
            
                #Move accepet check 
                move_accepted, move_type = accept_move_func(cost_diff=cost_diff, temp=temperature)
                if move_accepted:
                    print(f"[INFO] ## Move is accepted ##") if print_info else None
                    if move_type==1:
                        print(f"[INFO] ## Move is accepted due to Improvement (cost_diff<0) - move_type {move_type}") if print_info else None
                    if move_type==2:
                        print(f"[INFO] ## Move is accepted due to probabilistically - move_type {move_type}") if print_info else None
                    if move_type==3:
                        print(f"[INFO] ## Move is rejected") if print_info else None
                    best_architecture = new_architecture
                    cur_architecture = new_architecture #Update the current architecture to the new one
                    best_cost = new_cost_val
                else:
                    print(f"[INFO] ## Move is rejected ##") if print_info else None
                    print(f"[INFO] ## Move type is {move_type}") if print_info else None
                
                #### ADD LOG ENTRY ####
            
                print(f"[INFO] Writing log entry ...") if print_info else None
                SA_run_loop += 1 #Increment loop variable as an iteration attemt was made
                
                ####### Dump details #######
                processed_row = process_iteration_wide(new_architecture, iteration_id=SA_run_loop)
                all_rows_data.append(processed_row)
                ############################
                
                log_entry = {
                    'temperature': temperature,
                    'inner_iter': iterations,
                    '1L_move': SA_first_level_move,
                    '2L_move': random_arch_move,
                    'SA_run_loop': SA_run_loop,
                    'best_cost': best_cost,
                    'new_cost': new_cost_val,
                    'cost_diff': cost_diff,
                    'move_accepted': move_accepted,
                    'move_type': move_type
                }
                log_entry.update(norm_cost_dict)
                log_entry.update(raw_cost_dict)
                SA_log_data.append(log_entry)

        #Cool down temperature 
        temperature *= cooling_rate
        
        result_df = pd.DataFrame(SA_log_data)
   

    cache.dump_cache()

    sim_data_csv_results = process_arch_details_dump(all_rows_data=all_rows_data)
    return best_cost, best_architecture, result_df, sim_data_csv_results    

##########################################



if __name__ == "__main__":
    #################

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=int, choices=[i+1 for i in range(len(GEMM_SHAPE))], 
                        help = "workload index to retrieve GEMM shapes from encoded list")
    parser.add_argument("--iteration", type = int, default=1,
                        help="Number of iterations running, default is 1")
    parser.add_argument("--run_name", type = str, default=None,
                        help="Name of current run, used to create work/log folder, default is None")
    parser.add_argument("--cache_file", type = str, default="cfg/static_cache/static_cache.csv",
                        help="Cache file used to accelerate the simulation")
    parser.add_argument("--cost_profile", type = str, default="t1",
                        help="Cost profiles used to calculate cost in SimAnnelaing. Options - t1, t2, t3, t4")

    parser.add_argument("--run_mode", type = str, default="run_sim_anneal",
                        help="run_mode to select type of sim to run. Options- run_sim_anneal, run_exhaustive_dse, run_dbg_check_mutations, run_dbg_find_cost")
    #parser.add_argument("--json_file_path", type = str, default=None,
    #                    help="Path to the JSON file for debugging purposes, default is script/analysis_dir/json_out/example.json")

    #Example: python -m main --workload 1 --cost_profile t1 --run_mode run_sim_anneal --run_name test_run

    args = parser.parse_args()
    
    run_name = args.run_name
    if run_name == None:
        run_name = ""

    wl_idx = args.workload
    iteration = args.iteration
    cache_file = args.cache_file
    cost_profile = args.cost_profile
    run_mode = args.run_mode
    #json_file_path = args.json_file_path

    if wl_idx is None:
        print(f"[Warning] Using Default Workload Index = 1")
        # parser.print_help()
        # exit(-1)
        wl_idx = 1
    

    shape = GEMM_SHAPE[wl_idx]
    GEMM_M = shape[0]
    GEMM_K = shape[1]
    GEMM_N = shape[2]

    file_run_name = f"wl{wl_idx}_{iteration}iteration_{run_name}_{cost_profile}"
    
    print(f"[INFO] Run name: {file_run_name}, cache_file: {cache_file}, Iteration: {iteration}")
    
    cache = SimulationCache(cache_file)
    #################

    for i in range(iteration):
        if run_mode == "run_sim_anneal": #Runs Simulated Annealing
            start_time = time.time()
            best_cost, best_arch, sa_details_csv, sim_results_csv = sim_annealing(
                                                                wl_idx=wl_idx,
                                                                cache_file = cache_file,
                                                                run_name=file_run_name,
                                                                cost_profile=cost_profile,
                                                                initial_temp=40, 
                                                                freezing_temp=1e-3, 
                                                                max_move_per_temp_step=5, #20
                                                                cooling_rate=0.3,
                                                                calibration_iterations=10
                                                                )
            
            dump_results(sa_details_csv, sim_results_csv, best_arch, best_cost, file_run_name)
            end_time = time.time()
            find_run_time(start_time,end_time)
        elif run_mode == "run_calibration": #Runs Calibration
            print(f"[INFO] Running Calibration only")
            start_time = time.time()
            
            calibration_iterations = 10
            print(f"[INFO] Running calibration for {calibration_iterations} iterations to get variation data")
            
            run_calibration(
                wl_idx=wl_idx,
                cache_file=cache_file,
                run_name=file_run_name,
                cost_profile=cost_profile,
                calibration_iterations=calibration_iterations 
            )
            
            end_time = time.time()
            find_run_time(start_time,end_time)
        
        else:
            print(f"[INFO] Please ensure run_mode is correct")
