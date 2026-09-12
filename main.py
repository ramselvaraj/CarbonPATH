from ast import arg
import hashlib
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
from system.utils.IntermediateMemoryPolicy import (
    INTERMEDIATE_POLICIES,
    build_boundary_mapping,
    plan_boundary,
)
from config import print_info, fast_test, latency_en, sram_selection_mode


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
WORKLOAD_CONFIGS = {int(k): v for k, v in workload.items()}
CALIBRATION_MODEL_VERSION = 3
######


def parse_workload_entry(workload_id, entry):
    """Normalize a legacy GEMM or an ordered, cold-memory GEMM sequence."""
    if isinstance(entry, list):
        sequence_name = f"workload_{workload_id}"
        raw_gemms = [{"name": "gemm_1", "shape": entry}]
    elif isinstance(entry, dict):
        sequence_name = entry.get("name", f"workload_{workload_id}")
        raw_gemms = entry.get("gemms")
        if not isinstance(raw_gemms, list) or len(raw_gemms) < 2:
            raise ValueError(f"Workload {workload_id} must define at least two GEMMs")
    else:
        raise ValueError(f"Workload {workload_id} must be a GEMM shape or sequence object")

    gemms = []
    names = set()
    for index, raw_gemm in enumerate(raw_gemms, start=1):
        if not isinstance(raw_gemm, dict):
            raise ValueError(f"GEMM {index} in workload {workload_id} must be an object")
        name = raw_gemm.get("name", f"gemm_{index}")
        shape = raw_gemm.get("shape")
        if not isinstance(name, str) or not name:
            raise ValueError(f"GEMM {index} in workload {workload_id} must have a name")
        if name in names:
            raise ValueError(f"GEMM names must be unique within workload {workload_id}")
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError(f"GEMM '{name}' must have three positive integer dimensions [M, K, N]")
        names.add(name)
        gemms.append({"name": name, "shape": tuple(shape)})

    for previous, current in zip(gemms, gemms[1:]):
        previous_m, _, previous_n = previous["shape"]
        current_m, current_k, _ = current["shape"]
        if current_m != previous_m or current_k != previous_n:
            raise ValueError(
                f"GEMM '{current['name']}' must consume '{previous['name']}' output: "
                f"expected M={previous_m}, K={previous_n}, got M={current_m}, K={current_k}"
            )

    return {"id": workload_id, "name": sequence_name, "gemms": gemms}



def build_scheduler_system(workload, arch_dict):

    system = ChipletSystem(arch_dict=arch_dict, BASE_FREQUENCY=FREQUENCY)
    scheduler = Scheduler(workload, system.core_dict.values(), mapping_dict=arch_dict['WL_mapping']['mapping'])
    scheduler.static_workload_scheduling()

    return scheduler, system



def prepare_single_gemm(cache: SimulationCache, arch_dict: dict, shape):
    wl = GEMMWorkload(*shape)
    scheduler, system = build_scheduler_system(wl, arch_dict)
    cache.all_cores_simulation_with_cache(scheduler.systolic_arrays)
    return scheduler, system


def simulate_single_gemm(
    cache: SimulationCache,
    arch_dict: dict,
    shape,
    prepared=None,
    activation_from_dram=True,
    output_to_dram=True,
):
    scheduler, system = prepared or prepare_single_gemm(cache, arch_dict, shape)
    latency_ns, energy_pj = scheduler.system_modeling(
        system,
        activation_from_dram=activation_from_dram,
        output_to_dram=output_to_dram,
    )
    sram_energy_pj = scheduler._get_sram_energy()
    return {
        "latency_ns": latency_ns,
        "dram_interconnect_energy_pj": energy_pj,
        "sram_energy_pj": sram_energy_pj,
    }


def collect_tile_mappings(scheduler):
    mappings = []
    for core in getattr(scheduler, "systolic_arrays", ()):
        for tile in core.workloads:
            mappings.append(
                {
                    "core_id": core.id,
                    "m": tile.m,
                    "k": tile.k,
                    "n": tile.n,
                    "m_offset": tile.m_offset,
                    "k_offset": tile.k_offset,
                    "n_offset": tile.n_offset,
                }
            )
    return mappings


def simulate_latency_energy(
    cache: SimulationCache,
    arch_dict: dict,
    workload_sequence,
    dbg=False,
    intermediate_policy="direct_forward",
):
    if intermediate_policy not in INTERMEDIATE_POLICIES:
        raise ValueError(f"Unknown intermediate-memory policy: {intermediate_policy}")
    prepared_gemms = [
        prepare_single_gemm(cache, arch_dict, gemm["shape"])
        for gemm in workload_sequence["gemms"]
    ]
    boundary_plans = []
    for index, (producer, consumer) in enumerate(
        zip(prepared_gemms, prepared_gemms[1:]),
        start=1,
    ):
        mapping = build_boundary_mapping(
            producer[0], producer[1], consumer[0], consumer[1]
        )
        boundary_plans.append(
            plan_boundary(
                boundary_index=index,
                intermediate_bytes=mapping.intermediate_bytes,
                transfers=mapping.transfers,
                cores=mapping.cores,
                policy=intermediate_policy,
                mapping_valid=mapping.valid,
                mapping_error=mapping.error,
            )
        )

    gemm_metrics = []
    for index, (gemm, prepared) in enumerate(
        zip(workload_sequence["gemms"], prepared_gemms)
    ):
        incoming_boundary = boundary_plans[index - 1] if index > 0 else None
        outgoing_boundary = boundary_plans[index] if index < len(boundary_plans) else None
        metrics = simulate_single_gemm(
            cache,
            arch_dict,
            gemm["shape"],
            prepared=prepared,
            activation_from_dram=(incoming_boundary is None or incoming_boundary.uses_dram),
            output_to_dram=(outgoing_boundary is None or outgoing_boundary.uses_dram),
        )
        if incoming_boundary is not None and incoming_boundary.selected_method == "direct_forward":
            metrics["latency_ns"] += incoming_boundary.latency_ns
            metrics["dram_interconnect_energy_pj"] += incoming_boundary.energy_pj
        metrics["tile_mappings"] = collect_tile_mappings(prepared[0])
        gemm_metrics.append({"name": gemm["name"], "shape": gemm["shape"], **metrics})

    latency_ns = sum(metric["latency_ns"] for metric in gemm_metrics)
    energy_pj = sum(metric["dram_interconnect_energy_pj"] for metric in gemm_metrics)
    sram_energy_pj = sum(metric["sram_energy_pj"] for metric in gemm_metrics)
    return latency_ns, energy_pj, sram_energy_pj, gemm_metrics, boundary_plans


def add_per_gemm_metrics(output, gemm_metrics, power):
    for index, metrics in enumerate(gemm_metrics, start=1):
        prefix = f"gemm_{index}"
        output[f"{prefix}_name"] = metrics["name"]
        output[f"{prefix}_shape"] = "x".join(str(value) for value in metrics["shape"])
        output[f"{prefix}_latency_ns"] = metrics["latency_ns"]
        output[f"{prefix}_dram_interconnect_energy_pj"] = metrics[
            "dram_interconnect_energy_pj"
        ]
        output[f"{prefix}_sram_energy_pj"] = metrics["sram_energy_pj"]
        output[f"{prefix}_total_energy_pj"] = (
            metrics["dram_interconnect_energy_pj"]
            + metrics["sram_energy_pj"]
            + power * metrics["latency_ns"] * 1000
        )


def add_boundary_metrics(output, boundary_plans):
    output["intermediate_policy_requested"] = (
        boundary_plans[0].requested_policy if boundary_plans else "direct_forward"
    )
    for plan in boundary_plans:
        prefix = f"boundary_{plan.boundary_index}"
        output[f"{prefix}_selected_method"] = plan.selected_method
        output[f"{prefix}_intermediate_bytes"] = plan.intermediate_bytes
        output[f"{prefix}_retained_bytes"] = plan.retained_bytes
        output[f"{prefix}_forwarded_bytes"] = plan.forwarded_bytes
        output[f"{prefix}_dram_spilled_bytes"] = plan.dram_spilled_bytes
        output[f"{prefix}_dram_traffic_bytes"] = plan.dram_traffic_bytes
        output[f"{prefix}_latency_ns"] = plan.latency_ns
        output[f"{prefix}_energy_pj"] = plan.energy_pj
        output[f"{prefix}_producer_cores"] = ",".join(map(str, plan.producer_cores))
        output[f"{prefix}_consumer_cores"] = ",".join(map(str, plan.consumer_cores))
        output[f"{prefix}_routes"] = ";".join(
            "->".join(map(str, route)) for route in plan.routes
        )
        output[f"{prefix}_fallback_reason"] = plan.fallback_reason

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

def calculate_cost(
    profile_name='t1',
    cost_avgerage=dict,
    system_dict=dict,
    cache=SimulationCache,
    workload_sequence=None,
    intermediate_policy="direct_forward",
    simulation_result=None,
    system_metrics=None,
):
    if workload_sequence is None:
        raise ValueError("A workload sequence is required for cost calculation")
    print("\n[INFO] --- System Analysis Metrics ---") if print_info else None
    power, area, dollar_cost = system_metrics or calculate_system_metrics(
        system_dict=system_dict
    )
    
    
    print(f"[DEBUG COST] ************** LATENCY ************** ") if print_info else None
    if latency_en and simulation_result is not None:
        latency, energy_comm, energy_sram, gemm_metrics, boundary_plans = (
            simulation_result
        )
    elif latency_en:
        print(f"[INFO] Working on calcuting performance ...") if print_info else None
        latency, energy_comm, energy_sram, gemm_metrics, boundary_plans = simulate_latency_energy(
            cache,
            system_dict,
            workload_sequence,
            intermediate_policy=intermediate_policy,
        )
    else: 
        latency = 0
        energy_comm = 0
        energy_sram = 0
        gemm_metrics = [
            {
                "name": gemm["name"],
                "shape": gemm["shape"],
                "latency_ns": 0,
                "dram_interconnect_energy_pj": 0,
                "sram_energy_pj": 0,
            }
            for gemm in workload_sequence["gemms"]
        ]
        boundary_plans = []
    
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

    add_per_gemm_metrics(raw_cost_dict, gemm_metrics, power)
    add_boundary_metrics(raw_cost_dict, boundary_plans)
    raw_cost_dict["intermediate_policy_requested"] = intermediate_policy
    return cost_val, norm_cost_dict, raw_cost_dict


def gen_initial_arch(config_path,stack_diff_size=True):
    
    system_builder = SystemGenerator(config_path=config_path,stack_diff_size=stack_diff_size)
    final_config = system_builder.generate_system()
    return final_config    


def calibration_identity(
    config_path, workload_sequence, intermediate_policy="direct_forward"
):
    if isinstance(config_path, dict):
        search_space = config_path
    else:
        with open(config_path, encoding="utf-8") as file:
            search_space = json.load(file)
    payload = {
        "model_version": CALIBRATION_MODEL_VERSION,
        "search_space": search_space,
        "workload": workload_sequence,
        "intermediate_policy": intermediate_policy,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_calibration(cost_averages):
    average_keys = {
        "avg_energy",
        "avg_area",
        "avg_dollar_cost",
        "avg_latency",
        "avg_embCarbon",
        "avg_opeCarbon",
    }
    metrics = ("energy", "latency", "area", "cost", "embCarbon", "opeCarbon")
    statistic_keys = {
        f"{metric}_{statistic}"
        for metric in metrics
        for statistic in ("min", "max", "stddev", "mean", "median")
    }
    required = average_keys | statistic_keys
    missing = sorted(required - cost_averages.keys())
    if missing:
        raise ValueError(f"Calibration is missing required metrics: {missing}")
    for key in required:
        value = cost_averages[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Calibration metric {key} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"Calibration metric {key} must be finite")
    for metric in metrics:
        if cost_averages[f"{metric}_min"] <= 0:
            raise ValueError(f"Calibration metric {metric}_min must be positive")
        if cost_averages[f"{metric}_median"] <= 0:
            raise ValueError(f"Calibration metric {metric}_median must be positive")
        if cost_averages[f"{metric}_max"] < cost_averages[f"{metric}_min"]:
            raise ValueError(f"Calibration range for {metric} is invalid")
    return cost_averages

def get_calib_cost_avg(
    calibration_iterations,
    config_path,
    cache,
    calibration_file_path,
    workload_sequence,
    intermediate_policy="direct_forward",
):
    if calibration_iterations <= 0:
        raise ValueError("calibration_iterations must be positive")
    expected_identity = calibration_identity(
        config_path, workload_sequence, intermediate_policy
    )
    
    # Check if the calibration file already exists.
    if os.path.exists(calibration_file_path):
        print(f"\n[INFO] --- Loading existing cost averages from {calibration_file_path} ---") if print_info else None
        with open(calibration_file_path, 'r') as f:
            cost_averages = json.load(f)
        if cost_averages.get("_calibration_identity") == expected_identity:
            validate_calibration(cost_averages)
            print(f"[INFO] --- Successfully loaded averages: {cost_averages} ---") if print_info else None
            return cost_averages
        print("[INFO] Existing calibration is stale; regenerating it")

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
        latency, energy_comm, energy_sram, gemm_metrics, boundary_plans = simulate_latency_energy(
            cache=cache,
            arch_dict=final_config,
            workload_sequence=workload_sequence,
            intermediate_policy=intermediate_policy,
        )
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
        energy_compute = power * latency * 1000
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
        add_per_gemm_metrics(cost_dict, gemm_metrics, power)
        add_boundary_metrics(cost_dict, boundary_plans)
        
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
        "_calibration_model_version": CALIBRATION_MODEL_VERSION,
        "_calibration_identity": expected_identity,
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
    validate_calibration(cost_averages)

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

def run_calibration(
    wl_idx,
    workload_sequence,
    cache_file,
    run_name,
    cost_profile,
    calibration_iterations=10000,
    intermediate_policy="direct_forward",
):
    
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
        calibration_file_path=calibration_file_path,
        workload_sequence=workload_sequence,
        intermediate_policy=intermediate_policy,
    )
    ###################################
    print(f"[CALIBRATION] Calibration is completed")
##########################################


def run_policy_comparison(
    wl_idx,
    workload_sequence,
    architecture_file,
    cache_file,
    run_name,
    cost_profile,
):
    if architecture_file is None:
        raise ValueError("--architecture_file is required for run_policy_compare")
    with open(architecture_file) as file:
        architecture = json.load(file)
    calibration_file = f"cfg/calibration/calibration_{wl_idx}.json"
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(
            f"Run calibration first; expected {calibration_file}"
        )
    with open(calibration_file) as file:
        cost_averages = json.load(file)
    expected_identity = calibration_identity(
        "cfg/parameters/input.json", workload_sequence
    )
    if cost_averages.get("_calibration_identity") != expected_identity:
        raise ValueError(
            f"Calibration is stale; rerun calibration for workload {wl_idx}"
        )

    cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)
    rows = []
    for policy in (
        "cold_dram",
        "ideal_on_chip",
        "local_sram",
        "direct_forward",
    ):
        objective, normalized, raw = calculate_cost(
            profile_name=cost_profile,
            cost_avgerage=cost_averages,
            system_dict=architecture,
            cache=cache,
            workload_sequence=workload_sequence,
            intermediate_policy=policy,
        )
        rows.append({"policy": policy, "objective": objective, **normalized, **raw})
    cache.dump_cache()

    output_name = run_name or f"wl{wl_idx}_{cost_profile}"
    output_path = f"reports/intermediate_policy_comparison_{output_name}.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    print(result[["policy", "objective", "latency", "energy"]].to_string(index=False))
    print(f"[INFO] Policy comparison written to {output_path}")
    return result, output_path



##########################################
####### Simulated Annealing Function

def sim_annealing(
    wl_idx,
    workload_sequence,
    cache_file,
    run_name,
    cost_profile,
    initial_temp=4000,
    freezing_temp=1e-3,
    max_move_per_temp_step=20,
    cooling_rate=0.99,
    calibration_iterations=200,
    intermediate_policy="direct_forward",
    random_seed=None,
    input_file_path="cfg/parameters/input.json",
    calibration_file_path=None,
    initial_architecture=None,
):
    if initial_temp <= 0:
        raise ValueError("initial_temp must be positive")
    if freezing_temp <= 0 or freezing_temp >= initial_temp:
        raise ValueError("freezing_temp must be positive and below initial_temp")
    if max_move_per_temp_step <= 0:
        raise ValueError("max_move_per_temp_step must be positive")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1")
    if random_seed is not None:
        random.seed(random_seed)
    
    print(f"[STANDALONE_MODE] Standalone framework mode is enabled")
    calibration_file_path = calibration_file_path or (
        f"cfg/calibration/calibration_{wl_idx}.json"
    )
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
        calibration_file_path=calibration_file_path,
        workload_sequence=workload_sequence,
        intermediate_policy=intermediate_policy,
    )
    ###################################

    # Calibration may be loaded or generated. Reset here so that cache state does
    # not alter seeded initial generation or the annealing trajectory.
    if random_seed is not None:
        random.seed(random_seed)
    
    
    # Use the new top-level function to generate the architecture
    cur_architecture = (
        copy.deepcopy(initial_architecture)
        if initial_architecture is not None
        else gen_initial_arch(config_path=input_file_path, stack_diff_size=True)
    )
    
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
        cache=cache,
        workload_sequence=workload_sequence,
        intermediate_policy=intermediate_policy,
        )
    
    #Debug
    if print_info:
        print("[INFO] Cost averages is ",json.dumps(cost_avg))
        print(f"[INFO] Calculated cost value is {cost_val}")
        print(f"[INFO] Normalized cost dict is {norm_cost_dict}")
        print(f"[INFO] Raw cost dict is {raw_cost_dict}")
    
    # Track the accepted annealing state separately from the global best.
    current_cost = cost_val
    best_cost = cost_val
    best_architecture = copy.deepcopy(cur_architecture)
    
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
                                                    cache=cache,
                                                    workload_sequence=workload_sequence,
                                                    intermediate_policy=intermediate_policy,
                )
                print(f"\n[INFO] The new cost is {new_cost_val} and current cost is {current_cost}") if print_info else None
            
                #Calcualte the cost delta
                cost_diff = new_cost_val - current_cost
            
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
                    cur_architecture = new_architecture #Update the current architecture to the new one
                    current_cost = new_cost_val
                    if new_cost_val < best_cost:
                        best_architecture = copy.deepcopy(new_architecture)
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
    parser.add_argument("--workload", type=int, choices=sorted(WORKLOAD_CONFIGS),
                        help = "workload index to retrieve an ordered GEMM sequence")
    parser.add_argument("--iteration", type = int, default=1,
                        help="Number of iterations running, default is 1")
    parser.add_argument("--run_name", type = str, default=None,
                        help="Name of current run, used to create work/log folder, default is None")
    parser.add_argument("--cache_file", type = str, default="cfg/static_cache/static_cache.csv",
                        help="Cache file used to accelerate the simulation")
    parser.add_argument("--cost_profile", type = str, default="t1",
                        help="Cost profiles used to calculate cost in SimAnnelaing. Options - t1, t2, t3, t4")
    parser.add_argument(
        "--intermediate_policy",
        choices=sorted(INTERMEDIATE_POLICIES),
        default="direct_forward",
        help="How sequential GEMM intermediates are transferred",
    )
    parser.add_argument(
        "--architecture_file",
        type=str,
        default=None,
        help="Fixed architecture JSON used by run_policy_compare",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible calibration and simulated-annealing randomness",
    )

    parser.add_argument("--run_mode", type = str, default="run_sim_anneal",
                        help="Options: run_sim_anneal, run_calibration, run_policy_compare")
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
    intermediate_policy = args.intermediate_policy
    architecture_file = args.architecture_file
    seed = args.seed
    #json_file_path = args.json_file_path

    if wl_idx is None:
        print(f"[Warning] Using Default Workload Index = 1")
        # parser.print_help()
        # exit(-1)
        wl_idx = 1
    

    workload_sequence = parse_workload_entry(wl_idx, WORKLOAD_CONFIGS[wl_idx])
    print(
        f"[INFO] Workload sequence: {workload_sequence['name']} "
        f"({len(workload_sequence['gemms'])} GEMM(s))"
    )

    file_run_name = f"wl{wl_idx}_{iteration}iteration_{run_name}_{cost_profile}"
    if intermediate_policy != "direct_forward":
        file_run_name += f"_{intermediate_policy}"
    
    print(f"[INFO] Run name: {file_run_name}, cache_file: {cache_file}, Iteration: {iteration}")
    
    if run_mode == "run_policy_compare":
        run_policy_comparison(
            wl_idx=wl_idx,
            workload_sequence=workload_sequence,
            architecture_file=architecture_file,
            cache_file=cache_file,
            run_name=file_run_name,
            cost_profile=cost_profile,
        )
        sys.exit(0)
    #################

    for i in range(iteration):
        iteration_seed = None if seed is None else seed + i
        if run_mode == "run_sim_anneal": #Runs Simulated Annealing
            start_time = time.time()
            best_cost, best_arch, sa_details_csv, sim_results_csv = sim_annealing(
                                                                wl_idx=wl_idx,
                                                                workload_sequence=workload_sequence,
                                                                cache_file = cache_file,
                                                                run_name=file_run_name,
                                                                cost_profile=cost_profile,
                                                                initial_temp=40, 
                                                                freezing_temp=1e-3, 
                                                                max_move_per_temp_step=5, #20
                                                                cooling_rate=0.3,
                                                                calibration_iterations=10,
                                                                intermediate_policy=intermediate_policy,
                                                                random_seed=iteration_seed,
                                                                )
            
            dump_results(sa_details_csv, sim_results_csv, best_arch, best_cost, file_run_name)
            end_time = time.time()
            find_run_time(start_time,end_time)
        elif run_mode == "run_calibration": #Runs Calibration
            print(f"[INFO] Running Calibration only")
            start_time = time.time()
            
            calibration_iterations = 10
            print(f"[INFO] Running calibration for {calibration_iterations} iterations to get variation data")
            
            if iteration_seed is not None:
                random.seed(iteration_seed)
            run_calibration(
                wl_idx=wl_idx,
                workload_sequence=workload_sequence,
                cache_file=cache_file,
                run_name=file_run_name,
                cost_profile=cost_profile,
                calibration_iterations=calibration_iterations,
                intermediate_policy=intermediate_policy,
            )
            
            end_time = time.time()
            find_run_time(start_time,end_time)
        
        else:
            print(f"[INFO] Please ensure run_mode is correct")
