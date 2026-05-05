from __future__ import annotations

import os
import pandas as pd
import numpy as np
from scalesim.scale_sim import scalesim
from scalesim.utilities.scalesim_report import ScalesimReport
from system.utils.GEMMWorkload import GEMMWorkload
from system.utils.SystolicArray import SystolicArray



class Simulator:
    def __init__(self, systolic_arrays: list[SystolicArray], home_dir = 'simulation', fast_test = False, mode = "USER"):
        self.systolic_array = systolic_arrays
        self.home_dir = home_dir
        self.results = {}
        self.fast_test = fast_test
        self.mode = mode

    def write_config_file(self, sa: SystolicArray) -> str:
        run_name = f"Core{sa.id}_{sa.width}_{sa.height}"
        general = [
            "[general]",
            f"run_name = {run_name}"
        ]
        architecture_presets = [
            "[architecture_presets]",
            f"ArrayHeight:    {sa.height}",
            f"ArrayWidth:     {sa.width}",
            f"IfmapSramSzkB:   {sa.buffer_size}",
            f"FilterSramSzkB:  {sa.buffer_size}",
            f"OfmapSramSzkB:   {sa.buffer_size}",
            "IfmapOffset:    0",
            "FilterOffset:   10000000",
            "OfmapOffset:    20000000",
            f"Bandwidth : {int(sa.dram_bandwidth)}",
            f"Dataflow : {sa.data_flow}",
            "MemoryBanks:    1"
        ]
        run_presets = [
            "[run_presets]",
            f"InterfaceBandwidth: {self.mode}"
        ]

        # Combine all lines with appropriate spacing
        lines = []
        lines.extend(general)
        lines.append('')  # Empty line between sections
        lines.extend(architecture_presets)
        lines.append('')  # Empty line between sections
        lines.extend(run_presets)
        
        # Write to file
        path = os.path.join(self.home_dir, run_name, run_name + '.cfg')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        return path, run_name

    def write_core_workload(self, core: SystolicArray, wl: list[GEMMWorkload] = None) -> str:
        
        if wl:
            workload = wl
        else:
            workload: list[GEMMWorkload] = list(core.workloads)


        num_layer = len(workload)
        if num_layer == 0:
            return None
        
        if num_layer == 1:
            M = workload[0].m
            N = workload[0].n
            K = workload[0].k
            data = {'Layer': 'Layer1', 'M': M, 'N': N, 'K': K}
            df = pd.DataFrame(data, index=[0])

        else:
            layers = [f'Layer{i}' for i in range(1, num_layer + 1)]
            M = []
            N = []
            K = []
            for wl in workload:
                M.append(wl.m)
                N.append(wl.n)
                K.append(wl.k)
            data = {'Layer': layers, 'M': M, 'N': N, 'K': K}
            df = pd.DataFrame(data)

        run_name = f'Core{core.id}_{core.width}_{core.height}'
        # run_name = f"Core{sa.id}_{sa.width}_{sa.height}"

        path = os.path.join(self.home_dir, run_name, run_name + '.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save with trailing comma on each line
        with open(path, 'w') as f:
            # Write header with trailing comma
            f.write('Layer,M,N,K,\n')
            for i in range(len(df)):
                row = df.iloc[i]
                f.write(f"{row['Layer']},{row['M']},{row['N']},{row['K']},\n")  # Trailing comma

        return path, run_name

    def simulate_all_cores(self):

        compute_latency = dict()
        for core in self.systolic_array:
            latency = self.simulate_single_core(core)
            compute_latency[core.id] = latency
        return compute_latency

    def simulate_single_core(self, core: SystolicArray):

        workload_return = self.write_core_workload(core)
        # current core as no workload assigned 
        if workload_return == None:
            print(f"Core[{core.id}]: {core.width}x{core.height} has no assigned workload, NO simulation performed")
            return 0
        workload_path, workload_file = workload_return 
        
        config_path, config_name = self.write_config_file(core)
        # sim_folder = os.path.join(self.home_dir, config_name)
        s = scalesim(config=config_path, 
                     topology=workload_path, 
                     input_type_gemm=True,
                     verbose=False,
                     save_disk_space=True)
        s.config.run_name = config_name
        if self.fast_test:
            total_latency = core.width * core.height ** 2
            core.cycle_per_layer = [total_latency] * len(core.workloads)
        else:
            s.run_scale(self.home_dir)
            # load compute and bandwidth report
            rpt = ScalesimReport()
            rpt.load_compute_report_data(self.home_dir, config_name)
            self.results[config_name] = rpt
            
            total_latency = sum(rpt.get_compute_cycles_all_layer())
            local_res = []
            for i in range(len(core.workloads)):
                    local_res.append(rpt.get_total_cycles_single_layer(i))


            rpt.load_detail_report_data(data_dir=self.home_dir, run_name=config_name)
            core.simulation_reporter = rpt
            
            assert core.simulation_reporter is not None

            core.cycle_per_layer = local_res
       
        core.total_latency = total_latency

        return total_latency     






    def report_latency(self):
        for key, rpt in self.results.items():
            cycles = sum(rpt.get_compute_cycles_all_layer())
            print(f'{key}: Total Cycles: {cycles}')
