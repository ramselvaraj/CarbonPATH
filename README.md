# CarbonPATH: Carbon-aware pathfinding and architecture optimization for chiplet-based AI systems

As chiplet-based accelerators and advanced packaging become mainstream for scaling DNN performance, **early design decisions**—spanning **workload mapping, architecture, chiplets, and packaging**—increasingly determine not only power, performance, and cost, but also the overall carbon footprint. As a result, **full-stack pathfinding frameworks** that jointly optimize PPAC and embodied + operational carbon are becoming essential to identify **robust, sustainability-aware accelerator configurations**.

This work introduces **CarbonPATH, an early-stage design and optimization framework that integrates embodied and operational carbon accounting** into the exploration of **chiplet-based accelerator architectures**. CarbonPATH enables **true system-level co-design across the full stack**, spanning:

- **Application-level decisions**: **workload mapping, dataflow** (e.g., **OS/IS/WS**), and **resource allocation**

- **Chip-architecture choices**: **memory type (DDR/HBM), chiplet count,** and **network topology**

- **Chiplet configuration parameters**: **technology node, systolic-array sizing,** and **SRAM capacity**

- **Package-level integration + interconnect: 2.5D / 3D / 2.5D+3D, protocol selection,** and **interconnect type**

By jointly modeling these interdependent choices, CarbonPATH supports systematic trade-off analysis across performance, cost, and carbon.


<p align="center">
<img src="figs/carbonPath.jpg" alt="drawing" width="600"/>
<br/>
  <em>CarbonPATH framework overview.</em></figcaption>
</p>

<p align="center">
<img src="figs/co-design-compute-stack.jpg" alt="drawing" width="500"/>
<br/>
  <em>System level co-design.</em></figcaption>
</p>


## Table of Contents 
-   [File structure](#file-structure)
-   [Getting started](#getting-started)
-   [Input parameters and configuration](#input-parameters-and-configuration)
-   [Running CarbonPATH](#running-carbon-path)
-   [Outputs and analysis](#outputs-and-analysis)


## File structure 
- [cfg](./cfg/)
- [chiplet](./chiplet/) 
- [config.py](./config.py)
- [main.py](./main.py)
- [Makefile](./Makefile)
- [script](./script/)
- [system](./system/)


## Getting started
### Prerequisites 

CarbonPATH requires the following: 
- python 3.9.18 
- pip 23.2.1
- python 3.9-venv


Additionally, please refer to the requirements.txt file in this repository. The packages in requirements.txt will be installed in a virtual environment.


### Download and install with bash 
```
Download the repo from https://github.com/ASU-VDA-Lab/CarbonPATH 
cd CarbonPATH
python3 -m venv carbonpath
source carbonpath/bin/activate
pip3 install -r requirements.txt
```

## Input parameters and configuration

### Parameters
CarbonPATH utilizes parameters and configurations stored in the cfg directory. The core parameters for the framework are located in the **cfg/parameter** folder. Below is a list of the various JSON files along with their details. The primary file that users need to modify is [input.json](./cfg/parameters/input.json), as it defines the overall search space for CarbonPATH. In addition to this, several other parameter files are available, as listed below:

```
cfg/parameters/
├── base_spec_7nm.json      [area and power for different systolic arrays at 7nm]
├── base_spec_sram_7nm.json [area and energy for different SRAM sizes at 7nm]
├── bonding_yield.json      [bonding yield for different package types]
├── cost_profiles.json      [weight factors for different optimization profiles]
├── d2d_input.json          [die-to-die datarate, efficiency, pitch information]
├── energy_eff.json         [die-to-die protocol and DRAM energy efficiency]
├── freq_scale.json         [frequency scale across tech nodes]
├── input.json              [main parameter file that defines the search space of PATH-AI]
├── mem_bw.json             [memory bandwidth for different systolic arrays]
├── scaling.json            [area and power scaling for logic]
├── sram_energy.json        [sram energy values and sram energy scaling]
└── sram_scaling.json       [sram area and energy scaling]

chiplet/carbon_model/arch_params/
├── architecture.json       [details of each chiplet]
├── designC.json            [design CFP parameters - architecture power, volume, design iterations]
├── operationalC.json       [operational CFP - lifetime value]
└── packageC.json           [parameters related to 2.5D/3D/2.5D+3D packages]
```
Users can input different parameters of their choice by modifying the above parameter files. CarbonPATH models the overall HI-system's area, power, energy, cost, embodied CFP, operational CFP, and cycle-accurate latency using data from the above JSON files. 

CarbonPATH also supports multiple optimization templates: ```t1, t2, t3, and t4``` ([cost_profiles.json](./cfg/parameters/cost_profiles.json)). Users can additionally customize the weight of various metrics according to their preferences.

### Calibration
Since the metrics used by CarbonPATH have different units and scales, normalization is necessary to prevent any single term from dominating the SA-Cost function. The files below contain an example calibration.json for all the six different workloads used in the paper. 
```
cfg/calibration/
├── calibration_1.json [calibration for WL1]
├── calibration_2.json [calibration for WL2]
├── calibration_3.json [calibration for WL3]
├── calibration_4.json [calibration for WL4]
├── calibration_5.json [calibration for WL5]
└── calibration_6.json [calibration for WL6]
```
Command to run calibration on a particular workload is shown below, by default it runs for 10,000 samples. You can modify ``` calibration_iterations = 10000 ``` to desiered value in [main.py](./main.py)
```
make calibration WORKLOAD=5 
```
**NOTE**: Please ensure the old cfg/calibration_*.json is deleted prior to running new calibration.  

### Simulation cache 
CarbonPATH computes cycle-accurate latency for the AI workloads it runs, which can be time-intensive. To address this, we implemented a lookup table–based simulation cache that dynamically stores key parameters such as systolic array size, workload shape, memory bandwidth, SRAM size, data flow, and the computed cycle count.
During the simulated annealing algorithm, the simulator is invoked only if a cache miss occurs (i.e., a configuration has not been encountered before). This approach significantly speeds up the computation. Additionally, the simulation cache is configured to automatically update on a miss, enabling faster execution for subsequent runs.


## Running Carbon-PATH
There are multiple ways CarbonPATH can be launched. CarbonPATH does an extensive design space exploration, and since the search space is vast, run times vary based on the workload. 

#### Include new GEMM workload
To run on a new GEMM workload update the [workload.json](./cfg/examples/workload.json) and use the provided workload number in the commands below. The [workload.json](./cfg/examples/workload.json) is in M, K, N format as shown below: 
```
"1": [128, 2048, 1000],
```
Here for workload 1, we have M=128, K=2048, and N=1000

#### Single workload commands
CarbonPATH provides a Makefile that allows users to launch the framework across multiple workloads.

###### Remove mobile and other profiles 
```
1. make run  
2. make run WORKLOAD=3 COST_PROFILE=t2  
3. make calibration WORKLOAD=5
```
**"make run"** will launch CarbonPATH framework for design space exploration, by default if WORKLOAD and COST_PROFILE is not mentioned it will use workload 1 and t1 optimization profile as default values. 

**"make calibration"** will launch the calibration only for 10,000 samples. 

#### Multiple runs in parallel
We provide a script that can run calibration and CarbonPATH simulated annealing in parallel across multiple workloads.
**NOTE**: For design space exploration with CarbonPATH, ensure that calibration is completed beforehand.

To delete old calibration files for all workloads either run
```
make clean_calibration 
OR
python script/clean_calibration.py 
```
To launch calibration in parallel for all workloads: in [run_parallel_calibration.sh](./script/run_parallel_calibration.sh) update the RUN_NAME and WORKLOADS of choice and it can be launched:
```
make parallel_calibration
OR 
./scripts/run_parallel_calibraiton.sh
```
To launch CarbonPATH across multiple workloads, update RUN_NAME and WORKLOADS in [run_parallel_carbonpath.sh](./script/run_parallel_carbonpath.sh) and it can be launched:
```
make parallel_carbonpath
OR 
./scripts/run_parallel_carbonpath.sh
```

#### Update simulated annealing hyperparameters 
To customize the default CarbonPATH hyperparameters for your workloads and use case, you can modify them in [main.py](./main.py)
By default, the framework runs with an initial temperature of 4000, 50 iterations per temperature, a cooling rate of 0.99, and a freezing temperature of 0.001.
```
initial_temp=4000, 
freezing_temp=1e-3, 
max_move_per_temp_step=50, #20
cooling_rate=0.99,
```


## Outputs and analysis
Once CarbonPATH completes, it creates a folder based on workload under the generated architecture directory (cfg/gen_arch)

It contains the best_arch*.json file along with other CSV files that are generated along with the runs. Below is an example of data that is generated for wl1 balance optimization profile case: 

```
Example shown below for WL5 templates T2:
cfg/gen_arch/wl5_1iteration__t2/ 
├── best_arch_wl5_1iteration__t2_2026-03-13T02-19-49.json
├── cost_v_iteration_wl5_1iteration__t2_2026-03-13T02-19-49.png
├── sa_arch_detail_wl5_1iteration__t2_2026-03-13T02-19-49.csv
└── sa_metrics_wl5_1iteration__t2_2026-03-13T02-19-49.csv
```
The best_arch_wl5*.json is the file that gives the best optimzied HI-system architecture. More details about it explained below. 

The sa_arch_.csv and sa_metrics_*.csv are data is generated for each iterations in simualted annealing algorithm, it contains all the metric for SA-Cost and also has the architecture information. 

##### Architecture file best_arch*.json 
This file is generated upon completion of CarbonPATH’s design space exploration. It captures the best-performing architecture, optimized for the given workload under the specified optimization profile. It is structured as below: 
- Chiplet information 
- Package information 
  - HI package type 
  - Connection topology between chiplets 
  - Protocol 3D and 2.5D 
  - Memory information 
- Workload mapping 

Differnt example arch.json files for multiple HI-types are shown in [cfg/gen_arch/temp_example](./cfg/gen_arch/temp_example/) directory for multiple chiplet numbers. 

##### CarbonPATH analysis
You can use CarbonPATH and perform carbon-aware early stage design space exploration, shown are some examples of the resutls from paper. 

CarbonPATH enables workload-mapping analysis across different HI-system configurations and reports multiple metrics, including Perf-SI.
<p align="center">
<img src="figs/carbon_wl_map.png" alt="drawing" width="500"/>
<br/>
  <em>Variation of Perf-SI with differnt workload mappings.</em></figcaption>
</p>

CarbonPATH also supports multiple package, protocol, combinations across all HI types.
<p align="center">
<img src="figs/carbon_scatter.jpg" alt="drawing" width="500"/>
<br/>
  <em>Scatter plot of Perf-SI with Dollar cost ($) for package-protocol combinations.</em></figcaption>
</p>

Due to space constraints in the paper, we provide additioanl results [paper_results](./paper_results/)

<!--
## Citation
If you find CarbonPATH useful or relevant to your research, please kindly cite our paper 
```
@misc{sudarshan2026carbonpathcarbonawarepathfindingarchitecture,
      title={CarbonPATH: Carbon-aware pathfinding and architecture optimization for chiplet-based AI systems}, 
      author={Chetan Choppali Sudarshan and Jiajun Hu and Aman Arora and Vidya A. Chhabria},
      year={2026},
      eprint={2603.03878},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2603.03878}, 
}
```
-->
