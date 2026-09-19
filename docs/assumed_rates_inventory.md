# Assumed Rates And Calculation Inputs

This inventory traces numeric rates, factors, and defaults that affect the
energy, performance, cost, yield, or carbon calculations. "Source" means the
source visible in this repository; it does not claim that an uncited table has
been independently validated.

## Executive Summary

| Category | Values | Repository source | Basis visible in code |
|---|---|---|---|
| Carbon intensity | `700 gCO2/kWh` | `chiplet/carbon_model/arch_params/designC.json:7`; defaults in `CO2_func.py:59,136,281` | Configured assumption; no external citation found |
| Product lifetime | `17,520 h` (2 years) | `arch_params/operationalC.json:2`; default formula `2*365*24` in `CO2_func.py:280` | Configured assumption |
| Design iterations / production volume | `90`, `100,000` | `arch_params/designC.json:3-4` | Scales design carbon in `calculate_CO2`; no external citation found |
| Transistors per gate | `8` | `designC.json:5`; defaults in `CO2_func.py:59,135,281` | Model assumption |
| Power per core | `10` | `designC.json:6`; defaults in `CO2_func.py:59,135,281` | Model assumption |
| Bonding yield | `1.0`, `0.99`, `0.95`, `0.90`, `0.80`, `0.75`, `0.70` | `cfg/parameters/bonding_yield.json:2-9` | Configured empirical assumptions; comments cite one 99% value |
| Wafer diameter | `450` | `CO2_func.py:59,266` | Function default; unit is presumably mm from the area formula |
| Packaging geometry | Interposer `65 nm`, layers `6/5/8`, pitches and sizes in `packageC.json` | `arch_params/packageC.json:2-12` | Configured packaging assumptions |
| Router/packaging geometry | `4.47`, `0.33`, EMIB `[5*5]`, node `22` | `CO2_func.py:164,192,200,210-214` | Hardcoded model assumptions; no citation attached to active factors |

## Carbon Model Rates And Factors

| Input | Value(s) / unit | Used by | Source and logic | Assessment |
|---|---|---|---|---|
| Carbon intensity | `700 gCO2/kWh` | `design_costs`, `power_chip`, packaging design carbon | `designC.json:7`; passed through `calculate_CO2` and used in `CO2_func.py:25-26,105-106` | Assumed/configured; no citation in repo |
| Carbon intensity (alternate representation) | `0.700 kgCO2/kWh` | Legacy/helper carbon path | `chiplet/n_utils.py:1717` | Numerically equivalent to `700 gCO2/kWh`; separate hardcoded representation |
| Lifetime | `17,520 h` = 2 years; fallback `2*365*24` | Operational carbon | `operationalC.json:2`; `CO2_func.py:280,313` | Assumed/configured |
| Alternate lifetime | `3 years` | `main.py:349,362,376,536,545` | Directly embedded in operational and embedded-carbon calculations | Separate assumption from carbon-model 2-year default |
| Manufacturing iterations | `90` | Design-carbon scaling | `designC.json:3`; `calculate_CO2` default `num_iter=90` at `CO2_func.py:279`; `CO2_func.py:316` | Assumed/configured |
| Production count | `100,000` | Design-carbon amortization | `designC.json:4`; `calculate_CO2` default `Ns=1e5` at `CO2_func.py:280`; `CO2_func.py:316` | Assumed/configured |
| Transistors per gate | `8` | Converts transistor count to gates | `designC.json:5`; `CO2_func.py:23` | Assumed/configured |
| Power per core | `10` | Design energy estimate | `designC.json:6`; `CO2_func.py:25` | Assumed/configured; unit is not stated in code |
| Yield model constants | `1e4`, `1e-6`, divisor `10`, exponent `-10` | Die yield | `CO2_func.py:14-16` | Hardcoded formula constants; unit conversions and model shape are undocumented |
| Defect density | `[0.2, 0.11, 0.09, 0.08, 0.07, 0.05]` | `yield_calc` | `tech_params/defect_density.json:2`; indexed by `[7,10,14,22,28,65]` in `tech_scaling.py:6,23-26` | External-looking table, but no citation/units in repo |
| Cost per area | `[26.13,19.49,16.84,16.74,14.138,8]` | Silicon manufacturing carbon/cost proxy | `tech_params/cpa_scaling.json:2`; `CO2_func.py:61,86` | External-looking table; units and citation absent |
| Transistor density | `[82.86e6,38.72e6,18.09e6,8.45e6,3.95e6] / mm2` | Design carbon | `tech_params/transistors_scaling.json:2`; `CO2_func.py:22` | External-looking table; node order is `[7,10,14,22,28]`; citation absent |
| Gates per core-hour | `3645.8333` (effectively same for all nodes) | Design carbon | `tech_params/gates_perhr_scaling.json:2`; `CO2_func.py:24` | Configured rate; citation and unit interpretation absent |
| BEOL/FEOL ratio | `[0.5090,0.4887,0.4916,0.4855,0.5675,0.5675]` | Active/passive/RDL/EMIB packaging carbon | `tech_params/beol_feol_scaling.json:2`; `CO2_func.py:170,205,207,214` | External-looking table; citation absent |
| Dynamic-power ratio | `[0.164167,0.258575,0.353535,0.507937,0.651163,0.8]` | Activity-adjusted operational power | `tech_params/dyn_pwr_scaling.json:2`; `CO2_func.py:101-105` | External-looking table; citation absent |
| Activity factors | `[0.2,0.667,0.1]` = active, on, average power | Operational power | `CO2_func.py:96-105,280` | Default assumption; names/units are only defined by variable assignment |
| Router area factor | `4.47` per chiplet for active packaging | Active interposer carbon | `CO2_func.py:163-170` | Hardcoded empirical/model factor; no citation attached |
| Router area factor | `0.33` converted using 14 nm area scaling | 3D/passive/RDL/EMIB router carbon | `CO2_func.py:192-203` | Hardcoded model factor; preceding comment references an unclear “16” conversion |
| EMIB area and node | `5*5` area per interface; `22 nm` | EMIB package carbon | `CO2_func.py:209-214` | Hardcoded packaging assumptions; units not stated |
| Bonding yields | 2D `1.0`; RDL `0.99`; EMIB/passive `0.95`; active `0.90`; TSV `0.80`; u-bump `0.75`; hybrid `0.70` | Divides package/router carbon at `CO2_func.py:218-219`; 3D yield exponentiated at `196` | `cfg/parameters/bonding_yield.json:2-9` | Configured assumptions. Comment at `CO2_func.py:140` says 99% is from “Cost-effective design of scalable high-performance systems using active and passive interposers”; other values have no source |
| RDL layer adjustment | `RDLLayers / numBEOL`, default `6/8` | RDL carbon | `CO2_func.py:207-208`; `packageC.json:3,11` | Derived from configured layer counts |
| Wafer geometry | `pi`, `/4`, `4`, `1/sqrt(2)` | Dies-per-wafer and silicon wastage | `CO2_func.py:266-272` | Geometric formula; not empirical rates |
| Wafer diameter | `450` | Wafer area/wastage | `CO2_func.py:59,76,266`; no value in `packageC.json` | Hardcoded default; unit not explicitly documented |

## Technology Scaling Tables

All rows below are arrays ordered by `tech_indices = [7, 10, 14, 22, 28]` nm,
except defect density, CPA, BEOL/FEOL, and dynamic-power ratio, which append
the packaging node `65` nm (`tech_scaling.py:6,23-46`). These are inputs, not
values derived by the Python code.

| Table | Values by node `[7,10,14,22,28]` (and `65` where applicable) | Source | Logic |
|---|---|---|---|
| Logic area | `[1,1,1,1,1]` | `tech_params/logic_scaling.json:2` | Area normalization |
| Logic delay | `[1,1.25268817,1.562502831,1.959590643,2.464222503]` | `logic_scaling.json:3` | Technology scaling factor |
| Logic energy | `[1,1.36425648,1.727115717,2.72479564,3.367003367]` | `logic_scaling.json:4` | Technology scaling factor |
| Logic EDP | `[1,1.333333333,2,3.164556962,4.166666667]` | `logic_scaling.json:5` | Technology scaling factor |
| Logic power | `[1,1,1,1,1]` | `logic_scaling.json:6` | Technology scaling factor |
| Logic throughput | `[1,0.925925926,0.883392226,0.851788756,0.788643533]` | `logic_scaling.json:7` | Technology scaling factor |
| Analog area/power/energy | Area `[1,1,1,1.974743319,2.278875162]`; power `[1,1,1,1.497716895,1.749333333]`; energy `[1,1,1,1.577656676,1.949494949]` | `analog_scaling.json:2,4,6` | Technology scaling factors |
| SRAM area/power/energy | Area `[1,1,1.902779873,3.757501843,4.336197791]`; power `[1,1,1.231707317,1.844748858,2.154666667]`; energy `[1,1,1.26597582,1.997275204,2.468013468]` | `sram_scaling.json:2,4,6` | Technology scaling factors |

The remaining analog/SRAM delay, EDP, and throughput arrays duplicate the
technology-scaled values shown in `analog_scaling.json:3,5,7` and
`sram_scaling.json:3,5,7`. No provenance metadata is stored alongside any of
these tables.

## Energy, Frequency, And Optimization Inputs

| Input | Values / unit | Source | Basis |
|---|---|---|---|
| Base frequency | `1,000,000,000 Hz` (1 GHz) | `cfg/parameters/freq_scale.json:2` | Configured baseline |
| Frequency scaling | 28 nm `0.36`; 22 nm `0.467`; 14 nm `0.6`; 10 nm `0.74`; 7 nm `1.0` | `freq_scale.json:3-8` | Configured technology assumptions |
| DRAM energy | DDR4 `30`; DDR5 `20`; HBM2 `5`; HBM3 `3` pJ/bit | `cfg/parameters/energy_eff.json:2-7` | Configured technology-rate table; citation absent |
| Die-to-die energy | UCIe standard `0.5`; UCIe advanced `0.25`; UCIe 3D `0.005`; AIB `0.25`; BoW `0.75` pJ/bit | `energy_eff.json:8-14` | Configured technology-rate table; citation absent |
| Cost-profile weights | `t1`: equal energy/performance/area/cost; `t2`, `t3`, `t4` use values `0.1`, `0.2`, `0.6`, `0.7`, `0.8` | `cfg/parameters/cost_profiles.json:2-32` | Decision/objective assumptions |
| Calibration sample count | `10` | `network.py:433` | Hardcoded operational assumption |
| Default cost profile | `t1` | `network.py:408` | Behavioral default, not a physical rate |

## Unit Conversions And Formula Constants

These are not empirical rates, but they materially affect results and should
not be mistaken for sourced assumptions.

| Constant | Location | Purpose |
|---|---|---|
| `1000` | `CO2_func.py:25,105`; `network.py:67,106,239`; `main.py:349` | Converts power/energy quantities to kWh or scales latency units |
| `1e3`, `1e6` | `network.py:224-225` | Reporting conversions to microseconds and microjoules |
| `365*24` | `CO2_func.py:280`; `main.py:362,376` | Calendar hours per year |
| `1e4`, `1e-6`, `/10`, `-10` | `CO2_func.py:15` | Yield model scaling and shape |
| `RDLLayers/numBEOL` | `CO2_func.py:208` | Layer-count ratio, not an independently sourced rate |

## Provenance Gaps And Inconsistencies

- The repository stores values in JSON/code but does not attach citations,
  dataset versions, or units to most technology tables.
- The carbon model defaults to two years (`17,520 h`), while `main.py` uses a
  three-year lifetime for another calculation path. These should be reconciled
  or explicitly documented as different scopes.
- `CO2_func.py:140-142` documents a 99% bonding yield and 10% area overhead as
  coming from “Cost-effective design of scalable high-performance systems
  using active and passive interposers.” The active code loads the bonding
  yield from JSON, and the stated 10% area overhead is commented out at
  `CO2_func.py:198`.
- `EMIBLayers` is accepted and configured, but the active EMIB calculation at
  `CO2_func.py:209-214` uses the hardcoded `5*5` interface area and does not
  visibly apply `EMIBLayers`.
- The apparent unit of `Power_per_core=10`, the defect-density units, CPA units,
  and the units of several packaging dimensions are not declared in code.
