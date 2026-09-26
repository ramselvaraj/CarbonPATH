# Workload 16 Convergence and Architecture Similarity

This documents the workload 16 annealing campaign and its architecture
similarity analysis for a wide-intermediate two-GEMM chain.

## Question

Does the annealing search consistently find the same objective and architecture
for a chain with a `1024 x 1024` intermediate tensor?

## Setup

- Workload: `16` = `wide_intermediate_chain`,
  `produce [1024, 32, 1024]` -> `consume [1024, 1024, 32]`
  (67,108,864 MACs total).
- One intermediate boundary: `1024 x 1024 = 1,048,576` bytes.
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once for workload 16 and reused by
  every annealing run.
- Runs: 10 independent runs, seeds `17000`–`17009`, each with its own cache.
- Schedule: `100,000` moves per run — `INITIAL_TEMP=4000`,
  `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`,
  `MAX_MOVE_PER_TEMP_STEP=50`.
- Metric: normalized objective (`best_cost_after`), lower is better.

## Results

| Schedule | Best | Worst | Relative spread | Largest 0.1% cluster | Within 0.1% of best | Distinct canonical fingerprints | Largest fingerprint group |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100000` | 38.399148 | 38.399148 | **0.00%** | **10/10** | **10/10** | 4 | 5 |

All ten runs reach exactly the same objective (`38.3991476879`). Workload 16
meets the convergence criterion with a perfect 10/10 objective cluster.

### Per-run results

| Run | Objective | Canonical fingerprint | Distance to best |
|---|---:|---|---:|
| r01 | 38.3991476879 | `054f4aab3860` | 0.000 |
| r02 | 38.3991476879 | `d9bac42e274c` | 0.048 |
| r03 | 38.3991476879 | `b7a2e2e9c1dc` | 0.095 |
| r04 | 38.3991476879 | `054f4aab3860` | 0.000 |
| r05 | 38.3991476879 | `054f4aab3860` | 0.000 |
| r06 | 38.3991476879 | `d9bac42e274c` | 0.048 |
| r07 | 38.3991476879 | `b6aab38cf06a` | 0.143 |
| r08 | 38.3991476879 | `054f4aab3860` | 0.000 |
| r09 | 38.3991476879 | `b6aab38cf06a` | 0.143 |
| r10 | 38.3991476879 | `054f4aab3860` | 0.000 |

### Exact architecture groups

| Group size | Runs |
|---:|---|
| 5 | r01, r04, r05, r08, r10 |
| 2 | r02, r06 |
| 2 | r07, r09 |
| 1 | r03 |

## Architecture similarity

All ten runs select two 7 nm chiplets with `64x64` arrays, 512 KB SRAM per
chiplet, HBM3, and a 3D `ucie_3d` stack. All use output-stationary dataflow,
with static tiling, merge tiles, K splitting, and chiplet data sharing disabled.

Only three fields vary across the runs:

| Field | Modal value | Agreement | Distribution |
|---|---|---:|---|
| `pkg.mem_chip1` | `8` | 70% | 8=7; 9=3 |
| `pkg.mem_chip2` | `8` | 70% | 8=7; 7=3 |
| `wl.assign_workload_in_ascending_order` | `1` | 60% | 1=6; 0=4 |

Package topology, memory type, chiplet count, technology, array size, SRAM
capacity, dataflow, and all other modeled controls agree across all ten runs.

### Feature distance

| Schedule | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.048 | 0.143 | 0.418 |

Because every run has the same objective, the residual distance correlation has
no optimization significance. The architectural variation is objective-neutral
within this workload and schedule.

### Interpretation

Workload 16 is a complete convergence result. The wide intermediate requires
the larger 512 KB SRAM configuration, but it does not create package-level
divergence: every run selects the same two-chiplet 3D design core. Only HBM
channel allocation and mapping order remain underdetermined.

## Cross-workload comparison

| Workload | Chain | Best | Relative spread | Largest 0.1% cluster | Distinct fingerprints | Converged design |
|---|---|---:|---:|---:|---:|---|
| 7 | 2 GEMMs | 22.875 | 0.00% | 10/10 | 4 | 2-chiplet 3D |
| 8 | 4 GEMMs | 10.411 | 0.00% | 10/10 | 3 | 2-chiplet 3D |
| 12 | 3 GEMMs | 21.318 | 26.65% | 9/10 | 6 | 2-chiplet 3D |
| 13 | 2 GEMMs | 23.158 | 0.00% | 10/10 | 6 | 2-chiplet 3D |
| 14 | 6 GEMMs | 10.802 | 71.03% | 7/10 | 7 | 2-chiplet 3D |
| 15 | 2 GEMMs | 36.738 | 10.30% | 7/10 | 8 | 3-chiplet 3D |
| 16 | 2 GEMMs | 38.399 | 0.00% | 10/10 | 4 | 2-chiplet 3D |

The wide intermediate changes the SRAM requirement from the 256 KB dominant
configuration seen in workloads 7, 8, 12, 13, and 14 to 512 KB, while
preserving a stable two-chiplet 3D topology.

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Generated report: `reports/architecture_similarity/workload16/report.md`
- Per-run table: `reports/architecture_similarity/workload16/runs.csv`
- Additional tables: `field_agreement_100000.csv` and
  `distance_matrix_100000.csv`
- Run outputs on the server:
  `cfg/gen_arch/wl16_1iteration_workload16_100k_convergence_r*_t1/`
