# Workload 14 Convergence and Architecture Similarity

This documents the workload 14 annealing campaign and its architecture
similarity analysis. It is the six-GEMM sequential-chain study.

## Question

Does the annealing search reach the same solution across independent runs for
a six-GEMM chain, and do the resulting architectures share a common design
core?

## Setup

- Workload: `14` = `six_gemm_chain`, six GEMMs with shape
  `[128, 128, 128]` (12,582,912 MACs total).
- Five intermediate boundaries, each with a `128 x 128` output
  (`16,384` bytes assuming the configured two-byte elements).
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once for workload 14 and reused by
  every annealing run.
- Runs: 10 independent runs, seeds `17000`–`17009`, each with its own cache.
- Schedule: `100,000` moves per run — `INITIAL_TEMP=4000`,
  `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`,
  `MAX_MOVE_PER_TEMP_STEP=50`.
- Metric: normalized objective (`best_cost_after`), lower is better.

## Results

| Schedule | Best | Worst | Relative spread | Largest 0.1% cluster | Within 0.1% of best | Distinct canonical fingerprints | Largest fingerprint group |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100000` | 10.802030 | 18.475030 | 71.03% | **7/10** | **7/10** | 7 | 3 |

Workload 14 **does not meet** the study's convergence criterion of at least
8/10 runs within 0.1% of the best objective. Seven runs reach the same best
objective, while r09 is a moderate outlier and r03/r08 settle in a much worse
basin.

### Per-run results

| Run | Objective | Canonical fingerprint | Distance to best |
|---|---:|---|---:|
| r01 | 10.8020295173 | `c8fc919fa4b2` | 0.000 |
| r02 | 10.8020295173 | `c8fc919fa4b2` | 0.000 |
| r03 | 18.4750302504 | `0700fff9adf1` | 0.381 |
| r04 | 10.8020295173 | `a049ec947400` | 0.143 |
| r05 | 10.8020295173 | `f7d202306585` | 0.143 |
| r06 | 10.8020295173 | `c8fc919fa4b2` | 0.000 |
| r07 | 10.8020295173 | `9b172d1aa0a2` | 0.095 |
| r08 | 18.4750302504 | `543cb7a13c99` | 0.333 |
| r09 | 13.5603242869 | `43588b26e228` | 0.143 |
| r10 | 10.8020295173 | `f7d202306585` | 0.143 |

### Exact architecture groups

| Group size | Runs |
|---:|---|
| 3 | r01, r02, r06 |
| 2 | r05, r10 |
| 1 | r03 |
| 1 | r04 |
| 1 | r07 |
| 1 | r08 |
| 1 | r09 |

## Architecture similarity

All ten runs use two chiplets at 7 nm with `64x64` systolic arrays, HBM3, and
output-stationary dataflow. The dominant design core is a 3D `ucie_3d`
package with 256 KB SRAM per chiplet.

The main disagreements are:

| Field | Modal value | Agreement | Distribution |
|---|---|---:|---|
| `chip2.sram_buf` | `256` | 90% | 256=9; 512=1 |
| `pkg.HI_pkg_type` | `3d` | 80% | 3d=8; 2.5d=2 |
| `pkg.protocol_3d` | `ucie_3d` | 80% | ucie_3d=8; na=2 |
| `pkg.mem_chip1` | `9` | 30% | 9=3; 7=3; 8=3; 10=1 |
| `pkg.mem_chip2` | `7` | 30% | 7=3; 9=3; 8=3; 6=1 |
| `wl.assign_workload_in_ascending_order` | `1` | 60% | 1=6; 0=4 |

The two worst runs, r03 and r08, are the only 2.5D runs. Their package
topology differs from the dominant 3D design, and both have objective
`18.4750302504`. Run r09 remains 3D but uses a 512 KB second-chiplet SRAM and
has the intermediate objective `13.5603242869`.

### Feature distance

| Schedule | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.138 | 0.381 | 0.879 |

The strong positive correlation is explained by the two 2.5D outliers and the
additional r09 deviation. Architecture differences are therefore materially
associated with objective quality for this workload.

### Interpretation

Workload 14 exposes a real package-level failure mode. The best basin is a
two-chiplet 3D design, but the search reaches a 2.5D basin in two runs and a
different 3D configuration in another. The six-GEMM chain therefore does not
meet the required convergence threshold at 100,000 moves.

## Cross-workload comparison

| Workload | Chain | Best | Relative spread | Largest 0.1% cluster | Distinct fingerprints | Converged design |
|---|---|---:|---:|---:|---:|---|
| 7 | 2 GEMMs | 22.875 | 0.00% | 10/10 | 4 | 2-chiplet 3D |
| 8 | 4 GEMMs | 10.411 | 0.00% | 10/10 | 3 | 2-chiplet 3D |
| 12 | 3 GEMMs | 21.318 | 26.65% | 9/10 | 6 | 2-chiplet 3D |
| 13 | 2 GEMMs | 23.158 | 0.00% | 10/10 | 6 | 2-chiplet 3D |
| 14 | 6 GEMMs | 10.802 | 71.03% | 7/10 | 7 | 2-chiplet 3D |

Workload 14 has the largest spread observed in the sequential set so far. Its
outliers are package-level rather than merely different HBM allocations.

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Generated report: `reports/architecture_similarity/workload14/report.md`
- Per-run table: `reports/architecture_similarity/workload14/runs.csv`
- Additional tables: `field_agreement_100000.csv` and
  `distance_matrix_100000.csv`
- Run outputs on the server:
  `cfg/gen_arch/wl14_1iteration_workload14_100k_convergence_r*_t1/`
