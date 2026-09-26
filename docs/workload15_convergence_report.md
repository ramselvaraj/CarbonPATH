# Workload 15 Convergence and Architecture Similarity

This documents the workload 15 annealing campaign and its architecture
similarity analysis for a chain whose intermediate tensor stresses capacity.

## Question

Does the annealing search consistently choose the same architecture for a
capacity-stressing two-GEMM chain with a large intermediate tensor?

## Setup

- Workload: `15` = `capacity_chain`,
  `produce [512, 64, 1024]` -> `consume [512, 1024, 64]`
  (67,108,864 MACs total).
- One intermediate boundary: `512 x 1024 = 524,288` bytes.
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once for workload 15 and reused by
  every annealing run.
- Runs: 10 independent runs, seeds `17000`–`17009`, each with its own cache.
- Schedule: `100,000` moves per run — `INITIAL_TEMP=4000`,
  `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`,
  `MAX_MOVE_PER_TEMP_STEP=50`.
- Metric: normalized objective (`best_cost_after`), lower is better.

## Results

| Schedule | Best | Worst | Relative spread | Largest 0.1% cluster | Within 0.1% of best | Distinct canonical fingerprints | Largest fingerprint group |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100000` | 36.737507 | 40.523032 | 10.30% | **7/10** | **7/10** | 8 | 2 |

Workload 15 **does not meet** the study's convergence criterion of at least
8/10 runs within 0.1% of the best objective. Seven runs are tightly grouped
near the best result; r06, r08, and r09 converge to a distinct, higher-cost
two-chiplet basin.

### Per-run results

| Run | Objective | Canonical fingerprint | Distance to best |
|---|---:|---|---:|
| r01 | 36.7388937165 | `254f9f340d52` | 0.077 |
| r02 | 36.7388937165 | `ec310777648a` | 0.115 |
| r03 | 36.7411838602 | `d7e2bbfd5beb` | 0.000 |
| r04 | 36.7388937165 | `0cef79579560` | 0.115 |
| r05 | 36.7375067073 | `d7e2bbfd5beb` | 0.000 |
| r06 | 40.5230316649 | `19bcf19503ad` | 0.346 |
| r07 | 36.7402807258 | `965621cb5404` | 0.115 |
| r08 | 40.5230316649 | `0ddfdccbd4f2` | 0.346 |
| r09 | 40.5230316649 | `5983441962aa` | 0.385 |
| r10 | 36.7402807258 | `965621cb5404` | 0.115 |

### Exact architecture groups

| Group size | Runs |
|---:|---|
| 2 | r03, r05 |
| 2 | r07, r10 |
| 1 | r01 |
| 1 | r02 |
| 1 | r04 |
| 1 | r06 |
| 1 | r08 |
| 1 | r09 |

## Architecture similarity

The seven near-best runs use three 7 nm chiplets with `64x64` arrays, 256 KB
SRAM per chiplet, HBM3, and a 3D `ucie_3d` stack. The three high-cost outliers
use only two chiplets, omitting the third chiplet needed by the dominant
capacity solution.

The main field distributions are:

| Field | Modal value | Agreement | Distribution |
|---|---|---:|---|
| `n_chiplets` | `3` | 70% | 3=7; 2=3 |
| `chip3.sram_buf` | `256` | 70% present | 256=7 |
| `pkg.HI_pkg_type` | `3d` | 100% | 3d=10 |
| `pkg.protocol_3d` | `ucie_3d` | 100% | ucie_3d=10 |
| `pkg.mem_chip1` | `5` | 40% | 5=4; 7=2; 6=2; 8=1; 9=1 |
| `pkg.mem_chip2` | `5` | 50% | 5=5; 6=2; 9=1; 8=1; 7=1 |
| `wl.assign_workload_in_ascending_order` | `1` | 50% | 1=5; 0=5 |

All runs remain 3D and use HBM3. The decisive difference is whether the search
finds the three-chiplet capacity configuration. The two-chiplet runs r06, r08,
and r09 all have objective `40.5230316649`.

### Feature distance

| Schedule | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.162 | 0.385 | 0.745 |

The distance/objective relationship is driven primarily by the three
two-chiplet outliers. Within the seven-run near-best group, the remaining
differences are mostly HBM allocation and mapping order.

### Interpretation

The large intermediate tensor makes chiplet capacity a meaningful search
decision. Most runs find a three-chiplet 3D design, but three runs settle in a
two-chiplet basin that is 10.30% above the best objective. Workload 15 therefore
fails the 8/10 convergence target despite having a clear dominant solution.

## Cross-workload comparison

| Workload | Chain | Best | Relative spread | Largest 0.1% cluster | Distinct fingerprints | Converged design |
|---|---|---:|---:|---:|---:|---|
| 7 | 2 GEMMs | 22.875 | 0.00% | 10/10 | 4 | 2-chiplet 3D |
| 8 | 4 GEMMs | 10.411 | 0.00% | 10/10 | 3 | 2-chiplet 3D |
| 12 | 3 GEMMs | 21.318 | 26.65% | 9/10 | 6 | 2-chiplet 3D |
| 13 | 2 GEMMs | 23.158 | 0.00% | 10/10 | 6 | 2-chiplet 3D |
| 14 | 6 GEMMs | 10.802 | 71.03% | 7/10 | 7 | 2-chiplet 3D |
| 15 | 2 GEMMs | 36.738 | 10.30% | 7/10 | 8 | 3-chiplet 3D |

Workload 15 is the first study in this set where the dominant solution uses
three chiplets; its outliers are lower-capacity two-chiplet designs.

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Generated report: `reports/architecture_similarity/workload15/report.md`
- Per-run table: `reports/architecture_similarity/workload15/runs.csv`
- Additional tables: `field_agreement_100000.csv` and
  `distance_matrix_100000.csv`
- Run outputs on the server:
  `cfg/gen_arch/wl15_1iteration_workload15_100k_convergence_r*_t1/`
