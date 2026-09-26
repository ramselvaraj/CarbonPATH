# Workload 12 Convergence and Architecture Similarity

This documents the workload 12 annealing campaign and its architecture
similarity analysis. It mirrors `docs/workload8_convergence_report.md`; the
related sequential studies are workloads 7, 8, 10, 11, 13, 14, 15, and 16.
See [Reproducing on another workload](#reproducing-on-another-workload).

## Question

Does the annealing search reach the optimum, and do the independent runs
converge to the same architecture for a **three-GEMM chain** with two
intermediate boundaries?

## Setup

- Workload: `12` = `three_gemm_chain`, three chained GEMMs
  `expand [128, 256, 256]` -> `compress [128, 256, 128]` ->
  `project [128, 128, 64]` (13,631,488 MACs total).
- Two intermediate boundaries: `128 x 256 = 32,768` bytes (32 KiB) and
  `128 x 128 = 16,384` bytes (16 KiB).
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once for workload 12 and reused by
  every annealing run (zero stale regenerations across all ten runs).
- Runs: 10 independent runs, seeds `17000`–`17009`, launched in parallel with
  `nohup`, each with its own simulation cache file.
- Schedule: `100,000` moves per run — `INITIAL_TEMP=4000`,
  `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`, `MAX_MOVE_PER_TEMP_STEP=50`.
- Metric: normalized objective (`best_cost_after`), lower is better.

## Results

| Schedule | Best | Worst | Relative spread | Largest 0.1% cluster | Within 0.1% of best | Distinct canonical fingerprints | Largest fingerprint group |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100000` | 21.317750 | 26.998669 | 26.65% | **9/10** | **9/10** | 6 | 4 |

Nine of the ten runs land on **exactly the same objective**
(`21.3177500865`, identical to every printed digit). The tenth run (r03)
settles in a genuinely worse basin, so the largest 0.1% cluster is 9/10, which
meets the ≥8/10 convergence criterion.

### Per-run results

| Run | Objective | Canonical fingerprint | Distance to best |
|---|---:|---|---:|
| r01 | 21.3177500865 | `0ddfdccbd4f2` | 0.000 |
| r02 | 21.3177500865 | `5983441962aa` | 0.143 |
| r03 | 26.9986692335 | `3e3bf61afaec` | 0.238 |
| r04 | 21.3177500865 | `c8fc919fa4b2` | 0.095 |
| r05 | 21.3177500865 | `c8fc919fa4b2` | 0.095 |
| r06 | 21.3177500865 | `0ddfdccbd4f2` | 0.000 |
| r07 | 21.3177500865 | `9b172d1aa0a2` | 0.095 |
| r08 | 21.3177500865 | `0ddfdccbd4f2` | 0.000 |
| r09 | 21.3177500865 | `0ddfdccbd4f2` | 0.000 |
| r10 | 21.3177500865 | `f7d202306585` | 0.048 |

### Exact architecture groups

| Group size | Runs |
|---:|---|
| 4 | r01, r06, r08, r09 |
| 2 | r04, r05 |
| 1 | r02 |
| 1 | r07 |
| 1 | r10 |
| 1 | r03 (outlier) |

## Architecture similarity

The converged design core, shared by the nine tied runs, is:

- 2 chiplets, tech node `7`, systolic array `64x64`, `256` KB SRAM each
- package `3d` with `ucie_3d` hybrid bonding, linear stack topology
- memory `hbm3`
- dataflow `os`; `static_tiling`, `merge_tiles`, `if_splitting_k`,
  `chiplet_data_sharing_enabled` all zero

Among the nine tied runs, only three knobs vary, and none changes the objective:

| Field | Modal value | Agreement | Distribution |
|---|---|---:|---|
| `pkg.mem_chip1` | `8` | 60% | 8=6; 9=3; 10=1 |
| `pkg.mem_chip2` | `8` | 60% | 8=6; 7=3; 6=1 |
| `wl.assign_workload_in_ascending_order` | `1` | 80% | 1=8; 0=2 |

The outlier r03 is the only run that selected a **2.5D** package
(`pkg.HI_pkg_type=2.5d`, `protocol_2.5d=ucie_std`, `2.5d_rdl` connections) and
is the sole source of the `pkg.HI_pkg_type`, `pkg.conn1`, `pkg.conn2`,
`pkg.protocol_2.5d`, and `pkg.protocol_3d` disagreement.

### Feature distance

Normalized Hamming distance over the union of canonical architecture fields,
measured from the best run:

| Schedule | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.071 | 0.238 | 0.224 |

The positive Spearman correlation is driven almost entirely by the single
2.5D outlier; among the nine tied runs the varying knobs do not move the
objective, so the ranks carry little signal.

### Interpretation

Workload 12 mostly converges: nine runs reach one objective and one design core
with only HBM channel counts and a mapping flag left undetermined. The
exception is r03, which stalled in a 2.5D basin at 26.999 (26.6% worse). This is
the first sequential workload in the study to show a genuine package-level
outlier.

The converged design core is the **same as workloads 7, 8, and 13**: 2 chiplets,
7 nm, `64x64`, 256 KB SRAM, 3D `ucie_3d`, HBM3, output-stationary. A
three-GEMM chain with a 32 KiB first intermediate did not move the optimum.

## Cross-workload comparison

| Workload | Chain | Schedule | Best | Relative spread | Largest 0.1% cluster | Distinct fingerprints | Largest group | Converged design |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 7 | 2 GEMMs | 100,000 | 22.875 | 0.00% | 10/10 | 4 | 5 | 2-chiplet 3D |
| 8 | 4 GEMMs | 100,000 | 10.411 | 0.00% | 10/10 | 3 | 5 | 2-chiplet 3D |
| 12 | 3 GEMMs | 100,000 | 21.318 | 26.65% | 9/10 | 6 | 4 | 2-chiplet 3D |
| 13 | 2 GEMMs | 100,000 | 23.158 | 0.00% | 10/10 | 6 | 2 | 2-chiplet 3D |

Chain length and shape did not change the optimum. Workload 12's single 2.5D
outlier is the only package-level divergence observed in the sequential set.

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Tests: `tests/test_architecture_similarity.py`
- Generated report (local, `reports/` is git-ignored):
  `reports/architecture_similarity/workload12/report.md`
- Per-run tables: `runs_100000.csv`, `distance_matrix_100000.csv`,
  `field_agreement_100000.csv`
- Run outputs (on the server):
  `cfg/gen_arch/wl12_1iteration_workload12_100k_convergence_r*_t1/`

## Reproducing on another workload

Replace `<N>` with the workload index. Run from the repository root on the
server with the virtual environment activated.

### 1. Calibration (once per workload)

```bash
nohup bash -lc '
  cd /absolute/path/to/CarbonPATH &&
  source carbonpath/bin/activate &&
  make calibration WORKLOAD=<N> CALIBRATION_ITERATIONS=10000 \
    RUN_NAME=workload<N>_calibration_10000 SEED=$((11000 + <N>)) \
    CACHE_FILE=cfg/static_cache/cache_cal_<N>.csv
' > logs/workload<N>_calibration_10000.log 2>&1 < /dev/null &
```

Use a private `CACHE_FILE` when other calibrations run concurrently, so they do
not clobber the shared `cfg/static_cache/static_cache.csv`. Wait for it to
finish, then confirm `cfg/calibration/calibration_<N>.json` contains
`"_calibration_samples": 10000`.

### 2. Ten parallel runs at the chosen schedule

```bash
for i in $(seq -w 1 10); do
  cache="cfg/static_cache/cache_w<N>_100k_r${i}.csv"
  cp -f cfg/static_cache/static_cache.csv "$cache"
  nohup bash -lc "
    cd /absolute/path/to/CarbonPATH &&
    source carbonpath/bin/activate &&
    make sim_anneal WORKLOAD=<N> ITERATION=1 CALIBRATION_ITERATIONS=10000 \
      INITIAL_TEMP=4000 FREEZING_TEMP=7.5e-6 MAX_MOVE_PER_TEMP_STEP=50 COOLING_RATE=0.99 \
      RUN_NAME=workload<N>_100k_convergence_r${i} SEED=$((17000 + 10#${i} - 1)) \
      CACHE_FILE=${cache}
  " > logs/workload<N>_100k_convergence_r${i}.log 2>&1 < /dev/null &
done
```

For a different budget, change `FREEZING_TEMP` (with `INITIAL_TEMP=4000`,
`COOLING_RATE=0.99`, `MAX_MOVE_PER_TEMP_STEP=50`):

| Target moves | `FREEZING_TEMP` |
|---:|---:|
| 50 | use the CLI defaults (`INITIAL_TEMP=40 FREEZING_TEMP=5e-4 MAX_MOVE_PER_TEMP_STEP=5 COOLING_RATE=0.3`) |
| 75,650 | `1e-3` |
| 100,000 | `7.5e-6` |

### 3. Architecture similarity report

```bash
python -m script.analyze_architecture_similarity \
  --workload <N> \
  --set 100000="cfg/gen_arch/wl<N>_1iteration_workload<N>_100k_convergence_r*_t1" \
  --output "reports/architecture_similarity/workload<N>"
```

Add more `--set LABEL=PATTERN` arguments to compare schedules side by side.
