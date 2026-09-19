# Workload 7 Convergence and Architecture Similarity

This documents the workload 7 annealing budget study and its architecture
similarity analysis. It mirrors `docs/workload6_convergence_report.md`; workload
1 results are in `docs/workload1_convergence_report.md` and workload 10 results
are in `docs/workload10_convergence_report.md`. See
[Reproducing on another workload](#reproducing-on-another-workload).

## Question

Does the annealing search reach the optimum, and do the independent runs
converge to the same architecture?

## Setup

- Workload: `7` = `two_gemm_demo`, a chained pair
  `projection [128, 256, 512]` + `classifier [128, 512, 64]`
  (20,971,520 MACs total).
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once for workload 7 and reused by
  every annealing run (zero stale regenerations across all ten runs).
- Runs: 10 independent runs, seeds `17000`–`17009`, launched in parallel with
  `nohup`, each with its own simulation cache file.
- Schedule: `100,000` moves per run — `INITIAL_TEMP=4000`,
  `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`, `MAX_MOVE_PER_TEMP_STEP=50`.
- Metric: normalized objective (`best_cost_after`), lower is better.

## Results

| Schedule | Best | Worst | Relative spread | Distinct canonical fingerprints | Largest fingerprint group |
|---|---:|---:|---:|---:|---:|
| `100000` | 22.875145 | 22.875145 | 0.00% | 4 | 5 |

All ten runs land on **exactly the same objective** (`22.87514484`, identical
to every printed digit). This is full objective convergence, stronger than
workload 6, where the best plateaued at a 1.50% spread.

### Per-run results

| Run | Objective | Canonical fingerprint | Distance to best |
|---|---:|---|---:|
| r01 | 22.87514484 | `9b172d1aa0a2` | 0.000 |
| r02 | 22.87514484 | `a049ec947400` | 0.143 |
| r03 | 22.87514484 | `0ddfdccbd4f2` | 0.095 |
| r04 | 22.87514484 | `f7d202306585` | 0.143 |
| r05 | 22.87514484 | `0ddfdccbd4f2` | 0.095 |
| r06 | 22.87514484 | `0ddfdccbd4f2` | 0.095 |
| r07 | 22.87514484 | `0ddfdccbd4f2` | 0.095 |
| r08 | 22.87514484 | `a049ec947400` | 0.143 |
| r09 | 22.87514484 | `0ddfdccbd4f2` | 0.095 |
| r10 | 22.87514484 | `f7d202306585` | 0.143 |

### Exact architecture groups

| Group size | Runs |
|---:|---|
| 5 | r03, r05, r06, r07, r09 |
| 2 | r02, r08 |
| 2 | r04, r10 |
| 1 | r01 |

## Architecture similarity

The design core is identical across all ten runs:

- 2 chiplets, tech node `7`, systolic array `64x64`, `256` KB SRAM each
- package `3d` with `ucie_3d` hybrid bonding, linear stack topology
- memory `hbm3`
- dataflow `os`; `static_tiling`, `merge_tiles`, `if_splitting_k`,
  `chiplet_data_sharing_enabled` all zero

Only three knobs vary, and none changes the objective:

| Field | Modal value | Agreement | Distribution |
|---|---|---:|---|
| `pkg.mem_chip1` | `8` | 70% | 8=7; 7=2; 10=1 |
| `pkg.mem_chip2` | `8` | 70% | 8=7; 9=2; 6=1 |
| `wl.assign_workload_in_ascending_order` | `1` | 60% | 1=6; 0=4 |

### Feature distance

Normalized Hamming distance over the union of canonical architecture fields,
measured from the best run:

| Schedule | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.105 | 0.143 | 0.612 |

The residual spread is tiny (max 0.143). The positive Spearman correlation is
an artifact of near-ties: the varying knobs do not move the objective, so the
ranks carry little signal.

### Interpretation

For workload 7 the search converges completely: one objective, one design, with
only HBM channel counts and a mapping flag left undetermined—none of which
affects the score. The converged design is the 2-chiplet, 3D-stacked, `64x64`,
7 nm, HBM3 architecture (fingerprint `0ddfdccbd4f2`), which also appears as the
workload 6 two-chiplet variant. Workload 6 additionally admits an equally good
3-chiplet design; workload 7 does not.

## Cross-workload comparison

| Workload | Schedule | Best | Relative spread | Distinct fingerprints | Largest group | Converged design |
|---|---:|---:|---:|---:|---:|---|
| 6 | 50 | 55.987 | 865.24% | 10 | 1 | none |
| 6 | 75,650 | 36.791 | 10.53% | 6 | 2 | 3-chiplet |
| 6 | 100,000 | 36.791 | 1.50% | 6 | 3 | 3-chiplet |
| 7 | 100,000 | 22.875 | **0.00%** | 4 | 5 | 2-chiplet |

Both workloads converge to the same 2-chiplet, 3D, HBM3, `os` design; workload
6 also has a tied 3-chiplet alternative that keeps its count of distinct
architectures higher. The analyzer is now validated on two workloads.

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Tests: `tests/test_architecture_similarity.py`
- Generated report (local, `reports/` is git-ignored):
  `reports/architecture_similarity/workload7/report.md`
- Per-run tables: `runs_100000.csv`, `distance_matrix_100000.csv`,
  `field_agreement_100000.csv`
- Run outputs (on the server):
  `cfg/gen_arch/wl7_1iteration_workload7_100k_convergence_r*_t1/`

## Reproducing on another workload

Replace `<N>` with the workload index. Run from the repository root on the
server with the virtual environment activated.

### 1. Calibration (once per workload)

```bash
nohup bash -lc '
  cd /absolute/path/to/CarbonPATH &&
  source carbonpath/bin/activate &&
  make calibration WORKLOAD=<N> CALIBRATION_ITERATIONS=10000 \
    RUN_NAME=workload<N>_calibration_10000 SEED=$((11000 + <N>))
' > logs/workload<N>_calibration_10000.log 2>&1 < /dev/null &
```

Wait for it to finish, then confirm `cfg/calibration/calibration_<N>.json`
contains `"_calibration_samples": 10000`.

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

Add more `--set LABEL=PATTERN` arguments to compare schedules side by side, for
example `--set 75650="cfg/gen_arch/wl<N>_1iteration_workload<N>_75650move_convergence_r*_t1"`.
