# Workload 8 Convergence and Architecture Similarity

This documents the workload 8 annealing campaign and its architecture
similarity analysis. It mirrors `docs/workload7_convergence_report.md`; workload
1 results are in `docs/workload1_convergence_report.md`, workload 4 in
`docs/workload4_convergence_report.md`, workload 6 in
`docs/workload6_convergence_report.md`, and workload 10 in
`docs/workload10_convergence_report.md`. See
[Reproducing on another workload](#reproducing-on-another-workload).

## Question

Does the annealing search reach the optimum, and do the independent runs
converge to the same architecture for a **four-GEMM chain** with three
intermediate boundaries?

## Setup

- Workload: `8` = `four_gemm_demo`, four chained GEMMs
  `block_1..block_4`, each `[128, 128, 128]` (8,388,608 MACs total).
- Three intermediate boundaries, each `128 x 128 = 16,384` bytes (16 KiB).
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once for workload 8 and reused by
  every annealing run (zero stale regenerations across all ten runs).
- Runs: 10 independent runs, seeds `17000`–`17009`, launched in parallel with
  `nohup`, each with its own simulation cache file.
- Schedule: `100,000` moves per run — `INITIAL_TEMP=4000`,
  `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`, `MAX_MOVE_PER_TEMP_STEP=50`.
- Metric: normalized objective (`best_cost_after`), lower is better.

## Results

| Schedule | Best | Worst | Relative spread | Largest 0.1% cluster | Within 0.1% of best | Distinct canonical fingerprints | Largest fingerprint group |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100000` | 10.410697 | 10.410697 | **0.00%** | **10/10** | **10/10** | 3 | 5 |

All ten runs land on **exactly the same objective** (`10.4106974578`, identical
to every printed digit). This is full objective convergence: the largest 0.1%
cluster is all ten runs, and the remaining spread is confined to memory
allocation and one mapping flag.

### Per-run results

| Run | Objective | Canonical fingerprint | Distance to best |
|---|---:|---|---:|
| r01 | 10.4106974578 | `f7d202306585` | 0.000 |
| r02 | 10.4106974578 | `0ddfdccbd4f2` | 0.048 |
| r03 | 10.4106974578 | `f7d202306585` | 0.000 |
| r04 | 10.4106974578 | `0ddfdccbd4f2` | 0.048 |
| r05 | 10.4106974578 | `f7d202306585` | 0.000 |
| r06 | 10.4106974578 | `f7d202306585` | 0.000 |
| r07 | 10.4106974578 | `f7d202306585` | 0.000 |
| r08 | 10.4106974578 | `0ddfdccbd4f2` | 0.048 |
| r09 | 10.4106974578 | `5983441962aa` | 0.095 |
| r10 | 10.4106974578 | `0ddfdccbd4f2` | 0.048 |

### Exact architecture groups

| Group size | Runs |
|---:|---|
| 5 | r01, r03, r05, r06, r07 |
| 4 | r02, r04, r08, r10 |
| 1 | r09 |

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
| `pkg.mem_chip1` | `8` | 90% | 8=9; 9=1 |
| `pkg.mem_chip2` | `8` | 90% | 8=9; 7=1 |
| `wl.assign_workload_in_ascending_order` | `0` | 60% | 0=6; 1=4 |

### Feature distance

Normalized Hamming distance over the union of canonical architecture fields,
measured from the best run:

| Schedule | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.029 | 0.095 | 0.758 |

The residual spread is tiny (max 0.095). The positive Spearman correlation is
an artifact of near-ties: the varying knobs do not move the objective, so the
ranks carry little signal.

### Interpretation

Workload 8 converges completely: one objective, one design core, with only HBM
channel counts and a mapping flag left undetermined—none of which affects the
score. The three fingerprints differ solely by `pkg.mem_chip1`,
`pkg.mem_chip2`, and `wl.assign_workload_in_ascending_order`.

Crucially, the converged design is the **same core as workload 7**: 2 chiplets,
7 nm, `64x64`, 256 KB SRAM, 3D `ucie_3d`, HBM3, output-stationary
(fingerprint `0ddfdccbd4f2`, which is the workload 7 modal design and also
appears as a workload 6 two-chiplet variant). Adding two more chained GEMMs and
two more boundaries therefore did **not** move the optimum away from the
two-chiplet 3D design.

## Cross-workload comparison

| Workload | Chain | Schedule | Best | Relative spread | Largest 0.1% cluster | Distinct fingerprints | Largest group | Converged design |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 7 | 2 GEMMs | 100,000 | 22.875 | 0.00% | 10/10 | 4 | 5 | 2-chiplet |
| 8 | 4 GEMMs | 100,000 | 10.411 | 0.00% | 10/10 | 3 | 5 | 2-chiplet |

Both workloads reach a single objective and the same 2-chiplet, 3D, HBM3, `os`
design core. Chain length (two versus four GEMMs, one versus three boundaries)
changed neither the optimum nor the convergence quality.

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Tests: `tests/test_architecture_similarity.py`
- Generated report (local, `reports/` is git-ignored):
  `reports/architecture_similarity/workload8/report.md`
- Per-run tables: `runs_100000.csv`, `distance_matrix_100000.csv`,
  `field_agreement_100000.csv`
- Run outputs (on the server):
  `cfg/gen_arch/wl8_1iteration_workload8_100k_convergence_r*_t1/`

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

Add more `--set LABEL=PATTERN` arguments to compare schedules side by side.
