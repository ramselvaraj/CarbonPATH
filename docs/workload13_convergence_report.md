# Workload 13 Convergence and Architecture Similarity

This documents the workload 13 annealing campaign and its architecture
similarity analysis. It mirrors `docs/workload12_convergence_report.md`; the
related sequential studies are workloads 7, 8, 10, 11, 12, 14, 15, and 16.
See [Reproducing on another workload](#reproducing-on-another-workload).

## Question

Does the annealing search reach the optimum, and do the independent runs
converge to the same architecture for a **wide -> narrow -> wide bottleneck
chain**?

## Setup

- Workload: `13` = `bottleneck_chain`, two chained GEMMs
  `compress [128, 512, 128]` -> `expand [128, 128, 512]`
  (16,777,216 MACs total).
- One intermediate boundary: `128 x 128 = 16,384` bytes (16 KiB), the narrow
  bottleneck shared by both GEMMs.
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once for workload 13 and reused by
  every annealing run (zero stale regenerations across all ten runs).
- Runs: 10 independent runs, seeds `17000`–`17009`, launched in parallel with
  `nohup`, each with its own simulation cache file.
- Schedule: `100,000` moves per run — `INITIAL_TEMP=4000`,
  `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`, `MAX_MOVE_PER_TEMP_STEP=50`.
- Metric: normalized objective (`best_cost_after`), lower is better.

## Results

| Schedule | Best | Worst | Relative spread | Largest 0.1% cluster | Within 0.1% of best | Distinct canonical fingerprints | Largest fingerprint group |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100000` | 23.157654 | 23.157654 | **0.00%** | **10/10** | **10/10** | 6 | 2 |

All ten runs land on **exactly the same objective** (`23.1576538406`, identical
to every printed digit). This is full objective convergence, matching workloads
7 and 8.

### Per-run results

| Run | Objective | Canonical fingerprint | Distance to best |
|---|---:|---|---:|
| r01 | 23.1576538406 | `9b172d1aa0a2` | 0.000 |
| r02 | 23.1576538406 | `5983441962aa` | 0.143 |
| r03 | 23.1576538406 | `19bcf19503ad` | 0.095 |
| r04 | 23.1576538406 | `c8fc919fa4b2` | 0.095 |
| r05 | 23.1576538406 | `0ddfdccbd4f2` | 0.095 |
| r06 | 23.1576538406 | `5983441962aa` | 0.143 |
| r07 | 23.1576538406 | `19bcf19503ad` | 0.095 |
| r08 | 23.1576538406 | `f7d202306585` | 0.143 |
| r09 | 23.1576538406 | `c8fc919fa4b2` | 0.095 |
| r10 | 23.1576538406 | `f7d202306585` | 0.143 |

### Exact architecture groups

| Group size | Runs |
|---:|---|
| 2 | r02, r06 |
| 2 | r03, r07 |
| 2 | r04, r09 |
| 2 | r08, r10 |
| 1 | r01 |
| 1 | r05 |

Four fingerprints appear twice and two appear once; every group differs only in
memory allocation and the mapping flag.

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
| `pkg.mem_chip1` | `9` | 40% | 9=4; 8=3; 7=2; 10=1 |
| `pkg.mem_chip2` | `7` | 40% | 7=4; 8=3; 9=2; 6=1 |
| `wl.assign_workload_in_ascending_order` | `1` | 60% | 1=6; 0=4 |

### Feature distance

Normalized Hamming distance over the union of canonical architecture fields,
measured from the best run:

| Schedule | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.105 | 0.143 | 0.721 |

The residual spread is tiny (max 0.143). The positive Spearman correlation is
an artifact of near-ties: the varying knobs do not move the objective, so the
ranks carry little signal.

### Interpretation

Workload 13 converges completely: one objective, one design core, with only HBM
channel counts and a mapping flag left undetermined. The narrow 16 KiB
bottleneck did not introduce a package-level or topology-level outlier; unlike
workload 12, no run settled on 2.5D.

The converged design core is the **same as workloads 7, 8, and 12**: 2 chiplets,
7 nm, `64x64`, 256 KB SRAM, 3D `ucie_3d`, HBM3, output-stationary. A
wide -> narrow -> wide bottleneck chain did not move the optimum.

## Cross-workload comparison

| Workload | Chain | Schedule | Best | Relative spread | Largest 0.1% cluster | Distinct fingerprints | Largest group | Converged design |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 7 | 2 GEMMs | 100,000 | 22.875 | 0.00% | 10/10 | 4 | 5 | 2-chiplet 3D |
| 8 | 4 GEMMs | 100,000 | 10.411 | 0.00% | 10/10 | 3 | 5 | 2-chiplet 3D |
| 12 | 3 GEMMs | 100,000 | 21.318 | 26.65% | 9/10 | 6 | 4 | 2-chiplet 3D |
| 13 | 2 GEMMs | 100,000 | 23.158 | 0.00% | 10/10 | 6 | 2 | 2-chiplet 3D |

Chain length and shape did not change the optimum. Workload 13 shows the
strongest convergence in the set (10/10 at a single objective).

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Tests: `tests/test_architecture_similarity.py`
- Generated report (local, `reports/` is git-ignored):
  `reports/architecture_similarity/workload13/report.md`
- Per-run tables: `runs_100000.csv`, `distance_matrix_100000.csv`,
  `field_agreement_100000.csv`
- Run outputs (on the server):
  `cfg/gen_arch/wl13_1iteration_workload13_100k_convergence_r*_t1/`

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
