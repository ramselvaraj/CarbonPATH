# Workload 6 Convergence and Architecture Similarity

This documents the workload 6 annealing budget study and the architecture
similarity analysis built to support it. It is reproducible for any workload;
see [Reproducing on another workload](#reproducing-on-another-workload).
Workload 1 results are in `docs/workload1_convergence_report.md`; workload 7
results are in `docs/workload7_convergence_report.md`; workload 10 results are
in `docs/workload10_convergence_report.md`.

## Question

Does more annealing budget improve the optimum, and do the independent runs
converge to the same architecture?

## Setup

- Workload: `6` = `[1316, 24, 144]` (4,548,096 MACs), the smallest of
  workloads 1–6.
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once for workload 6 and reused by
  every annealing run (zero stale regenerations across all 30 runs).
- Runs: 10 independent runs per schedule, seeds `17000`–`17009`, launched in
  parallel with `nohup`, each with its own simulation cache file.
- Metric: normalized objective (`best_cost_after`), lower is better.

### Schedules

| Label | Initial temp | Freezing temp | Cooling | Moves/level | Total moves |
|---|---:|---:|---:|---:|---:|
| `50` | 40 | 5e-4 | 0.3 | 5 | 50 |
| `75650` | 4000 | 1e-3 | 0.99 | 50 | 75,650 |
| `100000` | 4000 | 7.5e-6 | 0.99 | 50 | 100,000 |

## Results

| Schedule | Best | Worst | Relative spread | Distinct canonical fingerprints | Largest fingerprint group |
|---|---:|---:|---:|---:|---:|
| `50` | 55.987 | 540.412 | 865.24% | 10 | 1 |
| `75650` | 36.791 | 40.667 | 10.53% | 6 | 2 |
| `100000` | 36.791 | 37.342 | 1.50% | 6 | 3 |

- The 50-move run is far too short: objectives span almost 10x and every run
  ends on a different architecture.
- 75,650 moves already reach the optimum (`36.791`).
- 100,000 moves do **not** beat that optimum but collapse the objective spread
  from 10.53% to 1.50% and grow the modal architecture cluster from 2 to 3.

## Architecture similarity

At 100,000 moves the design core is identical across all ten runs:

- tech node `7`, systolic array `64x64`
- package `3d` with `ucie_3d` hybrid bonding
- memory `hbm3`
- dataflow `os`; `static_tiling`, `merge_tiles`, `if_splitting_k`,
  `chiplet_data_sharing_enabled` all zero
- linear 3D stack topology

Only four knobs vary:

| Group | Runs | Chiplets | SRAM per chiplet | HBM channels | Objective |
|---|---|---|---:|---|---:|
| A | r10 | 3 | 256,256,256 | 5,5,6 | 36.791413 |
| B | r01,r04,r08 | 3 | 256,256,256 | 5,6,5 | 36.791448 |
| C | r02,r09 | 3 | 256,256,256 | 5,6,5 | 36.791448 |
| D | r05,r07 | 3 | 256,256,256 | 5,5,6 | 36.791768 |
| E | r03 | 3 | 256,256,256 | 6,5,5 | 36.791839 |
| F | r06 | 2 | 256,256 | 8,8 | 37.342296 |

So nine of ten runs are the same design point with parameter-level variation;
the only structural outlier is the two-chiplet run (F, +1.5%).

### Feature distance

Normalized Hamming distance over the union of canonical architecture fields,
measured from the best run:

| Schedule | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `50` | 0.610 | 0.800 | 0.867 |
| `75650` | 0.127 | 0.346 | 0.636 |
| `100000` | 0.104 | 0.346 | 0.685 |

Mean structural distance to the best run falls from 0.610 (50 moves) to 0.104
(100,000 moves). The positive Spearman correlation means farther architectures
generally score worse, but the relationship is not monotonic.

### Interpretation

Short runs are not just noisier — they explore a genuinely different, more
diverse region of the design space and land on worse designs. Long runs collapse
onto a single design point; the residual variation is confined to HBM channel
attachment, one mapping flag, and (in one run) chiplet count. For the thesis the
honest claim is: *the search converges to a 3-chiplet, 3D-stacked, 64x64, 7 nm,
HBM3 design; extra budget beyond ~75k moves buys reproducibility, not a better
optimum.*

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Tests: `tests/test_architecture_similarity.py`
- Generated report (local, `reports/` is git-ignored):
  `reports/architecture_similarity/workload6/report.md`
- Per-run and per-set tables:
  `runs.csv`, `runs_<set>.csv`, `distance_matrix_<set>.csv`,
  `field_agreement_<set>.csv`
- Run outputs (on the server): `cfg/gen_arch/wl6_1iteration_workload6_100k_convergence_r*_t1/`
  and `cfg/gen_arch/wl6_1iteration_workload6_75650move_convergence_r*_t1/`

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
