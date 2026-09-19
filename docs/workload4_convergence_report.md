# Workload 4 Convergence and Architecture Similarity

This documents the workload 4 annealing budget study and its architecture
similarity analysis. It mirrors the other convergence reports; workload 1 is in
`docs/workload1_convergence_report.md`, workload 3 in
`docs/workload3_convergence_report.md`, workload 6 in
`docs/workload6_convergence_report.md`, workload 7 in
`docs/workload7_convergence_report.md`, and workload 10 in
`docs/workload10_convergence_report.md`. See
[Reproducing on another workload](#reproducing-on-another-workload).

## Question

Do the independent runs converge to the same score and (nearly) the same
architecture—at least 80% tied within ~0.1%, differing only in minor knobs such
as memory placement or mapping flags?

## Setup

- Workload: `4` = a single GEMM `[128, 2048, 1000]` (262,144,000 MACs).
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once and reused by every annealing
  run (zero stale regenerations across all ten runs).
- Runs: 10 independent runs, seeds `17000`–`17009`, launched in parallel with
  `nohup`, each with its own simulation cache file.
- Schedule: `100,000` moves per run — `INITIAL_TEMP=4000`,
  `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`, `MAX_MOVE_PER_TEMP_STEP=50`.

## Result: the 80% same-score bar is met

| Metric | Value |
|---|---:|
| Best | 89.016232 |
| Worst | 96.723024 |
| Relative spread | 8.66% |
| Distinct canonical fingerprints | 5 |
| Largest 0.1% cluster | **9/10** |
| Within 0.1% of the best | 1/10 |

**9 of 10 runs share an identical score** (`96.7230244109`, equal to ten
decimal places), and within that cluster the *only* differing fields are the HBM
channel split (`pkg.mem_chip1` / `pkg.mem_chip2`, always summing to 16). Every
other architecture field is identical. So the "same score, only memory
differences" pattern holds—but that cluster is **not** the optimum.

The tenth run (r07) is a unique outlier that found a better design, 8.66% below
the cluster.

### Per-run results

| Run | Objective | Canonical fingerprint | Distance to best |
|---|---:|---|---:|
| r07 | 89.01623198 | `bd88cc2cc06a` | 0.000 |
| r01 | 96.72302441 | `6aa0acd72e5c` | 0.538 |
| r02 | 96.72302441 | `6aa0acd72e5c` | 0.538 |
| r03 | 96.72302441 | `6aa0acd72e5c` | 0.538 |
| r04 | 96.72302441 | `4a8e20dd0500` | 0.500 |
| r05 | 96.72302441 | `4a8e20dd0500` | 0.500 |
| r06 | 96.72302441 | `9105f206bb1f` | 0.538 |
| r08 | 96.72302441 | `6aa0acd72e5c` | 0.538 |
| r09 | 96.72302441 | `6aa0acd72e5c` | 0.538 |
| r10 | 96.72302441 | `4ba478690c76` | 0.538 |

### Exact architecture groups

| Group size | Runs |
|---:|---|
| 5 | r01, r02, r03, r08, r09 (`6aa0acd72e5c`) |
| 2 | r04, r05 (`4a8e20dd0500`) |
| 1 | r06 (`9105f206bb1f`) |
| 1 | r10 (`4ba478690c76`) |
| 1 | r07 (`bd88cc2cc06a`) |

## Architecture similarity

### The tied cluster (9 runs)

All nine are the **same design**, constant in every field except the HBM
channel split:

- 2 chiplets, tech node `7`
- chiplet 1 `128x128` / `1024` KB SRAM; chiplet 2 `64x64`
- `3d` package, `ucie_3d`, `3d_u_bump` connections, HBM3, `os` dataflow
- `static_tiling`, `merge_tiles`, `if_splitting_k`,
  `chiplet_data_sharing_enabled`, `assign_workload_in_ascending_order` all fixed

| Field | Values | Note |
|---|---|---|
| `pkg.mem_chip1` | 11, 12, 13, 14 | |
| `pkg.mem_chip2` | 2, 3, 4, 5 | always `mem_chip1 + mem_chip2 = 16` |

Because these four fingerprints (`6aa0acd72e5c`, `4a8e20dd0500`,
`9105f206bb1f`, `4ba478690c76`) produce a **bit-identical objective**, the
memory split is a no-op for the `t1` score.

### The optimum (r07)

The one run that escaped found a **3-chiplet** design:

- chiplets: `96x96`/512, `96x96`/512, `64x64`/768 (all 7 nm)
- linear `3d_hyb_bond` stack, `ucie_3d`, HBM3, `os` dataflow
- HBM channels 7 / 5 / 4 (16 total)
- `assign_workload_in_ascending_order = 1`

So the tied cluster's 2-chiplet, `128x128`/1024 design is a local optimum; a
3-chiplet configuration does 8.66% better. As with workloads 1 and 3, the
chiplet count is the decisive structural knob—but here nine runs still agreed on
the same score.

### Feature distance

| Set | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.477 | 0.538 | 0.818 |

## Cross-workload comparison

| Workload | Best | Relative spread | Largest 0.1% cluster | Within 0.1% of best | Meets 80% same-score bar? |
|---|---:|---:|---:|---:|---|
| 1 (150k) | 107.327362 | 11.80% | 4/10 | 1/10 | no |
| 1 (100k) | 110.210733 | 7.22% | 5/10 | 2/10 | no |
| 3 (100k) | 86.357035 | 24.85% | 4/10 | 4/10 | no |
| 4 (100k) | 89.016232 | 8.66% | **9/10** | 1/10 | **yes** (cluster is suboptimal) |
| 6 (100k) | 36.791413 | 1.50% | 9/10 | 9/10 | yes |
| 7 (100k) | 22.875145 | 0.00% | 10/10 | 10/10 | yes |
| 10 (100k) | 35.573191 | 21.76% | 8/10 | 8/10 | yes |

The "same score, only memory differences" regime appears for workloads 4, 6, 7,
and 10. Workload 4 is the interesting case where the tied majority is *not* the
optimum: nine runs agree on a 2-chiplet design while one finds a better
3-chiplet design.

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Tests: `tests/test_architecture_similarity.py`
- Generated report (local, `reports/` is git-ignored):
  `reports/architecture_similarity/workload4/report.md`
- Per-run tables: `runs_100000.csv`, `distance_matrix_100000.csv`,
  `field_agreement_100000.csv`
- Run outputs (on the server):
  `cfg/gen_arch/wl4_1iteration_workload4_100k_convergence_r*_t1/`

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
| 150,000 | `3.24e-10` |

### 3. Architecture similarity report

```bash
python -m script.analyze_architecture_similarity \
  --workload <N> \
  --set 100000="cfg/gen_arch/wl<N>_1iteration_workload<N>_100k_convergence_r*_t1" \
  --output "reports/architecture_similarity/workload<N>"
```

Add more `--set LABEL=PATTERN` arguments to compare budgets side by side.
