# Workload 1 Convergence and Architecture Similarity

This documents the workload 1 annealing budget study (100,000 and 150,000
moves) and its architecture similarity analysis. It mirrors the other
convergence reports; workload 6 is in `docs/workload6_convergence_report.md`,
workload 7 in `docs/workload7_convergence_report.md`, and workload 10 in
`docs/workload10_convergence_report.md`. See
[Reproducing on another workload](#reproducing-on-another-workload).

## Question

For the largest single-GEMM workload, do the independent runs converge to the
same score and (nearly) the same architecture? The target pattern is at least
80% of runs tied within ~0.1% and differing only in minor knobs such as memory
placement or mapping flags.

## Setup

- Workload: `1` = a single GEMM `[512, 768, 3072]` (1,207,959,552 MACs) — the
  largest workload in this study, ~36x workload 10.
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Runs: 10 independent runs per budget, seeds `17000`–`17009`, launched in
  parallel with `nohup`, each with its own simulation cache file.
- Metric: normalized objective (`best_cost_after`), lower is better.

| Set | Calibration | Schedule | Moves/run |
|---|---|---:|---:|
| `100000` | 10,000 samples | `INITIAL_TEMP=4000`, `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`, 50/level | 100,000 |
| `150000` | 15,000 samples | `INITIAL_TEMP=4000`, `FREEZING_TEMP=3.24e-10`, `COOLING_RATE=0.99`, 50/level | 150,000 |

## Result: the 80% convergence bar is not met

| Set | Best | Worst | Relative spread | Distinct canonical fingerprints | Best found by |
|---|---:|---:|---:|---:|---:|
| `100000` | 110.210733 | 118.171777 | 7.22% | 6 | 2/10 |
| `150000` | 107.327362 | 119.988936 | 11.80% | 7 | 1/10 |

Score agreement for the 150,000-move set (best = `107.327362`):

| Tolerance | Runs within best |
|---|---:|
| 0.1% | 1/10 |
| 1% | 1/10 |
| 2% | 4/10 |
| 5% | 5/10 |
| 10% | 9/10 |

The best score is unique. The largest identical-score clusters are only 3–4
runs, and they sit ~9.6% above the best. More budget **widened** the spread
(7.22% -> 11.80%); it did not buy convergence.

### 150,000-move per-run results

| Run | Objective | Canonical fingerprint | Chiplets | Distance to best |
|---|---:|---|---:|---:|
| r05 | 107.32736196 | `cafd3fbe2a2a` | 4 | 0.000 |
| r07 | 109.39751722 | `0ac1f0a55e7b` | 4 | 0.531 |
| r08 | 109.39751722 | `0ac1f0a55e7b` | 4 | 0.531 |
| r02 | 109.39758394 | `3a39d0d59ea6` | 4 | 0.500 |
| r06 | 111.50407090 | `96583726909a` | 3 | 0.562 |
| r09 | 117.67556494 | `11595201a379` | 2 | 0.688 |
| r03 | 117.67569303 | `11595201a379` | 2 | 0.688 |
| r10 | 117.67569303 | `11595201a379` | 2 | 0.688 |
| r04 | 117.67569303 | `62079071694a` | 2 | 0.719 |
| r01 | 119.98893634 | `abe957dd0f0f` | 3 | 0.500 |

### 100,000-move per-run results

| Run | Objective | Canonical fingerprint | Chiplets |
|---|---:|---|---:|
| r02 | 110.21073317 | `3a39d0d59ea6` | 4 |
| r10 | 110.21073317 | `3a39d0d59ea6` | 4 |
| r05 | 111.47931941 | `8f103782f910` | 4 |
| r01 | 112.10708563 | `76bdd1764c69` | 3 |
| r07 | 112.10708563 | `96583726909a` | 3 |
| r09 | 118.17164869 | `11595201a379` | 2 |
| r03 | 118.17177707 | `62079071694a` | 2 |
| r04 | 118.17177707 | `62079071694a` | 2 |
| r06 | 118.17177707 | `11595201a379` | 2 |
| r08 | 118.17177707 | `11595201a379` | 2 |

## Architecture similarity

Both budgets leave the **chiplet count unsettled** and explore 2, 3, and 4
chiplets. The shared core is thin: every chiplet is 7 nm, the package is `3d`
with `ucie_3d`, memory is `hbm3`, dataflow is `os`, and the tiling/split/merge/
data-sharing flags are zero.

At 150,000 moves:

| Field | Modal value | Agreement | Distribution |
|---|---|---:|---|
| `n_chiplets` | `4` / `2` | 40% | 4=4; 2=4; 3=2 |
| `chip1.sys_array_size` | `96x96` | 80% | 96x96=8; 128x128=1; 64x64=1 |
| `chip1.sram_buf` | `512` | 80% | 512=8; 1024=1; 256=1 |
| `pkg.HI_pkg_type` | `3d` | 90% | 3d=9; 2.5d_3d=1 |
| `wl.assign_workload_in_ascending_order` | `1` | 90% | 1=9; 0=1 |

No single architecture dominates. Even the tied clusters are not
architecturally identical: the `109.3975` trio mixes `0ac1f0a55e7b` (r07, r08)
with `3a39d0d59ea6` (r02), and the `117.6757` cluster mixes `11595201a379`
(r03, r09, r10) with `62079071694a` (r04).

### Feature distance

| Set | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.356 | 0.500 | 0.479 |
| `150000` | 0.541 | 0.719 | 0.503 |

## Budget effect (calibration caveat)

The 150,000-move rerun changed two inputs at once: the calibration grew from
10,000 to 15,000 samples **and** the move budget grew. Because the objective is
normalized by calibration statistics, the two sets' objective values are on
different scales and are not directly comparable. This is easy to see: the same
architecture `3a39d0d59ea6` scores `110.210733` at 100k (10k calibration) and
`109.397584` at 150k (15k calibration).

Re-evaluating both best architectures under one shared (15,000-sample)
calibration isolates the budget effect:

| Architecture | Normalized objective | Latency (ns) | Energy (pJ) |
|---|---:|---:|---:|
| 100k best (`3a39d0d59ea6`) | 109.397584 | 84,538.9 | 1.840e9 |
| 150k best (`cafd3fbe2a2a`) | 107.327362 | 133,916.2 | 1.902e9 |

So the extra budget did find a genuinely better design (~1.9% lower normalized
objective), despite higher latency and energy, by trading against area and
dollar cost. But it did **not** improve convergence: the 150k best is a unique
outlier, and the run population is more architecturally diverse than at 100k.

## Cross-workload comparison

| Workload | Best | Relative spread | Distinct fingerprints | Best found | Meets 80%-tied bar? |
|---|---:|---:|---:|---:|---|
| 1 (150k) | 107.327362 | 11.80% | 7 | 1/10 | no |
| 1 (100k) | 110.210733 | 7.22% | 6 | 2/10 | no |
| 6 | 36.791 | 1.50% | 6 | 3/10 exact | near (90% within 0.1%) |
| 7 | 22.875 | 0.00% | 4 | 10/10 | yes |
| 10 | 35.573 | 21.76% | 6 | 8/10 | yes for score, memory-only variation |

Only the small chains (workloads 7 and 10) reach the "same score, minor
differences" regime. The large single GEMM (workload 1) does not, at either
budget, because its optimum depends on chiplet count, which the search leaves
unsettled.

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Tests: `tests/test_architecture_similarity.py`
- Generated report (local, `reports/` is git-ignored):
  `reports/architecture_similarity/workload1/report.md`
- Per-run tables: `runs_100000.csv`, `runs_150000.csv`, plus distance and
  field-agreement CSVs for each set
- Run outputs (on the server):
  `cfg/gen_arch/wl1_1iteration_workload1_100k_convergence_r*_t1/` and
  `cfg/gen_arch/wl1_1iteration_workload1_150k_convergence_r*_t1/`

## Reproducing on another workload

Replace `<N>` with the workload index. Run from the repository root on the
server with the virtual environment activated.

### 1. Calibration (once per workload)

```bash
nohup bash -lc '
  cd /absolute/path/to/CarbonPATH &&
  source carbonpath/bin/activate &&
  make calibration WORKLOAD=<N> CALIBRATION_ITERATIONS=15000 \
    RUN_NAME=workload<N>_calibration_15000 SEED=$((11000 + <N>))
' > logs/workload<N>_calibration_15000.log 2>&1 < /dev/null &
```

Wait for it to finish, then confirm `cfg/calibration/calibration_<N>.json`
contains `"_calibration_samples": 15000`.

### 2. Ten parallel runs at the chosen schedule

```bash
for i in $(seq -w 1 10); do
  cache="cfg/static_cache/cache_w<N>_150k_r${i}.csv"
  cp -f cfg/static_cache/static_cache.csv "$cache"
  nohup bash -lc "
    cd /absolute/path/to/CarbonPATH &&
    source carbonpath/bin/activate &&
    make sim_anneal WORKLOAD=<N> ITERATION=1 CALIBRATION_ITERATIONS=15000 \
      INITIAL_TEMP=4000 FREEZING_TEMP=3.24e-10 MAX_MOVE_PER_TEMP_STEP=50 COOLING_RATE=0.99 \
      RUN_NAME=workload<N>_150k_convergence_r${i} SEED=$((17000 + 10#${i} - 1)) \
      CACHE_FILE=${cache}
  " > logs/workload<N>_150k_convergence_r${i}.log 2>&1 < /dev/null &
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
  --set 150000="cfg/gen_arch/wl<N>_1iteration_workload<N>_150k_convergence_r*_t1" \
  --output "reports/architecture_similarity/workload<N>"
```

Add more `--set LABEL=PATTERN` arguments to compare budgets side by side, for
example `--set 100000="cfg/gen_arch/wl<N>_1iteration_workload<N>_100k_convergence_r*_t1"`.
