# Workload 10 Convergence and Architecture Similarity

This documents the workload 10 annealing budget study and its architecture
similarity analysis. It mirrors `docs/workload6_convergence_report.md` and
`docs/workload7_convergence_report.md`; workload 1 results are in
`docs/workload1_convergence_report.md`. See
[Reproducing on another workload](#reproducing-on-another-workload).

> Note: workload 10's head was widened from `N=16` to `N=256` for this study
> (shapes `[128, 256, 512]` and `[128, 512, 256]`). This changes the workload
> definition, so the workload-10 rows of the earlier
> `reports/fixed_sa_baseline_w7_w9_w10_v2` campaign no longer match the current
> config, and `cfg/calibration/calibration_10.json` was regenerated.

## Question

Does the annealing search reach a single optimum and architecture when the
second GEMM of the chain carries real weight?

## Setup

- Workload: `10` = `projection_head_demo`, chained
  `projection [128, 256, 512]` + `head [128, 512, 256]`
  (33,554,432 MACs total; the head alone is 16,777,216).
- Intermediate policy: `direct_forward`; cost profile: `t1`
  (energy + latency + area + dollar, zero carbon weight).
- Calibration: `10,000` samples, generated once and reused by every annealing
  run (zero stale regenerations across all ten runs).
- Runs: 10 independent runs, seeds `17000`–`17009`, launched in parallel with
  `nohup`, each with its own simulation cache file.
- Schedule: `100,000` moves per run — `INITIAL_TEMP=4000`,
  `FREEZING_TEMP=7.5e-6`, `COOLING_RATE=0.99`, `MAX_MOVE_PER_TEMP_STEP=50`.
- Metric: normalized objective (`best_cost_after`), lower is better.

## Results

| Schedule | Best | Worst | Relative spread | Distinct canonical fingerprints | Largest fingerprint group |
|---|---:|---:|---:|---:|---:|
| `100000` | 35.573191 | 43.312609 | 21.76% | 6 | 3 |

Eight of ten runs reach the same best objective (`35.573191`); two lag behind.
Unlike workload 7, the larger head leaves the search only **partially
converged**.

### Per-run results

| Run | Objective | Canonical fingerprint | Distance to best |
|---|---:|---|---:|
| r02 | 35.57319104 | `f7d202306585` | 0.000 |
| r03 | 35.57319104 | `f7d202306585` | 0.000 |
| r04 | 35.57319104 | `0ddfdccbd4f2` | 0.048 |
| r05 | 35.57319104 | `5983441962aa` | 0.095 |
| r07 | 35.57319104 | `f7d202306585` | 0.000 |
| r08 | 35.57319104 | `5983441962aa` | 0.095 |
| r09 | 35.57319104 | `0ddfdccbd4f2` | 0.048 |
| r10 | 35.57319104 | `58abb45f8049` | 0.095 |
| r06 | 37.68557295 | `a42c493b95ba` | 0.190 |
| r01 | 43.31260862 | `29abe46d960a` | 0.333 |

### Exact architecture groups

| Group size | Runs |
|---:|---|
| 3 | r02, r03, r07 (`f7d202306585`) |
| 2 | r04, r09 (`0ddfdccbd4f2`) |
| 2 | r05, r08 (`5983441962aa`) |
| 1 | r01 (`29abe46d960a`) |
| 1 | r06 (`a42c493b95ba`) |
| 1 | r10 (`58abb45f8049`) |

## Architecture similarity

The design core still favours the small 3D stack:

- 2 chiplets, tech node `7`, systolic array `64x64`
- chiplet 1 SRAM `256` KB
- memory `hbm3`
- dataflow `os`; `static_tiling`, `merge_tiles`, `if_splitting_k`,
  `chiplet_data_sharing_enabled` all zero

But more fields vary than for workloads 6 and 7:

| Field | Modal value | Agreement | Distribution |
|---|---|---:|---|
| `chip2.sram_buf` | `256` | 90% | 256=9; 512=1 |
| `pkg.HI_pkg_type` | `3d` | 90% | 3d=9; 2.5d=1 |
| `pkg.protocol_3d` | `ucie_3d` | 90% | ucie_3d=9; na=1 |
| `pkg.protocol_2.5d` | `na` | 90% | na=9; aib=1 |
| `pkg.conn1` | `Chiplet_1->Chiplet_2:3d_hyb_bond@stack0_base` | 90% | 3d_hyb_bond=9; 2.5d_emib=1 |
| `pkg.conn2` | `Chiplet_2->na:3d_hyb_bond@stack0_top` | 90% | 3d_hyb_bond=9; 2.5d_emib=1 |
| `wl.assign_workload_in_ascending_order` | `0` | 70% | 0=7; 1=3 |
| `pkg.mem_chip1` | `8` | 50% | 8=5; 7=2; 9=2; 10=1 |
| `pkg.mem_chip2` | `8` | 50% | 8=5; 9=2; 7=2; 6=1 |

The single `2.5d` run (r01) also switches to the `aib` 2.5D protocol and a
`2.5d_emib` connection—a structurally different package, not just a parameter
tweak—and it is the worst run (`43.313`).

### Feature distance

Normalized Hamming distance over the union of canonical architecture fields,
measured from the best run:

| Schedule | Mean distance to best | Max distance to best | Spearman (distance vs objective) |
|---|---:|---:|---:|
| `100000` | 0.090 | 0.333 | 0.915 |

The strong positive correlation (0.915) is meaningful here: farther designs
really do score worse, unlike the near-tie artifact seen for workload 7.

### Interpretation

Widening the head to `N=256` makes the chain substantially harder to converge.
The search settles on the same 2-chiplet, 7 nm, `64x64`, HBM3, `os` family and
finds the best objective in 8 of 10 runs, but the remaining knobs (second
chiplet's SRAM, HBM channel counts, mapping flag) stay undecided, and one run
escapes to a `2.5d`/`aib` package that costs ~22%. The best objective
(`35.573`) is higher than workload 7's (`22.875`) and workload 6's (`36.791`),
but objectives are workload-specific and not comparable across workloads.

## Cross-workload comparison

| Workload | Schedule | Best | Relative spread | Distinct fingerprints | Largest group | Convergence |
|---|---:|---:|---:|---:|---:|---|
| 6 | 50 | 55.987 | 865.24% | 10 | 1 | none |
| 6 | 75,650 | 36.791 | 10.53% | 6 | 2 | partial |
| 6 | 100,000 | 36.791 | 1.50% | 6 | 3 | near |
| 7 | 100,000 | 22.875 | 0.00% | 4 | 5 | full |
| 10 (N=256) | 100,000 | 35.573 | 21.76% | 6 | 3 | partial |

The pattern across workloads: convergence quality depends on the workload, even
at a fixed 100,000-move budget. The two-GEMM chain with a narrow head
(workload 7) converges completely; the same chain with a wide head (workload 10)
and the single-GEMM workload 6 do not.

## Artifacts

- Analyzer: `script/analyze_architecture_similarity.py`
- Tests: `tests/test_architecture_similarity.py`
- Generated report (local, `reports/` is git-ignored):
  `reports/architecture_similarity/workload10/report.md`
- Per-run tables: `runs_100000.csv`, `distance_matrix_100000.csv`,
  `field_agreement_100000.csv`
- Run outputs (on the server):
  `cfg/gen_arch/wl10_1iteration_workload10_100k_convergence_r*_t1/`
- Workload definition: `cfg/examples/workload.json` (workload 10)

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
