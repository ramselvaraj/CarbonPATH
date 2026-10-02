# Remote modular-equivalence campaign

This is the final statistical acceptance test for modular CarbonPATH. Run it
only on the remote compute server. The commands below are not intended for a
developer laptop.

## Acceptance rule

For each workload:

- run original CarbonPATH 20 times with 20 distinct initial seeds and 20
  distinct search seeds;
- run modular CarbonPATH 20 times with the same workload and annealing schedule,
  again with 20 distinct initial seeds and 20 distinct search seeds;
- find the final result returned most often by original CarbonPATH;
- require at least 16 of the 20 modular runs to return either:
  - that same physical architecture and effective GEMM mapping; or
  - that same physical architecture with only GEMM mapping differences and a
    score within 0.1% of the dominant original score.

The comparison includes SA count and properties, memory, package topology,
protocol, and transfer hardware. It reads modular GEMM mapping from the saved
evaluation profile, where the new flow owns those settings.

## 1. Create two clean checkouts on the remote server

Use the last pre-modular campaign commit as the original reference. Replace
`<MODULAR_COMMIT>` with the committed revision containing the completed modular
implementation.

```bash
git worktree add ../carbonpath-original 41c059b63217b53369866e0142570978ae72e2bd
git worktree add ../carbonpath-modular <MODULAR_COMMIT>
```

Before starting, verify that both worktrees are clean and create/use the normal
project Python environment in each checkout.

```bash
git -C ../carbonpath-original status --short
git -C ../carbonpath-modular status --short
```

Choose a result directory outside both worktrees so results survive checkout
changes:

```bash
export CARBONPATH_RESULTS="$PWD/../carbonpath-equivalence-results"
mkdir -p "$CARBONPATH_RESULTS"
```

## 2. Run the original cohort

From `../carbonpath-original`:

```bash
python -m script.run_baseline_campaign prepare \
  --output-root "$CARBONPATH_RESULTS/original" \
  --workloads 7 9 10 \
  --runs 20 \
  --calibration-samples 10

python -m script.run_baseline_campaign start \
  --output-root "$CARBONPATH_RESULTS/original" \
  --workloads 7 9 10 \
  --runs 20 \
  --max-workers 4
```

`--max-workers` may be increased or reduced for the remote machine. It does not
change the per-run search.

Validate before moving on:

```bash
python -m script.run_baseline_campaign validate \
  --output-root "$CARBONPATH_RESULTS/original"
```

If interrupted, rerun the `start` command as `resume`; completed valid runs are
not repeated.

## 3. Run the modular cohort

From `../carbonpath-modular`, use the same workload list, schedule, run count,
and calibration sample count:

```bash
python -m script.run_baseline_campaign prepare \
  --output-root "$CARBONPATH_RESULTS/modular" \
  --workloads 7 9 10 \
  --runs 20 \
  --calibration-samples 10

python -m script.run_baseline_campaign start \
  --output-root "$CARBONPATH_RESULTS/modular" \
  --workloads 7 9 10 \
  --runs 20 \
  --max-workers 4

python -m script.run_baseline_campaign validate \
  --output-root "$CARBONPATH_RESULTS/modular"
```

Each modular run must contain both `best_architecture.json` and
`best_profile.json`. Campaign validation rejects incomplete or mismatched
profile artifacts.

## 4. Apply the equivalence gate

From `../carbonpath-modular`:

```bash
python -m script.validate_modular_equivalence_campaign \
  --original-root "$CARBONPATH_RESULTS/original" \
  --modular-root "$CARBONPATH_RESULTS/modular" \
  --output "$CARBONPATH_RESULTS/equivalence-report"
```

The command exits with status 0 only when every workload passes the 16-of-20
rule. It writes:

- `equivalence_report.json` for machine-readable evidence;
- `equivalence_runs.csv` for per-run inspection;
- `equivalence_report.md` for the concise pass/fail summary.

The validator also rejects mismatched workload definitions, mismatched
annealing schedules, missing artifacts, non-finite scores, and cohorts that do
not contain independently seeded runs.
