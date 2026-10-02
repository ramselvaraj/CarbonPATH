# Legacy-to-modular equivalence experiment

This is the operational runbook for proving that modular CarbonPATH preserves
the result of original CarbonPATH when the same sequential-GEMM computation is
supplied as a Keras/HGQ2 network.

Run annealing only on the remote compute server.

## Experiment contract

Compare exactly two cohorts for each workload:

```text
10 Legacy runs
Legacy workload -> Legacy evaluator -> Legacy annealer

10 modular runs
Keras/HGQ2 network -> ATLAS graph -> modular evaluator -> modular annealer
```

Runs 1--10 use the same initial seeds, search seeds, annealing schedule, search
space, and calibration size in both cohorts. The final comparison is across the
two independently started cohorts; run 1 is not required to match run 1.

A modular run is equivalent to a result produced by the Legacy cohort when:

- the physical SA, SRAM, memory, and package architecture is the same after
  canonical chiplet relabeling;
- the objective differs by no more than 0.1%; and
- any remaining differences are GEMM mapping decisions: dataflow, split-K, or
  assignment order.

Memory-channel allocation and interconnect topology are physical differences.
A better score on different hardware is interesting, but it is not a parity
match. At least 8 of the 10 modular runs must match a Legacy result class.

Adapter-generated ATLAS graphs are diagnostic inputs. Only a graph produced
from the real Keras/HGQ2 frontend can be used for the modular acceptance cohort.

## Fixed configuration

```text
Legacy commit:        f997887ca447a1e3ae28008143945c5283455dd5
Modular branch:       modular-carbonpath
Runs per cohort:      10
Initial seeds:        12000..12009
Search seeds:         13000..13009
Calibration seed:     10000 + workload ID
Calibration samples:  10
Initial temperature:  4000
Freezing temperature: 0.001
Moves per level:      50
Cooling rate:         0.99
Moves per run:        75650
Movement policy:      direct_forward
Objective:            calibrated t1
```

## 1. Verify one Legacy run is reproducible

Before changing modular code, rerun one saved Legacy case using its exact
commit, saved calibration, saved `initial_architecture.json`, search seed, and
schedule. Require the new search trace, best score, and best architecture to
match the saved run.

The baseline currently lives at:

```text
~/work/CarbonPATH/reports/fixed_sa_baseline_w7_w9_w10_v2
```

If it must be reconstructed, create a clean Legacy worktree from the fixed
commit and use `script.run_baseline_campaign` with `--runs 10`.

Completion criterion: one exact Legacy replay passes. If it does not, stop and
diagnose baseline reproducibility before testing modular CarbonPATH.

## 2. Generate the graph from the real Keras network

Keras sources committed in CarbonPATH:

```text
cfg/examples/atlas/workload7_two_gemm.keras.py
cfg/examples/atlas/workload9_single_gemm.keras.py
```

Copy the selected source into the ATLAS checkout so its upward search finds
`run_atlas_flow.py`, then run it with the ATLAS Keras/HGQ2 environment:

```bash
cp "$CARBONPATH/cfg/examples/atlas/workload9_single_gemm.keras.py" \
   "$ATLAS/example/carbonpath_workload9_single_gemm.py"

cd "$ATLAS"
.venv/bin/python example/carbonpath_workload9_single_gemm.py
```

This performs graph-only conversion:

```text
dump_graph=True
stop_after_convert=True
```

It does not run synthesis or EDA. For workload 9, the artifact is:

```text
$ATLAS/outputs/carbonpath_workload9_single_gemm/graph_dump.json
```

Completion criterion: CarbonPATH parses exactly the expected operation types,
dependencies, and GEMM dimensions. A mismatch stops the experiment.

## 3. Prepare exactly 10 modular runs

Use a clean modular worktree and a new output directory:

```bash
cd "$CARBONPATH_MODULAR"

python -m script.run_baseline_campaign prepare \
  --output-root "$RESULTS/modular-workload9" \
  --workloads 9 \
  --runs 10 \
  --calibration-samples 10 \
  --atlas-graph "9=$ATLAS/outputs/carbonpath_workload9_single_gemm/graph_dump.json"
```

Preparation copies and hashes the graph, checks that its evaluated operation
sequence matches workload 9, and generates calibration through that Keras graph.

Inspect `manifest.json` and require:

```text
input_flow: keras_atlas
runs_per_workload: 10
workloads: [9]
atlas_graphs.9.sha256: present
```

Completion criterion: the manifest binds the cohort to the expected graph,
commit, seeds, schedule, profile, search space, and calibration.

## 4. Launch, monitor, and validate

```bash
python -m script.run_baseline_campaign start \
  --output-root "$RESULTS/modular-workload9" \
  --workloads 9 \
  --runs 10 \
  --max-workers 4
```

Resume an interrupted controller without repeating valid completed runs:

```bash
python -m script.run_baseline_campaign resume \
  --output-root "$RESULTS/modular-workload9" \
  --workloads 9 \
  --runs 10 \
  --max-workers 4
```

Check progress:

```bash
python -m script.run_baseline_campaign status \
  --output-root "$RESULTS/modular-workload9"
```

Validate all saved artifacts:

```bash
python -m script.run_baseline_campaign validate \
  --output-root "$RESULTS/modular-workload9"
```

Completion criterion: exactly 10 of 10 runs validate. Stop and compare them;
additional runs require an explicit decision.

## 5. Apply the 8-of-10 equivalence gate

The comparison must consider every result class actually produced by the 10
Legacy runs. A modular result may match any Legacy class satisfying the physical
architecture and 0.1%-score rules above.

The current `script.validate_modular_equivalence_campaign` instead selects one
dominant complete Legacy fingerprint. That behavior does not implement this
contract when Legacy itself produces several equivalent final architectures.

**Required next implementation step:** update the validator to:

1. select one workload from larger campaign manifests;
2. compare exactly the first 10 independently seeded runs;
3. build the set of Legacy-equivalent result classes;
4. count each modular run that matches any class;
5. require at least 8 matches; and
6. write per-run field differences for every failure.

Do not declare a workload passed using the existing dominant-only gate.

## Current remote evidence

### Legacy cohort

```text
~/work/CarbonPATH/reports/fixed_sa_baseline_w7_w9_w10_v2
```

Contains 10 Legacy runs each for workloads 7, 9, and 10 at commit `f997887`.

### Workload 7 modular-adapter diagnostic

```text
~/work/carbonpath-workload7-2819ef0
```

The first 10 modular runs all have the Legacy objective. Eight have the same
physical architecture as a Legacy result with only GEMM mapping differences.
This is useful annealer evidence, but it is not the Keras acceptance cohort.

Detailed field comparison:

```text
~/work/carbonpath-workload7-2819ef0/architecture-vs-legacy-first10/report.md
```

### Workload 9 Keras-modular diagnostic

```text
~/work/carbonpath-keras-workload9-0703beb
```

All 20 runs used the real Keras-derived graph and modular flow. Only runs 1--10
belong to the matched experiment; runs 11--20 are extra robustness data.

Among runs 1--10, three are clear physical-plus-mapping matches and seven have a
score produced by Legacy. The cohort does not meet the 8-of-10 gate.

Detailed field comparison:

```text
~/work/carbonpath-keras-workload9-0703beb/architecture-vs-legacy-first10/report.md
```

## Next session

Start by reading this file. Then:

1. implement the corrected validator described in step 5;
2. replay one Legacy workload-9 run exactly;
3. build a short Legacy-versus-modular trace comparison from the same saved
   start and search seed;
4. locate and fix the first meaningful annealing divergence without calling the
   Legacy search as the modular implementation;
5. run short regression checks; and
6. only after those pass, launch exactly 10 fresh workload-9 Keras runs.

Workload 7 Keras runs come after workload 9 passes. Workload 10 follows workload
7. No other annealing cohort should be launched before the current workload's
equivalence report is reviewed.
