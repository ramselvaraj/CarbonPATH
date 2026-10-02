# Modular CarbonPATH Implementation Plan

## Implementation status — 2026-10-01

Completed locally so far:

- canonical, label-independent architecture identity;
- architecture validation and `ChipletSystem` construction for one through six
  heterogeneous SAs, with optional FPGA endpoints and connected routes;
- explicit SA endpoint groups and all-SA GEMM placement;
- evaluator-owned dataflow, split-K, and assignment-order settings;
- multi-SA tile coverage, per-tile result reporting, and explicit split-K
  reduction metrics;
- `cold_dram`, `ideal_on_chip`, `local_sram`, and `direct_forward` as modular
  movement-policy implementations;
- original `t1` through `t4` scoring and all normalization modes behind the
  modular objective interface;
- multi-SA value mutations and evaluator-setting mutations without unchanged
  proposals;
- SA-count mutation with atomic memory/link regeneration;
- 2D, 2.5D, 3D, and hybrid 2.5D+3D package construction with compatible
  protocol regeneration;
- a domain-independent annealing loop, now used by modular ATLAS searches,
  with fixed or adaptive cooling, invalid/no-op handling, current-vs-best
  tracking, deterministic tie-breaking, and complete attempt accounting.
- explicit per-family mutation weights in every tracked modular search space,
  with validation and public-seam reachability coverage for every move family;
- an identity-pinned original CarbonPATH two-GEMM fixture and live differential
  test covering all four intermediate-memory behaviors;
- pointwise equality for latency, combined dynamic energy, tile mappings,
  boundary decisions, and DRAM-use behavior; this gate found and removed a
  duplicate modular DRAM-boundary charge;
- a frozen two-SA split-K fixture proving the modular evaluator reproduces the
  original mapping, forwarding decision, latency, and dynamic energy;
- exhaustive scoring of an eight-design reduced space, with identical raw
  metrics, t1 scores, complete ranking, and best design in both evaluators;
- exhaustive neighborhood comparison across 4,512 reduced valid designs,
  including every original mapping and hardware move family; this gate restored
  the original one-SA memory, package-topology, and protocol-regeneration
  constraints for sequential-GEMM searches;
- a deterministic sequential-GEMM-to-ATLAS adapter, so the original workload
  input now enters the normal modular graph representation;
- modular calibration sampling using the same design-point evaluator as the
  search, rather than the original top-level workload evaluator;
- the normal sequential-workload annealing entry point now assembles a modular
  graph, evaluation profile, original-domain move weights, calibrated
  objective, and the generic annealer. Raw ATLAS graphs use that same annealer;
- the unreachable original combined annealing-loop body has been removed from
  the production entry point;
- the standalone calibration and intermediate-policy comparison commands also
  evaluate sequential GEMMs through the modular graph/design-point path;
- the production modular search configuration again contains every original
  package, memory, and protocol option alongside the newer FPGA and transfer
  options;
- the remote campaign workflow now persists and validates the complete best
  design (architecture plus evaluation profile), pins its starting profile,
  rejects dirty modular checkouts, and provides a machine-checked equivalence
  report;
- 249 local tests pass. No full annealing cohort was run locally.

Still required before compatibility can be claimed:

- execute the documented original and modular 20-run cohorts on the remote
  server;
- apply the equivalence gate and confirm at least 16 of 20 modular runs match
  the dominant original result or an equivalent-score, physically equivalent
  architecture.

No production annealing campaign has been run on the local laptop.

## Outcome

CarbonPATH will have one production flow:

```text
Keras/HGQ2 network
    → ATLAS graph
    → modular design point
    → modular evaluation
    → objective
    → generic simulated annealing
    → best architecture and evaluation profile
```

An equivalent sequential-GEMM workload, SA-only search definition, evaluation
configuration, calibration, and objective must reproduce original CarbonPATH's
final-result behavior. FPGA evaluation, heterogeneous graphs, replaceable
models, and explicit transfer modeling remain available through the same flow.

There must not be an `if all_gemm: run_original_pipeline()` production branch.
Validated low-level engines are reused behind modular interfaces; the original
top-level evaluator and annealing path are not reused by production code.

## Settled design decisions

1. **One modular executor.** Every supported graph uses the same placement,
   operation-evaluation, movement, transfer, aggregation, objective, and search
   flow.
2. **GEMM mapping stays inside the GEMM evaluator.** Dataflow, split-K, and
   assignment-order settings are evaluator configuration, not a new top-level
   policy.
3. **No catch-all system model.** Physical responsibilities remain distributed:
   endpoint characterization owns SA/FPGA area and power; movement owns tensor
   residency and spilling; transfer owns route latency and energy; design-point
   metric calculation owns package cost and carbon; the objective combines the
   metrics.
4. **Placement selects an endpoint or SA group.** The first complete SA
   placement implementation assigns GEMMs to all eligible SAs. The GEMM
   evaluator maps tiles within the selected group.
5. **Existing low-level mathematics are retained.** `Scheduler`, SCALE-Sim,
   `SimulationCache`, intermediate-memory calculations, package generation,
   and physical/cost/carbon helpers are refactored behind modular interfaces.
6. **Architecture inputs are canonicalized.** Hardware choices are
   authoritative; derived area, power, routes, bandwidth, and package values
   are regenerated before evaluation.
7. **Calibration belongs to objective configuration.** It does not become an
   evaluation-profile policy.
8. **Compatibility constrains the search definition, not the production
   architecture.** The sequential-GEMM comparison fixes the ordinary modular
   implementations and searches the original SA-only domain. Other runs may
   include FPGA endpoints and alternative implementations.
9. **Search trajectories need not match.** The feasible designs, scores, and
   final-result behavior must match; proposal order and acceptance traces may
   differ.

## Execution policy: laptop versus remote server

The local laptop is for implementation and fast feedback only. Real simulated-
annealing experiments run on the remote compute server.

### Allowed locally

- edit and review code and documentation;
- run format, import, schema, and static checks;
- run unit tests for architecture handling, profiles, placement, movement,
  transfer, objectives, mutations, and annealer state transitions;
- run fixture-backed point comparisons that do not invoke new SCALE-Sim work;
- run mocked or deliberately tiny deterministic search tests whose purpose is
  control-flow correctness rather than hardware optimization;
- prepare and validate campaign manifests without starting their jobs.

### Remote only

- calibration generation beyond tiny test fixtures;
- any real SCALE-Sim-backed annealing run;
- the original 20-run reference cohorts;
- reduced-space enumeration when it invokes real simulation;
- cold-cache and warm-cache performance campaigns;
- the modular 20-run final-result cohorts;
- full workload/objective campaigns and cutover validation.

No default local test command may launch a real annealing campaign. Remote-only
work is exposed through explicit campaign commands, not ordinary unit-test
discovery.

### Remote campaign interface

Reuse the existing `script/run_baseline_campaign.py` workflow and conventions.
The modular equivalence campaign runner must provide the same operational
commands:

```text
prepare → start/resume → status → validate → report
```

The runner must:

- freeze the git commit, workload/graph fingerprints, search space, evaluation
  profile, objective, calibration, simulator model, schedule, and seed panel in
  a manifest before jobs start;
- reject a dirty or mismatched remote checkout unless its exact patch identity
  is explicitly captured;
- assign each run its own initial-architecture seed, search seed, cache, output
  directory, and log;
- allow bounded remote parallelism without sharing mutable run caches;
- resume interrupted campaigns without rerunning valid completed jobs;
- re-evaluate every reported best design and validate artifact hashes before a
  result is admitted to the cohort;
- produce a machine-readable report containing exact-result rate,
  mapping-only-equivalent rate, failures, runtimes, and simulator calls.

### Artifact handoff

1. Prepare the manifest locally or on the remote checkout without starting the
   campaign.
2. Run a small remote smoke job and validate its artifacts.
3. Run a remote pilot before launching all 20 independent jobs.
4. Launch or resume the full remote cohort with an explicit worker limit.
5. Run `validate` and `report` remotely.
6. Copy back only the manifest, validated results, reports, and selected
   immutable fixtures. Do not copy credentials or machine-specific paths into
   the repository.

Connection details remain in `docs/server_connection.local.md`. Before the
first modular remote run, extend that local-only runbook with the actual remote
repository path, environment activation command, worker limit, storage path,
and modular campaign commands. Do not commit passwords, keys, host secrets, or
other credentials.

## Required functionality

The implementation is incomplete until the modular flow supports all of the
following original capabilities:

- one through six SA chiplets;
- SA array size, technology node, and SRAM choices;
- split-K, `ws`/`os`/`is` dataflow, and assignment ordering;
- multi-SA tiling, scheduling, execution, and reductions;
- 2D, 2.5D, and 3D package generation and validity constraints;
- supported memory, interconnect, and protocol choices;
- `cold_dram`, `ideal_on_chip`, `local_sram`, and `direct_forward` intermediate
  behavior;
- architecture power, area, dollar cost, embodied carbon, and operational
  carbon;
- existing calibration modes and calibration identity rules;
- `t1` through `t4` objectives;
- original architecture-generation and mutation families;
- simulator/cache model identity and cold-/warm-cache correctness.

The following modular additions must remain supported:

- FPGA endpoints and FPGA resource/frequency/implementation changes;
- mixed GEMM, ReLU, and Softmax graphs;
- heterogeneous operation placement;
- explicit tensor residency and movement;
- replaceable operation evaluators, placement policies, movement policies,
  transfer cost models, and objectives;
- profile selection and profile identity in results.

## Target module ownership

### Graph adapter

Accepts the supplied Keras/HGQ2-derived ATLAS graph and returns immutable graph
operations with shapes, tensors, element counts, and dependencies. It never
contains hardware or annealing decisions.

### Architecture and endpoint characterization

Accepts architecture JSON, normalizes endpoint identities and ordering,
validates package constraints, regenerates derived fields, and builds SA/FPGA
endpoints and routes. It supports old architecture JSON through a
backwards-compatible adapter.

### Evaluation profile

Selects:

- operation evaluators;
- operation placement policy;
- tensor movement policy;
- transfer cost model.

Evaluator-specific settings are nested under the selected evaluator. For
example:

```yaml
evaluators:
  gemm:
    id: carbonpath_gemm_v1
    settings:
      dataflow: ws
      split_k: true
      assignment_order: ascending
```

### Placement

Assigns each operation to one endpoint or endpoint group. It does not tile a
GEMM. The complete SA placement implementation returns the group of all
eligible SAs; future implementations may choose subsets without changing the
executor.

### GEMM evaluator

Given a GEMM, selected SA group, access plan, and evaluator settings, it:

1. creates the GEMM workload;
2. applies dataflow, split-K, and assignment-order settings;
3. partitions and maps tiles across the selected SAs;
4. invokes the existing simulation/cache engine;
5. handles reductions and local memory effects;
6. returns latency, dynamic energy, and the resolved tile mapping.

### Tensor movement

Tracks tensor residency and chooses retention, forwarding, replication, or
DRAM spill. The four original intermediate behaviors become ordinary movement
implementations using the validated intermediate-memory calculations.

### Transfer cost model

Receives an already resolved movement path and payload, then returns movement
latency and energy using the selected memory, topology, link, and protocol
models. It does not choose operation placement or GEMM mapping.

### Design-point metric calculation and objective

Metric calculation aggregates compute and movement results and derives power,
area, cost, and carbon with the existing validated helpers. The objective then
applies its selected weights, normalization mode, and calibration artifact.

### Annealer

Operates only on a modular design point. It receives proposal and evaluation
dependencies rather than branching on workload type. It owns temperature,
acceptance, current/best tracking, logging, and stopping. Mutation-family
probabilities are explicit search configuration.

## Implementation phases

### Phase 0 — Freeze reference evidence

1. Select the existing sequential-GEMM workloads used by the compatibility
   campaign and create equivalent Keras/HGQ2 inputs.
2. Assert that their ATLAS graphs preserve the GEMM sequence and dimensions.
3. Capture immutable point-evaluation fixtures from original CarbonPATH,
   including:
   - architecture and mapping;
   - per-GEMM tile mappings;
   - boundary movement decisions;
   - raw and normalized metrics;
   - final objective;
   - workload, search-space, calibration, simulator, and cache fingerprints.
4. On the remote server, run the original full search 20 independent times per
   campaign workload and objective configuration. Record every seed,
   independently generated starting architecture, final result, and score.
5. Keep a live original-vs-modular differential harness during migration. It is
   test-only and cannot be imported by the modular production executor.
6. Add the remote campaign manifest and runbook fields needed by the execution
   policy above. Do not run the full campaign on the laptop.

**Exit gate:** reproducible fixtures and reference cohorts exist before
behavior is moved.

### Phase 1 — Canonical architecture and multi-endpoint support

1. Separate authoritative hardware choices from derived characterization.
2. Add deterministic architecture canonicalization:
   - normalize chiplet identifiers and ordering;
   - retain every evaluation-affecting field;
   - omit only log-only/generated noise;
   - provide a stable fingerprint.
3. Generalize architecture validation and `ChipletSystem` construction from one
   SA/one FPGA to arbitrary supported SA and FPGA endpoint collections.
4. Add SA endpoint-group representation.
5. Reuse package generation and validation for topology, stacking, channel, and
   protocol constraints.
6. Recompute area, power, SRAM properties, link bandwidth, and routes whenever
   authoritative hardware choices change.

**Tests:** old-JSON loading, canonical round trips, identifier-order
independence, one-to-six SAs, heterogeneous SA properties, package validity,
derived-field refresh, and invalid-combination rejection.

**Exit gate:** every original SA architecture can be represented, rebuilt, and
fingerprinted without requiring the original top-level flow.

### Phase 2 — Complete modular GEMM evaluation

1. Extend the evaluation-profile schema so evaluator selections may carry
   validated nested settings while preserving existing string-based profiles.
2. Make placement return the complete eligible SA group for GEMM.
3. Move construction of `Scheduler` inputs behind the GEMM evaluator interface.
4. Feed dataflow, split-K, assignment order, and multi-SA topology into the
   existing scheduler and SCALE-Sim/cache engines.
5. Return per-operation latency, energy, reductions, and tile mappings as
   structured evaluation output.
6. Remove assumptions that the GEMM endpoint is one SA.

**Tests:** every mapping setting, one and multiple SAs, split-K reductions,
tile coverage without duplication, supported dataflows, cold/warm cache, and
pointwise comparison with original GEMM evaluation.

**Exit gate:** a fixed sequential-GEMM graph and fixed architecture produce the
same tile decisions and compute metrics through the modular evaluator.

### Phase 3 — Complete movement and transfer behavior

1. Adapt the existing intermediate-memory calculations behind the modular
   movement interface.
2. Implement all four original intermediate behaviors as selectable movement
   implementations.
3. Represent SRAM-capacity failure, incompatible producer/consumer mappings,
   route availability, forwarding, and DRAM fallback explicitly.
4. Ensure the transfer model prices resolved routes using the same bandwidth
   and energy calculations as the original flow.
5. Preserve first-input DRAM reads and final-output DRAM writes.

**Tests:** every movement behavior, same-SA retention, cross-SA forwarding,
SRAM capacity edges, mapping mismatch fallback, DRAM traffic, route validation,
and per-boundary latency/energy parity.

**Exit gate:** complete sequential-GEMM latency, communication energy, and SRAM
energy match for every original intermediate behavior at fixed design points.

### Phase 4 — Physical metrics, calibration, and objectives

1. Centralize raw design-point metric production without moving calculations
   into the objective.
2. Reuse the existing area, power, cost, embodied-carbon, and
   operational-carbon helpers.
3. Add objective implementations for `t1` through `t4`.
4. Support all existing normalization modes.
5. Load or generate calibration through objective configuration using the same
   identity inputs and post-calibration RNG reset rules.
6. Retain the modular raw weighted objective as an ordinary alternative for
   experiments that request it.

**Tests:** physical-metric parity, calibration identity and staleness,
normalization formulas, every objective profile, deterministic calibration,
and complete pointwise raw/normalized score comparisons.

**Exit gate:** the modular path gives the same score and ranking for every
fixed compatibility design point.

### Phase 5 — Restore the complete modular search space

1. Add modular mutation implementations for:
   - SA count;
   - SA array size;
   - technology node;
   - SRAM capacity;
   - GEMM evaluator mapping settings;
   - memory technology;
   - package/interconnect topology;
   - communication protocol.
2. Retain FPGA, transfer-model, and profile changes for general modular runs.
3. Regenerate and canonicalize the architecture after every hardware change.
4. Make mutation-family probabilities explicit configuration. Do not require
   original proposal probabilities for compatibility.
5. Reject invalid or unchanged proposals consistently.
6. Add deterministic tie-breaking over canonical design identity when scores
   are numerically equal.

**Tests:** every move changes only its owned settings, inputs remain immutable,
all allowed values are reachable, invalid moves are rejected, package
constraints survive mutation, and reduced-space neighbor sets are complete.

**Exit gate:** the modular compatibility search reaches the same feasible
design set as original CarbonPATH.

### Phase 6 — Collapse onto one generic annealing flow

1. Make the annealer consume only:
   - an initial modular design point;
   - a proposal implementation;
   - a design-point evaluator;
   - an objective;
   - the annealing schedule and random source.
2. Remove workload-type branching from candidate evaluation and mutation.
3. Preserve acceptance behavior, cooling/adaptive control, current-vs-best
   tracking, attempt accounting, and reproducible logging.
4. Return the complete best result: architecture, evaluation profile and
   evaluator settings, raw metrics, objective value, and trace.
5. Route Keras/HGQ2-derived graphs and directly supplied ATLAS graphs through
   the same executor.

**Tests:** deterministic replay for a fixed start/seed, current-vs-best
semantics, acceptance/cooling behavior, unsupported-evaluation handling, and
complete result serialization.

**Exit gate:** production evaluation and search no longer require the original
top-level branch.

### Phase 7 — Prove compatibility and retain modular additions

Run all acceptance layers below. A failure blocks cutover.

#### A. Pointwise equivalence

For representative workloads, canonical architectures, mapping settings,
movement behaviors, normalization modes, and objectives:

- canonical discrete decisions must match exactly;
- raw metrics, normalized metrics, and objective values must match with relative
  tolerance `1e-9`;
- cold-cache and warm-cache results must agree;
- live differential results and immutable fixtures must both agree.

Any relaxed numerical tolerance must be metric-specific, justified, and
recorded.

#### B. Feasible-space and neighborhood equivalence

Construct a reduced finite search space with at least two values for every
original search dimension. Enumerate:

- every canonical valid design;
- every design's valid one-move neighbors.

Compare sets rather than random proposal order.

#### C. Exhaustive optimum equivalence

Score every design in the reduced space through both evaluators. Assert:

- equal ordering after numerical tie handling;
- the same globally optimal canonical design class;
- the same objective value.

#### D. Independent-run final-result campaign

This campaign is remote-only.

For each selected workload and objective configuration:

1. Run original CarbonPATH 20 times from independently generated starting
   architectures and independent seeds.
2. Run modular CarbonPATH 20 times under the same workload and search
   definition, also with independent starts and seeds.
3. Determine the dominant original final-result class.
4. Require at least 16 of 20 modular runs to return either:
   - that same physical architecture and effective GEMM mapping; or
   - the same physical architecture and an equivalent score, with differences
     limited to GEMM mapping.

The remote procedure and machine-checked gate are documented in
`docs/modular_equivalence_remote_runbook.md`. The default equivalent-score
tolerance is 0.1% and is recorded in the generated report.
5. A remaining run is acceptable only if its physical architecture matches the
   dominant class, its score matches within the declared tolerance, and its
   differences are limited to GEMM mapping choices.
6. Record all starts, seeds, results, mappings, scores, and failure reasons.

This is a cohort comparison, not same-seed trace comparison.

#### E. Modular-addition regression suite

Retain or add tests for:

- GEMM → ReLU → GEMM and GEMM → Softmax graphs;
- SA-to-FPGA and FPGA-to-SA tensor movement;
- FPGA resources, frequency, and operation-design changes;
- alternative evaluator, placement, movement, transfer, and objective
  implementations;
- profile identity and replacement;
- deliberately different profiles producing predictably different results;
- fixed-seed deterministic modular annealing.

#### F. Test execution tiers

- **Per-commit CI:** focused unit tests, representative pointwise parity, reduced
  exhaustive search, and modular regressions.
- **Remote nightly/manual campaign:** all existing sequential workloads,
  objectives, and 20-run independent cohorts using the real simulator. These
  commands are never part of the laptop's default test suite.

### Phase 8 — Cut over and remove the duplicate production path

1. Switch library and CLI entry points to the single modular flow.
2. Run the complete CI locally, then run the independent-result campaign on the
   remote server from both a clean cache and a warm cache.
3. Remove the original top-level production evaluator/annealing branch.
4. Retain reused low-level engines, immutable expected fixtures, and the modular
   compatibility tests.
5. Remove the temporary live differential harness after fixtures and final
   campaign results are verified.
6. Update architecture/profile examples, user documentation, and the modular
   annealing diagram to match the shipped implementation.

## Suggested implementation order and blocking edges

```text
Reference evidence
    ↓
Canonical architecture and endpoint groups
    ↓
Complete GEMM evaluator
    ↓
Movement and transfer
    ↓
Physical metrics, calibration, objectives
    ↓
Complete mutation/search space
    ↓
Single generic annealer
    ↓
Full compatibility and modular-regression gates
    ↓
Production cutover and duplicate-path removal
```

Pointwise tests are added with each phase rather than postponed to Phase 7.
Phase 7 assembles those tests into the final proof and runs the statistical
campaign.

## Definition of done

The work is complete only when:

- every item in the required-functionality checklist has a modular owner and a
  passing test;
- no production code dispatches an all-GEMM graph to the original top-level
  pipeline;
- fixed-design pointwise metrics and scores meet the strict parity tolerance;
- the reduced exhaustive search has the same optimal canonical design;
- at least 16 of 20 independent modular runs per campaign workload/configuration
  reproduce the dominant original final-result class;
- remaining accepted outcomes differ only in GEMM mapping while retaining the
  same physical architecture and equivalent score;
- all heterogeneous/FPGA/profile/transfer regression tests pass;
- CLI and library entry points use the single modular executor;
- the duplicate original production path is removed and immutable reference
  evidence remains.
