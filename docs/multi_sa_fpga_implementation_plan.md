# Multi-SA + FPGA Workloads: Implementation Plan

**Status:** implemented and locally verified (2026-10-02)  
**Scope:** alternating GEMM/ReLU ATLAS graphs, one to six SAs, one FPGA  
**Primary workload:** `cfg/examples/atlas/dense_relu_funnel.graph_dump.json`

## Outcome

Allow modular annealing to add or remove SAs while evaluating GEMM/ReLU graphs
without rejecting SA-group-to-FPGA tensor movement.

The annealer already changes the SA count. The GEMM evaluator already maps
tiles across every SA in the candidate architecture. The missing behavior is a
movement policy that knows:

1. which SA owns each GEMM output region;
2. which input regions every following GEMM needs;
3. when an activation must be replicated to more than one SA; and
4. how to charge every routed transfer exactly once.

The selected policy will be named:

```text
distributed_activation_boundary_v1
```

The existing `activation_boundary_v1` policy remains unchanged so the current
single-SA result stays reproducible.

## Fixed V1 decisions

These are implementation decisions, not open questions:

- The architecture may contain one to six SAs and exactly one FPGA.
- GEMMs use `all_sas_single_fpga_v1`; every available SA is eligible.
- The existing scheduler decides the GEMM tiling and SA assignment.
- The existing SCALE-Sim evaluator remains the source of tile cycle counts.
- ReLU uses the existing `legacy_fpga_relu_v1` evaluator.
- Each tensor element is one byte, matching the current transfer model.
- Operations execute in graph order.
- Compute and tensor movement do not overlap.
- Transfers sharing a physical link serialize.
- Transfers whose routes share no links may execute concurrently.
- Movement energy is always the sum of every physical transfer's energy.
- Split-K output is owned by the existing final reduction owner after reduction.
- The first input still comes from DRAM and the final output still returns to
  DRAM.
- V1 supports GEMM -> ReLU, ReLU -> GEMM, and same-FPGA handoffs. Other mixed
  operation sequences fail with a clear `UnsupportedEvaluation`.

## Locked decisions (2026-10-02)

These were pinned in a design interview before implementation:

- A **physical link** is the undirected edge `tuple(sorted((a, b)))`, matching
  `Scheduler.__form_links`. Opposite-direction traffic on one edge serializes.
- `ChipletSystem.link_cost(a, b)` returns one directed physical link's
  `(reciprocal bandwidth ns/byte, energy pJ/bit)`. A missing edge raises
  `UnsupportedEvaluation`; it never returns zero or `None`.
- `DistributedMovementPlan.charged_latency_ns` is the maximum slice completion
  estimate (zero when retained); `charged_energy_pj` is the sum of slice
  energies; `logical_bytes` is the tensor size; `physical_bytes` is the sum of
  slice bytes (replication included).
- The executor always calls `binding.prepare` for every operation and passes the
  artifact through `context.prepared_operation`; the
  `requires_prepared_operations` branching is gone from the executor.
- `distributed_activation_boundary_v1` accepts only `gemm->relu`, `relu->gemm`,
  and same-FPGA `relu->relu`; it rejects `gemm->gemm`, Softmax, and every other
  pair. `gemm->gemm` remains the mapped GEMM adapters' responsibility.
- `GenericAnnealer.anneal` catches `UnsupportedEvaluation` from a candidate
  evaluation, records an `evaluation_error` row with current/best unchanged,
  and consumes the attempt. `_run_modular_annealing` skips `architecture_rows`
  when the candidate evaluation is `None`.
- ATLAS `boundaries.csv` holds one row per routed slice (with a per-slice
  `slice_bytes` column); handoff totals repeat on every row. `layers.csv` is
  unchanged and no aggregate file is added.
- The one-SA gate compares the `GOLDEN` summary at relative tolerance `1e-9`
  and every per-handoff latency, energy, and physical-byte value, ignoring
  profile-identity fields.
- The reference fixture is
  `tests/fixtures/dense_relu_funnel_one_sa_pointwise.json`; `GOLDEN` remains the
  fast assertion.
- The search space's `profiles` catalog points at the new profile. The plan's
  earlier claim that the search space needs no modification was wrong.

## Design

### Movement seam

Replace the executor's two movement call shapes with one interface:

```python
movement_policy.plan(request) -> MovementPlan
```

`MovementRequest` contains only the facts needed to plan one adjacent graph
handoff:

```text
handoff index
tensor ID and tensor shape
producer operation and assignment
consumer operation and assignment
producer prepared evaluator artifact
consumer prepared evaluator artifact
chiplet system and transfer-model settings
```

The executor does not inspect GEMM tiles or branch on GEMM/ReLU. That
implementation stays behind the movement seam.

The three real adapters at this seam will be:

- `activation_boundary_v1`: current single-endpoint behavior;
- the existing mapped GEMM intermediate-memory adapters;
- `distributed_activation_boundary_v1`: new SA-group/FPGA behavior.

This gives the seam leverage: the executor learns one interface while each
adapter owns its movement rules.

### Movement result

`DistributedMovementPlan` will expose aggregate values used by the executor:

```text
method
source endpoint or endpoint group
destination endpoint or endpoint group
logical tensor bytes
physical transferred bytes
charged latency
charged energy
transfer slices
```

Each `RoutedTensorSlice` will record:

```text
tensor ID
2-D region: row offset/count and feature offset/count
source endpoint
destination endpoint
byte count
resolved route
setup latency
hop latency
serialization latency
slice latency
slice energy
```

Logical bytes describe the tensor once. Physical bytes include replication and
may therefore be larger than logical bytes.

## Mapping rules

### GEMM output ownership

For a GEMM tile:

```text
output rows     = [m_offset, m_offset + m)
output features = [n_offset, n_offset + n)
```

Without split-K, the tile's assigned SA owns that output region.

With split-K, all K shards for one output region are reduced first. The
completed output region is owned by the scheduler's existing reduction owner.
Only the completed region is transferred to the FPGA.

The policy must prove that the unique output regions cover the complete
`M x N` tensor exactly once. Missing coverage or conflicting ownership is an
unsupported evaluation.

### GEMM input requirements

For a consumer GEMM tile:

```text
input rows     = [m_offset, m_offset + m)
input features = [k_offset, k_offset + k)
```

Input regions repeated by several output-N tiles on the same SA are counted
once for that SA. The same input region needed by two different SAs is sent
twice and charged twice.

This replication rule is essential. For example, two SAs producing different
output columns can both require the entire ReLU activation.

### GEMM -> ReLU

1. Build the unique output-region ownership map from the prepared GEMM
   scheduler.
2. Create one routed transfer for each owned region from its SA to the FPGA.
3. Verify that the slices cover the complete ReLU input.
4. Aggregate link occupancy, latency, and energy.

### ReLU -> GEMM

1. Treat the completed ReLU tensor as resident on the FPGA.
2. Build the unique input requirements for each consumer SA.
3. Create one routed transfer per destination-SA region.
4. Preserve replication across different destination SAs.
5. Aggregate link occupancy, latency, and energy.

### FPGA -> FPGA

Retain the tensor locally with zero movement latency and zero movement energy.

## Transfer calculations

`ChipletSystem.link_cost(a, b)` returns each directed physical link's cost; a
physical link is keyed by the undirected edge `tuple(sorted((a, b)))` so
opposite-direction traffic on one edge serializes.

For each routed slice:

```text
bytes = row_count * feature_count

slice energy
  = bytes * 8 * sum(link energy in pJ/bit along the route)

fixed route latency
  = setup latency + hop_count * hop latency
```

For a movement phase, calculate the serialization load on every physical link:

```text
link load
  = sum(slice bytes * link reciprocal bandwidth)
    for every slice using that link
```

For each slice, its completion estimate is its fixed route latency plus the
link loads on its route. The movement phase latency is the maximum slice
completion estimate. This makes disjoint routes parallel and shared links
serialized.

All intermediate values must be retained in the movement result so the final
number can be audited.

## One candidate evaluation inside annealing

For every proposed architecture/profile pair:

1. Validate the architecture and package connectivity.
2. Build the SA endpoints, FPGA endpoint, memory channels, and routes.
3. Place every GEMM on the complete SA group and every ReLU on the FPGA.
4. Prepare every GEMM mapping:
   - tile M/N;
   - optionally tile K;
   - assign tiles to SAs according to compute capacity;
   - run/cache SCALE-Sim for each assigned tile.
5. Plan every adjacent tensor handoff with the selected movement policy.
6. Evaluate every GEMM:
   - DRAM activation reads when required;
   - DRAM weight reads;
   - SCALE-Sim cycles divided by SA frequency;
   - maximum SA completion time;
   - split-K transfer and reduction when enabled;
   - final-output DRAM write when required.
7. Evaluate every ReLU:
   - lanes from FPGA resources;
   - `cycles = ceil(elements / lanes)`;
   - `latency = cycles / FPGA frequency`;
   - `dynamic energy = elements * energy_per_element_pj` when characterized.
8. Aggregate:
   - total latency = operation latency + incoming movement latency in graph
     order;
   - dynamic energy = GEMM + ReLU + movement energy;
   - baseline energy = full-system power * total latency;
   - area, cost, embodied carbon, and operational carbon.
9. Calculate the selected objective score.
10. Apply the annealing acceptance rule and record the proposal.

The SA count is an architecture choice made by annealing. The movement policy
does not choose the number of SAs; it correctly uses and prices whichever SAs
exist in the candidate.

## Files

### Production and configuration: 7 files

1. `system/utils/ChipletSystem.py`
   - add `link_cost(a, b)` for one directed physical link, raising
     `UnsupportedEvaluation` on a missing edge.

2. `system/utils/TensorMovement.py`
   - add `MovementRequest`, `RoutedTensorSlice`, and
     `DistributedMovementPlan`;
   - add `distributed_activation_boundary_v1` and its registry entry;
   - adapt existing movement implementations to the common `plan` interface;
   - implement region coverage, replication, route resolution, and link-load
     aggregation behind the interface.

3. `main.py`
   - always prepare operation mappings before movement planning;
   - construct one `MovementRequest` per adjacent operation pair;
   - remove movement-policy-specific branching from the executor;
   - continue passing the prepared GEMM artifact into evaluation.

4. `network.py`
   - populate ATLAS `boundaries.csv` with one row per routed slice;
   - include logical bytes, physical bytes, routes, latency, energy,
     per-slice bytes, and handoff totals;
   - preserve current layer and summary totals.

5. `system/utils/GenericAnnealer.py`
   - treat `UnsupportedEvaluation` from a candidate evaluation as an invalid
     proposal instead of terminating the complete search;
   - keep initial-design evaluation failures fatal.

6. `cfg/profiles/atlas_multi_sa_fpga_v1.json`
   - reuse the current GEMM/ReLU/Softmax evaluators and assignment policy;
   - select `distributed_activation_boundary_v1`.

7. `cfg/experiments/atlas_modular_search_space.json`
   - point the `profiles` catalog at the new profile so the trace records the
     profile actually used.

8. `system/utils/Scheduler.py`
   - return zero interconnect latency when a split-K design has no inter-SA link
     to its reduction owner (single SA), instead of crashing on an empty `max()`.

### Tests: 6 files

9. `tests/test_tensor_movement.py` (new)
   - pure policy tests for coverage, ownership, replication, split-K,
     shared/disjoint links, and aggregation.

10. `tests/test_atlas_graph_evaluation.py`
    - one-SA equivalence against the frozen fixture;
    - two-SA GEMM/ReLU execution;
    - exact per-handoff and graph totals;
    - detailed routed-slice reporting.

11. `tests/test_atlas_annealing.py`
    - SA-count candidates evaluate successfully;
    - trace contains one-SA and multi-SA candidates;
    - objective and identity remain deterministic.

12. `tests/test_generic_annealer.py`
    - candidate evaluation failures become invalid trace rows;
    - current/best state is unchanged after an invalid candidate.

13. `tests/fixtures/dense_relu_funnel_one_sa_pointwise.json` (new)
    - fingerprints, summary metrics, and every reference handoff.

14. `tests/test_chiplet_system.py`
    - single-SA split-K reports zero interconnect latency.

Excluding this document, the complete change touches **14 files**: eight
production/configuration files and six test-side files. Including this
document, it touches 15.

One defect fix was added beyond the original plan: a single-SA candidate with
split-K enabled reached
`Scheduler.__calculate_interconnect_latency_energy_with_separate_controller`
with no inter-SA link to the reduction owner, so `max()` on an empty list raised
`ValueError` and killed the campaign. That path now returns zero interconnect
latency when there are no links to accumulate, with a regression test in
`tests/test_chiplet_system.py`. No other changes are planned for `Scheduler.py`,
`Simulator.py`, `OperationAssignment.py`, `AtlasAnnealingMoves.py`, or the
SCALE-Sim internals.

## Implementation order

### 1. Freeze the one-SA reference

- Record the current dense-ReLU-funnel layer totals and graph totals.
- Record every current SA -> FPGA and FPGA -> SA transfer.
- Add an equality test before changing the executor.

**Gate:** the reference fixture is reproducible.

### 2. Add the common movement interface

- Add the request/result types.
- Adapt existing movement implementations.
- Change the executor to call `plan(request)` for every adjacent pair.

**Gate:** the existing one-SA profile remains pointwise equal.

### 3. Implement distributed region mapping

- Derive GEMM output ownership from prepared scheduler tiles.
- Derive per-SA GEMM input requirements.
- Deduplicate regions within one endpoint.
- Preserve replication across endpoints.
- Handle split-K final ownership.

**Gate:** pure mapping tests account for every logical and physical byte.

### 4. Implement routed batch costing

- Resolve every slice route.
- Calculate link occupancy, critical movement latency, and total energy.
- Return all intermediate calculations in the plan.

**Gate:** hand-calculated shared-link and disjoint-link fixtures match exactly.

### 5. Add graph integration and reporting

- Evaluate the alternating GEMM/ReLU graph with two and three SAs.
- Populate `boundaries.csv` with routed slices.
- Verify layer totals and graph totals are additive.

**Gate:** every reported aggregate can be reconstructed from its slice rows.

### 6. Enable multi-SA annealing

- Use `atlas_multi_sa_fpga_v1.json`.
- Use `atlas_modular_search_space.json`, not the `_single` search space.
- Record invalid candidate evaluations without stopping the run.

**Gate:** a deterministic smoke run visits at least two SA counts and finishes.

### 7. Remote verification

- Run the full dense-ReLU-funnel annealing job on the remote machine.
- Re-evaluate the winning architecture without annealing.
- Export architecture, profile, layer table, movement-slice table, mapping
  table, summary, trace, and exact command/revision metadata.

**Gate:** the winning architecture re-evaluates to the same objective and all
reported totals tally.

## Required invariants

- Every non-replicated producer tensor region has exactly one owner.
- GEMM -> FPGA unique slice bytes equal the logical tensor bytes.
- FPGA -> GEMM physical bytes equal the sum of unique requirements per
  destination SA and may exceed logical tensor bytes.
- No region is silently dropped.
- No same-SA duplicate input region is charged twice.
- Cross-SA replication is never deduplicated.
- Each physical transfer is charged once for latency accounting and once for
  energy accounting.
- Movement totals equal the aggregation of reported slice rows.
- Graph totals equal the aggregation of operation and movement rows.
- A one-SA evaluation using the new policy matches the current policy.
- Candidate evaluation failure cannot change the annealer's current or best
  design.

## Verification commands

Targeted tests:

```bash
python -m unittest \
  tests.test_tensor_movement \
  tests.test_atlas_graph_evaluation \
  tests.test_atlas_annealing \
  tests.test_generic_annealer -v
```

Full suite:

```bash
python -m unittest discover -s tests -v
```

The remote smoke run must use the new profile and the existing multi-SA search
space. A full annealing campaign starts only after the targeted and full test
suites pass and the one-SA equality fixture is unchanged.

## Explicitly deferred

- Multiple FPGAs.
- Branching or residual graphs.
- Streaming/pipelined overlap between transfer and compute.
- Choosing only a subset of installed SAs for one GEMM.
- Tensor sizes derived from ATLAS precision strings instead of one byte per
  element.
- A new FPGA ReLU characterization.
- Changes to SCALE-Sim internals.

