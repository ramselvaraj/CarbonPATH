# Modular CarbonPATH Overview

This document describes the modular ATLAS evaluation path: what it evaluates,
what its inputs are, how it is wired, and where simulated annealing currently
stands relative to it.

## 1. It estimates one fixed architecture and one fixed profile

The modular evaluator is a single-point estimator. Given one graph, one
architecture, and one profile, it produces one result.

```text
one ATLAS graph (fixed)
      x
one architecture (fixed)
      x
one profile (fixed)
      -> one evaluation -> one result
```

Within a run, nothing varies:

```text
endpoints        fixed   SA(0), FPGA(1)
resources        fixed   CLBs, BRAMs, DSPs, frequency
routes           fixed   SA(0) <-> FPGA(1)
placement rules  fixed
models           fixed   per operation type
```

The loop only walks the graph's operations. It does not search, mutate, or
compare alternatives. Variation across runs comes only from changing an input:

```text
different architecture JSON -> new run
different profile JSON      -> new run
different ATLAS graph       -> new run
```

## 2. Inputs

```text
atlas_path    raw ATLAS graph JSON        the workload (operations to execute)
architecture  hardware description JSON   the hardware (endpoints, resources, routes)
profile       evaluation profile JSON     the model and policy selection
cache         simulation cache            internal runtime dependency (GEMM results)
```

Architecture describes what hardware exists:

```text
Chiplet_1 : SA   (sys_array_size, sram_buf, area, power)
Chiplet_2 : FPGA (clbs, brams, dsps, frequency, relu_implementation)
pkg       : inter-chiplet connections
transfer_model : setup/hop latency
```

Profile describes which models and policies run on it:

```text
evaluators       : {gemm, relu, softmax} -> model IDs
placement_policy : ID
movement_policy  : ID
transfer_model   : ID
```

## 3. Policies

Each policy is named in the profile and resolves to one implementation.

- `placement_policy` — decides which endpoint an operation runs on
  (`fixed_single_sa_single_fpga_v1`: gemm to SA, relu/softmax to FPGA).
- `movement_policy` — decides how a produced tensor reaches its consumer
  (`activation_boundary_v1`: retain if the same endpoint, else resolve a route).
- `transfer_model` — computes the latency and energy of one movement over a
  resolved route (`route_transfer_v1`).

What they look like in the profile:

```json
{
  "placement_policy": "fixed_single_sa_single_fpga_v1",
  "movement_policy": "activation_boundary_v1",
  "transfer_model": "route_transfer_v1",
  "evaluators": {
    "gemm": "legacy_scale_sim_gemm_v1",
    "relu": "placeholder_hls_relu_v0",
    "softmax": "placeholder_fpga_softmax_v0"
  }
}
```

What they look like in code:

```text
placement_policy -> FixedSingleSaSingleFpgaPlacement.endpoint_for(type)
movement_policy  -> TensorMovementService.move(...)
transfer_model   -> TransferEstimator.estimate(request)
```

## 4. Models

- **Evaluator** — the replaceable model that estimates one operation's compute
  latency and dynamic energy; it never reads the graph or raw JSON.
- **Input adapter** — the evaluator's partner that turns an operation (plus the
  access plan) into the exact typed input that evaluator needs, reading the
  ATLAS source view only when required.
- **Evaluator binding** — the registered pair of one evaluator and its input
  adapter, looked up by the profile's evaluator ID.

```text
evaluator      -> LegacyScaleSimGemmEvaluator / FpgaReluEvaluator / PlaceholderSoftmaxEvaluator
input adapter  -> LegacyScaleSimGemmInputAdapter / PlaceholderSoftmaxInputAdapter
binding        -> EvaluatorBinding(input_adapter, evaluator)
```

Example model inputs:

```text
GEMM    : M, K, N, activation_from_dram, output_to_dram
ReLU    : element_count  (placeholder HLS adds reuse_factor)
Softmax : element_count, axis
```

Every evaluator returns the same contract:

```text
compute_latency_ns
dynamic_energy_pj
```

## 5. Pseudocode

```text
evaluate_atlas(atlas_path, architecture, profile):
    graph   = parse_atlas_graph(read_json(atlas_path))   # ATLAS JSON -> operations
    system  = build_hardware(architecture)               # SA + FPGA + routes
    policy  = resolve(profile.placement_policy, system)
    mover   = resolve(profile.movement_policy, profile.transfer_model)
    models  = resolve(profile.evaluators)                # evaluator + input adapter

    residency = none
    for operation in graph.operations:
        access      = dram_access(operation, position)     # DRAM in/out
        endpoint    = policy.endpoint_for(operation)       # SA or FPGA
        movement    = mover.move(residency -> endpoint)    # inter-chiplet

        model_input = models.input_adapter(operation, access)
        estimate    = models.evaluator(model_input, endpoint)

        record(operation, endpoint, movement, estimate)
        residency = endpoint_of(operation)

    return aggregate(records, system)
```

Stage summary:

```text
parse     : ATLAS JSON        -> graph of operations
resolve   : profile names     -> policies + models
place     : operation         -> SA or FPGA
move      : producer endpoint -> consumer endpoint
adapt     : operation + DRAM  -> model-specific input
estimate  : model input       -> latency + energy
aggregate : per-op results    -> totals + reports
```

## 6. Simulated annealing over the modular evaluator

`main.sim_annealing` now has an ATLAS branch. When `atlas_graph` is supplied it
skips calibration and the legacy GEMM sequence, and calls the modular evaluator
for every proposal:

```text
candidate architecture + candidate profile
        |
        v
   evaluate_atlas_design_point()          # evaluate_atlas_graph + metrics
        |
        v
   AtlasObjective.score(design_point)     # replaceable objective model
        |
        v
   accept_move_func(cost_diff, temperature)
        |
        +-- accept -> current = candidate
        +-- reject -> keep current
```

Where it searches:

```text
architecture knobs   SA sys_array, tech node, SRAM; FPGA CLBs, BRAMs, DSPs,
                     frequency, relu_implementation; package protocol/memory;
                     transfer model
profile knobs        evaluator per operation type, placement policy,
                     movement policy, transfer cost model
```

Because the modular placement policy requires exactly one SA and one FPGA, the
modular moves mutate the two fixed endpoints and never add or delete chiplets;
`validate_atlas_architecture` enforces the invariant.

What stays fixed:

```text
the ATLAS graph (the workload) and its order
```

The objective is a replaceable model. The default
`raw_weighted_sum_v0` is uncalibrated: it applies configuration coefficients to
raw design-point metrics (`cfg/parameters/atlas_objective.json`). The legacy
calibration path is not used for ATLAS. A calibrated objective is a new
`AtlasObjective` subclass plus one registry entry and no change to
`sim_annealing`.

A placeholder policy catalog under `cfg/profiles/placeholders/` and
`system/utils/PlaceholderPolicies.py` supplies deliberately bad policies so a
check can prove the search scores a known bad policy badly and does not select
it. Those files are isolated for later deletion.

Entry point:

```bash
python -m main --run_mode run_sim_anneal \
  --atlas_graph cfg/examples/atlas/dense_relu_funnel.graph_dump.json \
  --architecture_file cfg/examples/sa_fpga_architecture.json \
  --evaluation_profile cfg/profiles/legacy_sa_fpga_v1.json \
  --atlas_search_space cfg/experiments/atlas_modular_search_space.json
```

`sim_annealing` returns the same four values as before; in ATLAS mode the trace
DataFrame carries `attrs["best_profile"]`, `attrs["best_profile_fingerprint"]`,
and `attrs["candidate_profiles"]`.

