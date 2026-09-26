# Annealing Search over the Modular Evaluator

This document shows the intended relationship between two layers:

- **Inner** — `evaluate_atlas`: one design point (one fixed architecture and one
  fixed profile), evaluated once.
- **Outer** — `search`: the future simulated-annealing loop that proposes design
  points and calls the inner evaluator.

The ATLAS graph (the workload) is fixed for the whole search.

## Inner: one design point

```text
evaluate_atlas(graph, architecture, profile):
    system  = build_hardware(architecture)               # SA + FPGA + routes
    policy  = resolve(profile.placement_policy, system)
    mover   = resolve(profile.movement_policy, profile.transfer_model)
    models  = resolve(profile.evaluators)                # input adapter + evaluator

    residency = none
    for operation in graph.operations:
        access      = dram_access(operation, position)     # DRAM in/out
        endpoint    = policy.endpoint_for(operation)       # SA or FPGA
        movement    = mover.move(residency -> endpoint) if residency else none
        model_input = models.input_adapter(operation, access)
        estimate    = models.evaluator(model_input, endpoint)
        record(operation, endpoint, movement, estimate)
        residency   = endpoint_of(operation)

    return aggregate(records, system)                     # latency + energy
```

## Outer: simulated annealing search

```text
search(graph, initial_architecture, initial_profile):
    graph stays fixed for the whole search

    current      = (initial_architecture, initial_profile)
    current_cost = objective(evaluate_atlas(graph, *current))
    best, best_cost = current, current_cost

    temperature = initial_temp
    while temperature > freezing_temp:
        for move in range(moves_per_step):
            candidate = mutate(current)                   # architecture and/or profile
            if candidate is invalid:
                continue
            cost = objective(evaluate_atlas(graph, *candidate))
            if accept(cost, current_cost, temperature):
                current, current_cost = candidate, cost
                if cost < best_cost:
                    best, best_cost = candidate, cost
        temperature *= cooling_rate

    return best, best_cost
```

This outer loop is implemented inside `main.sim_annealing`: the ATLAS branch
replaces `objective(evaluate_atlas(...))` with
`evaluate_atlas_design_point` plus a replaceable `AtlasObjective`, and
`mutate(current)` with `mutate_atlas_design_point`. The legacy branch is
unchanged.

## The two swap points the search plays with

```text
mutate(current):
    architecture, profile = current
    pick one:
        architecture knob  -> chiplet count, size, tech node, SRAM,
                              package, protocol, WL mapping, FPGA resources
        profile knob       -> placement policy, movement policy,
                              transfer model, evaluator per operation type
    return (new_architecture, new_profile)

objective(evaluation):
    return latency / energy / cost / carbon from the evaluation
```

## Flow

```text
search
  -> propose (architecture, profile)
  -> evaluate_atlas (one design point)
  -> objective
  -> accept / reject
  -> repeat until temperature freezes
```

## Current state

The ATLAS branch is implemented. `sim_annealing` now calls
`evaluate_atlas_design_point` and `mutate_atlas_design_point` when given an
`atlas_graph`, and keeps the legacy path when given a `workload_sequence`.

The objective is a replaceable model. The default `raw_weighted_sum_v0` is
uncalibrated and reads coefficients from `cfg/parameters/atlas_objective.json`;
it does not use calibration averages. Replacing it with a calibrated objective
is a new `AtlasObjective` subclass plus one registry entry, with no change to the
outer loop.

A placeholder policy catalog supplies deliberately bad movement, transfer, and
GEMM policies so `script/validate_atlas_policy_scores.py` and
`tests/test_atlas_annealing.py` can check that a known bad policy produces a bad
score and is not selected. Those placeholder files are isolated for deletion.

