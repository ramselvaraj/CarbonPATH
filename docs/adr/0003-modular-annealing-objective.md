# 3. Modular annealing objective and isolated placeholder policies

Date: 2026-09-25

## Status

Accepted

## Context

CarbonPATH's modular ATLAS evaluator (`main.evaluate_atlas_graph`) is a fixed
single-point estimator with no scalar cost, while the legacy annealer scores
calibration-normalized metrics from the legacy GEMM path. Enabling simulated
annealing over the modular flow requires a scalar objective for a modular design
point, a mutation strategy for `(architecture, evaluation profile)`, and a way
to test that the search penalizes a known bad policy.

The modular placement policy requires exactly one systolic-array endpoint and one
FPGA endpoint, so the legacy chiplet-count mutations are not usable. The legacy
objective also depends on a calibration pass that the modular path does not have.

## Decision

1. The modular objective is a replaceable model behind
   `system/utils/AtlasObjective.py`, selected by configuration, exactly like the
   existing evaluator, placement, movement, and transfer registries. The default
   `raw_weighted_sum_v0` is **uncalibrated**: configuration coefficients applied
   to raw design-point metrics. A calibrated objective can replace it as a new
   subclass plus one registry entry, with no change to `sim_annealing`.
2. Modular moves mutate the two fixed endpoints and the evaluation profile, never
   adding or deleting chiplets, and validate the one-SA/one-FPGA invariant.
3. Deliberately bad placeholder policies (movement, transfer, GEMM evaluator)
   live in isolated files (`system/utils/PlaceholderPolicies.py`,
   `cfg/profiles/placeholders/`, and a placeholder check script) so tests can
   prove a known bad policy produces a bad score and is never the annealer's
   best result. These files are intended to be deleted as a unit later.

## Consequences

- Modular annealing can run today without a calibration pass, and the objective
  can be upgraded without touching the search loop.
- The default raw weighted sum is provisional and is not a physically calibrated
  cost; it must not be reported as such.
- The placeholder catalog is a temporary testing seam. Removing it means deleting
  one module, one config directory, one registration call, and the
  placeholder-dependent tests.
