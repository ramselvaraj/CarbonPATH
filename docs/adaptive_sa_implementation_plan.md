# Adaptive Simulated-Annealing Implementation Plan

## Goal

Add a feedback-controlled simulated-annealing mode and an unattended convergence campaign. The experiment must pursue both the lowest verified objective score and repeatability: at least 8 of 10 formal runs must return one exact canonical architecture.

This is an extension to PATH-AI. The paper uses a fixed schedule (`4000`, `0.001`, cooling `0.99`, 50 moves per level) and does not specify adaptive control.

## Experiment Contract

- Workload: 7 initially; workload 9 only after workload 7 reaches a terminal result.
- Cost profile: `t1`.
- Intermediate policy: `direct_forward`.
- Fixed move budget: 75,650 attempted moves.
- Feedback block: 100 attempted moves.
- Initial target uphill acceptance: 80%.
- Final target uphill acceptance: 5%.
- Minimum temperature: `1e-3`.
- Temperature multiplier bounds: `0.5` to `1.5`.
- Fallback cooling: `0.98`.
- Stagnation threshold: 5,000 attempted moves.
- Maximum reheats: 2, each at most a 2x temperature increase.
- Reheating is disabled after 80% of the move budget.
- Objective equality epsilon: `1e-12`.
- Quality tolerance: 1% relative to the best verified score.

The initial temperature is derived from the median positive `cost_diff` values in the model-version-3 workload-7 reference traces:

```text
T0 = -median_positive_cost_diff / ln(0.80)
```

The current reference gives a median delta of approximately `2.737`, or an initial temperature near `12.27`.

## Adaptive Algorithm

After every 100 attempted moves, calculate the target uphill acceptance using an exponential curve from 80% to 5%:

```text
target = 0.80 * (0.05 / 0.80) ** progress
```

Count only valid, changed, strictly uphill proposals. Exclude invalid proposals, unchanged proposals, improving proposals, and equal-cost proposals. Use Jeffreys smoothing:

```text
observed = (accepted_uphill + 0.5) / (uphill_proposals + 1)
```

When at least five uphill proposals exist:

```text
multiplier = clamp(exp(target - observed), 0.5, 1.5)
next_temperature = clamp(current_temperature * multiplier, min_temperature, max_temperature)
```

With insufficient evidence, apply fallback cooling. Record target, observed rate, evidence, multiplier, temperature, action, stagnation, and reheat count in `temperature_trace.csv`.

Reheat only when there has been no strict best-score improvement for 5,000 moves, progress is below 80%, and fewer than two reheats have occurred. Adaptive runs terminate at exactly 75,650 moves. Fixed-mode behavior must remain unchanged.

Equal-cost candidates remain subject to the existing best-architecture behavior. This experiment tests whether adaptive search improves exact convergence; it does not add a tie-breaker.

## Code Structure

Add:

- `system/utils/AnnealingSchedule.py`: pure configuration, statistics, temperature decisions, and validation. No Pandas, simulation, filesystem, or randomness.
- `script/run_adaptive_sa_convergence.py`: frozen calibration, process workers, private caches, state machine, gates, status, watch, resume, and reports.
- `tests/test_annealing_schedule.py`.
- `tests/test_simulated_annealing_hooks.py`.
- `tests/test_adaptive_sa_convergence.py`.

Modify:

- `main.py`: optional adaptive controller, level callback, move budget, common move records, and adaptive trace fields.
- `script/run_optimizer_experiments.py`: forward adaptive controls, require validated existing calibration, and write adaptive metadata.
- `Makefile`: add an explicit adaptive-convergence target only after the script is stable.
- `CURRENT_CODE_FLOW.md`: document the new flow and caveats.

Use subprocesses, not threads. Every worker gets a private cache, simulator directory, run directory, log, and progress file. Calibration is prepared and validated before workers start. Progress and state files are written atomically.

## Campaign State Machine

```text
PREPARING -> PILOT_3_RUNNING -> PILOT_3_ASSESSING
           -> PILOT_5_RUNNING -> PILOT_5_ASSESSING
           -> FORMAL_10_RUNNING -> FORMAL_10_ASSESSING
           -> COMPLETE_PASS or COMPLETE_FAIL
```

Pilot seeds are initial `7000..7004`, search `8000..8004`. Formal seeds are initial `12000..12009`, search `13000..13009`.

Proceed from pilot 3 when the largest exact fingerprint group is at least 2/3 and objective quality is valid. Proceed from pilot 5 only when at least 4/5 share a fingerprint. Pass formal testing only when at least 8/10 share one fingerprint, the formal modal fingerprint matches the pilot modal fingerprint, objective quality is within 1%, and every trace and reevaluation is valid.

The controller must stop on corrupt data or repeated operational failure. A valid divergent result is evidence and is never retried. `resume` skips valid completed runs and never duplicates active workers.

## Required Tests

Test configuration validation, target acceptance, invalid/no-op exclusion, Jeffreys smoothing, heating/cooling, bounds, fallback cooling, stagnation, reheat limits, deterministic decisions, exact move-budget termination, legacy fixed-mode behavior, callback delivery, invalid-proposal recording, malformed controller output, result identity validation, gate transitions, quality gaps for negative objectives, atomic state/progress writes, worker limits, resume, locking, and terminal reports.

## Verification

Run targeted tests first:

```bash
python -m unittest tests.test_annealing_schedule tests.test_simulated_annealing_hooks tests.test_adaptive_sa_convergence -v
```

Then run the full suite:

```bash
python -m unittest discover -s tests -v
```

Before the overnight campaign, run a separate 300-move smoke campaign and verify temperature blocks, progress, reevaluation, status, and resume. The final report must include code revision, calibration identity, schedule/controller configuration, seeds, objective distributions, exact fingerprints, raw metrics, temperature history, reheats, runtime, simulator calls, and explicit limitations.
