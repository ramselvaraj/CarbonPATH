# Adaptive SA Pilot Gate Summary

## Outcome

The workload-7 adaptive simulated-annealing campaign stopped at the pilot-3 gate. It did not crash, restart, or encounter an operational failure.

All three pilot runs completed the full 75,650-move search and independently reevaluated their best architectures successfully.

## Results

Each run found the same objective:

```text
-2.0314255921830546
```

The runs did not return the same exact canonical architecture:

```text
5983441962aa: 1 run
c8fc919fa4b2: 1 run
0ddfdccbd4f2: 1 run
```

The common architecture summary was:

```text
2 chiplets
64x64 arrays
7 nm technology
256 KiB SRAM per chiplet
3D package
HBM3 memory
Output-stationary dataflow
No split-K
```

The differences were in equal-cost configuration details:

- Workload assignment order differed between runs.
- One run used an 8/8 memory allocation instead of 7/9.
- One run reversed the chiplet stack and interconnect direction.

## Why The Campaign Stopped

The campaign requires exact architecture convergence, not only objective convergence.

The pilot-3 gate requires at least two of three runs to return one identical canonical architecture:

```text
Required: 2 of 3
Observed: 1 of 3
```

The objective-quality condition passed because the modal objective had a zero relative gap to the best verified objective. The exact-fingerprint condition failed, so the controller correctly stopped before pilot runs 4 and 5.

## Interpretation

The adaptive search repeatedly found the same best score, but the objective function treats several different architectures as equivalent. The result demonstrates objective convergence without exact architecture convergence.

The controller did not retry the valid divergent results because they are scientific evidence, not operational failures. Re-running the same campaign with different seeds may reproduce the same behavior if these equal-cost alternatives remain available.

## Available Decisions

### Investigate Equivalent Optima

Determine whether the differing workload order, stack order, and memory allocation fields are physically or scientifically meaningful. If they are equivalent for the research question, define an equivalence-level architecture identity for convergence reporting.

### Keep Exact Architecture Convergence

Keep the current gate unchanged and treat this campaign as a valid convergence failure. Investigate why the objective cannot distinguish the architectures before another formal campaign.

### Add A Deterministic Tie-Breaker

Use a deterministic secondary ordering when objective values are equal. This could force exact reproducibility, but it changes the original experiment contract.

### Use Objective Convergence

Change the gate to accept repeated equivalent objective values. Under that rule, the pilot result is 3/3 objective convergence, but this changes the research question from exact architecture repeatability to score repeatability.

## Current Campaign State

- Workload: 7
- Stage: pilot 3
- State: `COMPLETE_FAIL`
- Formal runs started: 0
- Operational errors: 0
- Remote campaign processes: none
- Formal output remains unlaunched
