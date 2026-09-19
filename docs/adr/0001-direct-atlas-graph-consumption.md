# 0001. CarbonPATH consumes ATLAS graph dumps directly

- Status: Accepted
- Date: 2026-09-18
- Branch: `modular-carbonpath`

## Context

CarbonPATH previously required an external parser to rewrite an ATLAS graph into
a CarbonPATH-owned normalized envelope (`atlas-normalized-v0`). That created a
second workload schema to maintain, tied graph ingestion to a parser step, and
left operation handling as hard-coded `gemm`/`relu` branches in
`main.simulate_operation_sequence`.

ATLAS already emits a stable artifact in graph-only mode: a top-level JSON array
of lowered nodes with explicit inputs, outputs, weights, and attributes. That
artifact is the real external contract.

## Decision

ATLAS owns the graph artifact. CarbonPATH consumes it directly through
`AtlasGraphAdapter` and exposes only the facts its models need. CarbonPATH does
not define or accept a normalized ATLAS workload format.

Graph execution is separated from operation modeling:

- The executor owns order, operation placement, tensor residency, tensor
  movement, and aggregation.
- An evaluation profile names the replaceable operation evaluators, placement
  policy, movement policy, and transfer cost model.
- An operation evaluator returns only compute latency and dynamic energy.
- A tensor movement service owns intermediate activation handoffs.
- GEMM tile mapping remains the Scheduler's responsibility and is distinct from
  operation placement.

The first profile (`legacy_sa_fpga_v1`) wraps the existing SCALE-Sim GEMM path
and the existing FPGA ReLU model, requires one SA and one FPGA endpoint, and
supports one linear chain of constant-weight GEMMs and ReLUs.

## Consequences

- ATLAS graph dumps are the supported input; `atlas-normalized-v0` and the
  external parser are removed.
- Replacing the ReLU model or adding a non-GEMM model is a new evaluator
  registration, not a change to the executor.
- Model selection is recorded in evaluation output through the profile
  fingerprint, so results from different model sets are not conflated.
- Unsupported graph features, profiles, or architectures stop the evaluation;
  there is no partial result.
- The first scope deliberately excludes branches, residuals, dynamic GEMMs,
  multi-endpoint placement, streaming, and overlap.
