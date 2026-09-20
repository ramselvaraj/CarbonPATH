# 0002. Evaluators own their ATLAS input adapters

- Status: Accepted
- Date: 2026-09-19
- Branch: `modular-carbonpath`

## Context

ADR 0001 made ATLAS graph dumps the direct input and separated graph execution
from operation modeling. The first adapter, however, reduced every node to a
narrow operation record (`GemmOperation` or `ReluOperation`) and discarded the
rest of the node. A new model that needed another ATLAS fact — a ReLU estimator
reading `reuse_factor`, or a Softmax estimator reading `axis` — could not get it
without changing the central parser and the executor. That is not a replaceable
model; it is a hard-coded one.

## Decision

One immutable `AtlasOperation` represents a recognized node. It carries the
generic execution facts (operation type, input and output `TensorSpec`, GEMM
dimensions) and an `AtlasNodeView` over the original node. The source view is
deeply frozen and is the only path to node-specific facts.

Each operation evaluator is paired with an operation input adapter:

```text
AtlasOperation
  -> evaluator input adapter   (reads operation.source and the access plan)
  -> typed evaluator input
  -> operation evaluator       (returns compute latency and dynamic energy)
```

The graph adapter validates graph-level structure (input nodes, linear-chain
order, tensor shapes). An evaluator input adapter validates the model-specific
facts its model needs, and fails with an unsupported evaluation when they are
absent.

The executor uses a type-neutral tensor access plan (first input from DRAM, last
output to DRAM, everything else resident or moved) instead of branching on
`relu` or `softmax`.

Evaluation profiles resolve all four selections through registries: evaluator,
operation placement policy, tensor movement policy, and transfer cost model.

## Consequences

- Adding a model is a new evaluator registration plus its input adapter, not a
  change to `AtlasGraphAdapter` or the executor.
- A graph that lacks an attribute one model needs still loads and evaluates for
  every model that does not need it.
- The graph fingerprint hashes the complete raw artifact, so it identifies the
  exact external workload rather than the subset CarbonPATH currently reads.
- Placeholder models (`placeholder_hls_relu_v0`, `placeholder_fpga_softmax_v0`)
  demonstrate the seam and are explicitly not hardware characterizations.
- `legacy_sa_fpga_v1` remains the default profile and its results are pinned by
  a compatibility test.
- Known limitation: ATLAS precision is preserved in the source view but transfer
  size is still one byte per element.
