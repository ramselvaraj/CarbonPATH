# Direct ATLAS Graph Evaluation

This document defines how CarbonPATH evaluates a raw ATLAS graph. ATLAS owns the
graph artifact; CarbonPATH owns an adapter and a replaceable set of operation
evaluators. CarbonPATH does not maintain a normalized ATLAS workload format.

## Input Contract

The input is the top-level JSON array emitted by ATLAS graph-only mode
(`run_atlas_flow(..., dump_graph=True, stop_after_convert=True)`). Each node
carries `name`, `class`, `inputs`, `output`, `weights`, and `attrs`.

First scope supports exactly one linear chain:

- `Gemm` nodes with `attrs.weights_in_core == True` (constant weights).
- `Activation` nodes with `attrs.activation == "relu"`.
- `Softmax` nodes (the form emitted by ATLAS).
- `Input` nodes are skipped as the graph entry.

Every other node class, dynamic two-operand GEMM, or non-linear graph shape is
reported as an unsupported evaluation.

GEMM dimensions come from ATLAS shapes:

```text
rank-2 input/output:  M, K = input_shape,  N = output_shape[1]
rank-1 input/output:  M = 1, K = input_shape[0], N = output_shape[0]
```

Non-GEMM element count is the product of the output shape. Tensors are treated as
one `int8` byte per element. ATLAS precision metadata is preserved in the source
view but not modeled in cost yet.

An example graph is
`cfg/examples/atlas/dense_relu_funnel.graph_dump.json`, a dense-ReLU funnel:

```text
Input(128,128)
-> Gemm(128,128,1024)  -> ReLU
-> Gemm(128,1024,512)  -> ReLU
-> Gemm(128,512,256)   -> ReLU
-> Gemm(128,256,64)
```

A second fixture, `cfg/examples/atlas/dense_softmax.graph_dump.json`, exercises
Softmax:

```text
Input(8,16)
-> Gemm(8,16,16) -> Softmax(axis=-1) -> Gemm(8,16,4)
```

## ATLAS Operation and Source View

The graph adapter produces one immutable `AtlasOperation` per recognized node. It
carries the generic execution facts (operation type, input and output
`TensorSpec`, GEMM dimensions) plus an `AtlasNodeView` over the original node.
The source view is deeply frozen and is the only path an evaluator input adapter
uses to read node-specific facts such as `reuse_factor` or `axis`.

The adapter validates graph-level structure (input nodes, linear-chain order,
tensor shapes). Model-specific source requirements are validated later by the
selected evaluator's input adapter, so a graph that lacks an attribute a model
needs still loads for every model that does not need it.

## Evaluation Profile

An evaluation profile is the named selection of models and policies for one
evaluation. It contains names and versions, not Python imports and not hardware
values:

```json
{
  "profile": "legacy_sa_fpga_v1",
  "version": 1,
  "evaluators": {
    "gemm": "legacy_scale_sim_gemm_v1",
    "relu": "legacy_fpga_relu_v1"
  },
  "assignment_policy": "fixed_single_sa_single_fpga_v1",
  "movement_policy": "activation_boundary_v1",
  "transfer_model": "route_transfer_v1"
}
```

The profile is selected per evaluation and its fingerprint is recorded with the
results. All four selections are resolved through registries: evaluator,
operation assignment policy, tensor movement policy, and transfer cost model.

A second, opt-in profile demonstrates the extension seam:

```json
{
  "profile": "placeholder_fpga_ops_v0",
  "version": 1,
  "evaluators": {
    "gemm": "legacy_scale_sim_gemm_v1",
    "relu": "placeholder_hls_relu_v0",
    "softmax": "placeholder_fpga_softmax_v0"
  },
  "assignment_policy": "fixed_single_sa_single_fpga_v1",
  "movement_policy": "activation_boundary_v1",
  "transfer_model": "route_transfer_v1"
}
```

## Operation Input Adapters and Evaluator Contract

Every evaluator receives a typed input view built by its paired operation input
adapter. The evaluator never reads the raw ATLAS artifact.

```text
AtlasOperation
  -> evaluator input adapter
  -> typed evaluator input
  -> operation evaluator
  -> compute latency + dynamic energy
```

Every evaluator returns only:

```text
compute_latency_ns
dynamic_energy_pj
```

The executor owns operation identity, operation order, assignment, tensor
residency, tensor movement, totals, and report formatting. Evaluators are
replaceable: a new GEMM or non-GEMM model is a new registration plus its input
adapter, not a new branch in the executor.

Current registrations:

- `legacy_scale_sim_gemm_v1`: existing SCALE-Sim GEMM path. Its dynamic energy is
  the GEMM's DRAM/interconnect plus SRAM energy.
- `legacy_fpga_relu_v1`: FPGA ReLU compute. Dynamic energy is the configured
  energy-per-element result, otherwise zero additional energy.
- `placeholder_hls_relu_v0`: **placeholder, not a hardware characterization.**
  Reads `attrs.reuse_factor` through its input adapter and multiplies the
  resource-derived cycle count by it.
- `placeholder_fpga_softmax_v0`: **placeholder, not a hardware
  characterization.** Reads the Softmax `axis` and uses one cycle per element.

## Tensor Access Plan

The executor derives a type-neutral access plan from graph position:

```text
first operation input  -> DRAM
intermediate tensor    -> resident or moved by the tensor movement service
last operation output  -> DRAM
```

Evaluator input adapters translate this plan into the facts their model needs.
The legacy GEMM input adapter maps it onto the SCALE-Sim `activation_from_dram`
and `output_to_dram` flags. The executor contains no `relu` or `softmax` branch.

## Operation Assignment

The first assignment policy requires exactly one systolic-array endpoint and one
FPGA endpoint:

```text
gemm    -> the SA endpoint
relu    -> the FPGA endpoint
softmax -> the FPGA endpoint
```

Operation assignment is distinct from GEMM tile mapping. The existing Scheduler
still assigns the tiles of one GEMM across its SA group.

## Tensor Movement

`TensorMovementService` (`activation_boundary_v1`) owns intermediate activation
handoffs. It records where each produced tensor becomes resident and, before the
next operation, resolves a route from the producer endpoint to the consumer
endpoint.

```text
same endpoint: retain locally, zero cost
different endpoint with a route: resolve route, charge latency and energy
no route: unsupported evaluation
```

The consumer owns its incoming movement, so every physical tensor movement is
charged exactly once. Weights and the graph's external inputs and outputs remain
the legacy GEMM wrapper's responsibility.

## Aggregation

```text
total latency = sum(operation compute latency + incoming movement latency)

total energy  = baseline architecture power * total latency
              + sum(operation dynamic energy)
              + sum(movement dynamic energy)
```

`legacy_fpga_relu_v1` returns zero additional dynamic energy until a
characterized coefficient exists. Baseline architecture power still covers the
currently modeled FPGA power, so this is not a claim that the operation consumes
no energy.

## Running

```bash
.venv/bin/python -m network validate \
  --network cfg/examples/atlas/dense_relu_funnel.graph_dump.json

.venv/bin/python -m network evaluate \
  --network cfg/examples/atlas/dense_relu_funnel.graph_dump.json \
  --architecture cfg/examples/sa_fpga_architecture.json \
  --output-dir reports/networks/dense_relu_funnel/evaluate

.venv/bin/python -m network evaluate \
  --network cfg/examples/atlas/dense_softmax.graph_dump.json \
  --architecture cfg/examples/sa_fpga_architecture.json \
  --evaluation-profile cfg/profiles/placeholder_fpga_ops_v0.json \
  --output-dir reports/networks/dense_softmax/evaluate
```

`--evaluation-profile` is ATLAS-only. Calibration and `compare-memory` are not
supported for ATLAS graphs.

## Modules

- `system/utils/AtlasGraphAdapter.py`: raw ATLAS graph front end; produces
  `AtlasOperation` and `AtlasNodeView`.
- `system/utils/OperationEvaluator.py`: evaluator, input adapter, binding, and
  registry contracts.
- `system/utils/EvaluationProfile.py`: profile loading and identity.
- `system/utils/OperationAssignment.py`: operation-to-endpoint assignment and
  registry.
- `system/utils/TensorMovement.py`: residency, movement plans, and registry.
- `system/utils/TransferEstimator.py`: resolved-route transfer cost model and
  registry.
- `system/utils/AtlasObjective.py`: replaceable objective models and the raw
  design point used by modular annealing.
- `system/utils/WorkloadMapping.py`: the workload mapping of a design point.
- `system/utils/DesignPointFile.py`: the one result file (architecture and
  workload mapping).
- `system/utils/AtlasAnnealingMoves.py`: architecture and workload-mapping moves.
- `main.evaluate_atlas_graph`: the graph executor.
- `main.evaluate_atlas_design_point`: the modular annealing objective bridge.
- `network.evaluate_atlas_network`: reporting for ATLAS graphs.

## Known Limitations

- One linear chain; no branches, joins, or residuals.
- Constant-weight GEMM only.
- Whole-tensor movement only; no streaming or tiling across FPGA endpoints.
- A workload mapping can use several SAs and several FPGA chiplets. An operation on FPGA chiplets splits **by rows**
  (`system/utils/FpgaSplit.py`): the rows divide in blocks that differ by at most one row, the chiplets run at the same
  time, the latency is the slowest chiplet, and the energy is the sum. Each chiplet has its own constants, buffer, and
  resource check. A split needs a distributed movement policy (`distributed_activation_boundary_v2`): each chiplet gets the
  rows it owns from the SAs and sends them on directly. The loads of the constants share the DRAM and run one after
  the other. The annealing moves are `fpga_count` (add or remove a chiplet; a new chiplet joins no operation) and
  `fpga_set` (add or remove a chiplet from one operation). The full search space lists `fpga.count: [1, 2]`; the placeholder and single
  spaces list `[1]`, so the count move is off there. A split needs the profile with `distributed_activation_boundary_v2`;
  with an older movement policy the split points are rejected.
- The per-chiplet resource check adds up the kernels of all operations on one FPGA chiplet and counts the input
  buffer once (`system/utils/ChipletResourceCheck.py`). See `docs/rule4ml_assumptions.md`.
- ReLU and Softmax are the only non-GEMM operation types.
- The HLS ReLU and Softmax evaluators are placeholders, not measured hardware
  models.
- Transfers and computation are serialized; no contention or overlap.
- `LayerNormalization` runs on the FPGA with `fpga_layernorm_v1` (`system/utils/FpgaLayerNorm.py`). rule4ml has no
  LayerNorm predictor, so the latency comes from the loop structure of the hls4ml kernel: one cycle for each
  dependent operation, `(seq_len - 1) * reuse_factor + 2 * ceil(log2(dim)) + 11` cycles. It is not validated against
  a synthesis report. It predicts no logic resources. It charges the input buffer of the FPGA (read delay and BRAM)
  and the parameters (scale, bias, and the rsqrt table, 8 bits each): their BRAM counts against the FPGA, and with
  `parameter_source: "dram"` they load over DRAM -> systolic array -> link -> FPGA
  (`system/utils/ParameterLoad.py`). The compute time of the previous operation, when it runs on another endpoint,
  hides the load: `exposed = max(0, load - previous compute)`. The energy is always paid. `"bitstream"` (the default) has no load.
  The profiles `atlas_modular_layernorm_v1` and `rule4ml_hybrid_layernorm_v1` name it; the
  older profiles are unchanged because their fingerprints are pinned.
- ATLAS precision metadata is preserved but ignored in cost; every tensor is one
  `int8` byte per element.
