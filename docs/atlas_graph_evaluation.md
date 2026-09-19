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
- `Input` nodes are skipped as the graph entry.

Every other node class, dynamic two-operand GEMM, or non-linear graph shape is
reported as an unsupported evaluation.

GEMM dimensions come from ATLAS shapes:

```text
rank-2 input/output:  M, K = input_shape,  N = output_shape[1]
rank-1 input/output:  M = 1, K = input_shape[0], N = output_shape[0]
```

ReLU element count is the product of its input shape. Tensors are treated as one
`int8` byte per element. ATLAS precision metadata is not modeled yet.

An example graph is
`cfg/examples/atlas/dense_relu_funnel.graph_dump.json`, a dense-ReLU funnel:

```text
Input(128,128)
-> Gemm(128,128,1024)  -> ReLU
-> Gemm(128,1024,512)  -> ReLU
-> Gemm(128,512,256)   -> ReLU
-> Gemm(128,256,64)
```

## Evaluation Profile

An evaluation profile is the named selection of models for one evaluation. It
contains names and versions, not Python imports and not hardware values:

```json
{
  "profile": "legacy_sa_fpga_v1",
  "version": 1,
  "evaluators": {
    "gemm": "legacy_scale_sim_gemm_v1",
    "relu": "legacy_fpga_relu_v1"
  },
  "placement_policy": "fixed_single_sa_single_fpga_v1",
  "movement_policy": "activation_boundary_v1",
  "transfer_model": "route_transfer_v1"
}
```

The profile is selected per evaluation and its fingerprint is recorded with the
results. The first profile selects an evaluator by ATLAS operation type only.

## Operation Evaluator Contract

Every evaluator returns only:

```text
compute_latency_ns
dynamic_energy_pj
```

The executor owns operation identity, operation order, placement, tensor
residency, tensor movement, totals, and report formatting. Evaluators are
replaceable: a new GEMM or non-GEMM model is a new registration, not a new branch
in the executor.

Current registrations:

- `legacy_scale_sim_gemm_v1`: existing SCALE-Sim GEMM path. Its dynamic energy is
  the GEMM's DRAM/interconnect plus SRAM energy.
- `legacy_fpga_relu_v1`: FPGA ReLU compute. Dynamic energy is the configured
  energy-per-element result, otherwise zero additional energy.

## Operation Placement

The first placement policy requires exactly one systolic-array endpoint and one
FPGA endpoint:

```text
gemm -> the SA endpoint
relu -> the FPGA endpoint
```

Operation placement is distinct from GEMM tile mapping. The existing Scheduler
still assigns the tiles of one GEMM across its SA group.

## Tensor Movement

`TensorMovementService` owns intermediate activation handoffs. It records where
each produced tensor becomes resident and, before the next operation, resolves a
route from the producer endpoint to the consumer endpoint.

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
```

Calibration and `compare-memory` are not supported for ATLAS graphs.

## Modules

- `system/utils/AtlasGraphAdapter.py`: raw ATLAS graph front end.
- `system/utils/OperationEvaluator.py`: evaluator contract and registry.
- `system/utils/EvaluationProfile.py`: profile loading and identity.
- `system/utils/OperationPlacement.py`: operation-to-endpoint placement.
- `system/utils/TensorMovement.py`: residency and movement plans.
- `system/utils/TransferEstimator.py`: resolved-route transfer cost model.
- `main.evaluate_atlas_graph`: the graph executor.
- `network.evaluate_atlas_network`: reporting for ATLAS graphs.

## Known Limitations

- One linear chain; no branches, joins, or residuals.
- Constant-weight GEMM only.
- Whole-tensor movement only; no streaming or tiling across FPGA endpoints.
- One SA endpoint and one FPGA endpoint.
- ReLU is the only non-GEMM evaluator.
- Transfers and computation are serialized; no contention or overlap.
- ATLAS precision metadata is ignored; every tensor is one `int8` byte per
  element.
