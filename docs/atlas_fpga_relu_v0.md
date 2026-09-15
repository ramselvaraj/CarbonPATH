# ATLAS Parser Workloads and FPGA ReLU V0

This document defines the first end-to-end path that evaluates a normalized
parser workload containing both GEMM and non-GEMM operations on a chiplet
system with a systolic-array (SA) chiplet and an FPGA chiplet.

## Scope

V0 supports exactly one sequential chain of operations:

```text
GEMM -> ReLU -> GEMM
```

Constraints:

- `int8` only.
- One systolic-array chiplet and one FPGA chiplet.
- No branching, residuals, fan-out, or joins.
- ReLU executes on the FPGA chiplet.
- Transfers and computation are serialized; there is no streaming overlap.
- No transfer fallback to DRAM: a missing route makes the placement invalid.
- No FPGA resource profile inference without synthesis data.
- No FPGA architecture search or mutation.

## External Parser Contract

CarbonPATH does not parse raw ATLAS graph dumps. An external parser converts an
ATLAS graph into the normalized envelope below (`format` =
`atlas-normalized-v0`). Load it with
`system.utils.AtlasWorkload.load_atlas_workload`.

### GEMM operation

```json
{
  "operation_id": "ffn_dense0",
  "operation_type": "gemm",
  "input_tensor_id": "ffn_in0",
  "weight_tensor_id": "ffn_dense0:weights",
  "output_tensor_id": "ffn_dense0:output",
  "input_shape": [8, 64],
  "weight_shape": [64, 256],
  "output_shape": [8, 256],
  "m": 8,
  "k": 64,
  "n": 256
}
```

### ReLU operation

```json
{
  "operation_id": "ffn_act0",
  "operation_type": "relu",
  "input_tensor_id": "ffn_dense0:output",
  "output_tensor_id": "ffn_act0:output",
  "input_shape": [8, 256],
  "output_shape": [8, 256],
  "element_count": 2048
}
```

Parser rules:

- `input_shape == [M, K]`, `weight_shape == [K, N]`, `output_shape == [M, N]`.
- ReLU `input_shape == output_shape` and `element_count == product(input_shape)`.
- Operation IDs are unique, tensor IDs are non-empty, and every operation after
  the first consumes the previous operation's output tensor.
- Precision metadata from ATLAS is intentionally ignored in V0; every tensor is
  treated as one `int8` byte per element.

See `cfg/examples/atlas_gemm_relu_gemm.json`.

## FPGA Chiplet Configuration

An FPGA chiplet is declared in the architecture with
`"chiplet_type": "fpga"`. It carries its resource counts and a characterized
per-lane ReLU implementation profile:

```json
{
  "chiplet_type": "fpga",
  "tech_node": "7",
  "area": 10.0,
  "power": 2.0,
  "frequency_hz": 300000000,
  "clbs": 10000,
  "brams": 200,
  "dsps": 500,
  "relu_implementation": {
    "clbs_per_lane": 5,
    "brams_per_lane": 0,
    "dsps_per_lane": 0,
    "max_parallel_lanes": 64,
    "energy_per_element_pj": null
  }
}
```

`parallel_lanes` is derived from resources, never assumed:

```text
clb_lanes  = floor(clbs / clbs_per_lane)
bram_lanes = floor(brams / brams_per_lane)   # only when brams_per_lane > 0
dsp_lanes  = floor(dsps / dsps_per_lane)     # only when dsps_per_lane > 0

parallel_lanes = min(all applicable limits)
parallel_lanes = min(parallel_lanes, max_parallel_lanes)  # when set

compute_cycles = ceil(element_count / parallel_lanes)
compute_latency_ns = compute_cycles / frequency_hz * 1e9
```

`clbs_per_lane`, `brams_per_lane`, and `dsps_per_lane` are implementation
inputs, not derivable from tensor dimensions. They should come from synthesis
data; V0 accepts them as configuration. If the derived lane count is zero the
operation is reported infeasible and no latency is produced.

ReLU compute energy is reported only when `energy_per_element_pj` is configured;
otherwise it is `null`, never zero. Transfer energy is always modeled.

## Transfer Model

The architecture may declare:

```json
{
  "transfer_model": {
    "setup_latency_ns": 5.0,
    "hop_latency_ns": 1.0
  }
}
```

The route itself (path, reciprocal bandwidth, energy per bit) is resolved by
`ChipletSystem.get_shortest_path`. `TransferEstimator` receives the resolved
route and a tensor payload:

```text
byte_count = element_count            # int8
hop_count  = max(0, len(path) - 1)

latency_ns = setup_latency_ns
           + hop_count * hop_latency_ns
           + byte_count * path_bandwidth_reciprocal_ns_per_byte

energy_pj  = byte_count * 8 * path_energy_pj_per_bit
```

A local route (same endpoint or no path) has zero transfer cost.

## Stage Composition and Accounting Ownership

`ReluStageComposer` composes one serialized stage:

```text
total_latency_ns = input_transfer_latency
                 + relu_compute_latency
                 + output_transfer_latency
```

For `GEMM 1 -> ReLU -> GEMM 2`:

- `GEMM 1` runs with `output_to_dram = False`; its output crosses to the FPGA
  and is charged by the ReLU stage's input transfer.
- The ReLU stage owns both the SA->FPGA input transfer and the FPGA->SA output
  transfer.
- `GEMM 2` runs with `activation_from_dram = False`; its input is supplied by
  the ReLU stage's output transfer.

This gives every physical tensor movement exactly one accounting owner. The
network-level compute energy continues to use the existing
`system_power_w * total_latency_ns * 1000` convention, which includes FPGA
power; the specialized ReLU compute energy is reported separately and is not
added again.

## Modules

- `system/utils/AtlasWorkload.py`: normalized parser frontend.
- `system/utils/FpgaChiplet.py`: FPGA endpoint representation.
- `system/utils/TransferEstimator.py`: route-based transfer estimate.
- `system/utils/NonGemmEstimator.py`: ReLU compute estimator, stage composer,
  and operation dispatcher.
- `system/utils/ChipletSystem.py`: builds SA and FPGA endpoints and routes
  between them.
- `main.simulate_operation_sequence`: mixed GEMM/non-GEMM sequence evaluator.
- `network.evaluate_atlas_network`: reporting for parser workloads.

## Running

```bash
.venv/bin/python -m network validate \
  --network cfg/examples/atlas_gemm_relu_gemm.json

.venv/bin/python -m network evaluate \
  --network cfg/examples/atlas_gemm_relu_gemm.json \
  --architecture cfg/examples/sa_fpga_architecture.json \
  --output-dir reports/networks/gemm_relu_gemm/evaluate
```

Calibration and memory-policy comparison are not supported for parser workloads
in V0.

## Known Limitations

- ReLU is the only non-GEMM operation with an estimator.
- One SA and one FPGA chiplet only; no tensor slicing across FPGA chiplets.
- The resource per-lane coefficients are uncalibrated configuration inputs;
  resource feasibility is only as trustworthy as those coefficients.
- Transfers and computation do not overlap.
- No link contention, packetization, or per-hop switching energy.
- FPGA area/power enter the existing physical metrics, but FPGA choice is fixed
  and not part of the architecture search.
