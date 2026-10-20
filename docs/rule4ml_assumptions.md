# Rule4ML Assumptions

Living document. Updated 2026-10-20.

Sources, in order of priority:

1. The latest `rule4ml` code on the remote machine (`~/work/rule4ml`, `main` at `ccfe3b6`, 2026-01-20, v0.2),
   including the shipped predictor weights in `rule4ml/models/weights/v2/gnn`.
2. The `wa-hls4ml` dataset and its README (local copy in `~/work/rule4ml/datasets/wa-hls4ml`).
   The README says the newer predictors were trained on this dataset.
3. The wa-hls4ml paper (Hawks et al. 2026): only the abstract and the poster in Zotero were read.
4. The 2024 Rule4ML paper (Rahimifar et al.): the first version of the tool, with an MLP predictor.
   It is kept in its own section below.

How each fact was checked is marked: "code", "data" (counted in the dataset files), "README", "paper", or
"not verified".

## Which predictor we use

| Item | Value | Source |
| --- | --- | --- |
| Default predictor | `TorchGNN` (graph neural network), weights `v2/gnn`, one model for each output | code: `wrappers.py:915` |
| Outputs | BRAM, DSP, FF, LUT, CYCLES, INTERVAL (six files, one GNN each) | code |
| Older predictor | MLP, weights `v1/mlp` (the 2024 paper) | code |
| Training data of the shipped weights | `wa-hls4ml` ("newer predictors"). Which subsets exactly: not verified. There is no training notebook in the repo. | README |
| Our code default | hls4ml 1.1.0 and Vivado 2024.2 (`system/utils/Rule4MlEvaluators.py`) | CarbonPATH code |

All the activation numbers in this document (sweep, 191 ns) come from this default GNN.

## Inputs and outputs

| | Item | Source |
| --- | --- | --- |
| Input | Keras or PyTorch model (the workload) | code |
| Input | Board: `pynq-z2`, `zcu102`, `alveo-u200`, `alveo-u250` (encoded as 1 to 4) | code: model config |
| Input | Strategy: `latency` or `resource` | code |
| Input | Precision (bit width), reuse factor | code |
| Input | hls4ml version: `0.8.1` or `1.1.0` | code |
| Input | Vivado version: `2019.1` to `2024.2` (12 values) | code |
| Not an input | Clock period. The parser reads it (`data_parser.py:325`), but the shipped model has no clock feature. | code |
| Output | LUT, FF, DSP, BRAM as counts (and percent of the board) | code |
| Output | `CYCLES` and `INTERVAL`: cycle counts, not time | code |

- Time = cycles x the clock period of the design (Fmax for the design). CarbonPATH uses `frequency_hz` of the FPGA chiplet.
- Board capacities (`rule4ml/parsers/supported_boards.json`): pynq-z2 (53,200 LUT, 220 DSP, 280 BRAM),
  zcu102 (274,080 LUT, 2,520 DSP, 1,824 BRAM), alveo-u200 (1,182,240 LUT, 6,840 DSP, 4,320 BRAM),
  alveo-u250 (1,728,000 LUT, 12,288 DSP, 5,376 BRAM).

## How the model sees a tensor

- The GNN input has whole-network features and per-layer features: layer input size, output size, parameter count,
  reuse factor, and counts of add, multiply, lookup, and logical operations. Layer types: input, dense, conv1d,
  conv2d, pooling, relu, sigmoid, tanh, softmax, batchnorm, add, concatenate, dropout, flatten. (code: config labels)
- The input and output size of a layer is the product of all dimensions of the tensor (`network_parser.py:158-159`).
  So `8x32`, `256`, and `(256,)` give the same prediction. My sweep showed this. (code)

## Training data: wa-hls4ml

Source: dataset README, and counts from the JSON files on the remote machine.
Total: 683,176 samples (608,679 fully connected, 31,278 conv1d, 43,219 conv2d). Split: 478,220 train,
102,472 validation, 102,484 test. Plus 887 exemplar samples.

| Subset | Samples checked | Clock (target) | Board | Tool | hls4ml | Strategy | I/O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2_20 (rule4ml-style) | train 6,677 | 10 ns (all) | zcu102 2,248, alveo-u200 2,244, pynq-z2 2,185 | Vivado, version field empty | 0.8.1 | not counted | io_parallel (all) |
| 2layer | train 45,716 | 5 ns (all) | alveo-u250 | Vitis 2023.2 | 0.8.1 | resource (all) | not stored |
| conv1d | train 21,894 | 5 ns (11,621); missing (10,273) | alveo-u250 | Vitis 2024.2 | 1.1.0 | resource (all) | io_stream (README) |
| latency (test) | 4,320 | 5 ns (3,676); missing (644) | alveo-u250 | Vitis 2024.2 | 1.1.0 | latency (all) | not stored |
| resource (test) | 3,278 | 5 ns (2,492); missing (786) | alveo-u250 | Vitis 2024.2 | 1.1.0 | resource (all) | not stored |
| conv2d (test) | 6,484 | 5 ns (2,043); missing (4,441) | alveo-u250 | Vitis 2024.2 | 1.1.0 | resource (all) | io_stream (README) |
| exemplar | 887 | 5 ns (448); 10 ns (439) | alveo-u200 452, alveo-u250 435 | Vivado 2019.1 and 2020.1 | 0.8.1 | resource 444, latency 443 | not stored |

Not checked: `train_3layer`, the validation files, and the other conv2d files (the 3layer file is 1.9 GB).

Network limits (README):

- Layers: 2 to 20 in the 2_20 subset; 2 to 7 in the others (3 to 7 for conv).
- Neurons or features: 8 to 128 (conv: 32 to 128 with 8 to 64 filters). Weight precision 2 to 16 bits (`ap_fixed<X,1>`).
- Target reuse factor: 1 to 4,093 (fully connected), 8,192 to 32,795 (conv).
- Activations: linear for most 2-layer and 3-layer networks. ReLU, tanh, and sigmoid for the other networks.
- The dataset card says: most of the dataset is very small (2 to 3 layer) dense networks without activations.
  It asks users to weight the data.
- Strategy: most networks use Resource. Latency is used for some 3 to 7 layer networks and the 2_20 subset.

## Kernel I/O assumptions

| Item | Value | Source |
| --- | --- | --- |
| I/O type, fully connected networks | `io_parallel`: the output of one layer is wired to the next layer. All values are present at once. | README, data (2_20) |
| I/O type, conv networks | `io_stream`: FIFO buffers between layers | README |
| Clock period | 10 ns (2_20 subset). 5 ns in the other subsets. The model has no clock input. | data, code |
| Latency | C synthesis (HLS) report: `cycles_min` and `cycles_max`. The training target is the average of the two. | code: `data_parser.py:730-740` |
| Resources | Logic synthesis (Vivado or Vitis) of the generated RTL. Trained as a percent of the board capacity. | README, code |
| Flow | Build flags in the samples: `synth` and `vsynth` on. `csim`, `cosim`, `validation`, `export`, and `bitfile` off. | data (2_20 samples) |
| Hardware or RTL simulation | None. No place-and-route either. | data |
| Estimated clock | The HLS report has an `estimated_clock`. It is stored in the data. The model does not predict it. | data |
| Batch size | Not stated. One inference at a time. | not verified |
| Weights | Constants on chip. Weight loading is not modeled. | hls4ml |

Consequences:

- Every input value is available at once for dense and activation layers. There is no streaming in that data.
- Cycles cover the compute inside the kernel. They do not cover moving data to or from the FPGA.
- Results are tool estimates, not measurements.

## Pragmas and parallelism

- Rule4ML has no pragma input. hls4ml writes the pragmas from strategy, reuse factor, precision, and I/O type.
- Training data used hls4ml 0.8.1 and 1.1.0. The version is a model input.
- hls4ml is not installed with Rule4ML. I read the fork `atlas/hls4ml-gemm` on the remote machine
  (`itsruthwik/hls4ml-gemm`, commit `b8cc1092`). It is newer than the training versions, so treat all of this as
  "very likely the same", not verified.

Whole-network pragma. The default `PipelineStyle` is `auto`. The rule (`backends/vivado/passes/pipeline_style.py`) is:

| Case | Whole-network pragma |
| --- | --- |
| `io_stream` | `DATAFLOW` |
| A conv layer with more than one partition | `DATAFLOW` |
| Any layer uses the Resource strategy (also with `io_parallel`) | `DATAFLOW` |
| `io_parallel` with Latency strategy | `PIPELINE` |

- `DATAFLOW`: each layer is its own stage, and stages overlap for successive inputs. One inference takes the sum of the
  stage latencies. Most of wa-hls4ml uses the Resource strategy, so most of it is `DATAFLOW`.
- `PIPELINE`: the whole network is one unrolled pipeline. Only Latency strategy with `io_parallel`.

Port pragmas of the network function (`writer/vivado_writer.py:442-462`):

| I/O type | Pragmas |
| --- | --- |
| `io_parallel` | `ARRAY_PARTITION` on the input and output arrays, `INTERFACE ap_vld` |
| `io_stream` | `INTERFACE axis`. FIFOs between layers (`STREAM` with a depth). No partition of the I/O. |

Layer pragmas:

| Level | Pragma | Where |
| --- | --- | --- |
| Dense, Latency strategy | `PIPELINE II=reuse_factor`, arrays `complete`, `ALLOCATION mul limit=multiplier_limit` | `nnet_dense_latency.h:26-33` |
| Dense, Resource strategy | `ARRAY_RESHAPE weights block`, `PIPELINE II=1 rewind`, weights in ROM | `nnet_dense_resource.h:29-56` |
| Dense, `io_stream` | `INLINE recursive`, `PIPELINE II=reuse_factor`, `ARRAY_PARTITION complete`, `UNROLL` | `nnet_dense_stream.h` |
| Activation | `PIPELINE` per element loop; table lookup for sigmoid and tanh | `nnet_activation.h` |
| Softmax | `PIPELINE`, exp and inverse tables, `array_partition complete` | `nnet_activation.h:176-270` |

Other pragma types in the fork (count of uses): `UNROLL` 357, `ARRAY_PARTITION` 233, `PIPELINE` 192, `INLINE` 116,
`ALLOCATION` 24, `LOOP_FLATTEN` 15, `ARRAY_RESHAPE` 14, `function_instantiate` 12, `STREAM` 10, `DATAFLOW` 6, `RESOURCE` 4,
`INTERFACE` 4, and one each of `BIND_STORAGE`, `AGGREGATE`, `data_pack`.

The `VivadoAccelerator` backend (used by `nn_synth.py`) adds an AXI wrapper `<project>_axi` with copy loops for the input
and output. The HLS top function is the network function, not the wrapper (`templates/vivado/build_prj.tcl:154`,
`set_top ${project_name}`). So the C synthesis cycle count does not include the wrapper loops.

What this means:

- Unrolling is not a free choice. Reuse factor sets `II` (Latency) or the block size (Resource). All of this comes from the config, not from the user.
- The pragmas enhance the design (pipeline, partition, unroll, dataflow). They are not a bare baseline.
- Sigmoid and tanh use a table (`table_size` 1024 by default), so they cost BRAM or LUTs for the table, not DSPs for the math.
- With `DATAFLOW` (Resource strategy), `CYCLES` is the pass latency and `INTERVAL` is the time before the next input.
  With `PIPELINE`, the two are close. CarbonPATH uses `CYCLES`.

## Activation predictions

Data: `docs/rule4ml_activation_sweep.csv`. Script: `script/rule4ml_activation_sweep.py` (run on the
remote machine with `~/work/venv-cp-rule4ml`). Default predictor (GNN). Default settings: zcu102, Latency,
`ap_fixed<16, 6>`, reuse 1, hls4ml 1.1.0, Vivado 2024.2. Each row is one single-activation Keras model.

| Op, 256 elements | Cycles | LUT | DSP | BRAM |
| --- | --- | --- | --- | --- |
| relu | 0.0 | 5,504 | 63 | 2.6 |
| sigmoid | 48.8 | 7,454 | 54 | 9.5 |
| tanh | 49.0 | 7,450 | 55 | 9.5 |
| softmax | 173.2 (interval 121.8) | 18,041 | 446 | 1.9 |

CarbonPATH run (`dense_rule4ml_activations` graph, profile `rule4ml_fpga_ops_v3`, architecture
`cfg/examples/sa_fpga_architecture.json`): relu 0.0 ns, sigmoid 191.07 ns, tanh 193.5 ns, softmax 177.1 ns.
I reproduced it on the remote machine:

| Op | Shape | Board (nearest) | Precision | Rule4ML cycles | FPGA clock | Time |
| --- | --- | --- | --- | --- | --- | --- |
| sigmoid | 8x32 | pynq-z2 | `ap_fixed<8, 2>` | 57.32 | 300 MHz | 191.07 ns |
| tanh | 8x32 | pynq-z2 | `ap_fixed<9, 3>` | 58.05 | 300 MHz | 193.5 ns |
| softmax | 8x16 | pynq-z2 | `ap_fixed<9, 3>` | 53.13 | 300 MHz | 177.1 ns |

- The clock comes from `frequency_hz` of the FPGA chiplet (300 MHz). Rule4ML does not get it. At 100 MHz the sigmoid would take 573 ns.
- The board is the nearest board to the chiplet (10,000 CLBs gives pynq-z2). The precision comes from the ATLAS tensor.
- Repro script: `script/rule4ml_workload_repro.py` (run from the repo root with `PYTHONPATH=.`).

Findings:

1. The element count is all that matters for an activation (see "How the model sees a tensor").
2. Cycles grow faster than linear with elements for sigmoid and tanh: 16 -> 0.34, 64 -> 5.6,
   256 -> 48.8, 1024 -> 359.7. A table lookup should not do this. The Resource strategy gives about 3x the
   cycles (149 at 256). Reuse factor changes cycles a little (49.6 at rf 4, 76.4 at rf 64).
3. Softmax cycles and DSP grow with elements (DSP 116 at 16, 446 at 256, 839 at 1024).
4. Sigmoid DSP does not grow with elements (65 at 16, 35 at 1024), and relu uses 63 DSP and 5.5k LUT
   for 0 cycles. A ReLU needs no DSP. So the DSP and LUT numbers for activations are model
   offsets, not a physical count. Hypothesis: the training networks were mostly dense layers, so the model learned a floor.
   Not verified.
5. The software versions change the result a lot. Sigmoid, 256 elements: hls4ml 1.1.0 + Vivado
   2024.2 -> 48.8 cycles, 54 DSP. hls4ml 0.8.1 + Vivado 2019.1 -> 159.2 cycles, 333 DSP.
   hls4ml 1.1.0 + Vivado 2019.1 -> 64.7 cycles, 152 DSP. The last pair may not exist in the training data.
6. Precision and board matter. Sigmoid at `ap_fixed<2, 1>`: 38 cycles, 4 DSP. At pynq-z2: 62.5 cycles,
   55 BRAM. Precision `ap_fixed<8, 3>` and `<8, 4>` give identical results.

Conclusion for now: use the cycle counts for sigmoid, tanh, and softmax as rough, size-dependent
estimates. Do not use the DSP and LUT numbers for a ReLU. Treat the 191 ns as 57 predicted cycles at an assumed 300 MHz clock, not as a measured kernel latency.

## Fused versus unfused sigmoid and tanh (experiment)

Data: `docs/rule4ml_fused_vs_unfused.csv`. Script: `script/rule4ml_unfused_experiment.py`. Profile `rule4ml_fused_dummy_v0`.
All 8 on/off combinations of fusing relu, sigmoid, and tanh, on 4 boards and 3 FPGA clocks (200, 300, 400 MHz).
Total latency in ns at a 300 MHz FPGA clock:

| Variant | pynq-z2 | zcu102 | alveo-u200 | alveo-u250 |
| --- | --- | --- | --- | --- |
| unfused | 1640 | 1514 | 1526 | 1401 |
| fused relu | 1626 | 1500 | 1512 | 1383 |
| fused sigmoid | 1469 | 1389 | 1399 | 1309 |
| fused tanh | 1466 | 1386 | 1397 | 1308 |
| fused relu + sigmoid | 1455 | 1375 | 1385 | 1290 |
| fused relu + tanh | 1452 | 1372 | 1382 | 1290 |
| fused sigmoid + tanh | 1295 | 1261 | 1270 | 1215 |
| fused all three | 1281 | 1247 | 1255 | 1197 |

- Fusing sigmoid alone saves about 171 ns, tanh about 174 ns, ReLU about 14 ns (pynq-z2). The savings add up.
- The fused cost is a dummy value (10 SA cycles for each SA width of elements). The unfused cost comes from Rule4ML.
- Rule4ML gives no energy. The FPGA activations get 0 pJ here, so energy is not comparable.
- Break-even: the unfused sigmoid costs about 211 ns (191 ns compute and about 20 ns movement). Fusing wins while the
  fused cost stays below that. The dummy 10 cycles gives 40 ns. Published designs (tables, rational approximation) need 1 to 5 cycles
  for each value. Not tested by a sweep yet.

## Accuracy and statements by the authors

wa-hls4ml (Hawks et al. 2026; abstract, poster, dataset card):

- GNN and transformer surrogate models. The abstract says the predictions for the 75th percentile are within several
  percent of the synthesized resources on the synthetic test set. The cycle accuracy of the shipped GNN was not read.
- The poster says one model predicts 6 values together. The shipped weights are one GNN for each output.
- The targets are the synthesized values: latency from C synthesis, resources from logic synthesis (dataset card).
- Planned: larger networks, skip connections, more hardware settings such as reuse factors, more samples.
- The dataset card warns: most samples are small dense networks without activations; use weights at training time.
- Not read: the full paper (ACM / arXiv 2511.05615) and the training details of the GNN.

2024 Rule4ML paper (MLP predictor):

- Latency means clock cycles for one inference of the whole network. Fixed 10 ns clock. Fully connected networks only.
- Features: counts of layer types and activations, average dense-layer sizes and reuse, fixed-point operation counts times precision.
- Resources trained as a percent of the board, capped at 200%. Cycles trained as an absolute value.
- Cycle error on the test set: median about 40, mean about 60, worst about 500. On the benchmarks, 85% of errors are below 100 cycles.
- Resource error: 81% of predictions are within 10%. BRAM 80%, DSP 89%, FF 88%, LUT 66%.
- The predictors tend to overestimate resources. Vivado HLS overestimates LUTs.
- The benchmarks are dense networks with ReLU and Softmax (one Sigmoid in Top Quarks). Tanh and standalone activations
  are never tested. Benchmark boards: ZCU102 and Pynq-Z2.
- Softmax correlates with DSP use even though HLS4ML builds it from lookup tables.
- Warnings from the authors: odd results at 2-bit and 8-bit; mostly large boards and low precision, so reuse factor has a weak effect;
  FF is hard to predict; a 5 ns clock raised FF use a lot.

## Future enhancements

From the 2024 paper and the README:

1. More clock periods. The new data has 5 ns and 10 ns, but the shipped model has no clock input.
2. More architectures. Done in part: wa-hls4ml has conv1d and conv2d.
3. Implementation (place-and-route) reports, not only synthesis. Not done. HLS overestimates LUTs.
4. A bigger dataset. Done: wa-hls4ml.
5. Known problem: some training networks gave odd results (reuse factor up, DSP not down) at 2-bit and 8-bit.

A lead for us: `datasets/analysis/` in the Rule4ML repo has scripts for single-operation data
(`rank_single_op.py`, `score_op.py`, `op_depth_grid.py`). They may help explain the single-activation predictions. Not checked.

## Caveats for CarbonPATH

| # | Caveat | Ignore? |
| --- | --- | --- |
| 1 | Dense and activation data is `io_parallel` (all values at once); the systolic array streams its data | open |
| 2 | Clock: data at 10 ns (2_20 subset) and 5 ns (others). The model has no clock input. We divide by the chiplet clock, which is an extrapolation. | open |
| 3 | Most training data is small dense networks without activations. A standalone activation model is out of distribution (possible cause of the 191 ns sigmoid). | open |
| 4 | Rule4ML sees only the element count of a tensor, not its shape | open |
| 5 | The hls4ml and Vivado version pair changes cycles by up to 3x; the code default is 1.1.0 + 2024.2 | open |
| 6 | ReLU gets 0 cycles but about 5.5k LUT and 63 DSP; resource numbers have an offset | open |
| 7 | Tensor shapes may fall outside the training ranges (8 to 128 neurons in the new data; 16 to 1,024 inputs in the paper) | open |
| 8 | Leaky ReLU and thresholded ReLU are not in the training data (no predictor) | open |
| 9 | The 2024 paper's cycle error (40 to 60 cycles) is as large as the predicted activation latency (57 cycles). The GNN cycle error was not read. | open |
| 10 | No standalone activation, and no tanh, in the benchmarks; sigmoid only once | open |
| 11 | Resources are trained as a percent of the board and tend to be overestimated | open |
| 12 | Board coverage: the 2_20 subset has 3 boards; most other data is alveo-u250; alveo-u200 only in the 2_20 subset and the exemplar set | open |
| 13 | Most synthetic data uses the Resource strategy; CarbonPATH uses Latency by default | open |
| 14 | Latency comes from the C synthesis report (average of min and max cycles). Resources come from logic synthesis. No simulation, no place-and-route. | open |
| 15 | Rule4ML says nothing about a GEMM done on a systolic array; it was never trained on one | open |

## Not verified

- Which subsets of wa-hls4ml the shipped GNN weights were trained on.
- Where the 2_20 samples store the Vivado version (the top-level field is empty).
- The `io_type` of the subsets other than 2_20 (README says io_parallel for fully connected, io_stream for conv).
- The pragmas of the training versions (hls4ml 0.8.1 and 1.1.0); I read a newer fork.
- The cycle accuracy of the GNN, and the full wa-hls4ml paper.
- The `train_3layer` and validation files.
