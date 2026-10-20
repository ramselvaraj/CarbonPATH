# Rule4ML Assumptions

Living document. Source: Rahimifar et al. 2024 (Zotero) and the Rule4ML repo
(`data_gen/nn_synth.py`, `data_gen/utils.py`). Update it as each analysis task finishes.

## Inputs and outputs

- Inputs: Keras workload, board, strategy, precision, reuse factor. Frequency is not an input.
- Outputs: LUT, DSP, BRAM, FF, and `CYCLES` / `INTERVAL`. These are cycle counts, not time.
- Time = cycles x the design clock period (Fmax for the design).

## Training data (task 7: kernel I/O assumptions)

| Item | Value | Source |
| --- | --- | --- |
| I/O type | `io_parallel` only | `nn_synth.py:86` |
| Backend | `VivadoAccelerator`, Vivado 2019.1 | `nn_synth.py:88-89` |
| Clock period | 10 ns, fixed | `nn_synth.py:85` |
| Boards | pynq-z2, zcu102, alveo-u200 (the repo board table also has alveo-u250) | `nn_synth.py:83` |
| Strategy | Latency, Resource (global) | `nn_synth.py:82` |
| Precision | `ap_fixed<2,1>`, `<8,3>`, `<8,4>`, `<16,6>` (global) | `nn_synth.py:74-81` |
| Reuse factor | 1 to 64, powers of 2 (per layer) | `nn_synth.py:73` |
| Input size | 16 to 1024, powers of 2 | paper, Table 1 |
| Output size | 1 to 200 | paper, Table 1 |
| Layers / neurons | 2 to 20 layers, 2 to 4096 neurons | paper, Table 1 |
| Activations | ReLU, Tanh, Sigmoid, Softmax | paper, Table 1 |
| Flow | C synthesis report only (cosim, csim, validation, vsynth off in the benchmark call; original dataset run not traced) | `nn_synth.py:747-753` |
| Latency source | `BestLatency` / `WorstLatency` from the C synthesis report | `data_gen/utils.py:164-168` |

Consequences:

- Every input is available at once. There is no streaming, FIFO, or input buffering in the data.
- Results are HLS estimates, not place-and-route results.
- The paper does not state a batch size.

## Pragmas and parallelism (task 6)

- Rule4ML has no pragma input. hls4ml writes the pragmas from strategy, reuse factor, and precision.
- Training data used two hls4ml versions: 0.8.1 and 1.1.0 (`data_parser.py:47`). The version is a model feature.
- hls4ml is not installed with Rule4ML. I read the fork `atlas/hls4ml-gemm` on the remote machine
  (`itsruthwik/hls4ml-gemm`). It is newer than the training versions, so treat these pragmas as
  "very likely the same", not verified.

| Level | Pragma | Where |
| --- | --- | --- |
| Whole model | `#pragma HLS PIPELINE` (default style, optional II) | `writer/vivado_writer.py:436` |
| Model I/O (`io_parallel`) | `ARRAY_PARTITION` on input and output, `INTERFACE ap_vld` | `writer/vivado_writer.py:443-450` |
| Dense, Latency strategy | `PIPELINE II=reuse_factor`, arrays `complete`, `ALLOCATION mul limit=multiplier_limit` | `nnet_dense_latency.h:26-33` |
| Dense, Resource strategy | `ARRAY_RESHAPE weights block`, `PIPELINE II=1 rewind`, weights in ROM | `nnet_dense_resource.h:29-56` |
| Activation | `PIPELINE` per element loop; table lookup for sigmoid and tanh | `nnet_activation.h` |
| Softmax | `PIPELINE`, exp and inverse tables, `array_partition complete` | `nnet_activation.h:176-270` |

What this means:

- Unrolling is not a free choice. Reuse factor sets `II` (Latency) or the block size (Resource). All of this comes from the config, not from the user.
- The whole model is one pipelined function. The reported cycles are the pass latency for one input, with all inputs read at once (`ap_vld`).
- Sigmoid and tanh use a table (`table_size` 1024 by default), so they cost BRAM or LUTs for the table, not DSPs for the math. Check this in item 8.
- Weights are constants in the design. The model assumes weights are stored on chip. It does not model weight loading.
- Status of the pragmas: they enhance the design (pipeline, partition, unroll). They are not a bare baseline.

## Activation predictions (task 8)

Data: `docs/rule4ml_activation_sweep.csv`. Script: `script/rule4ml_activation_sweep.py` (run on the
remote machine with `~/work/venv-cp-rule4ml`). Default settings: zcu102, Latency, `ap_fixed<16, 6>`,
reuse 1, hls4ml 1.1.0, Vivado 2024.2. Each row is one single-activation Keras model.

| Op, 256 elements | Cycles | LUT | DSP | BRAM |
| --- | --- | --- | --- | --- |
| relu | 0.0 | 5,504 | 63 | 2.6 |
| sigmoid | 48.8 | 7,454 | 54 | 9.5 |
| tanh | 49.0 | 7,450 | 55 | 9.5 |
| softmax | 173.2 (interval 121.8) | 18,041 | 446 | 1.9 |

Earlier CarbonPATH run (`dense_rule4ml_activations` graph, profile `rule4ml_fpga_ops_v3`, architecture
`cfg/examples/sa_fpga_architecture.json`): relu 0.0 ns, sigmoid 191.07 ns, tanh 193.5 ns, softmax 177.1 ns.
I reproduced it on the remote machine. The 191 ns is fully explained:

| Op | Shape | Board (nearest) | Precision | Rule4ML cycles | FPGA clock | Time |
| --- | --- | --- | --- | --- | --- | --- |
| sigmoid | 8x32 | pynq-z2 | `ap_fixed<8, 2>` | 57.32 | 300 MHz | 191.07 ns |
| tanh | 8x32 | pynq-z2 | `ap_fixed<9, 3>` | 58.05 | 300 MHz | 193.5 ns |
| softmax | 8x16 | pynq-z2 | `ap_fixed<9, 3>` | 53.13 | 300 MHz | 177.1 ns |

- The clock comes from `frequency_hz` of the FPGA chiplet in the architecture file (300 MHz).
  Rule4ML does not get it. Its data is at 10 ns, so at 100 MHz the sigmoid would take 573 ns.
- The board is the nearest board to the chiplet (10,000 CLBs gives pynq-z2). The precision comes
  from the ATLAS tensor. My sweep used zcu102 and `ap_fixed<16, 6>`, so its cycle counts differ.
- Sigmoid on pynq-z2 gets 53.7 BRAM (of 140). The table lookup shows up as BRAM there.
- Repro script: `script/rule4ml_workload_repro.py` (run from the repo root with `PYTHONPATH=.`).

Findings:

1. Rule4ML sees only the element count. `8x32`, `256`, and `(256,)` give identical numbers.
2. Cycles grow faster than linear with elements for sigmoid and tanh: 16 -> 0.34, 64 -> 5.6,
   256 -> 48.8, 1024 -> 359.7. A table lookup should not do this. The Resource strategy gives about 3x the
   cycles (149 at 256). Reuse factor changes cycles a little (49.6 at rf 4, 76.4 at rf 64).
3. Softmax cycles and DSP grow with elements (DSP 116 at 16, 446 at 256, 839 at 1024). That fits
   a fully unrolled design with one set of multipliers for each element.
4. Sigmoid DSP does not grow with elements (65 at 16, 35 at 1024), and relu uses 63 DSP and 5.5k LUT
   for 0 cycles. A ReLU needs no DSP. So the DSP and LUT numbers for activations are model
   offsets, not a physical count. My hypothesis: all training networks had dense layers, so the
   model learned a floor. Not verified.
5. The software versions change the result a lot. Sigmoid, 256 elements: hls4ml 1.1.0 + Vivado
   2024.2 -> 48.8 cycles, 54 DSP. hls4ml 0.8.1 + Vivado 2019.1 -> 159.2 cycles, 333 DSP.
   hls4ml 1.1.0 + Vivado 2019.1 -> 64.7 cycles, 152 DSP. The last pair may not exist in the training
   data, so it may be an extrapolation. Not checked.
6. Precision and board matter. Sigmoid at `ap_fixed<2, 1>`: 38 cycles, 4 DSP. At pynq-z2: 62.5 cycles,
   55 BRAM. Precision `ap_fixed<8, 3>` and `<8, 4>` give identical results.

Conclusion for now: use the cycle counts for sigmoid, tanh, and softmax as rough, size-dependent
estimates. Do not use the DSP and LUT numbers for a ReLU. Treat the 191 ns as 57 predicted cycles at an assumed 300 MHz clock, not as a measured kernel latency.

## What the paper says about accuracy and scope (task 4: paper read in full)

Source: Rahimifar et al. 2024, sections 3 to 6.

- Features: counts of layers and activation types, average dense-layer size and reuse, and counts of fixed-point
  operations (add, multiply, logical, lookup) times the precision. Tensor shapes are not features.
  This is why only the element count matters (finding 1 above).
- Resources are trained as a percentage of the board, capped at 200%. Cycles are trained as an absolute error.
- Latency means clock cycles for one inference of the whole network.
- Cycle error on the test set: median about 40, mean about 60, worst about 500 cycles. On the benchmarks, 85% of
  errors are below 100 cycles. The paper calls this "less than 1 us at 10 ns".
- Resource error: 81% of predictions are within 10%. By resource: BRAM 80%, DSP 89%, FF 88%, LUT 66%.
- The predictors tend to overestimate resources. Vivado HLS also overestimates LUTs.
- The benchmarks are only dense networks with ReLU and Softmax (Top Quarks has one Sigmoid). Tanh and
  standalone activations are never tested. Benchmark boards are ZCU102 and Pynq-Z2. Alveo-U200 was left out
  as "similar to ZCU102".
- Only FCNNs. The authors expect CNNs to use more DSPs.
- Softmax correlates with DSP use even though HLS4ML builds it from lookup tables. This matches our softmax DSP count.
- Vivado fails on layers over 4096 parameters in `Latency` strategy (MNIST). Not a limit for our activations.
- A 5 ns clock raised FF use a lot. A 10 ns clock is the only one in the data.

What it means for us: the sigmoid prediction of 57 cycles is as large as the typical error of the cycle model
(40 to 60 cycles). Treat the activation cycle counts as order of magnitude, not exact.

## Future enhancements stated by Rule4ML

From the paper (Discussion) and the repo README ("Limitations"):

1. More clock periods. Today it is 10 ns only. A 5 ns clock raised FF use a lot. They plan a larger dataset with a range of clock periods. This matters for item 10.
2. More architectures: FCNN, CNN, RNN, with more layer types, larger layers, more activations, other precisions, and higher reuse factors.
3. Implementation reports from Vivado, not only the HLS synthesis report. HLS overestimates LUTs, and the real count after implementation is lower.
4. A bigger training set. The README says newer predictors use the `wa-hls4ml` dataset (Hugging Face: `fastmachinelearning/wa-hls4ml` and `wa-hls4ml-projects`). I have not checked what is in it.
5. Known problem: some training networks gave odd results (reuse factor up, DSP not down) at 2-bit and 8-bit precision.

Not stated by the authors, but a lead for us: `datasets/analysis/` in the repo has scripts for single-op data (`rank_single_op.py`, `score_op.py`, `op_depth_grid.py`). The 1.1.0 data may contain single-operation networks. That would help explain the single-activation predictions. Not checked yet.

## Caveats for CarbonPATH (task 9)

| # | Caveat | Ignore? |
| --- | --- | --- |
| 1 | Data is `io_parallel`; the systolic array streams its data | open |
| 2 | Fixed 10 ns clock; ASIC systolic array runs at a different frequency | open |
| 3 | Training precisions differ slightly from the paper's 2, 8, 16 bit | open |
| 4 | Training data is dense networks; a single activation model is out of distribution (possible cause of 191 ns sigmoid) | open |
| 7 | Rule4ML sees only the element count, not the tensor shape | open |
| 8 | The hls4ml and Vivado version pair changes cycles by up to 3x; the code default is 1.1.0 + 2024.2 | open |
| 9 | ReLU gets 0 cycles but about 5.5k LUT and 63 DSP; resource numbers have an offset | open |
| 5 | Tensor shapes may fall outside the training ranges | open |
| 6 | Leaky ReLU and thresholded ReLU are not in the training activations | open |
| 10 | The typical cycle error (40 to 60 cycles) is as large as the predicted activation latency (57 cycles) | open |
| 11 | No standalone activation, and no tanh, in the paper benchmarks; sigmoid only once | open |
| 12 | Resources are percentages of the board, capped at 200%, and tend to be overestimated | open |
| 13 | Alveo-U200 was not validated; Alveo-U250 is not in the training data | open |
| 14 | Only FCNNs; Rule4ML says nothing about a GEMM done on a systolic array | open |
