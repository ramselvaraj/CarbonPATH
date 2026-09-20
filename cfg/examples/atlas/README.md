# ATLAS Graph Fixtures

This directory holds raw ATLAS graph dumps and the Keras models that produced
them. CarbonPATH consumes the dumps directly through `AtlasGraphAdapter`; the
Keras sources are committed for provenance and reproducibility.

## Contents

- `dense_relu_funnel.keras.py`: Keras/HGQ2 model source.
- `dense_relu_funnel.graph_dump.json`: raw ATLAS graph dump.
- `dense_softmax.keras.py`: Keras/HGQ2 model source.
- `dense_softmax.graph_dump.json`: raw ATLAS graph dump.

## Provenance

- ATLAS checkout: `atlas-working` (copy of the ATLAS `main` checkout used to
  generate the artifacts).
- ATLAS revision: `9501bd5`.
- Generation mode: graph only
  (`run_atlas_flow(..., dump_graph=True, stop_after_convert=True)`). No
  synthesis, cosimulation, or EDA tool was run.

## dense_relu_funnel

```text
Input(128,128)
-> QDense(1024) -> relu
-> QDense(512)  -> relu
-> QDense(256)  -> relu
-> QDense(64)
```

Expected operations: four `Gemm` nodes and three `relu` `Activation` nodes with
GEMM shapes `(128,128,1024)`, `(128,1024,512)`, `(128,512,256)`, `(128,256,64)`.

## dense_softmax

```text
Input(8,16)
-> QDense(16) -> Softmax(axis=-1) -> QDense(4)
```

Expected operations: two `Gemm` nodes and one `Softmax` node with output shape
`(8,16)` and `attrs.axis == -1`.

## Regenerating

Normal tests consume the committed dumps and do not require ATLAS or Keras. To
regenerate, run the Keras source with the ATLAS Keras-3/HGQ2 virtual environment
and copy the emitted `graph_dump.json` back into this directory.
