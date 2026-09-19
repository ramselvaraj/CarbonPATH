# ATLAS Graph Fixture

This directory holds a raw ATLAS graph dump and the Keras model that produced
it. CarbonPATH consumes the dump directly through `AtlasGraphAdapter`; the Keras
source is committed for provenance and reproducibility.

## Contents

- `dense_relu_funnel.keras.py`: Keras/HGQ2 model source.
- `dense_relu_funnel.graph_dump.json`: raw ATLAS graph dump.

## Provenance

- ATLAS checkout: `atlas-working` (copy of the ATLAS `main` checkout used to
  generate the artifact).
- ATLAS revision: `9501bd5`.
- Generation mode: graph only
  (`run_atlas_flow(..., dump_graph=True, stop_after_convert=True)`). No
  synthesis, cosimulation, or EDA tool was run.

## Model

```text
Input(128,128)
-> QDense(1024) -> relu
-> QDense(512)  -> relu
-> QDense(256)  -> relu
-> QDense(64)
```

Expected operations: four `Gemm` nodes and three `relu` `Activation` nodes with
GEMM shapes `(128,128,1024)`, `(128,1024,512)`, `(128,512,256)`, `(128,256,64)`.

## Regenerating

Normal tests consume the committed dump and do not require ATLAS or Keras. To
regenerate, run the Keras source with the ATLAS Keras-3/HGQ2 virtual environment
and copy the emitted `graph_dump.json` back into this directory.
