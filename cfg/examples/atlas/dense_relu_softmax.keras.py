#!/usr/bin/env python3
"""ATLAS dense ReLU + Softmax mixed chain for CarbonPATH.

Input(8, 16) -> QDense(16) -> relu -> QDense(32) -> Softmax(axis=-1) -> QDense(8)

Graph only: dumps the pre-codegen hls4ml ModelGraph, no code or EDA tool.
Run with the Keras-3 / HGQ2 frontend (.venv).
"""
import sys
from pathlib import Path

# Find the ATLAS repo root (holds run_atlas_flow.py) by searching upward.
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "run_atlas_flow.py").exists())))
from run_atlas_flow import run_atlas_flow  # noqa: E402

import keras
from hgq.layers import QDense

HLS_CONFIG = {
    "Model": {"Precision": {"default": "fixed<16,6>"}, "Strategy": "GEMM"},
}


def build_model():
    inp = keras.layers.Input((8, 16))
    x = QDense(16, name="stem")(inp)
    x = keras.layers.Activation("relu", name="stem_relu")(x)
    x = QDense(32, name="expand")(x)
    x = keras.layers.Softmax(axis=-1, name="softmax")(x)
    out = QDense(8, name="head")(x)
    return keras.Model(inp, out)


if __name__ == "__main__":
    run_atlas_flow(build_model(), "outputs/carbonpath_dense_relu_softmax",
                   hls_config=HLS_CONFIG, io_type="io_stream",
                   dump_graph=True, stop_after_convert=True)
