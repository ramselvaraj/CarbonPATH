#!/usr/bin/env python3
"""ATLAS dense-ReLU funnel for CarbonPATH.

Input(128, 128) -> QDense(1024) -> relu -> QDense(512) -> relu
                -> QDense(256) -> relu -> QDense(64)

Graph only: dumps the pre-codegen hls4ml ModelGraph, no code or EDA tool.
Run with the Keras-3 / HGQ2 frontend (.venv):

    .venv/bin/python example/carbonpath_dense_relu_funnel.py
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
    inp = keras.layers.Input((128, 128))
    x = QDense(1024, name="expand")(inp)
    x = keras.layers.Activation("relu", name="expand_relu")(x)
    x = QDense(512, name="contract1")(x)
    x = keras.layers.Activation("relu", name="contract1_relu")(x)
    x = QDense(256, name="contract2")(x)
    x = keras.layers.Activation("relu", name="contract2_relu")(x)
    out = QDense(64, name="head")(x)
    return keras.Model(inp, out)


if __name__ == "__main__":
    run_atlas_flow(build_model(), "outputs/carbonpath_dense_relu_funnel",
                   hls_config=HLS_CONFIG, io_type="io_stream",
                   dump_graph=True, stop_after_convert=True)
