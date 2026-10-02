#!/usr/bin/env python3
"""ATLAS graph-only source for CarbonPATH legacy workload 9.

Input(128, 256) -> QDense(512)

The layer has no bias or activation, matching workload 9's single GEMM.
Run this file from an ATLAS checkout so the upward search can find
``run_atlas_flow.py``.
"""

import sys
from pathlib import Path


sys.path.insert(
    0,
    str(
        next(
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "run_atlas_flow.py").exists()
        )
    ),
)
from run_atlas_flow import run_atlas_flow  # noqa: E402

import keras  # noqa: E402
from hgq.layers import QDense  # noqa: E402


HLS_CONFIG = {
    "Model": {"Precision": {"default": "fixed<16,6>"}, "Strategy": "GEMM"},
}


def build_model():
    inputs = keras.layers.Input((128, 256), name="workload_input")
    outputs = QDense(
        512,
        activation=None,
        use_bias=False,
        name="projection",
    )(inputs)
    return keras.Model(inputs, outputs, name="workload9_single_gemm")


if __name__ == "__main__":
    run_atlas_flow(
        build_model(),
        "outputs/carbonpath_workload9_single_gemm",
        hls_config=HLS_CONFIG,
        io_type="io_stream",
        dump_graph=True,
        stop_after_convert=True,
    )
