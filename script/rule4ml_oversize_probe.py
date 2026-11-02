"""Probe Rule4ML with designs far larger than the training range and the board.

Run from the repo root on the machine that has rule4ml:
    PYTHONPATH=. python script/rule4ml_oversize_probe.py > docs/rule4ml_oversize_probe.csv

Questions:
  1. Does Rule4ML predict more than 100% of a board (resources above the maximum)?
  2. What does it do at very large sizes: error, NaN, negative values, or a plateau?

Cases: Dense layers (square, up to 4096 x 4096) and activations (up to 2^22 elements).
All values are ap_fixed<8, 3>, Latency strategy, reuse factor 1, 10 ns clock.
"""
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import keras
from rule4ml.models.wrappers import MultiModelWrapper

from system.utils.Rule4MlBoards import board_chiplet_fields

BOARDS = ["pynq-z2", "zcu102", "alveo-u200", "alveo-u250"]
DENSE_SIZES = [64, 256, 1024, 2048, 4096]
ACTIVATIONS = ["relu", "sigmoid", "tanh", "softmax"]
ACTIVATION_ELEMENTS = [2**12, 2**14, 2**16, 2**18, 2**20, 2**22]
PRECISION = "ap_fixed<8, 3>"
CLOCK_NS = 10.0
TRAINED_MAX = 1024

wrapper = MultiModelWrapper()
wrapper.load_default_models()


def dense_model(size):
    inputs = keras.Input(shape=(size,))
    return keras.Model(inputs, keras.layers.Dense(size)(inputs))


def activation_model(activation, elements):
    inputs = keras.Input(shape=(elements // 1024, 1024))
    return keras.Model(inputs, keras.layers.Activation(activation)(inputs))


def cases():
    for size in DENSE_SIZES:
        yield "dense", f"{size}x{size}", size * size, (lambda s=size: dense_model(s))
    for activation in ACTIVATIONS:
        for elements in ACTIVATION_ELEMENTS:
            yield (activation, f"{elements // 1024}x1024", elements,
                   (lambda a=activation, e=elements: activation_model(a, e)))


columns = ("board;kind;shape;elements;cycles;interval;time_ns;lut;ff;dsp;bram;"
           "lut_pct;ff_pct;dsp_pct;bram_pct;over_100pct;beyond_1024;note").split(";")
print(";".join(columns))
for board in BOARDS:
    cap = board_chiplet_fields(board)
    for kind, shape, elements, build in cases():
        try:
            frame = wrapper.predict(
                build(),
                hls_configs=[{"board": board, "model": {
                    "strategy": "latency", "precision": PRECISION, "reuse_factor": 1}}],
                hls4ml_versions=["1.1.0"], vivado_versions=["2024.2"],
            )
            row = frame.iloc[0]
            cycles = float(row["CYCLES"])
            lut, ff, dsp, bram = (float(row[k]) for k in ("LUT", "FF", "DSP", "BRAM"))
            pct = (100 * lut / cap["luts"], 100 * ff / cap["ffs"],
                   100 * dsp / cap["dsps"], 100 * bram / cap["brams"])
            values = (cycles, float(row["INTERVAL"]), lut, ff, dsp, bram)
            note = ""
            if any(v != v for v in values):
                note = "nan"
            elif any(v < 0 for v in values):
                note = "negative"
            print(";".join(str(x) for x in (
                board, kind, shape, elements, round(cycles, 2), round(float(row["INTERVAL"]), 2),
                round(cycles * CLOCK_NS, 1), round(lut), round(ff), round(dsp, 1), round(bram, 1),
                *(round(p, 1) for p in pct), "yes" if max(pct) > 100 else "no",
                "yes" if elements > TRAINED_MAX else "no", note)), flush=True)
        except Exception as error:  # an edge condition: record it, keep going
            message = f"{type(error).__name__}: {error}".replace(";", ",").replace("\n", " ")[:120]
            print(";".join(str(x) for x in (
                board, kind, shape, elements, *([""] * 11), "yes" if elements > TRAINED_MAX else "no",
                message)), flush=True)
