"""Rule4ML predictions for each activation, on each board, for many tensor sizes.

Run from the repo root on the machine that has rule4ml:
    PYTHONPATH=. python script/rule4ml_board_dimension_sweep.py > docs/rule4ml_board_dimension_sweep.csv

Rule4ML sees only the element count of a tensor, so each size is one shape.
Time = cycles x 10 ns, the clock period of the Rule4ML data.
"""
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import keras
from rule4ml.models.wrappers import MultiModelWrapper

from system.utils.Rule4MlBoards import board_chiplet_fields

BOARDS = ["pynq-z2", "zcu102", "alveo-u200", "alveo-u250"]
ACTIVATIONS = ["relu", "sigmoid", "tanh", "softmax"]
SHAPES = [
    (4, 4), (4, 8), (8, 8), (8, 16), (8, 32), (8, 64), (8, 128), (8, 256),
    (16, 256), (32, 256), (64, 256), (128, 128),
]
STRATEGY = "latency"
PRECISION = "ap_fixed<8, 3>"
REUSE = 1
HLS4ML = "1.1.0"
VIVADO = "2024.2"
CLOCK_NS = 10.0
TRAINED_MAX = 1024  # largest input size in the 2024 paper's range

wrapper = MultiModelWrapper()
wrapper.load_default_models()


def model(activation, shape):
    inputs = keras.Input(shape=shape)
    return keras.Model(inputs, keras.layers.Activation(activation)(inputs))


columns = ("board;activation;shape;elements;cycles;interval;time_ns_10ns;lut;ff;dsp;bram;"
           "lut_pct;ff_pct;dsp_pct;bram_pct;beyond_1024").split(";")
print(";".join(columns))
for board in BOARDS:
    cap = board_chiplet_fields(board)
    for activation in ACTIVATIONS:
        for shape in SHAPES:
            elements = shape[0] * shape[1]
            frame = wrapper.predict(
                model(activation, shape),
                hls_configs=[{"board": board, "model": {
                    "strategy": STRATEGY, "precision": PRECISION, "reuse_factor": REUSE}}],
                hls4ml_versions=[HLS4ML], vivado_versions=[VIVADO],
            )
            row = frame.iloc[0]
            cycles = float(row["CYCLES"])
            lut, ff, dsp, bram = (float(row[k]) for k in ("LUT", "FF", "DSP", "BRAM"))
            print(";".join(str(x) for x in (
                board, activation, f"{shape[0]}x{shape[1]}", elements,
                round(cycles, 2), round(float(row["INTERVAL"]), 2), round(cycles * CLOCK_NS, 1),
                round(lut), round(ff), round(dsp, 1), round(bram, 1),
                round(100 * lut / cap["luts"], 1), round(100 * ff / cap["ffs"], 1),
                round(100 * dsp / cap["dsps"], 1), round(100 * bram / cap["brams"], 1),
                "yes" if elements > TRAINED_MAX else "no")), flush=True)
