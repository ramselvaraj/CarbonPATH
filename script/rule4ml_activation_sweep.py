import os, sys, itertools, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import keras
from rule4ml.models.wrappers import MultiModelWrapper

w = MultiModelWrapper()
w.load_default_models()


def model(act, shape):
    i = keras.Input(shape=shape)
    return keras.Model(i, keras.layers.Activation(act)(i), name=f"{act}_{'x'.join(map(str, shape))}")


def run(act, shape, board="zcu102", strategy="latency", precision="ap_fixed<16, 6>",
        reuse=1, hv="1.1.0", vv="2024.2"):
    f = w.predict(
        model(act, shape),
        hls_configs=[{"board": board, "model": {"strategy": strategy, "precision": precision,
                                                "reuse_factor": reuse}}],
        hls4ml_versions=[hv], vivado_versions=[vv],
    )
    r = f.iloc[0]
    g = lambda k: float(r[k]) if k in r else float("nan")
    print(";".join(str(x) for x in (act, "x".join(map(str, shape)), board, strategy, precision,
          reuse, hv, vv, g("CYCLES"), g("INTERVAL"), g("LUT"), g("FF"), g("DSP"), g("BRAM"))),
          flush=True)


print("act;shape;board;strategy;precision;reuse;hls4ml;vivado;cycles;interval;LUT;FF;DSP;BRAM")
acts = ["relu", "sigmoid", "tanh", "softmax"]
# A: the CarbonPATH shape, default settings
for a in acts:
    run(a, (8, 32))
# B: shape sweep (flat sizes inside the 16..1024 training range)
for a in acts:
    for s in [(16,), (64,), (256,), (1024,), (8, 16), (8, 32), (16, 64)]:
        run(a, s)
# C: reuse factor, strategy
for a in acts:
    for st in ["latency", "resource"]:
        for rf in [1, 4, 16, 64]:
            run(a, (256,), strategy=st, reuse=rf)
# D: precision, board, versions
for a in acts:
    for p in ["ap_fixed<2, 1>", "ap_fixed<8, 3>", "ap_fixed<8, 4>", "ap_fixed<16, 6>"]:
        run(a, (256,), precision=p)
    for b in ["pynq-z2", "zcu102", "alveo-u200"]:
        run(a, (256,), board=b)
    for hv, vv in [("0.8.1", "2019.1"), ("1.1.0", "2024.2"), ("1.1.0", "2019.1")]:
        try:
            run(a, (256,), hv=hv, vv=vv)
        except Exception as e:
            print(f"{a},256,ERR {hv} {vv}: {str(e)[:80]}")
