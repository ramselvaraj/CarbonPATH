#!/usr/bin/env python3
"""Write the two large ATLAS graph dumps by editing layers of an existing dump.

The Keras/HGQ2 flow is not needed: each layer copies the schema of a layer in
``dense_rule4ml_activations.graph_dump.json`` and changes names, shapes and sizes.
The C++ configuration strings are dropped; CarbonPATH does not read them.

    wide_k_mlp   M=32 rows. Two feed-forward blocks (64 -> 1024 -> relu -> 64) make
                 the GEMMs large (K up to 1024). The 1024-wide ReLU tensors hold
                 32768 values, far above Rule4ML's trained range of 1024; the sweep
                 shows ReLU stays cheap there. Sigmoid, tanh and softmax tensors
                 hold at most 2048 values.
    row_split    M=128 rows and narrow layers. Each activation tensor has 2048
                 values (softmax 2048), so a row split over FPGA chiplets halves it.

Run from the repository root:  python3 cfg/examples/atlas/synthetic_graphs.py
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = json.loads((HERE / "dense_rule4ml_activations.graph_dump.json").read_text())
BY_CLASS = {}
for layer in TEMPLATE:
    key = layer["attrs"].get("activation", layer["class"])
    BY_CLASS.setdefault(key if layer["class"] == "Activation" else layer["class"], layer)

DROP = ("config_cpp", "function_cpp", "include_header")


def clean(layer):
    for key in DROP:
        layer["attrs"].pop(key, None)
    return layer


def build(spec, m):
    """spec: [(name, N, activation or None)], input is (m, K0)."""
    k0 = spec[0][3]
    layers, index = [], 1
    inp = clean(copy.deepcopy(BY_CLASS["Input"]))
    inp["output"]["shape"] = [m, k0]
    inp["attrs"]["input_shape"] = [m, k0]
    inp["index"] = inp["attrs"]["index"] = index
    layers.append(inp)
    previous, previous_shape, k = "input_layer", [m, k0], k0
    for name, n, activation, _ in spec:
        index += 1
        gemm = clean(copy.deepcopy(BY_CLASS["Gemm"]))
        gemm.update(index=index, name=f"gemm_{name}")
        gemm["attrs"].update(
            index=index, n_in=k, n_out=n, n_patches=m, gemm_m=m,
            _gemm_output_shape=[m, n],
        )
        gemm["inputs"] = [{"name": previous, "shape": [m, k],
                           "precision": gemm["inputs"][0]["precision"]}]
        gemm["output"]["shape"] = [m, n]
        gemm["weights"][0].update(name=f"weight{index}", shape=[n, k])
        gemm["weights"][1].update(name=f"bias{index}", shape=[n])
        layers.append(gemm)
        previous, previous_shape, k = name, [m, n], n
        if activation:
            index += 1
            template = BY_CLASS["Softmax" if activation == "softmax" else activation]
            act = clean(copy.deepcopy(template))
            act_name = "softmax" if activation == "softmax" else f"{name}_{activation}"
            act.update(index=index, name=act_name)
            act["attrs"].update(name=act_name, index=index, n_in=m * n, inputs=[name])
            act["inputs"] = [{"name": name, "shape": [m, n],
                              "precision": act["inputs"][0]["precision"]}]
            act["output"]["shape"] = [m, n]
            layers.append(act)
            previous = act_name
    return layers


# (name, N, activation, K) -- K of the first layer sets the input width.
WIDE_K_MLP = [
    ("up1", 1024, "relu", 64),
    ("down1", 64, "sigmoid", None),
    ("up2", 1024, "relu", None),
    ("down2", 64, "tanh", None),
    ("score", 32, "softmax", None),
    ("head", 16, None, None),
]
ROW_SPLIT = [
    ("stem", 16, "relu", 64),
    ("mid", 16, "sigmoid", None),
    ("mix", 16, "tanh", None),
    ("score", 16, "softmax", None),
    ("head", 8, None, None),
]

if __name__ == "__main__":
    for file_name, spec, m in (
        ("wide_k_mlp", WIDE_K_MLP, 32),
        ("row_split", ROW_SPLIT, 128),
    ):
        layers = build(spec, m)
        path = HERE / f"{file_name}.graph_dump.json"
        path.write_text(json.dumps(layers, indent=1) + "\n")
        print(path.name, len(layers), "layers")
