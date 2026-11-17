"""Build one single-layer model with neurigami and print its estimate as JSON.

Run this with the Python of a neurigami checkout (it needs onnx, qonnx, amaranth
and scipy; CarbonPATH's own environment does not):

    <neurigami python> script/neurigami_single_layer.py \
        --home <neurigami checkout> --op relu --elements 64 --clock-ns 4.0 \
        [--param X] [--vector N] [--part xczu7ev]
        [--budget-lut N --budget-ff N --budget-bram36 N --budget-dsp N] [--work-dir DIR]

The model is one 8-bit Quant -> op -> 8-bit Quant chain over a [1, elements]
tensor ([1, vector] for softmax, whose ``rows`` = elements / vector rows stream through the same hardware). neurigami compiles it, and the script reads the area and timing that
neurigami's own estimator reports in settings.yaml. Nothing is synthesised.

The one-line JSON on stdout is {"no_fit": why} when no design fits the budget, else holds: elements, rows, lut, ff, bram36, dsp, uram,
interval (cycles per sample), first_latency (cycles), clock_ns, part.

An activation has no lanes to fold: neurigami streams it one element a cycle, so
its interval equals the element count and an ``interval`` knob would change nothing.
"""

import argparse
import json
import os
import sys
import tempfile


def _quant(name, source, target, scale, signed):
    import numpy as np
    from onnx import helper, numpy_helper

    initializers = [
        numpy_helper.from_array(np.array(scale, np.float32), name + "_scale"),
        numpy_helper.from_array(np.array(0, np.float32), name + "_zero"),
        numpy_helper.from_array(np.array(8, np.float32), name + "_bits"),
    ]
    node = helper.make_node(
        "Quant",
        [source, name + "_scale", name + "_zero", name + "_bits"],
        [target],
        name=name,
        domain="qonnx.custom_op.general",
        signed=signed,
        narrow=0,
        rounding_mode="ROUND",
    )
    return initializers, node


#: op name -> (ONNX op type, extra node attributes from the parameter, constant inputs from the parameter).
#: ``param`` is the op's one number: the slope of leaky_relu, the threshold of thresholded_relu, the max value of clipped_relu.
_OPS = {
    "relu": ("Relu", lambda param: {}, lambda param: []),
    "leaky_relu": ("LeakyRelu", lambda param: {"alpha": 0.01 if param is None else param}, lambda param: []),
    "thresholded_relu": ("ThresholdedRelu", lambda param: {"alpha": 1.0 if param is None else param}, lambda param: []),
    "sigmoid": ("Sigmoid", lambda param: {}, lambda param: []),
    "tanh": ("Tanh", lambda param: {}, lambda param: []),
    "hard_sigmoid": ("HardSigmoid", lambda param: {}, lambda param: []),
    "gelu": ("Gelu", lambda param: {}, lambda param: []),
    "clip": ("Clip", lambda param: {}, lambda param: [("lo", -1.0), ("hi", 1.0)]),
    # hls4ml's hard_tanh at its defaults is clip(x, -1, 1); ClippedReLU is clip(x, 0, max_value).
    "hard_tanh": ("Clip", lambda param: {}, lambda param: [("lo", -1.0), ("hi", 1.0)]),
    "clipped_relu": ("Clip", lambda param: {}, lambda param: [("lo", 0.0), ("hi", 6.0 if param is None else param)]),
    "softmax": ("Softmax", lambda param: {"axis": -1}, lambda param: []),
}
OPS = tuple(_OPS)
_OPSET = {"gelu": 20}


def single_layer_model(op, elements, param=None, vector=None):
    """A QONNX ModelProto of one ``op`` over ``elements`` 8-bit values.

    ``vector`` is the length of the last axis; only softmax reads it (it normalises over that axis). Every other
    op is elementwise and sees one flat vector of ``elements``.

    Input and output are both signed 8-bit at one scale. A standalone activation whose input and output types
    differ makes neurigami insert a requantizer that needs a scale-file directory a one-layer model does not
    have, so the types match.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    if op not in _OPS:
        raise SystemExit(f"op '{op}' is not modelled; known: {sorted(_OPS)}")
    onnx_type, attrs, constants = _OPS[op]
    width = vector if (op == "softmax" and vector) else elements
    init_in, quant_in = _quant("q_in", "x", "xq", 1 / 32, signed=1)
    init_out, quant_out = _quant("q_out", "r", "y", 1 / 32, signed=1)
    extra = [numpy_helper.from_array(np.array(v, np.float32), f"{op}_{n}") for n, v in constants(param)]
    layer = helper.make_node(onnx_type, ["xq"] + [t.name for t in extra], ["r"], name=f"{op}0", **attrs(param))
    graph = helper.make_graph(
        [quant_in, layer, quant_out],
        f"{op}_{elements}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, width])],
        initializer=init_in + init_out + extra,
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", _OPSET.get(op, 13)),
            helper.make_opsetid("qonnx.custom_op.general", 1),
        ],
    )


def _scale_dir_without_weights(directory):
    """Let neurigami write scale files into ``directory`` when the graph holds no arrays.

    A single activation has no weights, and neurigami's ``numeric._weights_dir`` raises then, because it
    borrows the directory of some node's array. The scale files only hold the quantizer's constants, so the
    estimate does not depend on where they go. neurigami itself is not changed.
    """
    import src.ir.numeric as numeric

    original = numeric._weights_dir

    def weights_dir(g):
        try:
            return original(g)
        except Exception:
            os.makedirs(directory, exist_ok=True)
            return directory

    numeric._weights_dir = weights_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--home", required=True, help="neurigami checkout")
    parser.add_argument("--op", default="relu", choices=OPS)
    parser.add_argument("--param", type=float, default=None, help="leaky_relu slope or thresholded_relu threshold")
    parser.add_argument("--vector", type=int, default=None, help="softmax: length of the normalised axis")
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--clock-ns", type=float, default=4.0)
    parser.add_argument("--part", default="xczu7ev")
    for name in ("lut", "ff", "bram36", "dsp"):
        parser.add_argument(f"--budget-{name}", type=float, default=None,
                            help=f"most {name} the design may use (what the chiplet has); lanes are chosen to fit")
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()
    if args.elements <= 0:
        raise SystemExit("--elements must be positive")

    sys.path.insert(0, os.path.abspath(args.home))
    os.chdir(args.home)
    import yaml
    from src.compile import build

    work = os.path.abspath(args.work_dir) if args.work_dir else tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    defaults = {"clock_ns": args.clock_ns, "part": args.part}
    for name in ("lut", "ff", "bram36", "dsp"):
        value = getattr(args, f"budget_{name}")
        if value is not None:
            defaults[f"budget_{name.rstrip('36')}"] = int(value)   # whole units: neurigami takes integers
    design = os.path.join(work, "design.yaml")
    with open(design, "w") as handle:
        yaml.safe_dump({"defaults": defaults}, handle)

    out_dir = os.path.join(work, "build")
    _scale_dir_without_weights(os.path.join(out_dir, "ir_work"))
    from src.ir.lanes_milp import NoFit

    try:
        build(single_layer_model(args.op, args.elements, args.param, args.vector), out_dir, design)
    except NoFit as error:                      # not even the smallest design fits the budget
        print(json.dumps({"no_fit": str(error)}))
        return
    with open(os.path.join(out_dir, "settings.yaml")) as handle:
        settings = yaml.safe_load(handle)
    total = settings["area"]["total"]
    timing = settings["timing"]
    rows = args.elements // args.vector if args.op == "softmax" and args.vector else 1
    print(json.dumps({
        "elements": args.elements,
        "rows": rows,
        "lut": float(total["lut"]),
        "ff": float(total["ff"]),
        "bram36": float(total["bram"]),
        "dsp": float(total["dsp"]),
        "uram": float(total["uram"]),
        "interval": int(timing["interval"]),
        "first_latency": int(timing["first_latency"]),
        "clock_ns": args.clock_ns,
        "part": settings["area"]["part"],
    }))


if __name__ == "__main__":
    main()
