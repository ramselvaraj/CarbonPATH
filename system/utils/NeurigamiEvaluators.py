"""neurigami operation evaluator, input adapter, and registration.

neurigami compiles a quantized network to streaming Verilog and reports its own
area estimate and exact timing. This evaluator uses it for one operation at a
time: the operation is built as a single-layer model of the same size, and the
single layer's estimate is the operation's estimate. The rest of the network is
not seen, so neighbouring layers do not change the result.

It is a drop-in operation evaluator: an evaluation profile names it, the
executor and movement policies are unchanged. neurigami runs in its own Python
environment as a subprocess (system/utils/NeurigamiRunner.py), so profiles that
do not name it load and run without neurigami installed.

Evaluator settings, read from the profile's per-operation ``settings`` object:

* ``neurigami_home``       neurigami checkout; default ``$NEURIGAMI_HOME``.
* ``neurigami_python``     its Python; default ``$NEURIGAMI_PYTHON`` or ``<home>/.venv``.
* ``part``                 neurigami part for the estimate; overrides the chiplet's own.
                           Default: the chiplet's ``calibration_part``, else its own part (or its
                           board's) when neurigami knows it, else the nearest calibrated part by size.
                           LIMITATION: the part only picks neurigami's timing calibration (fitted on
                           xck26). A free chiplet spec far from it gets resources that follow its
                           budget but timing that may be off.
* ``latency_metric``       ``first_latency`` (default) or ``interval``, in cycles.
* ``energy_per_element_pj``  dynamic energy per element; default ``0.0``.
* ``cache_dir``            result cache; default ``output/neurigami_cache``.

The chiplet's LUT, FF, BRAM and DSP counts go to neurigami as the area budget, so a foldable op (softmax) gets
the fastest lanes that fit the chiplet. A softmax over a [rows, vector] tensor is built for one row; the other
rows stream through the same hardware one interval apart. The ATLAS ``activ_param`` attribute gives the slope of
leaky_relu, the threshold of thresholded_relu and the max value of clipped_relu. hard_sigmoid and hard_tanh
are built as one table like sigmoid: a table's cost does not depend on its function, so the slope and shift of
the ATLAS HardActivation node do not change the estimate.

The clock is the FPGA chiplet's own: neurigami pipelines for that period and
the cycle counts convert to nanoseconds at that frequency.

Units. neurigami counts BRAM in 36 Kb tiles; CarbonPATH counts 18 Kb units, so a
tile is two units. neurigami reports URAM, which CarbonPATH does not model; a
design that uses any is an unsupported evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from system.utils import NeurigamiRunner
from system.utils.ChipletResourceCheck import ChipletUse
from system.utils.OperationEvaluator import (
    OperationEstimate,
    OperationEvaluator,
    OperationInputAdapter,
)
from system.utils.FpgaParts import neurigami_part_for
from system.utils.Rule4MlBoards import fpga_resource_vector
from system.utils.Rule4MlEvaluators import check_resources_fit
from system.utils.Rule4MlInputBuffer import VALUE_BITS, input_buffer
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation

#: Operation types neurigami can build as a single-layer model here.
NEURIGAMI_SUPPORTED_OPERATIONS = (
    "relu", "leaky_relu", "thresholded_relu", "sigmoid", "tanh", "softmax",
    "clipped_relu", "hard_sigmoid", "hard_tanh",
)
#: Operations that carry one number in the ATLAS ``activ_param`` attribute, with neurigami's default for it.
_PARAMETER_DEFAULTS = {"leaky_relu": 0.01, "thresholded_relu": 1.0, "clipped_relu": 6.0}

_SUPPORTED_SETTINGS = {
    "neurigami_home",
    "neurigami_python",
    "part",
    "latency_metric",
    "energy_per_element_pj",
    "cache_dir",
}
_BRAM18_PER_BRAM36 = 2


@dataclass(frozen=True)
class NeurigamiOperationInput:
    operation_id: str
    operation_type: str
    element_count: int
    shape: tuple
    # The slope of leaky_relu or the threshold of thresholded_relu; None for the others.
    parameter: float | None = None
    # Softmax normalises over the last axis; its length. None for elementwise operations.
    vector: int | None = None


class NeurigamiInputAdapter(OperationInputAdapter):
    """Reads the size of the operation; neurigami models it as 8-bit."""

    operation_type = ""
    input_adapter_id = ""

    def build_input(self, operation, access_plan):
        if operation.operation_type != self.operation_type:
            raise UnsupportedEvaluation(
                f"{type(self).__name__} cannot read "
                f"'{operation.operation_type}' operation '{operation.operation_id}'"
            )
        if operation.element_count <= 0:
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: the operation has no elements"
            )
        shape = tuple(operation.output_tensor.shape)
        parameter = None
        if self.operation_type in _PARAMETER_DEFAULTS:
            parameter = operation.source.attribute(
                "activ_param", _PARAMETER_DEFAULTS[self.operation_type]
            )
            if isinstance(parameter, bool) or not isinstance(parameter, (int, float)):
                raise UnsupportedEvaluation(
                    f"{operation.operation_id}: ATLAS attribute 'activ_param' must be a number"
                )
        return NeurigamiOperationInput(
            operation_id=operation.operation_id,
            operation_type=operation.operation_type,
            element_count=operation.element_count,
            shape=shape,
            parameter=None if parameter is None else float(parameter),
            vector=shape[-1] if self.operation_type == "softmax" else None,
        )


class NeurigamiEvaluator(OperationEvaluator):
    """Estimates one operation by compiling it alone with neurigami."""

    operation_type = ""
    evaluator_id = ""
    input_adapter_id = ""

    def evaluate(self, evaluator_input, assignment, context) -> OperationEstimate:
        op_id = evaluator_input.operation_id
        if self.operation_type not in NEURIGAMI_SUPPORTED_OPERATIONS:
            raise UnsupportedEvaluation(
                f"{op_id}: neurigami is not wired for '{self.operation_type}'; "
                "it supports " + ", ".join(NEURIGAMI_SUPPORTED_OPERATIONS)
            )
        fpga = context.system.fpga_chiplet_dict.get(assignment.endpoint_id)
        if fpga is None:
            raise UnsupportedEvaluation(
                f"no FPGA endpoint {assignment.endpoint_id} for '{op_id}'"
            )
        if fpga.frequency_hz <= 0:
            raise UnsupportedEvaluation(f"{op_id}: fpga frequency must be positive")

        settings = dict(context.evaluator_settings or {})
        unknown = sorted(set(settings) - _SUPPORTED_SETTINGS)
        if unknown:
            raise UnsupportedEvaluation(
                "unsupported neurigami evaluator setting(s): " + ", ".join(unknown)
            )
        metric = settings.get("latency_metric", "first_latency")
        if metric not in ("first_latency", "interval"):
            raise UnsupportedEvaluation(
                "neurigami latency_metric must be 'first_latency' or 'interval'"
            )

        clock_ns = 1e9 / fpga.frequency_hz
        buffer = input_buffer(
            evaluator_input.element_count, VALUE_BITS, fpga.input_buffer_blocks
        )
        available = fpga_resource_vector(fpga)
        result = NeurigamiRunner.estimate_single_layer(
            op=self.operation_type,
            elements=evaluator_input.element_count,
            clock_ns=clock_ns,
            part=settings.get("part") or neurigami_part_for(fpga),
            param=evaluator_input.parameter,
            vector=evaluator_input.vector,
            # What the chiplet has: neurigami picks the lanes of a foldable op (softmax) to fit it. Its BRAM is
            # in 36 Kb tiles; the chiplet's is in 18 Kb units. The input buffer's blocks come off the BRAM.
            budget={
                "lut": available["lut"],
                "ff": available["ff"],
                "bram36": max(0.0, (available["bram"] - buffer.blocks) / _BRAM18_PER_BRAM36),
                "dsp": available["dsp"],
            },
            home=settings.get("neurigami_home"),
            python=settings.get("neurigami_python"),
            cache_dir=settings.get("cache_dir", NeurigamiRunner.DEFAULT_CACHE_DIR),
        )
        if result["uram"] > 0:
            raise UnsupportedEvaluation(
                f"{op_id}: neurigami used {result['uram']:.0f} URAM, which CarbonPATH does not model"
            )
        kernel = {
            "LUT": result["lut"],
            "FF": result["ff"],
            "BRAM": result["bram36"] * _BRAM18_PER_BRAM36,
            "DSP": result["dsp"],
        }
        check_resources_fit(op_id, fpga, {**kernel, "BRAM": kernel["BRAM"] + buffer.blocks})

        # Rows beyond the first (softmax over a [rows, vector] tensor) stream through the same hardware one
        # interval apart; an elementwise op has one row.
        rows = int(result.get("rows", 1))
        cycles = result["interval"] * rows if metric == "interval" else (
            result["first_latency"] + (rows - 1) * result["interval"]
        )
        latency_ns = (cycles + buffer.read_cycles) * clock_ns
        return OperationEstimate(
            operation_id=op_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=latency_ns,
            dynamic_energy_pj=evaluator_input.element_count
            * float(settings.get("energy_per_element_pj", 0.0)),
            chiplet_use=(
                ChipletUse(
                    endpoint_id=assignment.endpoint_id,
                    kernel=kernel,
                    buffer_blocks=buffer.blocks,
                ),
            ),
        )


def neurigami_input_adapter(operation_type):
    return type(
        f"Neurigami_{operation_type}_InputAdapter",
        (NeurigamiInputAdapter,),
        {
            "operation_type": operation_type,
            "input_adapter_id": f"neurigami_{operation_type}_input_v1",
        },
    )()


def neurigami_evaluator(operation_type):
    return type(
        f"Neurigami_{operation_type}_Evaluator",
        (NeurigamiEvaluator,),
        {
            "operation_type": operation_type,
            "evaluator_id": f"neurigami_fpga_{operation_type}_v1",
            "input_adapter_id": f"neurigami_{operation_type}_input_v1",
        },
    )()


def register_neurigami_evaluators(registry):
    """Register the neurigami evaluators into ``registry``."""
    if registry is None:
        raise ValueError("register_neurigami_evaluators requires an evaluator registry")
    for operation_type in NEURIGAMI_SUPPORTED_OPERATIONS:
        registry.register(
            neurigami_evaluator(operation_type), neurigami_input_adapter(operation_type)
        )
    return registry
