"""FPGA chiplet endpoint representation.

An FPGA chiplet is a non-GEMM compute endpoint in the CarbonPATH architecture.
It owns the resources (CLB, BRAM, DSP) used to build element-wise kernels such
as the ReLU activation. Its per-lane implementation profile is supplied as
configuration; the estimator never derives resources from synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from system.utils.ParameterLoad import PARAMETER_SOURCES
from system.utils.Rule4MlBoards import board_chiplet_fields, load_boards


@dataclass(frozen=True)
class FpgaReluImplementation:
    clbs_per_lane: int
    brams_per_lane: int = 0
    dsps_per_lane: int = 0
    max_parallel_lanes: int | None = None
    energy_per_element_pj: float | None = None

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("relu_implementation must be an object")
        allowed = {
            "clbs_per_lane",
            "brams_per_lane",
            "dsps_per_lane",
            "max_parallel_lanes",
            "energy_per_element_pj",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"Unsupported relu_implementation field(s): {', '.join(unknown)}"
            )
        clbs_per_lane = value.get("clbs_per_lane")
        if not isinstance(clbs_per_lane, int) or clbs_per_lane <= 0:
            raise ValueError("relu_implementation.clbs_per_lane must be positive")
        brams_per_lane = value.get("brams_per_lane", 0)
        dsps_per_lane = value.get("dsps_per_lane", 0)
        for field, count in (
            ("brams_per_lane", brams_per_lane),
            ("dsps_per_lane", dsps_per_lane),
        ):
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"relu_implementation.{field} must be >= 0")
        max_parallel_lanes = value.get("max_parallel_lanes")
        if max_parallel_lanes is not None and (
            not isinstance(max_parallel_lanes, int) or max_parallel_lanes <= 0
        ):
            raise ValueError(
                "relu_implementation.max_parallel_lanes must be positive or null"
            )
        energy_per_element_pj = value.get("energy_per_element_pj")
        if energy_per_element_pj is not None and (
            not isinstance(energy_per_element_pj, (int, float))
            or energy_per_element_pj < 0
        ):
            raise ValueError(
                "relu_implementation.energy_per_element_pj must be >= 0 or null"
            )
        return cls(
            clbs_per_lane=clbs_per_lane,
            brams_per_lane=brams_per_lane,
            dsps_per_lane=dsps_per_lane,
            max_parallel_lanes=max_parallel_lanes,
            energy_per_element_pj=energy_per_element_pj,
        )


@dataclass(frozen=True)
class FpgaSoftmaxImplementation:
    """Softmax implementation parameters; the default matches the placeholder."""

    energy_per_element_pj: float = 2.0

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("softmax_implementation must be an object")
        unknown = sorted(set(value) - {"energy_per_element_pj"})
        if unknown:
            raise ValueError(
                f"Unsupported softmax_implementation field(s): {', '.join(unknown)}"
            )
        energy = value.get("energy_per_element_pj", cls.energy_per_element_pj)
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or energy < 0:
            raise ValueError(
                "softmax_implementation.energy_per_element_pj must be >= 0"
            )
        return cls(energy_per_element_pj=float(energy))


@dataclass(frozen=True)
class FpgaLayerNormImplementation:
    """LayerNorm implementation parameters; the energy rate is provisional."""

    energy_per_element_pj: float = 2.0

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("layernorm_implementation must be an object")
        unknown = sorted(set(value) - {"energy_per_element_pj"})
        if unknown:
            raise ValueError(
                f"Unsupported layernorm_implementation field(s): {', '.join(unknown)}"
            )
        energy = value.get("energy_per_element_pj", cls.energy_per_element_pj)
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or energy < 0:
            raise ValueError(
                "layernorm_implementation.energy_per_element_pj must be >= 0"
            )
        return cls(energy_per_element_pj=float(energy))


@dataclass
class FpgaChiplet:
    id: int
    name: str
    frequency_hz: float
    clbs: int
    brams: int
    dsps: int
    area: float
    power: float
    node: int
    relu_implementation: FpgaReluImplementation
    location: str = "chiplet"
    luts: int | None = None
    ffs: int | None = None
    softmax_implementation: FpgaSoftmaxImplementation = field(
        default_factory=FpgaSoftmaxImplementation
    )
    layernorm_implementation: FpgaLayerNormImplementation = field(
        default_factory=FpgaLayerNormImplementation
    )
    board: str | None = None
    # Where the constants of the kernels come from: "dram" loads them over the
    # route to the memory-attached systolic array; "bitstream" bakes them into
    # the configuration, so there is no load per inference.
    parameter_source: str = "bitstream"
    # BRAM blocks of the input buffer read in parallel; None sizes the buffer
    # for each operation at the fewest blocks that hold its input.
    input_buffer_blocks: int | None = None

    @property
    def chiplet_type(self):
        return "fpga"

    @classmethod
    def from_dict(cls, name, value):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be an object")
        allowed = {
            "chiplet_type",
            "tech_node",
            "area",
            "power",
            "frequency_hz",
            "clbs",
            "brams",
            "dsps",
            "luts",
            "ffs",
            "board",
            "relu_implementation",
            "softmax_implementation",
            "layernorm_implementation",
            "parameter_source",
            "input_buffer_blocks",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unsupported {name} field(s): {', '.join(unknown)}")

        tech_node = value.get("tech_node")
        if tech_node is None:
            raise ValueError(f"{name} must define tech_node")
        area = value.get("area")
        if not isinstance(area, (int, float)) or area <= 0:
            raise ValueError(f"{name}.area must be positive")
        power = value.get("power", 0.0)
        frequency_hz = value.get("frequency_hz")
        if not isinstance(frequency_hz, (int, float)) or frequency_hz <= 0:
            raise ValueError(f"{name}.frequency_hz must be positive")

        counts = {}
        for field in ("clbs", "brams", "dsps"):
            count = value.get(field)
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"{name}.{field} must be a non-negative integer")
            counts[field] = count

        optional_counts = {}
        for field in ("luts", "ffs"):
            count = value.get(field)
            if count is not None and (not isinstance(count, int) or count < 0):
                raise ValueError(
                    f"{name}.{field} must be a non-negative integer when present"
                )
            optional_counts[field] = count

        board = value.get("board")
        if board is not None:
            boards = load_boards()
            if board not in boards:
                raise ValueError(
                    f"{name}.board must be one of: {', '.join(boards)}"
                )
            expected = board_chiplet_fields(board, boards)
            for field in ("clbs", "brams", "dsps"):
                if counts[field] != expected[field]:
                    raise ValueError(
                        f"{name}.{field} is {counts[field]} but board '{board}' "
                        f"has {expected[field]}"
                    )

        input_buffer_blocks = value.get("input_buffer_blocks")
        if input_buffer_blocks is not None and (
            isinstance(input_buffer_blocks, bool)
            or not isinstance(input_buffer_blocks, int)
            or input_buffer_blocks <= 0
        ):
            raise ValueError(
                f"{name}.input_buffer_blocks must be a positive integer when present"
            )

        parameter_source = value.get("parameter_source", "bitstream")
        if parameter_source not in PARAMETER_SOURCES:
            raise ValueError(
                f"{name}.parameter_source must be one of: "
                f"{', '.join(PARAMETER_SOURCES)}"
            )

        implementation = FpgaReluImplementation.from_dict(
            value.get("relu_implementation", {})
        )

        chiplet_id = int(name.split("_")[1]) - 1
        return cls(
            id=chiplet_id,
            name=name,
            frequency_hz=float(frequency_hz),
            clbs=counts["clbs"],
            brams=counts["brams"],
            dsps=counts["dsps"],
            area=float(area),
            power=float(power),
            node=int(tech_node),
            relu_implementation=implementation,
            luts=optional_counts["luts"],
            ffs=optional_counts["ffs"],
            softmax_implementation=FpgaSoftmaxImplementation.from_dict(
                value.get("softmax_implementation", {})
            ),
            layernorm_implementation=FpgaLayerNormImplementation.from_dict(
                value.get("layernorm_implementation", {})
            ),
            board=board,
            parameter_source=parameter_source,
            input_buffer_blocks=input_buffer_blocks,
        )
