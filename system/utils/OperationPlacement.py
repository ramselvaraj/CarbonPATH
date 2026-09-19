"""Operation placement policies.

Operation placement assigns a graph operation to an endpoint or endpoint group.
It is distinct from GEMM tile mapping, which assigns the tiles of one GEMM to
systolic arrays within its assigned SA group.
"""

from __future__ import annotations

from dataclasses import dataclass

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


@dataclass(frozen=True)
class EndpointPlacement:
    endpoint_id: int
    endpoint_kind: str


class FixedSingleSaSingleFpgaPlacement:
    """First placement policy: one SA endpoint and one FPGA endpoint.

    GEMM runs on the SA; every non-GEMM operation runs on the FPGA. This
    reproduces the current behavior explicitly rather than hiding it in the
    executor.
    """

    policy_id = "fixed_single_sa_single_fpga_v1"

    def __init__(self, system):
        sa_ids = sorted(system.core_dict)
        fpga_ids = sorted(system.fpga_chiplet_dict)
        if len(sa_ids) != 1:
            raise UnsupportedEvaluation(
                "ATLAS evaluation requires exactly one systolic-array endpoint"
            )
        if len(fpga_ids) != 1:
            raise UnsupportedEvaluation(
                "ATLAS evaluation requires exactly one FPGA endpoint"
            )
        self.sa_id = sa_ids[0]
        self.fpga_id = fpga_ids[0]

    def endpoint_for(self, operation_type):
        if operation_type == "gemm":
            return EndpointPlacement(self.sa_id, "sa")
        if operation_type == "relu":
            return EndpointPlacement(self.fpga_id, "fpga")
        raise UnsupportedEvaluation(
            f"placement policy has no endpoint for operation type '{operation_type}'"
        )
