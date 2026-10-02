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
    endpoint_ids: tuple[int, ...]
    endpoint_kind: str

    def __post_init__(self):
        endpoint_ids = self.endpoint_ids
        if isinstance(endpoint_ids, int):
            endpoint_ids = (endpoint_ids,)
        else:
            endpoint_ids = tuple(endpoint_ids)
        if not endpoint_ids:
            raise ValueError("operation placement cannot be empty")
        if len(set(endpoint_ids)) != len(endpoint_ids):
            raise ValueError("operation placement cannot contain duplicates")
        object.__setattr__(self, "endpoint_ids", endpoint_ids)

    @property
    def endpoint_id(self):
        """The scalar endpoint ID, or ``None`` when this is a group."""
        return self.endpoint_ids[0] if len(self.endpoint_ids) == 1 else None


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
        if operation_type in ("relu", "softmax"):
            return EndpointPlacement(self.fpga_id, "fpga")
        raise UnsupportedEvaluation(
            f"placement policy has no endpoint for operation type '{operation_type}'"
        )


class AllSystolicArraysSingleFpgaPlacement:
    """Place GEMMs on every SA and supported elementwise ops on one FPGA."""

    policy_id = "all_sas_single_fpga_v1"

    def __init__(self, system):
        self.sa_ids = tuple(sorted(system.core_dict))
        self.fpga_ids = tuple(sorted(system.fpga_chiplet_dict))
        if not self.sa_ids:
            raise UnsupportedEvaluation(
                "all-SA placement requires at least one systolic-array endpoint"
            )
        if len(self.fpga_ids) > 1:
            raise UnsupportedEvaluation(
                "all-SA placement currently supports at most one FPGA endpoint"
            )

    def endpoint_for(self, operation_type):
        if operation_type == "gemm":
            return EndpointPlacement(self.sa_ids, "sa")
        if operation_type in ("relu", "softmax"):
            if not self.fpga_ids:
                raise UnsupportedEvaluation(
                    f"operation type '{operation_type}' requires an FPGA endpoint"
                )
            return EndpointPlacement(self.fpga_ids, "fpga")
        raise UnsupportedEvaluation(
            f"placement policy has no endpoint for operation type '{operation_type}'"
        )


PLACEMENT_POLICIES = {
    FixedSingleSaSingleFpgaPlacement.policy_id: FixedSingleSaSingleFpgaPlacement,
    AllSystolicArraysSingleFpgaPlacement.policy_id: AllSystolicArraysSingleFpgaPlacement,
}


def build_placement_policy(policy_id, system):
    """Resolve an evaluation profile's operation placement policy ID."""
    try:
        policy_class = PLACEMENT_POLICIES[policy_id]
    except KeyError:
        raise UnsupportedEvaluation(f"unknown placement policy '{policy_id}'")
    return policy_class(system)
