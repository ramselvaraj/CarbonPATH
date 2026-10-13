"""Distributed movement policy for any FPGA activation operation.

``distributed_activation_boundary_v1`` routes GEMM <-> ReLU handoffs. This
policy applies the same slice, ownership, and routing rules to every FPGA
activation operation (ReLU and Softmax), and retains a tensor that stays on
one FPGA between two activations. It is selected by an evaluation profile's
``movement_policy`` field; the v1 policy is unchanged.
"""

from system.utils.TensorMovement import (
    MOVEMENT_POLICIES,
    DistributedActivationMovementService,
)
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation

FPGA_ACTIVATION_OPERATIONS = ("relu", "softmax")


class FpgaActivationMovementService(DistributedActivationMovementService):
    policy_id = "distributed_activation_boundary_v2"

    _SUPPORTED_PAIRS = {
        **{
            ("gemm", activation): "_plan_gemm_to_fpga"
            for activation in FPGA_ACTIVATION_OPERATIONS
        },
        **{
            (activation, "gemm"): "_plan_fpga_to_gemm"
            for activation in FPGA_ACTIVATION_OPERATIONS
        },
    }

    def plan(self, request):
        producer_type = request.producer_operation.operation_type
        consumer_type = request.consumer_operation.operation_type

        if (
            producer_type in FPGA_ACTIVATION_OPERATIONS
            and consumer_type in FPGA_ACTIVATION_OPERATIONS
            and request.producer_assignment.endpoint_ids
            == request.consumer_assignment.endpoint_ids
        ):
            return self._retained_plan(request)

        handler_name = self._SUPPORTED_PAIRS.get((producer_type, consumer_type))
        if handler_name is None:
            raise UnsupportedEvaluation(
                f"movement policy '{self.policy_id}' supports gemm<->relu, "
                f"gemm<->softmax, and same-FPGA activation handoffs, not "
                f"{producer_type}->{consumer_type}"
            )
        return getattr(self, handler_name)(request)


def register_fpga_activation_movement():
    """Register the policy. Idempotent."""
    MOVEMENT_POLICIES.setdefault(
        FpgaActivationMovementService.policy_id, FpgaActivationMovementService
    )
