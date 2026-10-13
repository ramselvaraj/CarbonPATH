"""Workload mapping: how one workload uses one architecture.

A design point has two parts. The architecture says which hardware exists. The
workload mapping says which endpoints run each operation, how each GEMM is
tiled, and how each tensor moves between two adjacent operations.

Annealing changes both parts. The evaluation profile is not part of the design
point: it names the models that score a design point and stays constant for
one run.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

from system.utils.ArchitectureIdentity import effective_gemm_mapping
from system.utils.ArchitectureModel import validate_modular_architecture
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


GEMM_DATAFLOWS = ("ws", "os", "is")
GEMM_TILE_ORDERS = ("ascending", "descending")


def endpoint_kind_for(operation_type):
    """The endpoint kind that runs one operation type."""
    return "sa" if operation_type == "gemm" else "fpga"


@dataclass(frozen=True)
class OperationMapping:
    """Where one operation runs, and the GEMM tiling choices when it is a GEMM."""

    operation_id: str
    operation_type: str
    endpoint_kind: str
    endpoint_ids: tuple
    dataflow: str | None = None
    split_k: bool | None = None
    tile_order: str | None = None
    merge_tiles: bool | None = None

    def __post_init__(self):
        endpoint_ids = tuple(sorted(self.endpoint_ids))
        if not endpoint_ids:
            raise UnsupportedEvaluation(
                f"operation '{self.operation_id}' has no endpoint"
            )
        if len(set(endpoint_ids)) != len(endpoint_ids):
            raise UnsupportedEvaluation(
                f"operation '{self.operation_id}' repeats an endpoint"
            )
        object.__setattr__(self, "endpoint_ids", endpoint_ids)
        if self.operation_type != "gemm":
            return
        if self.dataflow not in GEMM_DATAFLOWS:
            raise UnsupportedEvaluation("GEMM dataflow must be one of: ws, os, is")
        if not isinstance(self.split_k, bool):
            raise UnsupportedEvaluation("GEMM split_k must be boolean")
        if self.tile_order not in GEMM_TILE_ORDERS:
            raise UnsupportedEvaluation(
                "GEMM tile_order must be ascending or descending"
            )
        if not isinstance(self.merge_tiles, bool):
            raise UnsupportedEvaluation("GEMM merge_tiles must be boolean")

    @property
    def is_gemm(self):
        return self.operation_type == "gemm"

    def evaluator_settings(self):
        """The settings a GEMM evaluator reads from this mapping."""
        if not self.is_gemm:
            return {}
        return {
            "dataflow": self.dataflow,
            "split_k": self.split_k,
            "assignment_order": self.tile_order,
            "merge_tiles": self.merge_tiles,
        }

    def canonical_dict(self):
        value = {
            "operation": self.operation_id,
            "type": self.operation_type,
            "endpoint_kind": self.endpoint_kind,
            "endpoints": list(self.endpoint_ids),
        }
        if self.is_gemm:
            value.update(
                {
                    "dataflow": self.dataflow,
                    "split_k": self.split_k,
                    "tile_order": self.tile_order,
                    "merge_tiles": self.merge_tiles,
                }
            )
        return value


@dataclass(frozen=True)
class WorkloadMapping:
    """One entry for each operation and one movement policy for each handoff.

    ``handoffs[i]`` is the movement policy between ``operations[i]`` and
    ``operations[i + 1]``.
    """

    operations: tuple
    handoffs: tuple

    def __post_init__(self):
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "handoffs", tuple(self.handoffs))
        if len(self.handoffs) != max(0, len(self.operations) - 1):
            raise UnsupportedEvaluation(
                "workload mapping needs one movement policy for each handoff"
            )

    @property
    def gemm_indexes(self):
        return tuple(
            index
            for index, operation in enumerate(self.operations)
            if operation.is_gemm
        )

    def handoff_kind(self, handoff_index):
        producer = self.operations[handoff_index]
        consumer = self.operations[handoff_index + 1]
        return f"{producer.endpoint_kind}_to_{consumer.endpoint_kind}"

    def with_operation(self, operation_index, **changes):
        operations = list(self.operations)
        operations[operation_index] = replace(operations[operation_index], **changes)
        return WorkloadMapping(operations=tuple(operations), handoffs=self.handoffs)

    def with_gemms(self, **changes):
        """Return a copy with the same change in every GEMM."""
        return WorkloadMapping(
            operations=tuple(
                replace(operation, **changes) if operation.is_gemm else operation
                for operation in self.operations
            ),
            handoffs=self.handoffs,
        )

    def with_handoff(self, handoff_index, movement_policy):
        handoffs = list(self.handoffs)
        handoffs[handoff_index] = movement_policy
        return WorkloadMapping(operations=self.operations, handoffs=tuple(handoffs))

    def with_endpoints_relabeled(self, endpoint_id_map, added_sa_ids=()):
        """Follow a chiplet relabel after an endpoint is added or removed.

        ``endpoint_id_map`` maps each old endpoint ID that still exists to its
        new ID. A removed endpoint leaves each operation. Each new SA joins
        every GEMM. An operation that loses its last endpoint is invalid.
        """
        operations = []
        for operation in self.operations:
            endpoint_ids = [
                endpoint_id_map[endpoint_id]
                for endpoint_id in operation.endpoint_ids
                if endpoint_id in endpoint_id_map
            ]
            if operation.is_gemm:
                endpoint_ids.extend(added_sa_ids)
            if not endpoint_ids:
                raise UnsupportedEvaluation(
                    f"operation '{operation.operation_id}' has no endpoint after "
                    "the architecture change"
                )
            operations.append(replace(operation, endpoint_ids=tuple(endpoint_ids)))
        return WorkloadMapping(operations=tuple(operations), handoffs=self.handoffs)

    def canonical_dict(self):
        return {
            "operations": [
                operation.canonical_dict() for operation in self.operations
            ],
            "handoffs": [
                {
                    "from": self.operations[index].operation_id,
                    "to": self.operations[index + 1].operation_id,
                    "movement_policy": movement_policy,
                }
                for index, movement_policy in enumerate(self.handoffs)
            ],
        }

    def fingerprint(self):
        serialized = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("ascii")).hexdigest()[:12]

    def shared_gemm_settings(self):
        """The GEMM settings when every GEMM uses the same ones, else ``None``."""
        settings = [
            self.operations[index].evaluator_settings()
            for index in self.gemm_indexes
        ]
        if not settings or any(value != settings[0] for value in settings):
            return None
        return settings[0]


def movement_policy_options(search_space, handoff_kind):
    """The movement policies the search may select for one handoff kind."""
    options = (search_space or {}).get("movement_policy", {})
    if not isinstance(options, dict):
        raise UnsupportedEvaluation("search space movement_policy must be an object")
    values = options.get(handoff_kind, [])
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise UnsupportedEvaluation(
            f"movement_policy options for '{handoff_kind}' must be a list of IDs"
        )
    return values


def initial_workload_mapping(graph, architecture, profile, search_space=None):
    """Build the first workload mapping for a run.

    Every GEMM starts on all SAs with the GEMM settings of the architecture and
    profile. Every other operation starts on the FPGA. Each handoff starts with
    the profile's movement policy; when the search space lists the policies for
    that handoff kind and the profile's policy is not one of them, the first
    listed policy is used.
    """
    endpoints = validate_modular_architecture(architecture)
    gemm_settings = effective_gemm_mapping(architecture, profile)
    operations = []
    for operation in graph.operations:
        endpoint_kind = endpoint_kind_for(operation.operation_type)
        if endpoint_kind == "sa":
            operations.append(
                OperationMapping(
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type,
                    endpoint_kind=endpoint_kind,
                    endpoint_ids=endpoints.sa_ids,
                    dataflow=gemm_settings["dataflow"],
                    split_k=gemm_settings["split_k"],
                    tile_order=gemm_settings["assignment_order"],
                    merge_tiles=bool(
                        architecture.get("WL_mapping", {})
                        .get("mapping", {})
                        .get("merge_tiles", 0)
                    ),
                )
            )
            continue
        if not endpoints.fpga_ids:
            raise UnsupportedEvaluation(
                f"operation type '{operation.operation_type}' requires an FPGA "
                "endpoint"
            )
        operations.append(
            OperationMapping(
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                endpoint_kind=endpoint_kind,
                endpoint_ids=endpoints.fpga_ids,
            )
        )

    mapping = WorkloadMapping(
        operations=tuple(operations),
        handoffs=(profile.movement_policy,) * max(0, len(operations) - 1),
    )
    for handoff_index in range(len(mapping.handoffs)):
        options = movement_policy_options(
            search_space, mapping.handoff_kind(handoff_index)
        )
        if options and profile.movement_policy not in options:
            mapping = mapping.with_handoff(handoff_index, options[0])
    return mapping
