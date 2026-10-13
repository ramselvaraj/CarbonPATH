"""The one result file of a run: an architecture and its workload mapping.

This module is the only code that knows the layout of the file. The rest of
CarbonPATH uses the in-memory architecture dictionary and ``WorkloadMapping``.

The file has four sections:

* ``workload``: the name and fingerprint of the graph the mapping is for.
* ``evaluated_with``: the name and fingerprint of the evaluation profile.
* ``architecture``: the chiplets, the package, and the link timing.
* ``workload_mapping``: one entry for each operation and for each handoff.

An architecture file of original CarbonPATH has no sections. It is read as an
architecture with no workload mapping.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from system.utils.ArchitectureModel import validate_modular_architecture
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.WorkloadMapping import OperationMapping, WorkloadMapping


FORMAT = "carbonpath_design_point"
VERSION = 1


@dataclass(frozen=True)
class LoadedDesignPoint:
    """An architecture, and the workload mapping when the file has one."""

    architecture: dict
    mapping: WorkloadMapping | None = None
    workload: dict | None = None
    evaluated_with: dict | None = None


def _chiplet_key(endpoint_id):
    return f"Chiplet_{endpoint_id + 1}"


def _endpoint_id(chiplet_key):
    try:
        return int(chiplet_key.split("_", 1)[1]) - 1
    except (AttributeError, IndexError, ValueError):
        raise UnsupportedEvaluation(f"invalid chiplet name '{chiplet_key}'")


def design_point_document(architecture, mapping, graph, profile):
    """Build the file content for one design point."""
    operations = []
    for operation in mapping.operations:
        entry = {
            "operation": operation.operation_id,
            "type": operation.operation_type,
            "endpoints": [
                _chiplet_key(endpoint_id) for endpoint_id in operation.endpoint_ids
            ],
        }
        if operation.is_gemm:
            entry.update(
                {
                    "dataflow": operation.dataflow,
                    "split_k": operation.split_k,
                    "tile_order": operation.tile_order,
                    "merge_tiles": operation.merge_tiles,
                }
            )
        operations.append(entry)
    return {
        "format": FORMAT,
        "version": VERSION,
        "workload": {"graph": graph.name, "fingerprint": graph.fingerprint()},
        "evaluated_with": {
            "profile": profile.name,
            "fingerprint": profile.fingerprint(),
        },
        "architecture": {
            key: copy.deepcopy(value)
            for key, value in architecture.items()
            if key != "WL_mapping"
        },
        "workload_mapping": {
            "operations": operations,
            "handoffs": [
                {
                    "from": mapping.operations[index].operation_id,
                    "to": mapping.operations[index + 1].operation_id,
                    "movement_policy": movement_policy,
                }
                for index, movement_policy in enumerate(mapping.handoffs)
            ],
        },
    }


def write_design_point(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(document, file, indent=4)


def is_design_point(document):
    return isinstance(document, dict) and document.get("format") == FORMAT


def _require(entry, field, label):
    if not isinstance(entry, dict) or field not in entry:
        raise UnsupportedEvaluation(f"{label} is missing '{field}'")
    return entry[field]


def _scheduler_mapping_block(mapping):
    """The mapping block the scheduler reads from the architecture dictionary.

    The workload mapping gives each GEMM its own values at evaluation time.
    The block holds the values of the first GEMM.
    """
    gemms = [operation for operation in mapping.operations if operation.is_gemm]
    first = gemms[0] if gemms else None
    return {
        "mapping": {
            "chiplet_data_sharing_enabled": 0,
            "if_splitting_k": int(first.split_k) if first else 0,
            "dataflow": [first.dataflow if first else "ws"],
            "assign_workload_in_ascending_order": int(
                first.tile_order == "ascending"
            )
            if first
            else 0,
            "static_tiling": 0,
            "merge_tiles": int(first.merge_tiles) if first else 0,
        }
    }


def parse_design_point(document):
    """Read the file content of a design point or of an original architecture."""
    if not isinstance(document, dict):
        raise UnsupportedEvaluation("design point file must hold a JSON object")
    if not is_design_point(document):
        return LoadedDesignPoint(architecture=document)
    if document.get("version") != VERSION:
        raise UnsupportedEvaluation(
            f"unsupported design point file version '{document.get('version')}'"
        )

    architecture = copy.deepcopy(_require(document, "architecture", "design point"))
    if not isinstance(architecture, dict):
        raise UnsupportedEvaluation("design point architecture must be an object")
    endpoints = validate_modular_architecture(architecture)
    endpoint_kinds = {
        **{key: "sa" for key in endpoints.sa_keys},
        **{key: "fpga" for key in endpoints.fpga_keys},
    }

    mapping_section = _require(document, "workload_mapping", "design point")
    operations = []
    for entry in _require(mapping_section, "operations", "workload mapping"):
        operation_id = _require(entry, "operation", "mapped operation")
        chiplet_keys = _require(entry, "endpoints", f"operation '{operation_id}'")
        if not isinstance(chiplet_keys, list) or not chiplet_keys:
            raise UnsupportedEvaluation(
                f"operation '{operation_id}' must name at least one endpoint"
            )
        unknown = [key for key in chiplet_keys if key not in endpoint_kinds]
        if unknown:
            raise UnsupportedEvaluation(
                f"operation '{operation_id}' names a chiplet that is not in the "
                "architecture: " + ", ".join(map(str, unknown))
            )
        kinds = {endpoint_kinds[key] for key in chiplet_keys}
        if len(kinds) != 1:
            raise UnsupportedEvaluation(
                f"operation '{operation_id}' mixes systolic-array and FPGA endpoints"
            )
        operations.append(
            OperationMapping(
                operation_id=operation_id,
                operation_type=_require(entry, "type", f"operation '{operation_id}'"),
                endpoint_kind=kinds.pop(),
                endpoint_ids=tuple(_endpoint_id(key) for key in chiplet_keys),
                dataflow=entry.get("dataflow"),
                split_k=entry.get("split_k"),
                tile_order=entry.get("tile_order"),
                merge_tiles=entry.get("merge_tiles"),
            )
        )

    handoffs = _require(mapping_section, "handoffs", "workload mapping")
    if not isinstance(handoffs, list) or len(handoffs) != max(0, len(operations) - 1):
        raise UnsupportedEvaluation(
            "workload mapping needs one handoff for each pair of adjacent operations"
        )
    for index, handoff in enumerate(handoffs):
        expected = (operations[index].operation_id, operations[index + 1].operation_id)
        if (handoff.get("from"), handoff.get("to")) != expected:
            raise UnsupportedEvaluation(
                f"handoff {index + 1} must go from '{expected[0]}' to '{expected[1]}'"
            )
    mapping = WorkloadMapping(
        operations=tuple(operations),
        handoffs=tuple(
            _require(handoff, "movement_policy", "handoff") for handoff in handoffs
        ),
    )
    architecture["WL_mapping"] = _scheduler_mapping_block(mapping)
    return LoadedDesignPoint(
        architecture=architecture,
        mapping=mapping,
        workload=document.get("workload"),
        evaluated_with=document.get("evaluated_with"),
    )


def read_design_point(path):
    with Path(path).open(encoding="utf-8") as file:
        return parse_design_point(json.load(file))


def latest_result_file(run_dir):
    """The newest result file of a run: a design point, else an old architecture."""
    run_dir = Path(run_dir)
    for pattern in ("best_design_point_*.json", "best_arch_*.json"):
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[-1]
    return None


def check_design_point_workload(loaded, graph):
    """Reject a workload mapping that was made for a different graph."""
    if loaded.mapping is None or not loaded.workload:
        return
    fingerprint = loaded.workload.get("fingerprint")
    if fingerprint != graph.fingerprint():
        raise UnsupportedEvaluation(
            f"the design point file is for workload "
            f"'{loaded.workload.get('graph')}' ({fingerprint}), not for "
            f"'{graph.name}' ({graph.fingerprint()})"
        )


def profile_differs(loaded, profile):
    """True when the file was scored with a different evaluation profile."""
    if not loaded.evaluated_with:
        return False
    return loaded.evaluated_with.get("fingerprint") != profile.fingerprint()
