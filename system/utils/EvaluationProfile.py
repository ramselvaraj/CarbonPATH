"""Evaluation profile: the named selection of models for one evaluation.

A profile names operation evaluators, a placement policy, a tensor movement
policy, and a transfer cost model. It carries names and versions only; hardware
values live in the architecture and evaluator-specific characterization is
configured elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


DEFAULT_PROFILE_PATH = Path("cfg/profiles/legacy_sa_fpga_v1.json")
_REQUIRED_FIELDS = {
    "profile",
    "version",
    "evaluators",
    "placement_policy",
    "movement_policy",
    "transfer_model",
}


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    version: int
    evaluators: dict
    placement_policy: str
    movement_policy: str
    transfer_model: str

    def evaluator_id_for(self, operation_type):
        try:
            return self.evaluators[operation_type]
        except KeyError:
            raise UnsupportedEvaluation(
                f"profile '{self.name}' has no evaluator for operation type "
                f"'{operation_type}'"
            )

    def canonical_dict(self):
        return {
            "profile": self.name,
            "version": self.version,
            "evaluators": dict(sorted(self.evaluators.items())),
            "placement_policy": self.placement_policy,
            "movement_policy": self.movement_policy,
            "transfer_model": self.transfer_model,
        }

    def fingerprint(self):
        serialized = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("ascii")).hexdigest()[:12]


def parse_evaluation_profile(entry):
    if not isinstance(entry, dict):
        raise UnsupportedEvaluation("evaluation profile must be an object")
    missing = sorted(_REQUIRED_FIELDS - set(entry))
    if missing:
        raise UnsupportedEvaluation(
            "evaluation profile is missing field(s): " + ", ".join(missing)
        )
    name = entry.get("profile")
    if not isinstance(name, str) or not name:
        raise UnsupportedEvaluation("evaluation profile name must be a non-empty string")
    version = entry.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise UnsupportedEvaluation("evaluation profile version must be positive")
    evaluators = entry.get("evaluators")
    if not isinstance(evaluators, dict) or not evaluators:
        raise UnsupportedEvaluation("evaluation profile must name evaluators")
    for operation_type, evaluator_id in evaluators.items():
        if not isinstance(operation_type, str) or not operation_type:
            raise UnsupportedEvaluation("evaluator operation types must be non-empty")
        if not isinstance(evaluator_id, str) or not evaluator_id:
            raise UnsupportedEvaluation(
                f"evaluator for '{operation_type}' must be a non-empty string"
            )
    for field in ("placement_policy", "movement_policy", "transfer_model"):
        value = entry.get(field)
        if not isinstance(value, str) or not value:
            raise UnsupportedEvaluation(f"evaluation profile {field} must be a string")
    return EvaluationProfile(
        name=name,
        version=version,
        evaluators=dict(evaluators),
        placement_policy=entry["placement_policy"],
        movement_policy=entry["movement_policy"],
        transfer_model=entry["transfer_model"],
    )


def load_evaluation_profile(path=None):
    profile_path = Path(path) if path else DEFAULT_PROFILE_PATH
    with profile_path.open(encoding="utf-8") as file:
        return parse_evaluation_profile(json.load(file))
