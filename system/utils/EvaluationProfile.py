"""Evaluation profile: the named selection of models for one evaluation.

A profile names operation evaluators, a placement policy, a tensor movement
policy, and a transfer cost model. An evaluator selection may also carry the
settings owned by that evaluator. Hardware values remain in the architecture.
"""

from __future__ import annotations

from copy import deepcopy
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
class EvaluatorSelection:
    evaluator_id: str
    settings: dict

    def canonical_value(self):
        if not self.settings:
            return self.evaluator_id
        return {"id": self.evaluator_id, "settings": deepcopy(self.settings)}


@dataclass(frozen=True)
class EvaluationProfile:
    name: str
    version: int
    evaluators: dict
    placement_policy: str
    movement_policy: str
    transfer_model: str

    def with_movement_policy(self, movement_policy):
        """Return the same model selection with a different movement policy."""
        if not isinstance(movement_policy, str) or not movement_policy:
            raise UnsupportedEvaluation("evaluation profile movement_policy must be a string")
        return EvaluationProfile(
            name=self.name,
            version=self.version,
            evaluators=self.evaluators,
            placement_policy=self.placement_policy,
            movement_policy=movement_policy,
            transfer_model=self.transfer_model,
        )

    def with_evaluator_setting(self, operation_type, setting, value):
        """Return a profile copy with one evaluator-owned setting replaced."""
        try:
            current = self.evaluators[operation_type]
        except KeyError:
            raise UnsupportedEvaluation(
                f"profile '{self.name}' has no evaluator for operation type "
                f"'{operation_type}'"
            )
        evaluators = dict(self.evaluators)
        settings = deepcopy(current.settings)
        settings[setting] = deepcopy(value)
        evaluators[operation_type] = EvaluatorSelection(
            evaluator_id=current.evaluator_id,
            settings=settings,
        )
        return EvaluationProfile(
            name=self.name,
            version=self.version,
            evaluators=evaluators,
            placement_policy=self.placement_policy,
            movement_policy=self.movement_policy,
            transfer_model=self.transfer_model,
        )

    def evaluator_id_for(self, operation_type):
        try:
            return self.evaluators[operation_type].evaluator_id
        except KeyError:
            raise UnsupportedEvaluation(
                f"profile '{self.name}' has no evaluator for operation type "
                f"'{operation_type}'"
            )

    def evaluator_settings_for(self, operation_type):
        try:
            return deepcopy(self.evaluators[operation_type].settings)
        except KeyError:
            raise UnsupportedEvaluation(
                f"profile '{self.name}' has no evaluator for operation type "
                f"'{operation_type}'"
            )

    def canonical_dict(self):
        return {
            "profile": self.name,
            "version": self.version,
            "evaluators": {
                operation_type: selection.canonical_value()
                for operation_type, selection in sorted(self.evaluators.items())
            },
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
    parsed_evaluators = {}
    for operation_type, evaluator_entry in evaluators.items():
        if not isinstance(operation_type, str) or not operation_type:
            raise UnsupportedEvaluation("evaluator operation types must be non-empty")
        if isinstance(evaluator_entry, str):
            evaluator_id = evaluator_entry
            settings = {}
        elif isinstance(evaluator_entry, dict):
            evaluator_id = evaluator_entry.get("id")
            settings = evaluator_entry.get("settings", {})
            if not isinstance(settings, dict):
                raise UnsupportedEvaluation(
                    f"evaluator settings for '{operation_type}' must be an object"
                )
        else:
            evaluator_id = None
            settings = {}
        if not isinstance(evaluator_id, str) or not evaluator_id:
            raise UnsupportedEvaluation(
                f"evaluator for '{operation_type}' must name a non-empty id"
            )
        parsed_evaluators[operation_type] = EvaluatorSelection(
            evaluator_id=evaluator_id,
            settings=deepcopy(settings),
        )
    for field in ("placement_policy", "movement_policy", "transfer_model"):
        value = entry.get(field)
        if not isinstance(value, str) or not value:
            raise UnsupportedEvaluation(f"evaluation profile {field} must be a string")
    return EvaluationProfile(
        name=name,
        version=version,
        evaluators=parsed_evaluators,
        placement_policy=entry["placement_policy"],
        movement_policy=entry["movement_policy"],
        transfer_model=entry["transfer_model"],
    )


def load_evaluation_profile(path=None):
    profile_path = Path(path) if path else DEFAULT_PROFILE_PATH
    with profile_path.open(encoding="utf-8") as file:
        return parse_evaluation_profile(json.load(file))
