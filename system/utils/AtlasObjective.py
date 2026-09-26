"""Replaceable objective models for modular ATLAS annealing.

The annealer needs one scalar per design point. This module owns that scalar and
nothing else: it never evaluates a graph, reads an architecture, or turns a
temperature. It receives an :class:`AtlasDesignPoint` and returns a number.

The default ``raw_weighted_sum_v0`` is intentionally simple and uncalibrated: a
weighted sum over raw design-point metrics read from
``cfg/parameters/atlas_objective.json``. It exists so annealing can run today; a
future calibrated objective (for example a ``calibrated_t1_v1``) is a new class
plus one registry entry, with no change to ``sim_annealing``.

This mirrors the replaceable-model registries used for evaluators, operation
placement, tensor movement, and transfer cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


DEFAULT_ATLAS_OBJECTIVE_PATH = Path("cfg/parameters/atlas_objective.json")

#: Raw metrics every design point must expose. A replacement objective may read
#: any subset of these through :meth:`AtlasDesignPoint.metric`.
RAW_METRICS = (
    "latency_ns",
    "total_energy_pj",
    "baseline_energy_pj",
    "compute_energy_pj",
    "movement_energy_pj",
    "power_w",
    "area_mm2",
    "cost_usd",
    "embodied_carbon_kg",
    "operational_carbon_kg",
)


@dataclass(frozen=True)
class AtlasDesignPoint:
    """One evaluated modular design point: identity plus raw metrics."""

    architecture_fingerprint: str
    profile_name: str
    profile_fingerprint: str
    latency_ns: float
    total_energy_pj: float
    baseline_energy_pj: float
    compute_energy_pj: float
    movement_energy_pj: float
    power_w: float
    area_mm2: float
    cost_usd: float
    embodied_carbon_kg: float
    operational_carbon_kg: float
    operation_count: int

    def metric(self, name):
        if name not in RAW_METRICS:
            raise UnsupportedEvaluation(f"unknown design-point metric '{name}'")
        return float(getattr(self, name))

    def metrics(self):
        return {name: self.metric(name) for name in RAW_METRICS}

    def raw_dict(self):
        """Legacy-flavoured raw metric keys plus the modular extras.

        The aliases (``latency``, ``energy``, ``area``, ``dollar``, ``embCarbon``,
        ``opeCarbon``) let existing trace/log helpers keep working unchanged.
        """
        raw = dict(asdict(self))
        raw.update(
            {
                "latency": self.latency_ns,
                "energy": self.total_energy_pj,
                "area": self.area_mm2,
                "dollar": self.cost_usd,
                "embCarbon": self.embodied_carbon_kg,
                "opeCarbon": self.operational_carbon_kg,
            }
        )
        return raw


class AtlasObjective:
    """Contract for one replaceable design-point objective."""

    objective_id = ""

    def score(self, design_point):  # pragma: no cover - interface
        raise NotImplementedError

    def canonical_dict(self):
        return {"objective": self.objective_id}


class RawWeightedSumObjective(AtlasObjective):
    """Uncalibrated weighted sum over raw metrics.

    Coefficients come from configuration. A coefficient of zero drops a metric;
    a positive coefficient makes a larger metric a worse score. This is a
    provisional default, not a claim that the weights are physically calibrated.
    """

    objective_id = "raw_weighted_sum_v0"

    DEFAULT_COEFFICIENTS = {"latency_ns": 1.0, "total_energy_pj": 1.0}

    def __init__(self, coefficients=None):
        coefficients = dict(coefficients or self.DEFAULT_COEFFICIENTS)
        cleaned = {}
        for name, value in coefficients.items():
            if name not in RAW_METRICS:
                raise UnsupportedEvaluation(
                    f"objective '{self.objective_id}' names unknown metric '{name}'"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise UnsupportedEvaluation(
                    f"objective coefficient for '{name}' must be numeric"
                )
            cleaned[name] = float(value)
        self.coefficients = cleaned

    def score(self, design_point):
        if not isinstance(design_point, AtlasDesignPoint):
            raise ValueError("objective requires an AtlasDesignPoint")
        return sum(
            coefficient * design_point.metric(name)
            for name, coefficient in self.coefficients.items()
        )

    def canonical_dict(self):
        return {
            "objective": self.objective_id,
            "coefficients": dict(sorted(self.coefficients.items())),
        }


ATLAS_OBJECTIVES = {
    RawWeightedSumObjective.objective_id: RawWeightedSumObjective,
}


def load_atlas_objective_config(config_path=DEFAULT_ATLAS_OBJECTIVE_PATH):
    path = Path(config_path)
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise UnsupportedEvaluation("atlas objective config must be an object")
    return config


def build_atlas_objective(
    objective_id=None, config=None, config_path=DEFAULT_ATLAS_OBJECTIVE_PATH
):
    """Resolve an objective ID plus coefficients from configuration."""
    config = config if config is not None else load_atlas_objective_config(config_path)
    objective_id = objective_id or config.get("default")
    if not isinstance(objective_id, str) or not objective_id:
        raise UnsupportedEvaluation("atlas objective config must name a default")
    try:
        objective_class = ATLAS_OBJECTIVES[objective_id]
    except KeyError:
        raise UnsupportedEvaluation(f"unknown atlas objective '{objective_id}'")
    coefficients = (
        config.get("objectives", {}).get(objective_id, {}).get("coefficients")
    )
    return objective_class(coefficients=coefficients)
