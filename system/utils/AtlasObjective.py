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

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


DEFAULT_ATLAS_OBJECTIVE_PATH = Path("cfg/parameters/atlas_objective.json")
DEFAULT_COST_PROFILES_PATH = Path("cfg/parameters/cost_profiles.json")

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
    diagnostics: dict = field(default_factory=dict, compare=False, repr=False)

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
        diagnostics = raw.pop("diagnostics")
        raw.update(diagnostics)
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


_CALIBRATED_METRICS = {
    "energy": ("total_energy_pj", "avg_energy"),
    "area": ("area_mm2", "avg_area"),
    "cost": ("cost_usd", "avg_dollar_cost"),
    "latency": ("latency_ns", "avg_latency"),
    "embCarbon": ("embodied_carbon_kg", "avg_embCarbon"),
    "opeCarbon": ("operational_carbon_kg", "avg_opeCarbon"),
}

_COEFFICIENT_KEYS = {
    "energy": "energy_coff",
    "area": "area_coeff",
    "cost": "cost_coeff",
    "latency": "perf_coeff",
    "embCarbon": "embc_coeff",
    "opeCarbon": "opec_coeff",
}


class CalibratedCarbonPathObjective(AtlasObjective):
    """The original t1-t4 scoring formula behind the modular objective seam."""

    NORMALIZATION_MODES = ("avg", "max", "max_minus_min", "mean_std", "min_median")

    def __init__(self, profile_name, coefficients, calibration, normalization_mode):
        if profile_name not in ("t1", "t2", "t3", "t4"):
            raise UnsupportedEvaluation(f"unknown CarbonPATH objective '{profile_name}'")
        if normalization_mode not in self.NORMALIZATION_MODES:
            raise UnsupportedEvaluation(
                f"unknown objective normalization mode '{normalization_mode}'"
            )
        self.objective_id = profile_name
        self.coefficients = {
            metric: float(coefficients[_COEFFICIENT_KEYS[metric]])
            for metric in _CALIBRATED_METRICS
        }
        self.calibration = dict(calibration)
        self.normalization_mode = normalization_mode
        self._validate_calibration()

    def _validate_calibration(self):
        required = set()
        for metric, (_, average_key) in _CALIBRATED_METRICS.items():
            required.add(average_key)
            required.update(
                f"{metric}_{statistic}"
                for statistic in ("min", "max", "mean", "stddev", "median")
            )
        missing = sorted(required - self.calibration.keys())
        if missing:
            raise UnsupportedEvaluation(
                "objective calibration is missing: " + ", ".join(missing)
            )
        for key in required:
            value = self.calibration[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise UnsupportedEvaluation(
                    f"objective calibration value '{key}' must be numeric"
                )
        for metric in _CALIBRATED_METRICS:
            minimum = self.calibration[f"{metric}_min"]
            maximum = self.calibration[f"{metric}_max"]
            if minimum <= 0 or maximum < minimum:
                raise UnsupportedEvaluation(
                    f"objective calibration range for '{metric}' is invalid"
                )

    def _normalized_value(self, metric, value):
        _, average_key = _CALIBRATED_METRICS[metric]
        mode = self.normalization_mode
        if mode == "avg":
            denominator = self.calibration[average_key]
            numerator = value
        elif mode == "max":
            denominator = self.calibration[f"{metric}_max"]
            numerator = value
        elif mode == "max_minus_min":
            minimum = self.calibration[f"{metric}_min"]
            denominator = self.calibration[f"{metric}_max"] - minimum
            numerator = value - minimum
        elif mode == "mean_std":
            denominator = self.calibration[f"{metric}_stddev"]
            numerator = value - self.calibration[f"{metric}_mean"]
        else:
            denominator = self.calibration[f"{metric}_median"]
            numerator = value - self.calibration[f"{metric}_min"]
        if denominator == 0:
            return 0.0 if numerator == 0 else float("inf")
        return numerator / denominator

    def normalized_metrics(self, design_point):
        if not isinstance(design_point, AtlasDesignPoint):
            raise ValueError("objective requires an AtlasDesignPoint")
        return {
            metric: self._normalized_value(
                metric, design_point.metric(design_point_metric)
            )
            for metric, (design_point_metric, _) in _CALIBRATED_METRICS.items()
        }

    def score(self, design_point):
        normalized = self.normalized_metrics(design_point)
        return sum(
            self.coefficients[metric]
            * (self.calibration[f"{metric}_max"] / self.calibration[f"{metric}_min"])
            * normalized[metric]
            for metric in _CALIBRATED_METRICS
        )

    def canonical_dict(self):
        return {
            "objective": self.objective_id,
            "normalization_mode": self.normalization_mode,
            "coefficients": dict(sorted(self.coefficients.items())),
            "calibration_identity": self.calibration.get("_calibration_identity"),
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
    objective_id=None,
    config=None,
    config_path=DEFAULT_ATLAS_OBJECTIVE_PATH,
    calibration=None,
    normalization_mode=None,
    cost_profiles_path=DEFAULT_COST_PROFILES_PATH,
):
    """Resolve an objective ID plus coefficients from configuration."""
    config = config if config is not None else load_atlas_objective_config(config_path)
    objective_id = objective_id or config.get("default")
    if not isinstance(objective_id, str) or not objective_id:
        raise UnsupportedEvaluation("atlas objective config must name a default")
    if objective_id in ("t1", "t2", "t3", "t4"):
        calibration = calibration or config.get("calibration")
        normalization_mode = normalization_mode or config.get("normalization_mode")
        if not isinstance(calibration, dict):
            raise UnsupportedEvaluation(
                f"objective '{objective_id}' requires calibration metrics"
            )
        if not normalization_mode:
            raise UnsupportedEvaluation(
                f"objective '{objective_id}' requires a normalization mode"
            )
        with Path(cost_profiles_path).open(encoding="utf-8") as file:
            profiles = json.load(file)
        return CalibratedCarbonPathObjective(
            objective_id,
            profiles[objective_id],
            calibration,
            normalization_mode,
        )
    try:
        objective_class = ATLAS_OBJECTIVES[objective_id]
    except KeyError:
        raise UnsupportedEvaluation(f"unknown atlas objective '{objective_id}'")
    coefficients = (
        config.get("objectives", {}).get(objective_id, {}).get("coefficients")
    )
    return objective_class(coefficients=coefficients)
