"""Calibration summaries for modular CarbonPATH design-point evaluations."""

from __future__ import annotations

import math
import statistics


_METRICS = {
    "energy": ("total_energy_pj", "avg_energy"),
    "area": ("area_mm2", "avg_area"),
    "cost": ("cost_usd", "avg_dollar_cost"),
    "latency": ("latency_ns", "avg_latency"),
    "embCarbon": ("embodied_carbon_kg", "avg_embCarbon"),
    "opeCarbon": ("operational_carbon_kg", "avg_opeCarbon"),
}


def summarize_design_points(design_points, calibration_identity, model_version):
    """Return the calibration schema consumed by the t1--t4 objectives."""
    points = tuple(design_points)
    if not points:
        raise ValueError("calibration requires at least one design point")

    summary = {
        "_calibration_model_version": model_version,
        "_calibration_identity": calibration_identity,
        "_calibration_samples": len(points),
    }
    for metric, (attribute, average_key) in _METRICS.items():
        values = [float(getattr(point, attribute)) for point in points]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"calibration metric {metric} must be finite")
        summary[average_key] = round(sum(values) / len(values), 2)
        summary[f"{metric}_min"] = min(values)
        summary[f"{metric}_max"] = max(values)
        summary[f"{metric}_stddev"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_median"] = statistics.median(values)
    return summary
