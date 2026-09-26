"""Validate that modular policy scores rank known-bad policies below good ones.

PLACEHOLDER CHECK (delete with cfg/profiles/placeholders/ and
system/utils/PlaceholderPolicies.py). It evaluates one fixed architecture and
ATLAS graph under the reference profile and every known-bad placeholder profile,
asserts the reference scores strictly best, and writes the score matrix.

Usage:
    python script/validate_atlas_policy_scores.py \
        --graph cfg/examples/atlas/dense_relu_funnel.graph_dump.json \
        --architecture cfg/examples/sa_fpga_architecture.json \
        --output-dir reports/atlas_policy_scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import evaluate_atlas_design_point  # noqa: E402
from system.utils.AtlasGraphAdapter import load_atlas_graph  # noqa: E402
from system.utils.AtlasObjective import build_atlas_objective  # noqa: E402
from system.utils.EvaluationProfile import load_evaluation_profile  # noqa: E402
from system.utils.SimulationCache import SimulationCache  # noqa: E402


DEFAULT_REFERENCE = "cfg/profiles/atlas_modular_v1.json"
DEFAULT_BAD = (
    "cfg/profiles/placeholders/ph_slow_gemm_v1.json",
    "cfg/profiles/placeholders/ph_dram_always_v1.json",
    "cfg/profiles/placeholders/ph_pessimistic_transfer_v1.json",
)


def _load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path, default=Path(DEFAULT_REFERENCE))
    parser.add_argument("--bad-profile", type=Path, action="append", default=None)
    parser.add_argument("--objective-config", type=Path, default=None)
    parser.add_argument("--cache-file", type=Path, default=Path("cfg/static_cache/static_cache.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/atlas_policy_scores"))
    args = parser.parse_args()

    bad_paths = args.bad_profile if args.bad_profile else [Path(p) for p in DEFAULT_BAD]
    profile_paths = [args.reference_profile, *bad_paths]
    architecture = _load_json(args.architecture)
    graph = load_atlas_graph(args.graph)
    objective = build_atlas_objective(
        config_path=args.objective_config or "cfg/parameters/atlas_objective.json"
    )

    rows = []
    with tempfile.TemporaryDirectory(prefix="carbonpath-atlas-scores-") as directory:
        cache_path = Path(directory) / "cache.csv"
        pd.DataFrame(
            columns=[
                "core_size", "data_flow", "bandwidth", "buffer_size",
                "M", "K", "N", "latency",
            ]
        ).to_csv(cache_path, index=False)
        if args.cache_file.exists():
            base = pd.read_csv(args.cache_file)
            base.to_csv(cache_path, index=False)
        cache = SimulationCache(cache_path, simulator_dir=str(Path(directory) / "sim"))
        for path in profile_paths:
            profile = load_evaluation_profile(path)
            design_point, score = evaluate_atlas_design_point(
                cache, architecture, graph, profile, objective
            )
            rows.append(
                {
                    "profile": profile.name,
                    "profile_fingerprint": profile.fingerprint(),
                    "objective": score,
                    "latency_ns": design_point.latency_ns,
                    "total_energy_pj": design_point.total_energy_pj,
                    "area_mm2": design_point.area_mm2,
                    "cost_usd": design_point.cost_usd,
                    "is_reference": path == args.reference_profile,
                }
            )

    results = pd.DataFrame(rows).sort_values("objective", ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "policy_scores.csv", index=False)
    print(results.to_string(index=False))

    reference_row = results.loc[results["is_reference"]].iloc[0]
    worse = results[(results["objective"] <= reference_row["objective"]) & (~results["is_reference"])]
    if not worse.empty:
        raise SystemExit(
            "FAIL: expected the reference profile to score strictly best, but "
            f"{list(worse['profile'])} scored at or below it"
        )
    print(f"PASS: reference scores strictly best at {reference_row['objective']:.6g}")


if __name__ == "__main__":
    main()
