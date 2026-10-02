"""Validate remote original-vs-modular CarbonPATH campaign results.

This command only reads completed campaign artifacts. It never runs annealing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from statistics import median

from system.utils.ArchitectureIdentity import (
    compatible_result_fingerprint,
    physical_architecture_fingerprint,
)


DEFAULT_RUNS = 20
DEFAULT_REQUIRED_MATCHES = 16
DEFAULT_RELATIVE_TOLERANCE = 1e-3
DEFAULT_ABSOLUTE_TOLERANCE = 1e-9


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _run_directory(root, workload_id, run):
    return root / "runs" / f"workload_{workload_id}" / f"run_{run:02d}"


def _load_runs(root, manifest, expected_runs):
    if manifest.get("runs_per_workload") != expected_runs:
        raise ValueError(
            f"{root} must contain exactly {expected_runs} runs per workload"
        )
    by_workload = {}
    for workload_id in manifest["workloads"]:
        records = []
        for run in range(1, expected_runs + 1):
            directory = _run_directory(root, workload_id, run)
            result_path = directory / "result.json"
            architecture_path = directory / "best_architecture.json"
            if not result_path.exists() or not architecture_path.exists():
                raise ValueError(f"missing completed run artifacts in {directory}")
            result = load_json(result_path)
            if result.get("workload_id") != workload_id or result.get("run") != run:
                raise ValueError(f"run metadata does not match {directory}")
            profile_path = directory / "best_profile.json"
            profile = load_json(profile_path) if profile_path.exists() else None
            if manifest.get("evaluation_flow") == "modular" and profile is None:
                raise ValueError(f"modular run is missing {profile_path}")
            architecture = load_json(architecture_path)
            score = float(result["verified_best_cost"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite score in {result_path}")
            records.append({
                "run": run,
                "initial_seed": result["initial_seed"],
                "search_seed": result["search_seed"],
                "score": score,
                "complete_fingerprint": compatible_result_fingerprint(
                    architecture, profile
                ),
                "physical_fingerprint": physical_architecture_fingerprint(
                    architecture
                ),
            })
        seed_pairs = {
            (record["initial_seed"], record["search_seed"]) for record in records
        }
        initial_seeds = {record["initial_seed"] for record in records}
        search_seeds = {record["search_seed"] for record in records}
        if not (
            len(seed_pairs) == expected_runs
            and len(initial_seeds) == expected_runs
            and len(search_seeds) == expected_runs
        ):
            raise ValueError(
                f"workload {workload_id} does not contain {expected_runs} "
                "independently seeded runs"
            )
        by_workload[workload_id] = records
    return by_workload


def _check_campaign_pair(original_manifest, modular_manifest):
    if original_manifest["workloads"] != modular_manifest["workloads"]:
        raise ValueError("campaign workload lists differ")
    if original_manifest.get("schedule_name") != modular_manifest.get("schedule_name"):
        raise ValueError("campaign annealing schedules differ")
    for workload_id in original_manifest["workloads"]:
        key = str(workload_id)
        original_hash = original_manifest.get("workload_config_sha256", {}).get(key)
        modular_hash = modular_manifest.get("workload_config_sha256", {}).get(key)
        if not original_hash or original_hash != modular_hash:
            raise ValueError(f"workload {workload_id} input definitions differ")


def compare_campaigns(
    original_root,
    modular_root,
    *,
    expected_runs=DEFAULT_RUNS,
    required_matches=DEFAULT_REQUIRED_MATCHES,
    relative_tolerance=DEFAULT_RELATIVE_TOLERANCE,
    absolute_tolerance=DEFAULT_ABSOLUTE_TOLERANCE,
):
    """Compare completed campaigns and return a machine-readable gate report."""
    original_root = Path(original_root)
    modular_root = Path(modular_root)
    original_manifest = load_json(original_root / "manifest.json")
    modular_manifest = load_json(modular_root / "manifest.json")
    _check_campaign_pair(original_manifest, modular_manifest)
    original_runs = _load_runs(original_root, original_manifest, expected_runs)
    modular_runs = _load_runs(modular_root, modular_manifest, expected_runs)

    workload_reports = []
    for workload_id in original_manifest["workloads"]:
        original = original_runs[workload_id]
        modular = modular_runs[workload_id]
        counts = Counter(record["complete_fingerprint"] for record in original)
        dominant_count = max(counts.values())
        candidates = sorted(
            fingerprint for fingerprint, count in counts.items()
            if count == dominant_count
        )
        # Stable tie-break: best median score, then fingerprint.
        target_fingerprint = min(
            candidates,
            key=lambda fingerprint: (
                median(
                    record["score"] for record in original
                    if record["complete_fingerprint"] == fingerprint
                ),
                fingerprint,
            ),
        )
        target_records = [
            record for record in original
            if record["complete_fingerprint"] == target_fingerprint
        ]
        target_score = median(record["score"] for record in target_records)
        target_physical = Counter(
            record["physical_fingerprint"] for record in target_records
        ).most_common(1)[0][0]

        exact = 0
        equivalent = 0
        details = []
        for record in modular:
            exact_match = record["complete_fingerprint"] == target_fingerprint
            equivalent_match = (
                not exact_match
                and record["physical_fingerprint"] == target_physical
                and math.isclose(
                    record["score"], target_score,
                    rel_tol=relative_tolerance,
                    abs_tol=absolute_tolerance,
                )
            )
            exact += int(exact_match)
            equivalent += int(equivalent_match)
            details.append({
                "run": record["run"],
                "score": record["score"],
                "exact_match": exact_match,
                "equivalent_score_and_hardware": equivalent_match,
                "passed": exact_match or equivalent_match,
            })
        matches = exact + equivalent
        workload_reports.append({
            "workload_id": workload_id,
            "dominant_original_fingerprint": target_fingerprint,
            "dominant_original_runs": dominant_count,
            "dominant_original_score": target_score,
            "dominant_original_physical_fingerprint": target_physical,
            "exact_match_runs": exact,
            "equivalent_score_runs": equivalent,
            "matching_runs": matches,
            "required_matches": required_matches,
            "passed": matches >= required_matches,
            "runs": details,
        })
    return {
        "passed": all(report["passed"] for report in workload_reports),
        "expected_runs_per_workload": expected_runs,
        "required_matches_per_workload": required_matches,
        "relative_score_tolerance": relative_tolerance,
        "absolute_score_tolerance": absolute_tolerance,
        "original_root": str(original_root),
        "modular_root": str(modular_root),
        "workloads": workload_reports,
    }


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "equivalence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "equivalence_runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "workload_id", "run", "score", "exact_match",
                "equivalent_score_and_hardware", "passed",
            ),
        )
        writer.writeheader()
        for workload in report["workloads"]:
            for run in workload["runs"]:
                writer.writerow({"workload_id": workload["workload_id"], **run})
    lines = [
        "# Modular CarbonPATH equivalence gate",
        "",
        f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "A run passes by returning the dominant original result, or the same "
        "physical architecture and an equivalent score when only GEMM mapping differs.",
        "",
        "| Workload | Exact | Mapping-equivalent | Total | Required | Result |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for workload in report["workloads"]:
        lines.append(
            f"| {workload['workload_id']} | {workload['exact_match_runs']} | "
            f"{workload['equivalent_score_runs']} | {workload['matching_runs']} | "
            f"{workload['required_matches']} | "
            f"{'PASS' if workload['passed'] else 'FAIL'} |"
        )
    (output / "equivalence_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--original-root", type=Path, required=True)
    command.add_argument("--modular-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--expected-runs", type=int, default=DEFAULT_RUNS)
    command.add_argument("--required-matches", type=int, default=DEFAULT_REQUIRED_MATCHES)
    command.add_argument(
        "--relative-tolerance", type=float, default=DEFAULT_RELATIVE_TOLERANCE
    )
    command.add_argument(
        "--absolute-tolerance", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE
    )
    return command


def main():
    args = parser().parse_args()
    if not 0 < args.required_matches <= args.expected_runs:
        raise SystemExit("--required-matches must be between 1 and --expected-runs")
    report = compare_campaigns(
        args.original_root,
        args.modular_root,
        expected_runs=args.expected_runs,
        required_matches=args.required_matches,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    write_report(report, args.output)
    print(json.dumps({
        "passed": report["passed"],
        "report": str(args.output / "equivalence_report.json"),
    }, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
