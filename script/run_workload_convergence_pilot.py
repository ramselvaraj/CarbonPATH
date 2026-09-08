from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main import WORKLOAD_CONFIGS, parse_workload_entry
from script.run_full_space_convergence import (
    INTERMEDIATE_POLICY,
    calibrate_workload,
    initialize_experiment_cache,
    run_one,
)
from script.run_optimizer_experiments import BASE_CACHE, load_json
from system.utils.ArchitectureIdentity import canonical_architecture_fingerprint


OUTPUT_ROOT = Path("reports/workload_convergence_pilot")
SCHEDULE = "requested_4000_75650"
RUNS = 1


def load_completed_runs(output_dir, panel, schedule):
    rows = []
    run_root = output_dir / "runs" / panel["name"] / schedule
    for result_path in sorted(run_root.glob("run_*/result.json")):
        row = load_json(result_path)
        architecture = load_json(result_path.parent / "best_architecture.json")
        row["canonical_fingerprint"] = canonical_architecture_fingerprint(architecture)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("run", ignore_index=True)


def write_report(output_dir, workload_id, workload, schedule, expected_runs):
    results = load_completed_runs(
        output_dir,
        {"name": f"workload_{workload_id}_pilot"},
        schedule,
    )
    if len(results) != expected_runs:
        return

    results.to_csv(output_dir / "runs.csv", index=False)
    counts = results["canonical_fingerprint"].value_counts()
    report = [
        f"# Workload {workload_id} Architecture Convergence Pilot",
        "",
        f"- Workload: `{workload}`",
        f"- Schedule: `{schedule}` ({expected_runs} independent runs)",
        f"- Intermediate policy: `{INTERMEDIATE_POLICY}`",
        f"- Exact canonical agreement: `{counts.iloc[0]}/{expected_runs}`",
        f"- Result: **{'READ ONLY' if expected_runs < 2 else ('PASS' if len(counts) == 1 else 'FAIL')}**",
        "",
        "| Canonical fingerprint | Runs | Median objective | Best objective | Worst objective |",
        "|---|---:|---:|---:|---:|",
    ]
    for fingerprint, group in results.groupby("canonical_fingerprint", sort=False):
        report.append(
            f"| `{fingerprint}` | {len(group)} | {group.best_cost.median():.12g} | "
            f"{group.best_cost.min():.12g} | {group.best_cost.max():.12g} |"
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if expected_runs >= 2 and len(counts) != 1:
        raise RuntimeError(
            f"Workload {workload_id} did not converge to one canonical architecture: {counts.to_dict()}"
        )


def run_workload(
    workload_id: int,
    output_root: Path,
    first_run: int,
    runs: int,
    expected_runs: int,
    schedule: str,
) -> None:
    output_dir = output_root / f"workload_{workload_id}"
    shared_cache = output_dir / "base_cache.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not shared_cache.exists():
        initialize_experiment_cache(BASE_CACHE, shared_cache)

    worker_cache = output_dir / "workers" / f"cache_run_{first_run:02d}.csv"
    worker_cache.parent.mkdir(parents=True, exist_ok=True)
    if not worker_cache.exists():
        initialize_experiment_cache(shared_cache, worker_cache)

    workload = parse_workload_entry(workload_id, WORKLOAD_CONFIGS[workload_id])
    calibration_path = calibrate_workload(
        output_dir=output_dir / "calibration",
        workload_id=workload_id,
        workload=workload,
        samples=10,
        seed=10000 + workload_id,
        base_cache=worker_cache,
    )
    panel = {
        "name": f"workload_{workload_id}_pilot",
        "workload_id": workload_id,
        "initial_seed": 7000,
        "search_seed": 8000,
    }
    last_run = first_run + runs - 1
    for run_number in range(first_run, last_run + 1):
        print(f"workload {workload_id}: run {run_number}/{expected_runs}", flush=True)
        run_one(
            output_dir=output_dir,
            panel=panel,
            run_index=run_number - 1,
            schedule_name=schedule,
            workload=workload,
            calibration_path=calibration_path,
            base_cache=worker_cache,
        )
    write_report(output_dir, workload_id, workload, schedule, expected_runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=int, choices=range(1, 10), required=True)
    parser.add_argument("--first-run", type=int, default=1)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--schedule", default=SCHEDULE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    expected_runs = args.expected_runs or args.runs
    if args.first_run < 1 or args.runs < 1 or expected_runs < args.first_run + args.runs - 1:
        parser.error("run range must be positive and fit within --expected-runs")
    run_workload(
        args.workload,
        args.output_root,
        args.first_run,
        args.runs,
        expected_runs,
        args.schedule,
    )


if __name__ == "__main__":
    main()
