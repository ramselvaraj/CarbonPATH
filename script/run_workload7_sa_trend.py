from __future__ import annotations

import argparse
import math
from pathlib import Path
import platform

import pandas as pd

from main import WORKLOAD_CONFIGS, parse_workload_entry
from script.run_full_space_convergence import (
    FULL_SEARCH_SPACE,
    INTERMEDIATE_POLICY,
    SCHEDULES,
    calibrate_workload,
    file_sha256,
    initialize_experiment_cache,
    planned_move_count,
    run_one,
    summarize_runs,
    write_json,
)


STAGE_ONE_SCHEDULES = (
    "thorough_4800",
    "slow_320_5632",
    "slow_640_5824",
)
INTENSIVE_SCHEDULES = {
    "slow_320_5632": "intensive_320_11264",
    "slow_640_5824": "intensive_640_11648",
}
SECONDARY_BASIN = -0.21223282206723632
PANEL = {
    "name": "workload_7_trend",
    "workload_id": 7,
    "runs": 12,
    "initial_seed": 11100,
    "search_seed": 11200,
}


def add_weak_basin_rate(summary, runs):
    weak_rates = {}
    for schedule, group in runs.groupby("schedule"):
        weak_rates[schedule] = sum(
            math.isclose(cost, SECONDARY_BASIN, rel_tol=1e-9, abs_tol=1e-9)
            for cost in group["best_cost"]
        ) / len(group)
    result = summary.copy()
    result["weak_basin_rate"] = result["schedule"].map(weak_rates)
    return result


def select_stage_one_winner(summary):
    candidates = summary.loc[summary["schedule"].isin(INTENSIVE_SCHEDULES)]
    winner = min(
        candidates.itertuples(index=False),
        key=lambda row: (
            row.median_cost,
            -row.within_10_percent,
            row.maximum_cost,
            row.late_improvement_rate,
        ),
    )
    return winner.schedule


def format_percent(value):
    return f"{100 * value:.1f}%"


def write_trend_report(
    output_dir, runs, summary, paired, stage_one_winner, intensive_schedule
):
    indexed = summary.set_index("schedule")
    control = indexed.loc["thorough_4800"]
    winner = indexed.loc[stage_one_winner]
    intensive = indexed.loc[intensive_schedule]
    checks = {
        "Stage 1 winner median beats paired control": (
            winner["median_cost"] < control["median_cost"]
        ),
        "Intensive median beats paired control": (
            intensive["median_cost"] < control["median_cost"]
        ),
        "Intensive median beats Stage 1 winner": (
            intensive["median_cost"] < winner["median_cost"]
        ),
        "Intensive within-10% rate exceeds 66.7%": (
            intensive["within_10_percent"] > 2 / 3
        ),
        "Intensive weak-basin rate is below 33.3%": (
            intensive["weak_basin_rate"] < 1 / 3
        ),
        "Intensive worst case beats paired control": (
            intensive["maximum_cost"] < control["maximum_cost"]
        ),
        "Intensive late-improvement rate is below 20%": (
            intensive["late_improvement_rate"] < 0.2
        ),
    }
    positive_checks = sum(checks.values())
    elapsed_seconds = float(runs["wall_seconds_with_evaluation"].sum())
    lines = [
        "# Workload-7 Simulated-Annealing Trend Test",
        "",
        "## Test Design",
        "",
        "Twelve new workload-7 initial architectures were each searched with the existing 4,800-move schedule and two slower-cooling schedules. The slower schedule with the lower median objective was selected for an intensive run on the same starts. All comparisons therefore use paired initial architectures and search seeds.",
        "",
        f"- Search space: `{FULL_SEARCH_SPACE}`.",
        f"- Intermediate policy: `{INTERMEDIATE_POLICY}`.",
        "- The current D2D protocol implementation was intentionally unchanged.",
        f"- Total measured runtime: {elapsed_seconds / 60:.1f} minutes.",
        f"- Stage 1 winner: `{stage_one_winner}`.",
        f"- Stage 2 schedule: `{intensive_schedule}`.",
        "",
        "## Schedules",
        "",
        "| Schedule | T0 | Tf | Moves/temperature | Cooling | Total moves |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for schedule_name in (*STAGE_ONE_SCHEDULES, intensive_schedule):
        schedule = SCHEDULES[schedule_name]
        lines.append(
            f"| `{schedule_name}` | {schedule['initial_temp']:g} | "
            f"{schedule['freezing_temp']:g} | "
            f"{schedule['max_move_per_temp_step']} | "
            f"{schedule['cooling_rate']:g} | {planned_move_count(schedule)} |"
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "`best_observed` is the lowest objective found across all 48 runs. Lower is better.",
            "",
            "| Schedule | Runs | Best | Median | Maximum | Within 5% | Within 10% | Weak basin | Late improvement |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.schedule}` | {row.runs} | {row.minimum_cost:.6g} | "
            f"{row.median_cost:.6g} | {row.maximum_cost:.6g} | "
            f"{format_percent(row.within_5_percent)} | "
            f"{format_percent(row.within_10_percent)} | "
            f"{format_percent(row.weak_basin_rate)} | "
            f"{format_percent(row.late_improvement_rate)} |"
        )
    lines.extend(
        [
            "",
            "## Paired Changes",
            "",
            "A positive objective reduction means the later schedule improved the result for that start.",
            "",
            f"- Median control-to-Stage-1 reduction: {paired['control_to_stage_one'].median():.6g}.",
            f"- Median Stage-1-to-intensive reduction: {paired['stage_one_to_intensive'].median():.6g}.",
            f"- Intensive improved over Stage 1 on {format_percent(paired['stage_one_to_intensive'].gt(0).mean())} of starts.",
            "",
            "## Trend Assessment",
            "",
        ]
    )
    for description, passed in checks.items():
        lines.append(f"- {description}: **{'yes' if passed else 'no'}**.")
    lines.extend(
        [
            "",
            f"Positive checks: **{positive_checks}/{len(checks)}**.",
            "",
            "This test measures empirical improvement across paired full-space searches. It does not prove an optimum.",
        ]
    )
    (Path(output_dir) / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the paired workload-7 slower-cooling trend test."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/workload_7_sa_trend"),
    )
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=Path("cfg/static_cache/static_cache.csv"),
    )
    parser.add_argument("--calibration-samples", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_cache = args.output_dir / "simulation_cache.csv"
    if not experiment_cache.exists():
        initialize_experiment_cache(args.base_cache, experiment_cache)

    workload = parse_workload_entry(7, WORKLOAD_CONFIGS[7])
    calibration_path = calibrate_workload(
        output_dir=args.output_dir / "calibration",
        workload_id=7,
        workload=workload,
        samples=args.calibration_samples,
        seed=11900,
        base_cache=experiment_cache,
    )
    manifest = {
        "python": platform.python_version(),
        "full_search_space": str(FULL_SEARCH_SPACE),
        "full_search_space_sha256": file_sha256(FULL_SEARCH_SPACE),
        "intermediate_policy": INTERMEDIATE_POLICY,
        "d2d_protocol_fix_skipped": True,
        "panel": PANEL,
        "stage_one_schedules": list(STAGE_ONE_SCHEDULES),
    }
    write_json(args.output_dir / "manifest.json", manifest)

    rows = []
    for run_index in range(PANEL["runs"]):
        for schedule_name in STAGE_ONE_SCHEDULES:
            print(
                f"[Stage 1 {len(rows) + 1}/36] run {run_index + 1} "
                f"{schedule_name}",
                flush=True,
            )
            rows.append(
                run_one(
                    output_dir=args.output_dir,
                    panel=PANEL,
                    run_index=run_index,
                    schedule_name=schedule_name,
                    workload=workload,
                    calibration_path=calibration_path,
                    base_cache=experiment_cache,
                )
            )
            pd.DataFrame(rows).to_csv(
                args.output_dir / "stage_one_runs.csv", index=False
            )

    stage_one_runs = pd.DataFrame(rows)
    stage_one_summary = add_weak_basin_rate(
        summarize_runs(stage_one_runs), stage_one_runs
    )
    stage_one_winner = select_stage_one_winner(stage_one_summary)
    intensive_schedule = INTENSIVE_SCHEDULES[stage_one_winner]
    manifest["stage_one_winner"] = stage_one_winner
    manifest["intensive_schedule"] = intensive_schedule
    write_json(args.output_dir / "manifest.json", manifest)
    stage_one_summary.to_csv(
        args.output_dir / "stage_one_summary.csv", index=False
    )
    print(
        f"Stage 1 winner: {stage_one_winner}; Stage 2: {intensive_schedule}",
        flush=True,
    )

    for run_index in range(PANEL["runs"]):
        print(
            f"[Stage 2 {run_index + 1}/12] run {run_index + 1} "
            f"{intensive_schedule}",
            flush=True,
        )
        rows.append(
            run_one(
                output_dir=args.output_dir,
                panel=PANEL,
                run_index=run_index,
                schedule_name=intensive_schedule,
                workload=workload,
                calibration_path=calibration_path,
                base_cache=experiment_cache,
            )
        )
        pd.DataFrame(rows).to_csv(args.output_dir / "runs.csv", index=False)

    runs = pd.DataFrame(rows)
    summary = add_weak_basin_rate(summarize_runs(runs), runs)
    paired = runs.pivot(index="run", columns="schedule", values="best_cost")
    paired["control_to_stage_one"] = (
        paired["thorough_4800"] - paired[stage_one_winner]
    )
    paired["stage_one_to_intensive"] = (
        paired[stage_one_winner] - paired[intensive_schedule]
    )
    paired.reset_index().to_csv(args.output_dir / "paired_results.csv", index=False)
    summary.to_csv(args.output_dir / "schedule_summary.csv", index=False)
    write_trend_report(
        args.output_dir,
        runs,
        summary,
        paired,
        stage_one_winner,
        intensive_schedule,
    )
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
