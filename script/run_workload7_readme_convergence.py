from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import pandas as pd

from main import WORKLOAD_CONFIGS, parse_workload_entry
from script.run_full_space_convergence import (
    INTERMEDIATE_POLICY,
    SCHEDULES,
    calibrate_workload,
    file_sha256,
    initialize_experiment_cache,
    run_one,
    summarize_runs,
)
from script.run_workload7_sa_trend import PANEL


OUTPUT_DIR = Path("reports/workload_7_readme_convergence")
SCHEDULE = "requested_4000_75650"
START_INDEX = 1
END_INDEX = 12
FIXED_REFERENCE = -4.799668202228307
OLD_TREND_DIR = Path("reports/workload_7_sa_trend")


def copy_run_one():
    source = OLD_TREND_DIR / "runs/workload_7_trend/requested_4000_75650/run_01"
    destination = OUTPUT_DIR / "runs/workload_7_readme/requested_4000_75650/run_01"
    if not (destination / "result.json").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)


def worker(first, last, worker_cache):
    output_dir = OUTPUT_DIR
    workload = parse_workload_entry(7, WORKLOAD_CONFIGS[7])
    calibration_path = output_dir / "calibration/workload_7/calibration.json"
    panel = {**PANEL, "name": "workload_7_readme"}
    rows = []
    for run_index in range(first, last + 1):
        print(f"worker run {run_index}/{END_INDEX}", flush=True)
        rows.append(
            run_one(
                output_dir=output_dir,
                panel=panel,
                run_index=run_index,
                schedule_name=SCHEDULE,
                workload=workload,
                calibration_path=calibration_path,
                base_cache=worker_cache,
            )
        )
        pd.DataFrame(rows).to_csv(
            output_dir / "workers" / f"worker_{first}_{last}.csv", index=False
        )


def load_rows():
    rows = []
    source = OUTPUT_DIR / "runs/workload_7_readme/requested_4000_75650"
    for result_path in sorted(source.glob("run_*/result.json")):
        with result_path.open(encoding="utf-8") as file:
            rows.append(json.load(file))
    return pd.DataFrame(rows)


def diagnostics(runs):
    rows = []
    trace_root = OUTPUT_DIR / "runs/workload_7_readme/requested_4000_75650"
    for run in runs.itertuples(index=False):
        trace = pd.read_csv(trace_root / f"run_{run.run:02d}" / "search_trace.csv")
        first_temperature = trace.iloc[0]["temperature"]
        first = trace.loc[trace["temperature"] == first_temperature]
        uphill = first.loc[first["cost_diff"] > 0]
        late = trace.iloc[-max(1, len(trace) // 5) :]
        invalid = int(trace["move_accepted"].isna().sum())
        rows.append(
            {
                "run": run.run,
                "attempted_moves": len(trace),
                "accepted_moves": int(trace["move_accepted"].eq(True).sum()),
                "invalid_mutations": invalid,
                "invalid_mutation_rate": invalid / len(trace),
                "first_temperature_acceptance": float(
                    first["move_accepted"].eq(True).mean()
                ),
                "first_temperature_uphill_acceptance": float(
                    uphill["move_accepted"].eq(True).mean()
                ),
                "late_acceptance": float(late["move_accepted"].eq(True).mean()),
                "last_improvement_move": run.last_improvement_move,
                "late_improvement": run.late_improvement,
                "simulator_calls": run.simulator_calls,
            }
        )
    return pd.DataFrame(rows)


def relative_gap(value, reference):
    return (value - reference) / max(abs(reference), 1e-12)


def write_report(runs, summary, pair, diagnostic, elapsed):
    reference = min(FIXED_REFERENCE, float(runs.best_cost.min()))
    summary = summary.copy()
    summary["within_5_percent_fixed"] = summary["best_observed"].map(
        lambda _: float(runs.best_cost.map(lambda value: relative_gap(value, FIXED_REFERENCE) <= 0.05).mean())
    )
    summary["within_10_percent_fixed"] = summary["best_observed"].map(
        lambda _: float(runs.best_cost.map(lambda value: relative_gap(value, FIXED_REFERENCE) <= 0.10).mean())
    )
    runs["gap_to_fixed_reference"] = runs.best_cost.map(
        lambda value: relative_gap(value, FIXED_REFERENCE)
    )
    runs.to_csv(OUTPUT_DIR / "runs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "schedule_summary.csv", index=False)
    diagnostic.to_csv(OUTPUT_DIR / "trace_diagnostics.csv", index=False)
    pair.to_csv(OUTPUT_DIR / "paired_comparison.csv", index=False)

    within5 = int(runs.gap_to_fixed_reference.le(0.05).sum())
    within10 = int(runs.gap_to_fixed_reference.le(0.10).sum())
    median_pairwise = float(summary.iloc[0]["median_pairwise_difference"])
    late_rate = float(runs.late_improvement.mean())
    criteria = {
        "within_5_percent": within5 >= 10,
        "within_10_percent": within10 == 12,
        "median_pairwise_difference": median_pairwise < 0.05,
        "late_improvement_rate": late_rate < 0.20,
        "complete_runs": len(runs) == 12 and runs.attempted_moves.eq(75650).all(),
    }
    lines = [
        "# Workload-7 README Annealing Convergence Test",
        "",
        "## Test",
        "",
        "Twelve full-space workload-7 searches used the same frozen calibration, policy, search space, and deterministic start/seed panel. The schedule is the GitHub README default. Run 1 was reused from the earlier exact-schedule execution; runs 2-12 were executed in four tmux workers with private simulation caches.",
        "",
        f"- Schedule: `T0=4000`, `Tf=1e-3`, `moves/temperature=50`, `cooling=0.99`.",
        f"- Attempted moves per run: `{75650}`.",
        f"- Fixed reference: `{FIXED_REFERENCE}`.",
        f"- Overall observed best: `{runs.best_cost.min():.12g}`.",
        f"- Wall-clock sum of run times: `{elapsed / 60:.1f}` minutes.",
        "",
        "## Results",
        "",
        "| Runs | Best | Median | Worst | Within 5% of fixed reference | Within 10% of fixed reference | Median pairwise delta | Late improvement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {len(runs)} | {runs.best_cost.min():.6g} | {runs.best_cost.median():.6g} | {runs.best_cost.max():.6g} | {within5}/12 ({within5 / 12:.1%}) | {within10}/12 ({within10 / 12:.1%}) | {median_pairwise:.1%} | {late_rate:.1%} |",
        "",
        "## Runtime And Trace Checks",
        "",
        f"- Complete 75,650-move traces: {'yes' if criteria['complete_runs'] else 'no'}.",
        f"- Median invalid-mutation rate: {diagnostic.invalid_mutation_rate.median():.1%}.",
        f"- Median first-temperature acceptance: {diagnostic.first_temperature_acceptance.median():.1%}.",
        f"- Median first-temperature uphill acceptance: {diagnostic.first_temperature_uphill_acceptance.median():.1%}.",
        f"- Median simulator calls: {diagnostic.simulator_calls.median():.0f}.",
        "",
        "## Paired Comparison",
        "",
        "The paired comparison uses the same 12 starts and seeds from the workload-7 trend test. It compares this README schedule with the available 2,400-, 4,800-, 5,632-, 5,824-, and 11,264-move result files where present.",
        "",
        "| Reference | README better | README worse | Median objective change |",
        "|---|---:|---:|---:|",
    ]
    for column in pair.columns:
        if not column.endswith("_delta"):
            continue
        label = column.removesuffix("_delta")
        delta = pair[column]
        lines.append(
            f"| `{label}` | {(delta < 0).sum()}/12 | {(delta > 0).sum()}/12 | {delta.median():.6g} |"
        )
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            f"Empirical convergence criteria: **{'met' if all(criteria.values()) else 'not met'}**.",
            "",
            "A positive result means the README schedule produces a stable and improving distribution relative to the fixed reference. It does not prove a global optimum.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize():
    runs = load_rows()
    if len(runs) != 12:
        raise RuntimeError(f"Expected 12 completed runs, found {len(runs)}")
    summary = summarize_runs(runs)
    diagnostic = diagnostics(runs)
    comparisons = {
        "control_2400": OLD_TREND_DIR / "supplemental_control_2400_runs.csv",
        "control_4800": OLD_TREND_DIR / "runs.csv",
        "slow_5632": OLD_TREND_DIR / "runs.csv",
    }
    pair = pd.DataFrame({"run": runs.run})
    for label, path in comparisons.items():
        if not path.exists():
            continue
        prior = pd.read_csv(path)
        prior = prior.loc[prior["run"].between(1, 12)]
        if "schedule" in prior:
            prior = prior.loc[prior.schedule.isin(["thorough_4800", "slow_320_5632", "slow_640_5824", "intensive_320_11264"])]
        if len(prior) != 12:
            continue
        pair[label] = prior.sort_values("run").best_cost.to_numpy()
        pair[f"{label}_delta"] = runs.sort_values("run").best_cost.to_numpy() - pair[label]
    write_report(runs, summary, pair, diagnostic, float(runs.wall_seconds_with_evaluation.sum()))


def wait_finalize():
    while True:
        if len(list((OUTPUT_DIR / "runs/workload_7_readme/requested_4000_75650").glob("run_*/result.json"))) >= 12:
            finalize()
            return
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "worker", "finalize", "wait-finalize"])
    parser.add_argument("--first", type=int)
    parser.add_argument("--last", type=int)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        copy_run_one()
        source_cache = OLD_TREND_DIR / "simulation_cache.csv"
        initialize_experiment_cache(source_cache, OUTPUT_DIR / "base_cache.csv")
        calibration_source = OLD_TREND_DIR / "calibration/workload_7/calibration.json"
        destination = OUTPUT_DIR / "calibration/workload_7/calibration.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(calibration_source, destination)
        write_manifest()
    elif args.mode == "worker":
        worker(args.first, args.last, args.cache)
    elif args.mode == "finalize":
        finalize()
    else:
        wait_finalize()


def write_manifest():
    write = {
        "schedule": SCHEDULES[SCHEDULE],
        "planned_moves": 75650,
        "runs": 12,
        "workload_id": 7,
        "intermediate_policy": INTERMEDIATE_POLICY,
        "fixed_reference": FIXED_REFERENCE,
        "search_space_sha256": file_sha256(Path("cfg/parameters/input.json")),
        "d2d_protocol_fix_skipped": True,
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(write, file, indent=2)


if __name__ == "__main__":
    main()
