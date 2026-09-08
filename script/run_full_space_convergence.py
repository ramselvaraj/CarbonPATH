from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import itertools
import json
import math
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time

import pandas as pd

from main import (
    WORKLOAD_CONFIGS,
    calculate_cost,
    calibration_identity,
    get_calib_cost_avg,
    parse_workload_entry,
)
from script.run_optimizer_experiments import (
    CURRENT_ANNEALING,
    FULL_SEARCH_SPACE,
    clone_cache,
    file_sha256,
    generate_initial_architecture,
    initialize_experiment_cache,
    load_json,
    run_search,
    write_json,
)
from system.utils.SimulationCache import SimulationCache


INTERMEDIATE_POLICY = "direct_forward"
SCHEDULES = {
    "current_45": CURRENT_ANNEALING,
    "moderate_600": {
        "initial_temp": 160,
        "freezing_temp": 1e-4,
        "max_move_per_temp_step": 20,
        "cooling_rate": 0.62,
    },
    "thorough_600": {
        "initial_temp": 320,
        "freezing_temp": 1e-5,
        "max_move_per_temp_step": 24,
        "cooling_rate": 0.5,
    },
    "thorough_1200": {
        "initial_temp": 320,
        "freezing_temp": 1e-5,
        "max_move_per_temp_step": 48,
        "cooling_rate": 0.5,
    },
    "thorough_2400": {
        "initial_temp": 320,
        "freezing_temp": 1e-5,
        "max_move_per_temp_step": 96,
        "cooling_rate": 0.5,
    },
    "thorough_4800": {
        "initial_temp": 320,
        "freezing_temp": 1e-5,
        "max_move_per_temp_step": 192,
        "cooling_rate": 0.5,
    },
    "slow_320_5632": {
        "initial_temp": 320,
        "freezing_temp": 1e-6,
        "max_move_per_temp_step": 64,
        "cooling_rate": 0.8,
    },
    "slow_640_5824": {
        "initial_temp": 640,
        "freezing_temp": 1e-6,
        "max_move_per_temp_step": 64,
        "cooling_rate": 0.8,
    },
    "intensive_320_11264": {
        "initial_temp": 320,
        "freezing_temp": 1e-6,
        "max_move_per_temp_step": 128,
        "cooling_rate": 0.8,
    },
    "intensive_640_11648": {
        "initial_temp": 640,
        "freezing_temp": 1e-6,
        "max_move_per_temp_step": 128,
        "cooling_rate": 0.8,
    },
    "requested_4000_75650": {
        "initial_temp": 4000,
        "freezing_temp": 1e-3,
        "max_move_per_temp_step": 50,
        "cooling_rate": 0.99,
    },
}
PANELS = (
    {
        "name": "workload_7_primary",
        "workload_id": 7,
        "runs": 8,
        "initial_seed": 5100,
        "search_seed": 5200,
        "schedules": ("current_45", "moderate_600", "thorough_600"),
    },
    {
        "name": "workload_7_heldout",
        "workload_id": 7,
        "runs": 4,
        "initial_seed": 6100,
        "search_seed": 6200,
        "schedules": ("thorough_600", "thorough_1200"),
    },
    {
        "name": "workload_8_crosscheck",
        "workload_id": 8,
        "runs": 4,
        "initial_seed": 7100,
        "search_seed": 7200,
        "schedules": ("current_45", "moderate_600", "thorough_600"),
    },
)
ROBUST_PANELS = (
    {
        "name": "workload_7_primary",
        "workload_id": 7,
        "runs": 30,
        "initial_seed": 8100,
        "search_seed": 8200,
        "schedules": ("thorough_600", "thorough_1200", "thorough_2400"),
    },
    {
        "name": "workload_7_heldout",
        "workload_id": 7,
        "runs": 12,
        "initial_seed": 9100,
        "search_seed": 9200,
        "schedules": ("thorough_4800",),
    },
    {
        "name": "workload_8_crosscheck",
        "workload_id": 8,
        "runs": 12,
        "initial_seed": 10100,
        "search_seed": 10200,
        "schedules": ("thorough_600", "thorough_2400"),
    },
)


def planned_move_count(schedule):
    levels = math.ceil(
        math.log(schedule["freezing_temp"] / schedule["initial_temp"])
        / math.log(schedule["cooling_rate"])
    )
    return levels * schedule["max_move_per_temp_step"]


def calibrate_workload(
    *, output_dir, workload_id, workload, samples, seed, base_cache
):
    calibration_dir = Path(output_dir) / f"workload_{workload_id}"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = calibration_dir / "calibration.json"
    expected_identity = calibration_identity(
        load_json(FULL_SEARCH_SPACE), workload, INTERMEDIATE_POLICY
    )
    if calibration_path.exists():
        calibration = load_json(calibration_path)
        if calibration.get("_calibration_identity") == expected_identity:
            return calibration_path

    with tempfile.TemporaryDirectory(
        prefix=f"carbonpath-convergence-calibration-{workload_id}-"
    ) as directory:
        cache = SimulationCache(
            clone_cache(base_cache, directory),
            simulator_dir=str(Path(directory) / "simulation"),
        )
        random.seed(seed)
        with (calibration_dir / "calibration.log").open(
            "w", encoding="utf-8"
        ) as log:
            with redirect_stdout(log), redirect_stderr(log):
                get_calib_cost_avg(
                    calibration_iterations=samples,
                    config_path=str(FULL_SEARCH_SPACE),
                    cache=cache,
                    calibration_file_path=str(calibration_path),
                    workload_sequence=workload,
                    intermediate_policy=INTERMEDIATE_POLICY,
                )
        cache.dump_cache()
        shutil.copy2(cache.dir, base_cache)
    return calibration_path


def evaluate_best_architecture(
    *, architecture, workload, calibration_path, base_cache, log_path
):
    with tempfile.TemporaryDirectory(
        prefix="carbonpath-convergence-evaluation-"
    ) as directory:
        cache = SimulationCache(
            clone_cache(base_cache, directory),
            simulator_dir=str(Path(directory) / "simulation"),
        )
        with Path(log_path).open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                objective, _, raw = calculate_cost(
                    profile_name="t1",
                    cost_avgerage=load_json(calibration_path),
                    system_dict=architecture,
                    cache=cache,
                    workload_sequence=workload,
                    intermediate_policy=INTERMEDIATE_POLICY,
                )
        cache.dump_cache()
        shutil.copy2(cache.dir, base_cache)
    return objective, raw


def run_one(
    *,
    output_dir,
    panel,
    run_index,
    schedule_name,
    workload,
    calibration_path,
    base_cache,
):
    run_dir = (
        Path(output_dir)
        / "runs"
        / panel["name"]
        / schedule_name
        / f"run_{run_index + 1:02d}"
    )
    result_path = run_dir / "result.json"
    if result_path.exists():
        return load_json(result_path)

    initial_seed = panel["initial_seed"] + run_index
    search_seed = panel["search_seed"] + run_index
    initial_path = (
        Path(output_dir)
        / "initial_architectures"
        / panel["name"]
        / f"run_{run_index + 1:02d}.json"
    )
    if initial_path.exists():
        initial = load_json(initial_path)
    else:
        initial = generate_initial_architecture(
            FULL_SEARCH_SPACE,
            initial_seed,
            initial_path.with_suffix(".log"),
        )
        write_json(initial_path, initial)

    started = time.perf_counter()
    result = run_search(
        label=f"{schedule_name}/run_{run_index + 1:02d}",
        output_dir=Path(output_dir) / "runs" / panel["name"],
        workload=workload,
        workload_id=panel["workload_id"],
        initial_architecture=initial,
        search_seed=search_seed,
        search_space=FULL_SEARCH_SPACE,
        calibration_path=calibration_path,
        base_cache=base_cache,
        annealing=SCHEDULES[schedule_name],
        intermediate_policy=INTERMEDIATE_POLICY,
    )
    evaluated_objective, raw = evaluate_best_architecture(
        architecture=result["best_architecture"],
        workload=workload,
        calibration_path=calibration_path,
        base_cache=base_cache,
        log_path=run_dir / "best_evaluation.log",
    )
    if not math.isclose(
        evaluated_objective, result["best_cost"], rel_tol=1e-10, abs_tol=1e-10
    ):
        raise RuntimeError(
            f"Best architecture objective mismatch: {result['best_cost']} vs "
            f"{evaluated_objective}"
        )

    trace = result["trace"]
    cutoff = max(1, math.ceil(len(trace) * 0.8))
    early_best = float(trace.iloc[cutoff - 1]["best_cost"])
    final_best = float(result["best_cost"])
    improving_rows = trace["best_cost"].diff().fillna(0).lt(-1e-12)
    last_improvement_move = (
        int(trace.loc[improving_rows, "SA_run_loop"].max())
        if improving_rows.any()
        else 0
    )
    row = {
        "panel": panel["name"],
        "workload_id": panel["workload_id"],
        "run": run_index + 1,
        "schedule": schedule_name,
        "initial_seed": initial_seed,
        "search_seed": search_seed,
        "initial_fingerprint": result["initial_fingerprint"],
        "best_fingerprint": result["best_fingerprint"],
        "best_cost": final_best,
        "attempted_moves": result["attempted_moves"],
        "accepted_moves": result["accepted_moves"],
        "acceptance_rate": result["accepted_moves"] / result["attempted_moves"],
        "late_improvement": final_best < early_best - 1e-12,
        "last_improvement_move": last_improvement_move,
        "last_improvement_fraction": last_improvement_move
        / result["attempted_moves"],
        "runtime_seconds": result["runtime_seconds"],
        "wall_seconds_with_evaluation": time.perf_counter() - started,
        "simulator_calls": result["simulator_calls"],
        "trace_fingerprint": result["trace_fingerprint"],
        "latency": float(raw["latency"]),
        "energy": float(raw["energy"]),
        "area": float(raw["area"]),
        "dollar": float(raw["dollar"]),
    }
    write_json(result_path, row)
    return row


def median_pairwise_relative_difference(values):
    differences = [
        abs(left - right) / min(abs(left), abs(right))
        for left, right in itertools.combinations(values, 2)
    ]
    return float(pd.Series(differences).median()) if differences else 0.0


def summarize_runs(runs):
    rows = []
    for (panel, schedule), group in runs.groupby(["panel", "schedule"], sort=False):
        panel_runs = runs.loc[runs["panel"] == panel]
        best_observed = float(panel_runs["best_cost"].min())
        gap_scale = max(abs(best_observed), 1e-12)
        gaps = (group["best_cost"] - best_observed) / gap_scale
        rows.append(
            {
                "panel": panel,
                "schedule": schedule,
                "runs": len(group),
                "moves": int(group["attempted_moves"].median()),
                "best_observed": best_observed,
                "minimum_cost": float(group["best_cost"].min()),
                "median_cost": float(group["best_cost"].median()),
                "maximum_cost": float(group["best_cost"].max()),
                "within_5_percent": float(gaps.le(0.05).mean()),
                "within_10_percent": float(gaps.le(0.10).mean()),
                "median_pairwise_difference": median_pairwise_relative_difference(
                    group["best_cost"].tolist()
                ),
                "late_improvement_rate": float(group["late_improvement"].mean()),
                "median_acceptance_rate": float(group["acceptance_rate"].median()),
                "unique_best_fingerprints": int(group["best_fingerprint"].nunique()),
                "median_latency": float(group["latency"].median()),
                "median_energy": float(group["energy"].median()),
                "median_area": float(group["area"].median()),
                "median_dollar": float(group["dollar"].median()),
                "runtime_seconds": float(group["runtime_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def paired_budget_summary(runs):
    for panel_name in ("workload_7_heldout", "workload_7_primary"):
        panel = runs.loc[runs["panel"] == panel_name]
        schedule_names = sorted(
            panel["schedule"].unique(), key=lambda name: planned_move_count(SCHEDULES[name])
        )
        if len(schedule_names) >= 2:
            break
    else:
        raise ValueError("A workload-7 panel must contain at least two move budgets")

    lower_schedule = schedule_names[0]
    higher_schedule = schedule_names[-1]
    paired = panel.pivot(index="run", columns="schedule", values="best_cost")
    paired["lower_cost"] = paired[lower_schedule]
    paired["higher_cost"] = paired[higher_schedule]
    paired["budget_gain"] = (
        paired["lower_cost"] - paired["higher_cost"]
    ) / paired["lower_cost"].abs().clip(lower=1e-12)
    paired["absolute_difference"] = (
        paired["lower_cost"] - paired["higher_cost"]
    ).abs() / paired[["lower_cost", "higher_cost"]].abs().min(axis=1).clip(lower=1e-12)
    return paired.reset_index(), panel_name, lower_schedule, higher_schedule


def percent(value):
    return f"{100 * value:.1f}%"


def write_report(
    output_dir,
    runs,
    summary,
    paired,
    paired_panel,
    lower_schedule,
    higher_schedule,
    elapsed_seconds,
):
    primary = summary.loc[summary["panel"] == "workload_7_primary"]
    candidate_schedule = max(
        primary["schedule"], key=lambda name: planned_move_count(SCHEDULES[name])
    )
    candidate = primary.loc[primary["schedule"] == candidate_schedule].iloc[0]
    stabilization = (
        candidate["within_5_percent"] >= 0.8
        and candidate["within_10_percent"] >= 0.95
        and candidate["median_pairwise_difference"] < 0.05
        and candidate["late_improvement_rate"] < 0.2
    )
    lines = [
        "# Full-Space Annealing Agreement Check",
        "",
        "## Scope",
        "",
        f"- Search space: `{FULL_SEARCH_SPACE}` (no reduction).",
        f"- Intermediate policy: `{INTERMEDIATE_POLICY}`.",
        "- The current D2D protocol implementation was intentionally left unchanged.",
        "- Results are empirical multi-start observations, not proof of an optimum.",
        f"- Total campaign wall time: {elapsed_seconds / 60:.1f} minutes.",
        "",
        "## Schedules",
        "",
        "| Schedule | T0 | Tf | Moves/level | Cooling | Planned moves |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    used_schedules = set(runs["schedule"])
    for name, schedule in SCHEDULES.items():
        if name not in used_schedules:
            continue
        lines.append(
            f"| `{name}` | {schedule['initial_temp']:g} | "
            f"{schedule['freezing_temp']:g} | "
            f"{schedule['max_move_per_temp_step']} | "
            f"{schedule['cooling_rate']:g} | {planned_move_count(schedule)} |"
        )
    lines.extend(
        [
            "",
            "## Multi-Start Results",
            "",
            "`best_observed` is the lowest objective seen within each panel across all listed schedules.",
            "",
            "| Panel | Schedule | Runs | Best observed | Median | Maximum | Within 5% | Within 10% | Median pairwise delta | Late improvement |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.panel}` | `{row.schedule}` | {row.runs} | "
            f"{row.best_observed:.6g} | {row.median_cost:.6g} | "
            f"{row.maximum_cost:.6g} | {percent(row.within_5_percent)} | "
            f"{percent(row.within_10_percent)} | "
            f"{percent(row.median_pairwise_difference)} | "
            f"{percent(row.late_improvement_rate)} |"
        )
    lines.extend(
        [
            "",
            "## Objective Components",
            "",
            "Medians below correspond to each schedule's final best architectures; lower is better.",
            "",
            "| Panel | Schedule | Latency | Energy | Area | Dollar |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.panel}` | `{row.schedule}` | {row.median_latency:.6g} | "
            f"{row.median_energy:.6g} | {row.median_area:.6g} | "
            f"{row.median_dollar:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Held-Out Budget Extension",
            "",
            f"Paired panel: `{paired_panel}`.",
            "",
            f"| Run | `{lower_schedule}` | `{higher_schedule}` | Budget gain | Absolute delta |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired.itertuples(index=False):
        lines.append(
            f"| {row.run} | {row.lower_cost:.6g} | "
            f"{row.higher_cost:.6g} | {percent(row.budget_gain)} | "
            f"{percent(row.absolute_difference)} |"
        )
    lines.extend(
        [
            "",
            f"Median signed gain: {percent(paired['budget_gain'].median())}.",
            f"Median absolute paired delta: {percent(paired['absolute_difference'].median())}.",
            "",
            "## Assessment",
            "",
            f"Primary workload-7 `{candidate_schedule}` empirical stabilization criteria: **{'met' if stabilization else 'not met'}**.",
            "",
            "The criteria require at least 80% of starts within 5% of `best_observed`, at least 95% within 10%, median pairwise objective difference below 5%, and fewer than 20% of runs improving in the final 20% of moves.",
            "",
            "See `runs.csv`, `schedule_summary.csv`, `heldout_budget.csv`, and each run's trace and architecture files for the underlying observations.",
        ]
    )
    (Path(output_dir) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run a resumable full-space annealing agreement campaign."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/full_space_convergence"),
    )
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=Path("cfg/static_cache/static_cache.csv"),
    )
    parser.add_argument("--calibration-samples", type=int, default=10)
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Run the 126-run, 600-to-4,800-move convergence campaign",
    )
    args = parser.parse_args()
    panels = ROBUST_PANELS if args.robust else PANELS

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_cache = args.output_dir / "simulation_cache.csv"
    if not experiment_cache.exists():
        initialize_experiment_cache(args.base_cache, experiment_cache)

    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        write_json(
            manifest_path,
            {
                "python": platform.python_version(),
                "full_search_space": str(FULL_SEARCH_SPACE),
                "full_search_space_sha256": file_sha256(FULL_SEARCH_SPACE),
                "intermediate_policy": INTERMEDIATE_POLICY,
                "d2d_protocol_fix_skipped": True,
                "schedules": SCHEDULES,
                "robust": args.robust,
                "panels": list(panels),
            },
        )

    workloads = {
        workload_id: parse_workload_entry(workload_id, WORKLOAD_CONFIGS[workload_id])
        for workload_id in {panel["workload_id"] for panel in panels}
    }
    calibration_paths = {
        workload_id: calibrate_workload(
            output_dir=args.output_dir / "calibration",
            workload_id=workload_id,
            workload=workload,
            samples=args.calibration_samples,
            seed=4900 + workload_id,
            base_cache=experiment_cache,
        )
        for workload_id, workload in workloads.items()
    }

    rows = []
    total_runs = sum(panel["runs"] * len(panel["schedules"]) for panel in panels)
    completed = 0
    for panel in panels:
        workload = workloads[panel["workload_id"]]
        for run_index in range(panel["runs"]):
            for schedule_name in panel["schedules"]:
                completed += 1
                print(
                    f"[{completed}/{total_runs}] {panel['name']} run "
                    f"{run_index + 1} {schedule_name}",
                    flush=True,
                )
                rows.append(
                    run_one(
                        output_dir=args.output_dir,
                        panel=panel,
                        run_index=run_index,
                        schedule_name=schedule_name,
                        workload=workload,
                        calibration_path=calibration_paths[panel["workload_id"]],
                        base_cache=experiment_cache,
                    )
                )
                pd.DataFrame(rows).to_csv(args.output_dir / "runs.csv", index=False)

    runs = pd.DataFrame(rows)
    summary = summarize_runs(runs)
    paired, paired_panel, lower_schedule, higher_schedule = paired_budget_summary(runs)
    summary.to_csv(args.output_dir / "schedule_summary.csv", index=False)
    paired.to_csv(args.output_dir / "heldout_budget.csv", index=False)
    write_report(
        args.output_dir,
        runs,
        summary,
        paired,
        paired_panel,
        lower_schedule,
        higher_schedule,
        float(runs["wall_seconds_with_evaluation"].sum()),
    )
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
