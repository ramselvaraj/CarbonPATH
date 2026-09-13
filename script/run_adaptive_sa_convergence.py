"""Unattended adaptive-SA pilot and formal convergence campaign."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import pandas as pd

from main import (
    WORKLOAD_CONFIGS,
    calibration_identity,
    parse_workload_entry,
    validate_calibration,
)
from script.run_full_space_convergence import (
    calibrate_workload,
    evaluate_best_architecture,
    initialize_experiment_cache,
)
from script.run_optimizer_experiments import (
    BASE_CACHE,
    FULL_SEARCH_SPACE,
    generate_initial_architecture,
    load_json,
    run_search,
    trace_fingerprint,
    wilson_interval,
    write_json,
)
from system.utils.AnnealingSchedule import (
    AdaptiveScheduleConfig,
    AdaptiveTemperatureController,
)
from system.utils.ArchitectureIdentity import (
    architecture_fingerprint,
    canonical_architecture_fingerprint,
)


MOVE_BUDGET = 75650
SCHEDULE = {
    "initial_temp": 12.27,
    "freezing_temp": 1e-3,
    "max_move_per_temp_step": 100,
    "cooling_rate": 0.98,
}
PILOT_SEEDS = [(7000 + i, 8000 + i) for i in range(5)]
FORMAL_SEEDS = [(12000 + i, 13000 + i) for i in range(10)]
REFERENCE_DELTA_EPSILON = 1e-12
DEFAULT_STALLED_SECONDS = 1800


def atomic_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_files(reference: Path) -> list[Path]:
    return sorted(reference.glob("runs/**/search_trace.csv"))


def read_positive_deltas(reference: Path) -> list[float]:
    values = []
    for trace in reference_files(reference):
        with trace.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                try:
                    delta = float(row["cost_diff"])
                except (KeyError, TypeError, ValueError):
                    continue
                if delta > 1e-12 and math.isfinite(delta):
                    values.append(delta)
    return values


def initial_temperature(reference: Path | None) -> float:
    deltas = read_positive_deltas(reference) if reference else []
    if not deltas:
        return SCHEDULE["initial_temp"]
    deltas.sort()
    middle = len(deltas) // 2
    median = deltas[middle] if len(deltas) % 2 else (deltas[middle - 1] + deltas[middle]) / 2
    return -median / math.log(0.80)


def reference_summary(reference: Path) -> dict:
    files = reference_files(reference)
    deltas = read_positive_deltas(reference)
    if not files or not deltas:
        raise ValueError("reference must contain usable positive cost deltas")
    deltas.sort()
    middle = len(deltas) // 2
    median = deltas[middle] if len(deltas) % 2 else (deltas[middle - 1] + deltas[middle]) / 2
    return {
        "path": str(reference.resolve()),
        "files": [
            {"path": str(path.resolve()), "sha256": sha256(path), "size": path.stat().st_size}
            for path in files
        ],
        "positive_delta_count": len(deltas),
        "median_positive_delta": median,
        "initial_temperature": -median / math.log(0.80),
    }


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def valid_result(
    path: Path,
    *,
    stage=None,
    run=None,
    workload=None,
    initial_seed=None,
    search_seed=None,
    move_budget=MOVE_BUDGET,
    expected_config=None,
) -> bool:
    if not path.exists():
        return False
    try:
        result = load_json(path)
        trace = path.parent / "search_trace.csv"
        required = ("run", "stage", "workload_id", "initial_seed", "search_seed",
                    "best_cost", "canonical_best_fingerprint", "trace_fingerprint",
                    "attempted_moves", "adaptive_config")
        if any(key not in result for key in required):
            return False
        if stage is not None and result["stage"] != stage:
            return False
        if run is not None and result["run"] != run:
            return False
        if workload is not None and result["workload_id"] != workload:
            return False
        if initial_seed is not None and result["initial_seed"] != initial_seed:
            return False
        if search_seed is not None and result["search_seed"] != search_seed:
            return False
        if not math.isfinite(float(result["best_cost"])):
            return False
        if result["attempted_moves"] != move_budget:
            return False
        if result["adaptive_config"].get("max_total_moves") != move_budget:
            return False
        if expected_config is not None and result["adaptive_config"] != expected_config:
            return False
        if not math.isfinite(float(result.get("verified_best_cost", result["best_cost"]))):
            return False
        architecture_path = path.parent / "best_architecture.json"
        initial_path = path.parent / "initial_architecture.json"
        if not initial_path.exists():
            return False
        if result["initial_fingerprint"] != architecture_fingerprint(load_json(initial_path)):
            return False
        architecture = load_json(architecture_path)
        if result["best_fingerprint"] != architecture_fingerprint(architecture):
            return False
        if result["canonical_best_fingerprint"] != canonical_architecture_fingerprint(architecture):
            return False
        if not (path.parent / "best_architecture.json").exists():
            return False
        with trace.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != move_budget:
            return False
        if [int(row["SA_run_loop"]) for row in rows] != list(range(1, move_budget + 1)):
            return False
        trace_frame = pd.read_csv(trace)
        if result["trace_fingerprint"] != trace_fingerprint(trace_frame):
            return False
        temperature_path = path.parent / "temperature_trace.csv"
        with temperature_path.open(newline="", encoding="utf-8") as stream:
            temperatures = list(csv.DictReader(stream))
        expected_levels = math.ceil(move_budget / 100)
        if len(temperatures) != expected_levels:
            return False
        if [int(row["level"]) for row in temperatures] != list(range(expected_levels)):
            return False
        if int(temperatures[-1]["moves_after"]) != move_budget:
            return False
        for row in temperatures:
            if not all(math.isfinite(float(row[key])) for key in ("temperature_before", "temperature_after")):
                return False
            if int(row["reheat_count"]) > 2:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_worker(args: argparse.Namespace) -> None:
    output = Path(args.output_root)
    workload = parse_workload_entry(args.workload, WORKLOAD_CONFIGS[args.workload])
    run_dir = output / args.stage / "runs" / f"run_{args.run:02d}"
    result_path = run_dir / "result.json"
    adaptive_config = AdaptiveScheduleConfig(max_total_moves=args.move_budget)
    if valid_result(result_path, stage=args.stage, run=args.run, workload=args.workload,
                    initial_seed=args.initial_seed, search_seed=args.search_seed,
                    move_budget=args.move_budget, expected_config=adaptive_config.as_dict()):
        return
    calibration_data = load_json(args.calibration)
    if args.manifest:
        manifest = load_json(args.manifest)
        if manifest.get("calibration_sha256") != sha256(Path(args.calibration)):
            raise RuntimeError("worker calibration does not match the campaign manifest")
        if not math.isclose(args.initial_temperature, manifest["initial_temperature"], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("worker initial temperature does not match the campaign manifest")
    expected_identity = calibration_identity(FULL_SEARCH_SPACE, workload, "direct_forward")
    if calibration_data.get("_calibration_identity") != expected_identity:
        raise RuntimeError("campaign calibration identity does not match the frozen search")
    validate_calibration(calibration_data)
    run_dir.mkdir(parents=True, exist_ok=True)
    initial_path = output / "initial_architectures" / args.stage / f"run_{args.run:02d}.json"
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    if initial_path.exists():
        initial = load_json(initial_path)
    else:
        initial = generate_initial_architecture(
            FULL_SEARCH_SPACE, args.initial_seed, initial_path.with_suffix(".log")
        )
        write_json(initial_path, initial)

    worker_cache = output / "workers" / f"cache_{args.stage}_{args.run:02d}.csv"
    worker_cache.parent.mkdir(parents=True, exist_ok=True)
    if not worker_cache.exists():
        initialize_experiment_cache(Path(args.base_cache), worker_cache)

    move_budget = args.move_budget
    controller = AdaptiveTemperatureController(
        args.initial_temperature, adaptive_config
    )
    progress_path = run_dir / "progress.json"
    atomic_write(progress_path, {
        "run": args.run, "stage": args.stage, "state": "running",
        "attempted_moves": 0, "max_total_moves": move_budget,
        "progress": 0.0, "pid": os.getpid(), "started_at": time.time(),
        "updated_at": time.time(),
    })

    def level_callback(decision, rows):
        atomic_write(
            progress_path,
            {
                "run": args.run,
                "stage": args.stage,
                "state": "running",
                "attempted_moves": decision.moves_after,
                "max_total_moves": move_budget,
                "progress": decision.moves_after / move_budget,
                "temperature": decision.temperature_after,
                "target_uphill_acceptance": decision.target_acceptance,
                "observed_uphill_acceptance": decision.observed_acceptance,
                "best_cost": decision.best_cost_after,
                "stagnation_moves": decision.stagnation_moves,
                "reheat_count": decision.reheat_count,
                "updated_at": time.time(),
            },
        )

    result = run_search(
        label=str(run_dir),
        output_dir=Path("."),
        workload=workload,
        workload_id=args.workload,
        initial_architecture=initial,
        search_seed=args.search_seed,
        search_space=FULL_SEARCH_SPACE,
        calibration_path=Path(args.calibration),
        base_cache=worker_cache,
        annealing={**SCHEDULE, "initial_temp": controller.initial_temperature},
        intermediate_policy="direct_forward",
        temperature_controller=controller,
        level_callback=level_callback,
        max_total_moves=move_budget,
    )
    best_architecture = result.pop("best_architecture")
    trace = result.pop("trace")
    result.update(
        {
            "run": args.run,
            "stage": args.stage,
            "workload_id": args.workload,
            "initial_seed": args.initial_seed,
            "search_seed": args.search_seed,
            "best_fingerprint": architecture_fingerprint(best_architecture),
            "canonical_best_fingerprint": canonical_architecture_fingerprint(best_architecture),
            "adaptive_config": adaptive_config.as_dict(),
            "temperature_levels": len(controller.history),
            "reheat_count": controller.reheat_count,
        }
    )
    evaluated_objective, raw_metrics = evaluate_best_architecture(
        architecture=best_architecture,
        workload=workload,
        calibration_path=Path(args.calibration),
        base_cache=worker_cache,
        log_path=run_dir / "best_evaluation.log",
    )
    if not math.isclose(evaluated_objective, result["best_cost"], rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError(
            f"best objective mismatch: search={result['best_cost']} evaluation={evaluated_objective}"
        )
    result["verified_best_cost"] = float(evaluated_objective)
    result["verified_metrics"] = {key: float(raw_metrics[key]) for key in ("latency", "energy", "area", "dollar")}
    run_dir.mkdir(parents=True, exist_ok=True)
    trace.to_csv(run_dir / "search_trace.csv", index=False)
    result["trace_fingerprint"] = trace_fingerprint(pd.read_csv(run_dir / "search_trace.csv"))
    write_json(run_dir / "best_architecture.json", best_architecture)
    with (run_dir / "temperature_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        rows = [decision.as_dict() for decision in controller.history]
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_write(result_path, result)
    atomic_write(progress_path, {"run": args.run, "stage": args.stage, "state": "complete", **result})


def run_worker(args: argparse.Namespace) -> None:
    run_dir = Path(args.output_root) / args.stage / "runs" / f"run_{args.run:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("run is already active") from error
        _run_worker(args)
        fcntl.flock(lock, fcntl.LOCK_UN)


def run_lock_available(run_dir: Path) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lock, fcntl.LOCK_UN)
        return True


def load_results(output: Path, stage: str, count=None, move_budget=MOVE_BUDGET, workload=None) -> list[dict]:
    results = []
    paths = sorted((output / stage / "runs").glob("run_*/result.json"))
    for path in paths:
        run = int(path.parent.name.removeprefix("run_"))
        if valid_result(path, stage=stage, run=run, workload=workload, move_budget=move_budget):
            results.append(load_json(path))
    if count is not None:
        expected = set(range(1, count + 1))
        actual = {result["run"] for result in results}
        if actual != expected:
            return []
    return results


def relative_gap(value, reference):
    return (value - reference) / max(abs(reference), 1e-12)


def gate(results: list[dict], required: int, *, pilot_fingerprint=None, all_results=None) -> tuple[bool, str]:
    if not results or any("verified_best_cost" not in result for result in results):
        return False, "missing verified results"
    counts = {}
    for result in results:
        key = result["canonical_best_fingerprint"]
        counts[key] = counts.get(key, 0) + 1
    modal_fingerprint, modal = max(counts.items(), key=lambda item: (item[1], item[0]))
    best_score = min(result["verified_best_cost"] for result in (all_results or results))
    modal_scores = [result["verified_best_cost"] for result in results
                    if result["canonical_best_fingerprint"] == modal_fingerprint]
    quality_gap = max(relative_gap(score, best_score) for score in modal_scores)
    passed = modal >= required and quality_gap <= 0.01
    if pilot_fingerprint is not None:
        passed = passed and modal_fingerprint == pilot_fingerprint
    detail = (f"modal={modal}/{len(results)} fingerprints={counts} "
              f"modal_fingerprint={modal_fingerprint} best_verified_cost={best_score:.17g} "
              f"modal_quality_gap={quality_gap:.6g}")
    if pilot_fingerprint is not None:
        detail += f" pilot_modal={pilot_fingerprint}"
    return passed, detail


def write_campaign_report(output: Path, *, pilot_results: list[dict], formal_results: list[dict], state: str) -> None:
    results = pilot_results + formal_results
    rows = []
    for result in results:
        row = dict(result)
        row["verified_metrics"] = json.dumps(row.get("verified_metrics", {}), sort_keys=True)
        rows.append(row)
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    counts = {}
    temperature_rows = []
    for result in results:
        fingerprint = result["canonical_best_fingerprint"]
        counts[fingerprint] = counts.get(fingerprint, 0) + 1
        temperature_rows.append({
            "stage": result["stage"], "run": result["run"],
            "temperature_levels": result.get("temperature_levels", 0),
            "reheat_count": result.get("reheat_count", 0),
        })
    with (output / "fingerprint_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fingerprint", "runs"])
        writer.writeheader()
        writer.writerows({"fingerprint": key, "runs": value} for key, value in sorted(counts.items()))
    with (output / "temperature_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "run", "temperature_levels", "reheat_count"])
        writer.writeheader()
        writer.writerows(temperature_rows)
    summary = {
        "state": state,
        "run_count": len(results),
        "pilot_count": len(pilot_results),
        "formal_count": len(formal_results),
        "fingerprint_counts": counts,
        "fingerprint_wilson_95": {
            fingerprint: wilson_interval(count, len(results))
            for fingerprint, count in counts.items()
        },
        "best_verified_cost": min((result["verified_best_cost"] for result in results), default=None),
        "runtime_seconds": sum(result.get("runtime_seconds", 0) for result in results),
        "simulator_calls": sum(result.get("simulator_calls", 0) for result in results),
        "temperature_levels": sum(result.get("temperature_levels", 0) for result in results),
        "reheats": sum(result.get("reheat_count", 0) for result in results),
        "runs": rows,
    }
    atomic_write(output / "summary.json", summary)
    lines = ["# Adaptive SA Campaign", "", f"- State: **{state}**", f"- Runs: `{len(results)}`", "",
             "| Fingerprint | Runs |", "|---|---:|"]
    lines.extend(f"| `{fingerprint}` | {count} |" for fingerprint, count in sorted(counts.items()))
    lines.extend(["", "The objective baseline is the best independently verified score across all completed campaign runs."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch_worker(args, stage, run, initial_seed, search_seed, calibration, output):
    command = [
        sys.executable,
        "-m",
        "script.run_adaptive_sa_convergence",
        "worker",
        "--workload",
        str(args.workload),
        "--output-root",
        str(output),
        "--stage",
        stage,
        "--run",
        str(run),
        "--initial-seed",
        str(initial_seed),
        "--search-seed",
        str(search_seed),
        "--calibration",
        str(calibration),
        "--base-cache",
        str(output / "base_cache.csv"),
        "--move-budget",
        str(args.move_budget),
        "--initial-temperature",
        str(args.initial_temperature),
    ]
    command.extend(["--manifest", str(output / "manifest.json")])
    if args.reference:
        command.extend(["--reference", str(args.reference)])
    log = output / "logs" / f"{stage}_run_{run:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("a", encoding="utf-8")
    return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def controller(args):
    output = Path(args.output_root)
    if args.command == "start" and output.exists() and any(output.iterdir()):
        raise RuntimeError("start requires a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("another campaign controller is already active") from error
    try:
        atomic_write(output / "state.json", {"state": "PREPARING"})
        manifest_path = output / "manifest.json"
        previous = load_json(manifest_path) if manifest_path.exists() else None
        if args.command == "start" and previous is not None:
            raise RuntimeError("start requires an output directory without a manifest")
        if args.command == "resume" and previous is None:
            raise RuntimeError("resume requires an existing campaign manifest")
        workload = parse_workload_entry(args.workload, WORKLOAD_CONFIGS[args.workload])
        if not args.reference:
            raise ValueError("--reference is required for a reproducible campaign")
        reference_info = reference_summary(args.reference)
        if previous is not None:
            if previous.get("workload") != args.workload or previous.get("move_budget") != args.move_budget:
                raise RuntimeError("resume arguments do not match the campaign manifest")
            if previous.get("reference") != reference_info:
                raise RuntimeError("reference contents do not match the campaign manifest")
            if previous.get("max_workers") != args.max_workers or previous.get("poll_seconds") != args.poll_seconds:
                raise RuntimeError("resume scheduling arguments do not match the campaign manifest")
        base_cache = output / "base_cache.csv"
        if not base_cache.exists():
            initialize_experiment_cache(Path(args.base_cache), base_cache)
        calibration = calibrate_workload(
            output_dir=output / "calibration",
            workload_id=args.workload,
            workload=workload,
            samples=10,
            seed=10000 + args.workload,
            base_cache=base_cache,
        )
        if previous is not None and previous.get("calibration_sha256") != sha256(calibration):
            raise RuntimeError("calibration contents do not match the campaign manifest")
        derived_temperature = reference_info["initial_temperature"]
        args.initial_temperature = derived_temperature
        manifest = {
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
            "python": platform.python_version(),
            "workload": args.workload,
            "search_space_sha256": sha256(FULL_SEARCH_SPACE),
            "calibration": str(calibration),
            "calibration_sha256": sha256(calibration),
            "schedule": SCHEDULE,
            "move_budget": args.move_budget,
            "controller": AdaptiveScheduleConfig(max_total_moves=args.move_budget).as_dict(),
            "initial_temperature": derived_temperature,
            "reference": reference_info,
            "base_cache_sha256": sha256(base_cache),
            "max_workers": args.max_workers,
            "poll_seconds": args.poll_seconds,
            "stalled_seconds": args.stalled_seconds,
            "created_at": time.time(),
        }
        if previous is None:
            atomic_write(output / "manifest.json", manifest)
        else:
            manifest = previous
        stages = [("pilot", 3, 2), ("pilot", 5, 4), ("formal_10", 10, 8)]
        pilot_fingerprint = None
        for stage, count, required in stages:
            atomic_write(
                output / "state.json",
                {"state": (f"PILOT_{count}_RUNNING" if stage == "pilot" else "FORMAL_10_RUNNING"), "stage": stage},
            )
            seeds = PILOT_SEEDS if stage == "pilot" else FORMAL_SEEDS
            running = {}
            attempts = {}
            started_at = {}
            while len(load_results(output, stage, count=count, move_budget=args.move_budget,
                                   workload=args.workload)) < count:
                results = load_results(output, stage, move_budget=args.move_budget, workload=args.workload)
                for run, (initial_seed, search_seed) in enumerate(seeds[:count], 1):
                    if run in {result["run"] for result in results} or run in running:
                        continue
                    if not run_lock_available(output / stage / "runs" / f"run_{run:02d}"):
                        continue
                    if len(running) >= args.max_workers:
                        break
                    attempts[run] = attempts.get(run, 0) + 1
                    if attempts[run] > 2:
                        atomic_write(output / "state.json", {
                            "state": "OPERATIONAL_ERROR", "stage": stage,
                            "run": run, "attempts": attempts[run],
                        })
                        return
                    running[run] = launch_worker(
                        args, stage, run, initial_seed, search_seed, calibration, output
                    )
                    started_at[run] = time.time()
                for run, process in list(running.items()):
                    progress = output / stage / "runs" / f"run_{run:02d}" / "progress.json"
                    last_update = progress.stat().st_mtime if progress.exists() else started_at.get(run, time.time())
                    if time.time() - last_update > args.stalled_seconds:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        attempts[run] = attempts.get(run, 0) + 1
                        del running[run]
                        if attempts[run] >= 2:
                            atomic_write(output / "state.json", {
                                "state": "OPERATIONAL_ERROR", "stage": stage,
                                "run": run, "attempts": attempts[run],
                                "error": "worker stalled",
                            })
                            return
                        continue
                    if process.poll() is not None:
                        if process.returncode != 0:
                            (output / "logs" / f"{stage}_run_{run:02d}.log").open("a").write(
                                f"\nworker exited with status {process.returncode}\n"
                            )
                        del running[run]
                time.sleep(args.poll_seconds)
            results = load_results(output, stage, count=count, move_budget=args.move_budget, workload=args.workload)
            assessment_state = f"PILOT_{count}_ASSESSING" if stage == "pilot" else "FORMAL_10_ASSESSING"
            atomic_write(output / "state.json", {"state": assessment_state, "stage": stage})
            if stage == "pilot":
                pilot_fingerprint = max(
                    {result["canonical_best_fingerprint"] for result in results},
                    key=lambda fingerprint: sum(result["canonical_best_fingerprint"] == fingerprint for result in results),
                )
            all_results = []
            if stage == "pilot":
                all_results.extend(load_results(output, "pilot", move_budget=args.move_budget,
                                                workload=args.workload))
            else:
                all_results.extend(load_results(output, "pilot", move_budget=args.move_budget,
                                                workload=args.workload))
                all_results.extend(load_results(output, "formal_10", move_budget=args.move_budget,
                                                workload=args.workload))
            passed, detail = gate(results, required, pilot_fingerprint=None if count == 3 else pilot_fingerprint,
                                  all_results=all_results)
            gate_path = output / (f"pilot_{count}_gate.json" if stage == "pilot" else "formal_10_gate.json")
            atomic_write(gate_path, {
                "passed": passed, "stage": stage, "count": count,
                "required": required, "detail": detail,
            })
            if not passed:
                pilot_results = load_results(output, "pilot", move_budget=args.move_budget,
                                             workload=args.workload)
                formal_results = load_results(output, "formal_10", move_budget=args.move_budget,
                                              workload=args.workload)
                atomic_write(
                    output / "state.json",
                    {"state": "COMPLETE_FAIL", "stage": f"{stage}_{count}", "detail": detail},
                )
                write_campaign_report(output, pilot_results=pilot_results,
                                      formal_results=formal_results, state="COMPLETE_FAIL")
                return
        pilot_results = load_results(output, "pilot", move_budget=args.move_budget, workload=args.workload)
        formal_results = load_results(output, "formal_10", move_budget=args.move_budget, workload=args.workload)
        atomic_write(output / "state.json", {"state": "COMPLETE_PASS", "stage": "formal_10"})
        write_campaign_report(output, pilot_results=pilot_results, formal_results=formal_results,
                              state="COMPLETE_PASS")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def status(args):
    output = Path(args.output_root)
    state = load_json(output / "state.json") if (output / "state.json").exists() else {"state": "not_started"}
    manifest = load_json(output / "manifest.json") if (output / "manifest.json").exists() else {}
    move_budget = manifest.get("move_budget", args.move_budget if hasattr(args, "move_budget") else MOVE_BUDGET)
    print(json.dumps(state, indent=2))
    for stage in ("pilot", "formal_10"):
        results = load_results(output, stage, move_budget=move_budget, workload=args.workload)
        print(stage, len(results), gate(results, 0)[1] if results else "")


def validate_campaign(args):
    output = Path(args.output_root)
    manifest = load_json(output / "manifest.json")
    state = load_json(output / "state.json")
    budget = manifest["move_budget"]
    pilot = load_results(output, "pilot", move_budget=budget, workload=manifest["workload"])
    formal = load_results(output, "formal_10", move_budget=budget, workload=manifest["workload"])
    expected_pilot = 5 if state.get("state") in {"PILOT_5_ASSESSING", "FORMAL_10_RUNNING", "FORMAL_10_ASSESSING", "COMPLETE_PASS", "COMPLETE_FAIL"} else 3
    if len(pilot) < expected_pilot:
        raise RuntimeError(f"expected at least {expected_pilot} valid pilot runs, found {len(pilot)}")
    if state.get("state") in {"FORMAL_10_ASSESSING", "COMPLETE_PASS", "COMPLETE_FAIL"} and len(formal) != 10:
        raise RuntimeError(f"expected 10 valid formal runs, found {len(formal)}")
    print(json.dumps({"state": state, "pilot_runs": len(pilot), "formal_runs": len(formal)}, indent=2))


def watch(args):
    while True:
        print("\033[2J\033[H", end="")
        status(args)
        if (Path(args.output_root) / "state.json").exists():
            state = load_json(Path(args.output_root) / "state.json").get("state")
            if state in {"COMPLETE_PASS", "COMPLETE_FAIL", "OPERATIONAL_ERROR"}:
                return
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workload", type=int, default=7)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--reference", type=Path)
    common.add_argument("--base-cache", type=Path, default=BASE_CACHE)
    start = subparsers.add_parser("start", parents=[common])
    start.add_argument("--max-workers", type=positive_int, default=4)
    start.add_argument("--poll-seconds", type=positive_int, default=30)
    start.add_argument("--stalled-seconds", type=positive_int, default=DEFAULT_STALLED_SECONDS)
    start.add_argument("--move-budget", type=positive_int, default=MOVE_BUDGET)
    start.set_defaults(function=controller)
    worker = subparsers.add_parser("worker", parents=[common])
    worker.add_argument("--stage", required=True)
    worker.add_argument("--run", type=int, required=True)
    worker.add_argument("--initial-seed", type=int, required=True)
    worker.add_argument("--search-seed", type=int, required=True)
    worker.add_argument("--calibration", type=Path, required=True)
    worker.add_argument("--move-budget", type=positive_int, default=MOVE_BUDGET)
    worker.add_argument("--initial-temperature", type=float, required=True)
    worker.add_argument("--manifest", type=Path)
    worker.set_defaults(function=run_worker)
    show = subparsers.add_parser("status", parents=[common])
    show.set_defaults(function=status)
    validate = subparsers.add_parser("validate", parents=[common])
    validate.set_defaults(function=validate_campaign)
    watch_parser = subparsers.add_parser("watch", parents=[common])
    watch_parser.add_argument("--interval", type=positive_int, default=30)
    watch_parser.set_defaults(function=watch)
    resume = subparsers.add_parser("resume", parents=[common])
    resume.add_argument("--max-workers", type=positive_int, default=4)
    resume.add_argument("--poll-seconds", type=positive_int, default=30)
    resume.add_argument("--stalled-seconds", type=positive_int, default=DEFAULT_STALLED_SECONDS)
    resume.add_argument("--move-budget", type=positive_int, default=MOVE_BUDGET)
    resume.set_defaults(function=controller)
    args = parser.parse_args()
    try:
        args.function(args)
    except Exception as error:
        if args.command in {"start", "resume"}:
            atomic_write(Path(args.output_root) / "state.json", {
                "state": "OPERATIONAL_ERROR",
                "command": args.command,
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_at": time.time(),
            })
        raise


if __name__ == "__main__":
    main()
