"""Run reproducible, fixed-schedule simulated-annealing baseline campaigns."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import pandas as pd

from main import (
    CALIBRATION_MODEL_VERSION,
    WORKLOAD_CONFIGS,
    parse_workload_entry,
    validate_calibration,
)
from script.run_full_space_convergence import (
    INTERMEDIATE_POLICY,
    calibrate_workload,
    evaluate_best_architecture,
)
from script.run_optimizer_experiments import (
    BASE_CACHE,
    FULL_SEARCH_SPACE,
    SIMULATION_MODEL_VERSION,
    generate_initial_architecture,
    initialize_experiment_cache,
    load_json,
    run_search,
    trace_fingerprint,
    write_json,
)
from system.utils.ArchitectureIdentity import (
    architecture_fingerprint,
    canonical_architecture_fingerprint,
)


SCHEDULE_NAME = "requested_4000_75650"
SCHEDULE = {
    "initial_temp": 4000,
    "freezing_temp": 1e-3,
    "max_move_per_temp_step": 50,
    "cooling_rate": 0.99,
}
INTERMEDIATE_POLICY = "direct_forward"
DEFAULT_WORKLOADS = (7, 9, 10)
DEFAULT_RUNS = 10
INITIAL_SEED_BASE = 12000
SEARCH_SEED_BASE = 13000
CALIBRATION_SEED_BASE = 10000


def atomic_write(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha256(value) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def planned_move_count(schedule=SCHEDULE) -> int:
    levels = math.ceil(
        math.log(schedule["freezing_temp"] / schedule["initial_temp"])
        / math.log(schedule["cooling_rate"])
    )
    return levels * schedule["max_move_per_temp_step"]


def task_list(workloads, runs):
    return [
        (workload_id, run)
        for run in range(1, runs + 1)
        for workload_id in workloads
    ]


def run_dir(output: Path, workload_id: int, run: int) -> Path:
    return output / "runs" / f"workload_{workload_id}" / f"run_{run:02d}"


def result_is_valid(path: Path, manifest: dict | None = None) -> bool:
    try:
        result = load_json(path)
        run_root = path.parent
        trace_path = run_root / "search_trace.csv"
        architecture_path = run_root / "best_architecture.json"
        initial_path = run_root / "initial_architecture.json"
        architecture_trace_path = run_root / "architecture_trace.csv"
        required = {
            "workload_id", "run", "initial_seed", "search_seed", "schedule",
            "best_cost", "verified_best_cost", "best_fingerprint",
            "canonical_best_fingerprint", "trace_fingerprint", "attempted_moves",
        }
        if not required.issubset(result) or not all(
            path.exists()
            for path in (trace_path, architecture_path, initial_path, architecture_trace_path)
        ):
            return False
        expected_workload = int(run_root.parent.name.removeprefix("workload_"))
        expected_run = int(run_root.name.removeprefix("run_"))
        if result["workload_id"] != expected_workload or result["run"] != expected_run:
            return False
        if manifest is not None:
            if result["schedule"] != manifest["schedule_name"]:
                return False
            if result["attempted_moves"] != manifest["planned_moves"]:
                return False
            expected_initial = manifest["seed_panel"]["initial_seed_base"] + result["run"] - 1
            expected_search = manifest["seed_panel"]["search_seed_base"] + result["run"] - 1
            if result["initial_seed"] != expected_initial or result["search_seed"] != expected_search:
                return False
            if result["search_space_sha256"] != manifest["search_space_sha256"]:
                return False
            if result["calibration_sha256"] != manifest["calibrations"][str(result["workload_id"])]["sha256"]:
                return False
            if result["workload_config_sha256"] != manifest["workload_config_sha256"][str(result["workload_id"])]:
                return False
        if not math.isfinite(float(result["best_cost"])):
            return False
        if not math.isclose(
            float(result["best_cost"]), float(result["verified_best_cost"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            return False
        architecture = load_json(architecture_path)
        initial = load_json(initial_path)
        if result["best_fingerprint"] != architecture_fingerprint(architecture):
            return False
        if result["canonical_best_fingerprint"] != canonical_architecture_fingerprint(architecture):
            return False
        trace = pd.read_csv(trace_path)
        if len(trace) != result["attempted_moves"]:
            return False
        if trace["SA_run_loop"].astype(int).tolist() != list(range(1, len(trace) + 1)):
            return False
        if result["trace_fingerprint"] != trace_fingerprint(trace):
            return False
        temperatures = trace["temperature"].drop_duplicates().tolist()
        expected_levels = manifest["planned_moves"] // manifest["schedule"]["max_move_per_temp_step"]
        if len(temperatures) != expected_levels:
            return False
        if result["initial_fingerprint"] != architecture_fingerprint(initial):
            return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def manifest_for(output: Path, workloads, runs, calibration_samples, base_cache) -> dict:
    return {
        "campaign_type": "fixed_sa_baseline",
        "campaign_version": 1,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "python": platform.python_version(),
        "workloads": list(workloads),
        "workload_config_sha256": {
            str(workload_id): value_sha256(WORKLOAD_CONFIGS[workload_id])
            for workload_id in workloads
        },
        "runs_per_workload": runs,
        "schedule_name": SCHEDULE_NAME,
        "schedule": SCHEDULE,
        "planned_moves": planned_move_count(),
        "profile_name": "t1",
        "intermediate_policy": INTERMEDIATE_POLICY,
        "search_space": str(FULL_SEARCH_SPACE),
        "search_space_sha256": sha256(FULL_SEARCH_SPACE),
        "calibration_model_version": CALIBRATION_MODEL_VERSION,
        "simulation_model_version": SIMULATION_MODEL_VERSION,
        "calibration_samples": calibration_samples,
        "seed_panel": {
            "initial_seed_base": INITIAL_SEED_BASE,
            "search_seed_base": SEARCH_SEED_BASE,
            "calibration_seed_base": CALIBRATION_SEED_BASE,
        },
        "base_cache_sha256": sha256(base_cache),
        "calibrations": {},
        "created_at": time.time(),
        "output_root": str(output),
    }


def prepare(args) -> None:
    output = args.output_root
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("prepare requires a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    base_cache = output / "base_cache.csv"
    initialize_experiment_cache(args.base_cache, base_cache)
    manifest = manifest_for(output, args.workloads, args.runs, args.calibration_samples, base_cache)
    for workload_id in args.workloads:
        workload = parse_workload_entry(workload_id, WORKLOAD_CONFIGS[workload_id])
        calibration = calibrate_workload(
            output_dir=output / "calibration",
            workload_id=workload_id,
            workload=workload,
            samples=args.calibration_samples,
            seed=CALIBRATION_SEED_BASE + workload_id,
            base_cache=base_cache,
        )
        manifest["calibrations"][str(workload_id)] = {
            "path": str(Path(calibration).resolve().relative_to(output.resolve())),
            "sha256": sha256(calibration),
            "identity": load_json(calibration).get("_calibration_identity"),
        }
    manifest["base_cache_sha256"] = sha256(base_cache)
    atomic_write(output / "manifest.json", manifest)
    atomic_write(output / "state.json", {"state": "PREPARED", "updated_at": time.time()})
    print(f"Prepared {output}")


def _worker(args) -> None:
    output = args.output_root
    manifest = load_json(output / "manifest.json")
    workload_id = args.workload
    run = args.run
    root = run_dir(output, workload_id, run)
    result_path = root / "result.json"
    if result_is_valid(result_path, manifest):
        return
    expected_initial = manifest["seed_panel"]["initial_seed_base"] + run - 1
    expected_search = manifest["seed_panel"]["search_seed_base"] + run - 1
    if args.initial_seed != expected_initial or args.search_seed != expected_search:
        raise RuntimeError("worker seeds do not match the prepared manifest")
    if sha256(FULL_SEARCH_SPACE) != manifest["search_space_sha256"]:
        raise RuntimeError("search space differs from the prepared manifest")
    if sha256(output / "base_cache.csv") != manifest["base_cache_sha256"]:
        raise RuntimeError("base cache differs from the prepared manifest")
    workload = parse_workload_entry(workload_id, WORKLOAD_CONFIGS[workload_id])
    if value_sha256(WORKLOAD_CONFIGS[workload_id]) != manifest["workload_config_sha256"][str(workload_id)]:
        raise RuntimeError("workload configuration differs from the prepared manifest")
    calibration_path = output / manifest["calibrations"][str(workload_id)]["path"]
    if sha256(calibration_path) != manifest["calibrations"][str(workload_id)]["sha256"]:
        raise RuntimeError("calibration differs from the prepared manifest")
    calibration = load_json(calibration_path)
    validate_calibration(calibration)
    root.mkdir(parents=True, exist_ok=True)
    initial_path = root / "initial_architecture.json"
    initial = (
        load_json(initial_path)
        if initial_path.exists()
        else generate_initial_architecture(FULL_SEARCH_SPACE, args.initial_seed, root / "initial_architecture.log")
    )
    write_json(initial_path, initial)
    worker_cache = output / "workers" / f"cache_w{workload_id}_r{run:02d}.csv"
    worker_cache.parent.mkdir(parents=True, exist_ok=True)
    if not worker_cache.exists():
        initialize_experiment_cache(output / "base_cache.csv", worker_cache)
    started = time.perf_counter()
    result = run_search(
        label=f"run_{run:02d}",
        output_dir=root.parent,
        workload=workload,
        workload_id=workload_id,
        initial_architecture=initial,
        search_seed=args.search_seed,
        search_space=FULL_SEARCH_SPACE,
        calibration_path=calibration_path,
        base_cache=worker_cache,
        annealing=SCHEDULE,
        intermediate_policy=INTERMEDIATE_POLICY,
    )
    best_architecture = result["best_architecture"]
    evaluated_objective, raw = evaluate_best_architecture(
        architecture=best_architecture,
        workload=workload,
        calibration_path=calibration_path,
        base_cache=worker_cache,
        log_path=root / "best_evaluation.log",
    )
    if not math.isclose(evaluated_objective, result["best_cost"], rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("independent best-architecture evaluation disagrees with search")
    row = {
        "campaign_type": manifest["campaign_type"],
        "workload_id": workload_id,
        "run": run,
        "schedule": SCHEDULE_NAME,
        "initial_seed": args.initial_seed,
        "search_seed": args.search_seed,
        "initial_fingerprint": result["initial_fingerprint"],
        "best_fingerprint": result["best_fingerprint"],
        "canonical_best_fingerprint": canonical_architecture_fingerprint(best_architecture),
        "best_cost": float(result["best_cost"]),
        "verified_best_cost": float(evaluated_objective),
        "attempted_moves": int(result["attempted_moves"]),
        "accepted_moves": int(result["accepted_moves"]),
        "acceptance_rate": result["accepted_moves"] / result["attempted_moves"],
        "trace_fingerprint": result["trace_fingerprint"],
        "runtime_seconds": result["runtime_seconds"],
        "wall_seconds_with_evaluation": time.perf_counter() - started,
        "simulator_calls": result["simulator_calls"],
        "search_space_sha256": manifest["search_space_sha256"],
        "calibration_sha256": manifest["calibrations"][str(workload_id)]["sha256"],
        "workload_config_sha256": manifest["workload_config_sha256"][str(workload_id)],
    }
    atomic_write(result_path, row)


def worker(args) -> None:
    root = run_dir(args.output_root, args.workload, args.run)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("run is already active") from error
        _worker(args)
        fcntl.flock(lock, fcntl.LOCK_UN)


def launch_worker(output: Path, workload_id: int, run: int, initial_seed: int, search_seed: int):
    log = output / "logs" / f"workload_{workload_id}_run_{run:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("a", encoding="utf-8")
    command = [
        sys.executable, "-m", "script.run_baseline_campaign", "worker",
        "--output-root", str(output), "--workload", str(workload_id), "--run", str(run),
        "--initial-seed", str(initial_seed), "--search-seed", str(search_seed),
    ]
    return subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def controller(args) -> None:
    output = args.output_root
    manifest = load_json(output / "manifest.json")
    if tuple(manifest["workloads"]) != tuple(args.workloads) or manifest["runs_per_workload"] != args.runs:
        raise RuntimeError("resume arguments do not match the campaign manifest")
    tasks = task_list(args.workloads, args.runs)
    pending = [task for task in tasks if not result_is_valid(run_dir(output, *task) / "result.json", manifest)]
    if args.stop_after is not None:
        pending = pending[:args.stop_after]
    limited = args.stop_after is not None
    running = {}
    attempts = {}
    lock = (output / "campaign.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_write(output / "state.json", {"state": "RUNNING", "pending": len(pending), "updated_at": time.time()})
        while pending or running:
            while pending and len(running) < args.max_workers:
                workload_id, run = pending.pop(0)
                process = launch_worker(
                    output,
                    workload_id,
                    run,
                    INITIAL_SEED_BASE + run - 1,
                    SEARCH_SEED_BASE + run - 1,
                )
                running[process] = (workload_id, run)
            for process, (workload_id, run) in list(running.items()):
                if process.poll() is None:
                    continue
                del running[process]
                if not result_is_valid(run_dir(output, workload_id, run) / "result.json", manifest):
                    key = (workload_id, run)
                    attempts[key] = attempts.get(key, 0) + 1
                    if attempts[key] > 1:
                        raise RuntimeError(
                            f"worker failed twice for workload {workload_id}, run {run}"
                        )
                    pending.append(key)
            atomic_write(output / "state.json", {
                "state": "RUNNING",
                "pending": len(pending),
                "running": len(running),
                "completed": sum(
                    result_is_valid(run_dir(output, *task) / "result.json", manifest) for task in tasks
                ),
                "updated_at": time.time(),
            })
            if pending or running:
                time.sleep(args.poll_seconds)
        state = "PARTIAL" if limited else "COMPLETE"
        atomic_write(output / "state.json", {"state": state, "updated_at": time.time()})
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    report(args)


def status(args) -> None:
    output = args.output_root
    manifest = load_json(output / "manifest.json")
    tasks = task_list(manifest["workloads"], manifest["runs_per_workload"])
    completed = [task for task in tasks if result_is_valid(run_dir(output, *task) / "result.json", manifest)]
    print(json.dumps({
        "state": load_json(output / "state.json").get("state", "UNKNOWN"),
        "completed": len(completed),
        "total": len(tasks),
        "pending": len(tasks) - len(completed),
    }, indent=2))


def validate(args) -> None:
    output = args.output_root
    manifest = load_json(output / "manifest.json")
    rows = []
    invalid = []
    for task in task_list(manifest["workloads"], manifest["runs_per_workload"]):
        path = run_dir(output, *task) / "result.json"
        if result_is_valid(path, manifest):
            rows.append(load_json(path))
        else:
            invalid.append(task)
    print(json.dumps({"valid": len(rows), "invalid": invalid}, indent=2))
    if invalid and not args.allow_partial:
        raise RuntimeError(f"invalid or incomplete runs: {invalid}")


def report(args) -> None:
    output = args.output_root
    manifest = load_json(output / "manifest.json")
    rows = [
        load_json(run_dir(output, *task) / "result.json")
        for task in task_list(manifest["workloads"], manifest["runs_per_workload"])
        if result_is_valid(run_dir(output, *task) / "result.json", manifest)
    ]
    if not rows:
        return
    runs = pd.DataFrame(rows)
    runs.to_csv(output / "runs.csv", index=False)
    fingerprints = (
        runs.groupby(["workload_id", "canonical_best_fingerprint"])
        .size().reset_index(name="runs")
    )
    fingerprints.to_csv(output / "fingerprint_summary.csv", index=False)
    summary = runs.groupby("workload_id").agg(
        runs=("run", "count"), best_observed=("verified_best_cost", "min"),
        median_cost=("verified_best_cost", "median"), worst_cost=("verified_best_cost", "max"),
        median_acceptance_rate=("acceptance_rate", "median"),
        median_runtime_seconds=("wall_seconds_with_evaluation", "median"),
        total_runtime_seconds=("wall_seconds_with_evaluation", "sum"),
        unique_canonical_fingerprints=("canonical_best_fingerprint", "nunique"),
    ).reset_index()
    summary.to_csv(output / "workload_summary.csv", index=False)
    lines = [
        "# Fixed-SA Baseline Campaign", "",
        f"- Schedule: `{SCHEDULE_NAME}` ({manifest['planned_moves']} moves per run)",
        f"- Policy: `{manifest['intermediate_policy']}`; profile: `t1`",
        f"- Valid runs: `{len(runs)}` / `{len(task_list(manifest['workloads'], manifest['runs_per_workload']))}`",
        "", "| Workload | Runs | Best | Median | Worst | Unique canonical architectures |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.workload_id} | {row.runs} | {row.best_observed:.12g} | "
            f"{row.median_cost:.12g} | {row.worst_cost:.12g} | "
            f"{row.unique_canonical_fingerprints} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument(
        "--workloads",
        type=int,
        nargs="+",
        choices=sorted(WORKLOAD_CONFIGS),
        default=list(DEFAULT_WORKLOADS),
    )
    common.add_argument("--runs", type=positive_int, default=DEFAULT_RUNS)
    common.add_argument("--base-cache", type=Path, default=BASE_CACHE)
    prepare_parser = sub.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--calibration-samples", type=positive_int, default=10)
    prepare_parser.set_defaults(function=prepare)
    controller_parser = sub.add_parser("start", parents=[common])
    controller_parser.add_argument("--max-workers", type=positive_int, default=4)
    controller_parser.add_argument("--poll-seconds", type=positive_int, default=10)
    controller_parser.add_argument("--stop-after", type=positive_int)
    controller_parser.set_defaults(function=controller)
    resume_parser = sub.add_parser("resume", parents=[common])
    resume_parser.add_argument("--max-workers", type=positive_int, default=4)
    resume_parser.add_argument("--poll-seconds", type=positive_int, default=10)
    resume_parser.add_argument("--stop-after", type=positive_int)
    resume_parser.set_defaults(function=controller)
    worker_parser = sub.add_parser("worker", parents=[common])
    worker_parser.add_argument("--workload", type=int, required=True)
    worker_parser.add_argument("--run", type=positive_int, required=True)
    worker_parser.add_argument("--initial-seed", type=int, required=True)
    worker_parser.add_argument("--search-seed", type=int, required=True)
    worker_parser.set_defaults(function=worker)
    status_parser = sub.add_parser("status", parents=[common])
    status_parser.set_defaults(function=status)
    validate_parser = sub.add_parser("validate", parents=[common])
    validate_parser.add_argument("--allow-partial", action="store_true")
    validate_parser.set_defaults(function=validate)
    report_parser = sub.add_parser("report", parents=[common])
    report_parser.set_defaults(function=report)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    if len(set(args.workloads)) != len(args.workloads):
        raise SystemExit("--workloads must not contain duplicates")
    args.function(args)
