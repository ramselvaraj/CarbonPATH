"""Annealing campaign for ATLAS networks with FPGA operations on several chiplets.

For every network and every arm (a list of allowed FPGA chiplet counts) the
campaign runs the same seeds, so the arms compare on equal starts. The objective
is ``t5``: latency, energy and area with equal weights (no dollar cost, no carbon).

Sub-commands::

    launch   run every missing job with a pool of worker processes (use nohup)
    status   one line per network and arm, from the status files of the runs
    report   write summary.csv and report.md from the finished runs
    run-one  run a single job (the launcher starts these)
    calibrate  calibrate the objective of one network

Each job writes into ``<out>/runs/<network>/<arm>/seed_<k>/``: ``status.json``
(updated at every temperature level, so a running job can be read), ``run.log``,
and at the end ``search_trace.csv``, ``best_design_point.json``, ``result.json``.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import pandas as pd

from config import calibration_mode
from main import (
    CALIBRATION_MODEL_VERSION,
    evaluate_atlas_design_point,
    gen_initial_arch,
    sim_annealing,
    summarize_design_points,
    validate_calibration,
)
from script.run_full_space_convergence import SCHEDULES
from script.run_optimizer_experiments import (
    BASE_CACHE,
    clone_cache,
    initialize_experiment_cache,
    load_json,
    write_json,
)
from system.utils.ArchitectureIdentity import canonicalize_chiplet_labels
from system.utils.ArchitectureModel import rebuild_package_for_endpoints
from system.utils.AtlasAnnealingMoves import validate_atlas_architecture
from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.AtlasObjective import RawWeightedSumObjective, build_atlas_objective
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.FpgaNewChiplet import new_fpga_chiplet
from system.utils.SimulationCache import SimulationCache
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation

# dense_relu_funnel and dense_relu_deep are left out: their ReLU tensors have
# 131,072 values or more, far beyond the sizes Rule4ML was trained on, and the
# input buffer would need at least 57 BRAM blocks. Every random design is
# rejected, so the search has no valid start.
NETWORKS = {
    "softmax_only": Path("cfg/examples/atlas/dense_softmax.graph_dump.json"),
    "relu_softmax": Path("cfg/examples/atlas/dense_relu_softmax.graph_dump.json"),
    "all_activations": Path(
        "cfg/examples/atlas/dense_rule4ml_activations.graph_dump.json"
    ),
}
# Each arm names the allowed FPGA chiplet counts and the policy for a new chiplet.
# A new chiplet is a copy of an existing one, or is drawn from the search space
# (so it can be a small board).
ARMS = {
    "A": {"count": [1], "new_chiplet": "copy_existing_v1"},
    "B": {"count": [1, 2], "new_chiplet": "copy_existing_v1"},
    "C": {"count": [1, 2, 3], "new_chiplet": "copy_existing_v1"},
    "D": {"count": [1, 2, 3], "new_chiplet": "from_search_space_v1"},
}
PROFILE = Path("cfg/profiles/rule4ml_fpga_ops_v3.json")
SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space.json")
# A suite names its networks and its search space. ``--suite`` selects one.
LARGE_NETWORKS = {
    "wide_k_mlp": Path("cfg/examples/atlas/wide_k_mlp.graph_dump.json"),
    "row_split": Path("cfg/examples/atlas/row_split.graph_dump.json"),
}
LARGE_SEARCH_SPACE = Path("cfg/experiments/atlas_large_search_space.json")
SUITES = ("small", "large")
LEGACY_SPACE = Path("cfg/parameters/input.json")
OBJECTIVE = "t5"
INITIAL_SEED_BASE = 12000
SEARCH_SEED_BASE = 13000
CALIBRATION_SEED_BASE = 10000
CALIBRATION_SAMPLES = 10000
SCHEDULES = {
    **SCHEDULES,
    # The defaults of main.py sim_annealing: 4000 -> 0.001 by 0.99, 20 moves per level.
    "main_default_30260": {
        "initial_temp": 4000,
        "freezing_temp": 1e-3,
        "max_move_per_temp_step": 20,
        "cooling_rate": 0.99,
    },
    # Same cooling as main.py with 50 moves per level.
    "long_75650": {
        "initial_temp": 4000,
        "freezing_temp": 1e-3,
        "max_move_per_temp_step": 50,
        "cooling_rate": 0.99,
    },
    "pilot": {
        "initial_temp": 8,
        "freezing_temp": 1e-1,
        "max_move_per_temp_step": 4,
        "cooling_rate": 0.5,
    },
}


def apply_suite(suite):
    """Point the module at the networks and the search space of ``suite``."""
    global SEARCH_SPACE
    if suite == "large":
        NETWORKS.clear()
        NETWORKS.update(LARGE_NETWORKS)
        SEARCH_SPACE = LARGE_SEARCH_SPACE


def planned_moves(schedule):
    levels = math.ceil(
        math.log(schedule["freezing_temp"] / schedule["initial_temp"])
        / math.log(schedule["cooling_rate"])
    )
    return levels * schedule["max_move_per_temp_step"]


def initial_architecture(seed, space):
    """A random systolic-array package with one FPGA chiplet, from ``seed``."""
    state = random.getstate()
    random.seed(seed)
    try:
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            base = gen_initial_arch(str(LEGACY_SPACE), stack_diff_size=True)
    finally:
        random.setstate(state)
    rng = random.Random(seed)
    moved = copy.deepcopy(base)
    used = [int(key.split("_", 1)[1]) for key in moved if key.startswith("Chiplet_")]
    params = {**space, "fpga": {**space["fpga"], "new_chiplet": "from_search_space_v1"}}
    moved[f"Chiplet_{max(used) + 1}"] = new_fpga_chiplet(moved, [], params, rng)
    architecture = canonicalize_chiplet_labels(
        rebuild_package_for_endpoints(moved, space, rng)
    )
    validate_atlas_architecture(architecture)
    return architecture


def arm_space(arm):
    space = load_json(SEARCH_SPACE)
    space["fpga"]["count"] = list(ARMS[arm]["count"])
    space["fpga"]["new_chiplet"] = ARMS[arm]["new_chiplet"]
    return space


def make_cache(base_cache, directory):
    return SimulationCache(
        clone_cache(base_cache, directory),
        simulator_dir=str(Path(directory) / "simulation"),
    )


# ---------------------------------------------------------------- calibration


_WORKER = {}


def _calibration_worker_init(out, network, directory_root):
    """Give each worker its own cache clone, graph and profile."""
    directory = tempfile.mkdtemp(prefix=f"w{os.getpid()}-", dir=directory_root)
    _WORKER["cache"] = make_cache(Path(out) / "base_cache.csv", directory)
    _WORKER["graph"] = load_atlas_graph(NETWORKS[network])
    _WORKER["profile"] = load_evaluation_profile(PROFILE)
    _WORKER["space"] = load_json(SEARCH_SPACE)
    _WORKER["scoreless"] = RawWeightedSumObjective({"latency_ns": 0.0})


def _calibration_sample(seed):
    """Evaluate one random package. Returns (seed, point or None, reason)."""
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        try:
            architecture = initial_architecture(seed, _WORKER["space"])
            point, _ = evaluate_atlas_design_point(
                _WORKER["cache"], architecture, _WORKER["graph"],
                _WORKER["profile"], _WORKER["scoreless"],
            )
        except UnsupportedEvaluation as error:
            return seed, None, str(error)
    return seed, point, ""


def calibrate(out, network, samples=CALIBRATION_SAMPLES, workers=1):
    """Calibrate the objective of one network with random packages that hold an FPGA.

    Seeds are evaluated in parallel but consumed in seed order, so the first
    ``samples`` valid packages are the same for any worker count.
    """
    Path(out).mkdir(parents=True, exist_ok=True)
    if not (Path(out) / "base_cache.csv").exists():
        initialize_experiment_cache(BASE_CACHE, Path(out) / "base_cache.csv")
    calibration_dir = Path(out) / "calibration" / network
    calibration_dir.mkdir(parents=True, exist_ok=True)
    path = calibration_dir / "calibration.json"
    if path.exists():
        return path
    points, skipped = [], []
    attempt = 0
    limit = 20 * samples
    seeds = range(CALIBRATION_SEED_BASE, CALIBRATION_SEED_BASE + limit)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"cp-calib-{network}-") as directory:
        context = multiprocessing.get_context("fork")
        with context.Pool(
            workers, _calibration_worker_init, (out, network, directory)
        ) as pool:
            for seed, point, reason in pool.imap(_calibration_sample, seeds, chunksize=8):
                attempt += 1
                if point is None:
                    skipped.append({"seed": seed, "reason": reason})
                else:
                    points.append(point)
                if attempt % 500 == 0:
                    (calibration_dir / "progress.txt").write_text(
                        f"{len(points)} valid, {len(skipped)} rejected, "
                        f"{time.time() - started:.0f} s\n"
                    )
                if len(points) >= samples:
                    pool.terminate()
                    break
    if len(points) < samples:
        raise RuntimeError(
            f"{network}: only {len(points)} of {samples} calibration "
            f"samples evaluated; {len(skipped)} were rejected"
        )
    identity = f"{network}|{PROFILE.name}|{CALIBRATION_MODEL_VERSION}|{samples}|{attempt}"
    cost_averages = summarize_design_points(
        points, calibration_identity=identity, model_version=CALIBRATION_MODEL_VERSION
    )
    validate_calibration(cost_averages)
    cost_averages["_calibration_samples"] = len(points)
    cost_averages["_skipped_samples"] = skipped[:200]
    cost_averages["_skipped_count"] = len(skipped)
    write_json(path, cost_averages)
    return path


# ------------------------------------------------------------------ one run


def job_dir(out, network, arm, k):
    return Path(out) / "runs" / network / arm / f"seed_{k:02d}"


def write_status(directory, **fields):
    write_json(Path(directory) / "status.json", {**fields, "updated": time.time()})


def run_one(out, network, arm, k, schedule_name):
    directory = job_dir(out, network, arm, k)
    directory.mkdir(parents=True, exist_ok=True)
    schedule = SCHEDULES[schedule_name]
    total = planned_moves(schedule)
    started = time.time()
    counts = {"attempted": 0, "accepted": 0, "invalid": 0, "evaluation_error": 0}
    state = {"best": None}
    write_status(directory, state="running", moves_done=0, moves_planned=total,
                 started=started, **counts)
    try:
        graph = load_atlas_graph(NETWORKS[network])
        profile = load_evaluation_profile(PROFILE)
        space = arm_space(arm)
        calibration = load_json(Path(out) / "calibration" / network / "calibration.json")
        objective = build_atlas_objective(
            objective_id=OBJECTIVE,
            config={"default": OBJECTIVE},
            calibration=calibration,
            normalization_mode=calibration_mode,
        )
        architecture = initial_architecture(INITIAL_SEED_BASE + k, space)
        write_json(directory / "initial_architecture.json", architecture)

        def on_level(_decision, rows):
            for row in rows:
                counts["attempted"] += 1
                reason = row.get("acceptance_reason")
                counts["accepted"] += bool(row.get("move_accepted"))
                counts["invalid"] += reason in ("invalid", "unchanged")
                counts["evaluation_error"] += reason == "evaluation_error"
                state["best"] = row["best_cost_after"]
            write_status(
                directory, state="running", moves_done=counts["attempted"],
                moves_planned=total, started=started, best_cost=state["best"],
                temperature=rows[-1]["temperature"] if rows else None, **counts,
            )

        with tempfile.TemporaryDirectory(prefix=f"cp-{network}-{arm}-{k}-") as work:
            base = Path(out) / "base_cache.csv"
            cache_path = clone_cache(base, work)
            with (directory / "run.log").open("w") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    best_cost, best_arch, trace, _ = sim_annealing(
                        wl_idx=None,
                        workload_sequence=None,
                        cache_file=str(cache_path),
                        run_name=str(Path(work) / "simulation"),
                        cost_profile=OBJECTIVE,
                        calibration_iterations=1,
                        random_seed=SEARCH_SEED_BASE + k,
                        initial_architecture=architecture,
                        level_callback=on_level,
                        atlas_graph=graph,
                        atlas_profile=profile,
                        atlas_search_space=space,
                        atlas_objective=objective,
                        **schedule,
                    )
        trace.to_csv(directory / "search_trace.csv", index=False)
        write_json(directory / "best_architecture.json", best_arch)
        write_json(directory / "best_design_point.json", trace.attrs["best_design_point"])
        write_json(
            directory / "result.json",
            summarize_run(trace, best_cost, best_arch, trace.attrs, time.time() - started),
        )
        write_status(directory, state="done", moves_done=len(trace), moves_planned=total,
                     started=started, best_cost=float(best_cost), **counts)
    except BaseException:
        (directory / "error.txt").write_text(traceback.format_exc())
        write_status(directory, state="failed", moves_done=counts["attempted"],
                     moves_planned=total, started=started, **counts)
        raise


def summarize_run(trace, best_cost, best_arch, attrs, seconds):
    valid = trace[trace["atlas_objective"].notna()] if "atlas_objective" in trace else trace
    best_row = valid.loc[valid["atlas_objective"].idxmin()] if len(valid) else None
    fpga = [
        key for key, value in best_arch.items()
        if key.startswith("Chiplet_") and value.get("chiplet_type") == "fpga"
    ]
    mapping = attrs.get("best_workload_mapping", {})
    operations = mapping.get("operations", [])
    split = [
        operation.get("operation")
        for operation in operations
        if operation.get("endpoint_kind") == "fpga"
        and len(operation.get("endpoints", ())) > 1
    ]
    reasons = trace["acceptance_reason"].value_counts().to_dict() if "acceptance_reason" in trace else {}
    return {
        "best_cost": float(best_cost),
        "latency_ns": float(best_row["latency_ns"]) if best_row is not None else None,
        "total_energy_pj": float(best_row["total_energy_pj"]) if best_row is not None else None,
        "area_mm2": float(best_row["area_mm2"]) if best_row is not None else None,
        "fpga_chiplets": len(fpga),
        "split_operations": split,
        "attempted_moves": int(len(trace)),
        "accepted_moves": int(trace["move_accepted"].eq(True).sum()),
        "evaluation_errors": int(reasons.get("evaluation_error", 0)),
        "invalid_or_unchanged": int(reasons.get("invalid", 0) + reasons.get("unchanged", 0)),
        "runtime_seconds": seconds,
    }


# ----------------------------------------------------------- launch / status


def jobs(networks, arms, runs):
    return [(n, a, k) for k in range(runs) for n in networks for a in arms]


def launch(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "controller.pid").write_text(str(os.getpid()))
    log = (out / "controller.log").open("a", buffering=1)

    def say(message):
        log.write(f"{time.strftime('%F %T')} {message}\n")

    if not (out / "base_cache.csv").exists():
        initialize_experiment_cache(BASE_CACHE, out / "base_cache.csv")
    write_json(out / "manifest.json", {
        "networks": {n: str(NETWORKS[n]) for n in args.networks},
        "arms": {a: ARMS[a] for a in args.arms},
        "runs": args.runs, "schedule": args.schedule,
        "schedule_values": SCHEDULES[args.schedule],
        "planned_moves": planned_moves(SCHEDULES[args.schedule]),
        "objective": OBJECTIVE, "profile": str(PROFILE),
        "search_space": str(SEARCH_SPACE),
        "seeds": {"initial": INITIAL_SEED_BASE, "search": SEARCH_SEED_BASE,
                  "calibration": CALIBRATION_SEED_BASE},
    })
    for network in args.networks:
        say(f"calibrate {network}")
        calibrate(out, network, args.calibration_samples, args.workers)
    pending = [
        job for job in jobs(args.networks, args.arms, args.runs)
        if not (job_dir(out, *job[:2], job[2]) / "result.json").exists()
    ]
    say(f"{len(pending)} jobs to run, {args.workers} workers")
    running = {}
    while pending or running:
        for job, process in list(running.items()):
            if process.poll() is not None:
                say(f"{'done' if process.returncode == 0 else 'FAILED'} {job}")
                del running[job]
        while pending and len(running) < args.workers:
            job = pending.pop(0)
            network, arm, k = job
            directory = job_dir(out, network, arm, k)
            directory.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                [sys.executable, "-W", "ignore", "-m", "script.run_fpga_campaign",
                 "run-one", "--out", str(out), "--network", network, "--arm", arm,
                 "--seed-index", str(k), "--schedule", args.schedule,
                 "--suite", args.suite],
                stdout=(directory / "proc.log").open("w"), stderr=subprocess.STDOUT,
            )
            running[job] = process
            say(f"start {job}")
        time.sleep(5)
    say("all jobs finished")


def read_runs(out):
    rows = []
    for status_path in sorted(Path(out).glob("runs/*/*/seed_*/status.json")):
        directory = status_path.parent
        network, arm = directory.parent.parent.name, directory.parent.name
        row = {"network": network, "arm": arm, "seed": directory.name,
               **load_json(status_path)}
        result = directory / "result.json"
        if result.exists():
            row.update({f"r_{key}": value for key, value in load_json(result).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def status(args):
    runs = read_runs(args.out)
    if runs.empty:
        print("no runs yet")
        return
    now = time.time()
    for (network, arm), group in runs.groupby(["network", "arm"]):
        done = (group["state"] == "done").sum()
        failed = (group["state"] == "failed").sum()
        running = (group["state"] == "running").sum()
        progress = (group["moves_done"] / group["moves_planned"]).mean() * 100
        best = group["best_cost"].dropna()
        print(f"{network:16s} {arm}  done {done:2d}  running {running:2d}  failed {failed:2d}  "
              f"progress {progress:5.1f}%  best {best.min() if len(best) else float('nan'):9.4f}  "
              f"median {best.median() if len(best) else float('nan'):9.4f}")
    if args.runs:
        print()
        for _, row in runs.iterrows():
            age = now - row["updated"]
            print(f"{row['network']:16s} {row['arm']} {row['seed']}  {row['state']:8s} "
                  f"{row['moves_done']}/{row['moves_planned']}  best {row.get('best_cost')}  "
                  f"errors {row.get('evaluation_error')}  (updated {age:.0f}s ago)")


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def report(args):
    runs = read_runs(args.out)
    done = runs[runs["state"] == "done"]
    if done.empty:
        print("no finished runs")
        return
    rows = []
    for (network, arm), group in done.groupby(["network", "arm"]):
        rows.append({
            "network": network, "arm": arm, "runs": len(group),
            "best_cost_min": group["r_best_cost"].min(),
            "best_cost_median": group["r_best_cost"].median(),
            "best_cost_mean": group["r_best_cost"].mean(),
            "latency_ns_median": group["r_latency_ns"].median(),
            "energy_pj_median": group["r_total_energy_pj"].median(),
            "area_mm2_median": group["r_area_mm2"].median(),
            "fpga_chiplets_mean": group["r_fpga_chiplets"].mean(),
            "share_with_2plus_chiplets": (group["r_fpga_chiplets"] >= 2).mean(),
            "share_with_split_operation": group["r_split_operations"].map(bool).mean(),
            "evaluation_error_share": (
                group["r_evaluation_errors"] / group["r_attempted_moves"]
            ).mean(),
            "runtime_minutes_median": group["r_runtime_seconds"].median() / 60,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(args.out) / "summary.csv", index=False)
    manifest = load_json(Path(args.out) / "manifest.json")
    lines = [
        "# FPGA chiplet annealing campaign", "",
        f"Objective `{manifest['objective']}` (latency, energy, area; no dollar cost, no carbon). "
        f"Schedule `{manifest['schedule']}` ({manifest['planned_moves']} moves). "
        f"Arms: {manifest['arms']}.", "",
        markdown_table(summary.round(4)), "",
    ]
    (Path(args.out) / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ALL_NETWORKS = {**NETWORKS, **LARGE_NETWORKS}
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "status", "report", "run-one", "calibrate"):
        command = sub.add_parser(name)
        command.add_argument("--out", required=True)
        command.add_argument("--suite", default="small", choices=SUITES)
        if name == "launch":
            command.add_argument("--networks", nargs="+", default=None, choices=ALL_NETWORKS)
            command.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
            command.add_argument("--runs", type=int, default=10)
            command.add_argument("--workers", type=int, default=10)
            command.add_argument("--schedule", default="pilot", choices=SCHEDULES)
        if name == "status":
            command.add_argument("--runs", action="store_true", help="also list every run")
        if name in ("launch", "calibrate"):
            command.add_argument("--calibration-samples", type=int, default=CALIBRATION_SAMPLES)
        if name == "calibrate":
            command.add_argument("--workers", type=int, default=1)
        if name in ("run-one", "calibrate"):
            command.add_argument("--network", required=True, choices=ALL_NETWORKS)
        if name == "run-one":
            command.add_argument("--arm", required=True, choices=ARMS)
            command.add_argument("--seed-index", type=int, required=True)
            command.add_argument("--schedule", required=True, choices=SCHEDULES)
    args = parser.parse_args()
    apply_suite(args.suite)
    if args.command == "launch" and args.networks is None:
        args.networks = list(NETWORKS)
    if args.command == "launch":
        launch(args)
    elif args.command == "status":
        status(args)
    elif args.command == "report":
        report(args)
    elif args.command == "calibrate":
        calibrate(args.out, args.network, args.calibration_samples, args.workers)
    else:
        run_one(args.out, args.network, args.arm, args.seed_index, args.schedule)


if __name__ == "__main__":
    main()
