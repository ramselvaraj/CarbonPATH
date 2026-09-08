from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
import time

import pandas as pd

from chiplet.n_disagg import PackageGenerator
from chiplet.n_utils import get_area_power, get_sram_area_energy
from main import (
    calculate_cost,
    calibration_identity,
    gen_initial_arch,
    get_calib_cost_avg,
    sim_annealing,
)
from network import compare_memory_policies, write_network_comparison
from system.utils.ArchitectureIdentity import architecture_fingerprint
from system.utils.NetworkWorkload import load_network
from system.utils.SimulationCache import SIMULATION_MODEL_VERSION, SimulationCache


FULL_SEARCH_SPACE = Path("cfg/parameters/input.json")
REDUCED_SEARCH_SPACE = Path("cfg/experiments/reduced_search_space.json")
BASE_CACHE = Path("cfg/static_cache/static_cache.csv")
BASELINE_CALIBRATION = Path("cfg/calibration/calibration_8.json")
NETWORK_DIR = Path("cfg/experiments/dnn_workloads")
BASELINE_NETWORK = NETWORK_DIR / "four_layer_mlp.json"

CURRENT_ANNEALING = {
    "initial_temp": 40,
    "freezing_temp": 1e-3,
    "max_move_per_temp_step": 5,
    "cooling_rate": 0.3,
}


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trace_fingerprint(trace):
    serialized = trace.to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def generate_initial_architecture(config_path, seed, log_path=None):
    state = random.getstate()
    random.seed(seed)
    try:
        if log_path is None:
            return gen_initial_arch(str(config_path), stack_diff_size=True)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                return gen_initial_arch(str(config_path), stack_diff_size=True)
    finally:
        random.setstate(state)


def clone_cache(base_cache, directory):
    cache_path = Path(directory) / "cache.csv"
    shutil.copy2(base_cache, cache_path)
    return cache_path


def initialize_experiment_cache(source, destination):
    cache = pd.read_csv(source)
    if "model_version" in cache:
        cache = cache.loc[
            cache["model_version"] == SIMULATION_MODEL_VERSION
        ].copy()
    else:
        cache = cache.iloc[0:0].copy()
        cache["model_version"] = pd.Series(dtype=int)
    cache.to_csv(destination, index=False)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def run_search(
    *,
    label,
    output_dir,
    workload,
    workload_id,
    initial_architecture,
    search_seed,
    search_space,
    calibration_path,
    base_cache,
    annealing=None,
    intermediate_policy="cold_dram",
):
    annealing = annealing or CURRENT_ANNEALING
    run_dir = Path(output_dir) / label
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "initial_architecture.json", initial_architecture)

    temp_label = str(label).replace("/", "-").replace("\\", "-")
    with tempfile.TemporaryDirectory(prefix=f"carbonpath-{temp_label}-") as directory:
        cache_path = clone_cache(base_cache, directory)
        log_path = run_dir / "run.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                best_cost, best_architecture, trace, architecture_trace = (
                    sim_annealing(
                        wl_idx=workload_id,
                        workload_sequence=workload,
                        cache_file=cache_path,
                        run_name=str(Path(directory) / "simulation"),
                        cost_profile="t1",
                        calibration_iterations=1,
                        intermediate_policy=intermediate_policy,
                        random_seed=search_seed,
                        input_file_path=str(search_space),
                        calibration_file_path=str(calibration_path),
                        initial_architecture=initial_architecture,
                        **annealing,
                    )
                )
        elapsed = time.perf_counter() - started
        shutil.copy2(cache_path, base_cache)

    trace.to_csv(run_dir / "search_trace.csv", index=False)
    architecture_trace.to_csv(run_dir / "architecture_trace.csv", index=False)
    write_json(run_dir / "best_architecture.json", best_architecture)
    simulator_calls = log_path.read_text(encoding="utf-8").count(
        "[INFO] Running simulator"
    )
    return {
        "label": label,
        "search_seed": search_seed,
        "initial_fingerprint": architecture_fingerprint(initial_architecture),
        "best_fingerprint": architecture_fingerprint(best_architecture),
        "best_cost": float(best_cost),
        "attempted_moves": int(len(trace)),
        "accepted_moves": int(trace["move_accepted"].eq(True).sum()),
        "trace_fingerprint": trace_fingerprint(trace),
        "runtime_seconds": elapsed,
        "simulator_calls": simulator_calls,
        "best_architecture": best_architecture,
        "trace": trace,
    }


def build_reduced_candidates(config_path):
    config = load_json(config_path)
    package = config["pkg"]
    candidates = []
    for array, tech_node, memory, split_k, dataflow, ascending in itertools.product(
        config["sys_array"],
        config["tech_nodes"],
        package["mem_pkg_architecture"],
        (0, 1),
        ("ws", "os", "is"),
        (0, 1),
    ):
        for sram_size in config["sram_buf_sizes"][array]:
            logic_area, power = get_area_power(array, tech_node)
            sram_area, _ = get_sram_area_energy(sram_size, tech_node)
            chiplets = {
                "Chiplet_1": {
                    "tech_node": tech_node,
                    "sys_array_size": array,
                    "sram_buf": sram_size,
                    "area": logic_area + sram_area,
                    "power": power,
                }
            }
            package_generator = PackageGenerator(
                chiplets,
                package["inter_pkg_architecture"],
                package["mem_pkg_architecture"],
                package["protocol"],
                stack_diff_size=True,
            )
            architecture = copy.deepcopy(chiplets)
            architecture["pkg"] = package_generator.generate(
                hi_pkg_type="2d", mem_pkg_type=memory
            )
            architecture["WL_mapping"] = {
                "mapping": {
                    "chiplet_data_sharing_enabled": 0,
                    "if_splitting_k": split_k,
                    "dataflow": [dataflow],
                    "assign_workload_in_ascending_order": ascending,
                    "static_tiling": 0,
                    "merge_tiles": 0,
                }
            }
            candidates.append(architecture)
    return candidates


def evaluate_reduced_space(
    output_dir,
    workload,
    candidates,
    calibration_path,
    base_cache,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_json(calibration_path)
    rows = []
    with tempfile.TemporaryDirectory(prefix="carbonpath-exhaustive-") as directory:
        cache = SimulationCache(
            clone_cache(base_cache, directory),
            simulator_dir=str(Path(directory) / "simulation"),
        )
        with (output_dir / "exhaustive.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                for index, architecture in enumerate(candidates):
                    cost, normalized, raw = calculate_cost(
                        profile_name="t1",
                        cost_avgerage=calibration,
                        system_dict=architecture,
                        cache=cache,
                        workload_sequence=workload,
                        intermediate_policy="cold_dram",
                    )
                    chiplet = architecture["Chiplet_1"]
                    mapping = architecture["WL_mapping"]["mapping"]
                    rows.append(
                        {
                            "candidate_id": index,
                            "fingerprint": architecture_fingerprint(architecture),
                            "objective": float(cost),
                            "array": chiplet["sys_array_size"],
                            "tech_node": chiplet["tech_node"],
                            "sram_kib": chiplet["sram_buf"],
                            "memory": architecture["pkg"]["mem_pkg_conn"]["mem_type"],
                            "split_k": mapping["if_splitting_k"],
                            "dataflow": mapping["dataflow"][0],
                            "ascending": mapping[
                                "assign_workload_in_ascending_order"
                            ],
                            "latency_ns": raw["latency"],
                            "total_energy_pj": raw["energy"],
                            "area_mm2": raw["area"],
                            "cost_usd": raw["dollar"],
                        }
                    )
        cache.dump_cache()
        shutil.copy2(cache.dir, base_cache)

    results = pd.DataFrame(rows).sort_values(
        ["objective", "fingerprint"], ignore_index=True
    )
    results.to_csv(output_dir / "exhaustive_results.csv", index=False)
    optimum = float(results.iloc[0]["objective"])
    tolerance = max(1e-12, abs(optimum) * 1e-12)
    optimum_rows = results[(results["objective"] - optimum).abs() <= tolerance]
    optimum_fingerprints = set(optimum_rows["fingerprint"])
    write_json(
        output_dir / "optimum.json",
        {
            "candidate_count": len(results),
            "objective": optimum,
            "tie_count": len(optimum_rows),
            "fingerprints": sorted(optimum_fingerprints),
            "candidates": optimum_rows.to_dict(orient="records"),
        },
    )
    return results, optimum, optimum_fingerprints


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def calibrate_network(
    *,
    network,
    output_dir,
    samples,
    seed,
    base_cache,
    search_space,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "calibration.json"
    if calibration_path.exists():
        calibration = load_json(calibration_path)
        expected_identity = calibration_identity(
            load_json(search_space),
            network.to_workload_sequence(),
            network.intermediate_policy,
        )
        if calibration.get("_calibration_identity") == expected_identity:
            return calibration, calibration_path, 0.0

    with tempfile.TemporaryDirectory(
        prefix=f"carbonpath-calibration-{network.name}-"
    ) as directory:
        cache = SimulationCache(
            clone_cache(base_cache, directory),
            simulator_dir=str(Path(directory) / "simulation"),
        )
        random.seed(seed)
        started = time.perf_counter()
        with (output_dir / "calibration.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                calibration = get_calib_cost_avg(
                    calibration_iterations=samples,
                    config_path=str(search_space),
                    cache=cache,
                    calibration_file_path=str(calibration_path),
                    workload_sequence=network.to_workload_sequence(),
                    intermediate_policy=network.intermediate_policy,
                )
        elapsed = time.perf_counter() - started
        cache.dump_cache()
        shutil.copy2(cache.dir, base_cache)
    return calibration, calibration_path, elapsed


def architecture_summary(architecture):
    chiplets = [
        value for key, value in architecture.items() if key.startswith("Chiplet_")
    ]
    mapping = architecture["WL_mapping"]["mapping"]
    return {
        "chiplet_count": len(chiplets),
        "arrays": ",".join(chiplet["sys_array_size"] for chiplet in chiplets),
        "tech_nodes": ",".join(str(chiplet["tech_node"]) for chiplet in chiplets),
        "sram_kib": ",".join(str(chiplet["sram_buf"]) for chiplet in chiplets),
        "package": architecture["pkg"]["HI_pkg_type"],
        "memory": architecture["pkg"]["mem_pkg_conn"]["mem_type"],
        "dataflow": mapping["dataflow"][0],
        "split_k": mapping["if_splitting_k"],
    }


def network_shape_summary(network):
    return {
        "layers": len(network.layers),
        "gemms": ";".join(
            f"{layer.batch_size}x{layer.in_features}x{layer.out_features}"
            for layer in network.layers
        ),
        "macs": sum(
            layer.batch_size * layer.in_features * layer.out_features
            for layer in network.layers
        ),
        "weight_bytes": sum(
            layer.in_features * layer.out_features for layer in network.layers
        ),
        "maximum_intermediate_bytes": max(
            (
                layer.batch_size * layer.out_features
                for layer in network.layers[:-1]
            ),
            default=0,
        ),
    }


def run_dnn_experiment(
    *,
    output_dir,
    network_paths,
    base_cache,
    calibration_samples,
):
    rows = []
    for index, network_path in enumerate(network_paths):
        network = load_network(network_path)
        network_dir = Path(output_dir) / network.name
        calibration, calibration_path, calibration_runtime = calibrate_network(
            network=network,
            output_dir=network_dir,
            samples=calibration_samples,
            seed=5000 + index,
            base_cache=base_cache,
            search_space=FULL_SEARCH_SPACE,
        )
        initial = generate_initial_architecture(
            FULL_SEARCH_SPACE,
            6000 + index,
            network_dir / "initial_generation.log",
        )
        search = run_search(
            label="architecture_search",
            output_dir=network_dir,
            workload=network.to_workload_sequence(),
            workload_id=network.name,
            initial_architecture=initial,
            search_seed=7000 + index,
            search_space=FULL_SEARCH_SPACE,
            calibration_path=calibration_path,
            base_cache=base_cache,
        )

        with tempfile.TemporaryDirectory(
            prefix=f"carbonpath-evaluation-{network.name}-"
        ) as directory:
            cache = SimulationCache(
                clone_cache(base_cache, directory),
                simulator_dir=str(Path(directory) / "simulation"),
            )
            comparison, evaluations = compare_memory_policies(
                network,
                search["best_architecture"],
                cache,
                cost_profile="t1",
                calibration=calibration,
            )
            write_network_comparison(
                comparison, evaluations, network_dir / "policy_comparison"
            )
            cache.dump_cache()
            shutil.copy2(cache.dir, base_cache)

        cold = comparison.loc[
            comparison["requested_policy"] == "cold_dram"
        ].iloc[0]
        local = comparison.loc[
            comparison["requested_policy"] == "local_sram"
        ].iloc[0]
        rows.append(
            {
                "network": network.name,
                **network_shape_summary(network),
                "initial_fingerprint": search["initial_fingerprint"],
                "best_fingerprint": search["best_fingerprint"],
                "search_objective": search["best_cost"],
                "search_runtime_seconds": search["runtime_seconds"],
                "calibration_samples": calibration_samples,
                "calibration_runtime_seconds": calibration_runtime,
                **architecture_summary(search["best_architecture"]),
                "cold_latency_ns": cold["latency_ns"],
                "cold_total_energy_pj": cold["total_energy_pj"],
                "cold_objective": cold["objective"],
                "local_latency_ns": local["latency_ns"],
                "local_total_energy_pj": local["total_energy_pj"],
                "local_objective": local["objective"],
                "local_methods": local["selected_methods"],
                "latency_reduction_percent": (
                    (cold["latency_ns"] - local["latency_ns"])
                    / cold["latency_ns"]
                    * 100
                ),
                "energy_reduction_percent": (
                    (cold["total_energy_pj"] - local["total_energy_pj"])
                    / cold["total_energy_pj"]
                    * 100
                ),
                "area_mm2": cold["system_area_mm2"],
                "power_w": cold["system_power_w"],
                "cost_usd": cold["system_cost_usd"],
                "embodied_carbon_kg": cold["embodied_carbon_kg"],
                "operational_carbon_kg": cold["operational_carbon_kg"],
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(Path(output_dir) / "dnn_summary.csv", index=False)
    return results


def markdown_table(frame, columns):
    selected = frame[columns].copy()
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output_dir,
    replay,
    reduced_runs,
    reduced_optimum,
    optimum_fingerprints,
    full_runs,
    dnn_results,
):
    output_dir = Path(output_dir)
    replay_passed = (
        len(replay) == 2
        and replay["search_seed"].nunique() == 1
        and replay["initial_fingerprint"].nunique() == 1
        and replay["attempted_moves"].nunique() == 1
        and replay["attempted_moves"].iloc[0] > 0
        and replay["best_fingerprint"].nunique() == 1
        and replay["trace_fingerprint"].nunique() == 1
        and replay["best_cost"].nunique() == 1
    )
    hits = int(reduced_runs["reached_global_optimum"].sum())
    full_objective_min = full_runs["best_cost"].min()
    full_objective_max = full_runs["best_cost"].max()
    full_relative_spread = (
        (full_objective_max - full_objective_min) / abs(full_objective_min)
        if full_objective_min != 0
        else float("nan")
    )
    full_objective_agreement = full_relative_spread <= 0.001
    full_architecture_stable = full_runs["best_fingerprint"].nunique() == 1

    lines = [
        "# CarbonPATH Optimizer and DNN Experiment",
        "",
        "## Scope",
        "",
        "This experiment separates deterministic replay, a reliability benchmark in a fully enumerated reduced space, empirical full-space agreement, and per-DNN architecture search. The DNNs are sequential int8 linear-layer proxies; activations, branching, convolution, normalization, and attention are not modeled.",
        "",
        "## Replay",
        "",
        f"Fixed-seed in-process replay: **{'PASS' if replay_passed else 'FAIL'}**. Each repetition regenerated the initial architecture and then ran the annealer with the same recorded seeds; the runs produced {replay['best_fingerprint'].nunique()} best architecture fingerprint(s) and {replay['trace_fingerprint'].nunique()} trace fingerprint(s). The first run populated the current-version cache and the second reused those entries, so this also checks numerical equivalence across cold-to-warm cache states rather than two independent cold executions.",
        "",
        markdown_table(
            replay,
            [
                "label",
                "search_seed",
                "initial_fingerprint",
                "best_fingerprint",
                "best_cost",
                "runtime_seconds",
            ],
        ),
        "",
        "## Reduced-Space Global Optimum",
        "",
        f"The declared one-chiplet reduced space contained {len(pd.read_csv(output_dir / 'reduced_space' / 'exhaustive_results.csv'))} candidates and was exhaustively evaluated. Its global minimum objective was `{reduced_optimum:.12g}` with {len(optimum_fingerprints)} tied order-normalized architecture fingerprint(s).",
        "",
        f"The current 45-move annealer reached that known optimum in **{hits}/{len(reduced_runs)} deterministic start/seed cases ({hits / len(reduced_runs):.1%})**. This stratified panel is a benchmark, not a random sample from a declared population, so no binomial confidence interval or population success probability is claimed. It failed to reach the optimum from every tested start.",
        "",
        markdown_table(
            reduced_runs,
            [
                "label",
                "initial_fingerprint",
                "best_fingerprint",
                "best_cost",
                "optimality_gap",
                "reached_global_optimum",
            ],
        ),
        "",
        "## Full-Space Multi-Start",
        "",
        f"{len(full_runs)} start/seed pairs produced {full_runs['best_fingerprint'].nunique()} order-normalized best architecture fingerprints. Objectives ranged from `{full_objective_min:.12g}` to `{full_objective_max:.12g}` (relative spread `{full_relative_spread:.3%}`). Objective agreement within 0.1%: **{'PASS' if full_objective_agreement else 'FAIL'}**. Exact serialized-architecture stability: **{'PASS' if full_architecture_stable else 'FAIL'}**. Because both the start and search stream changed, this tests the combined multi-start protocol rather than isolating either factor. The full space was not enumerated, so these are best-known results, not proven global optima.",
        "",
        markdown_table(
            full_runs,
            [
                "label",
                "initial_fingerprint",
                "best_fingerprint",
                "best_cost",
                "runtime_seconds",
            ],
        ),
        "",
        "## DNN Workloads",
        "",
        markdown_table(
            dnn_results,
            [
                "network",
                "gemms",
                "best_fingerprint",
                "chiplet_count",
                "arrays",
                "memory",
                "dataflow",
                "cold_latency_ns",
                "cold_total_energy_pj",
                "local_latency_ns",
                "local_total_energy_pj",
                "latency_reduction_percent",
                "energy_reduction_percent",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "A fixed seed tests software replay only. Different starts test robustness, while exhaustive enumeration is required to claim a global optimum within a declared finite space. CarbonPATH uses finite geometric cooling, so full-space multi-start agreement is empirical evidence rather than proof.",
        "",
        "The current 45-move configuration did not reach the known optimum from every reduced-space start and did not produce objective agreement in the full-space panel. It should therefore be reported as start-and-search-stream sensitive under this combined protocol, not as yielding a unique correct architecture.",
        "",
        "Each DNN architecture is the best result from one seeded 45-move run using a separate ten-sample calibration. The workload, calibration panel, initial architecture, and search stream all differ, so neither normalized objectives nor architecture differences can be attributed solely to workload shape. These runs are functional demonstrations, not comparative or statistical evidence. The `t1` profile assigns zero weight to embodied and operational carbon, so carbon is reported but did not influence architecture selection.",
        "",
        "On each cold-DRAM-selected fixed architecture, local SRAM reduced latency and energy for the bottleneck and four-layer workloads. These are post-hoc policy evaluations, not policy-aware architecture searches. The classifier has no internal boundary. The ViT FFN and LLM projection pair fell back to cold DRAM because their producer/consumer mappings did not form a valid complete intermediate mapping; this was not a pure SRAM-capacity failure.",
        "",
        "## Sources",
        "",
        "Methodological and local implementation sources are recorded in [`../sources.md`](../sources.md). The run manifest, raw traces, architectures, calibration data, and CSV summaries in this directory provide the experiment evidence.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def regenerate_report(output_dir):
    output_dir = Path(output_dir)
    replay = pd.read_csv(output_dir / "replay" / "summary.csv")
    reduced_runs = pd.read_csv(
        output_dir / "reduced_space" / "multistart_summary.csv"
    )
    reduced_optimum_data = load_json(output_dir / "reduced_space" / "optimum.json")
    full_runs = pd.read_csv(
        output_dir / "full_space" / "multistart_summary.csv"
    )
    dnn_path = output_dir / "dnn_workloads" / "dnn_summary.csv"
    dnn_results = pd.read_csv(dnn_path)
    if "latency_reduction_percent" not in dnn_results:
        dnn_results["latency_reduction_percent"] = (
            (dnn_results["cold_latency_ns"] - dnn_results["local_latency_ns"])
            / dnn_results["cold_latency_ns"]
            * 100
        )
    if "energy_reduction_percent" not in dnn_results:
        dnn_results["energy_reduction_percent"] = (
            (
                dnn_results["cold_total_energy_pj"]
                - dnn_results["local_total_energy_pj"]
            )
            / dnn_results["cold_total_energy_pj"]
            * 100
        )
    if "local_methods" not in dnn_results:
        dnn_results["local_methods"] = [
            pd.read_csv(
                output_dir
                / "dnn_workloads"
                / network
                / "policy_comparison"
                / "policy_comparison.csv"
            )
            .loc[lambda frame: frame["requested_policy"] == "local_sram"]
            .iloc[0]["selected_methods"]
            for network in dnn_results["network"]
        ]
    dnn_results.to_csv(dnn_path, index=False)
    write_report(
        output_dir,
        replay,
        reduced_runs,
        reduced_optimum_data["objective"],
        set(reduced_optimum_data["fingerprints"]),
        full_runs,
        dnn_results,
    )


def git_value(*args):
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Run reproducibility, optimizer, and linear-DNN experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/optimizer_experiments/results"),
    )
    parser.add_argument("--base-cache", type=Path, default=BASE_CACHE)
    parser.add_argument("--reduced-runs", type=positive_int, default=20)
    parser.add_argument("--full-runs", type=positive_int, default=5)
    parser.add_argument("--calibration-samples", type=positive_int, default=10)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate report.md and derived DNN columns from existing results",
    )
    args = parser.parse_args()

    if args.report_only:
        regenerate_report(args.output_dir)
        print(f"Experiment report: {args.output_dir / 'report.md'}")
        return

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Choose a new path."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_cache = args.output_dir / "simulation_cache.csv"
    initialize_experiment_cache(args.base_cache, experiment_cache)

    baseline_network = load_network(BASELINE_NETWORK)
    baseline_workload = baseline_network.to_workload_sequence()
    manifest = {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "base_cache": str(args.base_cache),
        "base_cache_sha256": file_sha256(args.base_cache),
        "full_search_space_sha256": file_sha256(FULL_SEARCH_SPACE),
        "reduced_search_space_sha256": file_sha256(REDUCED_SEARCH_SPACE),
        "annealing": CURRENT_ANNEALING,
        "replay": {"initial_seed": 1101, "search_seed": 2101, "runs": 2},
        "reduced": {"runs": args.reduced_runs, "search_seed_start": 3100},
        "full": {"runs": args.full_runs, "initial_seed_start": 4100},
        "dnn_calibration_samples": args.calibration_samples,
        "source_sha256": {
            str(path): file_sha256(path)
            for path in [
                Path("main.py"),
                Path("network.py"),
                Path("config.py"),
                Path("system/utils/ArchitectureIdentity.py"),
                Path("system/utils/IntermediateMemoryPolicy.py"),
                Path("system/utils/NetworkWorkload.py"),
                Path("system/utils/Scheduler.py"),
                Path("system/utils/SimulationCache.py"),
                Path("system/utils/Simulator.py"),
                Path(__file__),
                FULL_SEARCH_SPACE,
                REDUCED_SEARCH_SPACE,
                Path("cfg/parameters/cost_profiles.json"),
                *sorted(NETWORK_DIR.glob("*.json")),
            ]
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    _, baseline_calibration_path, _ = calibrate_network(
        network=baseline_network,
        output_dir=args.output_dir / "baseline_calibration",
        samples=args.calibration_samples,
        seed=4900,
        base_cache=experiment_cache,
        search_space=FULL_SEARCH_SPACE,
    )
    _, reduced_calibration_path, _ = calibrate_network(
        network=baseline_network,
        output_dir=args.output_dir / "reduced_calibration",
        samples=args.calibration_samples,
        seed=4901,
        base_cache=experiment_cache,
        search_space=REDUCED_SEARCH_SPACE,
    )

    replay_dir = args.output_dir / "replay"
    replay_rows = []
    for index in range(2):
        initial = generate_initial_architecture(
            FULL_SEARCH_SPACE,
            1101,
            replay_dir / f"initial_generation_{index + 1}.log",
        )
        result = run_search(
            label=f"run_{index + 1}",
            output_dir=replay_dir,
            workload=baseline_workload,
            workload_id=8,
            initial_architecture=initial,
            search_seed=2101,
            search_space=FULL_SEARCH_SPACE,
            calibration_path=baseline_calibration_path,
            base_cache=experiment_cache,
        )
        replay_rows.append({key: value for key, value in result.items() if key not in {"best_architecture", "trace"}})
    replay = pd.DataFrame(replay_rows)
    replay.to_csv(replay_dir / "summary.csv", index=False)

    reduced_dir = args.output_dir / "reduced_space"
    candidates = build_reduced_candidates(REDUCED_SEARCH_SPACE)
    _, reduced_optimum, optimum_fingerprints = evaluate_reduced_space(
        reduced_dir,
        baseline_workload,
        candidates,
        reduced_calibration_path,
        experiment_cache,
    )
    reduced_rows = []
    for index in range(args.reduced_runs):
        candidate_index = round(index * (len(candidates) - 1) / max(args.reduced_runs - 1, 1))
        result = run_search(
            label=f"run_{index + 1:02d}",
            output_dir=reduced_dir / "runs",
            workload=baseline_workload,
            workload_id=8,
            initial_architecture=candidates[candidate_index],
            search_seed=3100 + index,
            search_space=REDUCED_SEARCH_SPACE,
            calibration_path=reduced_calibration_path,
            base_cache=experiment_cache,
        )
        gap = result["best_cost"] - reduced_optimum
        reduced_rows.append(
            {
                **{
                    key: value
                    for key, value in result.items()
                    if key not in {"best_architecture", "trace"}
                },
                "start_candidate_id": candidate_index,
                "optimality_gap": gap,
                "reached_global_optimum": (
                    abs(gap) <= max(1e-12, abs(reduced_optimum) * 1e-12)
                    and result["best_fingerprint"] in optimum_fingerprints
                ),
            }
        )
    reduced_runs = pd.DataFrame(reduced_rows)
    reduced_runs.to_csv(reduced_dir / "multistart_summary.csv", index=False)

    full_dir = args.output_dir / "full_space"
    full_rows = []
    for index in range(args.full_runs):
        initial = generate_initial_architecture(
            FULL_SEARCH_SPACE,
            4100 + index,
            full_dir / f"initial_generation_{index + 1}.log",
        )
        result = run_search(
            label=f"run_{index + 1:02d}",
            output_dir=full_dir,
            workload=baseline_workload,
            workload_id=8,
            initial_architecture=initial,
            search_seed=4200 + index,
            search_space=FULL_SEARCH_SPACE,
            calibration_path=baseline_calibration_path,
            base_cache=experiment_cache,
        )
        full_rows.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"best_architecture", "trace"}
            }
        )
    full_runs = pd.DataFrame(full_rows)
    full_runs.to_csv(full_dir / "multistart_summary.csv", index=False)

    network_paths = sorted(NETWORK_DIR.glob("*.json"))
    dnn_results = run_dnn_experiment(
        output_dir=args.output_dir / "dnn_workloads",
        network_paths=network_paths,
        base_cache=experiment_cache,
        calibration_samples=args.calibration_samples,
    )
    write_report(
        args.output_dir,
        replay,
        reduced_runs,
        reduced_optimum,
        optimum_fingerprints,
        full_runs,
        dnn_results,
    )
    manifest["baseline_calibration_sha256"] = file_sha256(
        baseline_calibration_path
    )
    manifest["reduced_calibration_sha256"] = file_sha256(
        reduced_calibration_path
    )
    manifest["experiment_cache_sha256"] = file_sha256(experiment_cache)
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Experiment report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
