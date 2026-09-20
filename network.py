from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from chiplet.n_utils import calculate_system_metrics
from main import (
    calculate_cost,
    calibration_identity,
    evaluate_atlas_graph,
    get_calib_cost_avg,
    simulate_latency_energy,
    validate_calibration,
)
from system.utils.AtlasGraphAdapter import (
    AtlasGraph,
    default_graph_name,
    parse_atlas_graph,
)
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.IntermediateMemoryPolicy import INTERMEDIATE_POLICIES
from system.utils.NetworkWorkload import (
    LinearNetwork,
    canonical_fingerprint,
    parse_network_entry,
)
from system.utils.SimulationCache import SimulationCache


EXPLICIT_MEMORY_POLICIES = (
    "cold_dram",
    "ideal_on_chip",
    "local_sram",
    "direct_forward",
)


@dataclass
class NetworkEvaluation:
    summary: dict
    layers: pd.DataFrame
    boundaries: pd.DataFrame
    mapping: pd.DataFrame


def _load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_workload(path):
    """Load either a legacy linear network or a direct ATLAS graph dump."""
    entry = _load_json(path)
    if isinstance(entry, list):
        return parse_atlas_graph(entry, default_graph_name(path))
    return parse_network_entry(entry)


ATLAS_LAYER_COLUMNS = [
    "layer_index",
    "layer_name",
    "operation_type",
    "evaluator_id",
    "endpoint_id",
    "endpoint_kind",
    "M",
    "K",
    "N",
    "element_count",
    "compute_latency_ns",
    "movement_latency_ns",
    "latency_ns",
    "compute_energy_pj",
    "movement_energy_pj",
    "movement_method",
    "movement_source_endpoint",
    "movement_destination_endpoint",
    "total_energy_pj",
]


def _atlas_operation_row(result, power):
    movement = result.movement
    movement_latency_ns = movement.latency_ns if movement is not None else 0.0
    movement_energy_pj = movement.energy_pj if movement is not None else 0.0
    latency_ns = result.compute_latency_ns + movement_latency_ns
    return {
        "layer_index": result.index,
        "layer_name": result.operation_id,
        "operation_type": result.operation_type,
        "evaluator_id": result.evaluator_id,
        "endpoint_id": result.endpoint_id,
        "endpoint_kind": result.endpoint_kind,
        "M": result.m,
        "K": result.k,
        "N": result.n,
        "element_count": result.element_count,
        "compute_latency_ns": result.compute_latency_ns,
        "movement_latency_ns": movement_latency_ns,
        "latency_ns": latency_ns,
        "compute_energy_pj": result.compute_energy_pj,
        "movement_energy_pj": movement_energy_pj,
        "movement_method": movement.method if movement is not None else None,
        "movement_source_endpoint": (
            movement.source_endpoint if movement is not None else None
        ),
        "movement_destination_endpoint": (
            movement.destination_endpoint if movement is not None else None
        ),
        "total_energy_pj": (
            power * latency_ns * 1000
            + result.compute_energy_pj
            + movement_energy_pj
        ),
    }


def evaluate_atlas_network(graph, architecture, cache, intermediate_policy=None, profile=None):
    power, area, dollar_cost = calculate_system_metrics(architecture)
    evaluation = evaluate_atlas_graph(
        cache,
        architecture,
        graph,
        profile=profile,
        transfer_model=architecture.get("transfer_model", {}),
    )
    latency_ns = evaluation.latency_ns
    baseline_energy_pj = power * latency_ns * 1000
    compute_energy_pj = evaluation.compute_energy_pj
    movement_energy_pj = evaluation.movement_energy_pj
    total_energy_pj = baseline_energy_pj + compute_energy_pj + movement_energy_pj

    layer_rows = [
        _atlas_operation_row(result, power) for result in evaluation.results
    ]
    operation_counts = {}
    for result in evaluation.results:
        operation_counts[result.operation_type] = (
            operation_counts.get(result.operation_type, 0) + 1
        )
    gemm_count = operation_counts.get("gemm", 0)
    relu_count = operation_counts.get("relu", 0)

    summary = {
        "network": graph.name,
        "network_fingerprint": graph.fingerprint(),
        "dtype": graph.dtype,
        "evaluation_profile": evaluation.profile.name,
        "evaluation_profile_version": evaluation.profile.version,
        "evaluation_profile_fingerprint": evaluation.profile.fingerprint(),
        "layer_count": len(evaluation.results),
        "operation_count": len(evaluation.results),
        "operation_counts": operation_counts,
        "gemm_count": gemm_count,
        "relu_count": relu_count,
        "placement_policy": evaluation.profile.placement_policy,
        "movement_policy": evaluation.profile.movement_policy,
        "transfer_model": evaluation.profile.transfer_model,
        "latency_ns": latency_ns,
        "baseline_energy_pj": baseline_energy_pj,
        "compute_energy_pj": compute_energy_pj,
        "movement_energy_pj": movement_energy_pj,
        "total_energy_pj": total_energy_pj,
        "system_power_w": power,
        "system_area_mm2": area,
        "system_cost_usd": dollar_cost,
        "calibrated": False,
        "cost_profile": None,
        "objective": None,
        "normalized_metrics": None,
        "embodied_carbon_kg": None,
        "operational_carbon_kg": None,
    }
    return NetworkEvaluation(
        summary=summary,
        layers=pd.DataFrame(layer_rows, columns=ATLAS_LAYER_COLUMNS),
        boundaries=pd.DataFrame(),
        mapping=pd.DataFrame(),
    )


def evaluate_network(
    network: LinearNetwork,
    architecture,
    cache,
    intermediate_policy=None,
    cost_profile="t1",
    calibration=None,
    profile=None,
):
    if isinstance(network, AtlasGraph):
        return evaluate_atlas_network(
            network, architecture, cache, intermediate_policy, profile=profile
        )

    if profile is not None:
        raise ValueError(
            "an evaluation profile is only supported for ATLAS graph evaluation"
        )

    policy = intermediate_policy or network.intermediate_policy

    power, area, dollar_cost = calculate_system_metrics(architecture)
    latency, communication, sram, gemm_metrics, boundary_plans = (
        simulate_latency_energy(
            cache,
            architecture,
            network.to_workload_sequence(),
            intermediate_policy=policy,
        )
    )
    compute_energy = power * latency * 1000
    total_energy = communication + sram + compute_energy
    objective = None
    normalized_metrics = None
    calibrated_raw_metrics = None
    if calibration is not None:
        objective, normalized_metrics, calibrated_raw_metrics = calculate_cost(
            profile_name=cost_profile,
            cost_avgerage=calibration,
            system_dict=architecture,
            cache=cache,
            workload_sequence=network.to_workload_sequence(),
            intermediate_policy=policy,
            simulation_result=(
                latency,
                communication,
                sram,
                gemm_metrics,
                boundary_plans,
            ),
            system_metrics=(power, area, dollar_cost),
        )

    layer_rows = []
    mapping_rows = []
    for index, metrics in enumerate(gemm_metrics, start=1):
        layer_rows.append(
            {
                "layer_index": index,
                "layer_name": metrics["name"],
                "op": "linear",
                "M": metrics["shape"][0],
                "K": metrics["shape"][1],
                "N": metrics["shape"][2],
                "latency_ns": metrics["latency_ns"],
                "dram_interconnect_energy_pj": metrics[
                    "dram_interconnect_energy_pj"
                ],
                "sram_energy_pj": metrics["sram_energy_pj"],
                "compute_energy_pj": power * metrics["latency_ns"] * 1000,
                "total_energy_pj": (
                    metrics["dram_interconnect_energy_pj"]
                    + metrics["sram_energy_pj"]
                    + power * metrics["latency_ns"] * 1000
                ),
            }
        )
        for tile_index, tile in enumerate(metrics["tile_mappings"], start=1):
            mapping_rows.append(
                {
                    "layer_index": index,
                    "layer_name": metrics["name"],
                    "tile_index": tile_index,
                    **tile,
                }
            )

    boundary_columns = [
        "boundary_index",
        "producer_layer",
        "consumer_layer",
        "tensor_shape",
        "intermediate_bytes",
        "requested_policy",
        "selected_method",
        "retained_bytes",
        "forwarded_bytes",
        "dram_spilled_bytes",
        "dram_traffic_bytes",
        "latency_ns",
        "energy_pj",
        "producer_cores",
        "consumer_cores",
        "routes",
        "fallback_reason",
    ]
    boundary_rows = []
    for plan in boundary_plans:
        producer = network.layers[plan.boundary_index - 1]
        consumer = network.layers[plan.boundary_index]
        boundary_rows.append(
            {
                "boundary_index": plan.boundary_index,
                "producer_layer": producer.name,
                "consumer_layer": consumer.name,
                "tensor_shape": f"{producer.batch_size}x{producer.out_features}",
                "intermediate_bytes": plan.intermediate_bytes,
                "requested_policy": plan.requested_policy,
                "selected_method": plan.selected_method,
                "retained_bytes": plan.retained_bytes,
                "forwarded_bytes": plan.forwarded_bytes,
                "dram_spilled_bytes": plan.dram_spilled_bytes,
                "dram_traffic_bytes": plan.dram_traffic_bytes,
                "latency_ns": plan.latency_ns,
                "energy_pj": plan.energy_pj,
                "producer_cores": ",".join(map(str, plan.producer_cores)),
                "consumer_cores": ",".join(map(str, plan.consumer_cores)),
                "routes": ";".join(
                    "->".join(map(str, route)) for route in plan.routes
                ),
                "fallback_reason": plan.fallback_reason,
            }
        )

    summary = {
        "network": network.name,
        "network_fingerprint": network.fingerprint(),
        "dtype": network.dtype,
        "layer_count": len(network.layers),
        "requested_policy": policy,
        "latency_ns": latency,
        "communication_energy_pj": communication,
        "sram_energy_pj": sram,
        "compute_energy_pj": compute_energy,
        "total_energy_pj": total_energy,
        "system_power_w": power,
        "system_area_mm2": area,
        "system_cost_usd": dollar_cost,
        "calibrated": calibration is not None,
        "cost_profile": cost_profile if calibration is not None else None,
        "objective": objective,
        "normalized_metrics": normalized_metrics,
        "embodied_carbon_kg": (
            calibrated_raw_metrics["embCarbon"]
            if calibrated_raw_metrics is not None
            else None
        ),
        "operational_carbon_kg": (
            calibrated_raw_metrics["opeCarbon"]
            if calibrated_raw_metrics is not None
            else None
        ),
    }
    return NetworkEvaluation(
        summary=summary,
        layers=pd.DataFrame(layer_rows),
        boundaries=pd.DataFrame(boundary_rows, columns=boundary_columns),
        mapping=pd.DataFrame(mapping_rows),
    )


def write_network_evaluation(evaluation, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation.summary, file, indent=2)
    evaluation.layers.to_csv(output_dir / "layers.csv", index=False)
    evaluation.boundaries.to_csv(output_dir / "boundaries.csv", index=False)
    evaluation.mapping.to_csv(output_dir / "mapping.csv", index=False)

    lines = [
        f"# Network Evaluation: {evaluation.summary['network']}",
        "",
        "## Summary",
        "",
        f"- Layers: {evaluation.summary['layer_count']}",
        "- Policy: `"
        f"{evaluation.summary.get('requested_policy') or evaluation.summary.get('movement_policy')}"
        "`",
        f"- Latency: {evaluation.summary['latency_ns'] / 1e3:.6f} us",
        f"- Total energy: {evaluation.summary['total_energy_pj'] / 1e6:.6f} uJ",
        f"- System power: {evaluation.summary['system_power_w']:.6f} W",
        f"- System area: {evaluation.summary['system_area_mm2']:.6f} mm2",
        "",
        "## Files",
        "",
        "- `summary.json`: aggregate network and architecture metrics",
        "- `layers.csv`: one row per linear layer",
        "- `boundaries.csv`: one row per intermediate tensor",
        "- `mapping.csv`: one row per GEMM tile assignment",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def compare_memory_policies(
    network,
    architecture,
    cache,
    cost_profile="t1",
    calibration=None,
):
    policies = list(EXPLICIT_MEMORY_POLICIES)
    evaluations = []
    rows = []
    for policy in policies:
        evaluation = evaluate_network(
            network,
            architecture,
            cache,
            intermediate_policy=policy,
            cost_profile=cost_profile,
            calibration=calibration,
        )
        evaluations.append(evaluation)
        selected_methods = ",".join(
            evaluation.boundaries.get("selected_method", pd.Series(dtype=str)).tolist()
        )
        rows.append(
            {
                "requested_policy": policy,
                "selected_methods": selected_methods,
                **evaluation.summary,
            }
        )
    return pd.DataFrame(rows), evaluations


def _default_output_dir(network, command):
    return Path("reports") / "networks" / network.name / command


def _make_cache(path, simulator_dir):
    return SimulationCache(path, simulator_dir=simulator_dir)


def _print_atlas_workload(network):
    rows = []
    for index, operation in enumerate(network.operations, start=1):
        rows.append(
            {
                "index": index,
                "operation": operation.operation_id,
                "type": operation.operation_type,
                "input": operation.input_tensor_id,
                "output": operation.output_tensor_id,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def _print_compiled_network(network):
    rows = []
    for index, layer in enumerate(network.layers, start=1):
        rows.append(
            {
                "index": index,
                "layer": layer.name,
                "op": "linear",
                "M": layer.batch_size,
                "K": layer.in_features,
                "N": layer.out_features,
                "intermediate_bytes": (
                    layer.batch_size * layer.out_features
                    if index < len(network.layers)
                    else 0
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def _network_calibration_path(network, search_space):
    fingerprint = network.fingerprint(search_space)
    return Path("cfg/calibration/networks") / f"{network.name}-{fingerprint}.json"


def _calibration_metadata(network, search_space, cost_averages):
    return {
        "schema_version": 1,
        "network": network.name,
        "network_fingerprint": network.fingerprint(),
        "search_space_fingerprint": canonical_fingerprint(search_space),
        "calibration_fingerprint": network.fingerprint(search_space),
        "cost_averages": cost_averages,
    }


def _load_network_calibration(
    path, network, search_space, intermediate_policy=None
):
    calibration = _load_json(path)
    expected = _calibration_metadata(network, search_space, {})
    for key in (
        "network_fingerprint",
        "search_space_fingerprint",
        "calibration_fingerprint",
    ):
        if calibration.get(key) != expected[key]:
            raise ValueError(
                f"Calibration {path} has a mismatched {key}: "
                f"expected {expected[key]}, got {calibration.get(key)!r}"
            )
    cost_averages = calibration.get("cost_averages")
    if not isinstance(cost_averages, dict):
        raise ValueError(f"Calibration {path} does not contain cost_averages")
    expected_identity = calibration_identity(
        search_space,
        network.to_workload_sequence(),
        intermediate_policy or network.intermediate_policy,
    )
    if cost_averages.get("_calibration_identity") != expected_identity:
        raise ValueError(f"Calibration {path} was produced by a stale model")
    return validate_calibration(cost_averages)


def write_network_comparison(comparison, evaluations, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "policy_comparison.csv", index=False)

    first = evaluations[0]
    summary = {
        "network": first.summary["network"],
        "network_fingerprint": first.summary["network_fingerprint"],
        "policy_count": len(evaluations),
        "calibrated": first.summary["calibrated"],
        "policies": [evaluation.summary for evaluation in evaluations],
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    report_columns = [
        "requested_policy",
        "selected_methods",
        "latency_ns",
        "total_energy_pj",
        "objective",
    ]
    report = [
        f"# Memory Policy Comparison: {summary['network']}",
        "",
        f"- Network fingerprint: `{summary['network_fingerprint']}`",
        f"- Calibrated: `{summary['calibrated']}`",
        "",
        "```text",
        comparison[report_columns].to_string(index=False),
        "```",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    for evaluation in evaluations:
        write_network_evaluation(
            evaluation,
            output_dir / evaluation.summary["requested_policy"],
        )


def main():
    parser = argparse.ArgumentParser(
        description="Validate and evaluate sequential linear neural networks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--network", type=Path, required=True)

    for command in ("evaluate", "compare-memory"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--network", type=Path, required=True)
        command_parser.add_argument("--architecture", type=Path, required=True)
        command_parser.add_argument(
            "--cache-file",
            type=Path,
            default=Path("cfg/static_cache/static_cache.csv"),
        )
        command_parser.add_argument("--cost-profile", default="t1")
        command_parser.add_argument("--calibration", type=Path)
        command_parser.add_argument(
            "--search-space",
            type=Path,
            default=Path("cfg/parameters/input.json"),
        )
        command_parser.add_argument("--output-dir", type=Path)
        if command == "evaluate":
            command_parser.add_argument(
                "--memory-policy", choices=INTERMEDIATE_POLICIES
            )
            command_parser.add_argument(
                "--evaluation-profile",
                type=Path,
                help="ATLAS graph evaluation profile JSON (default: legacy profile)",
            )

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--network", type=Path, required=True)
    calibrate_parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("cfg/parameters/input.json"),
    )
    calibrate_parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("cfg/static_cache/static_cache.csv"),
    )
    calibrate_parser.add_argument("--samples", type=int, default=10)
    calibrate_parser.add_argument("--run-name", default="network_calibration")

    args = parser.parse_args()
    network = load_workload(args.network)
    if args.command == "validate":
        if isinstance(network, AtlasGraph):
            print(
                f"ATLAS graph '{network.name}' is valid: "
                f"{len(network.operations)} operation(s), dtype={network.dtype}"
            )
            _print_atlas_workload(network)
        else:
            print(
                f"Network '{network.name}' is valid: {len(network.layers)} linear layer(s), "
                f"dtype={network.dtype}, policy={network.intermediate_policy}"
            )
            _print_compiled_network(network)
        return

    profile = None
    evaluation_profile_path = getattr(args, "evaluation_profile", None)
    if evaluation_profile_path is not None:
        if not isinstance(network, AtlasGraph):
            raise ValueError(
                "--evaluation-profile is only supported for ATLAS graph evaluation"
            )
        profile = load_evaluation_profile(evaluation_profile_path)

    if isinstance(network, AtlasGraph) and args.command == "calibrate":
        raise ValueError("Calibration is not supported for ATLAS graph evaluation")

    if isinstance(network, AtlasGraph) and args.command == "compare-memory":
        raise ValueError(
            "Memory-policy comparison is not supported for ATLAS graph evaluation"
        )

    if args.command == "calibrate":
        if args.samples <= 0:
            raise ValueError("--samples must be a positive integer")
        search_space = _load_json(args.search_space)
        calibration_path = _network_calibration_path(network, search_space)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        if calibration_path.exists():
            try:
                _load_network_calibration(calibration_path, network, search_space)
            except ValueError:
                calibration_path.unlink()
                print("Existing network calibration is stale; regenerating it")
            else:
                print(f"Calibration: {calibration_path}")
                return
        cache = _make_cache(args.cache_file, args.run_name)
        cost_averages = get_calib_cost_avg(
            calibration_iterations=args.samples,
            config_path=str(args.search_space),
            cache=cache,
            calibration_file_path=str(calibration_path),
            workload_sequence=network.to_workload_sequence(),
            intermediate_policy=network.intermediate_policy,
        )
        with calibration_path.open("w") as file:
            json.dump(
                _calibration_metadata(network, search_space, cost_averages),
                file,
                indent=2,
            )
        cache.dump_cache()
        print(f"Calibration: {calibration_path}")
        return

    architecture = _load_json(args.architecture)
    search_space = _load_json(args.search_space)
    if isinstance(network, AtlasGraph):
        calibration_policy = None
    else:
        calibration_policy = (
            args.memory_policy or network.intermediate_policy
            if args.command == "evaluate"
            else network.intermediate_policy
        )
    calibration = (
        _load_network_calibration(
            args.calibration,
            network,
            search_space,
            calibration_policy,
        )
        if args.calibration and calibration_policy is not None
        else None
    )
    output_dir = args.output_dir or _default_output_dir(network, args.command)
    cache = _make_cache(args.cache_file, str(output_dir / "simulation"))

    if args.command == "evaluate":
        evaluation = evaluate_network(
            network,
            architecture,
            cache,
            intermediate_policy=args.memory_policy,
            cost_profile=args.cost_profile,
            calibration=calibration,
            profile=profile,
        )
        write_network_evaluation(evaluation, output_dir)
        cache.dump_cache()
        print(json.dumps(evaluation.summary, indent=2))
        print(f"Results: {output_dir}")
        return


    comparison, evaluations = compare_memory_policies(
        network,
        architecture,
        cache,
        cost_profile=args.cost_profile,
        calibration=calibration,
    )
    write_network_comparison(comparison, evaluations, output_dir)
    cache.dump_cache()
    print(
        comparison[
            [
                "requested_policy",
                "selected_methods",
                "latency_ns",
                "total_energy_pj",
            ]
        ].to_string(index=False)
    )
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
