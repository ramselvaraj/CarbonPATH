import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chiplet.n_utils import get_area_power, get_sram_area_energy
from main import parse_workload_entry, simulate_latency_energy
from system.utils.ChipletSystem import ChipletSystem
from system.utils.SimulationCache import SimulationCache


CACHE_COLUMNS = [
    "core_size",
    "data_flow",
    "bandwidth",
    "buffer_size",
    "M",
    "K",
    "N",
    "latency",
]
VALIDATION_POLICIES = (
    "cold_dram",
    "ideal_on_chip",
    "local_sram",
    "direct_forward",
)
SRAM_KIB = 256
SRAM_BYTES = SRAM_KIB * 1024


def build_validation_architecture():
    logic_area, power = get_area_power("64x64", "7")
    sram_area, _ = get_sram_area_energy(SRAM_KIB, "7")
    return {
        "Chiplet_1": {
            "tech_node": "7",
            "sys_array_size": "64x64",
            "sram_buf": SRAM_KIB,
            "area": logic_area + sram_area,
            "power": power,
        },
        "pkg": {
            "HI_pkg_type": "2d",
            "inter_pkg_conn": "2d_na",
            "protocol_3d": "na",
            "protocol_2.5d": "na",
            "mem_pkg_conn": {"mem_type": "ddr5", "Chiplet_1": 2},
        },
        "WL_mapping": {
            "mapping": {
                "assign_workload_in_ascending_order": 1,
                "chiplet_data_sharing_enabled": 0,
                "dataflow": ["ws"],
                "if_splitting_k": 0,
                "merge_tiles": 0,
                "static_tiling": 0,
            }
        },
    }


def build_remote_validation_architecture(
    connection_type="2.5d_emib", protocol="ucie_adv"
):
    architecture = build_validation_architecture()
    architecture["Chiplet_2"] = architecture["Chiplet_1"].copy()
    architecture["pkg"] = {
        "HI_pkg_type": "2.5d",
        "inter_pkg_conn": [
            {
                "from": "Chiplet_1",
                "to": "Chiplet_2",
                "connection_type": connection_type,
                "loc": "2.5d_chiplet",
            }
        ],
        "protocol_3d": "na",
        "protocol_2.5d": protocol,
        "mem_pkg_conn": {
            "mem_type": "ddr5",
            "Chiplet_1": 1,
            "Chiplet_2": 1,
        },
    }
    architecture["WL_mapping"]["mapping"]["if_splitting_k"] = 1
    return architecture


def build_validation_cases():
    definitions = {
        "fits": {
            "name": "capacity_fits_exactly",
            "gemms": [
                {"name": "producer", "shape": [512, 64, 512]},
                {"name": "consumer", "shape": [512, 512, 64]},
            ],
        },
        "overflows": {
            "name": "capacity_overflows_by_512_bytes",
            "gemms": [
                {"name": "producer", "shape": [512, 64, 513]},
                {"name": "consumer", "shape": [512, 513, 64]},
            ],
        },
    }
    cases = {}
    for name, definition in definitions.items():
        sequence = parse_workload_entry(name, definition)
        producer_m, _, producer_n = sequence["gemms"][0]["shape"]
        cases[name] = {
            "sequence": sequence,
            "intermediate_bytes": producer_m * producer_n,
        }
    return cases


def build_two_gemm_validation_case():
    return parse_workload_entry(
        7,
        {
            "name": "two_gemm_demo",
            "gemms": [
                {"name": "projection", "shape": [128, 256, 512]},
                {"name": "reduction", "shape": [128, 512, 64]},
            ],
        },
    )


def build_mixed_boundary_validation_case():
    return parse_workload_entry(
        "mixed_boundary",
        {
            "name": "mixed_local_and_remote_boundaries",
            "gemms": [
                {"name": "wide_projection", "shape": [32, 32, 128]},
                {"name": "narrow_projection", "shape": [32, 128, 32]},
                {"name": "local_output", "shape": [32, 32, 32]},
            ],
        },
    )


def build_policy_validation_architectures():
    architectures = {
        "2.5d_rdl_ucie": build_remote_validation_architecture(
            connection_type="2.5d_rdl", protocol="ucie_std"
        ),
        "2.5d_emib_bow": build_remote_validation_architecture(protocol="bow"),
        "2.5d_active_ucie_adv": build_remote_validation_architecture(
            connection_type="2.5d_active", protocol="ucie_adv"
        ),
    }
    three_d = build_remote_validation_architecture()
    three_d["pkg"].update(
        {
            "HI_pkg_type": "3d",
            "inter_pkg_conn": [
                {
                    "from": "Chiplet_1",
                    "to": "Chiplet_2",
                    "connection_type": "3d_hyb_bond",
                    "loc": "stack0_base",
                },
                {
                    "from": "Chiplet_2",
                    "to": "na",
                    "connection_type": "3d_hyb_bond",
                    "loc": "stack0_top",
                },
            ],
            "protocol_3d": "ucie_3d",
            "protocol_2.5d": "na",
        }
    )
    architectures["3d_hybrid_bond"] = three_d
    for architecture in architectures.values():
        architecture["WL_mapping"]["mapping"]["if_splitting_k"] = 0
    return architectures


def _expected_selected_method(case_name, policy):
    if policy == "cold_dram":
        return "cold_dram"
    if policy == "ideal_on_chip":
        return "ideal_on_chip"
    if case_name == "fits":
        return "local_sram"
    return "cold_dram"


def _expected_byte_placement(selected_method, intermediate_bytes):
    if selected_method == "cold_dram":
        return {
            "retained": 0,
            "forwarded": 0,
            "spilled": intermediate_bytes,
            "dram_traffic": 2 * intermediate_bytes,
        }
    return {
        "retained": intermediate_bytes,
        "forwarded": 0,
        "spilled": 0,
        "dram_traffic": 0,
    }


def _expected_forwarding_metrics(architecture):
    """Calculate forwarding metrics independently from d2d_bw_calc()."""
    connection = architecture["pkg"]["inter_pkg_conn"][0]
    connection_type = connection["connection_type"]
    is_3d = connection_type.startswith("3d_")
    protocol = (
        architecture["pkg"]["protocol_3d"]
        if is_3d
        else architecture["pkg"]["protocol_2.5d"]
    )
    with (REPO_ROOT / "cfg/parameters/d2d_input.json").open() as file:
        d2d_data = json.load(file)
    with (REPO_ROOT / "cfg/parameters/energy_eff.json").open() as file:
        energy_data = json.load(file)

    pitch_mm = d2d_data["D2D_pitch_pkg"][connection_type] / 1000.0
    sram_area, _ = get_sram_area_energy(SRAM_KIB, "7")
    area = architecture["Chiplet_1"]["area"] - sram_area
    connections = area / pitch_mm**2 if is_3d else math.sqrt(area) / pitch_mm
    bandwidth_gbps = (
        connections
        * d2d_data["D2D_RATES_BY_NODE"][protocol]["7"]
        * d2d_data["eff_protocol"][protocol]
        / 8
    )
    forwarded_bytes = 2048
    return (
        forwarded_bytes / bandwidth_gbps,
        forwarded_bytes * 8 * energy_data["Die2Die_pj_per_bit"][protocol],
    )


def _error_percent(actual, expected):
    denominator = abs(expected) if expected else 1
    return abs(actual - expected) / denominator * 100


def run_capacity_validation(cache, tolerance=1e-9):
    architecture = build_validation_architecture()
    reference_system = ChipletSystem(arch_dict=architecture)
    core = reference_system.core_dict[0]
    dram_bandwidth_bytes_per_ns = core.dram_bandwidth * core.frequency / 1e9
    system_power_w = architecture["Chiplet_1"]["power"]
    rows = []

    for case_name, case in build_validation_cases().items():
        measurements = {}
        for policy in VALIDATION_POLICIES:
            measurements[policy] = simulate_latency_energy(
                cache,
                architecture,
                case["sequence"],
                intermediate_policy=policy,
            )

        cold_latency, cold_communication, cold_sram, _, _ = measurements[
            "cold_dram"
        ]
        intermediate_bytes = case["intermediate_bytes"]
        boundary_latency = 2 * intermediate_bytes / dram_bandwidth_bytes_per_ns
        boundary_communication = (
            2 * intermediate_bytes * 8 * core.dram_energy_scale
        )

        for policy in VALIDATION_POLICIES:
            latency, communication, sram, _, boundary_plans = measurements[policy]
            plan = boundary_plans[0]
            expected_method = _expected_selected_method(case_name, policy)
            expected_placement = _expected_byte_placement(
                expected_method, intermediate_bytes
            )
            removes_boundary_dram = expected_method != "cold_dram"
            expected_latency = cold_latency - (
                boundary_latency if removes_boundary_dram else 0
            )
            expected_communication = cold_communication - (
                boundary_communication if removes_boundary_dram else 0
            )
            expected_total_energy = (
                expected_communication
                + cold_sram
                + system_power_w * expected_latency * 1000
            )
            actual_total_energy = (
                communication + sram + system_power_w * latency * 1000
            )
            latency_error = _error_percent(latency, expected_latency)
            communication_error = _error_percent(
                communication, expected_communication
            )
            total_energy_error = _error_percent(
                actual_total_energy, expected_total_energy
            )
            bytes_conserved = (
                plan.retained_bytes
                + plan.forwarded_bytes
                + plan.dram_spilled_bytes
                == intermediate_bytes
            )
            fallback_expected = (
                case_name == "overflows"
                and policy in ("local_sram", "direct_forward")
            )
            fallback_valid = (
                "capacity" in plan.fallback_reason.lower()
                if fallback_expected
                else not plan.fallback_reason
            )
            placement_valid = (
                plan.retained_bytes == expected_placement["retained"]
                and plan.forwarded_bytes == expected_placement["forwarded"]
                and plan.dram_spilled_bytes == expected_placement["spilled"]
                and plan.dram_traffic_bytes == expected_placement["dram_traffic"]
            )
            validation_passed = (
                plan.selected_method == expected_method
                and bytes_conserved
                and placement_valid
                and fallback_valid
                and math.isclose(latency, expected_latency, rel_tol=tolerance, abs_tol=tolerance)
                and math.isclose(
                    communication,
                    expected_communication,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
                and math.isclose(
                    actual_total_energy,
                    expected_total_energy,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
            )
            rows.append(
                {
                    "case": case_name,
                    "requested_policy": policy,
                    "expected_selected_method": expected_method,
                    "actual_selected_method": plan.selected_method,
                    "intermediate_bytes": intermediate_bytes,
                    "sram_capacity_bytes": SRAM_BYTES,
                    "expected_retained_bytes": expected_placement["retained"],
                    "retained_bytes": plan.retained_bytes,
                    "expected_forwarded_bytes": expected_placement["forwarded"],
                    "forwarded_bytes": plan.forwarded_bytes,
                    "expected_dram_spilled_bytes": expected_placement["spilled"],
                    "dram_spilled_bytes": plan.dram_spilled_bytes,
                    "expected_dram_traffic_bytes": expected_placement["dram_traffic"],
                    "dram_traffic_bytes": plan.dram_traffic_bytes,
                    "expected_boundary_dram_latency_ns": boundary_latency,
                    "expected_boundary_dram_energy_pj": boundary_communication,
                    "expected_latency_ns": expected_latency,
                    "actual_latency_ns": latency,
                    "latency_error_percent": latency_error,
                    "expected_communication_energy_pj": expected_communication,
                    "actual_communication_energy_pj": communication,
                    "communication_energy_error_percent": communication_error,
                    "expected_total_energy_pj": expected_total_energy,
                    "actual_total_energy_pj": actual_total_energy,
                    "total_energy_error_percent": total_energy_error,
                    "fallback_reason": plan.fallback_reason,
                    "validation_passed": validation_passed,
                }
            )
    return pd.DataFrame(rows)


def run_two_gemm_mapping_validation(cache, tolerance=1e-9):
    workload = build_two_gemm_validation_case()
    architectures = {
        "same_core": build_validation_architecture(),
        "remote_core": build_remote_validation_architecture(),
    }
    expected_placement = {
        "same_core": {"retained": 65536, "forwarded": 0, "route": ()},
        "remote_core": {"retained": 32768, "forwarded": 32768, "route": (0, 1)},
    }
    rows = []

    for case_name, architecture in architectures.items():
        cold = simulate_latency_energy(
            cache, architecture, workload, intermediate_policy="cold_dram"
        )
        ideal = simulate_latency_energy(
            cache, architecture, workload, intermediate_policy="ideal_on_chip"
        )
        direct = simulate_latency_energy(
            cache, architecture, workload, intermediate_policy="direct_forward"
        )
        default = simulate_latency_energy(cache, architecture, workload)

        direct_plan = direct[4][0]
        placement = expected_placement[case_name]
        boundary_dram_latency = cold[0] - ideal[0]
        boundary_dram_energy = cold[1] - ideal[1]
        expected_latency = cold[0] - boundary_dram_latency + direct_plan.latency_ns
        expected_communication = (
            cold[1] - boundary_dram_energy + direct_plan.energy_pj
        )
        default_matches_direct = (
            all(
                math.isclose(
                    default[index],
                    direct[index],
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
                for index in range(3)
            )
            and default[4] == direct[4]
        )
        routes = tuple(direct_plan.routes)
        placement_valid = (
            direct_plan.intermediate_bytes == 65536
            and direct_plan.retained_bytes == placement["retained"]
            and direct_plan.forwarded_bytes == placement["forwarded"]
            and direct_plan.dram_spilled_bytes == 0
            and direct_plan.dram_traffic_bytes == 0
            and routes == (() if not placement["route"] else (placement["route"],))
        )
        validation_passed = (
            direct_plan.selected_method
            == ("local_sram" if case_name == "same_core" else "direct_forward")
            and placement_valid
            and not direct_plan.fallback_reason
            and default_matches_direct
            and math.isclose(
                direct[0], expected_latency, rel_tol=tolerance, abs_tol=tolerance
            )
            and math.isclose(
                direct[1],
                expected_communication,
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
        )
        rows.append(
            {
                "case": case_name,
                "chiplet_count": len(
                    [name for name in architecture if name.startswith("Chiplet_")]
                ),
                "split_k": architecture["WL_mapping"]["mapping"]["if_splitting_k"],
                "selected_method": direct_plan.selected_method,
                "intermediate_bytes": direct_plan.intermediate_bytes,
                "retained_bytes": direct_plan.retained_bytes,
                "forwarded_bytes": direct_plan.forwarded_bytes,
                "dram_spilled_bytes": direct_plan.dram_spilled_bytes,
                "producer_cores": ",".join(map(str, direct_plan.producer_cores)),
                "consumer_cores": ",".join(map(str, direct_plan.consumer_cores)),
                "routes": ";".join(
                    "->".join(map(str, route)) for route in direct_plan.routes
                ),
                "cold_latency_ns": cold[0],
                "boundary_dram_latency_ns": boundary_dram_latency,
                "forwarding_latency_ns": direct_plan.latency_ns,
                "expected_direct_latency_ns": expected_latency,
                "actual_direct_latency_ns": direct[0],
                "cold_communication_energy_pj": cold[1],
                "boundary_dram_energy_pj": boundary_dram_energy,
                "forwarding_energy_pj": direct_plan.energy_pj,
                "expected_direct_communication_energy_pj": expected_communication,
                "actual_direct_communication_energy_pj": direct[1],
                "default_matches_direct": default_matches_direct,
                "fallback_reason": direct_plan.fallback_reason,
                "validation_passed": validation_passed,
            }
        )

    return pd.DataFrame(rows)


def run_policy_architecture_validation(cache, tolerance=1e-9):
    workload = build_mixed_boundary_validation_case()
    policies = ("cold_dram", "local_sram", "direct_forward")
    expected_boundaries = {
        "cold_dram": (
            ("cold_dram", 0, 0, 4096, 8192),
            ("cold_dram", 0, 0, 1024, 2048),
        ),
        "local_sram": (
            ("cold_dram", 0, 0, 4096, 8192),
            ("local_sram", 1024, 0, 0, 0),
        ),
        "direct_forward": (
            ("direct_forward", 2048, 2048, 0, 0),
            ("local_sram", 1024, 0, 0, 0),
        ),
    }
    rows = []

    for architecture_name, architecture in build_policy_validation_architectures().items():
        measurements = {
            policy: simulate_latency_energy(
                cache, architecture, workload, intermediate_policy=policy
            )
            for policy in policies
        }
        default = simulate_latency_energy(cache, architecture, workload)
        direct = measurements["direct_forward"]
        default_matches_direct = (
            all(
                math.isclose(
                    default[index],
                    direct[index],
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
                for index in range(3)
            )
            and default[4] == direct[4]
        )
        latency_values = [float(measurements[policy][0]) for policy in policies]
        communication_values = [float(measurements[policy][1]) for policy in policies]
        metrics_are_distinct = (
            all(
                not math.isclose(
                    latency_values[left],
                    latency_values[right],
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
                for left in range(len(policies))
                for right in range(left + 1, len(policies))
            )
            and all(
                not math.isclose(
                    communication_values[left],
                    communication_values[right],
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
                for left in range(len(policies))
                for right in range(left + 1, len(policies))
            )
        )
        system_power_w = sum(
            component["power"]
            for name, component in architecture.items()
            if name.startswith("Chiplet_")
        )
        connection = architecture["pkg"]["inter_pkg_conn"][0]
        protocol = (
            architecture["pkg"]["protocol_3d"]
            if connection["connection_type"].startswith("3d_")
            else architecture["pkg"]["protocol_2.5d"]
        )
        cold_latency, cold_communication, _, _, cold_plans = measurements["cold_dram"]
        expected_forwarding_latency, expected_forwarding_energy = (
            _expected_forwarding_metrics(architecture)
        )

        for policy in policies:
            latency, communication, sram, _, plans = measurements[policy]
            expected = expected_boundaries[policy]
            plans_match = len(plans) == len(expected) and all(
                (
                    plan.selected_method,
                    plan.retained_bytes,
                    plan.forwarded_bytes,
                    plan.dram_spilled_bytes,
                    plan.dram_traffic_bytes,
                )
                == expected_plan
                for plan, expected_plan in zip(plans, expected)
            )
            routes_valid = (
                plans[0].routes == ((0, 1),)
                if policy == "direct_forward"
                else not plans[0].routes
            )
            fallback_valid = (
                "mappings differ" in plans[0].fallback_reason
                if policy == "local_sram"
                else not plans[0].fallback_reason
            )
            if policy == "cold_dram":
                expected_latency = cold_latency
                expected_communication = cold_communication
            elif policy == "local_sram":
                expected_latency = cold_latency - cold_plans[1].latency_ns
                expected_communication = cold_communication - cold_plans[1].energy_pj
            else:
                expected_latency = (
                    cold_latency
                    - sum(plan.latency_ns for plan in cold_plans)
                    + expected_forwarding_latency
                )
                expected_communication = (
                    cold_communication
                    - sum(plan.energy_pj for plan in cold_plans)
                    + expected_forwarding_energy
                )
            forwarding_oracle_matches = policy != "direct_forward" or (
                math.isclose(
                    plans[0].latency_ns,
                    expected_forwarding_latency,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
                and math.isclose(
                    plans[0].energy_pj,
                    expected_forwarding_energy,
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                )
            )
            accounting_matches = math.isclose(
                latency,
                expected_latency,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ) and math.isclose(
                communication,
                expected_communication,
                rel_tol=tolerance,
                abs_tol=tolerance,
            )
            row = {
                    "architecture": architecture_name,
                    "connection_type": connection["connection_type"],
                    "protocol": protocol,
                    "policy": policy,
                    "boundary_methods": ";".join(
                        plan.selected_method for plan in plans
                    ),
                    "retained_bytes": sum(plan.retained_bytes for plan in plans),
                    "forwarded_bytes": sum(plan.forwarded_bytes for plan in plans),
                    "dram_spilled_bytes": sum(
                        plan.dram_spilled_bytes for plan in plans
                    ),
                    "dram_traffic_bytes": sum(
                        plan.dram_traffic_bytes for plan in plans
                    ),
                    "forwarding_latency_ns": sum(
                        plan.latency_ns
                        for plan in plans
                        if plan.selected_method == "direct_forward"
                    ),
                    "forwarding_energy_pj": sum(
                        plan.energy_pj
                        for plan in plans
                        if plan.selected_method == "direct_forward"
                    ),
                    "expected_latency_ns": expected_latency,
                    "latency_ns": latency,
                    "expected_communication_energy_pj": expected_communication,
                    "communication_energy_pj": communication,
                    "sram_energy_pj": sram,
                    "total_energy_pj": (
                        communication + sram + system_power_w * latency * 1000
                    ),
                    "default_matches_direct": default_matches_direct,
                    "all_policy_metrics_distinct": metrics_are_distinct,
                    "accounting_matches": accounting_matches,
                    "forwarding_oracle_matches": forwarding_oracle_matches,
                    "validation_passed": (
                        plans_match
                        and routes_valid
                        and fallback_valid
                        and default_matches_direct
                        and metrics_are_distinct
                        and accounting_matches
                        and forwarding_oracle_matches
                    ),
                }
            for plan in plans:
                prefix = f"boundary_{plan.boundary_index}"
                row.update(
                    {
                        f"{prefix}_selected_method": plan.selected_method,
                        f"{prefix}_intermediate_bytes": plan.intermediate_bytes,
                        f"{prefix}_retained_bytes": plan.retained_bytes,
                        f"{prefix}_forwarded_bytes": plan.forwarded_bytes,
                        f"{prefix}_dram_spilled_bytes": plan.dram_spilled_bytes,
                        f"{prefix}_dram_traffic_bytes": plan.dram_traffic_bytes,
                        f"{prefix}_latency_ns": plan.latency_ns,
                        f"{prefix}_energy_pj": plan.energy_pj,
                        f"{prefix}_routes": ";".join(
                            "->".join(map(str, route)) for route in plan.routes
                        ),
                    }
                )
            rows.append(row)

    return pd.DataFrame(rows)


def plot_results(results, output_path):
    labels = [
        f"{row.case}\n{row.requested_policy.replace('_', ' ')}"
        for row in results.itertuples()
    ]
    x = np.arange(len(results))
    width = 0.38

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), facecolor="#f5f1e8")
    fig.subplots_adjust(hspace=0.55, wspace=0.3, top=0.84)

    latency_ax, energy_ax, placement_ax, summary_ax = axes.flat
    latency_ax.bar(
        x - width / 2,
        results["expected_latency_ns"] / 1e3,
        width,
        color="#b8b2a6",
        label="Expected",
    )
    latency_ax.bar(
        x + width / 2,
        results["actual_latency_ns"] / 1e3,
        width,
        color="#0b6e75",
        label="Actual",
    )
    latency_ax.set_title("A. Expected vs Actual Latency", loc="left", weight="bold")
    latency_ax.set_ylabel("Sequence latency (us)")
    latency_ax.legend(frameon=False, ncols=2)

    energy_ax.bar(
        x - width / 2,
        results["expected_total_energy_pj"] / 1e6,
        width,
        color="#b8b2a6",
        label="Expected",
    )
    energy_ax.bar(
        x + width / 2,
        results["actual_total_energy_pj"] / 1e6,
        width,
        color="#8c4f66",
        label="Actual",
    )
    energy_ax.set_title("B. Expected vs Actual Energy", loc="left", weight="bold")
    energy_ax.set_ylabel("Sequence energy (uJ)")
    energy_ax.legend(frameon=False, ncols=2)

    policy_rows = results[results["requested_policy"].isin(["local_sram", "direct_forward"])]
    placement_x = np.arange(len(policy_rows))
    denominator = policy_rows["intermediate_bytes"]
    placement_ax.bar(
        placement_x,
        policy_rows["retained_bytes"] / denominator * 100,
        color="#25766c",
        label="Retained",
    )
    placement_ax.bar(
        placement_x,
        policy_rows["dram_spilled_bytes"] / denominator * 100,
        bottom=policy_rows["retained_bytes"] / denominator * 100,
        color="#9b4a3c",
        label="DRAM spilled",
    )
    placement_ax.set_title("C. Capacity Decision", loc="left", weight="bold")
    placement_ax.set_ylabel("Intermediate bytes (%)")
    placement_ax.set_ylim(0, 108)
    placement_ax.set_xticks(
        placement_x,
        [
            f"{row.case}\n{row.requested_policy.replace('_', ' ')}"
            for row in policy_rows.itertuples()
        ],
    )
    placement_ax.legend(frameon=False, ncols=2)

    fit_local = results[
        (results["case"] == "fits")
        & (results["requested_policy"] == "local_sram")
    ].iloc[0]
    overflow_local = results[
        (results["case"] == "overflows")
        & (results["requested_policy"] == "local_sram")
    ].iloc[0]
    summary_ax.axis("off")
    summary_ax.text(
        0,
        1,
        "VALIDATION SUMMARY\n\n"
        f"SRAM capacity       {SRAM_BYTES:,} B\n"
        f"Fit intermediate    {fit_local['intermediate_bytes']:,.0f} B\n"
        f"Fit selected        {fit_local['actual_selected_method']}\n\n"
        f"Overflow intermediate {overflow_local['intermediate_bytes']:,.0f} B\n"
        f"Overflow selected   {overflow_local['actual_selected_method']}\n"
        f"Fallback             {overflow_local['fallback_reason']}\n\n"
        f"Rows passed          {int(results['validation_passed'].sum())}/{len(results)}\n"
        f"Max latency error    {results['latency_error_percent'].max():.3e}%\n"
        f"Max energy error     {results['total_energy_error_percent'].max():.3e}%",
        va="top",
        family="monospace",
        fontsize=10.5,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "white", "edgecolor": "#d1ccc0"},
    )

    for axis in (latency_ax, energy_ax):
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    placement_ax.grid(axis="y", alpha=0.18)
    placement_ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Intermediate SRAM Capacity Validation",
        x=0.06,
        y=0.96,
        ha="left",
        fontsize=19,
        weight="bold",
        color="#17363a",
    )
    fig.text(
        0.06,
        0.915,
        "Controlled one-core experiment: exact fit versus 512-byte overflow",
        color="#4d5555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(results, output_path, figure_path, csv_path):
    fit_local = results[
        (results["case"] == "fits")
        & (results["requested_policy"] == "local_sram")
    ].iloc[0]
    overflow_local = results[
        (results["case"] == "overflows")
        & (results["requested_policy"] == "local_sram")
    ].iloc[0]
    relative_figure = figure_path.relative_to(output_path.parent)
    lines = [
        "# Intermediate SRAM Capacity Validation",
        "",
        "## Experiment",
        "",
        "A deterministic 64x64 single-core architecture with a nominal 256 KiB policy capacity isolates the configured capacity decision. The fit sequence produces exactly 256 KiB; the overflow sequence produces 256 KiB plus 512 bytes.",
        "",
        f"![Capacity validation]({relative_figure.as_posix()})",
        "",
        "## Results",
        "",
        "| Case | Requested | Expected selected | Actual selected | Expected latency | Actual latency | Passed |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in results.itertuples():
        lines.append(
            f"| {row.case} | `{row.requested_policy}` | `{row.expected_selected_method}` | "
            f"`{row.actual_selected_method}` | {row.expected_latency_ns / 1e3:.6f} us | "
            f"{row.actual_latency_ns / 1e3:.6f} us | {'yes' if row.validation_passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Capacity Outcome",
            "",
            f"The exact-fit local request retained {fit_local['retained_bytes']:,.0f} bytes and matched ideal-on-chip latency. The overflow local request selected cold DRAM, moved {overflow_local['dram_traffic_bytes']:,.0f} DRAM bytes, and matched the cold baseline.",
            "",
            f"Fallback reason: `{overflow_local['fallback_reason']}`.",
            "",
            f"Maximum latency error: `{results['latency_error_percent'].max():.3e}%`.",
            "",
            f"Maximum communication-energy error: `{results['communication_energy_error_percent'].max():.3e}%`.",
            "",
            "Expected sequence values use the measured cold run as the compute baseline and independently calculate the boundary DRAM delta. This validates CarbonPATH policy accounting, not absolute hardware timing or concurrent SRAM occupancy.",
            "",
            f"Raw data: `{csv_path.as_posix()}`.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_two_gemm_report(results, output_path, csv_path):
    lines = [
        "# Workload 7 Intermediate Mapping Validation",
        "",
        "## Experiment",
        "",
        "Workload 7 chains a 128x256x512 GEMM into a 128x512x64 GEMM, producing a 65,536-byte intermediate. A one-chiplet architecture validates same-core retention. A two-chiplet split-K architecture gives the completed producer output one reduction owner while the consumer spans both cores, forcing a routed remote transfer.",
        "",
        "## Results",
        "",
        "| Case | Chiplets | Split K | Selected | Retained | Forwarded | Route | Default matches | Passed |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in results.itertuples():
        lines.append(
            f"| {row.case} | {row.chiplet_count} | {row.split_k} | "
            f"`{row.selected_method}` | {row.retained_bytes:,} B | "
            f"{row.forwarded_bytes:,} B | `{row.routes or 'local'}` | "
            f"{'yes' if row.default_matches_direct else 'no'} | "
            f"{'yes' if row.validation_passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Accounting Check",
            "",
            "Expected direct-forward values remove the measured boundary DRAM delta between cold DRAM and ideal on-chip, then add the independently planned forwarding latency and energy.",
            "",
            "| Case | Cold latency | Removed DRAM | Added forwarding | Expected direct | Actual direct |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in results.itertuples():
        lines.append(
            f"| {row.case} | {row.cold_latency_ns:.6f} ns | "
            f"{row.boundary_dram_latency_ns:.6f} ns | "
            f"{row.forwarding_latency_ns:.6f} ns | "
            f"{row.expected_direct_latency_ns:.6f} ns | "
            f"{row.actual_direct_latency_ns:.6f} ns |"
        )
    lines.extend(
        [
            "",
            "The same-core case resolves `direct_forward` to `local_sram`, with all bytes retained and no transfer cost. The remote case retains half on the reduction-owner core and forwards half over the 0-to-1 route, with no DRAM spill.",
            "",
            f"Raw data: `{csv_path.as_posix()}`.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def plot_policy_architecture_results(results, output_path):
    architectures = list(results["architecture"].drop_duplicates())
    policies = ("cold_dram", "local_sram", "direct_forward")
    colors = {
        "cold_dram": "#9b4a3c",
        "local_sram": "#25766c",
        "direct_forward": "#315b8a",
    }
    labels = {
        "2.5d_rdl_ucie": "2.5D RDL\nUCIe",
        "2.5d_emib_bow": "2.5D EMIB\nBoW",
        "2.5d_active_ucie_adv": "2.5D active\nUCIe Adv",
        "3d_hybrid_bond": "3D hybrid\nbond",
    }
    x = np.arange(len(architectures))
    width = 0.24
    fig, (latency_ax, energy_ax) = plt.subplots(
        1, 2, figsize=(13, 5.4), facecolor="#f5f1e8"
    )

    for offset, policy in enumerate(policies):
        policy_rows = results.set_index(["architecture", "policy"])
        latency = [
            policy_rows.loc[(architecture, policy), "latency_ns"] / 1e3
            for architecture in architectures
        ]
        energy = [
            policy_rows.loc[(architecture, policy), "communication_energy_pj"]
            / 1e6
            for architecture in architectures
        ]
        position = x + (offset - 1) * width
        latency_ax.bar(
            position,
            latency,
            width,
            color=colors[policy],
            label=policy.replace("_", " "),
        )
        energy_ax.bar(position, energy, width, color=colors[policy])

    for axis in (latency_ax, energy_ax):
        axis.set_xticks(x, [labels[name] for name in architectures])
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    latency_ax.set_title("A. Sequence Latency", loc="left", weight="bold")
    latency_ax.set_ylabel("Latency (us)")
    latency_ax.legend(frameon=False, ncols=3, loc="upper right")
    energy_ax.set_title("B. Communication Energy", loc="left", weight="bold")
    energy_ax.set_ylabel("Communication energy (uJ)")
    fig.suptitle(
        "Intermediate Policy by Interconnect Architecture",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=18,
        weight="bold",
        color="#17363a",
    )
    fig.text(
        0.06,
        0.91,
        "Mixed sequence with one remote boundary and one same-core boundary",
        color="#4d5555",
    )
    fig.subplots_adjust(top=0.78, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_policy_architecture_report(results, output_path, figure_path, csv_path):
    relative_figure = figure_path.relative_to(output_path.parent)
    lines = [
        "# Intermediate Policy Architecture Matrix",
        "",
        "## Experiment",
        "",
        "The controlled three-GEMM sequence is `32x32x128 -> 32x128x32 -> 32x32x32`. Its first 4,096-byte boundary is split evenly between local retention and a remote consumer. Its second 1,024-byte boundary is entirely same-core. This mixed mapping makes the three policy totals legitimately distinct:",
        "",
        "- `cold_dram` spills both boundaries.",
        "- `local_sram` spills the remote boundary but retains the same-core boundary.",
        "- `direct_forward` forwards the remote half of the first boundary and retains all local bytes.",
        "",
        f"![Policy architecture matrix]({relative_figure.as_posix()})",
        "",
        "## Results",
        "",
        "| Architecture | Policy | Boundary methods | B1 latency | B1 energy | B2 latency | B2 energy | Retained | Forwarded | DRAM spilled | Latency | Communication energy | Passed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results.itertuples():
        lines.append(
            f"| {row.architecture} | `{row.policy}` | `{row.boundary_methods}` | "
            f"{row.boundary_1_latency_ns:.6f} ns | {row.boundary_1_energy_pj:.2f} pJ | "
            f"{row.boundary_2_latency_ns:.6f} ns | {row.boundary_2_energy_pj:.2f} pJ | "
            f"{row.retained_bytes:,} B | {row.forwarded_bytes:,} B | "
            f"{row.dram_spilled_bytes:,} B | {row.latency_ns / 1e3:.6f} us | "
            f"{row.communication_energy_pj / 1e6:.6f} uJ | "
            f"{'yes' if row.validation_passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Pattern",
            "",
            "Every architecture produces three distinct sequence latency and communication-energy totals. The 2.5D cold-DRAM and local-SRAM values remain identical because their compute cores and memory channels are held constant; the 3D stack adds its path energy to top-die DRAM accesses. Local SRAM consistently removes the second boundary's DRAM traffic. Direct forwarding additionally removes the first boundary's DRAM traffic, while its latency changes with the selected package route.",
            "",
            "The 2.5D routes trade much lower communication energy for route latency at this small transfer size. The 3D hybrid-bond route is both lower latency and lower communication energy than the local-SRAM policy's DRAM fallback.",
            "",
            f"Raw data: `{csv_path.as_posix()}`.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(
        description="Validate exact-fit and overflowing intermediate SRAM policies."
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Optional existing cache; omit it to force fresh SCALE-Sim execution",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as simulator_dir:
        if args.cache_file is None:
            cache_path = Path(simulator_dir) / "cache.csv"
            pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        else:
            cache_path = args.cache_file
        cache = SimulationCache(cache_path, simulator_dir=simulator_dir)
        results = run_capacity_validation(cache)
        mapping_results = run_two_gemm_mapping_validation(cache)
        policy_results = run_policy_architecture_validation(cache)

    csv_path = args.output_dir / "intermediate_capacity_validation.csv"
    figure_path = args.output_dir / "figures" / "intermediate_capacity_validation.png"
    report_path = args.output_dir / "intermediate_capacity_validation.md"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    plot_results(results, figure_path)
    write_report(results, report_path, figure_path, csv_path)
    mapping_csv_path = args.output_dir / "intermediate_workload_7_mapping_validation.csv"
    mapping_report_path = args.output_dir / "intermediate_workload_7_mapping_validation.md"
    mapping_results.to_csv(mapping_csv_path, index=False)
    write_two_gemm_report(mapping_results, mapping_report_path, mapping_csv_path)
    policy_csv_path = args.output_dir / "intermediate_policy_architecture_matrix.csv"
    policy_figure_path = (
        args.output_dir / "figures" / "intermediate_policy_architecture_matrix.png"
    )
    policy_report_path = args.output_dir / "intermediate_policy_architecture_matrix.md"
    policy_results.to_csv(policy_csv_path, index=False)
    plot_policy_architecture_results(policy_results, policy_figure_path)
    write_policy_architecture_report(
        policy_results, policy_report_path, policy_figure_path, policy_csv_path
    )

    display_columns = [
        "case",
        "requested_policy",
        "actual_selected_method",
        "expected_latency_ns",
        "actual_latency_ns",
        "validation_passed",
    ]
    print(results[display_columns].to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"Figure: {figure_path}")
    print(f"Report: {report_path}")
    print(f"Workload 7 CSV: {mapping_csv_path}")
    print(f"Workload 7 report: {mapping_report_path}")
    print(f"Policy matrix CSV: {policy_csv_path}")
    print(f"Policy matrix figure: {policy_figure_path}")
    print(f"Policy matrix report: {policy_report_path}")
    if not results["validation_passed"].all() or not mapping_results[
        "validation_passed"
    ].all() or not policy_results["validation_passed"].all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
