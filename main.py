from ast import arg
from dataclasses import dataclass
import hashlib
import json 
import os
import random
import pandas as pd
import time
import csv
import itertools
import copy
import sys
import math
import statistics
import types
from pathlib import Path

from chiplet.n_utils import get_area_power, read_json_input_params, calculate_system_metrics, write_solution_to_json, \
                            calculate_system_normalized_metrics,\
                            dump_results, calculate_memory_bandwidth,\
                            find_run_time, process_iteration_wide, process_arch_details_dump, \
                            flatten_dict, find_config_diff, get_sram_area_energy, recursive_split_updated, \
                            calc_HI_dimension, find_connections, update_inter_pkg_connections, opC_from_pj, total_opC,\
                            build_design_tables
from chiplet.n_disagg import ChipletGenerator, PackageGenerator, WLMappingGenerator
from chiplet.carbon_model.ECO_chip import find_carbon

from system.utils.GEMMWorkload import GEMMWorkload
from system.utils.Scheduler import CHIP2CHIP_TRANSFER, Scheduler
from system.utils.ChipletSystem import ChipletSystem
from system.utils.SimulationCache import SimulationCache
from system.utils.IntermediateMemoryPolicy import (
    INTERMEDIATE_POLICIES,
    build_boundary_mapping,
    plan_boundary,
)
from system.utils.ArchitectureIdentity import architecture_fingerprint
from system.utils.AtlasGraphAdapter import gemm_sequence_to_atlas_graph, load_atlas_graph
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.NonGemmEstimator import (
    FpgaReluEvaluator,
    LegacyReluInputAdapter,
    PlaceholderHlsReluEvaluator,
    PlaceholderHlsReluInputAdapter,
    PlaceholderSoftmaxEvaluator,
    PlaceholderSoftmaxInputAdapter,
)
from system.utils.OperationEvaluator import (
    EvaluationContext,
    EvaluatorRegistry,
    OperationEstimate,
    OperationEvaluator,
    OperationInputAdapter,
)
from system.utils.OperationPlacement import build_placement_policy
from system.utils.TensorMovement import TensorResidency, build_movement_service
from system.utils.TransferEstimator import build_transfer_cost_model
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation
from system.utils.AtlasObjective import (
    AtlasDesignPoint,
    RawWeightedSumObjective,
    build_atlas_objective,
)
from system.utils.AtlasAnnealingMoves import (
    candidate_profile_paths,
    mutate_atlas_design_point,
    sequential_gemm_search_space,
    validate_atlas_architecture,
)
from system.utils.Calibration import summarize_design_points
from system.utils.GenericAnnealer import anneal
from system.utils.PlaceholderPolicies import register_placeholder_policies
from config import calibration_mode, print_info, fast_test, latency_en, sram_selection_mode


# Register placeholder movement/transfer policies at import so profile resolution
# sees them. Evaluator registration happens in build_default_evaluator_registry.
# DELETE-ME with system/utils/PlaceholderPolicies.py.
register_placeholder_policies()


#########
#Freq config read
with open('cfg/parameters/freq_scale.json', 'r') as f:
    freq_config = json.load(f)

FREQUENCY = freq_config['BASE_FREQUENCY']
scaling_factors = {int(k): v for k, v in freq_config['freq_scaling_factors'].items()}
#########



######
## Workload 
with open("cfg/examples/workload.json") as f:
    workload = json.load(f)
WORKLOAD_CONFIGS = {int(k): v for k, v in workload.items()}
CALIBRATION_MODEL_VERSION = 3
CLI_INITIAL_TEMP = 40
CLI_FREEZING_TEMP = 5e-4
CLI_MAX_MOVE_PER_TEMP_STEP = 5
CLI_COOLING_RATE = 0.3
######


def parse_workload_entry(workload_id, entry):
    """Normalize a legacy GEMM or an ordered, cold-memory GEMM sequence."""
    if isinstance(entry, list):
        sequence_name = f"workload_{workload_id}"
        raw_gemms = [{"name": "gemm_1", "shape": entry}]
    elif isinstance(entry, dict):
        sequence_name = entry.get("name", f"workload_{workload_id}")
        raw_gemms = entry.get("gemms")
        if not isinstance(raw_gemms, list) or len(raw_gemms) < 2:
            raise ValueError(f"Workload {workload_id} must define at least two GEMMs")
    else:
        raise ValueError(f"Workload {workload_id} must be a GEMM shape or sequence object")

    gemms = []
    names = set()
    for index, raw_gemm in enumerate(raw_gemms, start=1):
        if not isinstance(raw_gemm, dict):
            raise ValueError(f"GEMM {index} in workload {workload_id} must be an object")
        name = raw_gemm.get("name", f"gemm_{index}")
        shape = raw_gemm.get("shape")
        if not isinstance(name, str) or not name:
            raise ValueError(f"GEMM {index} in workload {workload_id} must have a name")
        if name in names:
            raise ValueError(f"GEMM names must be unique within workload {workload_id}")
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError(f"GEMM '{name}' must have three positive integer dimensions [M, K, N]")
        names.add(name)
        gemms.append({"name": name, "shape": tuple(shape)})

    for previous, current in zip(gemms, gemms[1:]):
        previous_m, _, previous_n = previous["shape"]
        current_m, current_k, _ = current["shape"]
        if current_m != previous_m or current_k != previous_n:
            raise ValueError(
                f"GEMM '{current['name']}' must consume '{previous['name']}' output: "
                f"expected M={previous_m}, K={previous_n}, got M={current_m}, K={current_k}"
            )

    return {"id": workload_id, "name": sequence_name, "gemms": gemms}


def _load_json_file(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _dump_best_profile(result_df, run_name):
    """Persist the best modular evaluation profile beside the best architecture."""
    best_profile = result_df.attrs.get("best_profile")
    if not best_profile:
        return
    rundir = os.path.join("cfg", "gen_arch", run_name)
    os.makedirs(rundir, exist_ok=True)
    path = os.path.join(rundir, "best_profile.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(best_profile, file, indent=2)
    print(f"[INFO] Best evaluation profile written to {path}")


def build_scheduler_system(workload, arch_dict):

    system = ChipletSystem(arch_dict=arch_dict, BASE_FREQUENCY=FREQUENCY)
    scheduler = Scheduler(workload, system.core_dict.values(), mapping_dict=arch_dict['WL_mapping']['mapping'])
    scheduler.static_workload_scheduling()

    return scheduler, system



def prepare_single_gemm(cache: SimulationCache, arch_dict: dict, shape):
    wl = GEMMWorkload(*shape)
    scheduler, system = build_scheduler_system(wl, arch_dict)
    cache.all_cores_simulation_with_cache(scheduler.systolic_arrays)
    return scheduler, system


def simulate_single_gemm(
    cache: SimulationCache,
    arch_dict: dict,
    shape,
    prepared=None,
    activation_from_dram=True,
    output_to_dram=True,
):
    scheduler, system = prepared or prepare_single_gemm(cache, arch_dict, shape)
    latency_ns, energy_pj = scheduler.system_modeling(
        system,
        activation_from_dram=activation_from_dram,
        output_to_dram=output_to_dram,
    )
    sram_energy_pj = scheduler._get_sram_energy()
    modeling = getattr(scheduler, "last_modeling_details", {})
    return {
        "latency_ns": latency_ns,
        "dram_interconnect_energy_pj": energy_pj,
        "sram_energy_pj": sram_energy_pj,
        "tile_mappings": tuple(collect_tile_mappings(scheduler)),
        "reduction_latency_ns": modeling.get("reduction_latency_ns", 0.0),
        "reduction_transfer_latency_ns": modeling.get(
            "reduction_transfer_latency_ns", 0.0
        ),
        "reduction_communication_energy_pj": modeling.get(
            "reduction_communication_energy_pj", 0.0
        ),
    }


def collect_tile_mappings(scheduler):
    mappings = []
    for core in getattr(scheduler, "systolic_arrays", ()):
        for tile in core.workloads:
            mappings.append(
                {
                    "core_id": core.id,
                    "m": tile.m,
                    "k": tile.k,
                    "n": tile.n,
                    "m_offset": tile.m_offset,
                    "k_offset": tile.k_offset,
                    "n_offset": tile.n_offset,
                }
            )
    return mappings


def simulate_latency_energy(
    cache: SimulationCache,
    arch_dict: dict,
    workload_sequence,
    dbg=False,
    intermediate_policy="direct_forward",
):
    if intermediate_policy not in INTERMEDIATE_POLICIES:
        raise ValueError(f"Unknown intermediate-memory policy: {intermediate_policy}")
    prepared_gemms = [
        prepare_single_gemm(cache, arch_dict, gemm["shape"])
        for gemm in workload_sequence["gemms"]
    ]
    boundary_plans = []
    for index, (producer, consumer) in enumerate(
        zip(prepared_gemms, prepared_gemms[1:]),
        start=1,
    ):
        mapping = build_boundary_mapping(
            producer[0], producer[1], consumer[0], consumer[1]
        )
        boundary_plans.append(
            plan_boundary(
                boundary_index=index,
                intermediate_bytes=mapping.intermediate_bytes,
                transfers=mapping.transfers,
                cores=mapping.cores,
                policy=intermediate_policy,
                mapping_valid=mapping.valid,
                mapping_error=mapping.error,
            )
        )

    gemm_metrics = []
    for index, (gemm, prepared) in enumerate(
        zip(workload_sequence["gemms"], prepared_gemms)
    ):
        incoming_boundary = boundary_plans[index - 1] if index > 0 else None
        outgoing_boundary = boundary_plans[index] if index < len(boundary_plans) else None
        metrics = simulate_single_gemm(
            cache,
            arch_dict,
            gemm["shape"],
            prepared=prepared,
            activation_from_dram=(incoming_boundary is None or incoming_boundary.uses_dram),
            output_to_dram=(outgoing_boundary is None or outgoing_boundary.uses_dram),
        )
        if incoming_boundary is not None and incoming_boundary.selected_method == "direct_forward":
            metrics["latency_ns"] += incoming_boundary.latency_ns
            metrics["dram_interconnect_energy_pj"] += incoming_boundary.energy_pj
        metrics["tile_mappings"] = collect_tile_mappings(prepared[0])
        gemm_metrics.append({"name": gemm["name"], "shape": gemm["shape"], **metrics})

    latency_ns = sum(metric["latency_ns"] for metric in gemm_metrics)
    energy_pj = sum(metric["dram_interconnect_energy_pj"] for metric in gemm_metrics)
    sram_energy_pj = sum(metric["sram_energy_pj"] for metric in gemm_metrics)
    return latency_ns, energy_pj, sram_energy_pj, gemm_metrics, boundary_plans


@dataclass(frozen=True)
class LegacyScaleSimGemmInput:
    operation_id: str
    m: int
    k: int
    n: int
    activation_from_dram: bool
    output_to_dram: bool


class LegacyScaleSimGemmInputAdapter(OperationInputAdapter):
    """Input adapter for the legacy SCALE-Sim GEMM path."""

    input_adapter_id = "legacy_scale_sim_gemm_input_v1"
    operation_type = "gemm"

    def build_input(self, operation, access_plan):
        if operation.operation_type != "gemm":
            raise UnsupportedEvaluation(
                f"LegacyScaleSimGemmInputAdapter cannot read "
                f"'{operation.operation_type}' operation "
                f"'{operation.operation_id}'"
            )
        if operation.gemm_shape is None:
            raise UnsupportedEvaluation(
                f"{operation.operation_id}: GEMM dimensions are missing"
            )
        m, k, n = operation.gemm_shape
        return LegacyScaleSimGemmInput(
            operation_id=operation.operation_id,
            m=m,
            k=k,
            n=n,
            activation_from_dram=access_plan.activation_from_dram,
            output_to_dram=access_plan.output_to_dram,
        )


class LegacyScaleSimGemmEvaluator(OperationEvaluator):
    """Operation evaluator wrapping the existing SCALE-Sim GEMM path.

    Dynamic energy is the GEMM's DRAM/interconnect plus SRAM energy. Compute
    latency is the full GEMM latency. The executor adds baseline architecture
    power over the elapsed latency and any tensor movement energy.
    """

    evaluator_id = "legacy_scale_sim_gemm_v1"
    operation_type = "gemm"
    input_adapter_id = "legacy_scale_sim_gemm_input_v1"

    @staticmethod
    def _configured_architecture(architecture, settings):
        if not settings:
            return architecture
        unknown = sorted(set(settings) - {"dataflow", "split_k", "assignment_order"})
        if unknown:
            raise UnsupportedEvaluation(
                "unsupported GEMM evaluator setting(s): " + ", ".join(unknown)
            )
        architecture = copy.deepcopy(architecture)
        mapping = architecture["WL_mapping"]["mapping"]
        if "dataflow" in settings:
            dataflow = settings["dataflow"]
            if dataflow not in ("ws", "os", "is"):
                raise UnsupportedEvaluation("GEMM dataflow must be one of: ws, os, is")
            mapping["dataflow"] = [dataflow]
        if "split_k" in settings:
            split_k = settings["split_k"]
            if not isinstance(split_k, bool):
                raise UnsupportedEvaluation("GEMM split_k must be boolean")
            mapping["if_splitting_k"] = int(split_k)
        if "assignment_order" in settings:
            assignment_order = settings["assignment_order"]
            if assignment_order not in ("ascending", "descending"):
                raise UnsupportedEvaluation(
                    "GEMM assignment_order must be ascending or descending"
                )
            mapping["assign_workload_in_ascending_order"] = int(
                assignment_order == "ascending"
            )
        return architecture

    def prepare(self, evaluator_input, placement, context):
        if placement.endpoint_ids != context.system.sa_endpoint_group.endpoint_ids:
            raise UnsupportedEvaluation(
                "the CarbonPATH GEMM evaluator requires placement on the complete SA group"
            )
        architecture = self._configured_architecture(
            context.architecture, context.evaluator_settings
        )
        return prepare_single_gemm(
            context.cache,
            architecture,
            (evaluator_input.m, evaluator_input.k, evaluator_input.n),
        )

    def evaluate(self, evaluator_input, placement, context) -> OperationEstimate:
        if placement.endpoint_ids != context.system.sa_endpoint_group.endpoint_ids:
            raise UnsupportedEvaluation(
                "the CarbonPATH GEMM evaluator requires placement on the complete SA group"
            )
        architecture = self._configured_architecture(
            context.architecture, context.evaluator_settings
        )
        result = simulate_single_gemm(
            context.cache,
            architecture,
            (evaluator_input.m, evaluator_input.k, evaluator_input.n),
            prepared=context.prepared_operation,
            activation_from_dram=evaluator_input.activation_from_dram,
            output_to_dram=evaluator_input.output_to_dram,
        )
        dynamic_energy_pj = (
            result["dram_interconnect_energy_pj"] + result["sram_energy_pj"]
        )
        return OperationEstimate(
            operation_id=evaluator_input.operation_id,
            evaluator_id=self.evaluator_id,
            compute_latency_ns=result["latency_ns"],
            dynamic_energy_pj=dynamic_energy_pj,
            tile_mappings=tuple(result.get("tile_mappings", ())),
            reduction_latency_ns=result.get("reduction_latency_ns", 0.0),
            reduction_communication_energy_pj=result.get(
                "reduction_communication_energy_pj", 0.0
            ),
        )


def build_default_evaluator_registry():
    registry = EvaluatorRegistry()
    registry.register(
        LegacyScaleSimGemmEvaluator(), LegacyScaleSimGemmInputAdapter()
    )
    registry.register(FpgaReluEvaluator(), LegacyReluInputAdapter())
    registry.register(
        PlaceholderHlsReluEvaluator(), PlaceholderHlsReluInputAdapter()
    )
    registry.register(
        PlaceholderSoftmaxEvaluator(), PlaceholderSoftmaxInputAdapter()
    )
    register_placeholder_policies(registry)
    return registry


@dataclass(frozen=True)
class TensorAccessPlan:
    """Type-neutral statement of where an operation's tensors come from and go.

    The first operation reads its activation from DRAM and the last writes its
    output to DRAM. Every other tensor is an intermediate that is resident or
    moved by the tensor movement service. Evaluator input adapters translate
    this plan into the facts their model needs.
    """

    activation_from_dram: bool
    output_to_dram: bool


@dataclass(frozen=True)
class AtlasOperationResult:
    index: int
    operation_id: str
    operation_type: str
    endpoint_ids: tuple
    endpoint_kind: str
    evaluator_id: str
    m: object
    k: object
    n: object
    element_count: int
    compute_latency_ns: float
    compute_energy_pj: float
    movement: object
    tile_mappings: tuple = ()
    reduction_latency_ns: float = 0.0
    reduction_communication_energy_pj: float = 0.0

    @property
    def endpoint_id(self):
        return self.endpoint_ids[0] if len(self.endpoint_ids) == 1 else None


@dataclass(frozen=True)
class AtlasEvaluation:
    graph: object
    profile: object
    results: tuple
    system: object

    @property
    def latency_ns(self):
        total = 0.0
        for result in self.results:
            total += result.compute_latency_ns
            if result.movement is not None:
                total += result.movement.charged_latency_ns
        return total

    @property
    def compute_energy_pj(self):
        return sum(result.compute_energy_pj for result in self.results)

    @property
    def movement_energy_pj(self):
        return sum(
            result.movement.charged_energy_pj
            for result in self.results
            if result.movement is not None
        )

    def total_energy_pj(self, system_power_w):
        return (
            system_power_w * self.latency_ns * 1000
            + self.compute_energy_pj
            + self.movement_energy_pj
        )


def evaluate_atlas_graph(
    cache: SimulationCache,
    arch_dict: dict,
    graph,
    profile=None,
    transfer_model=None,
    registry=None,
):
    """Evaluate a direct ATLAS graph on a chiplet architecture.

    The executor owns operation order, operation placement, tensor residency,
    tensor movement, and aggregation. Evaluators own only compute latency and
    dynamic energy, chosen by the evaluation profile per operation type.
    """
    profile = profile or load_evaluation_profile()
    transfer_model = (
        transfer_model
        if transfer_model is not None
        else arch_dict.get("transfer_model", {})
    )
    system = ChipletSystem(arch_dict=arch_dict, BASE_FREQUENCY=FREQUENCY)
    placement_policy = build_placement_policy(profile.placement_policy, system)
    movement_service = build_movement_service(
        profile.movement_policy, build_transfer_cost_model(profile.transfer_model)
    )
    registry = registry or build_default_evaluator_registry()

    operations = graph.operations
    planned = []
    for index, operation in enumerate(operations):
        placement = placement_policy.endpoint_for(operation.operation_type)
        access_plan = TensorAccessPlan(
            activation_from_dram=(index == 0),
            output_to_dram=(index == len(operations) - 1),
        )
        binding = registry.get(profile.evaluator_id_for(operation.operation_type))
        context = EvaluationContext(
            cache=cache,
            architecture=arch_dict,
            system=system,
            evaluator_settings=profile.evaluator_settings_for(
                operation.operation_type
            ),
            activation_from_dram=access_plan.activation_from_dram,
            output_to_dram=access_plan.output_to_dram,
        )
        prepared = None
        if getattr(movement_service, "requires_prepared_operations", False):
            prepared = binding.prepare(operation, access_plan, placement, context)
        planned.append(
            {
                "operation": operation,
                "placement": placement,
                "binding": binding,
                "context": context,
                "prepared": prepared,
            }
        )

    mapped_movements = [None] * len(planned)
    if getattr(movement_service, "requires_prepared_operations", False):
        for index in range(1, len(planned)):
            producer = planned[index - 1]
            consumer = planned[index]
            mapped_movements[index] = movement_service.move_between(
                boundary_index=index,
                producer_operation=producer["operation"],
                consumer_operation=consumer["operation"],
                producer_prepared=producer["prepared"],
                consumer_prepared=consumer["prepared"],
            )

    results = []
    residency = None

    for index, item in enumerate(planned):
        operation = item["operation"]
        placement = item["placement"]

        movement = mapped_movements[index]
        if residency is not None and not getattr(
            movement_service, "requires_prepared_operations", False
        ):
            movement = movement_service.move(
                tensor_id=residency.tensor_id,
                element_count=residency.element_count,
                source_endpoint=residency.endpoint_ids,
                destination_endpoint=placement.endpoint_ids,
                system=system,
                transfer_model=transfer_model,
            )

        incoming_uses_dram = movement is not None and getattr(
            movement, "uses_dram", False
        )
        outgoing = mapped_movements[index + 1] if index + 1 < len(planned) else None
        outgoing_uses_dram = outgoing is not None and getattr(
            outgoing, "uses_dram", False
        )
        access_plan = TensorAccessPlan(
            activation_from_dram=(index == 0 or incoming_uses_dram),
            output_to_dram=(index == len(operations) - 1 or outgoing_uses_dram),
        )
        context = item["context"]
        context.prepared_operation = item["prepared"]
        context.activation_from_dram = access_plan.activation_from_dram
        context.output_to_dram = access_plan.output_to_dram
        estimate = item["binding"].estimate(
            operation, access_plan, placement, context
        )

        shape = operation.gemm_shape if operation.operation_type == "gemm" else None
        results.append(
            AtlasOperationResult(
                index=index + 1,
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
                endpoint_ids=placement.endpoint_ids,
                endpoint_kind=placement.endpoint_kind,
                evaluator_id=estimate.evaluator_id,
                m=shape[0] if shape else None,
                k=shape[1] if shape else None,
                n=shape[2] if shape else None,
                element_count=operation.element_count,
                compute_latency_ns=estimate.compute_latency_ns,
                compute_energy_pj=estimate.dynamic_energy_pj,
                movement=movement,
                tile_mappings=estimate.tile_mappings,
                reduction_latency_ns=estimate.reduction_latency_ns,
                reduction_communication_energy_pj=(
                    estimate.reduction_communication_energy_pj
                ),
            )
        )
        residency = TensorResidency(
            tensor_id=operation.output_tensor_id,
            endpoint_ids=placement.endpoint_ids,
            endpoint_kind=placement.endpoint_kind,
            element_count=operation.element_count,
        )

    return AtlasEvaluation(
        graph=graph,
        profile=profile,
        results=tuple(results),
        system=system,
    )


def _atlas_embodied_carbon(architecture):
    """Embodied carbon in kg, using the same design tables as the legacy path."""
    segments = build_design_tables(architecture)
    total_embodied_kg = 0.0
    for mode, df, pkg in segments:
        args = types.SimpleNamespace(
            design=df,
            pkg_type=pkg,
            lifetime=3 * 365 * 24,
        )
        embodied_kg, _, _ = find_carbon(args)
        total_embodied_kg += embodied_kg
    return total_embodied_kg


def evaluate_atlas_design_point(
    cache: SimulationCache,
    architecture: dict,
    graph,
    profile,
    objective,
    registry=None,
):
    """Evaluate one modular design point and score it.

    This is the ATLAS branch of the annealing objective. It reuses the modular
    executor and the architecture-level metrics, then hands a raw
    :class:`AtlasDesignPoint` to the replaceable objective model.
    """
    evaluation = evaluate_atlas_graph(
        cache,
        architecture,
        graph,
        profile=profile,
        transfer_model=architecture.get("transfer_model", {}),
        registry=registry,
    )
    power_w, area_mm2, cost_usd = calculate_system_metrics(system_dict=architecture)
    latency_ns = evaluation.latency_ns
    baseline_energy_pj = power_w * latency_ns * 1000
    compute_energy_pj = evaluation.compute_energy_pj
    movement_energy_pj = evaluation.movement_energy_pj
    total_energy_pj = baseline_energy_pj + compute_energy_pj + movement_energy_pj
    operational_carbon_kg = total_opC(
        energy_sram=compute_energy_pj,
        energy_compute=baseline_energy_pj,
        energy_comm=movement_energy_pj,
        lifetime_years=3,
    )
    design_point = AtlasDesignPoint(
        architecture_fingerprint=architecture_fingerprint(architecture),
        profile_name=profile.name,
        profile_fingerprint=profile.fingerprint(),
        latency_ns=latency_ns,
        total_energy_pj=total_energy_pj,
        baseline_energy_pj=baseline_energy_pj,
        compute_energy_pj=compute_energy_pj,
        movement_energy_pj=movement_energy_pj,
        power_w=power_w,
        area_mm2=area_mm2,
        cost_usd=cost_usd,
        embodied_carbon_kg=_atlas_embodied_carbon(architecture),
        operational_carbon_kg=operational_carbon_kg,
        operation_count=len(evaluation.results),
    )
    return design_point, objective.score(design_point)


def add_per_gemm_metrics(output, gemm_metrics, power):
    for index, metrics in enumerate(gemm_metrics, start=1):
        prefix = f"gemm_{index}"
        output[f"{prefix}_name"] = metrics["name"]
        output[f"{prefix}_shape"] = "x".join(str(value) for value in metrics["shape"])
        output[f"{prefix}_latency_ns"] = metrics["latency_ns"]
        output[f"{prefix}_dram_interconnect_energy_pj"] = metrics[
            "dram_interconnect_energy_pj"
        ]
        output[f"{prefix}_sram_energy_pj"] = metrics["sram_energy_pj"]
        output[f"{prefix}_total_energy_pj"] = (
            metrics["dram_interconnect_energy_pj"]
            + metrics["sram_energy_pj"]
            + power * metrics["latency_ns"] * 1000
        )


def add_boundary_metrics(output, boundary_plans):
    output["intermediate_policy_requested"] = (
        boundary_plans[0].requested_policy if boundary_plans else "direct_forward"
    )
    for plan in boundary_plans:
        prefix = f"boundary_{plan.boundary_index}"
        output[f"{prefix}_selected_method"] = plan.selected_method
        output[f"{prefix}_intermediate_bytes"] = plan.intermediate_bytes
        output[f"{prefix}_retained_bytes"] = plan.retained_bytes
        output[f"{prefix}_forwarded_bytes"] = plan.forwarded_bytes
        output[f"{prefix}_dram_spilled_bytes"] = plan.dram_spilled_bytes
        output[f"{prefix}_dram_traffic_bytes"] = plan.dram_traffic_bytes
        output[f"{prefix}_latency_ns"] = plan.latency_ns
        output[f"{prefix}_energy_pj"] = plan.energy_pj
        output[f"{prefix}_producer_cores"] = ",".join(map(str, plan.producer_cores))
        output[f"{prefix}_consumer_cores"] = ",".join(map(str, plan.consumer_cores))
        output[f"{prefix}_routes"] = ";".join(
            "->".join(map(str, route)) for route in plan.routes
        )
        output[f"{prefix}_fallback_reason"] = plan.fallback_reason

class SystemGenerator:
    
    def __init__(self, config_path, stack_diff_size=False):
        (self.max_chiplet, self.sys_array, self.tech_nodes, self.sram_buf_sizes, 
         self.inter_pkg_arch, self.mem_pkg_arch, self.protocol_arch) = read_json_input_params(config_path)
        self.chiplet_gen = ChipletGenerator(self.max_chiplet, self.sys_array, self.tech_nodes, self.sram_buf_sizes, get_area_power, get_sram_area_energy)
        self.wl_mapping_gen = WLMappingGenerator()
        self.stack_diff_size = stack_diff_size

    def generate_system(self, max_retries=20):
        for attempt in range(max_retries):
            try:
                # 1. Generate the chiplets. This is the base for everything.
                chiplets_dict = self.chiplet_gen.generate()
                
                # 2. Instantiate PackageGenerator and attempt to create the package.
                #    This is the step that may fail if stacking rules are not met.
                package_gen = PackageGenerator(chiplets_dict, self.inter_pkg_arch, self.mem_pkg_arch, self.protocol_arch ,stack_diff_size=self.stack_diff_size)
                package_dict = package_gen.generate()
                # 3. Generate the workload mapping.
                mapping_details = self.wl_mapping_gen.generate()
                # 4. Assemble the final dictionary.
                final_system = {}
                final_system.update(chiplets_dict)
                final_system["pkg"] = package_dict
                final_system["WL_mapping"] = {"mapping": mapping_details}
                
                #Update connections for 2.5d grid topology 
                final_system_size, final_system_info = calc_HI_dimension(final_system)
                final_connections = find_connections(final_system_info)
                final_system = update_inter_pkg_connections(final_system, final_connections)
                
                return final_system
            except ValueError as e:
                print(f"[WARNING] Attempt {attempt + 1}/{max_retries} failed to generate valid package: {e}. Retrying...")
        
        raise RuntimeError(f"[ERROR] Failed to generate a valid system after {max_retries} attempts.")

def calculate_cost(
    profile_name='t1',
    cost_avgerage=dict,
    system_dict=dict,
    cache=SimulationCache,
    workload_sequence=None,
    intermediate_policy="direct_forward",
    simulation_result=None,
    system_metrics=None,
):
    if workload_sequence is None:
        raise ValueError("A workload sequence is required for cost calculation")
    print("\n[INFO] --- System Analysis Metrics ---") if print_info else None
    power, area, dollar_cost = system_metrics or calculate_system_metrics(
        system_dict=system_dict
    )
    
    
    print(f"[DEBUG COST] ************** LATENCY ************** ") if print_info else None
    if latency_en and simulation_result is not None:
        latency, energy_comm, energy_sram, gemm_metrics, boundary_plans = (
            simulation_result
        )
    elif latency_en:
        print(f"[INFO] Working on calcuting performance ...") if print_info else None
        latency, energy_comm, energy_sram, gemm_metrics, boundary_plans = simulate_latency_energy(
            cache,
            system_dict,
            workload_sequence,
            intermediate_policy=intermediate_policy,
        )
    else: 
        latency = 0
        energy_comm = 0
        energy_sram = 0
        gemm_metrics = [
            {
                "name": gemm["name"],
                "shape": gemm["shape"],
                "latency_ns": 0,
                "dram_interconnect_energy_pj": 0,
                "sram_energy_pj": 0,
            }
            for gemm in workload_sequence["gemms"]
        ]
        boundary_plans = []
    
    total_operational_C_kg = total_opC(energy_sram=energy_sram, energy_comm=energy_comm, energy_compute=power*latency*1000, lifetime_years=3)
    print(f"[CARBON DEBUG] Operational Carbon for 3 years (kgs) = {total_operational_C_kg} ") if print_info else None
    
    print(f"EMB_CARBON_START\n") if print_info else None
    segments = build_design_tables(system_dict)
    if len(segments) == 1:
        mode, df, pkg = segments[0]
        # process single segment
        print(f"[EMB CARBON] mode is {mode} pkg is {pkg}")  if print_info else None 
        print(f"[EMB CARBON] df is \n{df}")  if print_info else None
        args = types.SimpleNamespace(
            design=df,  # pandas DataFrame
            pkg_type=pkg,      # or whatever applies
            lifetime= 3*365*24         # hrs 
            )
        total_embC_kg, unused_opeC_kg, unused_totalC_kg = find_carbon(args)
        print(f"[EMB CARBON] Embedded Carbon (kgs) = {total_embC_kg} ")   if print_info else None
        print(f"EMB_CARBON_END\n") if print_info else None
    else:
        total_embC_kg = 0
        for mode, df, pkg in segments:
            # process each of '2.5d' and '3d' in fixed order
            print(f"[EMB CARBON] mode is {mode} pkg is {pkg}")   if print_info else None
            print(f"[EMB CARBON] df is \n{df}")  if print_info else None
            args = types.SimpleNamespace(
                design=df,  # pandas DataFrame
                pkg_type=pkg,      # or whatever applies
                lifetime= 3*365*24         # hrs 
            )
            embC_kg, unused_opeC_kg, unused_totalC_kg = find_carbon(args)
            print(f"[EMB CARBON] Embedded Carbon (kgs) = {embC_kg} ") if print_info else None
            print(f"EMB_CARBON_END\n") if print_info else None
            total_embC_kg += embC_kg
            
        
     
    print("\n[INFO] --- System Normalized Metrics calculation ---") if print_info else None
    cost_val, norm_cost_dict, raw_cost_dict = calculate_system_normalized_metrics(
        area=area,
        power=power, 
        energy=energy_comm, 
        energy_sram=energy_sram,
        dollar=dollar_cost,
        latency=latency,
        embCarbon=total_embC_kg,
        opeCarbon=total_operational_C_kg,
        profile_name=profile_name,
        cost_averages=cost_avgerage,
        arch_dict=system_dict
    )

    add_per_gemm_metrics(raw_cost_dict, gemm_metrics, power)
    add_boundary_metrics(raw_cost_dict, boundary_plans)
    raw_cost_dict["intermediate_policy_requested"] = intermediate_policy
    return cost_val, norm_cost_dict, raw_cost_dict


def gen_initial_arch(config_path,stack_diff_size=True):
    
    system_builder = SystemGenerator(config_path=config_path,stack_diff_size=stack_diff_size)
    final_config = system_builder.generate_system()
    return final_config    


def calibration_identity(
    config_path, workload_sequence, intermediate_policy="direct_forward"
):
    if isinstance(config_path, dict):
        search_space = config_path
    else:
        with open(config_path, encoding="utf-8") as file:
            search_space = json.load(file)
    payload = {
        "model_version": CALIBRATION_MODEL_VERSION,
        "search_space": search_space,
        "workload": workload_sequence,
        "intermediate_policy": intermediate_policy,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def calibration_is_current(cost_averages, expected_identity, calibration_iterations):
    return (
        cost_averages.get("_calibration_identity") == expected_identity
        and cost_averages.get("_calibration_samples") == calibration_iterations
    )


def validate_calibration(cost_averages):
    average_keys = {
        "avg_energy",
        "avg_area",
        "avg_dollar_cost",
        "avg_latency",
        "avg_embCarbon",
        "avg_opeCarbon",
    }
    metrics = ("energy", "latency", "area", "cost", "embCarbon", "opeCarbon")
    statistic_keys = {
        f"{metric}_{statistic}"
        for metric in metrics
        for statistic in ("min", "max", "stddev", "mean", "median")
    }
    required = average_keys | statistic_keys
    missing = sorted(required - cost_averages.keys())
    if missing:
        raise ValueError(f"Calibration is missing required metrics: {missing}")
    for key in required:
        value = cost_averages[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Calibration metric {key} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"Calibration metric {key} must be finite")
    for metric in metrics:
        if cost_averages[f"{metric}_min"] <= 0:
            raise ValueError(f"Calibration metric {metric}_min must be positive")
        if cost_averages[f"{metric}_median"] <= 0:
            raise ValueError(f"Calibration metric {metric}_median must be positive")
        if cost_averages[f"{metric}_max"] < cost_averages[f"{metric}_min"]:
            raise ValueError(f"Calibration range for {metric} is invalid")
    return cost_averages

def get_calib_cost_avg(
    calibration_iterations,
    config_path,
    cache,
    calibration_file_path,
    workload_sequence,
    intermediate_policy="direct_forward",
):
    if calibration_iterations <= 0:
        raise ValueError("calibration_iterations must be positive")
    expected_identity = calibration_identity(
        config_path, workload_sequence, intermediate_policy
    )
    
    # Check if the calibration file already exists.
    if os.path.exists(calibration_file_path):
        print(f"\n[INFO] --- Loading existing cost averages from {calibration_file_path} ---") if print_info else None
        with open(calibration_file_path, 'r') as f:
            cost_averages = json.load(f)
        if calibration_is_current(
            cost_averages, expected_identity, calibration_iterations
        ):
            validate_calibration(cost_averages)
            print(f"[INFO] --- Successfully loaded averages: {cost_averages} ---") if print_info else None
            return cost_averages
        print("[INFO] Existing calibration is stale; regenerating it")

    print(f"\n[INFO] --- Running new calibration for {calibration_iterations} iterations in {calibration_file_path} ---") if print_info else None
    
    # Lists to store metrics from each run
    powers, areas, costs, latency_vals = [], [], [], []
    energys = [] 
    embCarbons = []
    opeCarbons = []
    
    calib_all_rows_data =[] #Stores data of all rows 
    calib_cost_stats = {
    'energy': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'cost': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'latency': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'area': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'embCarbon': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0},
    'opeCarbon': {'max': float('-inf'), 'min': float('inf'), 'count': 0, 'mean': 0.0, 'M2': 0.0}
    }
    calib_cost_values = {  # <-- NEW
    'energy': [],
    'cost': [],
    'latency': [],
    'area': [],
    'embCarbon': [],
    'opeCarbon': []
}

    for i in range(calibration_iterations):
        # Generate a new system architecture
        final_config = gen_initial_arch(config_path=config_path)
        print(f"[INFO] Initial architecture json written at cfg/gen_arch directory") if print_info else None
        
        power, area, cost = calculate_system_metrics(system_dict=final_config)
        latency, energy_comm, energy_sram, gemm_metrics, boundary_plans = simulate_latency_energy(
            cache=cache,
            arch_dict=final_config,
            workload_sequence=workload_sequence,
            intermediate_policy=intermediate_policy,
        )
        print(f"[INFO] Calibration stage, power, area, cost done. Now computing latency ....") if print_info else None 
        ope_carbon_kg = total_opC(energy_sram=energy_sram, energy_comm=energy_comm, energy_compute=power*latency*1000, lifetime_years=3)
        seg = build_design_tables(final_config)
        if len(seg) ==1:
            mode, df, pkg = seg[0]
            args = types.SimpleNamespace(
                design=df,  # pandas DataFrame
                pkg_type=pkg,      # or whatever applies
                lifetime= 3*365*24         # hrs 
            )
            emb_carbon_kg, unused_opeC_kg, unused_totalC_kg = find_carbon(args)
        else:
            emb_carbon_kg = 0
            for mode, df, pkg in seg:
                args = types.SimpleNamespace(
                    design=df,  # pandas DataFrame
                    pkg_type=pkg,      # or whatever applies
                    lifetime= 3*365*24         # hrs 
                )
                embC_kg_part, unused_opeC_kg, unused_totalC_kg = find_carbon(args)
                emb_carbon_kg += embC_kg_part
        
        areas.append(area)
        costs.append(cost)
        latency_vals.append(latency)
        embCarbons.append(emb_carbon_kg)
        opeCarbons.append(ope_carbon_kg)
        #####
        powers.append(power) # Keeping this for now, but we will use energy in the future
        energy_compute = power * latency * 1000
        energy = energy_comm + energy_compute + energy_sram
        energys.append(energy)
        #####

        print(f"*****************************************************************************") 
        print(f"\n Calibraiton iteration is {i+1}/{calibration_iterations} ....")
        print(f"\n  Iteration {i+1}/{calibration_iterations}: Power={power}W, Area={area}mm^2, Cost=${cost}, Latency={latency}, Energy={energy} pJ, EmbCarbon={emb_carbon_kg} kgs, OpeCarbon={ope_carbon_kg} kgs")
        print(f"*****************************************************************************") if print_info else None

        
        ##########
        #Dict of current iteration cost metrics
        cost_dict = {
            "energy": energy, 
            "area": area, 
            "latency": latency,
            "dollar_cost": cost,
            "embCarbon": emb_carbon_kg,
            "opeCarbon": ope_carbon_kg
        }
        add_per_gemm_metrics(cost_dict, gemm_metrics, power)
        add_boundary_metrics(cost_dict, boundary_plans)
        
        #Add to current arch dict 
        final_config['cost_metrics'] = cost_dict
        
        processed_dict_rows = process_iteration_wide(final_config, iteration_id=i)
        calib_all_rows_data.append(processed_dict_rows)
        
        calib_data_csv_results = process_arch_details_dump(all_rows_data=calib_all_rows_data)
        
        ##########
        
        for key, val in zip(['energy', 'cost', 'latency', 'area', 'embCarbon', 'opeCarbon'], [energy, cost, latency, area, emb_carbon_kg, ope_carbon_kg]):
            calib_cost_values[key].append(val)  # <-- NEW
            
            # Update min/max
            calib_cost_stats[key]['max'] = max(calib_cost_stats[key]['max'], val)
            calib_cost_stats[key]['min'] = min(calib_cost_stats[key]['min'], val)

            # Welford's online algorithm for mean and variance
            calib_cost_stats[key]['count'] += 1
            delta = val - calib_cost_stats[key]['mean']
            calib_cost_stats[key]['mean'] += delta / calib_cost_stats[key]['count']
            delta2 = val - calib_cost_stats[key]['mean']
            calib_cost_stats[key]['M2'] += delta * delta2
         
    for key in calib_cost_stats:
        count = calib_cost_stats[key]['count']
        if count > 1:
            variance = calib_cost_stats[key]['M2'] / (count - 1)
            calib_cost_stats[key]['stddev'] = math.sqrt(variance)
        else:
            calib_cost_stats[key]['stddev'] = 0.0 
        
        calib_cost_stats[key]['median'] = statistics.median(calib_cost_values[key]) if calib_cost_values[key] else 0.0

    # Calculate the average of each metric
    avg_power = round(sum(powers) / len(powers), 2) if powers else 0 #TODO 1: Remove Power 
    avg_energy = round(sum(energys) / len(energys), 2) if energys else 0
    avg_area = round(sum(areas) / len(areas), 2) if areas else 0
    avg_cost = round(sum(costs) / len(costs), 2) if costs else 0
    avg_latency = round(sum(latency_vals) / len(latency_vals),2) if latency_vals else 0
    avg_embCarbon = round(sum(embCarbons) / len(embCarbons),2) if embCarbons else 0
    avg_opeCarbon = round(sum(opeCarbons) / len(opeCarbons),2) if opeCarbons else 0
    
    energy_min = calib_cost_stats['energy']['min']
    energy_max = calib_cost_stats['energy']['max']
    energy_stddev = calib_cost_stats['energy']['stddev']
    energy_mean = calib_cost_stats['energy']['mean']
    energy_median = calib_cost_stats['energy']['median']
    latency_min = calib_cost_stats['latency']['min']
    latency_max = calib_cost_stats['latency']['max']
    latency_stddev = calib_cost_stats['latency']['stddev']
    latency_mean = calib_cost_stats['latency']['mean']
    latency_median = calib_cost_stats['latency']['median']
    area_min = calib_cost_stats['area']['min']
    area_max = calib_cost_stats['area']['max']
    area_stddev = calib_cost_stats['area']['stddev']
    area_mean = calib_cost_stats['area']['mean']
    area_median = calib_cost_stats['area']['median']
    cost_min = calib_cost_stats['cost']['min']
    cost_max = calib_cost_stats['cost']['max']
    cost_stddev = calib_cost_stats['cost']['stddev']
    cost_mean = calib_cost_stats['cost']['mean']
    cost_median = calib_cost_stats['cost']['median']
    embcarbon_min = calib_cost_stats['embCarbon']['min']
    embcarbon_max = calib_cost_stats['embCarbon']['max']
    embcarbon_stddev = calib_cost_stats['embCarbon']['stddev']
    embcarbon_mean = calib_cost_stats['embCarbon']['mean']
    embcarbon_median = calib_cost_stats['embCarbon']['median']
    opecarbon_min = calib_cost_stats['opeCarbon']['min']
    opecarbon_max = calib_cost_stats['opeCarbon']['max']
    opecarbon_stddev = calib_cost_stats['opeCarbon']['stddev']
    opecarbon_mean = calib_cost_stats['opeCarbon']['mean']
    opecarbon_median = calib_cost_stats['opeCarbon']['median']
    
    cost_averages = {
        "_calibration_model_version": CALIBRATION_MODEL_VERSION,
        "_calibration_identity": expected_identity,
        "_calibration_samples": calibration_iterations,
        "avg_energy": avg_energy,  
        "avg_area": avg_area,
        "avg_dollar_cost": avg_cost,
        "avg_latency": avg_latency,
        "avg_embCarbon": avg_embCarbon,
        "avg_opeCarbon": avg_opeCarbon,
        "energy_min": energy_min,
        "energy_max": energy_max,
        "energy_stddev": energy_stddev,
        "energy_mean": energy_mean,
        "energy_median": energy_median,
        "latency_min": latency_min,
        "latency_max": latency_max,
        "latency_stddev": latency_stddev,
        "latency_mean": latency_mean,
        "latency_median": latency_median,
        "area_min": area_min,
        "area_max": area_max,
        "area_stddev": area_stddev,
        "area_mean": area_mean,
        "area_median": area_median,
        "cost_min": cost_min,
        "cost_max": cost_max,
        "cost_stddev": cost_stddev,
        "cost_mean": cost_mean,
        "cost_median": cost_median,
        "embCarbon_min": embcarbon_min,
        "embCarbon_max": embcarbon_max,
        "embCarbon_stddev": embcarbon_stddev,
        "embCarbon_mean": embcarbon_mean,
        "embCarbon_median": embcarbon_median,
        "opeCarbon_min": opecarbon_min,
        "opeCarbon_max": opecarbon_max,
        "opeCarbon_stddev": opecarbon_stddev,
        "opeCarbon_mean": opecarbon_mean,
        "opeCarbon_median": opecarbon_median
    }
    validate_calibration(cost_averages)

    #Dump sim_annealing arch info
    print(f"[CALIBRATION] Dumping Calibration Results csv ...... ") #if print_info else None
    print(calib_data_csv_results.head())
    csv_file_path = os.path.splitext(calibration_file_path)[0] + ".csv"
    calib_data_csv_results.to_csv(f"{csv_file_path}", index=False)
    print("[INFO] Done dumping Simulation Results csv")

    # Save the new averages to the file for future use
    with open(calibration_file_path, 'w') as f:
        json.dump(cost_averages, f, indent=4)
        
    print(f"[INFO] --- Saved new cost averages to {calibration_file_path}: {cost_averages} ---") if print_info else None
    
    return cost_averages


def get_modular_calib_cost_avg(
    calibration_iterations,
    config_path,
    cache,
    calibration_file_path,
    workload_sequence,
    intermediate_policy,
    graph,
    profile,
    registry=None,
):
    """Calibrate t1--t4 using the same modular evaluator used by annealing."""
    if calibration_iterations <= 0:
        raise ValueError("calibration_iterations must be positive")
    expected_identity = calibration_identity(
        config_path, workload_sequence, intermediate_policy
    )
    calibration_path = Path(calibration_file_path)
    if calibration_path.exists():
        with calibration_path.open(encoding="utf-8") as file:
            cost_averages = json.load(file)
        if calibration_is_current(
            cost_averages, expected_identity, calibration_iterations
        ):
            return validate_calibration(cost_averages)

    scoreless_objective = RawWeightedSumObjective({"latency_ns": 0.0})
    design_points = []
    for _ in range(calibration_iterations):
        architecture = gen_initial_arch(
            config_path=config_path,
            stack_diff_size=True,
        )
        design_point, _ = evaluate_atlas_design_point(
            cache,
            architecture,
            graph,
            profile,
            scoreless_objective,
            registry=registry,
        )
        design_points.append(design_point)

    cost_averages = summarize_design_points(
        design_points,
        calibration_identity=expected_identity,
        model_version=CALIBRATION_MODEL_VERSION,
    )
    validate_calibration(cost_averages)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with calibration_path.open("w", encoding="utf-8") as file:
        json.dump(cost_averages, file, indent=4)
    return cost_averages

def run_calibration(
    wl_idx,
    workload_sequence,
    cache_file,
    run_name,
    cost_profile,
    calibration_iterations=10000,
    intermediate_policy="direct_forward",
):
    print("[STANDALONE_MODE] Standalone framework mode is enabled")
    input_file_path = "cfg/parameters/input.json"
    calibration_file_path = f"cfg/calibration/calibration_{wl_idx}.json"
    print(f"[STANDALONE_MODE] Input file path is {input_file_path}")
    print(f"[STANDALONE_MODE] Calibration file path is {calibration_file_path}")
    cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)
    graph = gemm_sequence_to_atlas_graph(workload_sequence)
    profile = load_evaluation_profile(
        "cfg/profiles/atlas_modular_v1.json"
    ).with_movement_policy(f"{intermediate_policy}_v1")
    cost_avg = get_modular_calib_cost_avg(
        calibration_iterations=calibration_iterations,
        config_path=input_file_path,
        cache=cache,
        calibration_file_path=calibration_file_path,
        workload_sequence=workload_sequence,
        intermediate_policy=intermediate_policy,
        graph=graph,
        profile=profile,
    )
    print("[CALIBRATION] Calibration is completed")
    return cost_avg


def run_policy_comparison(
    wl_idx,
    workload_sequence,
    architecture_file,
    cache_file,
    run_name,
    cost_profile,
):
    if architecture_file is None:
        raise ValueError("--architecture_file is required for run_policy_compare")
    with open(architecture_file) as file:
        architecture = json.load(file)
    calibration_file = f"cfg/calibration/calibration_{wl_idx}.json"
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(
            f"Run calibration first; expected {calibration_file}"
        )
    with open(calibration_file) as file:
        cost_averages = json.load(file)
    expected_identity = calibration_identity(
        "cfg/parameters/input.json", workload_sequence
    )
    if cost_averages.get("_calibration_identity") != expected_identity:
        raise ValueError(
            f"Calibration is stale; rerun calibration for workload {wl_idx}"
        )

    cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)
    graph = gemm_sequence_to_atlas_graph(workload_sequence)
    base_profile = load_evaluation_profile("cfg/profiles/atlas_modular_v1.json")
    rows = []
    for policy in (
        "cold_dram",
        "ideal_on_chip",
        "local_sram",
        "direct_forward",
    ):
        profile = base_profile.with_movement_policy(f"{policy}_v1")
        objective_model = build_atlas_objective(
            objective_id=cost_profile,
            config={"default": cost_profile},
            calibration=cost_averages,
            normalization_mode=calibration_mode,
        )
        design_point, objective = evaluate_atlas_design_point(
            cache,
            architecture,
            graph,
            profile,
            objective_model,
        )
        normalized = {
            f"norm_{name}": value
            for name, value in objective_model.normalized_metrics(
                design_point
            ).items()
        }
        rows.append(
            {
                "policy": policy,
                "objective": objective,
                **normalized,
                **design_point.raw_dict(),
            }
        )
    cache.dump_cache()

    output_name = run_name or f"wl{wl_idx}_{cost_profile}"
    output_path = f"reports/intermediate_policy_comparison_{output_name}.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    print(result[["policy", "objective", "latency", "energy"]].to_string(index=False))
    print(f"[INFO] Policy comparison written to {output_path}")
    return result, output_path



##########################################
####### Simulated Annealing Function

def _run_modular_annealing(
    *,
    cache,
    architecture,
    graph,
    profile,
    search_space,
    objective,
    candidate_profiles,
    initial_temp,
    freezing_temp,
    max_move_per_temp_step,
    cooling_rate,
    temperature_controller,
    level_callback,
    max_total_moves,
    registry,
):
    """Adapt an ATLAS design point to the domain-independent annealer."""
    proposals = []

    def identity(design):
        design_architecture, design_profile = design
        return (
            architecture_fingerprint(design_architecture),
            design_profile.fingerprint(),
        )

    def propose(design, rng):
        current_architecture, current_profile = design
        try:
            moved_architecture, moved_profile, move = mutate_atlas_design_point(
                current_architecture,
                current_profile,
                search_space,
                candidate_profiles,
                rng=rng,
            )
        except UnsupportedEvaluation:
            proposals.append(None)
            raise
        candidate = (moved_architecture, moved_profile)
        proposals.append(candidate)
        return candidate, move

    def evaluate(design):
        design_architecture, design_profile = design
        return evaluate_atlas_design_point(
            cache,
            design_architecture,
            graph,
            design_profile,
            objective,
            registry=registry,
        )

    result = anneal(
        initial_design=(architecture, profile),
        propose=propose,
        evaluate=evaluate,
        identity=identity,
        rng=random,
        initial_temperature=initial_temp,
        freezing_temperature=freezing_temp,
        moves_per_temperature=max_move_per_temp_step,
        cooling_rate=cooling_rate,
        temperature_controller=temperature_controller,
        level_callback=level_callback,
        max_total_moves=max_total_moves,
    )

    reason_codes = {"improvement": 1, "metropolis": 2, "rejected": 3}
    trace_rows = []
    architecture_rows = []
    for generic_row, candidate in zip(result.trace, proposals):
        evaluation = generic_row.get("candidate_evaluation")
        candidate_profile = candidate[1] if candidate is not None else None
        candidate_architecture = candidate[0] if candidate is not None else None
        row = {
            "temperature": generic_row["temperature"],
            "inner_iter": generic_row["inner_iter"],
            "1L_move": 1,
            "2L_move": generic_row.get("move_name"),
            "SA_run_loop": generic_row["attempt"],
            "best_cost": generic_row["best_cost_after"],
            "new_cost": generic_row.get("candidate_cost"),
            "cost_diff": generic_row.get("cost_diff"),
            "move_accepted": generic_row["move_accepted"],
            "move_type": reason_codes.get(generic_row["acceptance_reason"]),
            "proposal_valid": generic_row["proposal_valid"],
            "proposal_changed": generic_row["proposal_changed"],
            "current_cost_before": generic_row["current_cost_before"],
            "current_cost_after": generic_row["current_cost_after"],
            "best_cost_before": generic_row["best_cost_before"],
            "best_cost_after": generic_row["best_cost_after"],
            "candidate_fingerprint": (
                architecture_fingerprint(candidate_architecture)
                if candidate_architecture is not None
                else None
            ),
        }
        if evaluation is not None:
            row.update(
                {
                    "atlas_objective": generic_row["candidate_cost"],
                    "profile_name": candidate_profile.name,
                    "profile_fingerprint": candidate_profile.fingerprint(),
                }
            )
            row.update(evaluation.raw_dict())
        trace_rows.append(row)
        architecture_rows.append(
            process_iteration_wide(
                candidate_architecture,
                iteration_id=generic_row["attempt"],
            )
        )

    cache.dump_cache()
    trace = pd.DataFrame(trace_rows)
    best_architecture, best_profile = result.best_design
    trace.attrs["atlas"] = True
    trace.attrs["best_profile"] = best_profile.canonical_dict()
    trace.attrs["best_profile_fingerprint"] = best_profile.fingerprint()
    trace.attrs["candidate_profiles"] = tuple(
        candidate.name for candidate in candidate_profiles
    )
    return (
        result.best_cost,
        best_architecture,
        trace,
        process_arch_details_dump(all_rows_data=architecture_rows),
    )

def sim_annealing(
    wl_idx,
    workload_sequence,
    cache_file,
    run_name,
    cost_profile,
    initial_temp=4000,
    freezing_temp=1e-3,
    max_move_per_temp_step=20,
    cooling_rate=0.99,
    calibration_iterations=200,
    intermediate_policy="direct_forward",
    random_seed=None,
    input_file_path="cfg/parameters/input.json",
    calibration_file_path=None,
    initial_architecture=None,
    temperature_controller=None,
    level_callback=None,
    max_total_moves=None,
    atlas_graph=None,
    atlas_profile=None,
    atlas_search_space=None,
    atlas_objective=None,
    candidate_profiles=None,
    registry=None,
):
    if initial_temp <= 0:
        raise ValueError("initial_temp must be positive")
    if freezing_temp <= 0 or freezing_temp >= initial_temp:
        raise ValueError("freezing_temp must be positive and below initial_temp")
    if max_move_per_temp_step <= 0:
        raise ValueError("max_move_per_temp_step must be positive")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1")
    if temperature_controller is not None and (
        max_total_moves is None or max_total_moves <= 0
    ):
        raise ValueError("adaptive annealing requires a positive move budget")

    graph_was_supplied = atlas_graph is not None
    if graph_was_supplied:
        if workload_sequence is not None:
            raise ValueError(
                "atlas annealing takes an atlas_graph, not a workload_sequence"
            )
        if atlas_profile is None:
            raise ValueError("atlas annealing requires an evaluation profile")
        if atlas_search_space is None:
            raise ValueError("atlas annealing requires an atlas search space")
        if initial_architecture is None:
            raise ValueError("atlas annealing requires an initial architecture")
        if atlas_objective is None:
            atlas_objective = build_atlas_objective()
        candidate_profiles = tuple(candidate_profiles or ())
        if not candidate_profiles:
            raise ValueError("atlas annealing requires candidate profiles")
        validate_atlas_architecture(initial_architecture)
    else:
        if workload_sequence is None:
            raise ValueError("annealing requires an ATLAS graph or GEMM workload sequence")
        atlas_graph = gemm_sequence_to_atlas_graph(workload_sequence)
        atlas_profile = atlas_profile or load_evaluation_profile(
            "cfg/profiles/atlas_modular_v1.json"
        )
        atlas_profile = atlas_profile.with_movement_policy(
            f"{intermediate_policy}_v1"
        )
        with open(input_file_path, encoding="utf-8") as file:
            input_parameters = json.load(file)
        atlas_search_space = atlas_search_space or sequential_gemm_search_space(
            input_parameters
        )
        candidate_profiles = tuple(candidate_profiles or (atlas_profile,))

    if random_seed is not None:
        random.seed(random_seed)
    
    print(f"[STANDALONE_MODE] Standalone framework mode is enabled")
    calibration_file_path = calibration_file_path or (
        f"cfg/calibration/calibration_{wl_idx}.json"
    )
    print(f"[STANDALONE_MODE] Input file path is {input_file_path}")
    print(f"[STANDALONE_MODE] Calibration file path is {calibration_file_path}")

    cache = SimulationCache(cache_file, fast_test=fast_test, simulator_dir=run_name)

    if not graph_was_supplied:
        cost_avg = get_modular_calib_cost_avg(
            calibration_iterations=calibration_iterations,
            config_path=input_file_path,
            cache=cache,
            calibration_file_path=calibration_file_path,
            workload_sequence=workload_sequence,
            intermediate_policy=intermediate_policy,
            graph=atlas_graph,
            profile=atlas_profile,
            registry=registry,
        )
        atlas_objective = atlas_objective or build_atlas_objective(
            objective_id=cost_profile,
            config={"default": cost_profile},
            calibration=cost_avg,
            normalization_mode=calibration_mode,
        )
        if random_seed is not None:
            random.seed(random_seed)
        if initial_architecture is None:
            initial_architecture = gen_initial_arch(
                config_path=input_file_path,
                stack_diff_size=True,
            )
        validate_atlas_architecture(initial_architecture)

    return _run_modular_annealing(
        cache=cache,
        architecture=initial_architecture,
        graph=atlas_graph,
        profile=atlas_profile,
        search_space=atlas_search_space,
        objective=atlas_objective,
        candidate_profiles=candidate_profiles,
        initial_temp=initial_temp,
        freezing_temp=freezing_temp,
        max_move_per_temp_step=max_move_per_temp_step,
        cooling_rate=cooling_rate,
        temperature_controller=temperature_controller,
        level_callback=level_callback,
        max_total_moves=max_total_moves,
        registry=registry,
    )


if __name__ == "__main__":
    #################

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=int, choices=sorted(WORKLOAD_CONFIGS),
                        help = "workload index to retrieve an ordered GEMM sequence")
    parser.add_argument("--iteration", type = int, default=1,
                        help="Number of iterations running, default is 1")
    parser.add_argument(
        "--calibration_iterations",
        type=int,
        default=10,
        help="Number of random architectures used for calibration, default is 10",
    )
    parser.add_argument(
        "--initial_temp",
        type=float,
        default=CLI_INITIAL_TEMP,
        help="Starting simulated-annealing temperature",
    )
    parser.add_argument(
        "--freezing_temp",
        type=float,
        default=CLI_FREEZING_TEMP,
        help="Temperature at which simulated annealing stops",
    )
    parser.add_argument(
        "--max_move_per_temp_step",
        type=int,
        default=CLI_MAX_MOVE_PER_TEMP_STEP,
        help="Move proposals evaluated at each temperature level",
    )
    parser.add_argument(
        "--cooling_rate",
        type=float,
        default=CLI_COOLING_RATE,
        help="Temperature multiplier applied after each level",
    )
    parser.add_argument("--run_name", type = str, default=None,
                        help="Name of current run, used to create work/log folder, default is None")
    parser.add_argument("--cache_file", type = str, default="cfg/static_cache/static_cache.csv",
                        help="Cache file used to accelerate the simulation")
    parser.add_argument("--cost_profile", type = str, default="t1",
                        help="Cost profiles used to calculate cost in SimAnnelaing. Options - t1, t2, t3, t4")
    parser.add_argument(
        "--intermediate_policy",
        choices=sorted(INTERMEDIATE_POLICIES),
        default="direct_forward",
        help="How sequential GEMM intermediates are transferred",
    )
    parser.add_argument(
        "--architecture_file",
        type=str,
        default=None,
        help="Fixed architecture JSON used by run_policy_compare",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible calibration and simulated-annealing randomness",
    )

    parser.add_argument("--run_mode", type = str, default="run_sim_anneal",
                        help="Options: run_sim_anneal, run_calibration, run_policy_compare")
    parser.add_argument(
        "--atlas_graph",
        type=str,
        default=None,
        help="Raw ATLAS graph dump; enables modular annealing over ATLAS",
    )
    parser.add_argument(
        "--evaluation_profile",
        type=str,
        default=None,
        help="Evaluation profile JSON for modular ATLAS annealing",
    )
    parser.add_argument(
        "--atlas_search_space",
        type=str,
        default="cfg/experiments/atlas_modular_search_space.json",
        help="Modular architecture/profile search space for ATLAS annealing",
    )
    parser.add_argument(
        "--atlas_objective_config",
        type=str,
        default=None,
        help="Objective config JSON (default: cfg/parameters/atlas_objective.json)",
    )
    #parser.add_argument("--json_file_path", type = str, default=None,
    #                    help="Path to the JSON file for debugging purposes, default is script/analysis_dir/json_out/example.json")

    #Example: python -m main --workload 1 --cost_profile t1 --run_mode run_sim_anneal --run_name test_run

    args = parser.parse_args()
    
    run_name = args.run_name
    if run_name == None:
        run_name = ""

    wl_idx = args.workload
    iteration = args.iteration
    cache_file = args.cache_file
    cost_profile = args.cost_profile
    run_mode = args.run_mode
    intermediate_policy = args.intermediate_policy
    architecture_file = args.architecture_file
    seed = args.seed
    calibration_iterations = args.calibration_iterations
    initial_temp = args.initial_temp
    freezing_temp = args.freezing_temp
    max_move_per_temp_step = args.max_move_per_temp_step
    cooling_rate = args.cooling_rate
    atlas_graph_path = args.atlas_graph
    evaluation_profile_path = args.evaluation_profile
    atlas_search_space_path = args.atlas_search_space
    atlas_objective_config_path = args.atlas_objective_config
    #json_file_path = args.json_file_path

    atlas_mode = atlas_graph_path is not None

    if wl_idx is None and not atlas_mode:
        print(f"[Warning] Using Default Workload Index = 1")
        # parser.print_help()
        # exit(-1)
        wl_idx = 1

    if iteration <= 0:
        parser.error("--iteration must be positive")
    if calibration_iterations <= 0:
        parser.error("--calibration_iterations must be positive")
    if initial_temp <= 0:
        parser.error("--initial_temp must be positive")
    if freezing_temp <= 0 or freezing_temp >= initial_temp:
        parser.error("--freezing_temp must be positive and below --initial_temp")
    if max_move_per_temp_step <= 0:
        parser.error("--max_move_per_temp_step must be positive")
    if not 0 < cooling_rate < 1:
        parser.error("--cooling_rate must be between 0 and 1")

    atlas_graph = None
    atlas_profile = None
    atlas_search_space = None
    candidate_profiles = None
    atlas_objective = None
    if atlas_mode:
        atlas_graph = load_atlas_graph(atlas_graph_path)
        atlas_profile = load_evaluation_profile(evaluation_profile_path)
        atlas_search_space = _load_json_file(atlas_search_space_path)
        candidate_profiles = [
            load_evaluation_profile(path)
            for path in candidate_profile_paths(atlas_search_space)
        ]
        atlas_objective = build_atlas_objective(
            config_path=atlas_objective_config_path
            or "cfg/parameters/atlas_objective.json"
        )
        workload_sequence = None
    else:
        workload_sequence = parse_workload_entry(wl_idx, WORKLOAD_CONFIGS[wl_idx])
        print(
            f"[INFO] Workload sequence: {workload_sequence['name']} "
            f"({len(workload_sequence['gemms'])} GEMM(s))"
        )

    if atlas_mode:
        file_run_name = f"atlas_{iteration}iteration_{run_name}_{atlas_profile.name}"
    else:
        file_run_name = f"wl{wl_idx}_{iteration}iteration_{run_name}_{cost_profile}"
        if intermediate_policy != "direct_forward":
            file_run_name += f"_{intermediate_policy}"
    
    print(f"[INFO] Run name: {file_run_name}, cache_file: {cache_file}, Iteration: {iteration}")
    
    if run_mode == "run_policy_compare":
        run_policy_comparison(
            wl_idx=wl_idx,
            workload_sequence=workload_sequence,
            architecture_file=architecture_file,
            cache_file=cache_file,
            run_name=file_run_name,
            cost_profile=cost_profile,
        )
        sys.exit(0)
    #################

    for i in range(iteration):
        iteration_seed = None if seed is None else seed + i
        iteration_run_name = (
            f"{file_run_name}_run{i + 1:02d}" if iteration > 1 else file_run_name
        )
        if run_mode == "run_sim_anneal": #Runs Simulated Annealing
            start_time = time.time()
            if atlas_mode:
                initial_architecture = (
                    _load_json_file(architecture_file)
                    if architecture_file
                    else None
                )
                best_cost, best_arch, sa_details_csv, sim_results_csv = sim_annealing(
                    wl_idx=None,
                    workload_sequence=None,
                    cache_file=cache_file,
                    run_name=iteration_run_name,
                    cost_profile=cost_profile,
                    initial_temp=initial_temp,
                    freezing_temp=freezing_temp,
                    max_move_per_temp_step=max_move_per_temp_step,
                    cooling_rate=cooling_rate,
                    calibration_iterations=calibration_iterations,
                    random_seed=iteration_seed,
                    initial_architecture=initial_architecture,
                    atlas_graph=atlas_graph,
                    atlas_profile=atlas_profile,
                    atlas_search_space=atlas_search_space,
                    atlas_objective=atlas_objective,
                    candidate_profiles=candidate_profiles,
                )
            else:
                best_cost, best_arch, sa_details_csv, sim_results_csv = sim_annealing(
                                                            wl_idx=wl_idx,
                                                            workload_sequence=workload_sequence,
                                                            cache_file = cache_file,
                                                            run_name=iteration_run_name,
                                                            cost_profile=cost_profile,
                                                            initial_temp=initial_temp,
                                                            freezing_temp=freezing_temp,
                                                            max_move_per_temp_step=max_move_per_temp_step,
                                                            cooling_rate=cooling_rate,
                                                            calibration_iterations=calibration_iterations,
                                                            intermediate_policy=intermediate_policy,
                                                            random_seed=iteration_seed,
                                                            )
            
            dump_results(
                sa_details_csv,
                sim_results_csv,
                best_arch,
                best_cost,
                iteration_run_name,
            )
            if atlas_mode:
                _dump_best_profile(sa_details_csv, iteration_run_name)
            end_time = time.time()
            find_run_time(start_time,end_time)
        elif run_mode == "run_calibration": #Runs Calibration
            print(f"[INFO] Running Calibration only")
            start_time = time.time()
            
            print(f"[INFO] Running calibration for {calibration_iterations} iterations to get variation data")
            
            if iteration_seed is not None:
                random.seed(iteration_seed)
            run_calibration(
                wl_idx=wl_idx,
                workload_sequence=workload_sequence,
                cache_file=cache_file,
                run_name=iteration_run_name,
                cost_profile=cost_profile,
                calibration_iterations=calibration_iterations,
                intermediate_policy=intermediate_policy,
            )
            
            end_time = time.time()
            find_run_time(start_time,end_time)
        
        else:
            print(f"[INFO] Please ensure run_mode is correct")
