"""Fused versus unfused sigmoid and tanh, on the dense_rule4ml_activations workload.

Run from the repo root: PYTHONPATH=. python script/rule4ml_unfused_experiment.py > out.csv

Variants (profile rule4ml_fused_dummy_v0): every on/off combination of fusing
relu, sigmoid and tanh (8 variants), named "unfused" or "fused_<types>".
Softmax is never fusable. It always runs on the FPGA.
One row per (variant, board, FPGA clock, operation).
"""
import copy, json, os, tempfile, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from main import evaluate_atlas_graph
from system.utils import Rule4MlEvaluators
from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.Rule4MlBoards import board_chiplet_fields
from system.utils.SimulationCache import SimulationCache
from system.utils.WorkloadMapping import FUSED_ACTIVATION_POLICY, initial_workload_mapping

ARCH = json.load(open("cfg/examples/sa_fpga_architecture.json"))
SEARCH = json.load(open("cfg/experiments/atlas_modular_search_space.json"))
GRAPH = load_atlas_graph(Path("cfg/examples/atlas/dense_rule4ml_activations.graph_dump.json"))
PROFILE = load_evaluation_profile(Path("cfg/profiles/rule4ml_fused_dummy_v0.json"))
BOARDS = ["pynq-z2", "zcu102", "alveo-u200", "alveo-u250"]
CLOCKS_MHZ = [200, 300, 400]
import itertools
_TYPES = ("relu", "sigmoid", "tanh")
VARIANTS = {
    ("fused_" + "+".join(c) if c else "unfused"): c
    for n in range(len(_TYPES) + 1)
    for c in itertools.combinations(_TYPES, n)
}

d = tempfile.mkdtemp()
pd.DataFrame(columns=["core_size", "data_flow", "bandwidth", "buffer_size", "M", "K", "N", "latency"]).to_csv(
    Path(d) / "c.csv", index=False)
cache = SimulationCache(Path(d) / "c.csv", simulator_dir=d)

orig = Rule4MlEvaluators.predict_activation
predictions = {}


def spy(**kw):
    r = orig(**kw)
    predictions[kw["activation"]] = r
    return r


COLS = ("variant;board;fpga_mhz;op;type;kind;compute_ns;compute_pj;move_ns_after;move_pj_after;"
        "total_ns;total_pj_dynamic;r4m_cycles;r4m_lut;r4m_ff;r4m_dsp;r4m_bram").split(";")
print(";".join(COLS))
for board in BOARDS:
    for mhz in CLOCKS_MHZ:
        arch = copy.deepcopy(ARCH)
        fpga = arch["Chiplet_2"]
        fpga.update(board_chiplet_fields(board))
        fpga["frequency_hz"] = mhz * 1_000_000
        for variant, fused_types in VARIANTS.items():
            mapping = initial_workload_mapping(GRAPH, arch, PROFILE, SEARCH)
            for index in mapping.fusable_indexes:
                if GRAPH.operations[index].operation_type in fused_types:
                    mapping = mapping.with_fusion_change(
                        index, "sa", mapping.operations[index - 1].endpoint_ids,
                        FUSED_ACTIVATION_POLICY, "local_sram_v1")
            predictions.clear()
            with patch.object(Rule4MlEvaluators, "predict_activation", side_effect=spy):
                ev = evaluate_atlas_graph(cache, arch, GRAPH, profile=PROFILE, mapping=mapping)
            for i, r in enumerate(ev.results):
                nxt = ev.results[i + 1].movement if i + 1 < len(ev.results) else None
                p = predictions.get(r.operation_type) if r.endpoint_kind == "fpga" else None
                g = lambda k: (p["resources"].get(k, "") if k != "CYCLES" else p["cycles"]) if p else ""
                print(";".join(str(x) for x in (
                    variant, board, mhz, r.operation_id, r.operation_type, r.endpoint_kind,
                    round(r.compute_latency_ns, 3), round(r.compute_energy_pj, 3),
                    round(nxt.charged_latency_ns, 3) if nxt else 0.0,
                    round(nxt.charged_energy_pj, 3) if nxt else 0.0,
                    "", "", g("CYCLES"), g("LUT"), g("FF"), g("DSP"), g("BRAM"))), flush=True)
            print(";".join(str(x) for x in (variant, board, mhz, "TOTAL", "", "", round(ev.latency_ns, 3),
                  round(ev.compute_energy_pj, 3), "", round(ev.movement_energy_pj, 3),
                  round(ev.latency_ns, 3), round(ev.compute_energy_pj + ev.movement_energy_pj, 3),
                  "", "", "", "", "")), flush=True)
