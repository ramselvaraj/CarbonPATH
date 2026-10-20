import json, tempfile, warnings, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from main import evaluate_atlas_graph
from system.utils import Rule4MlEvaluators
from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.SimulationCache import SimulationCache

arch = json.load(open("cfg/examples/sa_fpga_architecture.json"))
print("FPGA:", {k: v for k, v in arch.items() if isinstance(v, dict) and v.get("chiplet_type") == "fpga"})
graph = load_atlas_graph(Path("cfg/examples/atlas/dense_rule4ml_activations.graph_dump.json"))
profile = load_evaluation_profile(Path("cfg/profiles/rule4ml_fpga_ops_v3.json"))
d = tempfile.mkdtemp()
cols = ["core_size","data_flow","bandwidth","buffer_size","M","K","N","latency"]
pd.DataFrame(columns=cols).to_csv(Path(d)/"c.csv", index=False)
cache = SimulationCache(Path(d)/"c.csv", simulator_dir=d)
orig = Rule4MlEvaluators.predict_activation
def spy(**kw):
    r = orig(**kw)
    print("PREDICT", {k: kw[k] for k in kw}, "->", r)
    return r
with patch.object(Rule4MlEvaluators, "predict_activation", side_effect=spy):
    ev = evaluate_atlas_graph(cache, arch, graph, profile=profile)
for r in ev.results:
    print("OP", r.operation_id, r.evaluator_id, r.compute_latency_ns)
