import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from main import evaluate_atlas_design_point
from system.utils.AtlasGraphAdapter import load_atlas_graph
from system.utils.AtlasObjective import (
    AtlasDesignPoint,
    RawWeightedSumObjective,
    build_atlas_objective,
)
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.SimulationCache import SimulationCache
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
GRAPH = Path("cfg/examples/atlas/dense_relu_funnel.graph_dump.json")
PROFILE_DIR = Path("cfg/profiles/placeholders")
REFERENCE_PROFILE = Path("cfg/profiles/atlas_modular_v1.json")
BAD_PROFILES = (
    PROFILE_DIR / "ph_slow_gemm_v1.json",
    PROFILE_DIR / "ph_dram_always_v1.json",
    PROFILE_DIR / "ph_pessimistic_transfer_v1.json",
)

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


def _load(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


class AtlasObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        self.graph = load_atlas_graph(GRAPH)
        self.objective = build_atlas_objective()

    def _cache(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        cache_path = Path(directory.name) / "cache.csv"
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(cache_path, index=False)
        return SimulationCache(cache_path, simulator_dir=directory.name)

    def _design_point(self, profile_path):
        profile = load_evaluation_profile(profile_path)
        with patch("main.simulate_single_gemm") as simulate:
            simulate.return_value = {
                "latency_ns": 1000.0,
                "dram_interconnect_energy_pj": 200.0,
                "sram_energy_pj": 100.0,
            }
            design_point, score = evaluate_atlas_design_point(
                self._cache(),
                self.architecture,
                self.graph,
                profile,
                self.objective,
            )
        return design_point, score

    def test_known_bad_profiles_score_strictly_worse(self):
        _, reference_score = self._design_point(REFERENCE_PROFILE)
        for path in BAD_PROFILES:
            _, bad_score = self._design_point(path)
            self.assertGreater(
                bad_score,
                reference_score,
                f"{path.name} should score worse than the reference profile",
            )

    def test_profile_change_changes_design_point_fingerprint(self):
        reference, _ = self._design_point(REFERENCE_PROFILE)
        for path in BAD_PROFILES:
            bad, _ = self._design_point(path)
            self.assertNotEqual(
                reference.profile_fingerprint, bad.profile_fingerprint
            )

    def test_penalty_raises_objective_monotonically(self):
        design_point = self._design_point(REFERENCE_PROFILE)[0]
        base = RawWeightedSumObjective({"latency_ns": 1.0}).score(design_point)
        heavier = RawWeightedSumObjective({"latency_ns": 3.0}).score(design_point)
        self.assertAlmostEqual(heavier, base * 3.0)

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(UnsupportedEvaluation):
            RawWeightedSumObjective({"not_a_metric": 1.0})

    def test_unknown_objective_id_is_rejected(self):
        with self.assertRaises(UnsupportedEvaluation):
            build_atlas_objective(
                objective_id="nope",
                config={"default": "raw_weighted_sum_v0", "objectives": {}},
            )

    def test_objective_requires_design_point(self):
        with self.assertRaises(ValueError):
            build_atlas_objective().score(object())

    def test_raw_dict_exposes_legacy_aliases(self):
        design_point = self._design_point(REFERENCE_PROFILE)[0]
        raw = design_point.raw_dict()
        for key in ("latency", "energy", "area", "dollar", "embCarbon", "opeCarbon"):
            self.assertIn(key, raw)

    def test_design_point_is_frozen_dataclass(self):
        self.assertTrue(hasattr(AtlasDesignPoint, "__dataclass_fields__"))


if __name__ == "__main__":
    unittest.main()
