import json
import random
import unittest
from pathlib import Path

from system.utils.AtlasAnnealingMoves import (
    MOVE_TYPES,
    candidate_profile_paths,
    mutate_atlas_design_point,
    validate_atlas_architecture,
)
from system.utils.EvaluationProfile import load_evaluation_profile
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation


ARCHITECTURE = Path("cfg/examples/sa_fpga_architecture.json")
SEARCH_SPACE = Path("cfg/experiments/atlas_modular_search_space_placeholders.json")


def _load(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


class AtlasAnnealingMoveTests(unittest.TestCase):
    def setUp(self):
        self.architecture = _load(ARCHITECTURE)
        self.search_space = _load(SEARCH_SPACE)
        self.profiles = [
            load_evaluation_profile(path)
            for path in candidate_profile_paths(self.search_space)
        ]

    def test_search_space_names_profiles(self):
        self.assertGreaterEqual(len(self.profiles), 2)

    def test_many_moves_preserve_single_sa_single_fpga(self):
        rng = random.Random(2024)
        architecture = self.architecture
        profile = self.profiles[0]
        for _ in range(300):
            architecture, profile, move = mutate_atlas_design_point(
                architecture,
                profile,
                self.search_space,
                self.profiles,
                rng=rng,
            )
            self.assertIn(move, MOVE_TYPES)
            sa_key, fpga_key = validate_atlas_architecture(architecture)
            self.assertTrue(architecture[fpga_key]["chiplet_type"] == "fpga")
            self.assertNotIn("chiplet_type", architecture[sa_key])
            self.assertIn(profile.name, {p.name for p in self.profiles})

    def test_mutation_does_not_mutate_input_architecture(self):
        original = json.dumps(self.architecture, sort_keys=True)
        rng = random.Random(7)
        mutate_atlas_design_point(
            self.architecture,
            self.profiles[0],
            self.search_space,
            self.profiles,
            rng=rng,
        )
        self.assertEqual(json.dumps(self.architecture, sort_keys=True), original)

    def test_reference_architecture_validates(self):
        sa_key, fpga_key = validate_atlas_architecture(self.architecture)
        self.assertEqual(sa_key, "Chiplet_1")
        self.assertEqual(fpga_key, "Chiplet_2")

    def test_architecture_without_fpga_is_rejected(self):
        broken = {
            key: value
            for key, value in self.architecture.items()
            if not key.startswith("Chiplet_2")
        }
        with self.assertRaises(UnsupportedEvaluation):
            validate_atlas_architecture(broken)

    def test_profile_move_is_available_only_with_candidates(self):
        rng = random.Random(1)
        architecture, profile, _ = mutate_atlas_design_point(
            self.architecture,
            self.profiles[0],
            self.search_space,
            candidate_profiles=(),
            rng=rng,
        )
        self.assertIs(profile, self.profiles[0])


if __name__ == "__main__":
    unittest.main()
