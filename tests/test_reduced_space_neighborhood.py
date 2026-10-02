import itertools
import json
import unittest
from dataclasses import dataclass, replace

from system.utils.AtlasAnnealingMoves import MOVE_TYPES, sequential_gemm_search_space


ARRAYS = ("64x64", "128x128")
TECH_NODES = ("7", "14")
SRAMS = {"64x64": (256, 512), "128x128": (1024, 2048)}
DATAFLOWS = ("ws", "os", "is")
ADVANCED_PROTOCOLS = ("ucie_adv", "aib")
LINKS = (
    ("2.5d_emib", ADVANCED_PROTOCOLS),
    ("2.5d_rdl", ("ucie_std",)),
    ("3d_tsv", ("ucie_3d",)),
    ("3d_hyb_bond", ("ucie_3d",)),
)


@dataclass(frozen=True)
class ReducedDesign:
    sa_specs: tuple
    dataflow: str
    split_k: bool
    ascending: bool
    memory: str
    link: str
    protocol: str


def _all_specs():
    return tuple(
        (array, tech, sram)
        for array in ARRAYS
        for tech in TECH_NODES
        for sram in SRAMS[array]
    )


SPECS = _all_specs()


def _package_choices(sa_count):
    if sa_count == 1:
        return tuple(
            (memory, "2d_na", "na") for memory in ("ddr4", "ddr5")
        )
    return tuple(
        (memory, link, protocol)
        for memory in ("ddr5", "hbm2")
        for link, protocols in LINKS
        for protocol in protocols
    )


def _designs():
    for sa_count in (1, 2):
        for specs in itertools.combinations_with_replacement(SPECS, sa_count):
            for dataflow in DATAFLOWS:
                for split_k in (False, True):
                    for ascending in (False, True):
                        for memory, link, protocol in _package_choices(sa_count):
                            yield ReducedDesign(
                                sa_specs=specs,
                                dataflow=dataflow,
                                split_k=split_k,
                                ascending=ascending,
                                memory=memory,
                                link=link,
                                protocol=protocol,
                            )


def _with_specs(design, specs):
    return replace(design, sa_specs=tuple(sorted(specs)))


def _with_package(design, choice):
    memory, link, protocol = choice
    return replace(design, memory=memory, link=link, protocol=protocol)


def _original_neighbors(design):
    neighbors = set()
    specs = design.sa_specs

    if len(specs) == 1:
        for added in SPECS:
            base = _with_specs(design, specs + (added,))
            neighbors.update(_with_package(base, choice) for choice in _package_choices(2))
    else:
        for index in range(2):
            base = _with_specs(design, (specs[1 - index],))
            neighbors.update(_with_package(base, choice) for choice in _package_choices(1))

    for index, (array, tech, sram) in enumerate(specs):
        for new_array in ARRAYS:
            if new_array != array:
                for new_sram in SRAMS[new_array]:
                    changed = list(specs)
                    changed[index] = (new_array, tech, new_sram)
                    neighbors.add(_with_specs(design, changed))
        for new_tech in TECH_NODES:
            if new_tech != tech:
                changed = list(specs)
                changed[index] = (array, new_tech, sram)
                neighbors.add(_with_specs(design, changed))
        for new_sram in SRAMS[array]:
            if new_sram != sram:
                changed = list(specs)
                changed[index] = (array, tech, new_sram)
                neighbors.add(_with_specs(design, changed))

    neighbors.update(
        replace(design, dataflow=value)
        for value in DATAFLOWS
        if value != design.dataflow
    )
    neighbors.add(replace(design, split_k=not design.split_k))
    neighbors.add(replace(design, ascending=not design.ascending))

    valid_packages = _package_choices(len(specs))
    neighbors.update(
        replace(design, memory=memory)
        for memory in {choice[0] for choice in valid_packages}
        if memory != design.memory
    )
    if design.link.startswith("2.5d_"):
        link_family = [entry for entry in LINKS if entry[0].startswith("2.5d_")]
    elif design.link.startswith("3d_"):
        link_family = [entry for entry in LINKS if entry[0].startswith("3d_")]
    else:
        link_family = []
    for link, protocols in link_family:
        if link != design.link:
            neighbors.update(
                replace(design, link=link, protocol=protocol)
                for protocol in protocols
            )
    if design.link == "2.5d_emib":
        neighbors.update(
            replace(design, protocol=protocol)
            for protocol in ADVANCED_PROTOCOLS
            if protocol != design.protocol
        )
    neighbors.discard(design)
    return neighbors


def _modular_neighbors(design, search_space):
    neighbors = set()
    enabled = {
        move for move in MOVE_TYPES if search_space["move_weights"].get(move, 0) > 0
    }
    specs = design.sa_specs

    if "sa_count" in enabled:
        if len(specs) == 1:
            for added in SPECS:
                base = _with_specs(design, specs + (added,))
                neighbors.update(
                    _with_package(base, choice) for choice in _package_choices(2)
                )
        else:
            for remaining in set(specs):
                base = _with_specs(design, (remaining,))
                neighbors.update(
                    _with_package(base, choice) for choice in _package_choices(1)
                )

    for index, (array, tech, sram) in enumerate(specs):
        if "sa_array" in enabled:
            for new_array in ARRAYS:
                if new_array != array:
                    for new_sram in SRAMS[new_array]:
                        changed = list(specs)
                        changed[index] = (new_array, tech, new_sram)
                        neighbors.add(_with_specs(design, changed))
        if "sa_tech_node" in enabled:
            for new_tech in TECH_NODES:
                if new_tech != tech:
                    changed = list(specs)
                    changed[index] = (array, new_tech, sram)
                    neighbors.add(_with_specs(design, changed))
        if "sa_sram" in enabled:
            for new_sram in SRAMS[array]:
                if new_sram != sram:
                    changed = list(specs)
                    changed[index] = (array, tech, new_sram)
                    neighbors.add(_with_specs(design, changed))

    if "gemm_dataflow" in enabled:
        neighbors.update(
            replace(design, dataflow=value)
            for value in DATAFLOWS
            if value != design.dataflow
        )
    if "gemm_split_k" in enabled:
        neighbors.add(replace(design, split_k=not design.split_k))
    if "gemm_assignment_order" in enabled:
        neighbors.add(replace(design, ascending=not design.ascending))
    if "mem_type" in enabled:
        memories = {choice[0] for choice in _package_choices(len(specs))}
        if len(specs) == 1:
            self_constraint = search_space["single_sa_ddr_only"]
            memories = {
                memory for memory in memories if not self_constraint or "ddr" in memory
            }
        neighbors.update(
            replace(design, memory=memory)
            for memory in memories
            if memory != design.memory
        )
    if "interconnect" in enabled and search_space[
        "preserve_package_type_on_interconnect"
    ]:
        prefix = (
            "2.5d_"
            if design.link.startswith("2.5d_")
            else "3d_"
            if design.link.startswith("3d_")
            else None
        )
        for link, protocols in LINKS:
            if prefix is not None and link.startswith(prefix) and link != design.link:
                neighbors.update(
                    replace(design, link=link, protocol=protocol)
                    for protocol in protocols
                )
    if "protocol" in enabled and design.link == "2.5d_emib":
        neighbors.update(
            replace(design, protocol=protocol)
            for protocol in ADVANCED_PROTOCOLS
            if protocol != design.protocol
        )
    neighbors.discard(design)
    return neighbors


class ReducedSpaceNeighborhoodTests(unittest.TestCase):
    def test_every_design_has_the_same_original_and_modular_neighbors(self):
        with open("cfg/parameters/input.json", encoding="utf-8") as file:
            search_space = sequential_gemm_search_space(json.load(file))
        designs = tuple(_designs())

        self.assertEqual(len(designs), 4512)
        for design in designs:
            with self.subTest(design=design):
                self.assertEqual(
                    _modular_neighbors(design, search_space),
                    _original_neighbors(design),
                )


if __name__ == "__main__":
    unittest.main()
