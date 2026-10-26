"""Policy for the new FPGA chiplet that the ``fpga_count`` move adds.

The search space names the policy in ``fpga.new_chiplet``. The default is
``copy_existing_v1``.

* ``copy_existing_v1``: a copy of an FPGA chiplet of the package, chosen at
  random. With no FPGA in the package, it builds one from the search space.
* ``from_search_space_v1``: a chiplet drawn from the search space (board,
  frequency, ReLU implementation, input buffer), even when the package already
  has FPGA chiplets.
"""

from __future__ import annotations

import copy

from system.utils.Rule4MlBoards import board_chiplet_fields
from system.utils.UnsupportedEvaluation import UnsupportedEvaluation

DEFAULT_NEW_CHIPLET_POLICY = "copy_existing_v1"


def _from_search_space(params, rng):
    fpga_space = params["fpga"]
    chiplet = {
        "chiplet_type": "fpga",
        "tech_node": str(rng.choice(params["tech_nodes"])),
        "frequency_hz": rng.choice(fpga_space["frequency_hz"]),
        "relu_implementation": copy.deepcopy(
            rng.choice(fpga_space["relu_implementation"])
        ),
        **board_chiplet_fields(rng.choice(fpga_space["boards"])),
    }
    if fpga_space.get("input_buffer_blocks"):
        chiplet["input_buffer_blocks"] = rng.choice(fpga_space["input_buffer_blocks"])
    return chiplet


def copy_existing_v1(architecture, fpga_keys, params, rng):
    if fpga_keys:
        return copy.deepcopy(architecture[rng.choice(fpga_keys)])
    return _from_search_space(params, rng)


def from_search_space_v1(architecture, fpga_keys, params, rng):
    return _from_search_space(params, rng)


NEW_CHIPLET_POLICIES = {
    "copy_existing_v1": copy_existing_v1,
    "from_search_space_v1": from_search_space_v1,
}


def new_fpga_chiplet(architecture, fpga_keys, params, rng):
    """Build the chiplet that the search space's ``fpga.new_chiplet`` names."""
    policy_id = params["fpga"].get("new_chiplet", DEFAULT_NEW_CHIPLET_POLICY)
    try:
        policy = NEW_CHIPLET_POLICIES[policy_id]
    except (KeyError, TypeError):
        raise UnsupportedEvaluation(f"unknown new FPGA chiplet policy '{policy_id}'")
    return policy(architecture, fpga_keys, params, rng)
