"""Shared failure type for ATLAS graph evaluations.

An unsupported evaluation means the ATLAS graph, evaluation profile, or
architecture uses a capability CarbonPATH does not yet model. It is a hard stop,
not a partial result.
"""


class UnsupportedEvaluation(Exception):
    """Raised when an evaluation uses a capability CarbonPATH does not model."""
