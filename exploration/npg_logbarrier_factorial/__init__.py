"""Isolated NPG × temporary categorical log-barrier experiments.

The package imports existing project objects read-only and never modifies
``vpg/`` or previous experiment archives.
"""

from .natural_step import NaturalStepResult, target_kl_natural_step

__all__ = ["NaturalStepResult", "target_kl_natural_step"]

