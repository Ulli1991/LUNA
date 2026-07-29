"""LUNA block-structured AMR (dimension-generic, JAX).

See amr/DESIGN.md for the architecture. Consumers import the submodules
directly; only ``AMRConfig`` and ``geometry`` are re-exported here.
"""

from amr.config import AMRConfig
from amr import geometry

__all__ = ["AMRConfig", "geometry"]
