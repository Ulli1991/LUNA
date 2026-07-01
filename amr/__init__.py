"""LUNA block-structured AMR (dimension-generic, JAX).

See DESIGN.md for the architecture and locked decisions. Public surface grows
as the layers land; for now the geometry core and config.
"""

from amr.config import AMRConfig
from amr import geometry

__all__ = ["AMRConfig", "geometry"]
