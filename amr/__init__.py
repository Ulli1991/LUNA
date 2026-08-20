"""LUNA block-structured AMR (dimension-generic, JAX).

Built on the **JAX-AMR** framework of Haocheng Wen and Faxuan Luo (MIT):
https://github.com/JA4S/JAX-AMR — https://doi.org/10.48550/arXiv.2504.13750

The approach is theirs: dynamically updated multi-layer blocks at fixed
positions and fixed shapes, JIT- and vmap-compatible throughout, with the
trailing NaN sentinel block marking ghosts that have no neighbour. LUNA
re-implements that bookkeeping dimension-generically; see amr/DESIGN.md for
what we changed and why.

Consumers import the submodules directly; only ``AMRConfig`` and
``geometry`` are re-exported here.
"""

from amr.config import AMRConfig
from amr import geometry

__all__ = ["AMRConfig", "geometry"]
