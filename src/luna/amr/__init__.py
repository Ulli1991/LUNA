"""LUNA block-structured AMR, dimension-generic, re-implementing the design of JAX-AMR
(Wen & Luo, https://github.com/JA4S/JAX-AMR, arXiv:2504.13750; MIT, see LICENSE-JAX-AMR).
What LUNA changed: amr/DESIGN.md."""

from luna.amr.config import AMRConfig
from luna.amr import geometry

__all__ = ["AMRConfig", "geometry"]
