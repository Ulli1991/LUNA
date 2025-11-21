"""
Grid abstractions for multi-GPU domain decomposition.

This module provides patch-based grid structures that support:
- Fixed uniform grids (immediate use case)
- Adaptive mesh refinement (future compatibility)
- Multi-GPU domain decomposition
"""

from grid.patch import GridPatch
from grid.domain import Domain
from grid.decomposition import create_uniform_decomposition
from grid.neighbors import setup_neighbor_connectivity

__all__ = [
    'GridPatch',
    'Domain',
    'create_uniform_decomposition',
    'setup_neighbor_connectivity',
]
