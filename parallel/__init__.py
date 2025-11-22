"""
Parallel communication primitives for multi-GPU/multi-node execution.

This module provides:
- Halo exchange (ghost cell communication)
- MPI-based halo exchange for multi-GPU
"""

from parallel.halo_exchange import exchange_halos, apply_boundary_conditions

__all__ = [
    'exchange_halos',
    'apply_boundary_conditions',
]
