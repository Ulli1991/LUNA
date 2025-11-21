"""
Parallel communication primitives for multi-GPU/multi-node execution.

This module provides:
- Halo exchange (ghost cell communication)
- Global reductions (CFL, diagnostics)
- Parallel I/O (future)
"""

from parallel.halo_exchange import exchange_halos, apply_boundary_conditions
from parallel.collective import global_reduce, global_min, global_max, global_sum

__all__ = [
    'exchange_halos',
    'apply_boundary_conditions',
    'global_reduce',
    'global_min',
    'global_max',
    'global_sum',
]
