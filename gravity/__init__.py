"""Self-gravity Poisson solver via multigrid V-cycle.

Phase 1 (single-process): core V-cycle, restriction, prolongation,
Jacobi smoother, periodic + Neumann halo helpers. Validated on Jeans
instability + uniform sphere.

Multi-process shard_map integration in Phase 2.
"""

from gravity.poisson_multigrid import solve_poisson, vcycle

__all__ = ["solve_poisson", "vcycle"]
