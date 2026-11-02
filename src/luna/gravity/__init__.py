"""Self-gravity: multigrid Poisson solvers (periodic, Neumann, isolated) and particle-mesh gravity.

See docs/ARCHITECTURE.md.
"""

from luna.gravity.poisson_multigrid import solve_poisson, vcycle

__all__ = ["solve_poisson", "vcycle"]
