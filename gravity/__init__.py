"""Self-gravity Poisson solvers.

Multigrid V-cycle / FMG on the uniform grid (single-device and
shard_map-sharded), red-black Gauss-Seidel smoother, boundaries:
periodic, Neumann/reflective, and isolated (Guillet & Teyssier 2011
multipole-Dirichlet faces). Particle-mesh CIC gravity for DM lives in
particle_mesh.py / particle_mesh_shardmap.py; differentiable FMG
wrappers in poisson_diff.py.
"""

from gravity.poisson_multigrid import solve_poisson, vcycle

__all__ = ["solve_poisson", "vcycle"]
