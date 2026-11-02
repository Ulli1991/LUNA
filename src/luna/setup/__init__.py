"""Setting up and running a simulation: `luna.main.run` walks mesh.py,
ics.py, assemble.py and evolve.py in that order (docs/ARCHITECTURE.md)."""

from .finalize_distributed import finalize_simulation_distributed
from .hydro import create_solver_distributed
from .ics import gas_grid
from .jax_init import initialize_jax_distributed
from .mesh import create_distributed_mesh

initialize_simulation_distributed = gas_grid     # the old name, for run scripts outside the repo

__all__ = [
    "initialize_jax_distributed",
    "create_distributed_mesh",
    "gas_grid",
    "initialize_simulation_distributed",
    "create_solver_distributed",
    "finalize_simulation_distributed",
]
