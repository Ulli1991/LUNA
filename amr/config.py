"""Immutable AMR configuration.

Replaces upstream JAX-AMR's global mutable module state (``set_amr`` setting
``Lx, Ly, n_block, ...`` as module globals). A frozen, hashable dataclass can be
passed as a ``static_argnames`` argument to ``jax.jit`` and closed over by
``shard_map`` without aliasing hazards across multiple configs.

All physical geometry is derived from three things: the domain ``box``, the base
grid ``base_grid``, and the fixed ``block_shape``. Refinement ratio is 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class AMRConfig:
    """Static AMR geometry.

    Parameters
    ----------
    ndim : int
        2 or 3. One code path serves both.
    box : tuple[float, ...]
        Physical domain extent per dim, length ``ndim`` (e.g. ``(L, L)`` /
        ``(L, L, L)``).
    base_grid : tuple[int, ...]
        Base (level-0) cell count per dim, length ``ndim``.
    block_shape : tuple[int, ...]
        Fixed cell count per block per dim, length ``ndim``. Same at every level
        (finer levels have smaller dx, not more cells). Must divide
        ``base_grid``.
    max_levels : int
        Number of refinement levels above the base (0 = uniform base grid only).
    n_ghost : int
        Ghost-cell halo width per face. Must cover the hydro stencil
        (MUSCL-Hancock slopes need 2).
    """

    ndim: int
    box: tuple[float, ...]
    base_grid: tuple[int, ...]
    block_shape: tuple[int, ...]
    max_levels: int = 2
    n_ghost: int = 2
    # Physical boundary condition (LUNA code: 0 periodic, 1 outflow/Neumann,
    # 2 reflective). Non-periodic domains need an explicit BC fill of the
    # domain-edge ghosts (no same-level neighbour, no coarse parent) — otherwise
    # they stay NaN and the unsplit MUSCL-Hancock predictor reads them.
    boundary: int = 0
    # Physics constants for refinement criteria that need them (Truelove/Jeans:
    # lambda_J = sqrt(pi gamma P / (G rho^2))). Pure geometry otherwise; the
    # defaults keep every non-self-gravity caller valid without change.
    gamma: float = 5.0 / 3.0
    grav_G: float = 1.0
    # Static (geometric) refinement region for criterion="static": a per-axis
    # ((lo, hi), ...) box in box-fraction coords [0, 1]. Solution-INDEPENDENT, so
    # the coarse-fine boundary is fixed (SMR convergence tests). None -> central
    # half on every axis. Blocks whose center falls inside are refined.
    static_region: tuple = None
    # Löhner (criterion="lohner") tunables: which conserved variable the normalized
    # 2nd-derivative estimator acts on (0=density) and the eps noise filter that
    # caps the switch where curvature is only discretisation ripple.
    lohner_var: int = 0
    lohner_eps: float = 0.01

    def __post_init__(self):
        nd = self.ndim
        if nd not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2 or 3; got {nd}")
        for name, val in (
            ("box", self.box),
            ("base_grid", self.base_grid),
            ("block_shape", self.block_shape),
        ):
            if len(val) != nd:
                raise ValueError(f"{name} must have length ndim={nd}; got {val}")
        for d in range(nd):
            if self.base_grid[d] % self.block_shape[d] != 0:
                raise ValueError(
                    f"base_grid {self.base_grid} not divisible by block_shape "
                    f"{self.block_shape} on axis {d}"
                )

    # --- derived geometry (cached; config is frozen so this is safe) ---
    @cached_property
    def refinement_ratio(self) -> int:
        return 2

    @cached_property
    def base_blocks(self) -> tuple[int, ...]:
        """Number of level-0 blocks per dim."""
        return tuple(self.base_grid[d] // self.block_shape[d] for d in range(self.ndim))

    @cached_property
    def cells_per_block(self) -> int:
        out = 1
        for s in self.block_shape:
            out *= s
        return out

    @cached_property
    def children_per_block(self) -> int:
        """A refined block splits into ``2**ndim`` children."""
        return self.refinement_ratio ** self.ndim

    def base_dx(self) -> tuple[float, ...]:
        return tuple(self.box[d] / self.base_grid[d] for d in range(self.ndim))

    def level_dx(self, level: int) -> tuple[float, ...]:
        """Cell size per dim at a given refinement level."""
        f = float(self.refinement_ratio ** level)
        return tuple(self.box[d] / self.base_grid[d] / f for d in range(self.ndim))

    def n_blocks_per_dim(self, level: int) -> tuple[int, ...]:
        """Total possible blocks per dim at a level (if fully refined)."""
        f = self.refinement_ratio ** level
        return tuple(self.base_blocks[d] * f for d in range(self.ndim))
