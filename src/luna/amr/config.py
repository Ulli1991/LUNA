"""Immutable AMR configuration: a frozen, hashable dataclass usable as a jit
static argument. All geometry derives from ``box``, ``base_grid`` and
``block_shape``; refinement ratio is 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class AMRConfig:
    """Static AMR geometry. ``block_shape`` is the same at every level (finer
    levels have smaller dx, not more cells) and must divide ``base_grid``;
    ``n_ghost`` must cover the hydro stencil (MUSCL-Hancock slopes need 2)."""

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
    # audit C8: an unphysical Hancock face state takes the cell's state (counted, logged) instead of NaN (refused)
    allow_first_order_faces: bool = False
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
    # Child-init prolongation limiter for newly refined blocks. "minmod"
    # (default, conservative + monotone, safe at shocks) is 1st order at smooth
    # extrema; "centered" is conservative unlimited centered-linear = 2nd order at
    # smooth extrema (for smooth-flow refinement) but may overshoot at shocks.
    prolong_limiter: str = "minmod"
    reflux: str = "repl"          # coarse-fine reflux mode, from Config.amr_reflux ('repl' | 'sharded' | 'iface')

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
        if self.boundary == 2:
            raise NotImplementedError(
                "AMR boundary=2 (reflective) is not implemented: the halo fill "
                "treats it as zero-gradient (mass leaves through the wall)")
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
