"""A user's own initial state for luna.launch: an array `U0` or a callable `ic(x, y, z, cfg)`.

Both give the CONSERVED variables, energy last, in cfg's own dimension, as a registered IC does:
(3, Nx) (rho, rho vx, E) in 1D, (4, Nx, Ny) (rho, rho vx, rho vy, E) in 2D, (5, Nx, Ny, Nz)
(rho, rho vx, rho vy, rho vz, E) in 3D; `conserved(rho, v, p, gamma)` builds that from primitives.
`ic` is called with this shard's cell centres (x = (i + 1/2) Lx / Nx, from 0; meshgrid, ij; y, z None
below their dimension) and returns this shard's cells; `U0` is the whole grid and each rank keeps its
slab. Either goes through `initial_conditions`, so main.run shards it as it shards a registered IC.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from luna.ic_generation._grid import local_range

VARIABLES = {1: "(rho, rho vx, E)", 2: "(rho, rho vx, rho vy, E)", 3: "(rho, rho vx, rho vy, rho vz, E)"}


def state_shape(cfg, x_slice=None, y_slice=None, z_slice=None):
    """(nvar, cells per axis) of cfg's conserved state on the slices (the whole grid without)."""
    d = cfg.SPATIAL_DIMENSION
    axes = zip((cfg.Nx, cfg.Ny, cfg.Nz)[:d], (x_slice, y_slice, z_slice)[:d])
    return (d + 2, *(hi - lo for lo, hi in (local_range(n, s) for n, s in axes)))


def cell_centres(cfg, x_slice=None, y_slice=None, z_slice=None):
    """(x, y, z) cell centres of the slices, meshgrid ij in cfg's dimension, None beyond it."""
    d = cfg.SPATIAL_DIMENSION
    axes = zip((cfg.Nx, cfg.Ny, cfg.Nz)[:d], (cfg.Lx, cfg.Ly, cfg.Lz)[:d], (x_slice, y_slice, z_slice)[:d])
    c = [(np.arange(*local_range(n, s)) + 0.5) * (L / n) for n, L, s in axes]
    return tuple(np.meshgrid(*c, indexing="ij")) + (None,) * (3 - d)


def conserved(rho, v, p, gamma):
    """Conserved (rho, rho v..., E) from rho, v (a sequence of 1-3 components) and p."""
    rho, p = np.asarray(rho, np.float64), np.asarray(p, np.float64)
    v = [np.broadcast_to(np.asarray(c, np.float64), rho.shape) for c in v]
    return np.stack([rho, *(rho * c for c in v), p / (gamma - 1.0) + 0.5 * rho * sum(c * c for c in v)])


def _what(cfg):
    d = cfg.SPATIAL_DIMENSION
    grid = ", ".join(f"{a}={getattr(cfg, a)}" for a in ("Nx", "Ny", "Nz")[:d])
    return f"SPATIAL_DIMENSION={d}, {grid} needs {state_shape(cfg)}: {VARIABLES[d]}"


@dataclass(frozen=True)
class UserIC:
    """The user's state: exactly one of U0 (whole grid) and ic (callable per shard)."""
    U0: Optional[Any] = None
    ic: Optional[Callable] = None

    def local(self, cfg, x_slice=None, y_slice=None, z_slice=None):
        """This shard's conserved state, in cfg's precision; a wrong shape or a non-finite value is refused."""
        want = state_shape(cfg, x_slice, y_slice, z_slice)
        if self.ic is not None:
            U = np.asarray(self.ic(*cell_centres(cfg, x_slice, y_slice, z_slice), cfg))
            if U.shape != want:
                raise ValueError(f"ic(x, y, z, cfg) returned shape {U.shape} for this shard's cells; "
                                 f"it must be {want}, {VARIABLES[cfg.SPATIAL_DIMENSION]}")
        else:
            d = cfg.SPATIAL_DIMENSION
            ranges = [local_range(n, s) for n, s in zip((cfg.Nx, cfg.Ny, cfg.Nz)[:d], (x_slice, y_slice, z_slice)[:d])]
            U = np.asarray(self.U0)[(slice(None), *(slice(lo, hi) for lo, hi in ranges))]
        U = np.asarray(U, np.float64 if cfg.enable_x64 else np.float32)
        if not np.isfinite(U).all():
            raise ValueError(f"the initial state has {int((~np.isfinite(U)).sum())} non-finite values")
        return U


def user_ic(cfg, U0=None, ic=None):
    """A UserIC for cfg from launch's U0 / ic, or None without either; whatever cannot hold is refused."""
    if U0 is None and ic is None:
        return None
    if U0 is not None and ic is not None:
        raise ValueError("give U0 or ic, not both")
    clash = [f"{k}={getattr(cfg, k)!r}" for k in ("ic_dir", "restart", "AMR", "disc_damping_enabled")
             if getattr(cfg, k)]
    if clash:
        raise ValueError(f"a user initial state (U0 / ic) cannot run with {', '.join(clash)}: ic_dir and a "
                         "restart are initial states of their own, AMR builds its own hierarchy, disc "
                         "damping relaxes towards the registered IC")
    from luna.setup.run_kind import GAS, run_kind
    if run_kind(cfg) != GAS:
        raise ValueError(f"run={cfg.run!r}: a user initial state (U0 / ic) is a gas grid; this run "
                         "reads MUSIC particles")
    if cfg.SPATIAL_DIMENSION not in VARIABLES:
        raise ValueError(f"SPATIAL_DIMENSION={cfg.SPATIAL_DIMENSION}: expected 1, 2 or 3")
    if ic is not None:
        if not callable(ic):
            raise ValueError(f"ic={ic!r} is not callable: ic(x, y, z, cfg) -> conserved state")
        return UserIC(ic=ic)
    shape = np.shape(U0)
    if shape != state_shape(cfg):
        raise ValueError(f"U0 has shape {shape}; {_what(cfg)}")
    return UserIC(U0=U0)
