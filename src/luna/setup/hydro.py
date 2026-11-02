"""The hydro a run steps with: `build_hydro(cfg, mesh) -> Hydro`, the
shard_map MUSCL-Hancock step(U, dt) and its CFL estimate compute_dt(U), on
the dense (x, y, z) mesh."""
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from luna.setup.gravity_energy import resolve_gravity_energy
from luna.time_integration.integrators import create_muscl_hancock_modular_step
from luna.utils.grid_params import GridParams


@dataclass
class Hydro:
    """The hydro step, its CFL estimate, the grid sharding, the grid."""
    step: Any
    compute_dt: Any
    sharding: Any
    grid: Any
    flops_per_cell: Optional[float] = None
    face_mass: Any = None         # (U, dt, acc) -> acc + (2, 6) face in/outflow; None = off
    scalars: Optional[str] = None  # source whose carry["S"] the step advects: step(U, S, dt)
    n_scalars: int = 0
    step_flux: Any = None         # (U, dt) -> (U, face mass fluxes), gravity_energy='flux'; None = off
    step_report: Any = None       # step with the face report last (integrators._face_report); the loop's
    step_flux_report: Any = None  # step_flux with it; None = the run does not report

    def warmup(self, U, is_main=True):
        """Compile compute_dt and step on the initial state without advancing it (that left a phase offset)."""
        t0 = time.time()
        dt = self.compute_dt(U)
        jax.block_until_ready(dt)
        if self.scalars:
            import jax.numpy as jnp
            S = jnp.zeros((self.n_scalars,) + U.shape[1:], U.dtype, device=U.sharding)
            jax.block_until_ready((self.step_report or self.step)(U, S, dt))
        elif self.step_flux is not None:
            jax.block_until_ready((self.step_flux_report or self.step_flux)(U, dt))
        else:
            jax.block_until_ready((self.step_report or self.step)(U, dt))
        if is_main:
            print(f"✓ Compilation complete ({time.time() - t0:.1f}s)\n", flush=True)
        if os.environ.get("LUNA_FLOP_COUNT", "0") == "1":
            from luna.utils.flop_counter import report_step_cost
            self.flops_per_cell = report_step_cost(self.step, self.compute_dt, U, dt,
                                                   self.grid.total_cells, is_main)
        return dt


def build_hydro(cfg, mesh, is_main=True):
    """The dense MUSCL-Hancock step and CFL estimate on `mesh`."""
    grid = GridParams.from_config(cfg)
    n_scalars = chemistry_scalars(cfg)
    compute_dt, step, sharding = create_solver_distributed(cfg, grid, mesh, n_scalars, face_report=True)
    if n_scalars:
        return Hydro(step=_without_report(step), compute_dt=compute_dt, sharding=sharding, grid=grid,
                     scalars="chemistry", n_scalars=n_scalars, step_report=step)
    step_flux = step_flux_r = None
    if resolve_gravity_energy(cfg) == "flux":
        step_flux_r = _behind_barrier(create_solver_distributed(cfg, grid, mesh, mass_flux=True,
                                                                face_report=True)[1])
        step_flux = _without_report(step_flux_r)
    step = _behind_barrier(step)
    return Hydro(step=_without_report(step), compute_dt=compute_dt, sharding=sharding,
                 grid=grid, face_mass=build_face_mass(cfg, grid, mesh), step_flux=step_flux,
                 step_report=step, step_flux_report=step_flux_r)


def chemistry_scalars(cfg):
    """Passive scalars the chemistry carries through the hydro (0: none)."""
    if getattr(cfg, "chemistry", "") == "net5":
        return 3
    return 0


def hydro_floors(cfg):
    """(smallr, smallc) in code units when cfg.preserve_face_states, else None (RAMSES floors, slopes.ramses_face)."""
    if not getattr(cfg, "preserve_face_states", False):
        return None
    return float(cfg.hydro_smallr), float(cfg.hydro_smallc)


def build_face_mass(cfg, grid, mesh):
    """cfg.boundary_mass_budget: the (2, 6) face in/outflow of one hydro step."""
    if not getattr(cfg, "boundary_mass_budget", False):
        return None
    from luna.time_integration.integrators import create_boundary_mass_fn
    from luna.utils.halo_exchange_shardmap import make_halo_shardmap
    return create_boundary_mass_fn(
        make_halo_shardmap(mesh, cfg.SPATIAL_DIMENSION), cfg.gamma, cfg.ORDER,
        hydro_limiter(cfg), grid.dx, grid.dy, grid.dz, grid.inv_dx, grid.inv_dy,
        grid.inv_dz, cfg.BOUNDARY, cfg.SPATIAL_DIMENSION, mesh=mesh,
        riemann_solver=cfg.riemann_solver, fused=cfg.hydro_fused,
        recon32=cfg.hydro_recon32, floors=hydro_floors(cfg),
        allow_first_order=bool(getattr(cfg, "allow_first_order_faces", False)))


def _behind_barrier(step):
    """The hydro step with its input behind an optimization barrier (docs/ARCHITECTURE.md)."""
    @jax.jit
    def guarded(U, *rest):
        return step(jax.lax.optimization_barrier(U), *rest)
    return guarded


def _without_report(step):
    """The step without its trailing face report: the old return, U or a tuple."""
    @jax.jit
    def plain(*args):
        out = step(*args)[:-1]
        return out[0] if len(out) == 1 else out
    return plain


def hydro_limiter(cfg):
    """cfg.LIMITER, or minmod with a smooth tangent (cfg.limiter_tangent_eps)."""
    eps = float(getattr(cfg, "limiter_tangent_eps", 0.0))
    if eps > 0.0:
        from luna.hydro.slopes import SmoothTangentMinmod
        return SmoothTangentMinmod(eps)
    return cfg.LIMITER


def hydro_kernel_name(cfg, n_scalars=0):
    """The hydro kernel the step runs, 'xla' or 'pallas': cfg.hydro_kernel resolved as the step resolves it."""
    from luna.hydro import muscl_hancock_pallas as mhp
    pallas = mhp.select(str(getattr(cfg, "hydro_kernel", "xla")), mhp.unsupported(
        spatial_dim=cfg.SPATIAL_DIMENSION, order=cfg.ORDER, limiter=hydro_limiter(cfg),
        riemann_solver=cfg.riemann_solver, boundary=cfg.BOUNDARY, fused=cfg.hydro_fused,
        recon32=cfg.hydro_recon32, autodiff=bool(getattr(cfg, "autodiff_mode", False)),
        n_scalars=n_scalars))
    return "pallas" if pallas else "xla"


def create_solver_distributed(
    cfg, grid: GridParams, mesh: Mesh, n_scalars: int = 0, mass_flux: bool = False, face_report: bool = False
) -> Tuple[Callable, Callable, NamedSharding]:
    """(compute_dt(U) -> CFL dt, step(U, dt) -> U, sharding): shard_map MUSCL-Hancock, ppermute halos; mass_flux: step -> (U, face mass fluxes); face_report: the face report last."""
    # Cache process info
    is_main_process = jax.process_index() == 0 and not mass_flux

    # Create NamedSharding for automatic sharding
    if cfg.SPATIAL_DIMENSION == 1:
        sharding = NamedSharding(mesh, P(None, "x"))
    elif cfg.SPATIAL_DIMENSION == 2:
        sharding = NamedSharding(mesh, P(None, "x", "y"))
    else:  # 3D
        sharding = NamedSharding(mesh, P(None, "x", "y", "z"))

    # shard_map is the single multi-GPU path: per-shard ppermute halos for every dimension
    # and boundary; only thin slabs cross the wire, and reverse mode flows through them
    from luna.utils.halo_exchange_shardmap import make_halo_shardmap

    pad_halo_fn = make_halo_shardmap(mesh, cfg.SPATIAL_DIMENSION)
    if is_main_process:
        print("Using shard_map with explicit per-shard ppermute halos")
        print(f"Sharding spec: {sharding.spec}")

    # The MUSCL-Hancock step: compute_dt_fn(U) -> dt_cfl (pure CFL, the runner
    # clamps it) and step_fn(U, dt) -> U_new.
    compute_dt_fn, step_fn = create_muscl_hancock_modular_step(
        pad_halo_fn,
        cfg.gamma,
        cfg.ORDER,
        hydro_limiter(cfg),
        grid.dx,
        grid.dy,
        grid.dz,
        grid.inv_dx,
        grid.inv_dy,
        grid.inv_dz,
        cfg.BOUNDARY,
        cfg.SPATIAL_DIMENSION,
        cfg.CFL,
        sharding=sharding,
        riemann_solver=cfg.riemann_solver,
        use_shardmap=True,
        mesh=mesh,
        fused=cfg.hydro_fused,
        recon32=cfg.hydro_recon32,
        kernel=str(getattr(cfg, "hydro_kernel", "xla")),
        autodiff=bool(getattr(cfg, "autodiff_mode", False)),
        n_scalars=n_scalars,
        mass_flux=mass_flux,
        floors=hydro_floors(cfg),
        allow_first_order=bool(getattr(cfg, "allow_first_order_faces", False)),
        face_report=face_report,
    )
    integrator_name = (
        f"MUSCL-Hancock ({cfg.riemann_solver.upper()}, "
        f"{hydro_kernel_name(cfg, n_scalars)} kernel, shard_map)"
    )

    if is_main_process:
        print(f"Using {integrator_name} time integration")
        print("Runner controls dt clamping (snapshot times, t_end)")

    return compute_dt_fn, step_fn, sharding
