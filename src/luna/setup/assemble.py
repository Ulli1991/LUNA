"""Build the physics of a run in one place: assemble(cfg, mesh, kind, layout, init) -> Physics
(hydro, clock, gravity, stages, projections); the parts and the four switches: docs/ARCHITECTURE.md.
"""
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import jax
import jax.numpy as jnp

from luna.setup.gravity_energy import resolve_gravity_energy
from luna.setup.clock import make_clock
from luna.setup.hydro import Hydro, build_hydro
from luna.setup.mesh import DENSE
from luna.setup.run_kind import COUPLED, DM
from luna.setup.step_timing import label_methods

# the gravity backend's calls the step timing and the profiler show as their own parts
GRAVITY_PARTS = ("source", "solve", "grad", "kick_gas", "kick_gas_work", "close_flux", "kick_drift", "gather",
                 "close_kick", "exchange", "resort")


@dataclass
class Physics:
    kind: str
    layout: str
    mesh: Any
    clock: Any
    hydro: Optional[Hydro] = None
    gravity: Any = None
    stages: List[Any] = field(default_factory=list)
    projections: List[Any] = field(default_factory=list)
    cooling: Any = None


def build_gravity(cfg, mesh, kind, layout, clock, is_main=True):
    """The gravity stage a config asks for, or None."""
    from luna.gravity import kick_pallas
    kernel = ("xla" if getattr(cfg, "autodiff_mode", False)
              else str(getattr(cfg, "gravity_kernel", "xla")))
    kick_pallas.select(kernel)               # rejects a bad value
    from luna.gravity.backend import DELTA, PROPER, LocalGravity, PoissonOpts, ShardedGravity
    from luna.gravity.kdk import Gravity
    from luna.setup.cosmo_common import gravity_precision_for, softening_from_config
    has_dm = kind in (DM, COUPLED)
    gas = kind != DM and (cfg.SELF_GRAVITY or cfg.sinks_enabled or clock.comoving)
    if not (has_dm or gas or cfg.sinks_enabled):
        return None
    if cfg.SPATIAL_DIMENSION != 3:
        raise ValueError("gravity needs SPATIAL_DIMENSION=3")
    pb = cfg.POISSON_BOUNDARY
    boundary = int(cfg.BOUNDARY if pb is None else pb)
    if boundary not in (0, 1, 2, 3):
        raise NotImplementedError(f"POISSON_BOUNDARY={boundary}: 0 periodic, 1/2 Neumann, "
                                  "3 isolated multipole-Dirichlet")
    comoving = clock.comoving
    # both clocks warm-start on the carried psi: dm_warm_vcycles (comoving), gravity_warm_vcycles (proper)
    prec = gravity_precision_for(cfg, comoving)
    opts = PoissonOpts.from_config(
        cfg, cfg.dm_warm_vcycles if comoving else cfg.gravity_warm_vcycles)
    mode = DELTA if comoving else PROPER
    fourpiG = 4.0 * 3.141592653589793 * float(cfg.GRAVITATIONAL_G)
    N3 = (int(cfg.Nx), int(cfg.Ny), int(cfg.Nz))
    box = float(cfg.Lx)
    dx = box / N3[0]
    if layout == DENSE and mesh.size == 1:
        from luna.gravity.particle_mesh import ParticleMeshGravity
        # the adjoint tasks differentiate through the solve: the FMG whose
        # reverse pass is the implicit adjoint (luna/gravity/poisson_diff.py)
        backend = LocalGravity(ParticleMeshGravity(cfg), box, opts, prec, mode, fourpiG,
                               adjoint=bool(cfg.autodiff_mode), kernel=kernel)
    else:
        backend = ShardedGravity(mesh, N3, dx, box, boundary, opts, prec, mode, fourpiG,
                                 adjoint=bool(cfg.autodiff_mode), kernel=kernel)

    label_methods(backend, "gravity", GRAVITY_PARTS)
    species = []
    if has_dm:
        from luna.species.dm import DarkMatter
        species.append(DarkMatter(
            backend, box=box, dx=dx, eta=cfg.dm_eta,
            eps_soft=softening_from_config(cfg, dx), prec=prec,
            capacity_factor=cfg.dm_capacity_factor,
            face_capacity_factor=cfg.dm_face_capacity_factor,
            resort_every=cfg.dm_resort_every, is_main=is_main))
    if cfg.sinks_enabled:
        from luna.species.sinks import particles_periodic
        if backend.sharded:
            from luna.species.sinks_sharded import ShardedSinks
            make_sinks = lambda **kw: ShardedSinks(backend.mesh, **kw)   # noqa: E731
        else:
            from luna.species.sinks import Sinks
            make_sinks = Sinks
        # the sinks read `boundary` as periodic-or-not plus a div v halo, which
        # knows 0/1/2: isolated gravity (3) is a non-periodic, edge-halo box
        species.append(make_sinks(
            N3=N3, box3=(cfg.Lx, cfg.Ly, cfg.Lz), G=cfg.GRAVITATIONAL_G,
            boundary=1 if boundary == 3 else boundary,
            n_max=cfg.sink_max_count, softening=cfg.sink_softening,
            sink_sink=getattr(cfg, "sink_sink_gravity", True),
            gamma=cfg.gamma,
            accretion_radius=cfg.sink_accretion_radius,
            creation_density=cfg.sink_creation_density,
            creation_separation=cfg.sink_creation_separation,
            create_frac=cfg.sink_create_frac,
            accretion=getattr(cfg, "sink_accretion", "threshold"),
            creation=getattr(cfg, "sink_creation_checks", "luna"),
            merging=getattr(cfg, "sink_merging", False),
            periodic_particles=particles_periodic(cfg), is_main=is_main))
    energy = resolve_gravity_energy(cfg)
    if is_main and gas:
        what = "comoving" if comoving else f"G={cfg.GRAVITATIONAL_G:.3e}"
        print(f"✓ Gravity ({'sharded' if backend.sharded else 'local'} FMG, {what}, "
              f"BC={boundary}, {prec}, warm={opts.warm_cycles}, energy={energy} [{cfg.gravity_energy}])", flush=True)
    return Gravity(backend, clock, species, gas, is_main=is_main,
                   traced=bool(cfg.autodiff_mode), energy=energy)


def build_stages(cfg, hydro, gravity, init, is_main=True, clock=None):
    """The operator-split terms, in step order."""
    from luna.sources.cc85 import setup_cc85_source
    from luna.sources.comoving_energy import setup_comoving_energy
    from luna.sources.cooling import setup_cooling
    from luna.sources.external_gravity import setup_external_gravity
    from luna.sources.ou_forcing import setup_ou_forcing
    from luna.sources.particles import setup_particles
    from luna.sources.radiation import setup_radiation
    from luna.sources.stages import (CellLocal, Cooling, Forcing, Orbits, Radiation, Wind,
                                     injection_rho_min)
    stages, cooling = [], None
    if hydro is None:
        return ([gravity] if gravity is not None else []), None
    sharding = hydro.sharding
    from luna.sources.chemistry import setup_chemistry
    chem = setup_chemistry(cfg, hydro, init, is_main)
    if chem is not None:                     # first, so its `after` closes the step
        stages.append(chem)
    extg = setup_external_gravity(cfg, sharding, is_main)
    if extg is not None:
        stages.append(CellLocal("external_gravity", extg["apply"]))
    comoving = setup_comoving_energy(cfg, clock, is_main)
    if comoving is not None:
        stages.append(comoving)
    cc85 = setup_cc85_source(cfg, sharding, is_main)
    if cc85 is not None:
        rho_min = injection_rho_min(init.U, cc85["sphere_mask"], sharding.mesh)
        dxs = (cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz)[:cfg.SPATIAL_DIMENSION]
        wind = Wind(cc85, gamma=cfg.gamma, dxs=dxs, CFL=cfg.CFL, rho_min=rho_min)
        if is_main:
            print(f"  wind injection bounds the step at dt <= {wind.dt_max:.3e}", flush=True)
        stages.append(wind)
    solver = setup_cooling(cfg, is_main)
    if solver is not None:
        cooling = Cooling(solver, cfg)
        stages.append(cooling)
    rad = setup_radiation(cfg, sharding, is_main)
    if rad is not None:
        stages.append(Radiation(rad))
    orbits = setup_particles(cfg, sharding, is_main)
    if orbits is not None:
        stages.append(Orbits(orbits))
    if gravity is not None:
        stages.append(gravity)
    ou = setup_ou_forcing(cfg, sharding, is_main)
    if ou is not None:
        stages.append(Forcing(ou))
    if cfg.sil_weights:                      # last, so its `after` follows the hydro
        stages.append(build_corrector(cfg, sharding.mesh, is_main))
    return stages, cooling


def build_corrector(cfg, mesh, is_main=True, params=None, meta=None):
    """The learned flux-correction stage: cfg.sil_weights, or `params` with `meta` = dict(rho0, c, amp)."""
    from luna.sources.ml_corrector import load_corrector3d, make_corrector3d, param_count
    from luna.sources.stages import Corrector
    if cfg.SPATIAL_DIMENSION != 3 or cfg.BOUNDARY != 0:
        raise NotImplementedError("the SiL corrector: 3D periodic boxes only")
    if params is None:
        params, meta = load_corrector3d(cfg.sil_weights)
    apply = make_corrector3d(mesh, dx=cfg.Lx / cfg.Nx, rho0=meta["rho0"], c=meta["c"],
                             gamma=float(cfg.gamma), depth=len(params), amp=meta["amp"])
    if is_main:
        print(f"✓ SiL flux corrector: {param_count(params)} parameters", flush=True)
    return Corrector(apply, params)


def _named(name, fn):
    """`fn` with the part name the step timing and the profiler show for it."""
    fn.name = name
    return fn


def build_projections(cfg, clock, gravity, kind, init, mesh, is_main=True):
    from luna.sources.eos_projection import barotropic_project, isothermal_project
    projections = []
    if kind == DM:
        return projections
    if cfg.isothermal_cs2 > 0.0:
        cs2, gamma = float(cfg.isothermal_cs2), float(cfg.gamma)
        projections.append(_named("isothermal", lambda U, a: isothermal_project(U, cs2, gamma)))
    if cfg.barotropic_eos:
        projections.append(_named("barotropic", lambda U, a: barotropic_project(U, cfg)))
    # The same floor stack in proper and comoving runs. On a proper clock a = 1
    # and 4 pi G rho_bar takes the place of the comoving 1.5 Om H0^2, so the
    # Jeans floor is Truelove's P >= G rho^2 (N dx)^2 / pi.
    floors_on = (cfg.cosmo_p_floor > 0.0 or cfg.cosmo_jeans_ncell > 0.0
                 or cfg.cosmo_T_floor > 0.0 or cfg.cosmo_rho_floor > 0.0)
    if floors_on:
        from luna.sources.cosmo_floors import CosmoFloors
        if gravity is not None:
            rho_bar = float(gravity.backend.grid_mean(gravity.backend.component(init.U, 0)))
        else:
            rho_bar = float(jax.jit(lambda U: jnp.mean(U[0]))(init.U))
        coeff_fac = (clock.coeff_fac if clock.comoving
                     else 4.0 * math.pi * float(cfg.GRAVITATIONAL_G) * rho_bar)
        floors = CosmoFloors(
            gamma=cfg.gamma, coeff_fac=coeff_fac, dx=cfg.Lx / cfg.Nx, rho_bar=rho_bar,
            p_floor=cfg.cosmo_p_floor, jeans_ncell=cfg.cosmo_jeans_ncell,
            T_floor=cfg.cosmo_T_floor, T_max=cfg.cosmo_Tmax, v_max=cfg.cosmo_vmax,
            mu=cfg.cosmo_mu, rho_floor_frac=cfg.cosmo_rho_floor,
            floor_buffer=cfg.cosmo_floor_buffer, mesh=mesh,
            periodic=int(cfg.BOUNDARY) == 0)
        if floors.active:
            projections.append(floors)
            if is_main:
                print(f"✓ {'Comoving' if clock.comoving else 'Proper'} {floors.describe()}", flush=True)
    return projections


def assemble(cfg, mesh, kind, layout, init, is_main=True):
    """Everything the loop calls, built once."""
    clock = make_clock(cfg)
    hydro = None if kind == DM else build_hydro(cfg, mesh, is_main)
    gravity = build_gravity(cfg, mesh, kind, layout, clock, is_main)
    stages, cooling = build_stages(cfg, hydro, gravity, init, is_main, clock)
    projections = build_projections(cfg, clock, gravity, kind, init, mesh, is_main)
    return Physics(kind=kind, layout=layout, mesh=mesh, clock=clock, hydro=hydro,
                   gravity=gravity, stages=stages, projections=projections, cooling=cooling)
