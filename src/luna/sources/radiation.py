"""Radiation stage wiring: the initial M1/M2 field and the two half-steps of the `Radiation` stage.

See docs/RADIATION.md ("The stage").
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from luna.radiation.coupling import absorb_emit, apply_force, substeps
from luna.radiation.ionization import (source_weights, transport_and_ionize,
                                       two_temperature)
from luna.time_integration.integrators import cell_dt

SKIP_DARK = True    # skip the substeps of a half step while nothing shines (tests turn it off)
LAG_MAX_CELLS = 2 ** 25     # cells per device up to which the next substep is queued before the last ends


def _per_group(scalar, per_group, G, name):
    """Return one scalar for every group, or a per-group array."""
    if per_group:
        v = tuple(float(x) for x in per_group)
        if len(v) != G:
            raise ValueError(f"{name} has {len(v)} values but rad_groups={G}")
        return jnp.asarray(v)
    return float(scalar)


def _warn_if_launches_are_async(is_main):
    """Warn that a differentiated radiation rollout on ROCm needs serialized launches (docs/AUTODIFF.md)."""
    # Warn only: the variables are read at runtime start, and serializing costs the multigrid 3x.
    import os
    import jax
    if not is_main or jax.default_backend() != "rocm":
        return
    if os.environ.get("HIP_LAUNCH_BLOCKING") or os.environ.get("AMD_SERIALIZE_KERNEL"):
        return
    print("  ⚠ differentiating radiation on ROCm: rollouts of ~8 steps or more "
          "segfault after a couple of gradient calls -- an async-dispatch race "
          "the size of the program triggers, not an autodiff bug. Keep the "
          "rollout short, lower rad_substeps, or export HIP_LAUNCH_BLOCKING=1 "
          "(free for hydro and radiation, 3x on the multigrid; ONE node only: "
          "it deadlocks the first cross-node collective). "
          "docs/AUTODIFF.md has the table.", flush=True)


def _restart_carry(cfg, mesh, ndim):
    """Return the radiation carry stored beside the restart checkpoint, or None."""
    from luna.file_io.checkpoint_utils import checkpoint_dir
    from luna.file_io.radiation_checkpoint import load_radiation_checkpoint
    from luna.file_io.read_checkpoint import find_latest_checkpoint, read_checkpoint

    f = cfg.restart_file or find_latest_checkpoint(checkpoint_dir(cfg))
    if f is None:
        return None
    step = int(read_checkpoint(f)["step"])
    return load_radiation_checkpoint(checkpoint_dir(cfg), step, mesh, ndim)


def _substep_count(h, per_sub, n_sub):
    """Return ceil(h / per_sub) clipped to [1, n_sub]; n_sub when h is traced."""
    if isinstance(h, (int, float, np.floating)):
        return min(n_sub, max(1, int(math.ceil(float(h) / per_sub - 1e-9))))
    return n_sub


def _halo_pad(halo, mesh, boundary, ndim):
    """Return pad(A, g, axes) from the shard_map halo; `pad.local_axes` lists the axes one device holds whole."""
    def pad(A, g, axes=None):
        return halo(A, boundary, ndim, width=g, axes=axes)
    # on these the fused sweeps take their ghosts in-kernel instead (m1_pallas.split_step)
    pad.local_axes = tuple(d for d, a in enumerate(("x", "y", "z")[:ndim]) if mesh.shape[a] == 1)
    return pad


def setup_radiation(cfg, sharding, is_main=True):
    """Return the radiation carry, its half-steps and the step bound, or None when off."""
    if not getattr(cfg, "radiation_enabled", False):
        return None
    ndim = int(cfg.SPATIAL_DIMENSION)
    G = int(cfg.rad_groups)
    chat = float(cfg.rad_chat)
    if G < 1:
        raise ValueError(f"rad_groups={G}: at least one frequency group")
    if chat <= 0.0:
        raise ValueError("radiation_enabled needs rad_chat > 0: the reduced "
                         "speed of light in code units, >~ 10x the fastest "
                         "signal the run has to resolve")
    boundary = int(cfg.BOUNDARY if cfg.rad_boundary is None else cfg.rad_boundary)
    if boundary not in (0, 1):
        raise NotImplementedError(
            f"rad_boundary={boundary}: the M1 transport has periodic (0) and "
            "zero-gradient (1); reflective and diode ghosts would have to "
            "negate the normal FLUX, which no test covers yet")
    n_sub = int(cfg.rad_substeps)
    if n_sub < 1:
        raise ValueError(f"rad_substeps={n_sub}: at least one")
    order = int(cfg.rad_order)
    scheme = str(getattr(cfg, "rad_scheme", "hancock_split"))
    dx = float(cfg.Lx) / int(cfg.Nx)
    # Longest substep: cfl dx/(ndim c_hat) unsplit; 2 cfl dx/c_hat split (x y z z y x at h/2 each).
    per_sub = float(cfg.rad_cfl) * dx / (ndim * chat)
    if scheme == "hancock_split":
        per_sub = 2.0 * float(cfg.rad_cfl) * dx / chat
    elif scheme != "rk2":
        raise ValueError(f"rad_scheme={scheme!r}: 'rk2' or 'hancock_split'")
    dt_max = 2.0 * n_sub * per_sub
    closure = str(getattr(cfg, "rad_closure", "m1"))
    from luna.radiation import m2_stage
    if closure not in m2_stage.CLOSURES:
        raise ValueError(f"rad_closure={closure!r}: one of {m2_stage.CLOSURES}")
    m2 = closure != "m1"
    if m2 and (ndim != 3 or scheme != "hancock_split" or cfg.autodiff_mode):
        raise NotImplementedError(
            f"rad_closure={closure!r} needs SPATIAL_DIMENSION=3 and "
            "rad_scheme='hancock_split', and has no adjoint path yet")
    params = m2_stage.load_params(str(getattr(cfg, "rad_closure_params", ""))) \
        if closure == "m2_learned" else None
    kernel = str(getattr(cfg, "rad_kernel", "xla"))
    if closure == "m2_entropy" and kernel == "pallas":
        raise ValueError("rad_closure='m2_entropy' with rad_kernel='pallas': the fused M2 kernel has "
                         "no entropy solve; rad_kernel='xla' or 'auto' (XLA for this closure)")
    ion_sub = int(getattr(cfg, "rad_ion_subcycles", 1))
    ion_solver = str(getattr(cfg, "rad_ion_solver", "implicit"))
    if ion_solver not in ("split", "implicit"):
        raise ValueError(f"rad_ion_solver={ion_solver!r}: 'split' or 'implicit'")

    kappa_a = _per_group(cfg.rad_kappa_a, cfg.rad_kappa_a_groups, G, "rad_kappa_a_groups")
    kappa_s = _per_group(cfg.rad_kappa_s, cfg.rad_kappa_s_groups, G, "rad_kappa_s_groups")
    kappa_t = kappa_a + kappa_s          # the flux feels both, the energy only kappa_a
    source = _per_group(cfg.rad_source, cfg.rad_source_groups, G, "rad_source_groups")
    c_true = float(cfg.rad_c_true)
    force = bool(cfg.rad_force)
    remat = bool(cfg.rad_remat)
    ionization = bool(cfg.rad_ionization)
    if ionization and G != 1:
        raise NotImplementedError(
            f"rad_ionization with rad_groups={G}: the hydrogen ionization "
            "prescription is grey (one ionizing band), as STARBENCH defines it")
    if ionization and ndim != 3:
        raise ValueError("rad_ionization needs SPATIAL_DIMENSION=3")

    mesh = sharding.mesh
    axes = tuple(sharding.spec)[1:]                     # ("x",), ("x","y"), ...
    from jax.sharding import NamedSharding, PartitionSpec as P
    rspec = P(None, None, *axes)
    rsharding = NamedSharding(mesh, rspec)
    shape = (G, 10 if m2 else 1 + ndim) + tuple(int(n) for n in
                                                (cfg.Nx, cfg.Ny, cfg.Nz)[:ndim])

    @jax.jit
    def _initial():
        R = jnp.zeros(shape)
        if m2:
            R = m2_stage.isotropic(jnp.full((G,) + shape[2:], float(cfg.rad_E0) / G))
        else:
            R = R.at[:, 0].set(float(cfg.rad_E0) / G)
        return jax.lax.with_sharding_constraint(R, rsharding)

    R0 = _initial()

    def _half(U, R, h, pad, reverse, n):
        """Run one half-step, or its mirror image, in `n` transport substeps."""
        def transport_and_force(U, R):
            R, dF = substeps(R, h, dx, chat=chat, kappa=kappa_t, n_sub=n,
                             boundary=boundary, order=order, pad=pad, scheme=scheme,
                             closure=closure, params=params, kernel=kernel)
            if force:
                U, R = apply_force(U, R, dF, chat=chat, c_true=c_true)
            return U, R

        def exchange(U, R):
            return absorb_emit(U, R, h, chat=chat, kappa_a=kappa_a, source=source)

        ops = (exchange, transport_and_force) if reverse else (transport_and_force,
                                                               exchange)
        for op in ops:
            U, R = op(U, R)
        return U, R

    if ionization:
        dxv = (float(cfg.Lx) / int(cfg.Nx), float(cfg.Ly) / int(cfg.Ny),
               float(cfg.Lz) / int(cfg.Nz))
        centre = (0.5 * float(cfg.Lx), 0.5 * float(cfg.Ly), 0.5 * float(cfg.Lz))
        src_r = float(cfg.rad_source_radius) or 2.0 * dxv[0]
        sigma, alpha_B = float(cfg.rad_sigma), float(cfg.rad_alpha_B)
        m_H, rate = float(cfg.rad_m_H), float(cfg.rad_photon_rate)
        cs2_n, cs2_i = float(cfg.rad_cs2_neutral), float(cfg.rad_cs2_ionized)
        table = None
        if cfg.rad_neutral_eos_table:
            from luna.radiation.ionization import load_neutral_table
            table = load_neutral_table(cfg.rad_neutral_eos_table)
        gamma = float(cfg.gamma)
        grid = tuple(int(n) for n in (cfg.Nx, cfg.Ny, cfg.Nz))

        @jax.jit
        def _weights():
            w = source_weights(grid, dxv, centre, src_r)
            return jax.lax.with_sharding_constraint(
                w, NamedSharding(mesh, P(*axes)))

        weight0 = _weights()

        @jax.jit
        def _x0():
            x = jnp.full(grid, float(cfg.rad_x0))
            return jax.lax.with_sharding_constraint(
                x, NamedSharding(mesh, P(*axes)))

        def _substep_ion(U, R, x, w, h, pad):
            """One (inject, transport, ionize) substep; advances (R, x) only, rho read from U."""
            # `w` is a rate density (so rate=1), an argument not a closure: moving sources must not retrace.
            return transport_and_ionize(
                R, x, U[0], h, dx, n_sub=1, chat=chat, sigma=sigma,
                alpha_B=alpha_B, m_H=m_H, weight=w, rate=1.0,
                boundary=boundary, order=order, pad=pad, scheme=scheme,
                ion_sub=ion_sub, ion_solver=ion_solver, closure=closure,
                params=params, kernel=kernel)

    # rad_remat recomputes the half-step in the backward pass instead of storing it (docs/AUTODIFF.md).
    def half(U, R, h, pad, reverse, n):
        f = _half
        if remat:
            f = jax.checkpoint(_half, static_argnums=(3, 4, 5))
        return f(U, R, h, pad, reverse, n)

    def count(h):
        return _substep_count(h, per_sub, n_sub)

    # Dark skip: with no light, no source and x = 0 a half step maps zeros to zeros.
    source_off = not bool(np.any(np.asarray(source) != 0.0))

    @jax.jit
    def _quiet(c):
        if ionization:
            R, x, w = c
            return (jnp.max(jnp.abs(R)) == 0.0) & (jnp.max(x) == 0.0) & (jnp.max(w) == 0.0)
        return jnp.max(jnp.abs(c)) == 0.0

    if ionization:
        _rest = jax.jit(lambda U, c: two_temperature(U, c[1], cs2_neutral=cs2_n,
                                                     cs2_ionized=cs2_i, gamma=gamma,
                                                     neutral_table=table))
    else:
        _rest = None

    # Dispatch one substep ahead (two programs' scratch alive) only where that fits (CODE_NOTES.md).
    cells = int(np.prod(shape[2:])) // mesh.size
    lag = cells <= LAG_MAX_CELLS

    def substep_loop(one, n, args):
        """Return args after n calls of one(*args): wait on each input, or on the previous substep's with `lag`."""
        prev = None
        for _ in range(n):
            # async dispatch holds every queued program's scratch at once (OOM at 512^3/GPU)
            if not lag:
                jax.block_until_ready(args)
            elif prev is not None:
                jax.block_until_ready(prev)
            prev = args
            args = one(*args)
        return args

    def quiet(c, h):
        return SKIP_DARK and (ionization or source_off) \
            and isinstance(h, (int, float, np.floating)) and bool(_quiet(c))

    def dispatch(make):
        """Return run(U, c, h): the half step as n calls of the one-substep program over h/n."""
        # n calls of one program, not a loop inside it: the in-program loop needs more scratch.
        one = make(1)

        def run(U, c, h):
            n = _substep_count(h, per_sub, n_sub)
            if quiet(c, h):
                return (U if _rest is None else _rest(U, c)), c
            hn = h / n
            return substep_loop(lambda U, c: one(U, c, hn), n, (U, c))
        return run

    def dispatch_ion(sub):
        """Return run(U, c, h): n calls of sub(U, R, x, w, h/n) -> (R, x), then the two-temperature EOS once."""
        def run(U, c, h):
            n = _substep_count(h, per_sub, n_sub)
            if quiet(c, h):
                return _rest(U, c), c
            R, x, w = c
            hn = h / n
            R, x = substep_loop(lambda R, x: sub(U, R, x, w, hn), n, (R, x))
            # EOS once: two_temperature reads rho, the momenta and x, and transport only rho
            return _rest(U, (R, x, w)), (R, x, w)
        return run

    if ionization and mesh.size == 1:
        sub = jax.jit(lambda U, R, x, w, h: _substep_ion(U, R, x, w, h, None))

        def build(reverse):
            return dispatch_ion(sub)
    elif ionization:
        from luna.utils.halo_exchange_shardmap import make_halo_shardmap
        from luna.utils.shmap import shard_map
        halo = make_halo_shardmap(mesh, ndim)
        uspec, xspec = P(None, *axes), P(*axes)

        def body(U, R, x, w, h):
            return _substep_ion(U, R, x, w, h,
                                _halo_pad(halo, mesh, boundary, ndim))
        sub = jax.jit(shard_map(body, mesh=mesh, in_specs=(uspec, rspec, xspec, xspec, P()),
                                out_specs=(rspec, xspec)))

        def build(reverse):
            return dispatch_ion(sub)
    elif mesh.size == 1:
        def build(reverse):
            return dispatch(lambda n: jax.jit(lambda U, R, h: half(U, R, h, None, reverse, n)))
    else:
        from luna.utils.halo_exchange_shardmap import make_halo_shardmap
        from luna.utils.shmap import shard_map
        halo = make_halo_shardmap(mesh, ndim)
        uspec = P(None, *axes)

        def build(reverse):
            def make(n):
                def body(U, R, h):
                    return half(U, R, h,
                                _halo_pad(halo, mesh, boundary, ndim), reverse, n)
                return jax.jit(shard_map(body, mesh=mesh,
                                         in_specs=(uspec, rspec, P()),
                                         out_specs=(uspec, rspec)))
            return dispatch(make)

    if cfg.autodiff_mode:
        _warn_if_launches_are_async(is_main)
    # The fused sweep kernel has no derivative: an adjoint run keeps XLA's.
    from luna.radiation import m1_pallas
    if kernel not in ("auto", "xla", "pallas"):
        raise ValueError(f"rad_kernel={kernel!r}: 'auto', 'xla' or 'pallas'")
    m1_pallas.KERNEL = "xla" if cfg.autodiff_mode else kernel

    if is_main:
        groups = "grey" if G == 1 else f"{G} groups"
        what = "ionizing" if ionization else groups
        print(f"✓ Radiation ({closure.upper()}, {what}, c_hat={chat:.3e}, up to {n_sub} substeps/half, "
              f"BC={boundary}{', force on gas' if force and not ionization else ''})\n"
              f"  radiation bounds the step at dt <= {dt_max:.3e}", flush=True)
        if cfg.enable_checkpointing and cfg.checkpoint_wall_time_interval > 0:
            print("  ⚠ checkpoints do not carry the radiation field: a restart "
                  "from one would begin again from rad_E0 (and is refused)",
                  flush=True)

    carry0 = (R0, _x0(), weight0 * rate) if ionization else R0
    if cfg.restart:
        # Refuse when absent: the run would silently restart from rad_E0.
        restored = _restart_carry(cfg, mesh, ndim)
        if restored is None:
            raise FileNotFoundError(
                "restart with radiation: no radiation file set beside the "
                "checkpoint being restarted from. A checkpoint written before "
                "radiation was checkpointable cannot be resumed with "
                "radiation on -- it would silently begin again from rad_E0.")
        carry0 = restored
        if is_main:
            print("  ✓ radiation field restored from its checkpoint set",
                  flush=True)
    dt_ion = _make_dt_ionized(cfg, mesh, axes) if ionization else None
    return {"R0": carry0, "before": build(False), "after": build(True),
            "dt_max": dt_max, "n_sub": n_sub, "count": count, "sharding": rsharding,
            "ionization": ionization, "dt_ion": dt_ion, "finite": _all_finite}


@jax.jit
def _all_finite(c):
    """Return whether every entry of the radiation carry (R, or R, x, w) is finite."""
    ok = jnp.bool_(True)
    for a in jax.tree.leaves(c):
        ok = ok & jnp.all(jnp.isfinite(a))
    return ok


def _make_dt_ionized(cfg, mesh, axes):
    """Return fn(U, carry) -> the hydro CFL step at the ionized sound speed once anything shines, else inf."""
    # The hydro CFL sees the pre-ionization pressure; ionizing mid-step raises it ~10x.
    cs_i = math.sqrt(float(cfg.rad_cs2_ionized))
    CFL = float(cfg.CFL)
    d = (float(cfg.Lx) / int(cfg.Nx), float(cfg.Ly) / int(cfg.Ny), float(cfg.Lz) / int(cfg.Nz))

    def local(U, x, w):
        rho = U[0]
        shining = jnp.max(w) > 0.0
        ionised = jnp.max(x) > 1e-6
        dt = jnp.min(cell_dt([U[1 + k] / rho for k in range(3)], cs_i, d, CFL))
        return dt, shining | ionised

    def bound(dt, on):
        return jnp.where(on, dt, jnp.inf)

    if mesh.size == 1:
        return jax.jit(lambda U, c: bound(*local(U, c[1], c[2])))
    from jax.sharding import PartitionSpec as P
    from luna.utils.shmap import shard_map

    def body(U, x, w):
        dt, on = local(U, x, w)
        return bound(jax.lax.pmin(dt, axes), jax.lax.pmax(on.astype(jnp.int32), axes) > 0)
    f = jax.jit(shard_map(body, mesh=mesh, in_specs=(P(None, *axes), P(*axes), P(*axes)),
                          out_specs=P()))
    return lambda U, c: f(U, c[1], c[2])
