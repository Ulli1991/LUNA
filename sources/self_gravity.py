"""Runner-side wiring for the self-gravity source term: each step solves
∇²Φ = 4πGρ via multigrid FMG/V-cycle, takes g = -∇Φ (centred difference),
then kicks momentum (ρv += dt·ρ·g) and energy (E += dt·(ρv)_mid·g).
Mirrors the setup_external_gravity API; auto-selects the shard_map path
when the mesh spans more than one device.
"""

import jax
import jax.numpy as jnp

from gravity.poisson_multigrid import (
    _max_levels,
    solve_poisson_fmg,
    solve_poisson_fmg_shardmap,
)
from gravity.poisson_diff import solve_poisson_fmg_diff
from gravity.poisson_bc import make_halo_3d


def setup_self_gravity(cfg, sharding, is_main_process: bool = True):
    """Return {"apply": fn(U, dt) -> U} or None when disabled."""
    if not getattr(cfg, "SELF_GRAVITY", False):
        return None
    if cfg.SPATIAL_DIMENSION != 3:
        raise ValueError("SELF_GRAVITY currently requires SPATIAL_DIMENSION=3.")
    # Poisson boundary independent of hydro boundary.  Default to hydro's.
    pb = getattr(cfg, "POISSON_BOUNDARY", None)
    poisson_boundary = int(cfg.BOUNDARY if pb is None else pb)
    if poisson_boundary not in (0, 1, 2, 3):
        raise NotImplementedError(
            f"POISSON_BOUNDARY={poisson_boundary} not supported; "
            "use 0 (periodic), 1/2 (Neumann), or 3 (isolated multipole-Dirichlet)."
        )

    dx = cfg.Lx / cfg.Nx
    fourpiG = 4.0 * jnp.pi * getattr(cfg, "GRAVITATIONAL_G", 1.0)
    n_levels = getattr(cfg, "POISSON_N_LEVELS", None)
    # FMG hyperparameters: one V-cycle per level on the walk-up is enough
    # for discretisation-accuracy in one pass; the optional n_extra_vcycles
    # adds insurance for hard problems.
    n_vcycles_per_level = getattr(cfg, "POISSON_FMG_VCYCLES_PER_LEVEL", 1)
    n_extra_vcycles = getattr(cfg, "POISSON_FMG_EXTRA_VCYCLES", 0)
    n_pre = getattr(cfg, "POISSON_N_PRE", 2)
    n_post = getattr(cfg, "POISSON_N_POST", 2)
    n_coarsest = getattr(cfg, "POISSON_N_COARSEST", 50)
    boundary = poisson_boundary
    # Use the sharded (ppermute) Poisson path whenever gravity spans more than
    # one device; a single device keeps the local halo path. Auto-detected from
    # the mesh — no config flag needed.
    _mesh = getattr(sharding, "mesh", None)
    use_shardmap = _mesh is not None and _mesh.size > 1

    halo_fn_local = make_halo_3d(boundary)  # for single-process path
    inv_2dx = 0.5 / dx

    # n_levels: bound by the per-shard side (each shard must still have at
    # least 4 cells per dim at the coarsest level we recurse to).
    if n_levels is None:
        n_levels = _max_levels(cfg.Nx)
    n_levels_static = int(n_levels)

    # For multi-process: a halo_fn that uses ppermute (built per-boundary
    # below). The gradient-of-phi step also needs cross-shard data when
    # phi is sharded, so we wrap solve + gradient + kick in one shard_map.
    if use_shardmap:
        from functools import partial
        from jax.experimental.shard_map import shard_map
        from jax.sharding import PartitionSpec as P

        from gravity.poisson_bc_shardmap import make_scalar_halo_3d_shardmap
        from gravity.poisson_multigrid import fmg_solve, vcycle

        halo_factory_sm = make_scalar_halo_3d_shardmap(sharding.mesh)
        halo_fn_sm = halo_factory_sm(boundary)
        mesh = sharding.mesh
        scalar_spec = P("x", "y", "z")
        vec_spec = P(None, "x", "y", "z")

    def _kick_body(U, phi, dt, halo_fn):
        rho = U[0]
        mx, my, mz = U[1], U[2], U[3]
        phi_pad = halo_fn(phi)
        gx = -(phi_pad[2:, 1:-1, 1:-1] - phi_pad[:-2, 1:-1, 1:-1]) * inv_2dx
        gy = -(phi_pad[1:-1, 2:, 1:-1] - phi_pad[1:-1, :-2, 1:-1]) * inv_2dx
        gz = -(phi_pad[1:-1, 1:-1, 2:] - phi_pad[1:-1, 1:-1, :-2]) * inv_2dx
        new_mx = mx + dt * rho * gx
        new_my = my + dt * rho * gy
        new_mz = mz + dt * rho * gz
        mx_mid = 0.5 * (mx + new_mx)
        my_mid = 0.5 * (my + new_my)
        mz_mid = 0.5 * (mz + new_mz)
        new_E = U[-1] + dt * (mx_mid * gx + my_mid * gy + mz_mid * gz)
        U_new = U.at[1].set(new_mx)
        U_new = U_new.at[2].set(new_my)
        U_new = U_new.at[3].set(new_mz)
        U_new = U_new.at[-1].set(new_E)
        return U_new

    if use_shardmap:
        @partial(jax.jit, static_argnames=())
        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(vec_spec, P()),
            out_specs=vec_spec,
        )
        def _apply_inner(U_local, dt):
            # Inside shard_map. Subtract mean of rho across the global mesh
            # (Jeans swindle) by averaging across all three mesh axes.
            rho_local = U_local[0]
            if boundary == 0:
                rho_mean_local = jnp.mean(rho_local)
                # Single all-reduce mean across all three mesh axes.
                rho_mean = jax.lax.pmean(rho_mean_local, ("x", "y", "z"))
                rho_src_local = rho_local - rho_mean
                # Debug: surface the per-shard rho_mean_local + global rho_mean
                # so we can compare against the single-process answer.
                jax.debug.print(
                    "[self_grav] rho_mean_local={l} rho_mean_global={g} "
                    "max_rho_src_local={s}",
                    l=rho_mean_local, g=rho_mean,
                    s=jnp.max(jnp.abs(rho_src_local)),
                )
            else:
                rho_src_local = rho_local
            rhs_local = fourpiG * rho_src_local
            phi_local = fmg_solve(
                rhs_local,
                dx,
                halo_fn_sm,
                n_levels_static,
                n_pre,
                n_post,
                n_coarsest,
                n_vcycles_per_level,
            )
            for _ in range(n_extra_vcycles):
                phi_local = vcycle(
                    phi_local,
                    rhs_local,
                    dx,
                    halo_fn_sm,
                    n_levels_static - 1,
                    n_pre,
                    n_post,
                    n_coarsest,
                )
            # sum(phi * rho_src) < 0 → Φ opposite-signed to ρ_src → attractive (correct).
            # sum(phi * rho_src) > 0 → Φ same-signed → repulsive (wrong).
            corr = jnp.sum(phi_local * rho_src_local)
            jax.debug.print(
                "[self_grav] sum(phi*rho_src)={c} max|phi|={p} max|rho_src|={r}",
                c=corr, p=jnp.max(jnp.abs(phi_local)),
                r=jnp.max(jnp.abs(rho_src_local)),
            )
            return _kick_body(U_local, phi_local, dt, halo_fn_sm)
    else:
        @jax.jit
        def _apply_inner(U, dt):
            rho = U[0]
            if boundary == 0:
                rho_src = rho - jnp.mean(rho)
            else:
                rho_src = rho
            # Differentiable Poisson: forward result is bit-identical to
            # solve_poisson_fmg (same tuning forwarded), but reverse-mode uses
            # the implicit adjoint (one extra solve) instead of unrolling every
            # V-cycle — the latter OOMs through a multi-step inversion at 256³.
            # custom_vjp nondiff args must be POSITIONAL.
            phi = solve_poisson_fmg_diff(
                rho_src,
                dx,
                fourpiG,
                boundary,
                n_vcycles_per_level,
                n_extra_vcycles,
                n_levels,
                n_pre,
                n_post,
                n_coarsest,
            )
            return _kick_body(U, phi, dt, halo_fn_local)

    def _apply(U, dt):
        return _apply_inner(U, dt)

    if is_main_process:
        print(
            f"✓ Self-gravity (multigrid V-cycle): "
            f"G={getattr(cfg, 'GRAVITATIONAL_G', 1.0):.3e}, "
            f"BC={ {0: 'periodic', 1: 'Neumann', 2: 'Neumann', 3: 'isolated/multipole-Dirichlet'}.get(boundary, '?') }, "
            f"FMG vcycles/level={n_vcycles_per_level}, extra={n_extra_vcycles}"
        )

    return {"apply": _apply}
