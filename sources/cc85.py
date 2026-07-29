"""Runner-side wiring for the CC85 central-injection source: deposits
Mdot = eta_M*SFR and Edot = eta_E*(SFR/100 Msun)*1e51 erg/yr uniformly
inside a sphere of radius R_*. The sphere mask is sharded like U and is
PASSED AS A JIT ARGUMENT (not a closure variable) so XLA cannot inline
the sharded multi-process array as an MLIR constant.
"""

import jax
import jax.numpy as jnp
import numpy as _np
from jax.sharding import NamedSharding, PartitionSpec

_SEC_PER_YR = 3.15576e7
_MSUN_G = 1.98892e33
_PC_CM = 3.08568e18


def setup_cc85_source(cfg, sharding, is_main_process: bool = True):
    """Return {"apply": fn(U, dt) -> U} or None when disabled."""
    if not getattr(cfg, "cc85_enabled", False):
        return None
    if not cfg.units.is_enabled:
        raise ValueError("cc85_enabled requires cfg.units (Arepo system).")

    Mdot_cgs = cfg.cc85_eta_M * cfg.cc85_SFR_Msun_per_yr * _MSUN_G / _SEC_PER_YR
    Edot_cgs = cfg.cc85_eta_E * cfg.cc85_SFR_Msun_per_yr / 100.0 * 1.0e51 / _SEC_PER_YR
    R_star_cgs = cfg.cc85_R_star_pc * _PC_CM

    Mdot_code = Mdot_cgs / cfg.units.UnitMass_in_g * cfg.units.UnitTime_in_s
    Edot_code = Edot_cgs / cfg.units.UnitEnergy_in_cgs * cfg.units.UnitTime_in_s
    R_star_code = R_star_cgs / cfg.units.UnitLength_in_cm
    V_sphere_code = (4.0 / 3.0) * jnp.pi * R_star_code**3

    dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
    cx, cy, cz = 0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz
    if jax.process_count() > 1:
        from utils.distributed import compute_local_slice

        xs, ys, zs = compute_local_slice(cfg.Nx, cfg.Ny, sharding.mesh, jax.process_index(), cfg.Nz)
        x_loc = (_np.arange(xs[0], xs[1], dtype=_np.float32) + 0.5) * dx - cx
        y_loc = (_np.arange(ys[0], ys[1], dtype=_np.float32) + 0.5) * dy - cy
        z_loc = (_np.arange(zs[0], zs[1], dtype=_np.float32) + 0.5) * dz - cz
        Xl, Yl, Zl = _np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")
        mask_local = ((Xl * Xl + Yl * Yl + Zl * Zl) < R_star_code**2).astype(_np.float32)
        coord_sharding = NamedSharding(sharding.mesh, PartitionSpec("x", "y", "z"))
        mask_on_dev = jax.device_put(mask_local, jax.local_devices()[0])
        sphere_mask = jax.make_array_from_single_device_arrays(
            shape=(cfg.Nx, cfg.Ny, cfg.Nz), sharding=coord_sharding, arrays=[mask_on_dev]
        )
    else:
        x_full = (jnp.arange(cfg.Nx, dtype=jnp.float32) + 0.5) * dx - cx
        y_full = (jnp.arange(cfg.Ny, dtype=jnp.float32) + 0.5) * dy - cy
        z_full = (jnp.arange(cfg.Nz, dtype=jnp.float32) + 0.5) * dz - cz
        X, Y, Z = jnp.meshgrid(x_full, y_full, z_full, indexing="ij")
        sphere_mask = ((X * X + Y * Y + Z * Z) < R_star_code**2).astype(jnp.float32)
        if cfg.SPATIAL_DIMENSION == 3:
            coord_sharding = NamedSharding(sharding.mesh, PartitionSpec(*sharding.spec[1:]))
            sphere_mask = jax.device_put(sphere_mask, coord_sharding)

    rho_dot_code = Mdot_code / V_sphere_code
    e_dot_code = Edot_code / V_sphere_code

    @jax.jit
    def _apply_inner(U, dt, mask):
        drho = rho_dot_code * dt * mask
        dE = e_dot_code * dt * mask
        U_new = U.at[0].add(drho)
        U_new = U_new.at[-1].add(dE)
        return U_new

    def _apply(U, dt):
        return _apply_inner(U, dt, sphere_mask)

    # Per-unit-loading rates: Mdot ∝ eta_M, Edot ∝ eta_E (linear), so the
    # differentiable M82 driver can take gradients w.r.t. (eta_M, eta_E) by
    # calling apply_eta with eta as a runtime arg. apply_eta(U, dt, (cfg_etaM,
    # cfg_etaE)) reproduces _apply exactly.
    rho_dot_per_etaM = rho_dot_code / cfg.cc85_eta_M
    e_dot_per_etaE = e_dot_code / cfg.cc85_eta_E

    def _apply_eta(U, dt, eta):
        drho = eta[0] * rho_dot_per_etaM * dt * sphere_mask
        dE = eta[1] * e_dot_per_etaE * dt * sphere_mask
        return U.at[0].add(drho).at[-1].add(dE)

    if is_main_process:
        n_inj = int(jnp.sum(sphere_mask)) if jax.process_count() == 1 else -1
        print(
            f"✓ CC85 wind source: R*={cfg.cc85_R_star_pc:.1f} pc, "
            f"SFR={cfg.cc85_SFR_Msun_per_yr} Msun/yr, "
            f"eta_M={cfg.cc85_eta_M}, eta_E={cfg.cc85_eta_E}"
        )
        print(f"  Mdot = {Mdot_cgs:.3e} g/s = {Mdot_code:.3e} code/code_t")
        print(f"  Edot = {Edot_cgs:.3e} erg/s = {Edot_code:.3e} code")
        print(f"  Injection cells: {n_inj}")

    return {"apply": _apply, "apply_eta": _apply_eta, "sphere_mask": sphere_mask}
