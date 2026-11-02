"""The gravity precision policy, the comoving Poisson prefactor and the DM
softening default. The clock lives in luna/setup/clock.py.
"""


def poisson_coeff_fac(Om, H0):
    """Comoving Poisson coefficient prefactor (3/2) Omega_m H0^2.

    The per-step coefficient is ``poisson_coeff_fac(Om, H0) * a``; for
    dimensionless pancake units (H0 = 1) it reduces to (3/2) Omega_m a.
    """
    return 1.5 * float(Om) * float(H0) * float(H0)


class GravityPrecision:
    """Resolved dtypes for one gravity_precision mode (docs/history/PRECISION_SURVEY.md §9 R6).

    mode      particles   deposit accumulator   V-cycle   phi carry / residual
    double    FP64        FP64  (int64 @2^40)   FP64      FP64
    mixed     FP64        FP32  (int32 @2^17)   FP32      FP64
    single    FP32        FP32  (int32 @2^17)   FP32      FP32
    """

    def __init__(self, mode, fixed_deposit=False):
        import jax.numpy as jnp
        if mode not in ("double", "mixed", "single"):
            raise ValueError(f"gravity_precision={mode!r}: "
                             "'double', 'mixed' or 'single'")
        self.mode = mode
        self.particles = jnp.float32 if mode == "single" else jnp.float64
        self.phi = jnp.float32 if mode == "single" else jnp.float64
        # None = "same as the incoming particles" (the deposit's own default).
        self.deposit = None if mode == "double" else jnp.float32
        self.cycle_mixed = mode != "double"   # FP32 cycle under an FP64 carry
        self.fixed_deposit = bool(fixed_deposit)
        self.fixed_int = jnp.int64 if mode == "double" else jnp.int32
        self.fixed_bits = 40 if mode == "double" else 17

    def __repr__(self):
        return (f"GravityPrecision({self.mode}: particles={self.particles.__name__}"
                f" phi={self.phi.__name__} deposit="
                f"{'fixed-' + self.fixed_int.__name__ + f'@2^{self.fixed_bits}' if self.fixed_deposit else (self.deposit.__name__ if self.deposit else 'native')}"
                f" cycle={'fp32' if self.cycle_mixed else 'fp64'})")


def gravity_precision_for(cfg, comoving):
    """gravity_precision comoving; else FP64, or mixed cycles with gravity_proper_mixed."""
    if comoving:
        return gravity_precision_from_config(cfg)
    return GravityPrecision("mixed" if cfg.gravity_proper_mixed else "double")


def gravity_precision_from_config(cfg):
    """cfg.gravity_precision and cfg.gravity_deposit_fixed as a GravityPrecision."""
    return GravityPrecision(str(cfg.gravity_precision), bool(cfg.gravity_deposit_fixed))


def softening_from_config(cfg, dx):
    """DM force softening: cfg.dm_softening, defaulting to the PM cell dx."""
    import builtins
    return builtins.float(cfg.dm_softening) or dx
