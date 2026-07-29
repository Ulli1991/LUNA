"""Random fields with prescribed spectral content (autodiff IC synthesis)."""

import jax
import jax.numpy as jnp


def spectral_ic_field(key, shape, k_max=0, amp=0.05, slope=0.0):
    """Random real field with prescribed spectral content, std normalized to
    `amp`. k_max <= 0: broadband power out to Nyquist (the honest input for an
    r(k) recovery test — collapse then measures the solver, not the band
    limit); k_max > 0: legacy band-limited injection at k <= k_max;
    slope: optional power-law shaping |F(k)| ~ k**slope (0 = white)."""
    f = jax.random.normal(key, shape)
    F = jnp.fft.rfftn(f)
    kx = jnp.fft.fftfreq(shape[0]) * shape[0]
    ky = jnp.fft.fftfreq(shape[1]) * shape[1]
    kz = jnp.arange(F.shape[2])
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    kk = jnp.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    if slope != 0.0:
        F = F * jnp.where(kk > 0, kk, 1.0) ** slope
    if k_max and k_max > 0:
        F = jnp.where(kk <= k_max, F, 0.0)
    g = jnp.fft.irfftn(F, s=shape)
    return g / (jnp.std(g) + 1e-12) * amp
