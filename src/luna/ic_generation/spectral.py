"""Random fields with prescribed spectral content (autodiff IC synthesis)."""

import jax
import jax.numpy as jnp


def spectral_ic_field(key, shape, k_max=0, amp=0.05, slope=0.0):
    """Random real field with std `amp` and |F(k)| ~ k**slope, band-limited to k <= k_max if > 0."""
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
