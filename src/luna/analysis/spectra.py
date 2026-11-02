"""Shell-averaged power spectra by a pencil-decomposed 3-D FFT (three all-to-all transposes)."""
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from luna.analysis.fields import AXES, primitives
from luna.utils.shmap import shard_map


def check_pencil(n_global, mesh_shape):
    """Raise unless the pencil transposes divide the grid evenly on this mesh."""
    px, py, pz = mesh_shape
    ny = n_global // py
    bad = [msg for ok, msg in (
        (n_global % px == 0 and n_global % py == 0 and n_global % pz == 0, "N divisible by every mesh axis"),
        (ny % pz == 0, "N / py divisible by pz"),
        (n_global % (py * pz) == 0, "N divisible by py * pz"))
        if not ok]
    if bad:
        raise ValueError(f"spectra: N={n_global} on mesh {mesh_shape} needs " + "; ".join(bad))


def fft3_pencil(x):
    """FFT of a block of P('x','y','z'); the result's local block is (kx: whole, ky over 'x', kz over ('y','z'))."""
    x = jax.lax.all_to_all(x, "z", 1, 2, tiled=True)              # z whole, y split over z
    x = jnp.fft.fft(x, axis=2)
    x = jax.lax.all_to_all(x, ("y", "z"), 2, 1, tiled=True)       # y whole, z over (y, z)
    x = jnp.fft.fft(x, axis=1)
    x = jax.lax.all_to_all(x, "x", 1, 0, tiled=True)              # x whole, y over x
    return jnp.fft.fft(x, axis=0)


def local_wavenumbers(shape_k, n_global, mesh_shape):
    """Integer wavenumbers (kx, ky, kz) of the local spectral block, broadcastable, signed."""
    _, _, pz = mesh_shape
    ix, iy, iz = (jax.lax.axis_index(a) for a in AXES)
    kx = jnp.arange(shape_k[0])
    ky = ix * shape_k[1] + jnp.arange(shape_k[1])
    kz = (iy * pz + iz) * shape_k[2] + jnp.arange(shape_k[2])
    signed = lambda k: jnp.where(k > n_global // 2, k - n_global, k)       # noqa: E731
    return (signed(kx)[:, None, None], signed(ky)[None, :, None], signed(kz)[None, None, :])


def n_shells(n_global):
    """Number of integer shells, k = 0 .. ceil(sqrt(3) N / 2)."""
    return int(3 ** 0.5 * n_global / 2) + 2


def build_spectra(mesh, gamma, box, n_global, single=False):
    """Return spectra(U) -> dict of replicated shell spectra summing to each field's mean square; single: FP32 FFTs."""
    mesh_shape = tuple(int(mesh.shape[a]) for a in AXES)
    check_pencil(n_global, mesh_shape)
    nb = n_shells(n_global)
    k0 = 2.0 * jnp.pi / box
    norm = float(n_global) ** 6

    def _spectra(U):
        rho, v, _ = primitives(U, gamma)
        rho_mean = jax.lax.psum(jnp.sum(rho), AXES) / float(n_global) ** 3
        rdt = jnp.float32 if single else U.dtype
        ctype = jnp.complex64 if rdt == jnp.float32 else jnp.complex128
        F = lambda f: fft3_pencil(f.astype(ctype))                           # noqa: E731
        vk = [F(v[i]) for i in range(3)]
        kx, ky, kz = local_wavenumbers(vk[0].shape, n_global, mesh_shape)
        kk = [kx.astype(rdt), ky.astype(rdt), kz.astype(rdt)]
        k2 = kk[0] ** 2 + kk[1] ** 2 + kk[2] ** 2
        shell = jnp.minimum(jnp.floor(jnp.sqrt(k2) + 0.5).astype(jnp.int32), nb - 1)
        shell = jnp.broadcast_to(shell, vk[0].shape).ravel()

        def bin_(p):
            return jax.lax.psum(jnp.zeros(nb, rdt).at[shell].add(p.ravel()), AXES) / norm

        def a2(z):
            return (z * jnp.conj(z)).real

        # one component at a time: at most the three velocity transforms and two more fields live
        kdotv = kk[0] * vk[0] + kk[1] * vk[1] + kk[2] * vk[2]
        inv_k2 = jnp.where(k2 == 0, 0, 1 / jnp.where(k2 == 0, 1, k2))
        E_v = E_s = E_c = E_w = 0.0
        for i in range(3):
            vc = kk[i] * kdotv * inv_k2
            E_v = E_v + bin_(a2(vk[i]))
            E_c = E_c + bin_(a2(vc))
            E_s = E_s + bin_(jnp.where(k2 == 0, 0, a2(vk[i] - vc)))
            j, m = (i + 1) % 3, (i + 2) % 3
            E_w = E_w + bin_(a2(kk[j] * vk[m] - kk[m] * vk[j]))
        out = {"k": jnp.arange(nb, dtype=rdt), "E_v": E_v, "E_v_solenoidal": E_s,
               "E_v_compressive": E_c, "E_vorticity": E_w * k0 * k0,
               "E_divergence": bin_(a2(kdotv)) * k0 * k0}
        del vk, kdotv
        out["E_rho"] = bin_(a2(F(rho)))
        out["E_lnrho"] = bin_(a2(F(jnp.log(rho / rho_mean))))
        w = rho ** (1.0 / 3.0)
        E_r = 0.0
        for i in range(3):
            E_r = E_r + bin_(a2(F(w * v[i])))
        out["E_rho13v"] = E_r
        return out

    spec = P(None, *AXES)
    return jax.jit(shard_map(_spectra, mesh=mesh, in_specs=(spec,), out_specs=P()))
