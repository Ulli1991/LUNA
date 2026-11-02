"""PDFs and moments of the turbulent state: local histograms and sums, one psum."""
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from luna.analysis.fields import AXES, primitives, vorticity_divergence
from luna.utils.shmap import shard_map

# Quantity -> default histogram range; samples outside land in the under/overflow bins.
RANGES = {"s": (-12.0, 8.0), "v": (-20.0, 20.0), "mach": (0.0, 40.0),
          "log10_omega": (-4.0, 4.0), "div": (-200.0, 200.0)}


def local_histogram(x, edges, w):
    """Weights summed into len(edges) - 1 bins, underflow first and overflow last (len(edges) + 1)."""
    idx = jnp.searchsorted(edges, x.ravel(), side="right")
    return jnp.zeros(edges.shape[0] + 1, w.dtype).at[idx].add(w.ravel())


def build_stats(mesh, gamma, dx, n_global, bins=128, ranges=None, isothermal=False):
    """Return stats(U) -> dict of replicated arrays: PDFs (volume- and mass-weighted) and moments."""
    rng = {**RANGES, **(ranges or {})}
    shape = tuple(int(mesh.shape[a]) for a in AXES)
    edges = {k: jnp.linspace(lo, hi, bins + 1) for k, (lo, hi) in rng.items()}

    def _stats(U):
        rho, v, cs = primitives(U, gamma, isothermal)
        omega, div = vorticity_divergence(v, dx, shape, n_global)
        tot = lambda x: jax.lax.psum(jnp.sum(x), AXES)                       # noqa: E731
        n = float(n_global) ** 3
        mass = tot(rho)
        rho_mean = mass / n
        s = jnp.log(rho / rho_mean)
        v2 = jnp.sum(v * v, axis=0)
        mach = jnp.sqrt(v2) / cs
        w_vol = jnp.ones_like(rho)
        w_mass = rho / rho_mean
        q = {"s": s, "mach": mach, "div": div,
             "log10_omega": jnp.log10(jnp.sqrt(jnp.sum(omega * omega, axis=0)) + 1e-300)}
        out = {"edges_" + k: e for k, e in edges.items()}
        for k, x in q.items():
            out["pdf_vol_" + k] = jax.lax.psum(local_histogram(x, edges[k], w_vol), AXES) / n
            out["pdf_mass_" + k] = jax.lax.psum(local_histogram(x, edges[k], w_mass), AXES) / n
        out["pdf_vol_v"] = sum(jax.lax.psum(local_histogram(v[i], edges["v"], w_vol), AXES)
                               for i in range(3)) / (3.0 * n)
        # raw moments 1..4 of s and of each velocity component, volume- and mass-weighted
        out["moments_vol_s"] = jnp.stack([tot(s ** p) / n for p in range(1, 5)])
        out["moments_mass_s"] = jnp.stack([tot(w_mass * s ** p) / n for p in range(1, 5)])
        out["moments_vol_v"] = jnp.stack([tot(v[i] ** p) / n for i in range(3) for p in range(1, 5)])
        out["mass"] = mass
        out["momentum"] = jnp.stack([tot(U[1 + i]) for i in range(3)])
        out["kinetic_energy"] = tot(0.5 * rho * v2)
        out["mach_rms_vol"] = jnp.sqrt(tot(mach ** 2) / n)
        out["mach_rms_mass"] = jnp.sqrt(tot(w_mass * mach ** 2) / n)
        out["enstrophy"] = tot(0.5 * jnp.sum(omega * omega, axis=0)) / n
        out["div_rms"] = jnp.sqrt(tot(div ** 2) / n)
        out["rho_min"] = jax.lax.pmin(jnp.min(rho), AXES)
        out["rho_max"] = jax.lax.pmax(jnp.max(rho), AXES)
        return out

    spec = P(None, *AXES)
    return jax.jit(shard_map(_stats, mesh=mesh, in_specs=(spec,), out_specs=P()))
