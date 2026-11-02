"""The Newton EOS (thermo.temperature_gamma_newton) against Glover's table: accuracy, convergence,
gradients and cost per cell.

    python tools/chem_eos/validate_newton.py [--cells 1e7] [--a079 energy-table-abhe0079.txt]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna.chemistry import tables as tb  # noqa: E402
from luna.chemistry import thermo  # noqa: E402

XS = np.array([0.1, 0.2, 0.3, 0.4, 0.5])


def stats(name, got, ref, where):
    """Print and return max / median relative error and where the max is."""
    r = np.abs(np.asarray(got) / ref - 1.0)
    k = int(np.argmax(r))
    print(f"  {name}: max {r[k]:.2e} at {where(k)}; median {np.median(r):.2e}")
    return r[k], float(np.median(r))


def on_grid(temp_ref, gamma_ref, a, label):
    """Newton at (x_k, ekn = 1..5e4) against a table's own T and gamma."""
    ekn = np.tile(np.arange(1.0, 50001.0), 5)
    x = np.repeat(XS, 50000)
    t, g = jax.jit(thermo.temperature_gamma_newton)(x, ekn, a)
    where = lambda k: f"x={x[k]:.1f}, ekn={ekn[k]:.0f}, T={temp_ref[k]:.5g}"  # noqa: E731
    print(f"{label} (a = {a}):")
    return stats("T", t, temp_ref, where), stats("gamma", g, gamma_ref, where)


def convergence(n, seed=0):
    """Max |G| and step after N_NEWTON + 1 steps over random (x, ekn, a), atomic edges included."""
    rng = np.random.default_rng(seed)
    ekn = 10 ** rng.uniform(-3, 8, n)
    x = rng.uniform(0, 0.5, n) * rng.choice([1.0, 1e-3, 1e-9, 0.0], n, p=[.7, .1, .1, .1])
    a = rng.choice([0.1, 0.079, 0.0], n)
    t, _ = jax.jit(thermo.temperature_gamma_newton)(x, ekn, a)
    t = np.asarray(t)
    lnt = jnp.log(t)
    phi, _ = thermo.eh2_over_t(lnt)
    ekn_back = (x * np.asarray(phi) * t + 1.5 * (1 + a - 2 * x) * t) / (1 + a - x)
    r = np.abs(ekn_back / ekn - 1)
    print(f"convergence over {n:.0e} random states: max |ekn(T)/ekn - 1| = {r.max():.1e}, "
          f"median {np.median(r):.1e}")
    return r.max()


def gradients():
    """jax.grad of T and gamma in (ekn, x, a) against central differences."""
    pts = [(37.3, 0.31, 0.079), (512.7, 0.05, 0.1), (3.3e3, 0.45, 0.079), (2.4e4, 0.2, 0.1),
           (0.7, 0.3, 0.079), (6e4, 0.1, 0.079), (18.2, 1e-4, 0.1), (9.1e3, 0.5, 0.079)]
    worst = 0.0
    for i in (0, 1):
        f = lambda e, x, a: thermo.temperature_gamma_newton(x, e, a)[i]  # noqa: E731
        g = jax.jit(jax.grad(f, argnums=(0, 1, 2)))
        for e, x, a in pts:
            ad = np.array(g(e, x, a))
            fd = []
            for j, v in enumerate((e, x, a)):
                h = 1e-6 * abs(v)
                args = [e, x, a]
                args[j] = v + h
                hi = float(f(*args))
                args[j] = v - h
                fd.append((hi - float(f(*args))) / (2 * h))
            fd = np.array(fd)
            scale = np.abs(fd * np.array([e, x, a])).max()
            err = np.abs((ad - fd) * np.array([e, x, a])).max() / scale
            worst = max(worst, err)
            print(f"  {'T' if i == 0 else 'gamma'} at ekn={e}, x={x}, a={a}: AD {ad} FD {fd} "
                  f"err {err:.1e}")
    print(f"gradients vs central differences: worst {worst:.1e}")
    return worst


def cost(n, reps=20):
    """Seconds per cell of the table lookup and the Newton solve, jitted, on the default device."""
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.uniform(0, 0.5, n))
    ekn = jnp.asarray(10 ** rng.uniform(0, 4.7, n))
    out = {}
    for name, fn in (("table", lambda x, e: thermo.temperature_gamma_table(x, e)),
                     ("newton", lambda x, e: thermo.temperature_gamma_newton(x, e, 0.079))):
        f = jax.jit(fn)
        jax.block_until_ready(f(x, ekn))
        best = np.inf
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(f(x, ekn))
            best = min(best, time.perf_counter() - t0)
        out[name] = best / n
        print(f"  {name}: {best * 1e3:.2f} ms for {n:.0e} cells = {best / n * 1e9:.3f} ns/cell")
    print(f"cost on {jax.devices()[0]}: newton / table = {out['newton'] / out['table']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=float, default=1e7)
    ap.add_argument("--a079", required=True)
    a = ap.parse_args()
    _, temp, gamma = tb.energy_table()
    on_grid(temp.ravel(), gamma.ravel(), 0.1, "newton vs Glover's table on its grid")
    tl, gl = thermo.temperature_gamma_table(np.repeat(XS, 50000), np.tile(np.arange(1.0, 50001.0), 5))
    where = lambda k: f"k={k}"  # noqa: E731
    print("table lookup vs the table itself (x on the blocks):")
    stats("T", tl, temp.ravel(), where)
    t79 = np.loadtxt(a.a079)
    on_grid(t79[:, 1], t79[:, 2], 0.079, "newton vs the exact rescale to He/H = 0.079")
    print("table lookup (He/H = 0.1) vs the He/H = 0.079 rescale:")
    stats("T", temp.ravel(), t79[:, 1], lambda k: f"x={XS[k // 50000]:.1f}, ekn={k % 50000 + 1}")
    stats("gamma", gamma.ravel(), t79[:, 2], lambda k: f"x={XS[k // 50000]:.1f}, ekn={k % 50000 + 1}")
    convergence(int(a.cells))
    gradients()
    cost(int(a.cells))


if __name__ == "__main__":
    main()
