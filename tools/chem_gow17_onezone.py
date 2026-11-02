"""One-zone network 5, fixes="chemcool" vs "gow17", on the 168 reference-grid conditions (CPU).

    python tools/chem_gow17_onezone.py <net5_trajectories.npz> <out.npz> [--zeta 3e-17] [--procs 48]

scipy BDF on the JAX rhs (jacobian from jacfwd); T_dust is carried between outputs as the Fortran does.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_F = {}


def _funcs(mode):
    """Jitted (rhs, jacobian, aux) of network 5 in one mode, per process."""
    if mode not in _F:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from luna.chemistry import rates_net5
        f = lambda y, p: rates_net5.rhs(y, p, fixes=mode)  # noqa: E731
        f(jnp.array([1e-3, 1e-4, 1e-8, 1e-12]), rates_net5.make_params(1.0, 1.7))  # fill table caches
        _F[mode] = (jax.jit(lambda y, p: f(y, p)[0]), jax.jit(jax.jacfwd(lambda y, p: f(y, p)[0])),
                    jax.jit(lambda y, p: f(y, p)[1]))
    return _F[mode]


def run_case(args):
    """Integrate one case in one mode; returns (outputs [nt, 6] = x_H2, x_H+, x_CO, e, T, T_dust, ok)."""
    mode, prm, fixed, times, y0, zeta = args
    import jax.numpy as jnp
    from scipy.integrate import solve_ivp

    from luna.chemistry import rates_net5
    fun, jac, auxf = _funcs(mode)
    n, _, g0, nh, nh2, nco = prm
    tdust = float(fixed["tdust0"])

    def params(td):
        return rates_net5.make_params(n, g0, nh, nh2, nco, zeta=zeta, dgr=fixed["dgr"],
                                      abundc=fixed["abundc"], abundo=fixed["abundo"], tdust=td)
    y = np.array(y0, float)
    atol = np.array([1e-14, 1e-14, 1e-16, 1e-4 * 1.5 * 1.380649e-16 * 5.0 * n])
    out = np.zeros((len(times), 6))
    ok = True
    for k, t1 in enumerate(times):
        p = params(tdust)
        if k > 0 and ok:
            r = solve_ivp(lambda t, u: np.asarray(fun(jnp.asarray(u), p)), (times[k - 1], t1), y,
                          method="BDF", jac=lambda t, u: np.asarray(jac(jnp.asarray(u), p)),
                          rtol=1e-6, atol=atol)
            ok = bool(r.success)
            y = r.y[:, -1]
        a = auxf(jnp.asarray(y), p)
        tdust = float(a["tdust"])
        out[k] = [*y, float(a["T"]), tdust]
    return out, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--zeta", type=float, default=None)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    d = np.load(a.ref)
    fixed = {k[6:]: float(d[k]) for k in d.files if k.startswith("fixed_")}
    zeta = fixed["zeta"] if a.zeta is None else a.zeta
    names = list(d["case_names"])
    jobs = []
    for mode in ("chemcool", "gow17"):
        for i, nm in enumerate(names):
            tr = d["traj_" + nm]
            jobs.append((mode, d["case_params"][i], fixed, tr[:, 0], tr[0, 1:5], zeta))
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(run_case, jobs, chunksize=1))
    nc = len(names)
    np.savez_compressed(a.out, case_names=np.array(names), case_params=d["case_params"],
                        times=d["traj_" + names[0]][:, 0], zeta=zeta,
                        chemcool=np.stack([r[0] for r in res[:nc]]),
                        gow17=np.stack([r[0] for r in res[nc:]]),
                        ok_chemcool=np.array([r[1] for r in res[:nc]]),
                        ok_gow17=np.array([r[1] for r in res[nc:]]),
                        columns=np.array(["xH2", "xHp", "xCO", "e", "T", "Tdust"]))
    print("failed:", sum(not r[1] for r in res), "of", len(res))


if __name__ == "__main__":
    main()
