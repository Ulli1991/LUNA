"""Kelvin-Helmholtz linear stability from LUNA's tangent propagator.

A smooth periodic double shear layer, u = (dU/2)[tanh((y-Ly/4)/d) -
tanh((y-3Ly/4)/d) - 1], uniform density and pressure, Lx = 2 pi d / 0.4446
(Michalke's most unstable wavenumber), Ly = 2 Lx. The production hydro step
(build_simulate, the same path as bench_autodiff.py) is linearised about it:

  theory     the compressible linear eigenproblem of the same profile (numpy,
             Fourier in y), and its check against Michalke 1964
  eig        |lambda| of M(tau) by forward-mode subspace iteration: modal rate
  svd        optimal perturbation over horizon T, subspace iteration on M^T M
  nonlinear  the NONLINEAR code seeded with the tangent's leading mode
  fair       one application: the tangent against differences of the
             nonlinear code, for a range of perturbation sizes

    python tools/kh_stability.py --nx 128 --base-delta 1e-3 --modes theory,eig,nonlinear,fair
    python tools/kh_stability.py --nx 256 --mesh 2,1 --tangent-eps 1e-3 --json out.json

Always minmod. On the exactly flat base minmod's JVP is the first-order
scheme's; the two remedies (docs/AUTODIFF.md, "Linear stability"):
--base-delta puts a small x- and y-structure into every primitive so no
limiter sits at a = b = 0, --tangent-eps gives minmod a smooth tangent.

Results and discussion: docs/AUTODIFF.md, "Linear stability".
"""
import argparse
import json
import math
import time

import numpy as np

from luna.hydro.exact_kh_linear import (K_MICHALKE, SIGMA_MICHALKE, cell_average_y,  # noqa: F401
                                        geometry, linear_mode, linear_theory, profile)


def theory(g, log=print):
    """The setup's own linear theory, converged in ny, and the Michalke check."""
    out = {}
    for ny in (256, 512, 1024):
        out[ny] = linear_theory(g["k"], g["Ly"], g["d"], g["dU"], g["cs"], g["gamma"], ny)
        log(f"  theory ny={ny}: sigma = {out[ny]:.6f}")
    # Michalke check: one isolated layer (separation 60 d), incompressible limit
    d = 1.0
    s_inc = linear_theory(K_MICHALKE, 120.0, d, 2.0, 200.0, 5.0 / 3.0, 1024)
    log(f"  Michalke check (u=tanh y, k=0.4446, M=0.005): sigma = {s_inc:.5f} "
        f"(Michalke 1964: {SIGMA_MICHALKE})")
    s_lowM = linear_theory(g["k"], g["Ly"], g["d"], g["dU"], 100.0, g["gamma"], 512)
    log(f"  the same double layer at Mach 0.005: sigma = {s_lowM:.6f}")
    sigma_mich = SIGMA_MICHALKE * (0.5 * g["dU"]) / g["d"]
    log(f"  Michalke single layer, incompressible, in these units: {sigma_mich:.6f}")
    return dict(sigma_setup=out[max(out)], sigma_by_ny=out, michalke_check=s_inc,
                sigma_double_layer_low_mach=s_lowM,
                sigma_michalke_incompressible=sigma_mich)


def base_structure(nx, ny, g, delta, seed):
    """delta x two waves per primitive, kx in 3..5 x 2 pi/Lx (k d > 1: no KH
    growth of their own), random ky and phase: (rho, u, v, p) factors/offsets."""
    rng = np.random.default_rng(seed)
    x = (np.arange(nx) + 0.5) * g["Lx"] / nx
    y = (np.arange(ny) + 0.5) * g["Ly"] / ny
    X, Y = np.meshgrid(x, y, indexing="ij")

    def wave():
        out = 0.0
        for _ in range(2):
            kx, ky = rng.integers(3, 6), rng.integers(1, 5)
            out = out + np.sin(2 * np.pi * (kx * X / g["Lx"] + ky * Y / g["Ly"])
                               + rng.uniform(0, 2 * np.pi))
        return 0.5 * delta * out
    return wave(), wave(), wave(), wave()


def build(nx, mesh_shape, horizon, cfl, g, delta=0.0, seed=0, tangent_eps=0.0,
          riemann="hllc"):
    """(prop, U0, dt, n_steps): the production minmod rollout over `horizon`
    about the shear layer (plus `delta` structure), on `mesh_shape`."""
    from dataclasses import replace
    import jax
    import jax.numpy as jnp
    from luna.ic_generation.ic import resolve
    from luna.setup.assemble import assemble
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    from luna.autodiff.rollout import build_simulate
    from luna.template_config import Config

    ny = 2 * nx
    dx = g["Lx"] / nx
    dt_cfl = cfl * dx / (0.5 * g["dU"] + g["cs"])
    n_steps = max(1, math.ceil(horizon / dt_cfl))
    dt = horizon / n_steps
    cfg = replace(Config(), test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2,
                  Nx=nx, Ny=ny, Nz=1, Lx=g["Lx"], Ly=g["Ly"], BOUNDARY=0,
                  gamma=g["gamma"], CFL=cfl, LIMITER=0, riemann_solver=riemann,
                  limiter_tangent_eps=tangent_eps,
                  enable_x64=True, MESH_SHAPE=tuple(mesh_shape), autodiff_mode=True,
                  autodiff_n_steps=n_steps, autodiff_fixed_dt=dt,
                  write_snapshots=False, enable_diagnostics=False,
                  enable_checkpointing=False)
    cfg = resolve(cfg)
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, False)
    physics = assemble(cfg, mesh, kind, layout, init, False)
    simulate, _, _, _, theta = build_simulate(init, cfg, physics, False)

    y = (np.arange(ny) + 0.5) * g["Ly"] / ny
    u = np.broadcast_to(profile(y, g["Ly"], g["d"], g["dU"])[None, :], (nx, ny))
    rho, p, v = np.full((nx, ny), g["rho"]), np.full((nx, ny), g["p"]), np.zeros((nx, ny))
    if delta:
        f_rho, f_u, f_v, f_p = base_structure(nx, ny, g, delta, seed)
        rho, p = rho * (1 + f_rho), p * (1 + f_p)
        u, v = u + g["cs"] * f_u, v + g["cs"] * f_v
    E = p / (g["gamma"] - 1.0) + 0.5 * rho * (u * u + v * v)
    U0 = jax.device_put(jnp.asarray(np.stack([rho, rho * u, rho * v, E])), init.U.sharding)
    return (lambda U: simulate(theta, U)), U0, dt, n_steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--mesh", default="1,1")
    ap.add_argument("--base-delta", type=float, default=0.0,
                    help="amplitude of the x/y structure in the base (0: the flat layer)")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--tangent-eps", type=float, default=0.0,
                    help="Config.limiter_tangent_eps: minmod's smooth tangent (0: off)")
    ap.add_argument("--fair-h", default="1e-4,1e-6,1e-8,1e-10",
                    help="fair: RMS perturbation per cell")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--cs", type=float, default=2.0)
    ap.add_argument("--modes", default="theory,eig,nonlinear,svd")
    ap.add_argument("--tau", type=float, default=0.5, help="eig: horizon of one application")
    ap.add_argument("--eig-iter", type=int, default=40)
    ap.add_argument("--horizons", default="0.5,1,2", help="svd horizons")
    ap.add_argument("--svd-iter", type=int, default=30)
    ap.add_argument("--amp", type=float, default=1e-6, help="nonlinear: RMS seed per cell")
    ap.add_argument("--n-apply", type=int, default=8, help="nonlinear: applications of tau")
    ap.add_argument("--n-vec", type=int, default=4, help="block size of the subspace iteration")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    g = geometry(cs=a.cs)
    res = {"args": vars(a), "geometry": g}
    modes = a.modes.split(",")
    print(f"KH: nx={a.nx} ny={2 * a.nx} d={g['d']:.5f} (d/dx={g['d'] * a.nx:.2f}) "
          f"Mach={0.5 * g['dU'] / g['cs']:.3f} delta={a.base_delta} seed={a.base_seed} "
          f"tangent_eps={a.tangent_eps} mesh={a.mesh}", flush=True)
    if "theory" in modes:
        res["theory"] = theory(g)
    if set(modes) - {"theory"}:
        import jax
        jax.config.update("jax_enable_x64", True)
        from luna.autodiff import stability as st
        mesh = tuple(int(v) for v in a.mesh.split(","))
        res["devices"] = jax.device_count()
        res["backend"] = jax.default_backend()
        log = lambda s: print(s, flush=True)

        kw = dict(delta=a.base_delta, seed=a.base_seed, tangent_eps=a.tangent_eps)
        if {"eig", "nonlinear", "fair"} & set(modes):
            prop, U0, dt, n = build(a.nx, mesh, a.tau, a.cfl, g, **kw)
            drift = float(jax.numpy.max(jax.numpy.abs(jax.jit(prop)(U0) - U0)))
            log(f"eig: tau={a.tau} dt={dt:.4e} steps={n}; base drift over tau {drift:.2e}")
            t0 = time.perf_counter()
            e = st.leading_mode(prop, U0, g["gamma"], n_iter=a.eig_iter, rtol=1e-9,
                                n_vec=a.n_vec, log=log)
            e_s = time.perf_counter() - t0
            sig = st.growth_rate(e["lam"], a.tau)
            log(f"eig: |lambda| = {e['lam']:.8f}  sigma = {sig:.6f}  ({e_s:.1f} s)")
            res["eig"] = dict(lam=e["lam"], sigma=sig, history=e["history"], seconds=e_s,
                              dt=dt, steps=n, base_drift=drift,
                              ritz=[[float(np.real(x)), float(np.imag(x))] for x in e["eigenvalues"]])
            if "nonlinear" in modes:
                eps = a.amp * math.sqrt(U0.size / U0.shape[0])
                amp = st.nonlinear_growth(prop, U0, g["gamma"], e["z"], eps, a.n_apply)
                rates = [math.log(amp[0]) / a.tau] + [
                    math.log(amp[i] / amp[i - 1]) / a.tau for i in range(1, len(amp))]
                log("nonlinear: amplitude " + " ".join(f"{x:.5e}" for x in amp))
                log("nonlinear: rate/tau  " + " ".join(f"{x:.5f}" for x in rates))
                res["nonlinear"] = dict(eps=eps, amp=amp, rates=rates)
            if "fair" in modes:
                cells = math.sqrt(U0.size / U0.shape[0])
                hs = [float(v) for v in a.fair_h.split(",")]
                err = st.tangent_vs_difference(prop, U0, g["gamma"], e["z"],
                                               [h * cells for h in hs])
                log("fair: h/cell " + " ".join(f"{h:.0e}" for h in hs)
                    + " -> |D_h - Mz|/|Mz| " + " ".join(f"{x:.3e}" for x in err))
                res["fair"] = dict(h=hs, err=err)

        if "svd" in modes:
            res["svd"] = []
            for T in (float(v) for v in a.horizons.split(",")):
                prop, U0, dt, n = build(a.nx, mesh, T, a.cfl, g, **kw)
                t0 = time.perf_counter()
                s = st.optimal_perturbation(prop, U0, g["gamma"], n_iter=a.svd_iter,
                                            rtol=1e-9, n_vec=a.n_vec, log=log)
                s_s = time.perf_counter() - t0
                sig = st.growth_rate(s["s"], T)
                log(f"svd: T={T} steps={n} s1 = {s['s']:.6f}  ln(s1)/T = {sig:.6f} "
                    f"({s_s:.1f} s, {len(s['history'])} iterations)")
                res["svd"].append(dict(T=T, s=s["s"], sigma=sig, history=s["history"],
                                       singular_values=[float(x) for x in s["singular_values"]],
                                       seconds=s_s, steps=n))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
