"""Driven, self-gravitating ISM box with the network-5 chemistry stage: per-stage cost per step.
    python tools/chem_coupled/run_turb_chem.py --n 512 --out DIR [--wall-h 23.5] [--restart]
"""
import argparse
import json
import math
import os
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna import units as un  # noqa: E402
from luna.ode import rodas3 as ig  # noqa: E402
from luna.chemistry import networks  # noqa: E402
from luna.chemistry import thermo  # noqa: E402

from luna.hydro.turbulence import OUDriverConfig  # noqa: E402
from luna.sources import chemistry as ch  # noqa: E402
from luna.ic_generation import driven_turbulence_3d  # noqa: E402
from luna.template_config import Config  # noqa: E402

YR = un.SEC_PER_YEAR
MYR = un.SEC_PER_MEGAYEAR
UNITS = dict(UnitLength_in_cm=un.PARSEC_CGS, UnitMass_in_g=un.SOLAR_MASS_CGS,
             UnitVelocity_in_cm_per_s=1e5)
BOX_PC = 100.0
N_H0 = 1.0
MACH = 2.0
G0, ZETA = 1.7, 3e-17


def equilibrium(n_h=N_H0, t0=8000.0, box_cm=BOX_PC * un.PARSEC_CGS, gamma=5.0 / 3.0):
    """One-zone network-5 equilibrium at n_H with the stage's local column: (x_H2, x_H+, x_CO, e, T, T_dust)."""
    from luna.chemistry import tables as tb
    mu_h = (1.0 + 4.0 * tb.ABHE) * tb.MH
    rho = n_h * mu_h
    x = np.array([0.0, 1e-2, 0.0])
    e = 1.5 * (1.1 + x[1] + ch.ABUNDC) * n_h * tb.KBOLTZ * t0
    y = np.array([*x, e])[None]
    tdust, hist = jnp.array([10.0]), []
    c = type("C", (), dict(g0=G0, zeta=ZETA))()
    for _ in range(400):
        cs2 = gamma * (gamma - 1.0) * y[0, 3] / rho
        l = min(math.sqrt(math.pi * cs2 / (tb.GRAVITY * rho)), box_cm)
        cols = tuple(jnp.array([v]) for v in (n_h * l, y[0, 0] * n_h * l, y[0, 2] * n_h * l))
        r = ig.integrate(jnp.asarray(y), ch.params(c, jnp.array([n_h]), cols, tdust), 10 * MYR, net=networks.NET5)
        ynew = np.asarray(r.y)
        tdust = r.params.tdust
        rel = np.max(np.abs(ynew - y) / np.maximum(np.abs(y), 1e-30) * (np.abs(ynew) > 1e-20))
        y = ynew
        hist.append(rel)
        if rel < 1e-6 and len(hist) > 3:
            break
    temp = temperature(jnp.asarray(y[:, :3]), jnp.asarray(y[:, 3]), jnp.array([n_h]))
    return y[0], float(temp[0]), float(tdust[0]), len(hist)


def temperature(x, e_cgs, n_h):
    """Gas temperature from network 5's own e(T) table."""
    x_e = x[..., 1] + jnp.maximum(0.0, ch.ABUNDC - x[..., 2])
    ekn = thermo.ekn_from_energy(e_cgs, x[..., 0], x_e, n_h)
    return thermo.temperature_gamma(x[..., 0], ekn)[0]


def make_config(n, eq):
    """The ISM box in pc / Msun / km/s, FP64, minmod, HLLC, Pallas, periodic gravity."""
    u = un.UnitSystem(**UNITS)
    from luna.chemistry import tables as tb
    rho0 = N_H0 * (1.0 + 4.0 * tb.ABHE) * tb.MH / u.UnitDensity_in_cgs
    p0 = (2.0 / 3.0) * eq[3] / u.UnitPressure_in_cgs
    cs0 = math.sqrt(5.0 / 3.0 * p0 / rho0)
    v_rms = MACH * cs0
    t_ac = 0.5 * BOX_PC / v_rms
    ou = OUDriverConfig(autocorrelation_time=t_ac, rms_acceleration=v_rms / t_ac, seed=7,
                        Lx=BOX_PC, Ly=BOX_PC, Lz=BOX_PC)
    cfg = replace(
        Config(), test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=n, Ny=n, Nz=n, Lx=BOX_PC, Ly=BOX_PC, Lz=BOX_PC, gamma=5.0 / 3.0,
        problem=driven_turbulence_3d.Params(rho0=rho0, P0=p0), units=u, ORDER=1, LIMITER=0, riemann_solver="hllc",
        hydro_kernel="pallas", ou_config=ou, ou_seed=7,
        SELF_GRAVITY=True, GRAVITATIONAL_G=u.GravityConstant, POISSON_BOUNDARY=0,
        chemistry="net5", chem_g0=G0, chem_zeta=ZETA, chem_x0=tuple(float(v) for v in eq[:3]),
        chem_tdust0=10.0, t_end=1e9, PLOT_AFTER_SIM=False,
        enable_x64=True, enable_checkpointing=False)
    return cfg, dict(rho0=rho0, p0=p0, cs0_kms=cs0, v_rms_kms=v_rms, t_ac=t_ac,
                     t_unit_s=u.UnitTime_in_s)


def build(cfg):
    """Mesh, ICs and physics exactly as luna.main.run builds them."""
    from luna.setup.assemble import assemble
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, True)
    return mesh, init, assemble(init.cfg, mesh, kind, layout, init, True)


def timed_step(state, dt, t_new, physics, sync):
    """evolve.step with a device sync after every stage: (state, {stage: seconds})."""
    tm = {}

    def run(name, f):
        t0 = time.perf_counter()
        out = f()
        sync(out)
        tm[name] = tm.get(name, 0.0) + time.perf_counter() - t0
        return out
    h = 0.5 * dt
    for st in physics.stages:
        state = run(st.name, lambda: st.before(state, h, dt))
    hyd = physics.hydro
    carry = state.sources[hyd.scalars]
    U1, S1 = run("hydro", lambda: hyd.step(state.U, carry["S"], dt))
    state = state.update(U=U1, t=t_new, a=physics.clock.scale(t_new)).with_source(
        hyd.scalars, {**carry, "S": S1})
    for st in reversed(physics.stages):
        state = run(st.name, lambda: st.after(state, h, dt))
    return state.update(step=state.step + 1), tm


@jax.jit
def substep_stats(ns, nr, fl, lo):
    """Histogram of accepted substeps (capped at 2000), multi-attempt / failed / floored counts."""
    hist = jnp.bincount(jnp.minimum(ns, 2000).reshape(-1), length=2001)
    return hist, jnp.sum((ns + nr) > 1), jnp.sum(fl), jnp.sum(nr), jnp.sum(lo)


def hist_summary(hist):
    """Mean, median, p90, p99, max of a histogram over 0..2000."""
    c = np.cumsum(hist)
    tot = c[-1]
    q = [int(np.searchsorted(c, f * tot)) for f in (0.5, 0.9, 0.99)]
    return dict(mean=float((hist * np.arange(hist.size)).sum() / tot), p50=q[0], p90=q[1],
                p99=q[2], max=int(np.nonzero(hist)[0].max()))


def make_phase(consts, mesh):
    """Jitted (n_H, T) histograms, volume- and mass-weighted, log n in [-3, 5], log T in [0, 8]."""
    from jax.sharding import PartitionSpec as P
    spec = P(None, "x", "y", "z")

    def nt(U, S):
        y0, n_h, _, _, _ = ch.cell_inputs(U, S, consts)
        return jnp.log10(n_h), jnp.log10(temperature(y0[..., :3], y0[..., 3], n_h))
    f = ch.shard_map_jit(lambda U, S: jnp.stack(nt(U, S)), mesh, (spec, spec), spec)

    @jax.jit
    def hist(U, S):
        ln, lt = f(U, S)
        i = jnp.clip(((ln + 3.0) / 8.0 * 160).astype(jnp.int32), 0, 159)
        j = jnp.clip((lt / 8.0 * 160).astype(jnp.int32), 0, 159)
        k = (i * 160 + j).reshape(-1)
        vol = jnp.bincount(k, length=160 * 160)
        mass = jnp.bincount(k, weights=U[0].reshape(-1), length=160 * 160)
        return vol.reshape(160, 160), mass.reshape(160, 160), jnp.max(ln), jnp.min(lt), \
            jnp.max(lt)
    return hist


def save_state(path, state, extra):
    """U, the chemistry carry, the OU carry, t and step to a directory of .npy files."""
    os.makedirs(path + ".tmp", exist_ok=True)
    np.save(f"{path}.tmp/U.npy", np.asarray(state.U))
    for k, v in state.sources["chemistry"].items():
        np.save(f"{path}.tmp/chem_{k}.npy", np.asarray(v))
    a, key = state.sources["forcing"]
    np.save(f"{path}.tmp/ou_a.npy", np.asarray(a))
    np.save(f"{path}.tmp/ou_key.npy", np.asarray(key))
    with open(f"{path}.tmp/meta.json", "w") as fh:
        json.dump(dict(t=float(state.t), step=int(state.step), **extra), fh)
    if os.path.exists(path):
        os.rename(path, path + ".old")
    os.rename(path + ".tmp", path)
    if os.path.exists(path + ".old"):
        import shutil
        shutil.rmtree(path + ".old")


def load_state(path, state, physics):
    """The inverse of save_state onto the run's shardings, each array freed before its replacement."""
    def put(old, f):
        sh = old.sharding
        old.delete()                      # the same slot again: no fragmentation
        return jax.device_put(np.load(f), sh)
    U = put(state.U, f"{path}/U.npy")
    carry = {k: (put(v, f"{path}/chem_{k}.npy") if os.path.exists(f"{path}/chem_{k}.npy") else v)
             for k, v in state.sources["chemistry"].items()}
    a = jnp.asarray(np.load(f"{path}/ou_a.npy"))
    _, key0 = state.sources["forcing"]
    key = jnp.asarray(np.load(f"{path}/ou_key.npy"), key0.dtype)
    meta = json.load(open(f"{path}/meta.json"))
    return state.update(U=U, t=meta["t"], step=meta["step"]).with_source(
        "chemistry", carry).with_source("forcing", (a, key)), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wall-h", type=float, default=23.5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--ckpt-min", type=float, default=60.0, help="checkpoint every this many minutes")
    ap.add_argument("--snap-myr", type=float, default=2.0, help="analysis states every this many Myr")
    ap.add_argument("--phase-every", type=int, default=50)
    ap.add_argument("--restart", action="store_true")
    a = ap.parse_args()
    t_wall0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    eqf = os.path.join(a.out, "equilibrium.json")
    if os.path.exists(eqf):
        eqd = json.load(open(eqf))
    else:
        y, temp, td, nit = equilibrium()
        eqd = dict(y=[float(v) for v in y], T=temp, tdust=td, chunks_10myr=nit)
        json.dump(eqd, open(eqf, "w"))
    print("equilibrium at n_H = 1:", eqd, flush=True)
    cfg, info = make_config(a.n, eqd["y"])
    cfg = replace(cfg, chem_tdust0=eqd["tdust"])
    info.update(n=a.n, box_pc=BOX_PC, mach=MACH, g0=G0, zeta=ZETA, T0=eqd["T"],
                devices=str(jax.devices()), jax=jax.__version__)
    print("setup:", info, flush=True)
    json.dump(info, open(os.path.join(a.out, "setup.json"), "w"))

    from luna.setup.evolve import initial_state, release_initial, timestep
    mesh, init, physics = build(cfg)
    state = initial_state(init, physics)
    release_initial(init, physics)
    ck = os.path.join(a.out, "checkpoint")
    if a.restart and os.path.exists(ck):
        state, meta = load_state(ck, state, physics)
        print(f"restarted at step {state.step}, t = {state.t:.6g}", flush=True)
    state = physics.gravity.init(state)
    chem_stage = next(s for s in physics.stages if s.name == "chemistry")
    phase = make_phase(chem_stage.consts, mesh)
    t_unit_myr = info["t_unit_s"] / MYR

    def sync(x):
        jax.block_until_ready(x.U if hasattr(x, "U") else x)
    log = open(os.path.join(a.out, "steps.jsonl"), "a")
    next_snap = (math.floor(state.t * t_unit_myr / a.snap_myr) + 1) * a.snap_myr
    t_ckpt = time.time()
    n_done = 0
    while True:
        if time.time() - t_wall0 > a.wall_h * 3600 or (a.max_steps and n_done >= a.max_steps):
            break
        t0 = time.perf_counter()
        dt, t_new = timestep(state, physics, 1e9)
        t_dt = time.perf_counter() - t0
        state, tm = timed_step(state, dt, t_new, physics, sync)
        physics.gravity.check(state)
        c = state.sources["chemistry"]
        hist, multi, failed, rej, floored = (np.asarray(v) for v in substep_stats(
            c["n_steps"], c["n_rejected"], c["failed"], c["floored"]))
        n_done += 1
        rec = dict(step=int(state.step), t_myr=float(state.t) * t_unit_myr,
                   dt_yr=float(dt) * info["t_unit_s"] / YR, t_dt=t_dt, **{f"t_{k}": v for k, v in tm.items()},
                   multi_attempt=int(multi), failed=int(failed), rejected=int(rej),
                   floored=int(floored),
                   **hist_summary(hist), first=n_done == 1)
        if n_done == 1 or state.step % a.phase_every == 0:
            vol, mass, lnmax, ltmin, ltmax = (np.asarray(v) for v in phase(state.U, c["S"]))
            np.savez_compressed(os.path.join(a.out, f"phase_{state.step:06d}.npz"), vol=vol,
                                mass=mass, t_myr=rec["t_myr"], hist=hist)
            rec.update(log_n_max=float(lnmax), log_T_min=float(ltmin), log_T_max=float(ltmax))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if rec["t_myr"] >= next_snap:
            save_state(os.path.join(a.out, f"state_{rec['t_myr']:05.1f}Myr"), state,
                       dict(dt=float(dt), t_myr=rec["t_myr"]))
            next_snap += a.snap_myr
        if time.time() - t_ckpt > a.ckpt_min * 60:
            save_state(ck, state, dict(dt=float(dt)))
            t_ckpt = time.time()
    save_state(ck, state, dict(dt=float(dt)))
    print(f"done: step {state.step}, t = {float(state.t) * t_unit_myr:.3f} Myr", flush=True)


if __name__ == "__main__":
    main()
