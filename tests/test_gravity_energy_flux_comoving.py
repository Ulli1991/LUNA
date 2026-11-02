"""gravity_energy='flux' on a comoving clock (super-comoving variables, Martel & Shapiro 1998): the face work on
Phibar = sbar psibar (s = 3/2 Omega_m H0^2 a, sbar and psibar step averages) makes the discrete Layzer-Irvine
statement d(E~ + W~) = (ds / 2)(W~0 / s0 + W~1 / s1) hold per step to the Poisson solve's tolerance, with W~ =
1/2 s sum (rho~ - <rho~>) psi dV from LUNA's own psi (the trapezoid of W~ d ln a, Layzer 1963, Irvine 1961). Gated
on the top-hat (tests/test_cosmo_tophat_collapse.py), the Zel'dovich pancake's adiabatic temperature T a^2 = K
rho~^(gamma - 1) and gas + dark matter."""
import functools
from dataclasses import replace

import numpy as np
import pytest

from tests import test_cosmo_tophat_collapse as TH
from tests._helpers import accuracy_cfg, check, gpu_only


def ledger(cfg, U0, a_end, dm=None, snaps=()):
    """Per-step Layzer-Irvine ledger through the production loop: a dict with the max over steps of |E~ + W~ - E~0 -
    W~0 - sum_n (ds / 2)(W~n / sn + W~n+1 / sn+1) - booked| / |W~| ('li'), the energy the projections (floors) and
    the comoving-energy stage put in (booked) and the positivity limiter moved, over |W~| at the end, the final U, W~,
    and U at each scale factor of `snaps` (landed on exactly; 'snaps': {a: U}).
    With dm (Particles), E~ includes K~_dm and the particles' own budget sum dK_dm + sum drho_dm Phibar dV is
    booked too ('own': its max / |W~|)."""
    import jax
    import jax.numpy as jnp

    from luna.setup.assemble import assemble
    from luna.setup.clock import make_clock
    from luna.setup.evolve import initial_state, step, timestep
    from luna.setup.ics import Initial
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind

    clock = make_clock(cfg)
    tau_end = clock.tau_of_a(a_end)
    targets = sorted([(clock.tau_of_a(a), a) for a in snaps if a < a_end]) + [(tau_end, a_end)]
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    physics = assemble(cfg, mesh, kind, layout, Initial(cfg=cfg, U=np.asarray(U0), dm=dm), is_main=False)
    grav = physics.gravity
    dV = (cfg.Lx / cfg.Nx) ** 3
    added = [0.0]

    def total_e(U):
        return float(jnp.sum(U[-1])) * dV

    def tracked(fn):
        def f(*a):
            out = fn(*a)
            U_in, U_out = a[0], out
            if hasattr(out, "U"):                       # a stage half: (state, h, dt) -> state
                U_in, U_out = a[0].U, out.U
            added[0] += total_e(U_out) - total_e(U_in)
            return out
        return f
    physics.projections[:] = [tracked(p) for p in physics.projections]
    for st in physics.stages:
        if st.name == "comoving_energy":
            st.before, st.after = tracked(st.before), tracked(st.after)
    state = initial_state(Initial(cfg=cfg, U=jax.device_put(jnp.asarray(U0), physics.hydro.sharding), dm=dm),
                          physics)
    state = grav.init(state)

    def parts(state):
        rho = state.U[0]
        if dm is None:
            return rho, 0.0, None
        d = state.species["dm"]
        act = jnp.asarray(d.active) > 0
        kdm = 0.5 * float(jnp.sum(jnp.where(act, jnp.asarray(d.mass) * jnp.sum(jnp.asarray(d.p) ** 2, axis=1), 0.0)))
        rdm = grav.deposit_of(state, "dm")
        return rho + rdm, kdm, rdm

    rho, kdm, rdm = parts(state)
    rbar = float(jnp.mean(rho))

    def w_of(state, rho):
        return 0.5 * float(state.s_kick) * float(jnp.sum((rho - rbar) * state.psi)) * dV

    w0 = w_prev = w_of(state, rho)
    s_prev, psi_prev = float(state.s_kick), state.psi
    e0 = total_e(state.U) + kdm
    li = own = moved = 0.0
    worst = worst_own = 0.0
    at = {}
    while state.t < tau_end:
        target, a_t = next((t, a) for t, a in targets if t > state.t)
        dt, t_new = timestep(state, physics, target)
        state = step(state, dt, t_new, physics)
        grav.check(state)
        moved += float(state.stats.get("flux_moved", 0.0))
        rho, k1, r1 = parts(state)
        w, s = w_of(state, rho), float(state.s_kick)
        li += 0.5 * (s - s_prev) * (w_prev / s_prev + w / s)
        if dm is not None:
            phibar = 0.5 * (s_prev + s) * 0.5 * (psi_prev + state.psi)
            own += k1 - kdm + float(jnp.sum((r1 - rdm) * phibar)) * dV
            kdm, rdm = k1, r1
            worst_own = max(worst_own, abs(own) / abs(w))
        res = total_e(state.U) + kdm + w - e0 - w0 - li - added[0] - own
        worst = max(worst, abs(res) / abs(w))
        w_prev, s_prev, psi_prev = w, s, state.psi
        if state.t == target and a_t != a_end:
            at[a_t] = np.asarray(state.U)
    return dict(snaps=at, li=worst, own=worst_own, booked=added[0] / abs(w_prev), moved=moved / abs(w_prev),
                U=np.asarray(state.U), W=w_prev, steps=int(state.step))


def tophat_cfg(N, energy="flux", tol=1e-4, **kw):
    cyc = dict(gravity_warm_max_cycles=40) if tol < 1e-6 else {}
    return TH.cfg_for(N, gravity_energy=energy, gravity_warm_tol=tol, **cyc, **kw)


@functools.lru_cache(maxsize=None)
def tophat(N, a_end, energy="flux", tol=1e-4, **kw):
    """The ledger of the top-hat (TH.ic) at N^3 to a_end, plus e_th diagnostics: mean e_th / (3/2 P0) in the IGM
    (comoving r > 2 R0, the underdense box outside the sphere) and in the collapsing core (rho~ > 2)."""
    U0, X = TH.ic(N, TH.tophat())
    out = ledger(tophat_cfg(N, energy, tol, **dict(kw)), U0, a_end)
    U = out["U"]
    eth = U[-1] - 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]
    r = np.sqrt((X ** 2).sum(0))
    floor = 1.5 * TH.P0
    out["eth_igm"] = float(eth[r > 2 * TH.R0].mean() / floor)
    core = U[0] > 2.0
    out["eth_core"] = float(eth[core].mean() / floor) if core.any() else float("nan")
    return out


def report(tag, o):
    print(f"MEASURED {tag}: LI per-step max |res|/|W| {o['li']:.3e}, floors+comoving booked/|W| {o['booked']:.3e}, "
          f"flux_moved/|W| {o['moved']:.3e}, e_th/(1.5 P0) IGM {o['eth_igm']:.4g} core {o['eth_core']:.4g}, "
          f"steps {o['steps']}")




# ---- CPU: 16^3 to a = 0.03 (the identity is per step; the resolution does not enter) ----------------------------
CPU_N, CPU_A = 16, 0.03
CPU_LI = 2.7e-11                 # tol 1e-10: 2.05e-11 (CPU-8, job 31401057)


def test_comoving_flux_layzer_irvine_per_step():
    """Top-hat in flux mode, solve to 1e-10: the per-step Layzer-Irvine ledger closes to round-off."""
    o = tophat(CPU_N, CPU_A, "flux", 1e-10)
    report(f"tophat flux N={CPU_N} a={CPU_A} tol=1e-10", o)
    check(f"tophat flux N={CPU_N} tol=1e-10 per-step LI |res|/|W|", o["li"], CPU_LI)


@gpu_only
def test_comoving_kick_misses_the_flux_bound_gpu():
    """Teeth: the cell-centred kick on the same run misses the CPU bound by orders."""
    k = tophat(CPU_N, CPU_A, "kick", 1e-10)
    report(f"tophat kick N={CPU_N} a={CPU_A} tol=1e-10", k)
    assert k["li"] > 100 * CPU_LI


def test_comoving_flux_is_explicit_only():
    """'auto' stays 'kick' on a comoving clock (not validated as the default); an explicit 'flux' runs."""
    from luna.setup.gravity_energy import resolve_gravity_energy
    cfg = tophat_cfg(CPU_N)
    assert resolve_gravity_energy(replace(cfg, gravity_energy="auto")) == "kick"
    assert resolve_gravity_energy(cfg) == "flux"
    cfg.validate()
    with pytest.raises(ValueError, match="cosmo_rho_floor=0"):
        replace(cfg, cosmo_rho_floor=1e-2).validate()


def test_comoving_flux_close_sharded_is_one_device():
    """The comoving close (sbar psibar) on a 2x2x2 block mesh equals one device to round-off (cold gas: the
    positivity limit acts, its two global sums are psums; tests/test_gravity_energy_flux.py, amp 1)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from luna.setup.assemble import build_gravity
    from luna.setup.clock import make_clock
    from luna.setup.hydro import build_hydro
    from luna.setup.mesh import DENSE
    from luna.setup.run_kind import GAS
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices (runs on CPU-8)")
    n = 16
    cfg = tophat_cfg(n)
    U = jnp.asarray(TH.ic(n, TH.tophat())[0])
    rng = np.random.default_rng(5)
    host = [rng.standard_normal((n, n, n)) for _ in range(2)]
    out = []
    for mesh in (Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z")),
                 Mesh(np.array(jax.devices()[:8]).reshape(2, 2, 2), ("x", "y", "z"))):
        hyd = build_hydro(cfg, mesh, is_main=False)
        b = build_gravity(cfg, mesh, GAS, DENSE, make_clock(cfg), is_main=False).backend
        U1, F = hyd.step_flux(jax.device_put(U, hyd.sharding), 1e-3)
        psi0, psi1 = (jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("x", "y", "z"))) for a in host)
        U2, moved = b.close_flux(U1, b.grad(psi1, 2), 2.0, 5e-4, psi0, psi1, 1.0, F, 0.0 * psi0, 1e-3)
        out.append((np.asarray(U1), np.asarray(U2), float(moved)))
    assert np.array_equal(out[0][0], out[1][0])
    rel = np.abs(out[0][1] - out[1][1]).max() / np.abs(out[0][1]).max()
    print(f"MEASURED comoving flux close sharded, limiter moved {out[0][2]:.3g}: max rel diff {rel:.2g}")
    assert out[0][2] > 0.0 and rel < 1e-14


# ---- GPU: 64^3 to a = 0.5 (turnaround), the measured table -----------------------------------------------------
# A100 FP64, 324 steps: 3.14e-5 (job 31404523), 2.58e-11 (31404530); kick 2.45e-3 at both tolerances
GPU_LI = {1e-4: 4.1e-5, 1e-10: 3.4e-11}


@gpu_only
@pytest.mark.parametrize("tol", (1e-4, 1e-10))
def test_comoving_flux_layzer_irvine_64_gpu(tol):
    """64^3 to turnaround: the per-step ledger at the solve's tolerance; the kick on the same run, the limiter's share
    and the IGM e_th (cold gas, P0 = 1e-10: the face work minus the kick's dK lands in e_th) printed beside it."""
    o = tophat(64, TH.A_TA, "flux", tol)
    report(f"tophat flux N=64 a=0.5 tol={tol:g}", o)
    k = tophat(64, TH.A_TA, "kick", tol)
    report(f"tophat kick N=64 a=0.5 tol={tol:g}", k)
    check(f"tophat flux N=64 tol={tol:g} per-step LI |res|/|W|", o["li"], GPU_LI[tol])
    assert k["li"] > 10 * o["li"]


GPU32_A = 0.1
# old-formula lower bound 1.72e-5 / 1.3 (job 31404533); gamma 1.4: 2.10e-11 (31404534)
GPU32_LI = {"old": 1.3e-5, "gamma": 2.7e-11}


@gpu_only
def test_comoving_flux_old_potential_misses_gpu(monkeypatch):
    """Teeth: the proper-time face potential (s0 psi0 + s1 psi1)/2 on the comoving run leaves O(ds dpsi) per step."""
    import luna.gravity.backend as backend
    real = backend.flux_work_body

    def proper_formula(*a, comoving=False, **k):
        return real(*a, **k)
    monkeypatch.setattr(backend, "flux_work_body", proper_formula)
    U0, _ = TH.ic(32, TH.tophat())
    o = ledger(tophat_cfg(32, "flux", 1e-10), U0, GPU32_A)
    print(f"MEASURED tophat flux old Phibar N=32 a={GPU32_A} tol=1e-10 per-step LI |res|/|W| {o['li']:.3e}")
    check("tophat flux old Phibar N=32 per-step LI |res|/|W|", o["li"], GPU32_LI["old"], lower=True)


@gpu_only
def test_comoving_flux_gamma_books_the_comoving_energy_term_gpu():
    """gamma = 1.4: the (3 gamma - 5) e_th term of the comoving_energy stage, booked, and the ledger closes the same."""
    o = tophat(32, GPU32_A, "flux", 1e-10, gamma=1.4)
    report(f"tophat flux gamma=1.4 N=32 a={GPU32_A} tol=1e-10", o)
    check("tophat flux gamma=1.4 N=32 per-step LI |res|/|W|", o["li"], GPU32_LI["gamma"])


# ---- Zel'dovich pancake: adiabatic pre-caustic, T a^2 = p~ / rho~ = K rho~^(2/3) exactly (gamma 5/3) ----------
def pancake_cfg(N, energy, **kw):
    """The pancake of scripts/cosmo/run_pancake.py (a_i 0.3, caustic 1.5, cs 1 % of the peak v~), floor 1e-3 K."""
    from luna.cosmology import Cosmology
    from luna.ic_generation import zeldovich
    a_i, a_c, cs_frac = 0.3, 1.5, 1.0e-2
    cs0 = cs_frac * Cosmology(Omega_m=1.0, Omega_L=0.0).dadtau(a_i, 1.0) / (a_c * 2 * np.pi)
    K = cs0 ** 2 / (5.0 / 3.0)
    return accuracy_cfg(**{**dict(
        test_problem="Zeldovich-Pancake", SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0,
        gamma=5.0 / 3.0, BOUNDARY=0, CosmologicalIntegration=True, cosmo_Omega_m=1.0, cosmo_Omega_L=0.0,
        cosmo_H0=1.0, a_init=a_i, a_end=1.0, cosmo_p_floor=1e-3 * K, gravity_energy=energy,
        problem=zeldovich.Params(a_caustic=a_c, cs_frac=cs_frac)), **kw}), K


A_CROSS = 1.5       # rho~ = 1 / (1 - (a / a_c) cos k q): shell crossing at a = a_caustic (zeldovich.exact)


def entropy_error(U, K):
    """(L1 over x, max) of |p~ / (K rho~^gamma) - 1|, i.e. of T a^2 / (K rho~^(2/3)) - 1, gamma 5/3."""
    p = (2.0 / 3.0) * (U[-1] - 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0])
    dev = (p / (K * U[0] ** (5.0 / 3.0)) - 1.0)[:, 0, 0]
    return float(np.abs(dev).mean()), float(np.abs(dev).max())


@functools.lru_cache(maxsize=None)
def pancake(N, energy, a_end=1.0, dm=False, tol=1e-4):
    """(entropy L1 at a_end, max, the ledger dict with 'pre': the entropy (L1, max) at 0.5 A_CROSS); dm: + DM."""
    from luna.ic_generation.ic import initial_conditions
    from luna.setup.state import Particles
    kw = dict(run="gas+dm") if dm else {}
    if tol < 1e-6:
        kw.update(gravity_warm_max_cycles=40)
    cfg, K = pancake_cfg(N, energy, a_end=a_end, gravity_warm_tol=tol, **kw)
    U0 = np.asarray(initial_conditions(cfg))
    parts = None
    if dm:
        from scripts.cosmo.run_pancake_coupled import dm_zeldovich_ic
        parts = Particles(*dm_zeldovich_ic(cfg))
    o = ledger(cfg, U0, a_end, dm=parts, snaps=(0.5 * A_CROSS,))
    if 0.5 * A_CROSS in o["snaps"]:
        o["pre"] = entropy_error(o["snaps"][0.5 * A_CROSS], K)
    return (*entropy_error(o["U"], K), o)


# L1 at a = 0.75 (A100, job 31406590): kick 275, 75.3, 19.3; flux 1246, 408, 38.3 at 16, 32, 64 (a = 1: kick 926,
# 266, 71; flux 3745, 976, 225). Both converge; flux heats the cold flow 2-5x more (e_th ~ 2e-4 of K~)
PANCAKE = {("kick", 16): 360.0, ("kick", 32): 98.0, ("kick", 64): 25.0,
           ("flux", 16): 1620.0, ("flux", 32): 530.0, ("flux", 64): 50.0}


@gpu_only
def test_pancake_adiabatic_temperature_gpu():
    """16^3, 32^3 and 64^3: T a^2 against K rho~^(2/3) (exact pre-caustic) at a = 0.5 A_CROSS, flux beside kick (a = 1
    printed only). Regression only for flux: the face work minus the kick's change of K~ lands in the cold gas's e_th."""
    for e in ("kick", "flux"):
        for n in (16, 32, 64):
            l1, mx, o = pancake(n, e, 1.0)
            pre = o["pre"]
            print(f"MEASURED pancake {e} N={n} a={0.5 * A_CROSS:g} |T a^2 / K rho~^(2/3) - 1| L1 {pre[0]:.3e} max "
                  f"{pre[1]:.3e}; a=1 L1 {l1:.3e} max {mx:.3e}; LI per-step {o['li']:.3e}, moved/|W| {o['moved']:.3e}, "
                  f"floor booked/|W| {o['booked']:.3e}")
    for e in ("kick", "flux"):
        for n in (16, 32, 64):
            check(f"pancake {e} N={n} a={0.5 * A_CROSS:g} entropy L1", pancake(n, e, 1.0)[2]["pre"][0], PANCAKE[e, n])


# ---- gas + dark matter: the gas part exact, the particles keep their own budget (as tests/test_gravity_energy_flux)
GASDM_BOUND = 2.1e-11         # tol 1e-10: 1.64e-11, kick 3.17e-3, particles' own 3.8e-4 (A100, job 31404544)


@gpu_only
def test_gas_dm_comoving_flux_gpu():
    """Coupled pancake, 32^3 gas + 32^3 particles to a = 0.6, solve to 1e-10: the per-step ledger less the particles'
    own budget (sum dK_dm + sum drho_dm Phibar dV) closes to the solve; the kick misses it."""
    _, _, o = pancake(32, "flux", 0.6, dm=True, tol=1e-10)
    _, _, k = pancake(32, "kick", 0.6, dm=True, tol=1e-10)
    print(f"MEASURED gas+dm pancake N=32 flux LI-own {o['li']:.3e} (own {o['own']:.3e}), "
          f"kick LI-own {k['li']:.3e} (own {k['own']:.3e})")
    check("gas+dm comoving flux N=32 |LI res - particles' own| / |W|", o["li"], GASDM_BOUND)
    assert k["li"] > 100 * GASDM_BOUND
