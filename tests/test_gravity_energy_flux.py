"""gravity_energy='flux': the gas gains -dt/dx F_rho dPhi on every face from the hydro step's face mass fluxes, with
Phi = (Phi^n + Phi^n+1)/2 (Mullen, Hanawa & Gammie 2020, eq. 57-68), half to each side or (next to a cell that would
lose half its internal energy) all to the donor cell, so E_kin + E_th + W is conserved to the Poisson solve's
tolerance with no pressure floor. A7 / A8 problems in flux mode; historical kick baselines stay explicit in the A7 / A8 files."""
import functools
from dataclasses import replace

import numpy as np
import pytest

from tests import test_gravity_evrard_energy as A8
from tests import test_gravity_jeans_wave as A7
from tests._helpers import accuracy_cfg, check, check_faces, gpu_only, self_gravity_energies

TIMES = A8.TIMES
# {N: |dE/E0| at TIMES}, Evrard hllc-minmod, flux, no floor, default solve (tol 1e-4); 1.3x the running max (CPU)
BOUNDS = {
    32: (3.6e-6, 2.2e-5) + (5.7e-5,) * 10,
    # 2.78e-6 1.64e-5 4.35e-5 3.69e-5 4.31e-5 2.26e-5 5.97e-6 ... 1.12e-6 at t = 3 (CPU, job 31175600)
    64: (2.5e-6, 9.5e-6) + (6.3e-5,) * 10,
    # 1.92e-6 7.32e-6 4.82e-5 7.40e-6 9.24e-6 5.04e-6 1.06e-5 ... 1.14e-5 at t = 3
}
# The same with HLLC + MC (ran into p < 0 at the bounce before the C8 face fix); the limiter never acts (CPU)
MC_BOUNDS = {
    32: (4.0e-6, 2.4e-5) + (6.4e-5,) * 10,
    # 3.02e-6 1.78e-5 4.89e-5 3.49e-5 1.94e-5 1.09e-5 1.25e-5 ... 1.92e-5 at t = 3
    64: (2.6e-6, 9.9e-6) + (1.1e-4,) * 10,
    # 1.98e-6 7.62e-6 5.53e-5 6.29e-5 7.74e-5 5.92e-5 5.51e-5 ... 5.47e-5 at t = 3
}
# |dE| / |E0| at solve tol 1e-4 / round-off at 1e-10 (Evrard 32, to t = 1.5)
SOLVE_BOUND = {1e-4: 5.7e-5, 1e-10: 5.8e-11}     # 4.35e-5, 4.42e-11
# energy the positivity limiter moved (conservatively) / |E0| by t = 3: 2.49e-4 at 32 (5 steps), none at 64
MOVED_BOUND = {32: 3.3e-4, 64: 0.0}


def run(cfg, U0, times, mesh=None, faces=None):
    """[(t, |dE/E0|, |dE - floor energy|/|E0|, max_d |dP_d| / sqrt(M E), min e_th / e_th0 min, limiter's energy /
    |E0|)] and the final U through the production loop; the floor energy is what the projections (cosmo_p_floor)
    added, the limiter's what the flux positivity limit moved (stats flux_moved); W from `self_gravity_energies`."""
    import jax
    import jax.numpy as jnp

    from luna.setup.assemble import assemble
    from luna.setup.evolve import initial_state, step, timestep
    from luna.setup.ics import Initial
    from luna.setup.mesh import DENSE, build_mesh, decomposition
    from luna.setup.run_kind import run_kind

    cfg.validate()
    kind = run_kind(cfg)
    if mesh is None:
        layout = decomposition(cfg, kind)
        mesh, _ = build_mesh(cfg, layout, kind)
    else:
        layout = DENSE
    physics = assemble(cfg, mesh, kind, layout, Initial(cfg=cfg, U=np.asarray(U0)), is_main=False)
    h = cfg.Lx / cfg.Nx
    added = [0.0]

    def tracked(proj):
        def f(U, a):
            out = proj(U, a)
            added[0] += float(jnp.sum(out[-1]) - jnp.sum(U[-1])) * h ** 3
            return out
        return f
    physics.projections[:] = [tracked(p) for p in physics.projections]
    state = initial_state(Initial(cfg=cfg, U=jax.device_put(jnp.asarray(U0), physics.hydro.sharding)), physics)
    state = physics.gravity.init(state)
    e0 = sum(self_gravity_energies(np.asarray(U0), h, cfg.GRAVITATIONAL_G))
    rows, moved = [], 0.0

    def eth(U):
        return U[-1] - 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]
    e_min0 = float(eth(np.asarray(U0)).min())
    from luna.setup.face_report import FaceLog
    log = FaceLog(cfg, physics.hydro.grid, is_main=False)
    for t_end in times:
        while state.t < t_end:
            dt, t_new = timestep(state, physics, t_end)
            reports = []
            state = step(state, dt, t_new, physics, reports=reports)
            log.push(reports[0], state.step, float(t_new))
            log.flush()                                   # refuses an unphysical face unless allowed
            physics.gravity.check(state)                  # as the production loop (evolve.py)
            moved += float(state.stats.get("flux_moved", 0.0))
        U = np.asarray(state.U)
        assert np.isfinite(U).all(), f"non-finite state at t={t_end}"
        de = sum(self_gravity_energies(U, h, cfg.GRAVITATIONAL_G)) - e0
        dp = max(abs(U[1 + d].sum() - U0[1 + d].sum()) for d in range(3)) / np.sqrt(U0[0].sum() * U0[-1].sum())
        rows.append((t_end, abs(de) / abs(e0), abs(de - added[0]) / abs(e0), float(dp),
                     float(eth(U).min()) / e_min0, moved / abs(e0)))
    check_faces(cfg, log.total, faces)
    return rows, U


# MC's faces go non-physical at the bounce (audit C8): the MC runs allow the first-order face state and assert
# its count, FACES_MC[N] = (lo, hi), exact (CPU-8 job 31665570 and A100 job 31663810 agree).
FACES_MC = {32: (4608, 4608), 64: (7440, 7440)}


def evrard_cfg(n, **kw):
    return replace(A8.cfg_for(n, "hllc-minmod"), gravity_energy="flux", **kw)


@functools.lru_cache(maxsize=None)
def evrard(n, times=TIMES, **kw):
    from luna.ic_generation.ic import initial_conditions
    cfg = evrard_cfg(n, **dict(kw))
    return run(cfg, np.asarray(initial_conditions(cfg)), times, faces=FACES_MC.get(n) if cfg.LIMITER == 1 else None)[0]


@pytest.mark.parametrize("n", list(BOUNDS))
def test_evrard_flux_energy(n):
    """No floor: |E(t) - E0| / |E0| at every output time, e_th > 0 throughout, and what the positivity limiter moved
    is bounded (none at 64); kick mode measures 3.0 / 1.5 at t = 3."""
    rows = evrard(n)
    for t, err, _, _, emin, moved in rows:
        print(f"MEASURED evrard flux N={n} t={t} |dE/E0|={err:.4g} min e_th/e_th0={emin:.3g} limiter {moved:.3g}")
    for (t, err, *_), b in zip(rows, BOUNDS[n]):
        check(f"evrard flux N={n} t={t} |dE/E0|", err, b)
    assert min(r[4] for r in rows) > 0.0
    check(f"evrard flux N={n} limiter energy / |E0|", rows[-1][5], MOVED_BOUND[n])


@pytest.mark.parametrize("n", list(MC_BOUNDS))
def test_evrard_flux_energy_mc(n):
    """MC through the bounce: |E(t) - E0| / |E0| at every output time, e_th > 0, nothing for the limiter to move."""
    rows = evrard(n, LIMITER=1, allow_first_order_faces=True)
    for (t, err, *_), b in zip(rows, MC_BOUNDS[n]):
        check(f"evrard flux mc N={n} t={t} |dE/E0|", err, b)
    assert min(r[4] for r in rows) > 0.0
    check(f"evrard flux mc N={n} limiter energy / |E0|", rows[-1][5], 0.0)


@pytest.mark.parametrize("tol", list(SOLVE_BOUND))
def test_evrard_flux_conserves_to_the_solve(tol):
    """The error is the Poisson solve's: tol 1e-10 brings it to round-off, cold cells and limiter included."""
    rows = evrard(32, TIMES[:6], gravity_warm_tol=tol, gravity_warm_max_cycles=40)
    worst = max(r[1] for r in rows)
    print(f"MEASURED evrard flux N=32 tol={tol} max |dE|/|E0| = {worst:.3g}")
    check(f"evrard flux tol={tol} |dE|/|E0|", worst, SOLVE_BOUND[tol])


def test_kick_misses_the_flux_bound():
    """Teeth: the default cell-centred kick on the same run (A8's own numbers)."""
    kick = A8.budget(32, "hllc-minmod")
    assert kick[5][4] > 100 * BOUNDS[32][5]


def test_flux_bound_catches_one_sided_potential(monkeypatch):
    """Teeth: the face work from Phi^n+1 alone instead of the step average (the centring eq. 59 asks for)."""
    import luna.gravity.backend as backend
    real = backend.flux_work_body

    def one_sided(psi0, psi1, s0, s1, F, dt, inv_dx, pad, donor=None):
        return real(psi1, psi1, s1, s1, F, dt, inv_dx, pad, donor)
    monkeypatch.setattr(backend, "flux_work_body", one_sided)
    from luna.ic_generation.ic import initial_conditions
    cfg = evrard_cfg(32, gravity_warm_tol=1e-10, gravity_warm_max_cycles=40)
    rows = run(cfg, np.asarray(initial_conditions(cfg)), TIMES[:2])[0]
    print(f"MEASURED one-sided potential |dE|/|E0| = {rows[-1][1]:.3g}")
    assert rows[-1][1] > 100 * SOLVE_BOUND[1e-10]


def test_flux_refuses_an_unconverged_solve():
    """The warm solve stopping at gravity_warm_max_cycles above gravity_warm_tol is an energy error: refused."""
    from luna.ic_generation.ic import initial_conditions
    cfg = evrard_cfg(32, gravity_warm_tol=1e-12, gravity_warm_max_cycles=1)
    with pytest.raises(RuntimeError, match="warm solve stopped at 1 cycles"):
        run(cfg, np.asarray(initial_conditions(cfg)), TIMES[:1])


def test_half_split_alone_goes_negative(monkeypatch):
    """Teeth: without the donor faces and the limiter (FLUX_ETH_KEEP -> -inf) the half split takes the cold Evrard
    ambient at 32^3 to a negative internal energy (it NaNs at the first step or the bounce)."""
    import luna.gravity.backend as backend
    monkeypatch.setattr(backend, "FLUX_ETH_KEEP", -1e300)
    from luna.ic_generation.ic import initial_conditions
    cfg = evrard_cfg(32)
    try:
        rows = run(cfg, np.asarray(initial_conditions(cfg)), TIMES[:5])[0]
    except RuntimeError as e:                            # require_finite (run_report.py:105) on the NaN dt
        print(f"MEASURED half split alone: {str(e)[:100]}")
        assert str(e).startswith("non-finite dt at step "), str(e)
        return
    except AssertionError as e:                          # run()'s own finiteness check
        print(f"MEASURED half split alone: {str(e)[:100]}")
        assert str(e).startswith("non-finite state at t="), str(e)
        return
    emin = min(r[4] for r in rows)
    print(f"MEASURED half split alone min e_th/e_th0 = {emin:.3g}")
    assert emin <= 0.0


# a random smooth self-gravitating box: no symmetry, warm gas, no floor
TURB_BOUND = 2.7e-8                   # 2.06e-8 (kick at 32: see the MEASURED line of the teeth below)


def turb_U(n, seed=7):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    kk = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)

    def field():
        f = np.real(np.fft.ifftn(np.fft.fftn(rng.standard_normal((n, n, n))) * ((kk > 0) & (kk <= 4))))
        return f / np.abs(f).max()
    rho = 1.0 + 0.3 * field()
    v = [0.3 * field() for _ in range(3)]
    p = np.full_like(rho, 0.36 / (5 / 3))
    return np.stack([rho, *(rho * vi for vi in v), p / (2 / 3) + 0.5 * rho * sum(vi * vi for vi in v)])


def turb_cfg(n, **kw):
    base = dict(test_problem="Jeans-Instability-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n, Lx=1.0, Ly=1.0, Lz=1.0,
                BOUNDARY=0, gamma=5 / 3, SELF_GRAVITY=True, GRAVITATIONAL_G=1.0, gravity_energy="flux")
    return accuracy_cfg(**{**base, **kw})


@pytest.mark.parametrize("energy", ["auto", "flux"])
def test_turbulent_box_flux_energy(energy):
    """Warm turbulence with gravity: no floor, |dE/E0| at the solve's tolerance (kick: 1.6e-4 at 64^3, t = 0.75)."""
    rows, _ = run(turb_cfg(32, gravity_energy=energy), turb_U(32), (0.25, 0.5))
    worst = max(r[1] for r in rows)
    print(f"MEASURED turbulent box flux 32 max |dE/E0| = {worst:.3g}")
    check("turbulent box flux |dE/E0|", worst, TURB_BOUND)


def test_turbulent_box_kick_misses_the_flux_bound():
    """Teeth: the explicit kick on the same box."""
    rows, _ = run(turb_cfg(32, gravity_energy="kick"), turb_U(32), (0.25, 0.5))
    worst = max(r[1] for r in rows)
    print(f"MEASURED turbulent box kick 32 max |dE/E0| = {worst:.3g}")
    assert worst > 100 * TURB_BOUND


def test_turbulent_box_momentum_is_the_kicks():
    """Momentum takes the same cell-centred kick in both modes: its drift (the solve's residual) is the kick's."""
    dp = {m: max(r[3] for r in run(turb_cfg(32, gravity_energy=m), turb_U(32), (0.25,))[0]) for m in ("kick", "flux")}
    print(f"MEASURED turbulent box max |dP|/sqrt(ME) kick {dp['kick']:.3g} flux {dp['flux']:.3g}")
    assert dp["flux"] <= 2.0 * dp["kick"] + 1e-15


# A7 in flux mode: {(njeans, scheme): {N: L1}}, 1.3x measured; kick-mode bounds in test_gravity_jeans_wave
JEANS_L1 = {
    (0.5, "hllc-minmod"): {32: 0.11, 64: 0.026},    # 0.0821, 0.0197: kick's to 4 digits
    (0.5, "hllc-mc"): {32: 0.029, 64: 0.0034},      # 0.0223, 0.00257
    (1.5, "hllc-minmod"): {32: 0.087, 64: 0.021},   # 0.0666, 0.0157 (kick 0.0683, 0.0160)
    (1.5, "hllc-mc"): {32: 0.054, 64: 0.012},       # 0.0413, 0.00932 (kick 0.0430, 0.00964)
}


@pytest.mark.parametrize("njeans,scheme", list(JEANS_L1))
def test_jeans_flux(njeans, scheme):
    """The linear Jeans wave and growing mode are as accurate in flux mode (same orders)."""
    for n, b in JEANS_L1[njeans, scheme].items():
        cfg = replace(A7.cfg_for(n, njeans, scheme), gravity_energy="flux")
        w2 = A7.omega2(njeans)
        t_end = 2 * np.pi / np.sqrt(w2) if w2 > 0 else 1 / np.sqrt(-w2)
        U0 = A7.initial(n, njeans)
        U = run(cfg, U0, (t_end,))[1]
        growth = 1.0 if w2 > 0 else np.e
        l1 = float(np.abs((U[0] - (A7.RHO0 + (U0[0] - A7.RHO0) * growth)) / A7.AMP).mean())
        print(f"MEASURED jeans flux nj={njeans} {scheme} N={n} L1={l1:.4g}")
        check(f"jeans flux nj={njeans} {scheme} N={n} L1", l1, b)


@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_pallas_flux_kick_is_the_xla_kick(dtype):
    """kick_pallas.kick_flux == backend.kick_gas_flux_body bitwise (interpret mode on the CPU, Triton on a GPU)."""
    import jax
    import jax.numpy as jnp

    from luna.gravity import kick_pallas
    from luna.gravity.backend import _flux_kick_fn, kick_gas_flux_body
    rng = np.random.default_rng(3)
    U = jnp.asarray(rng.random((5, 8, 8, 64)) + 0.5, dtype)
    g = tuple(jnp.asarray(rng.standard_normal((8, 8, 64)), dtype) for _ in range(3))
    dE = jnp.asarray(rng.standard_normal((8, 8, 64)), dtype)
    a = np.asarray(jax.jit(kick_pallas.kick_flux)(U, g, 0.7, 0.013, dE))
    b = np.asarray(jax.jit(kick_gas_flux_body)(U, g, 0.7, 0.013, dE))
    assert a.dtype == b.dtype == np.dtype(dtype) and np.abs(b - np.asarray(U)).max() > 0.0
    np.testing.assert_array_equal(a, b)
    assert _flux_kick_fn("pallas", True) is kick_gas_flux_body          # autodiff: XLA
    with pytest.raises(ValueError, match="gravity_kernel='pallas'"):
        _flux_kick_fn("pallas", False)(U, tuple(q.astype("float16") for q in g), 0.7, 0.013, dE)


def _one_mesh():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))


def _block_mesh():
    import jax
    if jax.device_count() < 2:
        pytest.skip("needs 2 or more devices")
    from luna.gravity.particle_mesh_blocks import make_block_mesh
    pdims = (2, 2, 2) if jax.device_count() >= 8 else (2, 1, 1)
    return make_block_mesh(jax.devices()[:int(np.prod(pdims))], pdims)


@pytest.mark.parametrize("amp", (1e-3, 1.0), ids=("local", "limited"))
@pytest.mark.parametrize("iso", (False, True), ids=("periodic", "isolated"))
def test_flux_close_sharded_is_one_device(iso, amp):
    """The closing flux kick (face work + kick) on a block mesh equals one device, same inputs: bitwise while the
    positivity limit is idle (amp 1e-3: nothing flagged), to round-off when it acts (amp 1: its two global sums are
    psums); the hydro step's face fluxes bitwise (seam faces computed on both sides)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from luna.setup.assemble import build_gravity
    from luna.setup.clock import ProperClock
    from luna.setup.hydro import build_hydro
    from luna.setup.mesh import DENSE
    from luna.setup.run_kind import GAS
    mesh8 = _block_mesh()
    n = 16
    over = dict(BOUNDARY=4, POISSON_BOUNDARY=3) if iso else {}
    cfg = turb_cfg(n, **over)
    U = jnp.asarray(turb_U(n))
    one = _one_mesh()
    out = []
    for mesh in (one, mesh8):
        hyd = build_hydro(cfg, mesh, is_main=False)
        grav = build_gravity(cfg, mesh, GAS, DENSE, ProperClock(0.0, 1.0), is_main=False)
        Ul = jax.device_put(U, hyd.sharding)
        U1, F = hyd.step_flux(Ul, 1e-3)
        sh = NamedSharding(mesh, P("x", "y", "z"))
        rng = np.random.default_rng(1)
        psi0, psi1, W1 = (jax.device_put(jnp.asarray(amp * rng.standard_normal((n, n, n))), sh) for _ in range(3))
        g = grav.backend.grad(psi1, 2)
        U2, moved = grav.backend.close_flux(U1, g, 1.0, 5e-4, psi0, psi1, 1.0, F, W1, 1e-3)
        out.append((np.asarray(U1), np.asarray(U2), float(moved)))
    assert np.array_equal(out[0][0], out[1][0])
    if amp < 1.0:
        assert out[0][2] == out[1][2] == 0.0
        assert np.array_equal(out[0][1], out[1][1])
    else:
        rel = np.abs(out[0][1] - out[1][1]).max() / np.abs(out[0][1]).max()
        print(f"MEASURED flux close sharded, limiter moved {out[0][2]:.3g}: max rel diff {rel:.2g}")
        assert out[0][2] > 0.0 and rel < 1e-14


def test_flux_sharded_run_is_one_device():
    """A short flux-mode run on a block mesh equals one device to the solve's layout round-off."""
    mesh8 = _block_mesh()
    cfg = turb_cfg(16, gravity_warm_tol=0.0, gravity_proper_mixed=False, gravity_warm_extrapolate=False)
    U0 = turb_U(16)
    r1, U1 = run(cfg, U0, (0.05,))
    r8, U8 = run(cfg, U0, (0.05,), mesh=mesh8)
    rel = np.abs(U1 - U8).max() / np.abs(U1).max()
    print(f"MEASURED flux sharded vs one device max rel diff {rel:.3g}, |dE/E0| {r1[0][1]:.3g} / {r8[0][1]:.3g}")
    assert rel < 1e-10


def _traced_step(energy, n=16, dt=1e-3):
    """One production step as a function of U (autodiff_mode: traced gravity, the differentiable FMG)."""
    import jax.numpy as jnp

    from luna.setup.assemble import build_gravity
    from luna.setup.clock import ProperClock
    from luna.setup.hydro import build_hydro
    from luna.setup.mesh import DENSE
    from luna.setup.run_kind import GAS
    from luna.setup.state import State
    cfg = turb_cfg(n, autodiff_mode=True, gravity_warm_tol=0.0, gravity_energy=energy)
    mesh = _one_mesh()
    hyd = build_hydro(cfg, mesh, is_main=False)
    grav = build_gravity(cfg, mesh, GAS, DENSE, ProperClock(0.0, 1.0), is_main=False)

    def step(U):
        st = State(U=U, t=jnp.asarray(0.0), a=jnp.asarray(1.0), step=jnp.asarray(0, jnp.int32))
        st = grav.before(grav.init(st), 0.5 * dt, dt)
        if energy == "flux":
            U1, F = hyd.step_flux(st.U, dt)
            st = st.update(U=U1, mass_flux=F)
        else:
            st = st.update(U=hyd.step(st.U, dt))
        return grav.after(st, 0.5 * dt, dt).U
    return step, grav


def test_flux_close_adjoint_is_the_transpose():
    """<J v, w> == <v, J^T w> for the closing flux kick in all its inputs (U, psi0, psi1, F, W1): round-off."""
    import jax
    import jax.numpy as jnp
    _, grav = _traced_step("flux")
    n, rng = 16, np.random.default_rng(4)
    r = lambda *s: jnp.asarray(rng.standard_normal(s))          # noqa: E731
    U = jnp.asarray(turb_U(n))
    x = (U, r(n, n, n), r(n, n, n), (r(n + 1, n, n), r(n, n + 1, n), r(n, n, n + 1)), r(n, n, n))

    def f(x):
        U, psi0, psi1, F, W1 = x
        return grav.backend.close_flux(U, grav.backend.grad(psi1, 2), 1.0, 5e-4, psi0, psi1, 1.0, F, W1, 1e-3)[0]
    v = jax.tree_util.tree_map(lambda a: jnp.asarray(rng.standard_normal(a.shape)), x)
    w = r(5, n, n, n)
    _, Jv = jax.jvp(f, (x,), (v,))
    (JTw,) = jax.vjp(f, x)[1](w)
    a = float(jnp.vdot(Jv, w))
    b = sum(float(jnp.vdot(p, q)) for p, q in zip(jax.tree_util.tree_leaves(v), jax.tree_util.tree_leaves(JTw)))
    print(f"MEASURED flux close transpose {abs(a - b) / abs(a):.2g}")
    assert abs(a - b) <= 1e-12 * abs(a)


def test_flux_gradient_matches_finite_differences():
    """The traced flux step's VJP against central differences, no worse than the default kick step's (minmod's kinks
    limit both)."""
    import jax
    import jax.numpy as jnp
    U = jnp.asarray(turb_U(16))
    rng = np.random.default_rng(2)
    u = jnp.asarray(rng.standard_normal(U.shape)) * 1e-3 * U
    w = jnp.asarray(rng.standard_normal(U.shape))
    rel = {}
    for energy in ("kick", "flux"):
        step = _traced_step(energy)[0]
        (gU,) = jax.vjp(step, U)[1](w)
        eps = 1e-4
        fd = float(jnp.vdot(w, step(U + eps * u) - step(U - eps * u))) / (2 * eps)
        ad = float(jnp.vdot(gU, u))
        rel[energy] = abs(ad - fd) / abs(fd)
        print(f"MEASURED {energy} VJP vs FD: {ad:.10g} {fd:.10g} rel {rel[energy]:.2g}")
    assert rel["flux"] <= 2.0 * rel["kick"] + 1e-9


@pytest.mark.parametrize("over,why", [
    (dict(sinks_enabled=True), "sinks"), (dict(sil_weights="w.npz"), "SiL corrector"),
    (dict(cosmo_rho_floor=1e-3), "cosmo_rho_floor=0"), (dict(cc85_enabled=True), "no cc85 wind"),
    (dict(hydro_fused=True), "hydro_fused"), (dict(ORDER=0), "ORDER=1"), (dict(SELF_GRAVITY=False), "SELF_GRAVITY")])
def test_flux_refuses_what_it_cannot_run(over, why):
    with pytest.raises(ValueError, match=why):
        replace(turb_cfg(16), **over).validate()


@gpu_only
def test_evrard_flux_energy_gpu():
    """128 and 256 on a GPU: |dE/E0| at t = 3 stays at the solve's tolerance (kick: 1.0 and 0.58)."""
    errs = [evrard(n)[-1][1] for n in (128, 256)]
    print(f"MEASURED evrard flux t=3 |dE/E0| 128, 256 = {errs}")
    check("evrard flux t=3 |dE/E0| at 128, 256", max(errs), 1e-4)


ISO_TIMES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
# isolated Evrard 64^3: |dE - booked| / |E0| at ISO_TIMES, 1.3x measured (A100); unbooked is 0.07, 0.20 at t = 2, 3
ISO_BOUNDS = (3.1e-7, 1.6e-6, 1.4e-5, 1.8e-5, 2.2e-5, 5.6e-5, 1.1e-4, 3.5e-4)
# 2.39e-7 1.21e-6 1.10e-5 1.37e-5 1.70e-5 4.32e-5 8.63e-5 2.67e-4 (job 31226347)


def run_isolated(n, times):
    """[(t, |dE/E0|, |dE - booked| / |E0|)]: Evrard with diode gas and isolated gravity (BOUNDARY 4, POISSON_BOUNDARY
    3) in flux mode, W = 1/2 sum rho psi from LUNA's own solve, and booked per step what crosses the box faces: the
    hydro's change of the total energy (its boundary energy flux) and F_rho Phi at the ghost on every boundary face."""
    import jax
    import jax.numpy as jnp

    from luna.gravity.isolated import extrap_pad
    from luna.ic_generation.ic import initial_conditions
    from luna.setup.assemble import assemble
    from luna.setup.evolve import initial_state, step, timestep
    from luna.setup.ics import Initial
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind

    cfg = evrard_cfg(n, BOUNDARY=4, POISSON_BOUNDARY=3)
    cfg.validate()
    U0 = np.asarray(initial_conditions(cfg))
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    physics = assemble(cfg, mesh, kind, layout, Initial(cfg=cfg, U=U0), is_main=False)
    h = cfg.Lx / cfg.Nx
    gb = physics.gravity.backend
    booked = [0.0]
    step_flux, close_flux = physics.hydro.step_flux, gb.close_flux

    def booked_step_flux(U, dt):
        U1, F = step_flux(U, dt)
        booked[0] += float(jnp.sum(U1[-1]) - jnp.sum(U[-1])) * h ** 3
        return U1, F

    def booked_close_flux(U, g, s, hh, psi0, psi1, s0, F, W1, dt):
        pp = np.asarray(extrap_pad(0.5 * (s0 * psi0 + s * psi1), 1))
        tot = 0.0
        for d in range(3):
            Fd = np.asarray(F[d])
            lo, hi, fl, fh = [slice(1, -1)] * 3, [slice(1, -1)] * 3, [slice(None)] * 3, [slice(None)] * 3
            lo[d], hi[d], fl[d], fh[d] = 0, -1, 0, -1
            tot += float(np.sum(Fd[tuple(fl)] * pp[tuple(lo)]) - np.sum(Fd[tuple(fh)] * pp[tuple(hi)]))
        booked[0] += tot * dt * h ** 2
        return close_flux(U, g, s, hh, psi0, psi1, s0, F, W1, dt)
    physics.hydro.step_flux, gb.close_flux = booked_step_flux, booked_close_flux

    def energy(U):
        """K + E_th + 1/2 sum rho psi dV, psi from FMG then 40 warm cycles."""
        src, _ = gb.source(jax.device_put(jnp.asarray(U), physics.hydro.sharding), None, None, None, None)
        psi = gb.solve(src, None)
        for _ in range(40):
            psi = gb.solve(src, psi)
        return float(U[-1].sum() * h ** 3 + 0.5 * np.sum(U[0] * np.asarray(psi)) * h ** 3)
    state = initial_state(Initial(cfg=cfg, U=jax.device_put(jnp.asarray(U0), physics.hydro.sharding)), physics)
    state = physics.gravity.init(state)
    e0, rows = energy(U0), []
    for t_end in times:
        while state.t < t_end:
            dt, t_new = timestep(state, physics, t_end)
            state = step(state, dt, t_new, physics)
            physics.gravity.check(state)
        U = np.asarray(state.U)
        assert np.isfinite(U).all(), f"non-finite state at t={t_end}"
        de = energy(U) - e0
        rows.append((t_end, abs(de) / abs(e0), abs(de - booked[0]) / abs(e0)))
    return rows


@gpu_only
def test_evrard_flux_isolated_energy_is_booked_gpu():
    """Isolated boundaries: with what crosses the box faces booked, E is conserved to ISO_BOUNDS at every time
    (the rest: the multipole Green's function's asymmetry and the solve); unbooked fails them from t = 0.25."""
    rows = run_isolated(64, ISO_TIMES)
    for t, err, err_b in rows:
        print(f"MEASURED evrard flux isolated 64 t={t} |dE/E0|={err:.4g} booked {err_b:.4g}")
    for (t, err, err_b), b in zip(rows, ISO_BOUNDS):
        check(f"evrard flux isolated 64 t={t} |dE - booked|/|E0|", err_b, b)
        assert err > b, f"t={t}: unbooked {err:.3g} within the booked bound {b:.3g}: the booking has no teeth"


# mass and energy put in between steps (the AMUSE worker's feedback), booked at the injection: 1.03e-7 (A100, job
# 31226361) with the stepper's fresh solve, 1.8e-4 when the step opens with the potential of the old density
INJECT_BOUND = 1.4e-7


def run_injected(n, refresh, t_end=0.5, t_inj=(0.1, 0.2, 0.3)):
    """|E - E0 - booked| / |E0| through `amuse_luna.stepper`: turbulent box in flux mode, an SN-like deposit
    (`inject_energy`) set as U at each of t_inj; refresh=False writes U past the stepper (no fresh solve)."""
    import jax

    from amuse_luna.stepper import build
    from luna.sources.feedback import inject_energy
    cfg = turb_cfg(n)
    h = 1.0 / n

    def energy(U):
        return sum(self_gravity_energies(np.asarray(U), h, 1.0))
    ls = build(cfg, U0=turb_U(n))
    e0, booked = energy(ls.U), 0.0
    for k, t in enumerate(t_inj):
        ls.evolve_to(t)
        U1 = inject_energy(ls.U, cfg, 0.3 + 0.2 * k, 0.5, 0.4, energy=0.02, mass=0.01, radius=3 * h)
        booked += energy(U1) - energy(ls.U)
        if refresh:
            ls.U = U1
        else:
            ls._s = ls._s.update(U=jax.device_put(U1, ls._sharding))
    ls.evolve_to(t_end)
    return abs(energy(ls.U) - e0 - booked) / abs(e0)


def test_injected_mass_gets_a_fresh_potential():
    """Mass set from outside the step: the stepper re-solves before the next flux step, so the budget closes to the
    solve's tolerance; with the old potential it misses by 1000x."""
    err = run_injected(32, True)
    check("flux, injected mass booked, |E - E0 - booked|/|E0|", err, INJECT_BOUND)
    stale = run_injected(32, False)
    print(f"MEASURED flux, injected mass, old potential: {stale:.3e}")
    assert stale > 100 * INJECT_BOUND, f"the old potential gives {stale:.3g}: the test has no teeth"


@pytest.mark.parametrize("override", [
    {}, {"SELF_GRAVITY": False}, {"SPATIAL_DIMENSION": 2}, {"ORDER": 0},
    {"hydro_fused": True}, {"hydro_kernel": "pallas"}, {"AMR": True},
    {"CosmologicalIntegration": True}, {"run": "dm"}, {"sinks_enabled": True},
    {"chemistry": "net5"}, {"hydro_kernel": "auto"},
])
def test_auto_energy_selection(override):
    from luna.setup.gravity_energy import resolve_gravity_energy
    from luna.template_config import Config

    cfg = replace(Config(), SELF_GRAVITY=True, SPATIAL_DIMENSION=3)
    cfg = replace(cfg, **override)
    supported = not override or override in ({"hydro_kernel": "auto"}, {"hydro_kernel": "pallas"})
    explicit_only = override == {"CosmologicalIntegration": True}     # runs when asked, 'auto' stays 'kick'
    assert cfg.gravity_energy == "auto"
    assert resolve_gravity_energy(cfg) == ("flux" if supported else "kick")
    assert resolve_gravity_energy(replace(cfg, gravity_energy="kick")) == "kick"
    explicit = replace(cfg, gravity_energy="flux")
    if supported or explicit_only:
        assert resolve_gravity_energy(explicit) == "flux"
    else:
        with pytest.raises(ValueError, match="gravity_energy='flux' needs"):
            resolve_gravity_energy(explicit)
        with pytest.raises(ValueError, match="gravity_energy='flux' needs"):
            explicit.validate()


def test_energy_defaults_and_historical_baselines():
    from luna.setup.gravity_energy import resolve_gravity_energy
    from luna.template_config import Config

    assert resolve_gravity_energy(Config()) == "kick"
    assert A8.cfg_for(32).gravity_energy == "kick"
    assert A7.cfg_for(32, 4, "hllc-mc").gravity_energy == "kick"


# ---- the Pallas hydro kernels' face mass fluxes (goal: flux mode with hydro_kernel='pallas') ----------
@pytest.mark.parametrize("mode", ("fp64", pytest.param("fp32", marks=pytest.mark.precision("fp32")),
                                  pytest.param("mixed", marks=pytest.mark.precision("mixed"))))
@pytest.mark.parametrize("limiter", (0, 1))
def test_pallas_mass_flux_is_the_xla_mass_flux(mode, limiter):
    """3-D: fused and split kernels' F_rho == the XLA wide step's, and asking for the fluxes leaves dU unchanged:
    bitwise under Triton; round-off in the CPU interpreter (MC mixed: 7 of 7680 dU values 1.6e-6 apart there)."""
    import jax
    import jax.numpy as jnp

    from luna.hydro import muscl_hancock_pallas as mhp
    from luna.hydro.muscl_hancock import _muscl_hancock_3d
    from tests.test_hydro_pallas import GAMMA, GPU, _SHAPES, _prims_nd
    shape = _SHAPES[3]
    W = _prims_nd(shape)
    if mode == "fp32":
        W = W.astype(jnp.float32)
    Wp = jnp.pad(W, [(0, 0)] + [(2, 2)] * 3, mode="wrap")
    dt, inv = 0.2 / max(shape) / 3.0, tuple(float(n) for n in shape)
    r32 = mode == "mixed"
    _, Fx = jax.jit(lambda Wp: _muscl_hancock_3d(Wp, dt, gamma=GAMMA, limiter=limiter, riemann_solver="hllc",
                                                 inv_dx=inv[0], inv_dy=inv[1], inv_dz=inv[2], boundary=0,
                                                 wide=True, recon32=r32, mass_flux=True))(Wp)
    tol = 1e-12 if mode == "fp64" else 1e-6
    for split in (False, True):
        def pal(Wp, m):
            return mhp.divergence(Wp, dt, gamma=GAMMA, inv=inv, recon32=r32, split=split, limiter=limiter,
                                  mass_flux=m)
        dU, Fp = jax.jit(lambda Wp: pal(Wp, True))(Wp)
        dU0 = jax.jit(lambda Wp: pal(Wp, False))(Wp)
        if GPU:
            np.testing.assert_array_equal(np.asarray(dU), np.asarray(dU0))
        assert np.abs(np.asarray(dU) - np.asarray(dU0)).max() <= tol * np.abs(np.asarray(dU0)).max()
        for d in range(3):
            a, b = np.asarray(Fx[d]), np.asarray(Fp[d])
            assert a.shape == b.shape == tuple(n + (k == d) for k, n in enumerate(shape)) and a.dtype == b.dtype
            if GPU:
                np.testing.assert_array_equal(b, a)
            assert np.abs(a - b).max() <= tol * np.abs(a).max()


@pytest.mark.parametrize("nd", (1, 2, 3))
def test_pallas_mass_flux_is_its_density_update(nd):
    """Every dimension: the returned faces telescope to the kernel's own dU_rho (the faces it used)."""
    import jax

    from luna.hydro import muscl_hancock_pallas as mhp
    from tests.test_hydro_pallas import GAMMA, _SHAPES, _prims_nd
    shape = _SHAPES[nd]
    Wp = jax.numpy.pad(_prims_nd(shape), [(0, 0)] + [(2, 2)] * nd, mode="wrap")
    dt, inv = 0.2 / max(shape) / 3.0, tuple(float(n) for n in shape) + (1.0,) * (3 - nd)
    for split in (False, True):
        dU, F = jax.jit(lambda Wp: mhp.divergence(Wp, dt, gamma=GAMMA, inv=inv, split=split, limiter=1,
                                                  mass_flux=True))(Wp)
        div = sum(np.diff(-np.asarray(F[d]), axis=d) * inv[d] for d in range(nd))
        dr = np.asarray(dU)[0]
        assert np.abs(div - dr).max() <= 1e-12 * np.abs(dr).max()


@gpu_only
@pytest.mark.parametrize("iso", (False, True), ids=("periodic", "isolated"))
def test_pallas_step_flux_is_the_xla_step_flux_gpu(iso):
    """The production flux-mode hydro step (shard_map, one GPU): hydro_kernel='pallas' == 'xla', U and the face mass
    fluxes bitwise, and 'auto' (Pallas on a GPU) the same."""
    import jax
    import jax.numpy as jnp

    from luna.setup.hydro import build_hydro
    n = 32
    over = dict(BOUNDARY=4, POISSON_BOUNDARY=3) if iso else {}
    U = jnp.asarray(turb_U(n))
    out = {}
    for kernel in ("xla", "pallas", "auto"):
        hyd = build_hydro(turb_cfg(n, hydro_kernel=kernel, **over), _one_mesh(), is_main=False)
        U1, F = hyd.step_flux(jax.device_put(U, hyd.sharding), 2e-3)
        out[kernel] = [np.asarray(U1)] + [np.asarray(f) for f in F]
    for kernel in ("pallas", "auto"):
        for a, b in zip(out["xla"], out[kernel]):
            np.testing.assert_array_equal(a, b)


# ---- gas + dark matter (proper clock): the gas coupling stays exact, the particles keep their leapfrog ----------
def dm_ic(n, fdm=0.5, seed=11):
    """One particle per cell on a lattice displaced by a smooth field (0.3 dx), smooth velocities (0.3), mean
    density fdm of the gas's 1."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    kk = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)

    def field():
        f = np.real(np.fft.ifftn(np.fft.fftn(rng.standard_normal((n, n, n))) * ((kk > 0) & (kk <= 4))))
        return (f / np.abs(f).max()).reshape(-1)
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    q = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    pos = np.mod(q + 0.3 * dx * np.stack([field() for _ in range(3)], axis=-1), 1.0)
    vel = 0.3 * np.stack([field() for _ in range(3)], axis=-1)
    return pos, vel, np.full(pos.shape[0], fdm * dx ** 3)


def run_dm(n, energy, t_end=0.25):
    """((E - E0) / |E0|, the particles' own budget sum_steps dK_dm + sum (rho_dm^n+1 - rho_dm^n) Phibar dV, / |E0|) with
    E = K_gas + E_th + K_dm + W(rho_gas + rho_dm), through the production loop."""
    import jax
    import jax.numpy as jnp

    from luna.setup.assemble import assemble
    from luna.setup.evolve import initial_state, step, timestep
    from luna.setup.ics import Initial
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import COUPLED
    from luna.setup.state import Particles
    cfg = turb_cfg(n, run="gas+dm", gravity_energy=energy)      # validate() wants a MUSIC file for gas+dm
    layout = decomposition(cfg, COUPLED)
    mesh, _ = build_mesh(cfg, layout, COUPLED)
    U0 = turb_U(n)
    dm = Particles(*dm_ic(n))
    physics = assemble(cfg, mesh, COUPLED, layout, Initial(cfg=cfg, U=U0, dm=dm), is_main=False)
    state = initial_state(Initial(cfg=cfg, U=jax.device_put(jnp.asarray(U0), physics.hydro.sharding), dm=dm),
                          physics)
    grav = physics.gravity
    state = grav.init(state)
    h = 1.0 / n

    def parts(state):
        U, d = np.asarray(state.U), state.species["dm"]
        act = np.asarray(d.active) > 0
        kdm = 0.5 * float((np.asarray(d.mass)[act] * (np.asarray(d.p)[act] ** 2).sum(axis=1)).sum())
        return U, kdm, np.asarray(grav.deposit_of(state, "dm"))

    def energy_of(U, kdm, rho_dm):
        k, th, _ = self_gravity_energies(U, h, 1.0)
        return k + th + kdm + self_gravity_energies(np.stack([U[0] + rho_dm] + [0 * U[0]] * 4), h, 1.0)[2]
    U, k0, r0 = parts(state)
    e0, own = energy_of(U, k0, r0), 0.0
    while state.t < t_end:
        dt, t_new = timestep(state, physics, t_end)
        psi0 = np.asarray(state.psi)
        state = step(state, dt, t_new, physics)
        grav.check(state)
        U, k1, r1 = parts(state)
        own += k1 - k0 + float(np.sum((r1 - r0) * 0.5 * (psi0 + np.asarray(state.psi)))) * h ** 3
        k0, r0 = k1, r1
    return (energy_of(U, k0, r0) - e0) / abs(e0), own / abs(e0)


def test_gas_dm_flux_energy():
    """Gas + DM in flux mode: the whole error is the particles' own budget (the gas's to the solve's tolerance), and
    smaller than the kick's; the particles' share does not shrink with dt (A100 64^3: 1.35e-4 at CFL 0.8 and 0.4 by
    t = 0.75), it is the PM force (CIC-gathered 4th-order gradient) not being the gradient of the CIC potential."""
    err, own = run_dm(32, "flux")
    kick, own_k = run_dm(32, "kick")
    print(f"MEASURED gas+dm 32 flux dE/|E0| {err:.3g} (particles' own {own:.3g}, rest {err - own:.3g}), "
          f"kick {kick:.3g} ({own_k:.3g})")
    check("gas+dm 32 flux |dE - particles' own| / |E0|", abs(err - own), 2.3e-9)     # 1.75e-9 (CPU)
    assert abs(err) < abs(kick)


# ---- P0 groundwork: a cell-centred external potential in the face work, mass booked by the pre-solve ops ---------
def test_flux_external_potential_and_booking_sharded_is_one_device():
    """close_flux with phi_ext and booked_work on a block mesh equal one device; phi_ext = 0 is the plain close,
    and booked_work is sum (drho - <drho>) Phibar dV (periodic)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from luna.setup.assemble import build_gravity
    from luna.setup.clock import ProperClock
    from luna.setup.hydro import build_hydro
    from luna.setup.mesh import DENSE
    from luna.setup.run_kind import GAS
    n = 16
    cfg = turb_cfg(n)
    U = jnp.asarray(turb_U(n))
    rng = np.random.default_rng(3)
    host = [1e-3 * rng.standard_normal((n, n, n)) for _ in range(5)]      # psi0, psi1, W1, phi_ext, drho
    out = []
    for mesh in (_one_mesh(), _block_mesh()):
        hyd = build_hydro(cfg, mesh, is_main=False)
        b = build_gravity(cfg, mesh, GAS, DENSE, ProperClock(0.0, 1.0), is_main=False).backend
        U1, F = hyd.step_flux(jax.device_put(U, hyd.sharding), 1e-3)
        psi0, psi1, W1, ext, drho = (jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("x", "y", "z")))
                                     for a in host)
        g = b.grad(psi1, 2)
        plain = np.asarray(b.close_flux(U1, g, 1.0, 5e-4, psi0, psi1, 1.0, F, W1, 1e-3)[0])
        zero = np.asarray(b.close_flux(U1, g, 1.0, 5e-4, psi0, psi1, 1.0, F, W1, 1e-3, phi_ext=0.0 * ext)[0])
        with_ext = np.asarray(b.close_flux(U1, g, 1.0, 5e-4, psi0, psi1, 1.0, F, W1, 1e-3, phi_ext=ext)[0])
        out.append((plain, zero, with_ext, float(b.booked_work(psi0, psi1, 1.0, 1.0, drho)),
                    float(b.booked_work(psi0, psi1, 1.0, 1.0, drho, phi_ext=ext))))
    d = host[4] - host[4].mean()
    phibar = 0.5 * (host[0] + host[1])
    want = (float(np.sum(d * phibar)) / n ** 3, float(np.sum(d * (phibar + host[3]))) / n ** 3)
    for one, blocks in zip(out[0][:3], out[1][:3]):
        np.testing.assert_array_equal(one, blocks)
    np.testing.assert_array_equal(out[0][0], out[0][1])
    assert np.abs(out[0][2] - out[0][0]).max() > 0.0
    for got in (out[0], out[1]):
        for g_, w in zip(got[3:], want):
            assert abs(g_ - w) <= 1e-13 * abs(w), (g_, w)


HOOK_BOUND = 1.8e-8          # |E + W - E0 - W0 - booked| / |E0| (32^3, t = 0.25): 1.35e-8 (CPU-8, job 31387079)


def run_hook(n, t_end=0.25, rate=0.2):
    """(|E - E0 - sum W_booked - sum E_booked| / |E0|, |sum W_booked| / |E0|): the turbulent box in flux mode with a
    pre-solve op that adds rate * dt * a Gaussian bump of mass each step (no momentum or energy)."""
    import jax
    import jax.numpy as jnp

    from luna.setup.assemble import assemble
    from luna.setup.evolve import initial_state, step, timestep
    from luna.setup.ics import Initial
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    cfg = turb_cfg(n)
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    U0 = turb_U(n)
    physics = assemble(cfg, mesh, kind, layout, Initial(cfg=cfg, U=U0), is_main=False)
    x = (np.arange(n) + 0.5) / n - 0.4
    X = np.meshgrid(x, x, x, indexing="ij")
    bump = jnp.asarray(np.exp(-sum(c * c for c in X) / (2 * 0.08 ** 2)))
    sharding = physics.hydro.sharding

    def add_mass(state, h, dt):
        return state.update(U=jax.device_put(state.U.at[0].add(rate * dt * bump), sharding))
    physics.gravity.pre_solve.append(add_mass)
    state = physics.gravity.init(initial_state(Initial(cfg=cfg, U=jax.device_put(jnp.asarray(U0), sharding)),
                                               physics))
    h = 1.0 / n
    e0 = sum(self_gravity_energies(U0, h, 1.0))
    booked_w = booked_e = 0.0
    while state.t < t_end:
        dt, t_new = timestep(state, physics, t_end)
        state = step(state, dt, t_new, physics)
        physics.gravity.check(state)
        booked_w += float(state.stats["W_booked"])
        booked_e += float(state.stats["E_booked"])
    de = sum(self_gravity_energies(np.asarray(state.U), h, 1.0)) - e0
    return abs(de - booked_w - booked_e) / abs(e0), abs(booked_w) / abs(e0)


def test_pre_solve_mass_is_booked():
    """Mass a pre-solve op adds is seen by the closing solve and booked: E + W closes to the solve's tolerance with
    sum W_booked, which itself is >> the bound (the booking has teeth)."""
    err, booked = run_hook(32)
    print(f"MEASURED flux pre-solve mass booked |dE - booked|/|E0| = {err:.3g}, booked W / |E0| = {booked:.3g}")
    check("flux pre-solve mass |E - E0 - booked| / |E0|", err, HOOK_BOUND)
    assert booked > 100 * HOOK_BOUND
