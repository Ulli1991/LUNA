"""The M2 closures as the radiation stage (`rad_closure`): beams cross where M1
piles them up, the coupling conserves what it conserves for M1, the sharded
half-step is the single-device one, and the default is bitwise what it was.

The crossing thresholds are set from measurements (docs/RADIATION.md, "M2 as a
stage"): the numbers the test prints are the ones the thresholds came from.
"""
import hashlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp  # noqa: E402

from luna.radiation import m2_3d  # noqa: E402
from luna.radiation.coupling import total_energy  # noqa: E402
from tests._helpers import check  # noqa: E402

CHAT = 3.0


def _mesh(shape):
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    n = int(np.prod(shape))
    return Mesh(mesh_utils.create_device_mesh(shape, jax.devices()[:n]), ("x", "y", "z"))


def _one():
    from jax.sharding import NamedSharding, PartitionSpec as P
    return NamedSharding(_mesh((1, 1, 1)), P(None, "x", "y", "z"))


def _cfg(n, **kw):
    from dataclasses import replace
    from luna.template_config import Config
    base = dict(SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n, Lx=1.0, Ly=1.0, Lz=1.0,
                BOUNDARY=0, enable_x64=True, radiation_enabled=True, rad_groups=2,
                rad_chat=CHAT, rad_E0=1.0, rad_kappa_a=0.7, rad_kappa_s=0.3,
                rad_source=0.4, rad_substeps=2)
    base.update(kw)
    return replace(Config(), **base)


def _gas(n, seed=0):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.5, 1.5, size=(n, n, n))
    mom = rng.normal(0.0, 0.1, size=(3, n, n, n))
    e_int = rng.uniform(1.0, 2.0, size=(n, n, n))
    E = e_int + 0.5 * np.sum(mom ** 2, axis=0) / rho
    return jnp.asarray(np.concatenate([rho[None], mom, E[None]], axis=0))


def _m2(n, g=2, seed=1, mix=0.2):
    """Realizable M2 groups: per cell a mixture of 1-3 beams and isotropic light."""
    rng = np.random.default_rng(seed)
    out = np.zeros((g, 10, n, n, n))
    for k in range(g):
        E = rng.uniform(0.1, 1.0, size=(n, n, n))
        for _ in range(3):
            d = rng.normal(size=(3, n, n, n))
            d /= np.sqrt((d ** 2).sum(axis=0))
            w = rng.uniform(0.0, 1.0, size=(n, n, n)) * (1.0 - mix) / 3.0
            u = [d[i] for i in range(3)]
            out[k, 1:4] += w * E * d
            out[k, 4:10] += np.stack([w * E * u[i] * u[j] for i, j in m2_3d._SYM])
        out[k, 0] = E
        iso = E - out[k, 4] - out[k, 5] - out[k, 6]
        out[k, 4:7] += iso / 3.0
    return jnp.asarray(out)


def _momentum(U, R, chat=CHAT):
    """Gas momentum plus radiation momentum F/c^2 = m1/c."""
    return jnp.sum(U[1:4], axis=(1, 2, 3)) + jnp.sum(R[:, 1:4], axis=(0, 2, 3, 4)) / chat


# -- beam crossing through the stage ------------------------------------------
N_CROSS, CT, SIG = 32, 0.2, 0.12


def _crossing(closure):
    """Two Gaussian beams head-on along x through the stage (no opacity, no
    force) to ct = 0.2 on 32^3; the error against the analytic answer (the
    beams ignore each other) and the mid-plane peak."""
    from luna.sources.radiation import setup_radiation
    n = N_CROSS
    cfg = _cfg(n, rad_closure=closure, rad_chat=1.0, rad_groups=1, rad_E0=0.0,
               rad_kappa_a=0.0, rad_kappa_s=0.0, rad_source=0.0, rad_force=False,
               rad_cfl=0.3, rad_substeps=1, BOUNDARY=1)
    rad = setup_radiation(cfg, _one(), is_main=False)
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    def blob(c):
        return np.exp(-0.5 * (((X - c) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) / SIG ** 2))

    eL, eR = blob(0.3), blob(0.7)
    if closure == "m1":
        R = jnp.asarray(np.stack([eL + eR, eL - eR, 0 * eL, 0 * eL]))[None]
    else:
        R = m2_3d.add(m2_3d.beam(jnp.asarray(eL), (1.0, 0.0, 0.0)),
                      m2_3d.beam(jnp.asarray(eR), (-1.0, 0.0, 0.0)))[None]
    steps = int(round(CT / (2.0 * cfg.rad_cfl / n)))          # the split CFL, 11 steps
    U = _gas(n)
    for _ in range(steps):
        U, R = rad["before"](U, R, CT / steps)
    exact = blob(0.3 + CT) + blob(0.7 - CT)
    E = np.asarray(R[0, 0])
    err = float(np.sqrt(np.mean((E - exact) ** 2) / np.mean(exact ** 2)))
    return err, float(E[n // 2].max()), float(exact[n // 2].max())


def test_the_beams_cross_with_the_learned_closure_and_pile_up_with_m1():
    """M1 sees m1 = 0 in the overlap, calls it isotropic and piles the energy up
    in the mid-plane; the learned M2 carries m2 and lets the beams pass.
    Measured (32^3, 11 split steps): M1 0.440, learned M2 0.092 (the
    entropy closure it approximates: 0.084; the trace closure, the network
    switched off: 0.207)."""
    m1_err, m1_peak, exact_peak = _crossing("m1")
    m2_err, m2_peak, _ = _crossing("m2_learned")
    print(f"crossing error M1 {m1_err:.4f} (mid-plane peak {m1_peak:.3f}), "
          f"learned M2 {m2_err:.4f} ({m2_peak:.3f}), exact peak {exact_peak:.3f}")
    assert m1_err > 0.40, f"M1 {m1_err:.4f}: the pile-up this test is built on is gone"
    assert m2_err < 0.11, f"learned M2 {m2_err:.4f}"
    assert m2_err < 0.25 * m1_err
    assert m1_peak > 1.15 * exact_peak and m2_peak < exact_peak


# -- conservation --------------------------------------------------------------
@pytest.mark.parametrize("closure", ["m2_learned", "m2_entropy"])
def test_the_m2_half_step_conserves_energy_and_momentum(closure):
    """A periodic box, absorption, emission, scattering and the force with
    c_true = c_hat: gas plus radiation energy and momentum are fixed to
    round-off, trace(m2) = m0 in every cell, and the gas was pushed."""
    from luna.sources.radiation import setup_radiation
    n = 8
    cfg = _cfg(n, rad_closure=closure, rad_c_true=CHAT, rad_substeps=1)
    rad = setup_radiation(cfg, _one(), is_main=False)
    U, R = _gas(n), _m2(n)
    p0, e0 = _momentum(U, R), float(total_energy(U, R))
    h = 1e-3 if closure == "m2_learned" else 2e-4
    U2, R2 = rad["before"](U, R, h)
    U2, R2 = rad["after"](U2, R2, h)
    assert R2.shape == R.shape and bool(jnp.all(jnp.isfinite(R2)))
    rel = abs(float(total_energy(U2, R2)) - e0) / e0
    assert rel < 1e-14, f"total energy moved by {rel:.2e}"
    dp = np.asarray(_momentum(U2, R2) - p0)
    assert float(np.max(np.abs(dp))) < 1e-13 * float(jnp.sum(R[:, 0])) / CHAT
    tr = R2[:, 4] + R2[:, 5] + R2[:, 6]
    assert float(jnp.max(jnp.abs(tr - R2[:, 0]))) < 1e-14 * float(jnp.max(R2[:, 0]))
    assert float(jnp.max(jnp.abs(U2[1:4] - U[1:4]))) > 1e-8


def test_the_stage_runs_in_the_real_step_with_m2_and_conserves_energy():
    """`test_radiation_coupled`'s integration gate with rad_closure="m2_learned"."""
    from luna.ic_generation.ic import resolve
    from luna.setup.assemble import assemble
    from luna.setup.evolve import initial_state, step, timestep
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    n = 16
    cfg = resolve(_cfg(n, rad_closure="m2_learned", test_problem="Soundwave-3D",
                       t_end=1.0))
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, False)
    physics = assemble(cfg, mesh, kind, layout, init, False)
    state = initial_state(init, physics)
    tot0 = float(total_energy(state.U, state.sources["radiation"]))
    m0, p0 = float(jnp.sum(state.U[0])), np.asarray(_momentum(state.U, state.sources["radiation"]))
    for _ in range(4):
        dt, t_new = timestep(state, physics, 1e30)
        state = step(state, dt, t_new, physics)
    R = state.sources["radiation"]
    assert R.shape == (2, 10, n, n, n) and bool(jnp.all(jnp.isfinite(R)))
    tot = float(total_energy(state.U, R))
    assert abs(tot - tot0) / abs(tot0) < 1e-12, f"total energy moved by {tot - tot0:.3e}"
    # periodic, c_true = c_hat (rad_c_true=0): gas mass, and gas + radiation momentum, to round-off
    check("rad real step |dM|/M", abs(float(jnp.sum(state.U[0])) / m0 - 1.0), 1e-13)
    check("rad real step |dP|/sqrt(M E)", float(np.abs(np.asarray(_momentum(state.U, R)) - p0).max())
          / np.sqrt(m0 * abs(tot0)), 1e-13)
    assert abs(float(R[:, 0].mean()) - cfg.rad_E0) > 1e-3 * cfg.rad_E0


def test_absorption_by_ionisation_keeps_an_m2_field_realizable():
    """The ionising path with M2: photons injected isotropically, absorbed into
    ionisations, every moment scaled together. Photons are conserved against
    ionisations (no recombination), and trace(m2) = m0."""
    from luna.sources.radiation import setup_radiation
    n = 16
    cfg = _cfg(n, rad_closure="m2_learned", rad_groups=1, rad_ionization=True, rad_E0=0.0,
               rad_kappa_a=0.0, rad_kappa_s=0.0, rad_source=0.0, BOUNDARY=0,
               rad_sigma=50.0, rad_alpha_B=0.0, rad_m_H=1.0, rad_photon_rate=20.0,
               rad_cs2_neutral=0.01, rad_cs2_ionized=1.0, rad_source_radius=0.1,
               gamma=5.0 / 3.0)
    rad = setup_radiation(cfg, _one(), is_main=False)
    U = _gas(n)
    c = rad["R0"]
    h = 2e-2
    for _ in range(3):
        U, c = rad["before"](U, c, h)
    R, x, _ = c
    dV = 1.0 / n ** 3
    photons = float(jnp.sum(R[0, 0]) * dV)
    ionised = float(jnp.sum(U[0] * x) * dV)                 # m_H = 1
    injected = 3 * h * cfg.rad_photon_rate
    assert float(jnp.max(x)) > 0.5, "nothing was ionised"
    assert abs(photons + ionised - injected) < 1e-12 * injected
    tr = R[0, 4] + R[0, 5] + R[0, 6]
    assert float(jnp.max(jnp.abs(tr - R[0, 0]))) < 1e-14 * float(jnp.max(R[0, 0]))


# -- the sharded path ---------------------------------------------------------
@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 1, 1)])
@pytest.mark.parametrize("ionization", [False, True])
def test_the_sharded_m2_half_step_is_the_single_device_one(ionization, shape):
    """The halo supplies the ghosts of the moments (one axis per sweep) and of
    the frozen closure (every axis, once per step); the local pad does on one
    device. Same answer. On two GPUs, (2,1,1) runs the fused kernels."""
    if jax.device_count() < int(np.prod(shape)):
        pytest.skip(f"the {shape} mesh needs {int(np.prod(shape))} devices")
    from dataclasses import replace
    from jax.sharding import NamedSharding, PartitionSpec as P
    from luna.sources.radiation import setup_radiation
    n = 16
    cfg = _cfg(n, rad_closure="m2_learned")
    if ionization:
        cfg = replace(cfg, rad_groups=1, rad_ionization=True, rad_E0=0.0, rad_kappa_a=0.0,
                      rad_kappa_s=0.0, rad_source=0.0, BOUNDARY=1, rad_boundary=1,
                      rad_sigma=50.0, rad_alpha_B=0.3, rad_m_H=1.0, rad_photon_rate=20.0,
                      rad_cs2_neutral=0.01, rad_cs2_ionized=1.0, rad_source_radius=0.1,
                      gamma=5.0 / 3.0)
    many = NamedSharding(_mesh(shape), P(None, "x", "y", "z"))
    rad1 = setup_radiation(cfg, _one(), is_main=False)
    rad8 = setup_radiation(cfg, many, is_main=False)
    U = _gas(n)
    U8 = jax.device_put(U, NamedSharding(many.mesh, P(None, "x", "y", "z")))
    if ionization:
        c1, c8, h, steps = rad1["R0"], rad8["R0"], 2e-2, 3
    else:
        c1 = _m2(n)
        c8 = jax.device_put(c1, rad8["sharding"])
        h, steps = 1e-3, 1
    for _ in range(steps):
        U, c1 = rad1["before"](U, c1, h)
        U8, c8 = rad8["before"](U8, c8, h)
    np.testing.assert_allclose(np.asarray(U8), np.asarray(U), rtol=1e-11, atol=1e-13)
    for a, b in zip(jax.tree.leaves(c1), jax.tree.leaves(c8)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-11, atol=1e-13)
    if ionization:
        assert float(np.max(np.asarray(c1[1]))) > 0.5, "nothing was ionised"


# -- the default is bitwise what it was ---------------------------------------
# sha256 of the default (M1) stage's output on the inputs below, per (backend,
# jax version), written by main at c5e92d2 -- before rad_closure existed.
DEFAULT_SHA = {
    ("cpu", "0.10.0"): {
        "R_after": "409f283d0a025b4c7a7bc4b4b9182c028d8574da4977535d95abf6662754664f",
        "R_before": "c1be14ecd8de64e2fc1cab88842ecbe3d9704ee47d1a4d194d0f1b88c8733c1a",
        "R_ion": "3585cfbcb9302583cd9d7303c3357890129e67205d75424111370763d57c0878",
        "U_after": "1d60d68ef8efa4fcb315de319d3e131c8521c3b24c85f6a790176b2b0a029b66",
        "U_before": "6a6e193a1a8b1f0d5714abcb3bfc0df686c7cc3d5ffc514e5b984d7d4c4e024b",
        "U_ion": "98e4aaad5c321c050926e0840edd6f3757f270ca7099100df93ca0a59bf5e8a5",
        "x_ion": "29080ba42c258ec14c153d2e1f1f57f7e1c2add312446d38aeb8d2ff485ba77d",
    },
    # MI300A, fused M1 kernels. No U_ion: main itself gave two different
    # hashes for it in two jobs (every other output reproduced in five runs)
    ("gpu", "0.10.2"): {
        "R_after": "200bfd27f2fb035bc0992ed0d703e208470f6945deeb98e95b5b539143ba8b58",
        "R_before": "e14a9c21de275214e040bbdbac55822c49b112e9d3d60f42100863bd83c638d6",
        "R_ion": "bb6d5efcb43ceb27e18a8fe9aafe70c8be86c1f6fb71faec7d44e3f9d18962f9",
        "U_after": "0fd7c1049e48f32ef9bea80f875eaf8d6d90262828cb2945df66902d7f469bc8",
        "U_before": "fcdaf93d4ab979e419706eec2fd215007a4aea8dac3c712915d5ac3448de4304",
        "x_ion": "acc09ebff251672433ba4e3f3569081b60899f5d4730b21c71ac613004e1656b",
    },
}


def default_outputs():
    """The default radiation stage on fixed inputs: the coupled M1 half-steps
    (two groups, opacities, source, force) and three ionising half-steps."""
    from dataclasses import replace
    from luna.sources.radiation import setup_radiation
    n = 16
    cfg = _cfg(n)
    rad = setup_radiation(cfg, _one(), is_main=False)
    rng = np.random.default_rng(3)
    U = _gas(n)
    E = rng.uniform(0.1, 1.0, size=(2, n, n, n))
    d = rng.normal(size=(2, 3, n, n, n))
    d /= np.sqrt((d ** 2).sum(axis=1))[:, None]
    f = rng.uniform(0.0, 0.9, size=(2, 1, n, n, n))
    R = jnp.asarray(np.concatenate([E[:, None], CHAT * E[:, None] * f * d], axis=1))
    U1, R1 = rad["before"](U, R, 1e-3)
    U2, R2 = rad["after"](U1, R1, 1e-3)
    out = {"U_before": U1, "R_before": R1, "U_after": U2, "R_after": R2}
    icfg = replace(cfg, rad_groups=1, rad_ionization=True, rad_E0=0.0, rad_kappa_a=0.0,
                   rad_kappa_s=0.0, rad_source=0.0, BOUNDARY=1, rad_boundary=1,
                   rad_sigma=50.0, rad_alpha_B=0.3, rad_m_H=1.0, rad_photon_rate=20.0,
                   rad_cs2_neutral=0.01, rad_cs2_ionized=1.0, rad_source_radius=0.1,
                   gamma=5.0 / 3.0)
    irad = setup_radiation(icfg, _one(), is_main=False)
    c, V = irad["R0"], U
    for _ in range(3):
        V, c = irad["before"](V, c, 2e-2)
    out.update(U_ion=V, R_ion=c[0], x_ion=c[1])
    return out


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()


def test_the_default_stage_is_bitwise_what_it_was():
    """rad_closure defaults to "m1", and the M1 stage -- coupled and ionising,
    the fused kernel on a GPU -- gives the same bits as before the option
    existed. A backend or jax version with no recorded reference skips."""
    from luna.template_config import Config
    assert Config().rad_closure == "m1"
    key = (jax.default_backend(), jax.__version__)
    ref = DEFAULT_SHA.get(key, {})
    if not ref:
        pytest.skip(f"no reference recorded for {key}")
    got = {k: sha(v) for k, v in default_outputs().items()}
    bad = [k for k in ref if got[k] != ref[k]]
    assert not bad, f"the default stage changed: {bad} (jax {jax.__version__})"


def test_the_config_refuses_m2_where_it_does_not_apply():
    from dataclasses import replace
    _cfg(8, rad_closure="m2_learned").validate()
    with pytest.raises(ValueError, match="rad_closure"):
        replace(_cfg(8), rad_closure="m3").validate()
    with pytest.raises(ValueError, match="hancock_split"):
        replace(_cfg(8), rad_closure="m2_learned", rad_scheme="rk2").validate()
    with pytest.raises(ValueError, match="SPATIAL_DIMENSION=3"):
        replace(_cfg(8), rad_closure="m2_learned", SPATIAL_DIMENSION=2).validate()


def test_the_entropy_closure_refuses_the_fused_kernel():
    """rad_kernel='pallas' with m2_entropy used to run XLA without a word: the
    fused sweep has no entropy solve. The config, the stage and the sweep
    refuse it; 'auto' picks XLA."""
    from dataclasses import replace
    from luna.radiation import m2_split
    from luna.sources.radiation import setup_radiation
    with pytest.raises(ValueError, match="rad_kernel"):
        replace(_cfg(8), rad_closure="m2_entropy", rad_kernel="pallas").validate()
    replace(_cfg(8), rad_closure="m2_entropy", rad_kernel="auto").validate()
    with pytest.raises(ValueError, match="rad_kernel"):
        setup_radiation(_cfg(8, rad_closure="m2_entropy", rad_kernel="pallas"), _one(), is_main=False)
    U = _m2(4, g=1)[0]
    with pytest.raises(ValueError, match="rad_kernel"):
        m2_split.split_step(U, 0.01, 0.25, chat=CHAT, closure_fn=m2_3d.third_moment, kernel="pallas")


def test_the_stage_solves_the_entropy_closure_in_blocks(monkeypatch):
    """Unblocked, the stage's entropy solve at 128^3 asked XLA for 110-163 GiB
    on a 40 GB A100; in blocks of m2_stage.ENTROPY_CHUNK states it fits."""
    from luna.radiation import m2_stage
    seen = []
    real = m2_3d.third_moment

    def spy(U, *a, **kw):
        seen.append(kw.get("chunk", 0))
        return real(U, *a, **kw)
    monkeypatch.setattr(m2_3d, "third_moment", spy)
    R = _m2(4, g=1)
    m2_stage.transport(R, 0.01, 0.25, chat=CHAT, boundary=0, pad=None, closure="m2_entropy",
                       kernel="xla")
    assert seen and all(c == m2_stage.ENTROPY_CHUNK for c in seen), seen


def test_the_shipped_network_loads():
    from luna.radiation import m2_stage
    params = m2_stage.load_params()
    assert [p[0].shape for p in params] == [(32, 8), (32, 32), (7, 32)]
