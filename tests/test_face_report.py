"""The face report (audit C8): unphysical Hancock faces are counted and located, refused by default, logged when allowed; it never changes the step."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from luna.hydro.conversions import prim_to_cons
from luna.time_integration.integrators import create_muscl_hancock_modular_step
from tests._helpers import gpu_only

G, N = 5.0 / 3.0, 32


def _ramp(dim):
    """An advected 4x-per-cell density ramp into 1e-8 at v = 50 (MC's faces go negative, test_preserve_face_states)."""
    rho = 1e-8 * 4.0 ** np.minimum(np.abs(np.arange(N) - N // 2), 12)
    return _prim(rho, np.full(N, 50.0), np.full(N, 1e-6), dim)


def _smooth(dim):
    """A subsonic sine wave: no face goes unphysical."""
    x = (np.arange(N) + 0.5) / N
    return _prim(1 + 0.3 * np.sin(2 * np.pi * x), 0.2 * np.cos(2 * np.pi * x), np.ones(N), dim)


def _prim(rho, v, p, dim):
    """(dim + 2, N, [4, [4]]) primitives varying along x."""
    sh = (N,) + (4,) * (dim - 1)
    b = lambda a: np.broadcast_to(a.reshape((N,) + (1,) * (dim - 1)), sh)  # noqa: E731
    return np.stack([b(rho), b(v), *[np.zeros(sh)] * (dim - 1), b(p)])


def _steps(dim, allow, report, mesh=None, kernel="xla", limiter=1):
    """(compute_dt, step) of the periodic MC/HLLC step at CFL 0.8."""
    kw = dict(use_shardmap=True, mesh=mesh) if mesh is not None else {}
    return create_muscl_hancock_modular_step(None, G, 1, limiter, 1 / N, 1 / 4, 1 / 4, N, 4, 4, 0, dim, 0.8,
                                             allow_first_order=allow, face_report=report, kernel=kernel, **kw)


def _first_report(W, dim, **kw):
    """(U before, U after, report) of the first of 40 steps with an unphysical face; every U equals the unreported step's."""
    cdt, step = _steps(dim, True, True, **kw)
    _, plain = _steps(dim, True, False, **kw)
    U = prim_to_cons(jnp.asarray(W), G)
    for _ in range(40):
        dt = cdt(U)
        U1, rep = step(U, dt)
        assert np.array_equal(np.asarray(U1), np.asarray(plain(U, dt)))
        if int(np.sum(rep["n"])) > 0:
            return U, dt, U1, rep
        U = U1
    raise AssertionError("no unphysical face in 40 steps")


def _cells(rep):
    c = np.asarray(rep["cells"])
    return {tuple(r) for r in c[(c >= 0).all(axis=1)]}


@pytest.mark.parametrize("dim", [1, 3])
def test_ramp_is_counted_located_and_refused(dim):
    """The ramp: count > 0; the refused step's NaN cells are the reported cells and their x-neighbours; refusal is NaN only there."""
    U, dt, U1, rep = _first_report(_ramp(dim), dim)
    n, cells = int(np.sum(rep["n"])), _cells(rep)
    assert n >= len(cells) > 0 and int(np.sum(rep["n_floor"])) == 0
    _, refuse = _steps(dim, False, False)
    bad = np.argwhere(~np.isfinite(np.asarray(refuse(U, dt))).all(axis=0))
    nan_x = {int(i[0]) for i in bad}
    rep_x = {c[0] for c in cells}
    assert rep_x <= nan_x <= {(i + s) % N for i in rep_x for s in (-1, 0, 1)} or len(cells) == 8
    # the face named: an x+ face alone poisons the flux into cell i + 1, an x- face alone the one into i - 1
    c_all, f_all = np.asarray(rep["cells"]), np.asarray(rep["faces"])
    ok = (c_all >= 0).all(axis=1)
    named = {(c[0] + (1 if m & 3 == 1 else -1)) % N for c, m in zip(c_all[ok], f_all[ok]) if m & 3 in (1, 2)}
    assert (f_all[ok] > 0).all() and named <= nan_x
    ok = np.isfinite(np.asarray(refuse(U, dt))).all(axis=0)
    assert np.array_equal(np.asarray(refuse(U, dt))[:, ok], np.asarray(U1)[:, ok])
    print(f"FACES dim={dim}: n={n} cells={sorted(cells)} NaN x={sorted(nan_x)}")


def test_smooth_reports_zero_and_is_bitwise():
    """The sine wave: 0 faces, no cell, and refuse == allow == unreported, bit for bit (minmod and MC)."""
    for lim in (0, 1):
        cdt, step = _steps(1, False, True, limiter=lim)
        _, allow = _steps(1, True, False, limiter=lim)
        U = prim_to_cons(jnp.asarray(_smooth(1)), G)
        for _ in range(10):
            dt = cdt(U)
            U1, rep = step(U, dt)
            assert int(np.sum(rep["n"])) == 0 and not _cells(rep)
            assert np.array_equal(np.asarray(U1), np.asarray(allow(U, dt)))
            U = U1


def test_sharded_report_equals_one_device():
    """Two shards along x: the same step bits, count and global cells as one device."""
    from jax.sharding import Mesh
    if len(jax.devices()) < 2:
        pytest.skip("needs 2 devices (pt.sh cpu gives 8)")
    U, dt, U1, rep = _first_report(_ramp(1), 1)
    mesh = Mesh(np.asarray(jax.devices()[:2]), ("x",))
    _, step2 = _steps(1, True, True, mesh=mesh)
    U2, rep2 = step2(U, dt)
    assert np.array_equal(np.asarray(U2), np.asarray(U1))
    assert int(np.sum(rep2["n"])) == int(np.sum(rep["n"])) and _cells(rep2) == _cells(rep)


def _cfg(tmp_path, **kw):
    from luna.template_config import Config
    return replace(Config(), SPATIAL_DIMENSION=1, Nx=N, Lx=1.0, BOUNDARY=0, gamma=G, ORDER=1, LIMITER=1,
                   CFL=0.8, t_end=1.0, max_steps=40, write_snapshots=False, PLOT_AFTER_SIM=False,
                   enable_checkpointing=False, enable_diagnostics=False, enable_x64=True,
                   output_dir=str(tmp_path), **kw)


def _run(cfg, W):
    """The production loop (setup.evolve.evolve) from the primitive state W."""
    from luna.setup.assemble import assemble
    from luna.setup.evolve import evolve
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, is_main=False)
    physics = assemble(cfg, mesh, kind, layout, init, is_main=False)
    init.U = jax.device_put(prim_to_cons(jnp.asarray(W), G), physics.hydro.sharding)
    return evolve(cfg, init, physics, mesh, is_main=True)


def test_run_refuses_by_default(tmp_path):
    """allow_first_order_faces=False (default): the run stops, naming step, t and cell."""
    with pytest.raises(RuntimeError, match=r"\[hydro\] step \d+ t=\S+: \d+ unphysical Hancock face states .*"
                                           r"first at the x[+-] face of cell \(\d+,\) x=\(\S+\).*allow_first_order_faces"):
        _run(_cfg(tmp_path), _ramp(1))


def test_run_logs_when_allowed(tmp_path, capsys):
    """allow_first_order_faces=True: each step with one logs when and where, the run end the total."""
    _run(_cfg(tmp_path, allow_first_order_faces=True), _ramp(1))
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[hydro] step")]
    assert lines and "first-order faces (C8), first at the x" in lines[0]
    assert "[hydro] run total:" in out
    print("\n".join(lines[:2]))


def test_smooth_run_is_silent(tmp_path, capsys):
    """A smooth run reports nothing and does not refuse."""
    _run(_cfg(tmp_path), _smooth(1))
    assert "[hydro]" not in capsys.readouterr().out


@gpu_only
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pallas_report_equals_xla_gpu(dim):
    """Pallas: the same step bits and the same count and cells as XLA, and the report changes neither."""
    W = _ramp(dim)
    U, dt, U1, rep = _first_report(W, dim)
    _, pal = _steps(dim, True, True, kernel="pallas")
    _, pal_plain = _steps(dim, True, False, kernel="pallas")
    Up, repp = pal(U, dt)
    assert np.array_equal(np.asarray(Up), np.asarray(pal_plain(U, dt)))
    assert np.array_equal(np.asarray(Up), np.asarray(U1))
    assert int(np.sum(repp["n"])) == int(np.sum(rep["n"])) and _cells(repp) == _cells(rep)
    print(f"FACES pallas dim={dim}: n={int(np.sum(repp['n']))}")


def test_amr_faces_refused_and_reported():
    """AMR (one 32 x 4 block, the 2-D ramp): the block face state rule counts what the dense step counts, refuses
    them as NaN by default, and the runner's flush names level, block, cell and face; allowed, the fluxes are
    finite and the flush logs the same count."""
    from luna.amr.config import AMRConfig
    from luna.amr.hydro import block_face_fluxes, report_amr_faces
    from luna.setup import face_report
    W = _ramp(2)
    U = prim_to_cons(jnp.asarray(W), G)
    cdt, step = _steps(2, True, True)
    dt = cdt(U)
    n_dense = int(np.sum(step(U, dt)[1]["n"]))
    Wg = jnp.pad(jnp.asarray(W), ((0, 0), (2, 2), (2, 2)), mode="wrap")[None]
    for allow in (False, True):
        cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(N, 4), block_shape=(N, 4), boundary=0,
                        allow_first_order_faces=allow)
        F, own = block_face_fluxes(Wg, dt, 0, cfg, gamma=G, limiter=1, report=True)
        n = int(jnp.sum(jax.lax.population_count(own)))
        assert n == n_dense > 0
        assert bool(jnp.isfinite(F[0]).all()) == allow
        report_amr_faces(own, 0, cfg, jnp.zeros((1, 2), jnp.int32))
        jax.effects_barrier()
        if allow:
            assert face_report.flush_amr(cfg, True, (1, 1), 0.0, float(dt)) == n
        else:
            with pytest.raises(RuntimeError, match=r"\[amr\] steps 1\.\.1 .*: \d+ unphysical Hancock face states "
                                                   r".*first at the x[+-] face of level 0 block 0 cell \(\d+, 0\) "
                                                   r"\(block coords \(0, 0\), x=\(\S+, \S+\)\).*Refused"):
                face_report.flush_amr(cfg, False, (1, 1), 0.0, float(dt))
