"""In-situ analysis: every statistic against numpy on the gathered field, N devices against one, and a full run."""
import json
from dataclasses import replace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

GAMMA = 5.0 / 3.0
N = 16
MESHES = [(2, 2, 2), (1, 2, 4), (4, 2, 1)]


def _mesh(shape):
    devs = np.array(jax.devices()[:int(np.prod(shape))]).reshape(shape)
    return Mesh(devs, ("x", "y", "z"))


def _state(seed=0, n=N, sigma_s=0.8, mach=3.0):
    """A generic supersonic-looking state: log-normal rho, random v, isothermal E (cs = 1)."""
    r = np.random.default_rng(seed)
    rho = np.exp(sigma_s * r.standard_normal((n, n, n)))
    v = mach * r.standard_normal((3, n, n, n))
    E = rho / (GAMMA - 1.0) + 0.5 * rho * np.sum(v * v, axis=0)
    return np.concatenate([rho[None], rho * v, E[None]]).astype(np.float64)


def _put(U, mesh):
    return jax.device_put(jnp.asarray(U), NamedSharding(mesh, P(None, "x", "y", "z")))


def _host(out):
    return {k: np.asarray(v.addressable_data(0)) for k, v in out.items()}


# ---------------- PDFs and moments -------------------------------------------------------------
def test_pdfs_and_budgets_match_numpy():
    from luna.analysis.pdfs import build_stats
    U = _state()
    out = _host(build_stats(_mesh((2, 2, 2)), GAMMA, 1.0 / N, N, isothermal=True)(_put(U, _mesh((2, 2, 2)))))
    rho, v = U[0], U[1:4] / U[0]
    s = np.log(rho / rho.mean())
    e = out["edges_s"]
    ref, _ = np.histogram(s, bins=e)
    np.testing.assert_allclose(out["pdf_vol_s"][1:-1] * N ** 3, ref, rtol=0, atol=1e-9)
    assert abs(out["pdf_vol_s"].sum() - 1.0) < 1e-12           # with under/overflow
    assert abs(out["pdf_mass_s"].sum() - 1.0) < 1e-12
    np.testing.assert_allclose(out["moments_vol_s"], [np.mean(s ** p) for p in range(1, 5)], rtol=1e-12)
    np.testing.assert_allclose(out["mass"], rho.sum(), rtol=1e-13)
    np.testing.assert_allclose(out["momentum"], U[1:4].sum(axis=(1, 2, 3)), rtol=1e-12, atol=1e-9)
    mach = np.sqrt(np.sum(v * v, axis=0))                        # cs = 1
    np.testing.assert_allclose(out["mach_rms_vol"], np.sqrt(np.mean(mach ** 2)), rtol=1e-12)


@pytest.mark.parametrize("shape", MESHES)
def test_stats_are_decomposition_independent(shape):
    from luna.analysis.pdfs import build_stats
    U = _state(1)
    one = _host(build_stats(_mesh((1, 1, 1)), GAMMA, 1.0 / N, N)(_put(U, _mesh((1, 1, 1)))))
    many = _host(build_stats(_mesh(shape), GAMMA, 1.0 / N, N)(_put(U, _mesh(shape))))
    for k in one:
        np.testing.assert_allclose(many[k], one[k], rtol=1e-12, atol=1e-12, err_msg=k)


def test_vorticity_of_a_shear_wave():
    """v_y = sin(2 pi x): omega_z = 2 pi cos(2 pi x), so the enstrophy is pi^2 (central differences: less)."""
    from luna.analysis.pdfs import build_stats
    n = 32
    x = (np.arange(n) + 0.5) / n
    rho = np.ones((n, n, n))
    vy = np.broadcast_to(np.sin(2 * np.pi * x)[:, None, None], (n, n, n))
    U = np.stack([rho, 0 * rho, vy, 0 * rho, rho / (GAMMA - 1) + 0.5 * vy ** 2])
    out = _host(build_stats(_mesh((2, 2, 2)), GAMMA, 1.0 / n, n)(_put(U, _mesh((2, 2, 2)))))
    exact = np.pi ** 2 * (np.sin(2 * np.pi / n) / (2 * np.pi / n)) ** 2   # the central difference's response
    np.testing.assert_allclose(out["enstrophy"], exact, rtol=1e-12)


# ---------------- spectra ----------------------------------------------------------------------
def _numpy_shells(f, n):
    F = np.fft.fftn(f)
    k = np.fft.fftfreq(n, 1.0 / n)
    kk = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    sh = np.floor(kk + 0.5).astype(int)
    from luna.analysis.spectra import n_shells
    return np.bincount(sh.ravel(), (np.abs(F) ** 2).ravel(), minlength=n_shells(n)) / n ** 6


@pytest.mark.parametrize("shape", MESHES)
def test_spectra_match_numpy_and_parseval(shape):
    from luna.analysis.spectra import build_spectra
    U = _state(2)
    out = _host(build_spectra(_mesh(shape), GAMMA, 1.0, N)(_put(U, _mesh(shape))))
    rho, v = U[0], U[1:4] / U[0]
    np.testing.assert_allclose(out["E_rho"], _numpy_shells(rho, N), rtol=1e-11, atol=1e-14)
    ref_v = sum(_numpy_shells(v[i], N) for i in range(3))
    np.testing.assert_allclose(out["E_v"], ref_v, rtol=1e-11, atol=1e-14)
    assert abs(out["E_v"].sum() - np.mean(np.sum(v * v, axis=0))) < 1e-10 * out["E_v"].sum()
    split = out["E_v_solenoidal"] + out["E_v_compressive"]
    np.testing.assert_allclose(split[1:], out["E_v"][1:], rtol=1e-11)   # k = 0 is the mean flow


def test_single_precision_spectra_agree_to_fp32_round_off():
    from luna.analysis.spectra import build_spectra
    U = _state(7)
    out = _host(build_spectra(_mesh((2, 2, 2)), GAMMA, 1.0, N, single=True)(_put(U, _mesh((2, 2, 2)))))
    assert out["E_v"].dtype == np.float32
    ref = sum(_numpy_shells(U[1 + i] / U[0], N) for i in range(3))
    np.testing.assert_allclose(out["E_v"], ref, rtol=2e-5, atol=1e-7 * ref.max())


def test_helmholtz_split_and_vorticity_of_pure_fields():
    """A gradient field has no solenoidal power, a curl field no compressive; E_omega = (2 pi k)^2 E_v."""
    from luna.analysis.spectra import build_spectra
    n = N
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k = 3
    rho = np.ones((n, n, n))
    for v, zero in ((np.stack([np.sin(2 * np.pi * k * X), 0 * X, 0 * X]), "E_v_solenoidal"),
                    (np.stack([0 * X, np.sin(2 * np.pi * k * X), 0 * X]), "E_v_compressive")):
        U = np.concatenate([rho[None], v, (rho / (GAMMA - 1) + 0.5 * np.sum(v * v, 0))[None]])
        out = _host(build_spectra(_mesh((2, 2, 2)), GAMMA, 1.0, n)(_put(U, _mesh((2, 2, 2)))))
        assert out[zero].max() < 1e-20
        assert abs(out["E_v"][k] - 0.5) < 1e-12 and abs(out["E_v"].sum() - 0.5) < 1e-12
        if zero == "E_v_compressive":
            np.testing.assert_allclose(out["E_vorticity"][k], (2 * np.pi * k) ** 2 * 0.5, rtol=1e-12)


def test_broken_pencil_transpose_is_caught(monkeypatch):
    """Teeth: transposing over 'y' alone (forgetting 'z') must fail the numpy comparison."""
    import luna.analysis.spectra as sp

    def broken(x):
        x = jax.lax.all_to_all(x, "z", 1, 2, tiled=True)
        x = jnp.fft.fft(x, axis=2)
        x = jax.lax.all_to_all(x, "y", 2, 1, tiled=True)
        x = jnp.fft.fft(x, axis=1)
        x = jax.lax.all_to_all(x, "x", 1, 0, tiled=True)
        return jnp.fft.fft(x, axis=0)
    monkeypatch.setattr(sp, "fft3_pencil", broken)
    U = _state(3)
    try:
        out = _host(sp.build_spectra(_mesh((2, 2, 2)), GAMMA, 1.0, N)(_put(U, _mesh((2, 2, 2)))))
    except Exception:
        return                                       # a shape error also counts as caught
    assert not np.allclose(out["E_rho"], _numpy_shells(U[0], N), rtol=1e-6)


def test_spectra_refuse_an_indivisible_mesh():
    from luna.analysis.spectra import check_pencil
    with pytest.raises(ValueError, match="needs"):
        check_pencil(12, (1, 2, 4))                  # N / py = 6 is not divisible by pz = 4


# ---------------- structure functions ----------------------------------------------------------
def _numpy_sf(f, lags, orders):
    SL, ST, S3 = [], [], []
    for lag in lags:
        sl, st, s3 = 0.0, 0.0, 0.0
        for d in range(3):
            du = np.roll(f, -lag, axis=1 + d) - f
            sl = sl + np.array([np.mean(np.abs(du[d]) ** p) for p in range(1, orders + 1)])
            st = st + np.array([np.mean(np.abs(np.stack([du[e] for e in range(3) if e != d])) ** p)
                                for p in range(1, orders + 1)])
            s3 = s3 + np.mean(du[d] ** 3)
        SL.append(sl / 3)
        ST.append(st / 3)
        S3.append(s3 / 3)
    return np.array(SL), np.array(ST), np.array(S3)


@pytest.mark.parametrize("shape", MESHES)
def test_structure_functions_match_numpy(shape):
    from luna.analysis.structure import build_structure_functions, default_lags
    U = _state(4)
    out = _host(build_structure_functions(_mesh(shape), GAMMA, N, orders=6)(_put(U, _mesh(shape))))
    rho, v = U[0], U[1:4] / U[0]
    lags = default_lags(N)
    assert tuple(out["lags"]) == lags == (1, 2, 4, 8)
    for name, f in (("v", v), ("w", rho ** (1 / 3) * v)):
        SL, ST, S3 = _numpy_sf(f, lags, 6)
        np.testing.assert_allclose(out["SL_" + name], SL, rtol=1e-11, err_msg=name)
        np.testing.assert_allclose(out["ST_" + name], ST, rtol=1e-11, err_msg=name)
        np.testing.assert_allclose(out["S3L_" + name], S3, rtol=1e-9, atol=1e-9, err_msg=name)


def test_second_order_structure_function_of_a_sine():
    """v_x = sin(2 pi x): S_2^L(l) = (1 - cos(2 pi l / N)) / 3 (only the x-lags see it)."""
    from luna.analysis.structure import build_structure_functions
    x = (np.arange(N) + 0.5) / N
    vx = np.broadcast_to(np.sin(2 * np.pi * x)[:, None, None], (N, N, N))
    rho = np.ones((N, N, N))
    U = np.stack([rho, vx, 0 * rho, 0 * rho, rho / (GAMMA - 1) + 0.5 * vx ** 2])
    out = _host(build_structure_functions(_mesh((2, 2, 2)), GAMMA, N, orders=2)(_put(U, _mesh((2, 2, 2)))))
    lags = np.asarray(out["lags"])
    np.testing.assert_allclose(out["SL_v"][:, 1], (1 - np.cos(2 * np.pi * lags / N)) / 3, rtol=1e-12)


# ---------------- snipshots and frames ---------------------------------------------------------
@pytest.mark.parametrize("mode", ["mean", "stride"])
def test_snipshot_round_trip_conserves_and_carries_its_geometry(tmp_path, mode, monkeypatch):
    from luna.analysis.reduced import build_coarsen, snip_header, write_snipshot
    f = 4
    mesh = _mesh((2, 2, 2))
    U = _state(5)
    Uc = build_coarsen(mesh, f, mode, isothermal=True)(_put(U, mesh))
    hdr = snip_header({"N": N, "box": 1.0, "time": 0.5, "step": 7}, f, mode, True, N, 1.0)
    out = tmp_path / "snip"
    write_snipshot(Uc, hdr, str(out))
    monkeypatch.chdir(tmp_path)                       # the virtual datasets resolve next to snip.h5
    with h5py.File(out / "snip.h5", "r") as h:
        rho, mom = h["rho"][...], h["mom"][...]
        attrs = dict(h.attrs)
        assert set(h.keys()) == {"rho", "mom"}        # no positions, no energy
    c = N // f
    if mode == "mean":
        ref = U[:4].reshape(4, c, f, c, f, c, f).mean(axis=(2, 4, 6))
        assert abs(rho.sum() * f ** 3 - U[0].sum()) < 1e-4 * U[0].sum()   # stored in FP32
    else:
        ref = U[:4, f // 2::f, f // 2::f, f // 2::f]
    np.testing.assert_allclose(rho, ref[0], rtol=1e-6)
    np.testing.assert_allclose(mom, ref[1:4], rtol=1e-6, atol=1e-6)
    assert json.loads(attrs["coarse_shape"]) == [c] * 3 and "positions" in attrs
    with h5py.File(out / "part.00000.h5", "r") as h:
        assert len(h.keys()) == 2                     # one slab per x-block, not one per GPU
    assert "not stored" in attrs["energy"]


def test_frames_match_numpy_projection_and_slice(tmp_path):
    from luna.analysis.reduced import build_frames, write_frame
    mesh = _mesh((2, 2, 2))
    U = _state(6, sigma_s=0.4)
    fn, rng = build_frames(mesh, GAMMA, 1.0, N, 2, ("column_density", "slice_s"))
    out = fn(_put(U, mesh))
    rho = U[0]
    for name, x in (("column_density", np.log10(rho.sum(axis=2) / N / rho.mean())),
                    ("slice_s", np.log(rho[:, :, N // 2] / rho.mean()))):
        lo, hi = rng[name]
        q = np.round((np.clip(x, lo, hi) - lo) / (hi - lo) * 65535).astype(np.uint16)
        img = np.asarray(out[name][0].addressable_data(0))
        assert img.shape == (N, N)
        assert np.abs(img.astype(int) - q.astype(int)).max() <= 1
    path = tmp_path / "frame.h5"
    write_frame(str(path), img, {"field": "slice_s"}, png_max=8)
    assert path.exists() and (tmp_path / "frame.png").exists()


# ---------------- the run loop -----------------------------------------------------------------
def test_driven_turbulence_run_writes_every_product(tmp_path):
    """The dummy example: a 16^3 driven box on a 2x2x2 mesh, every analysis on, a few steps."""
    from luna.analysis.config import InsituConfig
    from luna.hydro.turbulence import OUDriverConfig
    from luna.template_config import Config
    from luna.setup.assemble import assemble
    from luna.setup.evolve import evolve
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    ou = OUDriverConfig(k_min=1.0, k_max=2.0, solenoidal_weight=0.5, autocorrelation_time=0.5,
                        rms_acceleration=4.0, seed=7)
    ins = InsituConfig(stats_dt=1e-3, spectra_dt=2e-3, sf_dt=2e-3, snip_dt=2e-3, frame_dt=1e-3,
                       snip_factor=2, frame_png_max=16)
    cfg = replace(Config(), test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
                  Nx=N, Ny=N, Nz=N, gamma=GAMMA, ORDER=1, LIMITER=0, t_end=1.0, max_steps=6,
                  ou_config=ou, ou_seed=7, isothermal_cs2=1.0, MESH_SHAPE=(2, 2, 2),
                  write_snapshots=False, PLOT_AFTER_SIM=False, enable_checkpointing=False,
                  enable_x64=True, output_dir=str(tmp_path), insitu=ins)
    cfg.validate()
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, is_main=False)
    physics = assemble(cfg, mesh, kind, layout, init, is_main=False)
    evolve(cfg, init, physics, mesh, is_main=True)
    root = tmp_path / "insitu"
    for kind_ in ("stats", "spectra", "sf"):
        files = sorted((root / kind_).glob("*.h5"))
        assert files, kind_
        with h5py.File(files[-1], "r") as h:
            assert "time" in h.attrs and "config_json" in h.attrs and "code_revision" in h.attrs
    with h5py.File(sorted((root / "stats").glob("*.h5"))[0], "r") as h0, \
            h5py.File(sorted((root / "stats").glob("*.h5"))[-1], "r") as h1:
        assert abs(h1["mass"][()] - h0["mass"][()]) < 1e-12 * h0["mass"][()]   # periodic: mass conserved
    snips = sorted((root / "snip").glob("*/snip.h5"))
    assert snips
    assert sorted((root / "frames" / "column_density").glob("*.png"))
    assert json.loads((root / "header.json").read_text())["N"] == N


def test_config_refuses_insitu_in_2d():
    from luna.analysis.config import InsituConfig
    from luna.template_config import Config
    cfg = replace(Config(), SPATIAL_DIMENSION=2, Nx=16, Ny=16, Nz=1, insitu=InsituConfig(stats_dt=1.0))
    with pytest.raises(ValueError, match="insitu"):
        cfg.validate()
