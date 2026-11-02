"""Run one shipped run script as `python -m <module> <argv>` would, and write what it did as JSON.

    python tests/_run_script_smoke.py OUT.json scripts.validation.runs.kh_run nx=16 max_steps=2
"""
import json
import os
import runpy
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cell_volume(cfg):
    """The volume (area, length) of one cell of the grid `cfg` describes."""
    d = cfg.SPATIAL_DIMENSION
    return cfg.Lx / cfg.Nx * (cfg.Ly / cfg.Ny if d > 1 else 1.0) * (cfg.Lz / cfg.Nz if d > 2 else 1.0)


def _gas_mass(U, dV):
    """Total gas mass, whether every cell is finite, and the dtype."""
    U = np.asarray(U)
    return float(np.sum(U[0], dtype=np.float64) * dV), bool(np.isfinite(U).all()), str(U.dtype)


def _cc85_injected(cfg, elapsed):
    """The mass CC85 puts into the injection cells over `elapsed`, from the run's own parameters."""
    from luna.sources.cc85 import _MSUN_G, _PC_CM, _SEC_PER_YR
    u = cfg.units
    mdot = cfg.cc85_eta_M * cfg.cc85_SFR_Msun_per_yr * _MSUN_G / _SEC_PER_YR / u.UnitMass_in_g * u.UnitTime_in_s
    R = cfg.cc85_R_star_pc * _PC_CM / u.UnitLength_in_cm
    c = [(np.arange(n, dtype=np.float32) + 0.5) * (L / n) - 0.5 * L
         for n, L in ((cfg.Nx, cfg.Lx), (cfg.Ny, cfg.Ly), (cfg.Nz, cfg.Lz))]
    X, Y, Z = np.meshgrid(*c, indexing="ij")
    n_inj = int(np.sum(X * X + Y * Y + Z * Z < R * R))
    return mdot / (4.0 / 3.0 * np.pi * R ** 3) * elapsed * n_inj * _cell_volume(cfg)


def _dense_snapshots(out):
    """(mass, finite, dtype) of the first and the last dense snapshot a refinement run wrote."""
    import glob
    from luna.file_io.read_snapshot import read_snapshot
    dirs = sorted(glob.glob(os.path.join(out, "snapdir_*")))
    rows = []
    for d in (dirs[0], dirs[-1]):
        s = read_snapshot(d)
        dV = s["dx"] * s.get("dy", 1.0) * s.get("dz", 1.0)
        rho = np.asarray(s["rho"])
        rows.append((float(np.sum(rho, dtype=np.float64) * dV),
                     bool(np.isfinite(rho).all() and np.isfinite(np.asarray(s["P"])).all()), str(rho.dtype)))
    return rows


def _leaves_finite(x):
    """True if every number reachable in `x` (dicts, lists, arrays) is finite."""
    if isinstance(x, dict):
        return all(_leaves_finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(_leaves_finite(v) for v in x)
    a = np.asarray(x)
    return bool(np.isfinite(a).all()) if a.dtype.kind in "fc" else True


def main(out, module, argv):
    import luna.main
    import luna.setup.ics
    rec = {"module": module, "argv": argv}
    read_ics, run = luna.setup.ics.read_ics, luna.main.run

    def spy_ics(cfg, *a, **k):
        init = read_ics(cfg, *a, **k)
        rec["ic"] = True
        rec["t0"] = float(init.cfg.a_init) if init.cfg.CosmologicalIntegration else float(init.t_start)
        if init.U is not None:
            rec["dV"] = _cell_volume(init.cfg)
            rec["m0"], rec["ic_finite"], rec["dtype"] = _gas_mass(init.U, rec["dV"])
        if init.dm is not None:
            rec["n_dm0"] = int(np.sum(np.asarray(init.dm.active) > 0)) if init.dm.active is not None \
                else int(np.asarray(init.dm.pos).shape[0])
            rec["m_dm0"] = float(np.sum(np.asarray(init.dm.mass, np.float64)))
        return init

    def spy_run(cfg, task=None, **kw):
        rec["boundary"] = int(cfg.BOUNDARY)
        res = run(cfg, task, **kw)
        rec["t"], rec["n_steps"] = res.t, res.n_steps
        ex = res.extra or {}
        if "U" in ex:
            rec["m1"], rec["finite"], _ = _gas_mass(ex["U"], rec["dV"])
        elif cfg.AMR:
            first, last = _dense_snapshots(cfg.output_dir)
            rec["m0"], rec["ic_finite"], rec["dtype"] = first
            rec["m1"], rec["finite"], _ = last
        if cfg.cc85_enabled and res.t is not None:
            rec["injected"] = _cc85_injected(cfg, res.t - rec["t0"])
        if "boundary_mass" in ex:
            rec["budget"] = np.asarray(ex["boundary_mass"], np.float64).tolist()
        if "dm_pos" in ex:
            act = np.asarray(ex["dm_active"]) > 0
            rec["n_dm1"] = int(act.sum())
            rec["m_dm1"] = None if ex.get("mass") is None else float(np.asarray(ex["mass"]))
            rec["dm_finite"] = bool(np.isfinite(np.asarray(ex["dm_pos"])[act]).all())
        if cfg.autodiff_mode:
            rec["task_sizes"] = {str(k): len(v) if isinstance(v, (list, tuple, dict)) else int(np.size(v))
                                 for k, v in ex.items()}
            rec["task_finite"] = _leaves_finite(ex)
        return res

    luna.setup.ics.read_ics, luna.main.run = spy_ics, spy_run
    sys.argv = [module] + list(argv)
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    rec["done"] = True
    with open(out, "w") as fh:
        json.dump(rec, fh)


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
