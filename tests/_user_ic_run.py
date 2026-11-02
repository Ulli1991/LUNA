"""User initial states (launch(cfg, ic=...)) in 1D, 2D, 3D for tests/test_user_ic.py, as one process or one of
several: `python tests/_user_ic_run.py out=DIR`; rank 0 writes DIR/U{dim}.npy (the gathered end state)."""
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIMS = (1, 2, 3)


def case(dim, out):
    """A registered problem of dimension dim on a grid that divides over 8 devices, 6 steps, quiet."""
    from luna.template_config import Config
    grid = {1: dict(test_problem="Sod-Shock", Nx=64),
            2: dict(test_problem="Gresho-Vortex-2D", Nx=32, Ny=32, BOUNDARY=0),
            3: dict(test_problem="Sedov-Blast-3D", Nx=16, Ny=16, Nz=16)}[dim]
    return replace(Config(), SPATIAL_DIMENSION=dim, enable_x64=True, t_end=1.0, max_steps=6,
                   write_snapshots=False, PLOT_AFTER_SIM=False, enable_checkpointing=False,
                   enable_diagnostics=False, output_dir=out, clear_output_dir=True, **grid)


def as_user(cfg):
    """cfg renamed to a problem that is not registered: the state must come from U0 / ic."""
    return replace(cfg, test_problem="User-State", problem=None)


def registered_ic(reg):
    """ic(x, y, z, cfg): the registered problem of cfg `reg` on the cells it is handed (from their centres)."""
    import numpy as np
    from luna.ic_generation.ic import initial_conditions

    def ic(x, y, z, cfg):
        cut = {}
        for name, c, n, L in (("x_slice", x, cfg.Nx, cfg.Lx), ("y_slice", y, cfg.Ny, cfg.Ly),
                              ("z_slice", z, cfg.Nz, cfg.Lz)):
            if c is not None:
                i = np.rint(np.asarray(c) * n / L - 0.5).astype(int)
                cut[name] = (int(i.min()), int(i.max()) + 1)
        return np.asarray(initial_conditions(reg, **cut))
    return ic


def main(argv):
    kv = dict(a.split("=", 1) for a in argv)
    import jax
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1 and os.environ.get("JAX_PLATFORMS", "").startswith("cpu"):
        jax.config.update("jax_cpu_collectives_implementation", "gloo")
    import numpy as np
    from jax.experimental import multihost_utils
    from luna import launch
    for dim in DIMS:
        reg = case(dim, os.path.join(kv["out"], f"run{dim}"))
        res = launch(as_user(reg), ic=registered_ic(reg))
        U = np.asarray(multihost_utils.process_allgather(res.extra["U"], tiled=True))
        if jax.process_index() == 0:
            np.save(os.path.join(kv["out"], f"U{dim}.npy"), U)
            print(f"RESULT dim={dim} steps={res.n_steps} processes={jax.process_count()} "
                  f"devices={jax.device_count()}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main(sys.argv[1:])
