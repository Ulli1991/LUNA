"""LUNA-emergence (BLASTWAVE) through main.run.

The diffhydro Fig. 12 top-row analog, wired through the autodiff orchestrator
(autodiff_task="luna_emergence", driver="blast") instead of the old standalone
loop. The free parameter is the FULL per-cell initial internal-energy field
e(x,0) of a uniform medium at rest; reverse-mode AD through the production 2D
solver arranges scattered Sedov sites whose expanding shells collide and merge
into the target word 'LUNA' at late time.

Env: LUNA_EMERG_NX (256), LUNA_EMERG_NSTEPS (300), LUNA_EMERG_NITER (200),
     LUNA_EMERG_TARGET ("LUNA").
"""
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/raven/u/uli/jax-hydro")

from dataclasses import replace

from main import run
from template_config import Config


def main():
    Nx = int(os.environ.get("LUNA_EMERG_NX", "256"))
    n_steps = int(os.environ.get("LUNA_EMERG_NSTEPS", "300"))
    n_iter = int(os.environ.get("LUNA_EMERG_NITER", "200"))
    target = os.environ.get("LUNA_EMERG_TARGET", "LUNA")
    # optional: emerge an arbitrary IMAGE target (a [0,1] .npy/png) instead of a
    # rendered word -- e.g. the cropped you+LUNA silhouette on a plain background.
    image = os.environ.get("LUNA_EMERG_IMAGE", "")

    cfg = replace(
        Config(),
        test_problem="Kelvin-Helmholtz",   # placeholder 2D problem; the IC is rebuilt
        SPATIAL_DIMENSION=2, Nx=Nx, Ny=Nx, Lx=1.0, Ly=1.0,
        gamma=5.0 / 3.0, BOUNDARY=0, ORDER=2, LIMITER=1, riemann_solver="hllc",
        autodiff_mode=True, autodiff_task="luna_emergence",
        autodiff_emergence_driver="blast",
        autodiff_emergence_target=(target if not image else
                                   os.path.splitext(os.path.basename(image))[0]),
        autodiff_emergence_image=image,
        autodiff_n_steps=n_steps, autodiff_n_iter=n_iter,
        PLOT_AFTER_SIM=False, production_mode=True,
    )
    run(cfg)


if __name__ == "__main__":
    main()
