"""LUNA-emergence (TURBULENCE) through main.run.

The diffhydro Fig. 12 bottom-row analog, wired through the autodiff orchestrator
(autodiff_task="luna_emergence", driver="turbulence") instead of the old
standalone loop. The free parameter is a band-limited initial velocity field
v(x,0) (a coarse Nc x Nc control grid upsampled to the mesh) of an isothermal,
uniform-density gas; reverse-mode AD through the production 2D solver arranges
the convergent (compressive) zones so the gas piles up into the target word
'LUNA' at late time -- the velocity field itself does not spell anything.

Env: LUNA_EMERG_NX (256), LUNA_EMERG_NC (32), LUNA_EMERG_NSTEPS (300),
     LUNA_EMERG_NITER (200), LUNA_EMERG_MACH (3.0), LUNA_EMERG_TARGET ("LUNA").
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
    Nc = int(os.environ.get("LUNA_EMERG_NC", "32"))
    n_steps = int(os.environ.get("LUNA_EMERG_NSTEPS", "300"))
    n_iter = int(os.environ.get("LUNA_EMERG_NITER", "200"))
    mach = float(os.environ.get("LUNA_EMERG_MACH", "3.0"))
    target = os.environ.get("LUNA_EMERG_TARGET", "LUNA")

    cfg = replace(
        Config(),
        test_problem="Kelvin-Helmholtz",   # placeholder 2D problem; the IC is rebuilt
        SPATIAL_DIMENSION=2, Nx=Nx, Ny=Nx, Lx=1.0, Ly=1.0,
        gamma=5.0 / 3.0, BOUNDARY=0, ORDER=2, LIMITER=1, riemann_solver="hllc",
        autodiff_mode=True, autodiff_task="luna_emergence",
        autodiff_emergence_driver="turbulence", autodiff_emergence_target=target,
        autodiff_emergence_nc=Nc, autodiff_emergence_mach=mach,
        autodiff_emergence_cfl=0.20, autodiff_emergence_lr=0.15,
        autodiff_n_steps=n_steps, autodiff_n_iter=n_iter,
        PLOT_AFTER_SIM=False, production_mode=True,
    )
    run(cfg)


if __name__ == "__main__":
    main()
