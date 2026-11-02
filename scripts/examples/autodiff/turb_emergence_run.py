"""LUNA-emergence by turbulence (the luna_emergence example).

    python -m scripts.examples.autodiff.turb_emergence_run nx=256 nc=32 mach=3

The diffhydro Fig. 12 bottom-row analog: autodiff_task="luna_emergence" with
the turbulence driver. The free parameter is a band-limited initial velocity
field (an nc x nc control grid upsampled to the mesh) of an isothermal
uniform gas; reverse-mode AD arranges the convergent zones so the gas piles
up into the target word.

Parameters: nx, n_steps, and the task's options (luna_emergence.py): nc
(control-grid side), n_iter, mach, target, movie=<dir>, movie_every,
movie_nframe, hostloop=1, maxit_per_proc, ckpt.
"""
import functools
from dataclasses import replace


from luna.setup.cli import script_main            # noqa: E402
from luna.template_config import Config           # noqa: E402
from scripts.examples.autodiff import luna_emergence


TASK = functools.partial(luna_emergence.task, driver="turbulence", cfl=0.20, lr=0.15,
                         n_iter=200)


def config(nx=256, n_steps=300):
    return replace(
        Config(),
        test_problem="Kelvin-Helmholtz",   # placeholder 2D problem; the IC is rebuilt
        SPATIAL_DIMENSION=2, Nx=nx, Ny=nx, Lx=1.0, Ly=1.0,
        gamma=5.0 / 3.0, BOUNDARY=0, ORDER=1, LIMITER=1, riemann_solver="hllc",
        autodiff_mode=True, autodiff_task="luna_emergence", autodiff_n_steps=n_steps,
        PLOT_AFTER_SIM=False, enable_x64=True,
    )


if __name__ == "__main__":
    script_main(config, task=TASK)
