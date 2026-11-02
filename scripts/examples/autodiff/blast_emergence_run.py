"""LUNA-emergence by blast waves (the luna_emergence example).

    python -m scripts.examples.autodiff.blast_emergence_run nx=256 n_steps=300 n_iter=200
    python -m scripts.examples.autodiff.blast_emergence_run image=target.png seeds=20

The diffhydro Fig. 12 top-row analog: autodiff_task="luna_emergence" with the
blast driver. The free parameter is the full per-cell initial internal-energy
field of a uniform medium at rest; reverse-mode AD arranges scattered Sedov
sites whose shells merge into the target word at late time.

Parameters: nx, n_steps, and the task's options (luna_emergence.py): n_iter,
target (a word), image (a [0,1] .npy/png target field, overrides the word),
seeds (scattered hot sites in e(x,0)), movie=<dir>, movie_every,
movie_nframe, hostloop=1, maxit_per_proc, ckpt.
"""
import functools
from dataclasses import replace


from luna.setup.cli import script_main            # noqa: E402
from luna.template_config import Config           # noqa: E402
from scripts.examples.autodiff import luna_emergence


TASK = functools.partial(luna_emergence.task, driver="blast", n_iter=200)


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
