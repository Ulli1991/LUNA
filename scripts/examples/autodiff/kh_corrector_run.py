"""A solver-in-the-loop CNN corrector on Kelvin-Helmholtz, in adjoint mode.

    python -m scripts.examples.autodiff.kh_corrector_run nx=32 n_steps=200 n_iter=100

autodiff_task="corrector_training": a CNN flux corrector is trained (reverse
mode, through the production coarse step) so the coarse rollout tracks a fine
reference (the production solver at 4x resolution, downsampled).

Parameters: nx (coarse), ny (default nx; 2 nx for the 1:2 box), n_steps, dt,
and the task's options (corrector_training.py): n_iter, fine_factor (4) and
n_train (0 = n_steps/3).
"""
import functools
from dataclasses import replace


from luna.setup.cli import script_main            # noqa: E402
from luna.template_config import Config           # noqa: E402
from scripts.examples.autodiff import corrector_training


TASK = functools.partial(corrector_training.task, n_iter=100)


def config(nx=32, ny=0, n_steps=200, dt=0.004):
    ny = ny or nx
    Ly = 2.0                  # shear layers at y = 0.5, 1.5; Lx follows (square cells)
    return replace(
        Config(),
        test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2, BOUNDARY=0,
        Nx=nx, Ny=ny, Lx=Ly * nx / ny, Ly=Ly, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1,
        autodiff_mode=True, autodiff_task="corrector_training",
        autodiff_n_steps=n_steps, autodiff_fixed_dt=dt,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/kh_corrector_Nx{nx}", enable_x64=True,
    )


def report(res):
    r = res.extra
    if isinstance(r, dict) and "corr_mse" in r:
        ce, pe = r["corr_mse"], r["coarse_mse"]          # [(t, density MSE), ...]
        print(f"RESULT corrector: coarse-only={pe[-1][1] if pe else 'nan'} "
              f"+CNN={ce[-1][1] if ce else 'nan'}", flush=True)


if __name__ == "__main__":
    script_main(config, report=report, task=TASK)
