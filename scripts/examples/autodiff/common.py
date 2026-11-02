"""Helpers the autodiff examples share: the rollout with its header line, and
the finite-difference check every gradient is held to."""
from luna.autodiff.rollout import build_simulate


def central_fd(f, x, i, h=1.0e-4):
    """Central finite difference of scalar f(x) along element/index i."""
    return (float(f(x.at[i].add(h))) - float(f(x.at[i].add(-h)))) / (2.0 * h)


def rollout(init, cfg, physics, is_main, name):
    """build_simulate, and the line saying what it built."""
    simulate, step_once, carry0, dt_info, eta_true = build_simulate(
        init, cfg, physics, is_main)
    if is_main:
        print(f"  autodiff: task={name}, n_steps(max)={int(cfg.autodiff_n_steps)}, "
              f"dt={dt_info}, eta_true={eta_true}", flush=True)
    return simulate, step_once, carry0, eta_true
