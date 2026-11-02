"""What kind of run a config describes, decided in one place.

Three kinds -- gas, dark matter, or both coupled -- and three independent
modifiers: comoving time, refinement, adjoint mode. `run_kind(cfg)` names the
kind; `check_modifiers(cfg)` refuses a combination no runner can do, instead of
the dispatch silently running whichever branch sits first.
"""

GAS = "gas"
DM = "dm"
COUPLED = "gas+dm"
KINDS = (GAS, DM, COUPLED)


def run_kind(cfg):
    """The kind of run, as `cfg.run` names it."""
    if cfg.run not in KINDS:
        raise ValueError(f"run={cfg.run!r}: expected one of {KINDS}")
    return cfg.run


def check_modifiers(cfg):
    """Refuse a modifier the chosen kind's runner cannot honour.

    Each of these used to run: the dispatch picked one branch and the other
    flag was silently ignored.
    """
    kind = run_kind(cfg)
    if cfg.autodiff_mode:
        if kind != GAS:
            raise ValueError(f"autodiff_mode runs the dense gas loop only, not {kind}")
        if cfg.AMR:
            raise ValueError("autodiff_mode and AMR: the adjoint loop has no refinement")
        if cfg.CosmologicalIntegration:
            raise ValueError("autodiff_mode and CosmologicalIntegration: the adjoint "
                             "loop integrates proper time only")
    if cfg.AMR and kind == DM:
        raise ValueError("AMR and run='dm': the refinement runner has no dark-matter-"
                         "only path (it does gas, and gas+dm)")
    return kind
