# ic.py
from .sod import sod_shock
from .kh import kelvin_helmholtz

def initial_conditions(cfg):
    """
    Select initial conditions based on cfg.test_problem
    """

    if cfg.test_problem == "Sod-Shock":
        return sod_shock(cfg)
    elif cfg.test_problem == "Kelvin-Helmholtz":
        return kelvin_helmholtz(cfg)
    else:
        raise ValueError(
            f"Unknown test problem '{cfg.test_problem}'. "
            "Available options: 'Sod-Shock', 'Kelvin-Helmholtz'."
        )
