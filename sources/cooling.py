"""Runner-side wiring for the radiative cooling source term.

Builds an ExplicitCoolingSolver from cfg, suitable for use as an
operator-split step:  U = apply_cooling(U, dt, cfg, solver).
"""

from hydro.cooling import ExplicitCoolingSolver, sutherland_dopita_cie


def setup_cooling(cfg, is_main_process: bool = True):
    """Return the ExplicitCoolingSolver or None when disabled."""
    if not getattr(cfg, "cooling_enabled", False):
        return None
    if not cfg.units.is_enabled:
        raise ValueError("cooling_enabled requires cfg.units (Arepo system).")
    if cfg.cooling_table == "sd93_solar":
        table = sutherland_dopita_cie(metallicity="solar")
    else:
        raise ValueError(f"Unknown cooling_table: {cfg.cooling_table}")
    mu = getattr(cfg, "cc85_mu", 0.6)
    n_sub = getattr(cfg, "cooling_n_sub", 16)
    solver = ExplicitCoolingSolver(table=table, mu=mu, n_sub=n_sub)
    if is_main_process:
        print(
            f"✓ Cooling enabled: table={cfg.cooling_table}, "
            f"T range [{table.T.min():.2e}, {table.T.max():.2e}] K, mu={mu}"
        )
    return solver
