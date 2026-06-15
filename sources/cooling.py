"""Runner-side wiring for the radiative cooling source term.

Builds an ExplicitCoolingSolver from cfg, suitable for use as an
operator-split step:  U = apply_cooling(U, dt, cfg, solver).
"""

from hydro.cooling import (ExplicitCoolingSolver, sutherland_dopita_cie,
                           koyama_inutsuka_2002, combine)


def setup_cooling(cfg, is_main_process: bool = True):
    """Return the ExplicitCoolingSolver or None when disabled.

    Tables: "sd93_solar" (Sutherland-Dopita CIE, T>1e4 K, no heating),
    "koyama_inutsuka"/"ki" (KI 2002 two-phase ISM cooling + photoelectric
    heating, for the cold/warm ISM), and "combined" (KI below 1.5e4 K, SD
    above — full two-phase-to-hot coverage).
    """
    if not getattr(cfg, "cooling_enabled", False):
        return None
    if not cfg.units.is_enabled:
        raise ValueError("cooling_enabled requires cfg.units (Arepo system).")
    ct = cfg.cooling_table
    if ct == "sd93_solar":
        table = sutherland_dopita_cie(metallicity="solar")
    elif ct in ("koyama_inutsuka", "ki", "ki02"):
        table = koyama_inutsuka_2002()
    elif ct in ("combined", "ki_sd93"):
        table = combine(koyama_inutsuka_2002(),
                        sutherland_dopita_cie(metallicity="solar"))
    else:
        raise ValueError(f"Unknown cooling_table: {ct}")
    mu = getattr(cfg, "cc85_mu", 0.6)
    n_sub = getattr(cfg, "cooling_n_sub", 16)
    solver = ExplicitCoolingSolver(table=table, mu=mu, n_sub=n_sub)
    if is_main_process:
        print(
            f"✓ Cooling enabled: table={cfg.cooling_table}, "
            f"T range [{table.T.min():.2e}, {table.T.max():.2e}] K, mu={mu}"
        )
    return solver
