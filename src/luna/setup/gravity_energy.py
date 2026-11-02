"""Resolve gas gravity energy coupling for both hydro and gravity setup."""


def flux_requirements(cfg):
    """Requirements unmet by the face mass flux coupling: an explicit 'flux' refuses any of them."""
    return [w for w, bad in (
        ("SELF_GRAVITY=True (or CosmologicalIntegration)", not (cfg.SELF_GRAVITY or cfg.CosmologicalIntegration)),
        ("SPATIAL_DIMENSION=3", cfg.SPATIAL_DIMENSION != 3),
        ("ORDER=1", cfg.ORDER != 1),
        ("hydro_fused=False", cfg.hydro_fused),
        ("no AMR", cfg.AMR),
        ("run='gas' or 'gas+dm'", cfg.run not in ("gas", "gas+dm")),
        ("no sinks", cfg.sinks_enabled),
        ("no chemistry scalars", cfg.chemistry == "net5"),
        # mass changed outside the hydro's fluxes breaks W^n+1 - W^n = sum (rho^n+1 - rho^n) Phi
        ("no SiL corrector (sil_weights: its own mass fluxes)", bool(cfg.sil_weights)),
        ("cosmo_rho_floor=0 (the floor adds mass)", cfg.cosmo_rho_floor > 0.0),
        ("no cc85 wind (it injects mass)", cfg.cc85_enabled),
    ) if bad]


def flux_not_default(cfg):
    """What an explicit 'flux' runs but 'auto' does not choose yet (no GPU validation of the default): auto is 'kick'."""
    return [w for w, bad in (
        ("a proper clock (comoving flux is explicit only)", cfg.CosmologicalIntegration),
    ) if bad]


def resolve_gravity_energy(cfg):
    """Choose a supported mode; explicit requests never fall back."""
    mode = cfg.gravity_energy
    if mode not in ("auto", "kick", "flux"):
        raise ValueError(f"gravity_energy={mode!r}: expected auto, kick or flux")
    if mode == "kick":
        return mode
    why = flux_requirements(cfg)
    if mode == "flux":
        if why:
            raise ValueError("gravity_energy='flux' needs " + ", ".join(why))
        return mode
    return "kick" if why or flux_not_default(cfg) else "flux"
