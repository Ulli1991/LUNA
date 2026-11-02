"""config.py as overrides on template_config.Config, and the table of retired
field names (docs/CONFIG_REDESIGN.md)."""

# The problem fields that moved into their IC module's Params: old name ->
# (test problems, name in cfg.problem).
_PROBLEM_FIELDS = {
    "jeans_": (("Jeans-Instability-3D",), ("rho0", "P0", "amp", "kx", "ky", "kz", "eigenmode")),
    "evrard_": (("Evrard-Collapse-3D",), ("M", "R", "u", "rho_ambient")),
    "be_": (("Bonnor-Ebert-3D",), ("xi_t", "cs2", "R", "rho_ambient")),
    "lane_": (("Lane-Emden-3D", "Pulsating-Polytrope-3D"), ("n", "rho_c", "R", "rho_ambient")),
    "mac_": (("Maclaurin-Spheroid-3D",), ("e", "a", "rho", "rho_ambient")),
    "bb_": (("Boss-Bodenheimer-3D",), ("rho0_cgs", "R_pc", "omega_rad_s", "amp", "m",
                                       "rho_ambient_frac", "beta")),
    "kh_": (("Kelvin-Helmholtz",), ("noise_amp",)),
    "soundwave_": (("Soundwave",), ("single_mode",)),
    "turb_": (("Driven-Turbulence-3D",), ("rho0", "P0")),
    "gresho_": (("Gresho-Vortex-2D",), ("Mmax",)),
    "hii_": (("HII-Region-3D",), ("rho0",)),
    "disc_": (("Disc-Planet-2D",), ("sigma0", "sigma_index", "R_in", "R_out", "floor",
                                    "edge_width")),
    "zel_": (("Zeldovich-Pancake",), ("a_caustic", "cs_frac")),
    "m82_disk_": (("M82-Disk-3D",), ("M_gas", "R_scale_kpc", "R_trunc_kpc", "T_K")),
    "cc85_": (("CC85-Wind-3D", "M82-Disk-3D"), ("n_ambient_cgs", "T_ambient_K")),
    "cloud_": (("Turbulent-Sphere-3D",), ("mass_msun", "radius_pc", "T_K", "mach", "turb_slope",
                                          "amb_frac", "profile", "edge_ratio", "amb_T_K",
                                          "amb_mu", "amb_n_cm3", "turb_spectrum", "turb_v0_kms",
                                          "turb_kmin", "turb_kmax", "zero_drift")),
}

# Every moved field: old flat name -> its new key ("problem.x" or a Config field).
MOVED = {
    **{prefix + n: f"problem.{n}" for prefix, (_, names) in _PROBLEM_FIELDS.items()
       for n in names},
    "pulse_amp": "problem.amp",
    "cc85_mu": "cooling_mu",
    "cloud_mu": "sink_star_mu",
    "cloud_seed": "seed",
}


def _moved_message(old):
    new = MOVED[old]
    if not new.startswith("problem."):
        return f"now {new}"
    probs = ("Pulsating-Polytrope-3D",) if old == "pulse_amp" else next(
        p for prefix, (p, names) in _PROBLEM_FIELDS.items()
        if old.startswith(prefix) and old[len(prefix):] in names)
    return (f"now {new} (test_problem {' or '.join(map(repr, probs))}): {new}=... on the "
            f"command line, problem=<IC module>.Params({new[8:]}=...) in a config file")


# Fields that left Config, and where their setting went. Kept one release so an
# old config.py, run script or command line fails with the new place named.
RETIRED = {
    "production_mode": "removed: nothing read it; delete the setting",
    "check_memory": "removed: nothing read it; delete the setting",
    "bb_rho_ambient": "removed: nothing read it (problem.rho_ambient_frac sets the ambient)",
    "be_rho_c": "removed: the Bonnor-Ebert IC always derives rho_c from problem.xi_t, R "
                "and cs2; delete the setting",
    "CosmoDM": "use run='dm'",
    "CosmoHydro": "use run='gas+dm'",
    "dm_fp32": "use gravity_precision='single'",
    "dm_mixed_precision": "use gravity_precision: 'mixed' (was True) or 'double' (was False)",
    "DECOMPOSITION": "removed: MESH_SHAPE names the process grid, None picks one",
    "cfl_rule": "removed: CFL is always the RAMSES rule; an old config uses the default CFL",
    # the autodiff experiments' parameters are the example tasks' own options
    # (scripts/examples/autodiff/), set as name=value after the run script
    "autodiff_n_iter": "a task option now: n_iter=N after the run script",
    **{f"autodiff_ic_{k}": f"an ic_inversion task option now: {k}=..."
       for k in ("field", "kmax", "slope", "lr")},
    **{f"autodiff_emergence_{k}": f"a luna_emergence task option now: {v}=..."
       for k, v in (("driver", "driver"), ("target", "target"), ("image", "image"),
                    ("init_seeds", "seeds"), ("emax", "emax"), ("contrast", "contrast"),
                    ("match", "match"), ("cfl", "cfl"), ("lr", "lr"), ("nc", "nc"),
                    ("mach", "mach"))},
    **{old: _moved_message(old) for old in MOVED},
}


def retired(name):
    """The message for a retired field name, or None."""
    msg = RETIRED.get(name)
    return None if msg is None else f"Config field {name!r} is retired: {msg}"


def translate(values):
    """`values` with each moved flat name under its new key (the AMUSE worker's old keys)."""
    out = {}
    for k, v in values.items():
        new = MOVED.get(k, k)
        if new != k:
            print(f"LUNA config: {k} is {new} now; translated")
        out[new] = v
    return out


def check_names(names, known=None):
    """Raise on any retired field name among `names`, and, given the `known`
    field names, on any other name that is not one of them."""
    import difflib
    msgs = [retired(n) for n in names if retired(n)]
    for n in names:
        if known is not None and n not in known and not retired(n):
            close = difflib.get_close_matches(n, known, n=3)
            hint = f" (did you mean {', '.join(close)}?)" if close else ""
            msgs.append(f"{n!r} is not a Config field{hint}")
    if msgs:
        raise ValueError("\n".join(msgs))


def from_user_config(user, warn=True):
    """Project a user config (the gitignored config.py's Config, often a stale
    copy of the template with fields missing) onto template_config.Config, so
    every run sees the full field set with template defaults for anything the
    user's file lacks, and validate() runs. Warns about template fields the user
    file is missing (defaults used); an attribute the template does not know (a
    typo, or a retired field) raises."""
    import warnings
    from dataclasses import fields, replace
    from luna.template_config import Config as Template
    names = [f.name for f in fields(Template)]
    have = {n: getattr(user, n) for n in names if hasattr(user, n)}
    missing = [n for n in names if n not in have]
    unknown = sorted(k for k, v in vars(user).items()
                     if not k.startswith("_") and k not in names and not callable(v))
    check_names(unknown, names)
    cfg = replace(Template(), **have)
    if warn and missing:
        warnings.warn(f"config.py vs template_config.py: {len(missing)} template "
                      f"field(s) missing from config.py, template defaults used: "
                      f"{missing}", stacklevel=2)
    return cfg
