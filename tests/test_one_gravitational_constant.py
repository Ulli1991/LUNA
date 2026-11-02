"""There must be exactly one G, derived from the CGS constants.

Three copies existed: `units.UnitSystem.GravityConstant` (derived), and the
literal 43007.1 in both `sources.external_gravity` and `cosmology.music_ic`. The
literal implies Msun = 1.98832e33 g against units.py's 1.98892e33, so the three
disagreed by 3.0e-4 and the two literals ignored `cfg.units` entirely -- any run
in a non-Arepo unit system got an external gravity field inconsistent with its
own units.
"""


from luna.cosmology.music_ic import G_AREPO                    # noqa: E402
from luna.sources.external_gravity import G_CODE                 # noqa: E402
from luna.units import arepo_unit_system                       # noqa: E402


def test_all_three_agree_exactly():
    g = arepo_unit_system().GravityConstant
    assert G_CODE == g, f"external_gravity G_CODE {G_CODE} != units {g}"
    assert G_AREPO == g, f"music_ic G_AREPO {G_AREPO} != units {g}"


def test_g_is_derived_not_a_literal():
    """Change the CGS constants and G must follow; a literal would not."""
    from luna import units
    g0 = arepo_unit_system().GravityConstant
    old = units.GRAVITY_CGS
    try:
        units.GRAVITY_CGS = old * 2.0
        assert abs(arepo_unit_system().GravityConstant - 2.0 * g0) < 1e-6 * g0
    finally:
        units.GRAVITY_CGS = old


def test_matches_the_textbook_arepo_value_to_a_few_parts_in_1e4():
    """Sanity: still the Arepo constant, just with a current solar mass."""
    assert abs(arepo_unit_system().GravityConstant - 43007.1) / 43007.1 < 1e-3


def _music_cfg():
    """A run config built from a minimal MUSIC IC header."""
    import numpy as np
    from luna.cosmology.music_ic import DMICs, music_run_config
    z = 49.0
    ic = DMICs(pos=np.zeros((1, 3)), vel=np.zeros((1, 3)), mass=np.ones(1),
               box_size=10000.0, a=1.0 / (1.0 + z), redshift=z,
               Omega_m=0.3, Omega_L=0.7, h=0.7)
    return music_run_config(ic, 32)


def test_a_music_run_carries_the_arepo_unit_system():
    """The run config must carry cfg.units, not just a bare G.

    Without it a MUSIC run is in dimensionless mode: every unit-dependent
    module (cooling, CC85) refuses to start, and the writers record no units
    group, so a snapshot cannot be converted back to CGS.
    """
    u = _music_cfg().units
    assert u.is_enabled, "MUSIC run config is still in dimensionless mode"
    assert u.UnitLength_in_cm == arepo_unit_system().UnitLength_in_cm
    assert u.UnitMass_in_g == arepo_unit_system().UnitMass_in_g
    assert u.UnitVelocity_in_cm_per_s == arepo_unit_system().UnitVelocity_in_cm_per_s


def test_the_music_g_is_the_one_its_own_units_imply():
    """G on the config and G derived from the config's units are one number."""
    cfg = _music_cfg()
    assert cfg.GRAVITATIONAL_G == cfg.units.GravityConstant
