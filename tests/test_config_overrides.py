"""config.py as overrides on template_config.Config (audit sec.10).

`python main.py` used to run the user's gitignored config.Config directly: a
stale copy is missing fields (204 of 225 at audit time) and has no validate(),
so validation was silently skipped and getattr fallbacks decided the physics.
setup/user_config.py projects the user object onto the template: full field
set, validate() runs; it warns about missing fields and refuses unknown
attributes (a typo used to warn and run with the default).
"""
import warnings
from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.fast


@dataclass
class _StaleUserConfig:
    """A config.py that predates several template fields."""
    Nx: int = 48
    CFL: float = 0.4
    test_problem: str = "Sod-Shock-3D"


@dataclass
class _TypoUserConfig(_StaleUserConfig):
    Nxx: int = 3            # typo: unknown to the template


def test_stale_user_config_is_projected_and_reported():
    from luna.setup.user_config import from_user_config
    from luna.template_config import Config
    with pytest.warns(UserWarning) as rec:
        cfg = from_user_config(_StaleUserConfig())
    msg = "\n".join(str(w.message) for w in rec)
    assert isinstance(cfg, Config)
    assert cfg.Nx == 48 and cfg.CFL == 0.4 and cfg.test_problem == "Sod-Shock-3D"
    assert cfg.enable_x64 is False            # template default for a missing field
    assert hasattr(cfg, "validate")
    assert "missing from config.py" in msg and "'enable_x64'" in msg


def test_an_unknown_attribute_raises():
    """A typo used to warn and run with the template default."""
    from luna.setup.user_config import from_user_config
    with pytest.raises(ValueError, match="'Nxx' is not a Config field.*did you mean Nx"):
        from_user_config(_TypoUserConfig())


def test_complete_template_config_projects_silently():
    from luna.setup.user_config import from_user_config
    from luna.template_config import Config
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = from_user_config(Config())
    assert cfg == Config()


@dataclass
class _RetiredUserConfig:
    """A config.py that still sets a field Config no longer has."""
    Nx: int = 48
    production_mode: bool = True


def test_a_retired_field_raises_with_its_new_place(tmp_path):
    """A retired field fails in every way a setting arrives -- a config.py
    object, a config file that builds a Config, the command line -- and the
    message says what became of it."""
    import textwrap
    from luna.setup.cli import load_config, overrides
    from luna.setup.user_config import from_user_config
    with pytest.raises(ValueError, match="'production_mode' is retired: removed"):
        from_user_config(_RetiredUserConfig())
    f = tmp_path / "old_run.py"
    f.write_text(textwrap.dedent("""
        from dataclasses import replace
        from luna.template_config import Config
        Config = replace(Config(), Nx=48, be_rho_c=2.0)
    """))
    with pytest.raises(ValueError, match="'be_rho_c' is retired"):
        load_config(str(f))
    with pytest.raises(ValueError, match="'check_memory' is retired"):
        overrides({"check_memory": "1"})
    for old, new in (("CosmoDM", "run='dm'"), ("CosmoHydro", "run='gas\\+dm'"),
                     ("dm_fp32", "gravity_precision='single'"),
                     ("dm_mixed_precision", "gravity_precision"),
                     ("DECOMPOSITION", "MESH_SHAPE")):
        with pytest.raises(ValueError, match=f"'{old}' is retired: .*{new}"):
            overrides({old: "1"})
