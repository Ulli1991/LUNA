"""The numerics options that used to be LUNA_* environment variables are
Config fields, and the fields are what reaches the solvers; the old
variables are not read any more."""


def test_config_declares_the_fields():
    from luna.template_config import Config
    cfg = Config()
    assert cfg.hydro_fused is False and cfg.hydro_recon32 is False
    assert cfg.gravity_mg_ca == "auto" and cfg.gravity_mg_smoother == "rbgs"
    assert cfg.amr_reflux == "repl"
    assert cfg.gravity_mg_agg_n == 128 and cfg.POISSON_N_COARSEST == 50
    assert cfg.gravity_warm_tol == 1e-4 and cfg.gravity_warm_max_cycles == 10
    assert cfg.gravity_warm_extrapolate is True and cfg.gravity_proper_mixed is True


def test_mg_options_reach_the_ladder(monkeypatch):
    import numpy as np
    import jax
    from jax.sharding import Mesh
    import pytest
    from luna.gravity import poisson_multigrid as pm
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    monkeypatch.setenv("LUNA_MG_MIXED", "0")     # retired: no effect
    assert not hasattr(pm, "_MG_OPTS")           # passed per solve, no process-wide state
    mesh = Mesh(np.asarray(jax.devices()[:2]).reshape(2, 1, 1), ("x", "y", "z"))
    *_, fn, ca, smoo = pm._shardmap_ladder((16, 16, 16), 0, mesh, None, 2, 2, 50, 0,
                                           ca="1", smoother="cheb")
    assert (ca, smoo) == ("1", "cheb") and fn is not None
    *_, fn, ca, smoo = pm._shardmap_ladder((16, 16, 16), 0, mesh, None, 2, 2, 50, 0)
    assert (ca, smoo) == ("auto", "rbgs") and fn is None


def test_hydro_paths_ignore_the_retired_variables(monkeypatch):
    from luna.time_integration.integrators import _hydro_paths
    monkeypatch.setenv("LUNA_HYDRO_FUSED", "1")
    monkeypatch.setenv("LUNA_HYDRO_RECON32", "1")
    assert _hydro_paths(2, 3, 0, False, False)[:2] == (False, False)
    assert _hydro_paths(2, 3, 0, True, False)[:2] == (True, False)


def test_proper_time_precision_switch():
    from dataclasses import replace
    from luna.setup.cosmo_common import gravity_precision_for
    from luna.template_config import Config
    cfg = Config()
    assert gravity_precision_for(cfg, False).mode == "mixed"
    assert gravity_precision_for(replace(cfg, gravity_proper_mixed=False), False).mode == "double"
    assert gravity_precision_for(replace(cfg, gravity_precision="single"), True).mode == "single"
    iso = replace(cfg, gravity_proper_mixed=True, POISSON_BOUNDARY=3)
    assert gravity_precision_for(iso, False).mode == "mixed"
    iso.validate()


def test_unsupported_warm_options_drop_out():
    from luna.gravity.backend import PoissonOpts, supported_opts
    o = PoissonOpts(warm_cycles=2, warm_tol=3e-3, extrapolate=True)
    assert supported_opts(o, adjoint=True).warm_tol == 0.0
    assert supported_opts(o, adjoint=True).extrapolate is True
    assert supported_opts(o, adjoint=False) == o
