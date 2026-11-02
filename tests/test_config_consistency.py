"""Config is the single source of truth: every ``getattr(cfg, "name", default)``
in library code must name a declared Config field.

Pure AST scan, no JAX. Fallback defaults that differ from the template are
reported (they are dead for configs built from ``Config()`` but mislead
readers); the differing set is pinned so it can only shrink.
"""
import ast
import glob
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join("src", "luna")          # the package, under the src layout
LIBRARY_DIRS = tuple(os.path.join(PKG, d) for d in
                     ("amr", "cosmology", "file_io", "gravity", "hydro",
                      "ic_generation", "radiation", "setup", "sources", "species",
                      "time_integration", "utils"))

# getattr fallbacks known to differ from the template default (2026-09-16).
# Shrink this set as the call sites are switched to plain ``cfg.name``.
# Emptied 2026-09-18: every library getattr(cfg, "<template field>", default) was
# replaced by a plain attribute read (377 sites); the only getattr(cfg, ...) left name
# AMRConfig fields (prolong_limiter, lohner_var, lohner_eps) that live on acfg.
KNOWN_DIFFERING = set()


def _class_fields(path, cls_name):
    src = open(os.path.join(ROOT, path)).read()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fields = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            fields[node.target.id] = node.value
    return fields


def _template_fields():
    return _class_fields(os.path.join(PKG, "template_config.py"), "Config")


def _amr_fields():
    """amr/ kernels receive the AMRConfig as `cfg`; its fields count as declared."""
    return _class_fields(os.path.join(PKG, "amr", "config.py"), "AMRConfig")


def _getattr_sites():
    for d in LIBRARY_DIRS:
        for f in glob.glob(os.path.join(ROOT, d, "**", "*.py"), recursive=True):
            tree = ast.parse(open(f).read())
            for n in ast.walk(tree):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == "getattr" and len(n.args) >= 2
                        and isinstance(n.args[0], ast.Name) and n.args[0].id == "cfg"
                        and isinstance(n.args[1], ast.Constant)):
                    rel = os.path.relpath(f, ROOT)
                    yield rel, n.lineno, n.args[1].value, (n.args[2] if len(n.args) == 3 else None)


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def test_every_getattr_names_a_declared_field():
    fields = _template_fields()
    amr = _amr_fields()
    undeclared = sorted({(name, f"{rel}:{ln}") for rel, ln, name, _ in _getattr_sites()
                         if name not in fields
                         and not (rel.startswith(os.path.join(PKG, "amr") + os.sep)
                                  and name in amr)})
    assert not undeclared, f"getattr(cfg, name) on undeclared fields: {undeclared}"


def test_getattr_fallbacks_match_template_or_are_known():
    fields = _template_fields()
    differing = set()
    for _rel, _ln, name, default in _getattr_sites():
        if default is None or name not in fields or fields[name] is None:
            continue
        d, t = _literal(default), _literal(fields[name])
        if d is None or t is None:
            continue
        if d != t:
            differing.add(name)
    new = differing - KNOWN_DIFFERING
    assert not new, f"new getattr fallbacks that contradict template_config.py: {sorted(new)}"


def test_config_instantiates_and_validates():
    pytest.importorskip("numpy")
    from luna.template_config import Config
    cfg = Config()
    assert len(_template_fields()) == len([f for f in type(cfg).__dataclass_fields__])
    cfg.validate()


def test_validate_rejects_the_removed_lhllc_solver():
    """L-HLLC was removed: asking for it fails with a message saying so; the
    two solvers left validate."""
    from dataclasses import replace
    from luna.template_config import Config
    with pytest.raises(ValueError, match="riemann_solver='lhllc': the L-HLLC solver was removed"):
        replace(Config(), riemann_solver="lhllc").validate()
    with pytest.raises(ValueError, match="riemann_solver='roe': expected one of"):
        replace(Config(), riemann_solver="roe").validate()
    for rs in ("hllc", "hll"):
        for sdim in (1, 2, 3):
            replace(Config(), riemann_solver=rs, SPATIAL_DIMENSION=sdim).validate()


def test_validate_warns_on_known_open_traps():
    """Audit sec.10 'validate() extensions (warn first)': configurations that run
    but hit a known-open bug or a silently ignored setting must warn, and a
    default Config must validate silently."""
    import warnings
    from dataclasses import replace
    import pytest
    from luna.template_config import Config
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a clean default must not warn
        Config().validate()
    # multi-GPU 2D periodic seams were fixed (b47b192) and the warning that
    # steered runs off them dropped (cd7cdc5): this must be silent now
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        replace(Config(), SPATIAL_DIMENSION=2, BOUNDARY=0, MESH_SHAPE=(2, 1, 1)).validate()
    # a comoving run restores from its checkpoint set: silent too
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        replace(Config(), restart=True, CosmologicalIntegration=True,
                enable_x64=True).validate()
    # the adjoint mode still starts from the initial condition -> warn
    cfg = replace(Config(), restart=True, autodiff_mode=True)
    with pytest.warns(UserWarning, match="restart=True is not honoured"):
        cfg.validate()


# Each option that used to pass validate() and then run something else, or fail
# deep inside a runner (docs/CONFIG_REDESIGN.md sec.1): (overrides, message).
UNIMPLEMENTED = [
    ({"ORDER": 2}, "ORDER=2: there is no PPM"),
    ({"ORDER": 3}, "ORDER=3: expected one of"),
    ({"BOUNDARY": 3}, "BOUNDARY=3: isolated is a gravity boundary only"),
    ({"POISSON_BOUNDARY": 4}, "POISSON_BOUNDARY=4: expected one of"),
    ({"run": "auto"}, "run='auto': expected one of"),
    ({"decomposition": "particles"}, "decomposition='particles': expected one of"),
    ({"cooling_table": "ki"}, "cooling_table='ki': spell it"),
    ({"cooling_table": "cie"}, "cooling_table='cie': expected one of"),
    ({"hydro_kernel": "cuda"}, "hydro_kernel='cuda': expected one of"),
    ({"gravity_kernel": "cuda"}, "gravity_kernel='cuda': expected one of"),
    ({"rad_kernel": "cuda"}, "rad_kernel='cuda': expected one of"),
    ({"rad_scheme": "rk3"}, "rad_scheme='rk3': expected one of"),
    ({"rad_order": 3}, "rad_order=3: expected one of"),
    ({"rad_ion_solver": "newton"}, "rad_ion_solver='newton': expected one of"),
    ({"gravity_mg_ca": "on"}, "gravity_mg_ca='on': expected one of"),
    ({"gravity_mg_smoother": "jacobi"}, "gravity_mg_smoother='jacobi': expected one of"),
    ({"particle_boundary": "reflect"}, "particle_boundary='reflect': expected one of"),
    ({"amr_reflux": "psum"}, "amr_reflux='psum': expected one of"),
    # hydro_kernel='pallas' used to run XLA silently outside its scope (A10, 2026-10-12)
    ({"hydro_kernel": "pallas", "ORDER": 0}, "hydro_kernel='pallas' cannot run ORDER=0"),
    ({"hydro_kernel": "pallas", "AMR": True}, "hydro_kernel='pallas' cannot run AMR"),
    ({"hydro_kernel": "pallas", "autodiff_mode": True},
     "hydro_kernel='pallas' cannot run autodiff_mode"),
    ({"hydro_kernel": "pallas", "SPATIAL_DIMENSION": 3, "hydro_fused": True},
     "hydro_kernel='pallas' cannot run hydro_fused=True"),
    ({"hydro_kernel": "pallas", "SPATIAL_DIMENSION": 2, "hydro_recon32": True},
     "hydro_kernel='pallas' cannot run hydro_recon32 in 2-D"),
    ({"hydro_kernel": "pallas", "BOUNDARY": 4}, "hydro_kernel='pallas' cannot run BOUNDARY=4 in 1-D"),
]


@pytest.mark.parametrize("over,msg", UNIMPLEMENTED, ids=[m.split(":")[0] for _, m in UNIMPLEMENTED])
def test_validate_refuses_what_is_not_implemented(over, msg):
    import re
    from dataclasses import replace
    from luna.template_config import Config
    with pytest.raises(ValueError, match=re.escape(msg)):
        replace(Config(), **over).validate()


IMPLEMENTED = [
    {"ORDER": 0}, {"ORDER": 1}, {"BOUNDARY": 4, "POISSON_BOUNDARY": 3},
    {"riemann_solver": "hll", "SPATIAL_DIMENSION": 3},
    {"amr_criterion": "static"}, {"cooling_table": "koyama_inutsuka"},
    {"cooling_table": "combined"}, {"rad_scheme": "rk2", "rad_order": 1},
    {"rad_ion_solver": "split"}, {"gravity_mg_ca": "1", "gravity_mg_smoother": "cheb"},
    {"particle_boundary": "outflow"}, {"amr_reflux": "iface"},
    {"decomposition": "amr_blocks"}, {"hydro_kernel": "xla", "gravity_kernel": "pallas"},
    {"hydro_kernel": "pallas", "LIMITER": 1},
    {"hydro_kernel": "pallas", "SPATIAL_DIMENSION": 2, "LIMITER": 2, "riemann_solver": "hll"},
    {"hydro_kernel": "pallas", "SPATIAL_DIMENSION": 3, "hydro_recon32": True, "BOUNDARY": 4},
    {"hydro_kernel": "auto", "ORDER": 0},
]


@pytest.mark.parametrize("over", IMPLEMENTED, ids=[str(o) for o in IMPLEMENTED])
def test_validate_accepts_what_is_implemented(over):
    """amr_criterion='static' is the driver's geometric refinement, refused
    before although implemented."""
    from dataclasses import replace
    from luna.template_config import Config
    replace(Config(), **over).validate()
