"""The initial conditions written once and read back per rank.

At 1024^3 the turbulent cloud is built once (`write_ic`) and each rank reads
its own cells (`cfg.ic_dir`). What comes back has to be what the generator
would have built, bit for bit: the whole box, any rank's cells, through the
dispatcher. A file written for another config is refused.

Also pinned: the generator's own per-rank path. It used to index with the
(start, end) pairs the ranks are given, taking two cells instead of a range.
"""
from dataclasses import replace

import numpy as np
import pytest

from luna.ic_generation.turbulent_sphere_3d import Params
from luna.ic_generation.ic import initial_conditions
from luna.ic_generation.ic_file import read_ic, write_ic
from luna.ic_generation.turbulent_sphere_3d import turbulent_sphere_3d
from luna.units import UnitSystem
from tests._helpers import _with_problem

N = 24
RANGES = ((6, 18), (0, 12), (12, 24))


def _cloud(**over):
    units = UnitSystem(UnitLength_in_cm=3.085677581e18, UnitMass_in_g=1.98892e33,
                       UnitVelocity_in_cm_per_s=1.0e5)
    base = dict(test_problem="Turbulent-Sphere-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
                Nx=N, Ny=N, Nz=N, Lx=8.0, Ly=8.0, Lz=8.0, gamma=5.0 / 3.0, units=units,
                mass_msun=2000.0, radius_pc=2.0, T_K=15.0,
                mach=8.0, sink_star_mu=2.3, seed=3)
    base.update(over)
    return _with_problem(Params, **base)


def _sub(U, ranges):
    (x0, x1), (y0, y1), (z0, z1) = ranges
    return U[:, x0:x1, y0:y1, z0:z1]


@pytest.fixture
def written(tmp_path):
    cfg = _cloud()
    write_ic(cfg, str(tmp_path), slab=5)          # 5 does not divide 24: a ragged last slab
    return replace(cfg, ic_dir=str(tmp_path)), turbulent_sphere_3d(cfg)


def test_the_whole_box_reads_back_bitwise(written):
    cfg, U = written
    np.testing.assert_array_equal(read_ic(cfg), U)


def test_a_ranks_cells_read_back_bitwise(written):
    cfg, U = written
    np.testing.assert_array_equal(read_ic(cfg, *RANGES), _sub(U, RANGES))


def test_the_dispatcher_reads_the_file_when_ic_dir_is_set(written):
    cfg, U = written
    np.testing.assert_array_equal(initial_conditions(cfg, *RANGES), _sub(U, RANGES))


def test_the_generators_own_rank_path_takes_a_range():
    cfg = _cloud()
    np.testing.assert_array_equal(turbulent_sphere_3d(cfg, *RANGES),
                                  _sub(turbulent_sphere_3d(cfg), RANGES))


def test_a_file_for_another_cloud_is_refused(written):
    cfg, _ = written
    with pytest.raises(ValueError, match="cloud_mach"):
        read_ic(replace(cfg, problem=replace(cfg.problem, mach=4.0)))
    with pytest.raises(ValueError, match="Nx"):
        read_ic(replace(cfg, Nx=32))
