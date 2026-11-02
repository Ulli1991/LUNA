"""AMUSE community-code interface for LUNA: in-worker implementation, legacy-function
contract, and the high-level `Luna` grid with FLASH units. See docs/TORCH.md.
"""

import os

import numpy as np

from amuse.community import (
    LegacyFunctionSpecification,
    legacy_function,
    LiteratureReferencesMixIn,
)
from amuse.community.interface.hydro import HydrodynamicsInterface
from amuse.support.interface import InCodeComponentImplementation
from amuse.units import generic_unit_system as gus

from amuse_luna.particles import PARKED_ATTRS

try:
    from amuse.rfi.core import PythonCodeInterface
except Exception:  # pragma: no cover - import-time safety on partial installs
    PythonCodeInterface = object


def _units():
    """Map the parked attributes' unit names to AMUSE generic units (FLASH's choice)."""
    return {
        "mass": gus.mass,
        "time": gus.time,
        "length": gus.length,
        "speed": gus.speed,
        "area": gus.length ** 2,
        "rate": gus.time ** -1,
        "photon_energy": gus.mass * gus.length ** 2 * gus.time ** -2,
        "mass_rate": gus.mass / gus.time,
    }


class LunaImplementation:
    """In-worker object: adapts each AMUSE legacy call to `LunaWorker`, returning an error code."""

    def __init__(self):
        self._worker = None
        self._params = {}
        self._strings = {}

    # -- lifecycle ----------------------------------------------------------
    def set_config_value(self, key, value):
        self._params[str(key)] = value
        return 0

    def set_config_string(self, key, value):
        """Set a string Config field (`set_config_value` marshals float64 only)."""
        self._strings[str(key)] = str(value)
        return 0

    # Config fields that must be ints (set_config_value marshals float64).
    _INT_PARAMS = {"Nx", "Ny", "Nz", "SPATIAL_DIMENSION", "ORDER", "LIMITER",
                   "BOUNDARY", "POISSON_BOUNDARY"}

    def initialize_code(self):
        from luna.template_config import Config
        kwargs = {k: (int(v) if k in self._INT_PARAMS else v)
                  for k, v in self._params.items()}
        kwargs.update(self._strings)
        from luna.setup.cli import apply
        from luna.setup.user_config import check_names, translate
        kwargs = translate(kwargs)          # old flat keys (cloud_*, ...), for one release
        check_names([k for k in kwargs if not k.startswith("problem.")])
        # $LUNA_CONFIG: a config file loaded as main.py would; values set
        # through set_config_value/string override it.
        base = os.environ.get("LUNA_CONFIG")
        if base:
            from luna.setup.cli import load_config
            # A named config means LUNA owns the problem, so build its ICs.
            return self._boot(apply(load_config(base), kwargs), build_ic=True)
        # The orchestrator fills the grid, but `assemble` still builds ICs and
        # the default problem is 1D; the worker is 3D-only.
        kwargs.setdefault("test_problem", "Soundwave-3D")
        kwargs.setdefault("SPATIAL_DIMENSION", 3)
        return self._boot(apply(Config(), kwargs))

    def _boot(self, cfg, build_ic=False):
        from amuse_luna.worker import LunaWorker
        self._worker = LunaWorker(cfg)
        return self._worker.initialize_code(build_ic=build_ic)

    def commit_parameters(self):
        return self._worker.commit_parameters()

    def cleanup_code(self):
        rc = self._worker.cleanup_code() if self._worker else 0
        self._worker = None
        return rc

    # AMUSE passes every declared parameter by keyword; OUT params are
    # ValueHolders (set `.value`), and the method returns the int error code.

    # -- geometry -----------------------------------------------------------
    def get_grid_range(self, imin, imax, jmin, jmax, kmin, kmax, index_of_grid):
        a, b, c, d, e, f = self._worker.get_grid_range()
        imin.value, imax.value = a, b
        jmin.value, jmax.value = c, d
        kmin.value, kmax.value = e, f
        index_of_grid.value = 0
        return 0

    def get_position_of_index(self, i, j, k, index_of_grid, x, y, z, n):
        px, py, pz = self._worker.get_position_of_index(i, j, k)
        x.value, y.value, z.value = px, py, pz
        return 0

    def get_index_of_position(self, x, y, z, i, j, k, n):
        ii, jj, kk = self._worker.get_index_of_position(x, y, z)
        i.value, j.value, k.value = ii, jj, kk
        return 0

    # -- grid state (rhoen is INTERNAL energy density; seam in LunaWorker) ---
    def set_grid_state(self, i, j, k, index_of_grid, rho, rhovx, rhovy, rhovz, rhoen, n):
        return self._worker.set_grid_state(i, j, k, rho, rhovx, rhovy, rhovz, rhoen)

    def get_grid_state(self, i, j, k, index_of_grid, rho, rhovx, rhovy, rhovz, rhoen, n):
        a, b, c, d, e = self._worker.get_grid_state(i, j, k)
        rho.value, rhovx.value, rhovy.value, rhovz.value, rhoen.value = a, b, c, d, e
        return 0

    # Declared by HydrodynamicsInterface; views on get_grid_state. Parameter
    # names must match its declarations, since AMUSE calls by keyword.
    def get_grid_density(self, i, j, k, index_of_grid, rho, number_of_points):
        rho.value = self._worker.get_grid_state(i, j, k)[0]
        return 0

    def set_grid_density(self, i, j, k, rho, index_of_grid, number_of_points):
        _, mx, my, mz, en = self._worker.get_grid_state(i, j, k)
        return self._worker.set_grid_state(i, j, k, rho, mx, my, mz, en)

    def get_grid_momentum_density(self, i, j, k, index_of_grid,
                                  rhovx, rhovy, rhovz, number_of_points):
        _, mx, my, mz, _ = self._worker.get_grid_state(i, j, k)
        rhovx.value, rhovy.value, rhovz.value = mx, my, mz
        return 0

    def set_grid_momentum_density(self, i, j, k, rhovx, rhovy, rhovz,
                                  index_of_grid, number_of_points):
        rho, _, _, _, en = self._worker.get_grid_state(i, j, k)
        return self._worker.set_grid_state(i, j, k, rho, rhovx, rhovy, rhovz, en)

    def get_grid_energy_density(self, i, j, k, index_of_grid, en,
                                number_of_points):
        en.value = self._worker.get_grid_state(i, j, k)[4]
        return 0

    def set_grid_energy_density(self, i, j, k, en, index_of_grid,
                                number_of_points):
        rho, mx, my, mz, _ = self._worker.get_grid_state(i, j, k)
        return self._worker.set_grid_state(i, j, k, rho, mx, my, mz, en)

    def get_number_of_grids(self, n):
        n.value = 1
        return 0

    def initialize_grid(self):
        return 0

    def recommit_parameters(self):
        return self._worker.commit_parameters()

    # -- time / evolution ---------------------------------------------------
    def evolve_model(self, time):
        return self._worker.evolve_model(time)

    def get_time(self, time):
        time.value = self._worker.get_time()
        return 0

    def get_timestep(self, timestep):
        timestep.value = self._worker.get_timestep()
        return 0

    # -- particles ----------------------------------------------------------
    def set_particle_pointers(self, part_type):
        return self._worker.set_particle_pointers(part_type)

    def get_number_of_particles(self, number_of_particles):
        number_of_particles.value = self._worker.get_number_of_particles()
        return 0

    def new_particle(self, tag, x, y, z, n):
        tag.value = self._worker.new_particle(x, y, z)
        return 0

    def delete_particle(self, tag, n):
        return self._worker.delete_particle(tag)

    def set_particle_mass(self, tag, mass, n):
        return self._worker.set_particle_mass(tag, mass)

    def get_particle_mass(self, tag, mass, n):
        mass.value = self._worker.get_particle_mass(tag)
        return 0

    def set_particle_position(self, tag, x, y, z, n):
        return self._worker.set_particle_position(tag, x, y, z)

    def get_particle_position_array(self, tag, x, y, z, n):
        px, py, pz = self._worker.get_particle_position(tag)
        x.value, y.value, z.value = px, py, pz
        return 0

    def set_particle_velocity(self, tag, vx, vy, vz, n):
        return self._worker.set_particle_velocity(tag, vx, vy, vz)

    def get_particle_velocity_array(self, tag, vx, vy, vz, n):
        ux, uy, uz = self._worker.get_particle_velocity(tag)
        vx.value, vy.value, vz.value = ux, uy, uz
        return 0

    # -- gravity bridge -----------------------------------------------------
    def get_gravity_at_point(self, eps, x, y, z, ax, ay, az, n):
        gx, gy, gz = self._worker.get_gravity_at_point(eps, x, y, z)
        ax.value, ay.value, az.value = gx, gy, gz
        return 0

    def get_potential_at_point(self, eps, x, y, z, phi, n):
        phi.value = self._worker.get_potential_at_point(eps, x, y, z)
        return 0

    def apply_gas_gravity_on_particles(self, dt):
        return self._worker.apply_gas_gravity_on_particles(dt)

    def apply_particle_gravity_on_gas(self, dt):
        return self._worker.apply_particle_gravity_on_gas(dt)

    # Torch's names for the same two half-kicks.
    def get_gravity_gas_on_particles(self, dt, kick_number):
        return self._worker.get_gravity_gas_on_particles(dt, kick_number)

    def get_gravity_particles_on_gas(self, dt, kick_number):
        return self._worker.get_gravity_particles_on_gas(dt, kick_number)

    # -- run bookkeeping Torch's loop reads ---------------------------------
    def get_current_step(self, step):
        step.value = self._worker.get_current_step()
        return 0

    def get_end_time(self, time):
        time.value = self._worker.get_end_time()
        return 0

    def get_max_num_steps(self, max_num_steps):
        max_num_steps.value = self._worker.get_max_num_steps()
        return 0

    def get_output_dir_wrapped(self, output_dir):
        output_dir.value = self._worker.get_output_dir()
        return 0

    def get_runtime_parameter(self, name, value):
        value.value = self._worker.get_runtime_parameter(name)
        return 0

    def particles_sort(self):
        return self._worker.particles_sort()

    def timer_summary(self):
        return self._worker.timer_summary()

    def IO_num(self, kind, num):
        num.value = self._worker.IO_num(kind)
        return 0

    def IO_out(self, kind, num):
        num.value = self._worker.IO_out(kind)
        return 0

    def write_chpt(self):
        return self._worker.write_chpt()

    # -- which particles are new since Torch last looked --------------------
    def get_number_of_new_tags(self, number_of_new_tags):
        number_of_new_tags.value = self._worker.get_number_of_new_tags()
        return 0

    def get_new_tags(self, index, tag, n):
        tag.value = self._worker.get_new_tags(index)
        return 0

    def get_particle_tags(self, index, tag, n):
        tag.value = self._worker.get_particle_tags(index)
        return 0

    def clear_new_tags(self):
        return self._worker.clear_new_tags()

    def set_starting_local_tag_numbers(self):
        return self._worker.set_starting_local_tag_numbers()

    # -- bulk create / remove (torch_sf spawns stars in one call) -----------
    def add_particles(self, tag, x, y, z, n):
        tag.value = self._worker.add_particles(x, y, z)
        return 0

    def remove_particles(self, tag, n):
        return self._worker.remove_particles(tag)

    def get_sink_mean_cs(self, tag, mean_cs, n):
        mean_cs.value = self._worker.get_sink_mean_cs(tag)
        return 0

    def form_stars(self, sink_rad, min_imf_mass, max_imf_mass, sample_imf_mass,
                   sample_imf_bins, sum_small, m_small, n_formed):
        n_formed.value = self._worker.form_stars(
            sink_rad, min_imf_mass, max_imf_mass, sample_imf_mass,
            int(sample_imf_bins), bool(sum_small), m_small)
        return 0

    def get_sink_state(self, tag, mass, x, y, z, vx, vy, vz, mean_cs, n):
        (mass.value, x.value, y.value, z.value, vx.value, vy.value, vz.value,
         mean_cs.value) = self._worker.get_sink_state(tag)
        return 0

    # -- photoionisation: the stars Torch owns, driving LUNA's M1 -----------
    def set_ionizing_sources(self, x, y, z, rate, active, n):
        self._worker.set_ionizing_sources(
            np.stack([np.atleast_1d(x), np.atleast_1d(y),
                      np.atleast_1d(z)], axis=1),
            np.atleast_1d(rate), np.atleast_1d(active))
        return 0

    def get_ionizing_photon_rate(self, rate):
        rate.value = self._worker.get_ionizing_photon_rate()
        return 0

    def energy_injection(self, energy, frac_kin, mass, x, y, z, dt):
        self._worker.energy_injection(energy, frac_kin, mass, x, y, z)
        # Return the CFL step, as FLASH does: Torch caps the next bridge interval with it.
        dt.value = float(self._worker.get_timestep())
        return 0


class LunaInterface(PythonCodeInterface, HydrodynamicsInterface, LiteratureReferencesMixIn):
    """LUNA hydro as an AMUSE community code (Python worker).

    .. [#] Steinwandel et al., LUNA (JAX hydro); coupled into Torch via AMUSE.
    """

    def __init__(self, **options):
        PythonCodeInterface.__init__(self, LunaImplementation, **options)
        LiteratureReferencesMixIn.__init__(self)

    # -- parameter passing --------------------------------------------------
    @legacy_function
    def set_config_value():
        f = LegacyFunctionSpecification()
        f.addParameter("key", dtype="string", direction=f.IN)
        f.addParameter("value", dtype="float64", direction=f.IN)
        f.result_type = "int32"
        return f

    @legacy_function
    def set_config_string():
        """Set a string Config field."""
        f = LegacyFunctionSpecification()
        f.addParameter("key", dtype="string", direction=f.IN)
        f.addParameter("value", dtype="string", direction=f.IN)
        f.result_type = "int32"
        return f

    # -- lifecycle ----------------------------------------------------------
    @legacy_function
    def initialize_code():
        f = LegacyFunctionSpecification()
        f.result_type = "int32"
        return f

    @legacy_function
    def commit_parameters():
        f = LegacyFunctionSpecification()
        f.result_type = "int32"
        return f

    @legacy_function
    def cleanup_code():
        f = LegacyFunctionSpecification()
        f.result_type = "int32"
        return f

    # -- geometry -----------------------------------------------------------
    @legacy_function
    def get_grid_range():
        f = LegacyFunctionSpecification()
        for name in ("imin", "imax", "jmin", "jmax", "kmin", "kmax", "index_of_grid"):
            f.addParameter(name, dtype="int32", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_position_of_index():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        for name in ("i", "j", "k", "index_of_grid"):
            f.addParameter(name, dtype="int32", direction=f.IN)
        for name in ("x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_index_of_position():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        for name in ("x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        for name in ("i", "j", "k"):
            f.addParameter(name, dtype="int32", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    # -- grid state ---------------------------------------------------------
    @legacy_function
    def set_grid_state():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        for name in ("i", "j", "k", "index_of_grid"):
            f.addParameter(name, dtype="int32", direction=f.IN)
        for name in ("rho", "rhovx", "rhovy", "rhovz", "rhoen"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_grid_state():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        for name in ("i", "j", "k", "index_of_grid"):
            f.addParameter(name, dtype="int32", direction=f.IN)
        for name in ("rho", "rhovx", "rhovy", "rhovz", "rhoen"):
            f.addParameter(name, dtype="float64", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    # -- time / evolution ---------------------------------------------------
    @legacy_function
    def evolve_model():
        f = LegacyFunctionSpecification()
        f.addParameter("time", dtype="float64", direction=f.IN)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_time():
        f = LegacyFunctionSpecification()
        f.addParameter("time", dtype="float64", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_timestep():
        f = LegacyFunctionSpecification()
        f.addParameter("timestep", dtype="float64", direction=f.OUT)
        f.result_type = "int32"
        return f

    # -- particles ----------------------------------------------------------
    @legacy_function
    def set_particle_pointers():
        f = LegacyFunctionSpecification()
        f.addParameter("part_type", dtype="string", direction=f.IN, default="mass")
        f.result_type = "int32"
        return f

    @legacy_function
    def get_number_of_particles():
        f = LegacyFunctionSpecification()
        f.addParameter("number_of_particles", dtype="int32", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def new_particle():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.OUT)
        for name in ("x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def delete_particle():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def set_particle_mass():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        f.addParameter("mass", dtype="float64", direction=f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_particle_mass():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        f.addParameter("mass", dtype="float64", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def set_particle_position():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        for name in ("x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_particle_position_array():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        for name in ("x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def set_particle_velocity():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        for name in ("vx", "vy", "vz"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_particle_velocity_array():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        for name in ("vx", "vy", "vz"):
            f.addParameter(name, dtype="float64", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    # -- gravity bridge -----------------------------------------------------
    @legacy_function
    def get_gravity_at_point():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        for name in ("eps", "x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        for name in ("ax", "ay", "az"):
            f.addParameter(name, dtype="float64", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_potential_at_point():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        for name in ("eps", "x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        f.addParameter("phi", dtype="float64", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def apply_gas_gravity_on_particles():
        f = LegacyFunctionSpecification()
        f.addParameter("dt", dtype="float64", direction=f.IN)
        f.result_type = "int32"
        return f

    @legacy_function
    def apply_particle_gravity_on_gas():
        f = LegacyFunctionSpecification()
        f.addParameter("dt", dtype="float64", direction=f.IN)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_gravity_gas_on_particles():
        f = LegacyFunctionSpecification()
        f.addParameter("dt", dtype="float64", direction=f.IN)
        f.addParameter("kick_number", dtype="int32", direction=f.IN, default=0)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_gravity_particles_on_gas():
        f = LegacyFunctionSpecification()
        f.addParameter("dt", dtype="float64", direction=f.IN)
        f.addParameter("kick_number", dtype="int32", direction=f.IN, default=0)
        f.result_type = "int32"
        return f

    # -- run bookkeeping ----------------------------------------------------
    @legacy_function
    def get_current_step():
        f = LegacyFunctionSpecification()
        f.addParameter("step", dtype="int32", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_end_time():
        f = LegacyFunctionSpecification()
        f.addParameter("time", dtype="float64", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_max_num_steps():
        f = LegacyFunctionSpecification()
        f.addParameter("max_num_steps", dtype="int32", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_output_dir_wrapped():
        f = LegacyFunctionSpecification()
        f.addParameter("output_dir", dtype="string", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_runtime_parameter():
        f = LegacyFunctionSpecification()
        f.addParameter("name", dtype="string", direction=f.IN)
        f.addParameter("value", dtype="float64", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def particles_sort():
        f = LegacyFunctionSpecification()
        f.result_type = "int32"
        return f

    @legacy_function
    def timer_summary():
        f = LegacyFunctionSpecification()
        f.result_type = "int32"
        return f

    @legacy_function
    def IO_num():
        f = LegacyFunctionSpecification()
        f.addParameter("kind", dtype="string", direction=f.IN)
        f.addParameter("num", dtype="int32", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def IO_out():
        f = LegacyFunctionSpecification()
        f.addParameter("kind", dtype="string", direction=f.IN)
        f.addParameter("num", dtype="int32", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def write_chpt():
        f = LegacyFunctionSpecification()
        f.result_type = "int32"
        return f

    # -- new-tag tracking ---------------------------------------------------
    @legacy_function
    def get_number_of_new_tags():
        f = LegacyFunctionSpecification()
        f.addParameter("number_of_new_tags", dtype="int32", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_new_tags():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("index", dtype="int32", direction=f.IN)
        f.addParameter("tag", dtype="int32", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_particle_tags():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("index", dtype="int32", direction=f.IN)
        f.addParameter("tag", dtype="int32", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def clear_new_tags():
        f = LegacyFunctionSpecification()
        f.result_type = "int32"
        return f

    @legacy_function
    def set_starting_local_tag_numbers():
        f = LegacyFunctionSpecification()
        f.result_type = "int32"
        return f

    # -- bulk create / remove ------------------------------------------------
    @legacy_function
    def add_particles():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.OUT)
        for name in ("x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def remove_particles():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_sink_mean_cs():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        f.addParameter("mean_cs", dtype="float64", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def form_stars():
        """Torch's sink -> star rule, run LUNA-side; new stars appear in get_new_tags."""
        f = LegacyFunctionSpecification()
        f.addParameter("sink_rad", dtype="float64", direction=f.IN)
        for name in ("min_imf_mass", "max_imf_mass", "sample_imf_mass"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        f.addParameter("sample_imf_bins", dtype="int32", direction=f.IN)
        f.addParameter("sum_small", dtype="int32", direction=f.IN)
        f.addParameter("m_small", dtype="float64", direction=f.IN)
        f.addParameter("n_formed", dtype="int32", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_sink_state():
        """Mass, position, velocity and accreted-gas sound speed of sinks, in one call."""
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        for name in ("mass", "x", "y", "z", "vx", "vy", "vz", "mean_cs"):
            f.addParameter(name, dtype="float64", direction=f.OUT)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def set_ionizing_sources():
        """Ionizing stars (position, photon rate) for LUNA's M1; `active` gates empty slots."""
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        for name in ("x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        f.addParameter("rate", dtype="float64", direction=f.IN)
        f.addParameter("active", dtype="float64", direction=f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    @legacy_function
    def get_ionizing_photon_rate():
        """Total photon rate currently in the radiation carry."""
        f = LegacyFunctionSpecification()
        f.addParameter("rate", dtype="float64", direction=f.OUT)
        f.result_type = "int32"
        return f

    @legacy_function
    def energy_injection():
        f = LegacyFunctionSpecification()
        f.addParameter("energy", dtype="float64", direction=f.IN)
        f.addParameter("frac_kin", dtype="float64", direction=f.IN)
        f.addParameter("mass", dtype="float64", direction=f.IN)
        for name in ("x", "y", "z"):
            f.addParameter(name, dtype="float64", direction=f.IN)
        f.addParameter("dt", dtype="float64", direction=f.OUT)
        f.result_type = "int32"
        return f


class Luna(InCodeComponentImplementation):
    def __init__(self, unit_converter=None, **options):
        # Set before the base init (define_converter reads it); the interface
        # constructor does not accept this keyword.
        self.unit_converter = unit_converter
        InCodeComponentImplementation.__init__(self, LunaInterface(**options), **options)

    def define_converter(self, handler):
        if self.unit_converter is not None:
            handler.set_converter(self.unit_converter.as_converter_from_si_to_generic())

    # -- shapes Torch expects: (n, 3), or a flat (3,) for one tag, as FLASH.
    def _three_vector(self, columns):
        x, y, z = columns
        out = np.stack([np.atleast_1d(c.value_in(c.unit)) for c in (x, y, z)],
                       axis=1) | x.unit
        return out.flatten() if not hasattr(x.number, "__len__") else out

    def get_particle_position(self, tags):
        return self._three_vector(self.get_particle_position_array(tags))

    def get_particle_velocity(self, tags):
        return self._three_vector(self.get_particle_velocity_array(tags))

    def get_output_dir(self):
        return self.get_output_dir_wrapped().strip()

    def get_index_range_inclusive(self, index_of_grid=1):
        """The grid's inclusive index bounds, as AMUSE's grid API asks for them."""
        # get_grid_range also returns index_of_grid as a seventh output.
        imin, imax, jmin, jmax, kmin, kmax, _ = self.get_grid_range()
        return (imin, imax, jmin, jmax, kmin, kmax)

    def define_methods(self, handler):
        """Declare each call's units (FLASH's, which Torch expects)."""
        u = _units()
        length, time, mass = gus.length, gus.time, gus.mass
        speed, density = gus.speed, gus.density
        momentum, energy = gus.momentum_density, gus.energy_density
        acc, potential = gus.acceleration, gus.potential
        index, nounit, err = handler.INDEX, handler.NO_UNIT, handler.ERROR_CODE

        # -- time and the run's control values -------------------------------
        handler.add_method("evolve_model", (time,), (err,))
        handler.add_method("get_time", (), (time, err,))
        handler.add_method("get_timestep", (), (time, err,))
        handler.add_method("get_end_time", (), (time, err,))
        handler.add_method("get_current_step", (), (nounit, err,))
        handler.add_method("get_max_num_steps", (), (nounit, err,))

        # -- the grid ---------------------------------------------------------
        handler.add_method("get_position_of_index",
                           (index, index, index, index),
                           (length, length, length, err,))
        handler.add_method("get_index_of_position", (length, length, length),
                           (index, index, index, err,))
        handler.add_method("get_grid_state", (index, index, index, index),
                           (density, momentum, momentum, momentum, energy, err,))
        handler.add_method("set_grid_state",
                           (index, index, index, index,
                            density, momentum, momentum, momentum, energy),
                           (err,))
        handler.add_method("get_grid_density", (index, index, index, index),
                           (density, err,))
        handler.add_method("get_grid_momentum_density",
                           (index, index, index, index),
                           (momentum, momentum, momentum, err,))
        handler.add_method("get_grid_energy_density",
                           (index, index, index, index), (energy, err,))

        # -- gravity ----------------------------------------------------------
        handler.add_method("get_gravity_at_point",
                           (length, length, length, length),
                           (acc, acc, acc, err,))
        handler.add_method("get_potential_at_point",
                           (length, length, length, length), (potential, err,))
        handler.add_method("get_gravity_gas_on_particles", (time, nounit), (err,))
        handler.add_method("get_gravity_particles_on_gas", (time, nounit), (err,))

        # -- particles ---------------------------------------------------------
        handler.add_method("set_particle_pointers", (nounit,), (err,))
        handler.add_method("get_number_of_particles", (), (nounit, err,))
        handler.add_method("add_particles", (length, length, length),
                           (nounit, err,))
        handler.add_method("remove_particles", (nounit,), (err,))
        handler.add_method("new_particle", (length, length, length),
                           (nounit, err,))
        handler.add_method("delete_particle", (nounit,), (err,))
        handler.add_method("get_particle_position_array", (index,),
                           (length, length, length, err,))
        handler.add_method("set_particle_position",
                           (index, length, length, length), (err,))
        handler.add_method("get_particle_velocity_array", (index,),
                           (speed, speed, speed, err,))
        handler.add_method("set_particle_velocity",
                           (index, speed, speed, speed), (err,))
        handler.add_method("get_particle_mass", (index,), (mass, err,))
        handler.add_method("set_particle_mass", (index, mass), (err,))
        handler.add_method("get_particle_tags", (index,), (nounit, err,))
        handler.add_method("get_new_tags", (index,), (nounit, err,))
        handler.add_method("get_number_of_new_tags", (), (nounit, err,))
        handler.add_method("get_sink_mean_cs", (index,), (speed, err,))
        handler.add_method("form_stars", (length, mass, mass, mass, nounit, nounit, mass),
                           (nounit, err,))
        handler.add_method("get_sink_state", (index,),
                           (mass, length, length, length, speed, speed, speed, speed, err,))
        handler.add_method("set_ionizing_sources",
                           (length, length, length, u["rate"], nounit), (err,))
        handler.add_method("get_ionizing_photon_rate", (), (u["rate"], err,))
        handler.add_method("energy_injection",
                           (gus.energy, nounit, mass, length, length, length),
                           (time, err,))

        # everything torch_se parks between steps
        for attr, unit in PARKED_ATTRS.items():
            q = nounit if unit is None else u[unit]
            handler.add_method("get_particle_" + attr, (index,), (q, err,))
            handler.add_method("set_particle_" + attr, (index, q), (err,))


# -- the parked attributes' get_/set_ pairs, generated --------------------------
def _parked_impl(attr, reading):
    """In-worker get_/set_ forwarder for `attr`, built with exec so its parameter is named `attr`."""
    verb = "get" if reading else "set"
    name = "{}_particle_{}".format(verb, attr)
    body = ("value.value = getattr(self._worker, name)(tag)\n        return 0"
            if reading else
            "return getattr(self._worker, name)(tag, value)")
    src = ("def {name}(self, tag, {attr}, n):\n"
           "        value = {attr}\n"
           "        {body}\n").format(name=name, attr=attr, body=body)
    ns = {"name": name}
    exec(src, ns)
    return ns[name]


def _parked_legacy(attr, reading):
    """The marshalling side: tag in, one float per tag in or out."""
    verb = "get" if reading else "set"
    name = "{}_particle_{}".format(verb, attr)

    def spec():
        f = LegacyFunctionSpecification()
        f.must_handle_array = True
        f.addParameter("tag", dtype="int32", direction=f.IN)
        f.addParameter(attr, dtype="float64",
                       direction=f.OUT if reading else f.IN)
        f.addParameter("n", dtype="int32", direction=f.LENGTH)
        f.result_type = "int32"
        return f

    spec.__name__ = name
    return legacy_function(spec)


for _attr in PARKED_ATTRS:
    for _reading in (True, False):
        _fn = _parked_impl(_attr, _reading)
        if not hasattr(LunaImplementation, _fn.__name__):
            setattr(LunaImplementation, _fn.__name__, _fn)
            setattr(LunaInterface, _fn.__name__, _parked_legacy(_attr, _reading))


# AMUSE leaves Nagle on for the worker's socket; its small multi-part replies
# then wait on delayed ACKs, so set TCP_NODELAY (docs/TORCH.md).
_NODELAY_DONE = False


def _nodelay_connections():
    global _NODELAY_DONE
    if _NODELAY_DONE:
        return
    import socket
    from amuse.rfi import python_code
    plain = python_code.socket.create_connection

    def create_connection(*args, **kwargs):
        s = plain(*args, **kwargs)
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return s

    python_code.socket.create_connection = create_connection
    _NODELAY_DONE = True


try:
    _nodelay_connections()
except Exception:                       # pragma: no cover - the worker still runs, slowly
    pass


# Time every call; each evolve_model prints the costliest since the last one.
CALL_TIMES = {}


def _timed(name, fn):
    import functools
    import time

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            n_t = CALL_TIMES.setdefault(name, [0, 0.0])
            n_t[0] += 1
            n_t[1] += time.perf_counter() - t0
            if name == "evolve_model":
                top = sorted(CALL_TIMES.items(), key=lambda kv: -kv[1][1])[:8]
                print("LUNA calls: " + "  ".join(f"{k} {n}x {t:.3f}s" for k, (n, t) in top),
                      flush=True)
                CALL_TIMES.clear()
    return wrapper


for _name, _fn in list(vars(LunaImplementation).items()):
    if callable(_fn) and not _name.startswith("_"):
        setattr(LunaImplementation, _name, _timed(_name, _fn))
