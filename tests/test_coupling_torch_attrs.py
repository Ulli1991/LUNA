"""What Torch parks on LUNA's particles between steps.

With `with_se` on, `torch_se.evolve_stars` writes the whole stellar bookkeeping
back to the hydro code every step -- emission properties and SeBa state alike --
and `torch_sf.add_particles_to_grav` reads it out again when a newly formed
star is handed to the N-body and stellar-evolution codes. None of it means
anything to LUNA; all of it has to survive the round trip.
"""

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from luna.setup.state import Particles  # noqa: E402
from luna.species.sinks import Sinks  # noqa: E402
from amuse_luna.particles import PARKED_ATTRS  # noqa: E402
from amuse_luna.worker import LunaWorker  # noqa: E402


def _worker(N=8, boundary=0):
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=boundary,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
    )
    w = LunaWorker(cfg)
    w.initialize_code()
    w.commit_parameters()
    return w


def test_every_parked_attribute_survives_the_round_trip():
    w = _worker()
    tags = w.new_particle([0.2, 0.7], [0.3, 0.6], [0.4, 0.5])

    for i, attr in enumerate(sorted(PARKED_ATTRS)):
        want = np.array([1.0 + i, 2.0 + i])
        getattr(w, "set_particle_" + attr)(tags, want)
        got = getattr(w, "get_particle_" + attr)(tags)
        np.testing.assert_allclose(
            got, want, err_msg="{} did not survive".format(attr))


def test_a_new_star_reports_its_own_mass_as_the_zams_mass():
    """SeBa has never written rel_mass for a star LUNA formed this step. Handing
    it the stored zero would give the SE code a zero-mass star."""
    w = _worker()
    tag = w.new_particle(0.5, 0.5, 0.5)
    w.set_particle_mass(tag, [12.0])

    np.testing.assert_allclose(w.get_particle_rel_mass(tag), [12.0])

    # Once SeBa does write one, that is what comes back.
    w.set_particle_rel_mass(tag, [11.5])
    np.testing.assert_allclose(w.get_particle_rel_mass(tag), [11.5])


def test_particles_sort_wraps_in_a_periodic_box():
    """FLASH drops particles that leave the domain. In a periodic box that
    would silently lose a star, so they wrap instead."""
    w = _worker(boundary=0)
    tag = w.new_particle(1.25, -0.25, 0.5)     # Lx = Ly = Lz = 1

    w.particles_sort()

    assert w.get_number_of_particles() == 1
    x, y, z = w.get_particle_position(tag)
    np.testing.assert_allclose([x[0], y[0], z[0]], [0.25, 0.75, 0.5])


def test_particles_sort_drops_escapees_when_the_box_is_not_periodic():
    w = _worker(boundary=1)
    inside = w.new_particle(0.5, 0.5, 0.5)
    w.new_particle(1.5, 0.5, 0.5)

    w.particles_sort()

    assert w.get_number_of_particles() == 1
    np.testing.assert_allclose(w.get_particle_position(inside)[0], [0.5])


def test_the_run_control_values_torch_reads_are_usable():
    """Torch loops `while step < max_num_steps`, so LUNA's 0-means-no-limit
    cannot cross the boundary as a zero."""
    w = _worker()
    assert w.get_max_num_steps() > 0
    assert w.get_current_step() == 0
    assert w.get_end_time() == w.cfg.t_end
    assert isinstance(w.get_output_dir(), str)


def _inject_sink(w, pos, mass):
    """One sink in LUNA's own carry, where the solver would have put it.

    Under `Sinks.name`, not a key spelled here. Spelling it is how this was
    wrong in the first place: the worker looked up "sink" while the species is
    called "sinks", and a test that spelled it the same wrong way agreed with
    the bug instead of catching it.
    """
    n = int(getattr(w.cfg, "sink_max_count", 4))
    p = np.zeros((n, 3)); p[0] = pos
    m = np.zeros(n); m[0] = mass
    act = np.zeros(n); act[0] = 1.0
    w.state.species[Sinks.name] = Particles(
        pos=jnp.asarray(p), p=jnp.zeros((n, 3)),
        mass=jnp.asarray(m), active=jnp.asarray(act), extra=None)


def test_the_pointer_switch_carries_sinks_both_ways():
    """LUNA forms sinks inside the step; Torch reads and drains them through
    the particle interface. torch_sf brackets that work in a switch to the
    'sink' pointer and back, so that is where the two have to meet."""
    w = _worker()
    _inject_sink(w, [0.5, 0.5, 0.5], mass=40.0)

    w.set_particle_pointers("sink")
    tags = w.get_particle_tags()
    assert len(tags) == 1
    np.testing.assert_allclose(w.get_particle_mass(tags), [40.0])

    # torch_sf decrements a sink as it spawns that sink's stars
    w.set_particle_mass(tags, [30.0])
    w.set_particle_pointers("mass")

    _, _, _, smass = w.get_active_sinks()
    np.testing.assert_allclose(smass, [30.0], rtol=1e-9)


def test_a_self_gravitating_worker_can_take_a_step():
    """`initial_state` leaves the opening gravity solve out on purpose, so a
    restore can happen first; `setup.evolve.evolve` does it immediately after.
    An AMUSE worker never goes through `evolve`, so if the stepper does not do
    it too, `state.g_gas` is None and the first KDK opening kick dies on the
    first step -- which is exactly where a coupled run starts."""
    cfg = _build_cfg_for_test(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, GRAVITATIONAL_G=1.0,
        SELF_GRAVITY=True, POISSON_BOUNDARY=0,
    )
    w = LunaWorker(cfg)
    w.initialize_code(build_ic=True)
    w.commit_parameters()

    t0 = w.get_time()
    w.evolve_model(t0 + w.get_timestep())
    assert w.get_time() > t0
    assert np.all(np.isfinite(np.asarray(w.state.U)))


def test_the_worker_turns_x64_on_when_the_config_asks():
    """A standalone run gets x64 from `setup.cli.launch`. An AMUSE worker is
    started by the orchestrator and never goes near it, so `enable_x64 = True`
    produced a float32 run and said nothing about it."""
    cfg = _build_cfg_for_test(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=8, Ny=8, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
        ORDER=1, LIMITER=0, enable_x64=True,
    )
    was = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        LunaWorker(cfg)                       # constructing is enough
        assert jax.config.jax_enable_x64
    finally:
        jax.config.update("jax_enable_x64", was)


def test_a_sink_luna_forms_itself_reaches_torch():
    """The one that matters, and the one a hand-injected sink cannot give you.

    Everything above puts a sink in the carry by hand, so it can only check the
    plumbing downstream of the key it used. This builds a real cloud, lets LUNA
    form its own sinks, and asks for them the way Torch does -- switch the
    pointer, count, list the tags. It is the only test here that would have
    noticed the worker reading `species["sink"]` when the species is "sinks".
    """
    from luna.ic_generation.turbulent_sphere_3d import Params
    from luna.units import UnitSystem

    pc, msun = 3.085677581e18, 1.98892e33
    units = UnitSystem(UnitLength_in_cm=pc, UnitMass_in_g=msun,
                       UnitVelocity_in_cm_per_s=1.0e5)
    cfg = _build_cfg_for_test(
        test_problem="Turbulent-Sphere-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=32, Ny=32, Nz=32, Lx=8.0, Ly=8.0, Lz=8.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, enable_x64=True, units=units,
        GRAVITATIONAL_G=units.GravityConstant,
        problem=Params(mass_msun=2000.0, radius_pc=2.0, T_K=15.0, mach=8.0),
        sink_star_mu=2.3, seed=1,
        SELF_GRAVITY=True, POISSON_BOUNDARY=0,
        barotropic_eos=True, barotropic_T0_K=15.0, barotropic_mu=2.3,
        barotropic_rho_AD_cgs=1.0e-14,
        sinks_enabled=True, sink_max_count=64, sink_sink_gravity=False,
        # just above the mean cloud density (59.7 Msun/pc^3), so the first
        # converging region makes a sink instead of waiting for a free-fall time
        sink_creation_density=70.0,
        cooling_enabled=False, radiation_enabled=False,
    )
    w = LunaWorker(cfg)
    w.initialize_code(build_ic=True)
    w.commit_parameters()

    for _ in range(30):
        w.evolve_model(w.get_time() + float(w.get_timestep()))
        if w.get_number_of_sinks() > 0:
            break
    assert w.get_number_of_sinks() > 0, "LUNA formed no sink to hand over"

    # now ask for them the way torch_sf does
    w.set_particle_pointers("sink")
    n = w.get_number_of_particles()
    assert n > 0, "LUNA has sinks but the Torch particle surface reports none"
    tags = w.get_particle_tags(range(1, n + 1))
    assert len(tags) == n
    assert float(np.sum(w.get_particle_mass(tags))) > 0.0

    # Torch seeds a new star's velocity dispersion from this. A sink that has
    # just been created has accreted nothing, so its accumulators are empty and
    # the exact moments come out as zeros -- which reads as "the gas was at
    # rest" and makes the sink's whole brood exactly co-moving. They then
    # free-fall into each other and the N-body code spends its life on the
    # encounters: one measured run spent 79 s of a 103 s step inside PeTar.
    cs = np.atleast_1d(np.asarray(w.get_sink_mean_cs(tags)))
    assert np.all(cs > 0.0), "a sink with no accretion history reports cs = 0"
    w.set_particle_pointers("mass")


def _radiating_cfg(**over):
    from luna.units import UnitSystem
    pc, msun = 3.085677581e18, 1.98892e33
    units = UnitSystem(UnitLength_in_cm=pc, UnitMass_in_g=msun,
                       UnitVelocity_in_cm_per_s=1.0e5)
    base = dict(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=16, Ny=16, Nz=16, Lx=4.0, Ly=4.0, Lz=4.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, enable_x64=True, units=units,
        radiation_enabled=True, rad_ionization=True, cooling_enabled=False,
        star_max_count=16,
        # the ionization prescription's own numbers; what they are does not
        # matter to this test, only that the carry exists to deposit into
        rad_chat=300.0,            # reduced c in km/s, >> any flow here
        rad_sigma=1.0e-4, rad_alpha_B=1.0e-4, rad_m_H=1.0,
        rad_cs2_neutral=0.05, rad_cs2_ionized=0.5, rad_x0=0.0,
    )
    base.update(over)
    return _build_cfg_for_test(**base)


def test_a_star_torch_gave_us_drives_the_radiation():
    """Torch writes each star's ionising photon rate with `set_particle_nion`
    every step and expects the hydro code to do something with it. Its own
    path is a ray trace it asks FLASH for; LUNA transports the photons itself,
    so the wire is the rate, deposited at the bridge cadence.

    What this pins: a star with a rate on it ends up in the radiation carry,
    conserving, and a star with no rate does not disturb it.
    """
    w = LunaWorker(_radiating_cfg())
    w.initialize_code(build_ic=True)
    w.commit_parameters()

    tag = w.new_particle(2.0, 2.0, 2.0)
    w.set_particle_mass(tag, [30.0])
    assert w.get_ionizing_photon_rate() == 0.0     # nothing emitting yet

    w.set_particle_nion(tag, [1.0e3])
    w.evolve_model(w.get_time() + float(w.get_timestep()))

    # deposit_sources conserves: sum(field) * dV == sum(active rates)
    np.testing.assert_allclose(w.get_ionizing_photon_rate(), 1.0e3, rtol=1e-9)


def test_two_stars_add_their_rates():
    w = LunaWorker(_radiating_cfg())
    w.initialize_code(build_ic=True)
    w.commit_parameters()

    tags = w.new_particle([1.0, 3.0], [2.0, 2.0], [2.0, 2.0])
    w.set_particle_mass(tags, [30.0, 40.0])
    w.set_particle_nion(tags, [1.0e3, 2.5e3])
    w.evolve_model(w.get_time() + float(w.get_timestep()))

    np.testing.assert_allclose(w.get_ionizing_photon_rate(), 3.5e3, rtol=1e-9)


def test_a_star_that_is_gone_stops_shining():
    """When the last emitting star goes (removed, or out of the box), the
    source field and Q must go to zero; they used to keep the old star's."""
    w = LunaWorker(_radiating_cfg())
    w.initialize_code(build_ic=True)
    w.commit_parameters()
    tag = w.new_particle(2.0, 2.0, 2.0)
    w.set_particle_mass(tag, [30.0])
    w.set_particle_nion(tag, [1.0e3])
    w.evolve_model(w.get_time() + float(w.get_timestep()))
    np.testing.assert_allclose(w.get_ionizing_photon_rate(), 1.0e3, rtol=1e-9)

    w.delete_particle(tag)
    w.evolve_model(w.get_time() + float(w.get_timestep()))
    assert w.get_ionizing_photon_rate() == 0.0
    assert "n_src=0 Q=0.000e+00" in w.feedback_report()


def test_the_feedback_line_counts_what_torch_handed_over():
    """Per bridge step the worker prints what feedback it has received and
    what the gas shows for it. The counts are Torch's calls, the energies
    their sums, and a star shining for a step leaves ionised gas behind."""
    import re
    w = LunaWorker(_radiating_cfg())
    w.initialize_code(build_ic=True)
    w.commit_parameters()
    line = w.feedback_report()
    assert "SN n=0" in line and "winds n=0" in line
    assert float(re.search(r"x_ion mass=(\S+)", line).group(1)) == 0.0

    w.energy_injection(2.0, 0.0, 0.0, 2.0, 2.0, 2.0)
    w.energy_injection(3.0, 0.3, 0.1, 1.0, 1.0, 1.0)
    w.wind_injection(0.5, 4.0, 3.0, 3.0, 3.0)
    tag = w.new_particle(2.0, 2.0, 2.0)
    w.set_particle_mass(tag, [30.0])
    w.set_particle_nion(tag, [1.0e3])
    w.evolve_model(w.get_time() + float(w.get_timestep()))

    line = w.feedback_report()
    u = w.cfg.units
    E_sn = float(re.search(r"SN n=2 E=(\S+)", line).group(1))
    E_w = float(re.search(r"winds n=1 dm=5\.000e-01 E=(\S+)", line).group(1))
    np.testing.assert_allclose(E_sn, u.to_cgs_energy(5.0), rtol=1e-3)
    np.testing.assert_allclose(E_w, u.to_cgs_energy(0.5 * 0.5 * 4.0 ** 2), rtol=1e-3)
    assert "n_src=1 Q=1.000e+03" in line
    assert float(re.search(r"x_ion mass=(\S+)", line).group(1)) > 0.0


def test_one_call_gives_every_sink_what_torch_asks_for_per_sink():
    """torch_sf reads mass, position, velocity and mean_cs one sink and one
    attribute at a time; get_sink_state answers all of them at once and must
    give the same numbers, from any pointer."""
    w = _worker()
    n = int(getattr(w.cfg, "sink_max_count", 4))
    p = np.zeros((n, 3)); p[0] = [0.5, 0.5, 0.5]; p[1] = [0.2, 0.7, 0.4]
    v = np.zeros((n, 3)); v[0] = [0.1, -0.2, 0.3]; v[1] = [-0.4, 0.0, 0.2]
    m = np.zeros(n); m[0], m[1] = 40.0, 12.5
    act = np.zeros(n); act[:2] = 1.0
    w.state.species[Sinks.name] = Particles(
        pos=jnp.asarray(p), p=jnp.asarray(v), mass=jnp.asarray(m),
        active=jnp.asarray(act), extra=None)

    w.set_particle_pointers("sink")
    tags = np.array(w.get_particle_tags()[::-1])       # out of order on purpose
    assert len(tags) == 2
    want = (w.get_particle_mass(tags), *w.get_particle_position(tags),
            *w.get_particle_velocity(tags), w.get_sink_mean_cs(tags))
    w.set_particle_pointers("mass")
    got = w.get_sink_state(tags)                        # from the 'mass' pointer
    assert w._pointer == "sink"
    for a, b in zip(got, want):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_allclose(sorted(np.asarray(got[0])), [12.5, 40.0])


def test_luna_forms_stars_from_its_sinks_the_way_torch_does():
    """form_stars: the sink's queue is IMF-sampled, the stars its mass covers
    are spawned within sink_rad with the sink's velocity, the sink hands the
    mass over exactly, and the stars arrive as new tags for Torch."""
    w = _worker()
    n = int(getattr(w.cfg, "sink_max_count", 4))
    p = np.zeros((n, 3)); p[0] = [0.5, 0.5, 0.5]
    v = np.zeros((n, 3)); v[0] = [0.1, -0.2, 0.3]
    m = np.zeros(n); m[0] = 500.0
    act = np.zeros(n); act[0] = 1.0
    w.state.species[Sinks.name] = Particles(
        pos=jnp.asarray(p), p=jnp.asarray(v), mass=jnp.asarray(m),
        active=jnp.asarray(act), extra=None)
    kw = dict(sink_rad=0.05, min_imf_mass=0.08, max_imf_mass=120.0,
              sample_imf_mass=1.0e4, sample_imf_bins=10, sum_small=False, m_small=1.0)

    n_new = w.form_stars(**kw)
    assert n_new > 100                                   # 500 Msun at a 0.6 Msun mean
    assert w.get_number_of_new_tags() == n_new
    tags = w.get_new_tags()
    w.set_particle_pointers("mass")
    stars = np.asarray(w.get_particle_mass(tags))
    pos = np.asarray(w.get_particle_position(tags))
    _, _, _, smass = w.get_active_sinks()
    np.testing.assert_allclose(float(smass[0]) + stars.sum(), 500.0, rtol=1e-12)
    assert stars.sum() <= 500.0
    assert np.all(np.linalg.norm(np.stack(pos, axis=1) - p[0], axis=1) <= 0.05 + 1e-12)
    assert 0.08 <= stars.min() and stars.max() <= 120.0

    w.clear_new_tags()
    assert w.form_stars(**kw) == 0                       # nothing accreted since
    assert w.get_number_of_new_tags() == 0
