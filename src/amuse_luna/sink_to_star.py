"""Sink-to-star handoff: turn accumulated LUNA sink mass into AMUSE star particles."""
import numpy as np
from amuse.datamodel import Particles
from amuse.units import units as u


def spawn_stars_from_sinks(worker, m_star, grav=None, se=None, store=True,
                           sigma_factor=0.0, rng=None):
    """Spawn one `m_star` star per sink with mass >= `m_star`, debit the sink, return the stars.

    Star velocity = sink velocity + `sigma_factor` x accreted-gas sound speed x N(0,1)."""
    sel, spos, svel, smass = worker.get_active_sinks()
    if len(sel) == 0:
        return Particles(0)
    mom = worker.get_sink_gas_moments()
    cs = np.asarray(mom["mean_cs"])
    rng = rng or np.random.default_rng(0)

    rows = []
    for a in range(len(sel)):
        if smass[a] < m_star:
            continue
        pos = spos[a]
        vel = svel[a].copy()
        if sigma_factor > 0.0 and a < len(cs):
            vel = vel + sigma_factor * cs[a] * rng.standard_normal(3)
        rows.append((int(sel[a]), pos, vel))
        worker.set_sink_mass(int(sel[a]), float(smass[a] - m_star))

    if not rows:
        return Particles(0)

    new = Particles(len(rows))
    new.mass = np.full(len(rows), m_star) | u.MSun
    new.position = np.array([r[1] for r in rows]) | u.parsec
    new.velocity = np.array([r[2] for r in rows]) | u.kms

    if grav is not None:
        grav.particles.add_particles(new)
    if se is not None:
        se.particles.add_particles(new)
    if store:
        worker.set_particle_pointers("mass")
        p = new.position.value_in(u.parsec)
        v = new.velocity.value_in(u.kms)
        tags = worker.new_particle(p[:, 0], p[:, 1], p[:, 2])
        worker.set_particle_mass(tags, new.mass.value_in(u.MSun))
        worker.set_particle_velocity(tags, v[:, 0], v[:, 1], v[:, 2])
    return new
