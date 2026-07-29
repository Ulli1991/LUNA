"""Mesh MUSIC gas particles onto the (optionally slab-sharded) grid as
super-comoving conserved fields."""

import jax
import jax.numpy as jnp
import numpy as np


def mesh_gas_to_grid(gas, pm, a):
    """One-time CIC mesh of MUSIC gas SPH particles -> super-comoving conserved
    grid state (rho~, rho~v~, E~). Gadget velocity u -> v~ = a^{3/2} u, and the
    specific internal energy -> e~_int = a^2 u_int (since P~ = a^5 P)."""
    pos = jnp.asarray(gas.pos)
    mass = jnp.asarray(gas.mass)
    ones = jnp.ones(pos.shape[0])
    vtil = jnp.asarray(gas.vel) * (a ** 1.5)             # super-comoving velocity
    etil = (a ** 2) * jnp.asarray(gas.u_int)             # super-comoving specific u
    rho = pm.cic_deposit(pos, mass, ones)
    mx = pm.cic_deposit(pos, mass * vtil[:, 0], ones)
    my = pm.cic_deposit(pos, mass * vtil[:, 1], ones)
    mz = pm.cic_deposit(pos, mass * vtil[:, 2], ones)
    E = pm.cic_deposit(pos, mass * (0.5 * jnp.sum(vtil ** 2, axis=1) + etil), ones)
    return jnp.stack([rho, mx, my, mz, E])


def mesh_gas_to_grid_slab(gas, mesh, Nx, Ny, Nz, dx, box, a, n_total_gas=None):
    """Sharded one-time CIC mesh of MUSIC gas particles -> super-comoving conserved
    grid (rho~, rho~v~, E~), each field a P('x',None,None) slab. The 5 conserved
    fields are deposited as separate CIC-weighted scatters via cic_deposit_slab.

    Single-process: host-side decompose_to_slabs + device_put (whole gas set).
    Multi-process (n_total_gas given): this rank holds only its index-chunk; we
    assemble the per-slab buffer from the local chunk, settle boundary crossers
    with one drift exchange (the 5 weights ride along as the exchange payload),
    then deposit -- never building the whole-N array on any rank."""
    import numpy as _np

    from gravity.particle_mesh_shardmap import (assemble_slab_from_chunk,
                                                cic_deposit_slab, decompose_to_slabs,
                                                exchange_particles_slab)
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    ndev = int(mesh.shape["x"])
    vtil = _np.asarray(gas.vel) * (a ** 1.5)                  # super-comoving velocity
    etil = (a ** 2) * _np.asarray(gas.u_int)                  # super-comoving specific u
    m = _np.asarray(gas.mass)
    # five conserved weights deposited as "mass": rho~, mom_{x,y,z}, E~
    w = _np.stack([m, m * vtil[:, 0], m * vtil[:, 1], m * vtil[:, 2],
                   m * (0.5 * _np.sum(vtil ** 2, axis=1) + etil)], axis=1)   # (n,5)

    if n_total_gas is not None:                              # multi-process path
        (posS, wS, massS), actS, _C = assemble_slab_from_chunk(
            [_np.asarray(gas.pos), w, m], [P("x", None), P("x", None), P("x")],
            n_total_gas, ndev, mesh, capacity_factor=1.3)
        posS, wS, _massS, actS, _nb, _oc = exchange_particles_slab(
            posS, wS, massS, actS, Nx, dx, box, mesh)        # settle to owning slab
        fields = [cic_deposit_slab(posS, wS[:, k], actS, Nx, Ny, Nz, dx, mesh)
                  for k in range(5)]
        return jnp.stack(fields)

    pos_g, w_g, _, act_g, _C = decompose_to_slabs(_np.asarray(gas.pos), w, m, Nx, ndev, dx, box)
    posS = jax.device_put(jnp.asarray(pos_g), NamedSharding(mesh, P("x", None)))
    actS = jax.device_put(jnp.asarray(act_g), NamedSharding(mesh, P("x")))
    fields = []
    for k in range(5):
        wkS = jax.device_put(jnp.asarray(w_g[:, k]), NamedSharding(mesh, P("x")))
        fields.append(cic_deposit_slab(posS, wkS, actS, Nx, Ny, Nz, dx, mesh))
    return jnp.stack(fields)                                  # (5, Nx, Ny, Nz) P(None,'x',..)
