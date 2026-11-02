"""Mesh MUSIC gas particles onto the grid as super-comoving conserved fields."""

import jax
import jax.numpy as jnp


def mesh_gas_to_grid(gas, pm, a):
    """CIC-mesh MUSIC gas particles to the super-comoving state (rho~, rho~v~, E~)."""
    pos = jnp.asarray(gas.pos)
    mass = jnp.asarray(gas.mass)
    ones = jnp.ones(pos.shape[0])
    vtil = jnp.asarray(gas.vel) * (a ** 1.5)
    etil = (a ** 2) * jnp.asarray(gas.u_int)
    rho = pm.cic_deposit(pos, mass, ones)
    mx = pm.cic_deposit(pos, mass * vtil[:, 0], ones)
    my = pm.cic_deposit(pos, mass * vtil[:, 1], ones)
    mz = pm.cic_deposit(pos, mass * vtil[:, 2], ones)
    E = pm.cic_deposit(pos, mass * (0.5 * jnp.sum(vtil ** 2, axis=1) + etil), ones)
    return jnp.stack([rho, mx, my, mz, E])


def mesh_gas_to_grid_blocks(gas, mesh, N, dx, box, a, n_total_gas=None):

    import numpy as _np
    from jax.sharding import PartitionSpec as P

    from luna.gravity.particle_mesh_blocks import (AXES, assemble_blocks_local,
                                              assemble_blocks_sharded, cic_deposit_blocks,
                                              block_of_process,
                                              exact_block_capacity,
                                              exact_settle_face_capacity,
                                              exchange_particles_blocks)
    from luna.utils.shmap import shard_map

    vtil = _np.asarray(gas.vel) * (a ** 1.5)
    etil = (a ** 2) * _np.asarray(gas.u_int)
    m = _np.asarray(gas.mass)
    w = _np.stack([m, m * vtil[:, 0], m * vtil[:, 1], m * vtil[:, 2],
                   m * (0.5 * _np.sum(vtil ** 2, axis=1) + etil)], axis=1)   # (n,5)
    pdims = tuple(int(mesh.shape[ax]) for ax in AXES)
    Pn, Pk = P(AXES), P(AXES, None)
    if n_total_gas is not None:
        # gas is meshed once and never migrates: size the buffers exactly
        (posS, wS, mS), actS, _C = assemble_blocks_local(
            [_np.asarray(gas.pos), w, m], [Pk, Pk, Pn], n_total_gas, mesh,
            capacity=exact_block_capacity(_np.asarray(gas.pos), N, pdims, dx, box))
        posS, wS, mS, actS, _, n_over = exchange_particles_blocks(
            posS, wS, mS, actS, N, dx, box, mesh,
            face_capacity=exact_settle_face_capacity(
                _np.asarray(gas.pos), N, pdims, dx, box, block_of_process(mesh)))
        if int(n_over) != 0:
            raise RuntimeError(
                f"gas IC settle: {int(n_over)} particles overflowed the block "
                "buffers, which the counted capacities should make impossible")
    else:
        posS, wS, mS, actS, _, _C = assemble_blocks_sharded(
            _np.asarray(gas.pos), w, m, N, pdims, dx, box, mesh)
    fields = [cic_deposit_blocks(posS, wS[:, k], actS, N, dx, mesh) for k in range(5)]
    # grid spec P('x','y','z'), not the particle-buffer spec P(('x','y','z'))
    Pg = P(*AXES)
    stack = jax.jit(shard_map(lambda *f: jnp.stack(f), mesh=mesh,
                              in_specs=(Pg,) * 5, out_specs=P(None, *AXES)))
    return stack(*fields)
