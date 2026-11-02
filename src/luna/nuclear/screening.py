"""Screen5 weak/intermediate/strong enhancement, with the reference's quantum and exponent limits."""

import jax.numpy as jnp

from luna.eos.helmholtz import _require_fp64
from luna.nuclear.aprox13 import MASS_NUMBERS, CHARGES, composition_valid


def screen5_log(
    density, temperature, mass_fractions, charge1, mass1, charge2, mass2, *, negative_tolerance=0.0
):
    """Log enhancement for one entrance pair; broadcast pair axes explicitly against composition."""
    _require_fp64()
    fractions = jnp.asarray(mass_fractions, jnp.float64)
    valid = composition_valid(fractions, negative_tolerance=negative_tolerance)
    density, temperature, charge1, mass1, charge2, mass2 = jnp.broadcast_arrays(
        *[
            jnp.asarray(x, jnp.float64)
            for x in (density, temperature, charge1, mass1, charge2, mass2)
        ]
    )
    valid = (
        valid
        & jnp.isfinite(density)
        & (density > 0)
        & jnp.isfinite(temperature)
        & (temperature > 0)
        & jnp.isfinite(charge1)
        & jnp.isfinite(charge2)
        & jnp.isfinite(mass1)
        & jnp.isfinite(mass2)
        & (charge1 > 0)
        & (charge2 > 0)
        & (mass1 >= charge1)
        & (mass2 >= charge2)
    )
    density, temperature, charge1, mass1, charge2, mass2 = [
        jnp.where(valid, x, safe)
        for x, safe in zip(
            (density, temperature, charge1, mass1, charge2, mass2), (1.0, 1e9, 2.0, 4.0, 2.0, 4.0)
        )
    ]
    fractions = jnp.where(valid[..., None], fractions, jnp.eye(13)[0])
    molar = fractions / jnp.asarray(MASS_NUMBERS)
    electron_sum = jnp.sum(molar * jnp.asarray(CHARGES), axis=-1)
    charge2_sum = jnp.sum(molar * jnp.asarray(CHARGES) ** 2, axis=-1)
    plasma_coupling = 2.27493e5 / temperature * jnp.cbrt(density * electron_sum)
    weak = (
        charge1
        * charge2
        * 1.88e8
        / temperature
        * jnp.sqrt(density / temperature * (electron_sum + charge2_sum))
    )
    charge_sum = charge1 + charge2
    pair_scale = 2 ** (1 / 3) * charge1 * charge2 / jnp.cbrt(charge_sum)
    pair_coupling = pair_scale * plasma_coupling
    barrier = (
        (4.248719e3 / 3)
        * temperature ** (-1 / 3)
        * jnp.cbrt(charge1**2 * charge2**2 * mass1 * mass2 / (mass1 + mass2))
    )
    quantum_parameter = jnp.minimum(pair_coupling / barrier, 1.6)
    pair_coupling = jnp.minimum(pair_coupling, 1.6 * barrier)
    plasma_coupling = pair_coupling / pair_scale
    charge_difference = charge_sum ** (5 / 3) - charge1 ** (5 / 3) - charge2 ** (5 / 3)
    quarter_difference = charge_sum ** (5 / 12) - charge1 ** (5 / 12) - charge2 ** (5 / 12)
    charge_log = (5 / 3) * jnp.log(charge1 * charge2 / charge_sum)
    classical = (
        0.896434 * plasma_coupling * charge_difference
        - 3.44740 * plasma_coupling**0.25 * quarter_difference
        - 0.5551 * (jnp.log(plasma_coupling) + charge_log)
        - 2.996
    )
    quantum3 = quantum_parameter**3
    barrier_correction = barrier * (
        5 / 32 - quantum_parameter * (0.014 + 0.0128 * quantum_parameter)
    )
    coupling_correction = (
        pair_coupling
        * quantum_parameter
        * (0.0055 + quantum_parameter * (-0.0098 + 0.0048 * quantum_parameter))
    )
    strong = classical - quantum3 * (barrier_correction + coupling_correction)
    strong += jnp.log(jnp.maximum(1 - 0.0562 * quantum3, 0.77))
    weight = jnp.clip((pair_coupling - 0.3) / 0.5, 0.0, 1.0)
    blended = jnp.where(pair_coupling > 0.3, (1 - weight) * weak + weight * strong, weak)
    return jnp.where(valid, jnp.maximum(blended, 0.0), jnp.nan)


def screen5(
    density, temperature, mass_fractions, charge1, mass1, charge2, mass2, *, negative_tolerance=0.0
):
    """Multiplicative enhancement capped at exp(300), as in the reference wrapper."""
    exponent = screen5_log(
        density,
        temperature,
        mass_fractions,
        charge1,
        mass1,
        charge2,
        mass2,
        negative_tolerance=negative_tolerance,
    )
    return jnp.exp(jnp.minimum(exponent, 300.0))


def channel_enhancements(density, temperature, mass_fractions, *, negative_tolerance=0.0):
    """One batched screen5 evaluation for every distinct entrance pair used by aprox13."""
    fractions = jnp.asarray(mass_fractions)
    density, temperature = jnp.broadcast_arrays(jnp.asarray(density), jnp.asarray(temperature))
    parent_charges = CHARGES[1:-1]
    parent_masses = MASS_NUMBERS[1:-1]
    odd_charges = tuple(z + 1 for z in CHARGES[4:-1])
    odd_masses = tuple(a + 3 for a in MASS_NUMBERS[4:-1])
    charge1 = (2, 2) + parent_charges + odd_charges + (6, 6, 8)
    mass1 = (4, 4) + parent_masses + odd_masses + (12, 12, 16)
    charge2 = (2, 4) + (2,) * 11 + (1,) * 8 + (6, 8, 8)
    mass2 = (4, 8) + (4,) * 11 + (1,) * 8 + (12, 16, 16)
    factors = screen5(
        density[..., None],
        temperature[..., None],
        fractions[..., None, :],
        jnp.asarray(charge1),
        jnp.asarray(mass1),
        jnp.asarray(charge2),
        jnp.asarray(mass2),
        negative_tolerance=negative_tolerance,
    )
    return (
        factors[..., 0] * factors[..., 1],
        factors[..., 2:13],
        factors[..., 13:21],
        factors[..., 21:24],
    )
