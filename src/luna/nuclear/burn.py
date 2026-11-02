"""Standalone constant-density self-heating alpha-chain RHS for the shared Rodas3 integrator."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from luna.ode.rodas3 import Network
from luna.eos.helmholtz import evaluate, evaluate_checked
from luna.eos.inversion import temperature_from_energy
from luna.nuclear.aprox13 import composition_valid, fixed_temperature_rhs, mean_composition
from luna.nuclear.neutrinos import sneut5


class BurnParams(NamedTuple):
    density: object
    temperature_guess: object


def initial_state(table, density, temperature, mass_fractions, *, coulomb=True):
    """Host boundary: construct the 13 mass fractions plus specific thermal energy."""
    fractions = jnp.asarray(mass_fractions, jnp.float64)
    mean_mass, mean_charge = mean_composition(fractions)
    thermo = evaluate_checked(table, density, temperature, mean_mass, mean_charge, coulomb=coulomb)
    shape = jnp.broadcast_shapes(fractions.shape[:-1], thermo.energy.shape)
    return jnp.concatenate(
        [
            jnp.broadcast_to(fractions, shape + (13,)),
            jnp.broadcast_to(thermo.energy, shape)[..., None],
        ],
        axis=-1,
    )


def make_network(table, *, screening, neutrinos, rtol=1e-6, coulomb=True):
    """Build a self-heating network with explicit screening and neutrino choices."""
    if screening not in ("none", "screen5") or neutrinos not in ("none", "sneut5"):
        raise NotImplementedError(
            'screening must be "none" or "screen5"; neutrinos must be "none" or "sneut5"'
        )
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be positive and finite")

    def rhs(state, params):
        fractions, energy = state[:13], state[13]
        mean_mass, mean_charge = mean_composition(fractions, negative_tolerance=1e-14)
        recovered = temperature_from_energy(
            table,
            params.density,
            energy,
            mean_mass,
            mean_charge,
            temperature_guess=params.temperature_guess,
            coulomb=coulomb,
        )
        # A converged root gets the implicit first derivative; do not differentiate its iteration history.
        root = jax.lax.stop_gradient(recovered.temperature)
        thermo = evaluate(table, params.density, root, mean_mass, mean_charge, coulomb=coulomb)
        temperature = root + (energy - thermo.energy) / jax.lax.stop_gradient(thermo.cv)
        derivative, heating = fixed_temperature_rhs(
            params.density, temperature, fractions, negative_tolerance=1e-14, screening=screening
        )
        cooling = (
            sneut5(params.density, temperature, mean_mass, mean_charge).total
            if neutrinos == "sneut5"
            else jnp.zeros_like(heating)
        )
        result = jnp.concatenate([derivative, (heating - cooling)[None]])
        valid = (
            recovered.converged
            & thermo.valid
            & composition_valid(fractions, negative_tolerance=1e-14)
            & jnp.all(jnp.isfinite(result))
        )
        return result, {
            "temperature_guess": temperature,
            "invalid": ~valid,
            "neutrino_loss": cooling,
        }

    return Network(
        rhs=rhs,
        rtol=rtol,
        atol=(1e-14,) * 13 + (0.0,),
        safety=(0.1,) * 14,
        ymax=lambda p: jnp.asarray([1.0] * 13 + [jnp.inf]),
        ctrl_floor=lambda p: jnp.asarray([1e-14] * 13 + [0.0]),
        hidden=("temperature_guess",),
    )
