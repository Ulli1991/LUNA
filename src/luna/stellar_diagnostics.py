"""Shell-integrated stellar budgets and radial profiles from FP64 conservative states."""

import jax
import numpy as np

from luna.gravity.spherical_1d import shell_gravity
from luna.nuclear.aprox13 import BINDING_MEV, MASS_NUMBERS, NA, MEV_TO_ERG, SPECIES

_gravity = jax.jit(shell_gravity)


def measure(state, temperature, dx):
    """Return global quantities and radial profiles, including the ambient material."""
    state, temperature = np.asarray(state), np.asarray(temperature)
    if state.ndim != 2 or state.shape[0] != 16 or temperature.shape != state.shape[1:]:
        raise ValueError("stellar diagnostic shape mismatch")
    if not np.isfinite(state).all() or not np.isfinite(temperature).all():
        raise ValueError("nonfinite stellar diagnostic state")
    if not np.isfinite(dx) or dx <= 0 or np.any(state[0] <= 0) or np.any(temperature <= 0):
        raise ValueError("positive density, temperature and spacing required")
    fractions = state[2:-1] / state[0]
    if fractions.min() < -1e-12 or np.max(abs(fractions.sum(axis=0) - 1)) > 3e-12:
        raise ValueError("invalid diagnostic composition")
    faces = dx * np.arange(state.shape[1] + 1)
    volumes = 4 * np.pi / 3 * np.diff(faces**3)
    masses = state[0] * volumes
    enclosed = np.cumsum(masses)
    velocity = state[1] / state[0]
    kinetic = 0.5 * masses * velocity**2
    gas_energy = float(np.dot(state[-1], volumes))
    gravity = _gravity(state[0], dx)
    binding = float(gravity.binding_energy)
    fastest = int(np.argmax(abs(velocity)))
    quantities = dict(
        mass_g=float(enclosed[-1]),
        isotope_masses_g=dict(zip(SPECIES, map(float, state[2:-1] @ volumes))),
        gas_energy_erg=gas_energy,
        kinetic_energy_erg=float(kinetic.sum()),
        thermal_energy_erg=gas_energy - float(kinetic.sum()),
        gravitational_energy_erg=binding,
        gas_plus_gravity_energy_erg=gas_energy + binding,
        maximum_temperature_K=float(temperature.max()),
        maximum_speed_cm_s=float(abs(velocity[fastest])),
        density_at_maximum_speed_g_cm3=float(state[0, fastest]),
        rms_speed_cm_s=float(np.sqrt(np.dot(masses, velocity**2) / enclosed[-1])),
        mass_radii_cm={
            str(percent): float(np.cbrt(np.interp(
                percent / 100 * enclosed[-1], np.r_[0, enclosed], faces**3
            ))) for percent in (50, 90, 99)
        },
    )
    profiles = dict(
        radius_cm=0.5 * (faces[1:] + faces[:-1]),
        density_g_cm3=state[0],
        velocity_cm_s=velocity,
        temperature_K=temperature,
        mass_fractions=fractions,
        enclosed_mass_g=enclosed,
        potential_erg_g=np.asarray(gravity.potential),
        thermal_energy_erg_g=state[-1] / state[0] - 0.5 * velocity**2,
    )
    return quantities, profiles


def budgets(initial, current, boundary_budget):
    """Account for nuclear binding changes and integrated inward boundary fluxes in CGS."""
    boundary = 4 * np.pi * np.asarray(boundary_budget)
    if boundary.shape != (16,) or not np.isfinite(boundary).all():
        raise ValueError("finite spherical boundary budget with 16 fields required")
    weights = np.asarray(BINDING_MEV) / np.asarray(MASS_NUMBERS) * NA * MEV_TO_ERG
    old = np.array([initial["isotope_masses_g"][name] for name in SPECIES])
    new = np.array([current["isotope_masses_g"][name] for name in SPECIES])
    change = new - old - boundary[2:-1]
    release = float(np.dot(weights, change))
    residual = (
        current["gas_plus_gravity_energy_erg"] - initial["gas_plus_gravity_energy_erg"]
        - boundary[-1] - release
    )
    scale = abs(initial["gas_energy_erg"]) + abs(initial["gravitational_energy_erg"])
    if scale <= 0:
        raise ValueError("positive initial energy scale required")
    return dict(
        nuclear_release_erg=release,
        net_co_consumed_g=float(-change[1:3].sum()),
        mass_budget_error=float(abs(current["mass_g"] - initial["mass_g"] - boundary[0])
                                / initial["mass_g"]),
        energy_budget_residual_erg=float(residual),
        energy_budget_error=float(abs(residual) / scale),
        net_boundary_mass_input_g=float(boundary[0]),
        net_boundary_energy_input_erg=float(boundary[-1]),
    )
