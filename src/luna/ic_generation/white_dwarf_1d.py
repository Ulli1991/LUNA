"""CPU construction of an isothermal Newtonian white dwarf with an explicit ambient cutoff."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from luna.eos.helmholtz import evaluate
from luna.gravity.spherical_1d import G
from luna.nuclear.aprox13 import mean_composition
from luna.nuclear.burn import initial_state


class WhiteDwarf(NamedTuple):
    state: object
    temperature: object
    atmosphere: object
    surface_radius: float


def isothermal_model(
    table, central_density, temperature, fractions, cells, dx, *, ambient_density=1e-4
):
    """Integrate dP/dr=-rho*G*M/r² and volume-average three-point shell quadrature."""
    if not (
        np.isfinite([central_density, temperature, dx, ambient_density]).all()
        and central_density > ambient_density > 0
        and temperature > 0
        and dx > 0
        and cells >= 4
    ):
        raise ValueError("positive finite model inputs and central density above ambient required")
    fractions = jnp.asarray(fractions, dtype=jnp.float64)
    mass, charge = mean_composition(fractions)
    eos = jax.jit(lambda table, rho: evaluate(table, rho, temperature, mass, charge))
    length_scale, mass_scale = 1e8, 1e33

    def rhs(radius, values):
        rho = np.exp(values[0])
        thermo = eos(table, rho)
        if not bool(thermo.valid):
            raise ValueError(f"white dwarf model left EOS domain at rho={rho}")
        r = radius * length_scale
        gradient = -G * values[1] * mass_scale / (r**2 * float(thermo.dpdr))
        return [gradient * length_scale, 4 * np.pi * r**2 * rho * length_scale / mass_scale]

    def surface(radius, values):
        return values[0] - np.log(ambient_density)

    surface.terminal, surface.direction = True, -1
    start = min(dx * 1e-3, 1e3)
    central = eos(table, central_density)
    if not bool(central.valid):
        raise ValueError("central white dwarf state is invalid")
    log_rho = np.log(central_density) - 2 * np.pi * G * central_density * start**2 / (
        3 * float(central.dpdr)
    )
    solution = solve_ivp(
        rhs,
        (start / length_scale, cells * dx / length_scale),
        [log_rho, 4 * np.pi * central_density * start**3 / (3 * mass_scale)],
        rtol=2e-9,
        atol=[1e-10, 1e-12],
        events=surface,
        dense_output=True,
        max_step=dx / length_scale,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    if not solution.t_events[0].size:
        raise ValueError("domain does not contain the white dwarf surface; increase cells*dx")
    surface_radius = float(solution.t_events[0][0] * length_scale)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    radii = dx * (np.arange(cells)[:, None] + 0.5 * (nodes + 1))
    rho = np.exp(
        solution.sol(np.maximum(np.minimum(radii, surface_radius), start).ravel() / length_scale)[0]
    ).reshape(cells, 3)
    rho = np.where(radii < surface_radius, rho, ambient_density)
    thermal = initial_state(table, rho, temperature, fractions)[..., -1]
    volume_weights = weights[None] * radii**2
    volume_weights /= volume_weights.sum(axis=1, keepdims=True)
    density = np.sum(rho * volume_weights, axis=1)
    energy = np.sum(rho * np.asarray(thermal) * volume_weights, axis=1)
    state = (
        jnp.zeros((16, cells))
        .at[0]
        .set(density)
        .at[2:-1]
        .set(density * fractions[:, None])
        .at[-1]
        .set(energy)
    )
    # Recover temperature from cell-averaged energy; a warm guess is sufficient.
    from luna.hydro.stellar_1d import make_solver

    recovered = make_solver(table).recover(state, jnp.full(cells, temperature))
    if not np.asarray(recovered.valid).all():
        raise ValueError("volume-averaged white dwarf state is outside the EOS domain")
    return WhiteDwarf(
        state, recovered.temperature, jnp.asarray(radii[:, 0] >= surface_radius), surface_radius
    )
