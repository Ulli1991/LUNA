"""FP64 Helmholtz EOS in CGS, following Microphysics 99b9b47; see docs/HELMHOLTZ.md."""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

KB = 1.380650424e-16
NA = 6.0221417930e23
AMU = 1.66053878283e-24
H = 6.6260689633e-27
ARAD = 4 * 5.67051e-5 / 2.99792458e10


class HelmholtzTable(NamedTuple):
    density: jax.Array  # rho Ye, not total density
    temperature: jax.Array
    free: jax.Array  # (T, rho Ye, density derivative, T derivative)
    dpdr: jax.Array


class Thermodynamics(NamedTuple):
    pressure: jax.Array
    energy: jax.Array  # specific internal energy, erg/g
    entropy: jax.Array  # erg/g/K
    dpdr: jax.Array  # fixed T, composition
    dpdt: jax.Array  # fixed rho, composition
    dedr: jax.Array
    cv: jax.Array
    gamma1: jax.Array
    sound_speed: jax.Array  # nonrelativistic sqrt(Gamma1 P/rho)
    valid: jax.Array
    coulomb_disabled: jax.Array  # correction would make P or e nonpositive


def _require_fp64():
    if not jax.config.jax_enable_x64:
        raise ValueError("Helmholtz EOS requires jax_enable_x64=True")


def load_table(path):
    """Read the standard 541 x 201 Timmes ASCII table once, outside jit."""
    _require_fp64()
    with Path(path).open() as stream:
        values = np.fromstring(stream.read(), sep=" ")
    n = 541 * 201
    if values.size != n * 21 or not np.isfinite(values).all():
        raise ValueError("expected a finite 541 x 201 Helmholtz table with 21 values per node")
    free = values[: 9 * n].reshape(201, 541, 9)
    dpdr = values[9 * n : 13 * n].reshape(201, 541, 4)
    return HelmholtzTable(
        jnp.asarray(np.logspace(-12, 15, 541)),
        jnp.asarray(np.logspace(3, 13, 201)),
        jnp.asarray(free[..., [[0, 2, 4], [1, 5, 7], [3, 6, 8]]]),
        jnp.asarray(dpdr[..., [[0, 2], [1, 3]]]),
    )


# Low-end Hermite basis coefficients in ascending powers; high end is reflected.
_QUINTIC = ((1, 0, 0, -10, 15, -6), (0, 1, 0, -6, 8, -3), (0, 0, 0.5, -1.5, 1.5, -0.5))
_CUBIC = ((1, 0, -3, 2), (0, 1, -2, 1))


def _basis(x, width, degree, derivative=0):
    coeffs = _QUINTIC if degree == 2 else _CUBIC
    weights = []
    for end in (0, 1):
        z = x if end == 0 else 1 - x
        for order, row in enumerate(coeffs):
            row = np.polynomial.polynomial.polyder(row, derivative)
            value = jnp.zeros_like(z)
            for coefficient in reversed(row):
                value = value * z + coefficient
            sign = (-1) ** (order + derivative) if end else 1
            weights.append(sign * width ** (order - derivative) * value)
    return jnp.stack(weights, axis=-1)


def _cell(nodes, x):
    log0 = jnp.log10(nodes[0])
    inv_step = (nodes.size - 1) / (jnp.log10(nodes[-1]) - log0)
    i = jnp.clip(jnp.floor((jnp.log10(x) - log0) * inv_step), 0, nodes.size - 2).astype(jnp.int32)
    width = nodes[i + 1] - nodes[i]
    return i, (x - nodes[i]) / width, width


def _interpolate(data, i, j, wx, wy):
    n = data.shape[-1]
    result = jnp.zeros_like(wx[..., 0])
    for di in (0, 1):
        for dj in (0, 1):
            block = data[j + dj, i + di]
            result = result + jnp.einsum(
                "...ab,...a,...b->...",
                block,
                wx[..., di * n : (di + 1) * n],
                wy[..., dj * n : (dj + 1) * n],
            )
    return result


@jax.custom_jvp
def _stable_pressure(value, density, temperature, dpdd, dpdt):
    """Use the tabulated compressibility instead of cancelling free-energy derivatives."""
    return value


@_stable_pressure.defjvp
def _stable_pressure_jvp(primals, tangents):
    value, density, temperature, dpdd, dpdt = primals
    return value, dpdd * tangents[1] + dpdt * tangents[2]


def _electrons(table, rho, temperature, electron_fraction):
    electron_mass_density = rho * electron_fraction
    i, x, dx = _cell(table.density, electron_mass_density)
    j, y, dy = _cell(table.temperature, temperature)
    wx = [_basis(x, dx, 2, k) for k in range(2)]
    wy = [_basis(y, dy, 2, k) for k in range(3)]

    def free(a, b):
        return _interpolate(table.free, i, j, wx[a], wy[b])

    free_energy, free_density = free(0, 0), free(1, 0)
    free_temperature, free_mixed, free_temperature2 = free(0, 1), free(1, 1), free(0, 2)
    dpdd = jnp.maximum(_interpolate(table.dpdr, i, j, _basis(x, dx, 1), _basis(y, dy, 1)), 0)
    density_squared = electron_mass_density**2
    dpdt = density_squared * free_mixed
    pressure = _stable_pressure(
        density_squared * free_density, electron_mass_density, temperature, dpdd, dpdt
    )
    energy = electron_fraction * (free_energy - temperature * free_temperature)
    entropy = -electron_fraction * free_temperature
    dedr = electron_fraction**2 * (free_density - temperature * free_mixed)
    cv = -electron_fraction * temperature * free_temperature2
    return pressure, energy, entropy, electron_fraction * dpdd, dpdt, dedr, cv


def _ions_radiation(rho, temperature, mean_mass):
    gas = KB * NA / mean_mass
    ion_pressure = rho * gas * temperature
    radiation_pressure = ARAD / 3 * temperature**4
    radiation_energy = 3 * radiation_pressure / rho
    entropy_ion = gas * (
        2.5
        + jnp.log(mean_mass**2.5 / (rho * NA) * (2 * jnp.pi * AMU * KB * temperature / H**2) ** 1.5)
    )
    return (
        ion_pressure + radiation_pressure,
        1.5 * gas * temperature + radiation_energy,
        entropy_ion + 4 * radiation_pressure / (rho * temperature),
        gas * temperature,
        rho * gas + 4 * radiation_pressure / temperature,
        -radiation_energy / rho,
        1.5 * gas + 4 * radiation_energy / temperature,
    )


def _coulomb(rho, temperature, mean_mass, mean_charge):
    gas = KB * NA / mean_mass
    coupling = (
        mean_charge
        * mean_charge
        * (4.8032042712e-10) ** 2
        / (KB * temperature)
        * (4 * jnp.pi / 3 * NA * rho / mean_mass) ** (1 / 3)
    )
    quarter_power = coupling**0.25
    strong = coupling >= 1
    energy_fit = -0.898004 * coupling + 0.96786 * quarter_power + 0.220703 / quarter_power - 0.86097
    energy_slope = -0.898004 + (0.96786 * quarter_power - 0.220703 / quarter_power) / (4 * coupling)
    weak_fit = 0.288675 * coupling**1.5 - 0.29561 / 3 * coupling**1.9885
    weak_slope = 1.5 * 0.288675 * coupling**0.5 - 0.29561 / 3 * 1.9885 * coupling**0.9885
    e = gas * temperature * jnp.where(strong, energy_fit, -3 * weak_fit)
    de_dg = gas * temperature * jnp.where(strong, energy_slope, -3 * weak_slope)
    dedr = de_dg * coupling / (3 * rho)
    cv = e / temperature - de_dg * coupling / temperature
    entropy_strong = -gas * (
        3 * 0.96786 * quarter_power
        - 5 * 0.220703 / quarter_power
        - 0.86097 * (jnp.log(coupling) - 1)
        - 2.5269
    )
    entropy_weak = -gas * (
        0.288675 * coupling**1.5 - 0.29561 * (1.9885 - 1) / 1.9885 * coupling**1.9885
    )
    return (
        rho * e / 3,
        e,
        jnp.where(strong, entropy_strong, entropy_weak),
        e / 3 + rho * dedr / 3,
        rho * cv / 3,
        dedr,
        cv,
    )


def evaluate(table, rho, temperature, mean_mass, mean_charge, *, coulomb=True):
    """Broadcast CGS inputs; invalid cells return valid=False and NaN thermodynamics, including under jit."""
    _require_fp64()
    rho, temperature, mean_mass, mean_charge = jnp.broadcast_arrays(
        *[jnp.asarray(a, jnp.float64) for a in (rho, temperature, mean_mass, mean_charge)]
    )
    electron_fraction = mean_charge / mean_mass
    valid = (
        jnp.isfinite(rho)
        & jnp.isfinite(temperature)
        & jnp.isfinite(mean_mass)
        & jnp.isfinite(mean_charge)
        & (rho > 0)
        & (temperature >= table.temperature[0])
        & (temperature <= table.temperature[-1])
        & (mean_mass >= 1)
        & (mean_charge > 0)
        & (mean_charge <= mean_mass)
        & (rho * electron_fraction >= table.density[0])
        & (rho * electron_fraction <= table.density[-1])
    )
    # Safe operands keep invalid cells from contaminating their valid neighbours.
    rho, temperature, mean_mass, mean_charge = [
        jnp.where(valid, a, b)
        for a, b in zip((rho, temperature, mean_mass, mean_charge), (1.0, 1e7, 4.0, 2.0))
    ]
    electrons = _electrons(table, rho, temperature, mean_charge / mean_mass)
    ions_rad = _ions_radiation(rho, temperature, mean_mass)
    terms = [a + b for a, b in zip(electrons, ions_rad)]
    disabled = jnp.zeros_like(valid)
    if coulomb:
        correction = _coulomb(rho, temperature, mean_mass, mean_charge)
        disabled = (terms[0] + correction[0] <= 0) | (terms[1] + correction[1] <= 0)
        terms = [a + jnp.where(disabled, 0.0, b) for a, b in zip(terms, correction)]
    p, e, s, dpdr, dpdt, dedr, cv = terms
    cs2 = dpdr + temperature * dpdt * dpdt / (rho * rho * cv)
    valid = valid & (p > 0) & (cv > 0) & (cs2 > 0)
    outputs = (*terms, rho * cs2 / p, jnp.sqrt(jnp.maximum(cs2, 0.0)))
    valid = valid & jnp.all(jnp.stack([jnp.isfinite(a) for a in outputs]), axis=0)
    return Thermodynamics(*[jnp.where(valid, a, jnp.nan) for a in outputs], valid, valid & disabled)


def evaluate_checked(table, rho, temperature, mean_mass, mean_charge, *, coulomb=True):
    """Host boundary: raise on any invalid cell; use evaluate inside jit and inspect valid."""
    out = evaluate(table, rho, temperature, mean_mass, mean_charge, coulomb=coulomb)
    if not np.asarray(out.valid).all():
        raise ValueError(
            "invalid Helmholtz state: check table bounds, composition and thermodynamic stability"
        )
    return out
