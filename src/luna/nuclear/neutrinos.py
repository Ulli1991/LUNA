"""Itoh 1996 thermal neutrino losses, matching default sneut5 without optional recombination."""

from typing import NamedTuple

import jax.numpy as jnp

from luna.eos.helmholtz import _require_fp64
from luna.nuclear._neutrino_coefficients import PHOTO_COSINE, PHOTO_SINE

_VECTOR = 0.5 + 2 * 0.2319
_OTHER_VECTOR = 1 - _VECTOR
_CURRENT_SUM = _VECTOR**2 + 0.25 + 2 * (_OTHER_VECTOR**2 + 0.25)
_CURRENT_DIFFERENCE = _VECTOR**2 - 0.25 + 2 * (_OTHER_VECTOR**2 - 0.25)


class NeutrinoLosses(NamedTuple):
    pair: object
    plasma: object
    photo: object
    bremsstrahlung: object
    total: object
    valid: object


def _pair(temperature, theta, scaled_density, zeta):
    correction = 1 - 13.04 * theta**2 + 133.5 * theta**4 + 1534 * theta**6 + 918.6 * theta**8
    low = temperature < 1e10
    suppression = jnp.exp(-jnp.where(low, 5.5924, 4.9924) * zeta)
    denominator = jnp.where(
        low,
        0.9383 / theta - 0.4141 / theta**2 + 0.05829 / theta**3,
        1.2383 / theta - 0.8141 / theta**2,
    )
    fit = (6.002e19 + 2.084e20 * zeta + 1.872e21 * zeta**2) * suppression / (zeta**3 + denominator)
    weak_correction = (1 + scaled_density / (7.692e7 * theta**3 + 9.715e6 * jnp.sqrt(theta))) ** (
        -0.3
    )
    weak_correction /= 10.748 * theta**2 + 0.3967 * jnp.sqrt(theta) + 1.005
    return (
        jnp.exp(-2 / theta)
        * fit
        * correction
        * 0.5
        * (_CURRENT_SUM + _CURRENT_DIFFERENCE * weak_correction)
    )


def _plasma(temperature, theta, scaled_density):
    gamma2 = (
        1.1095e11
        * scaled_density
        / (temperature**2 * jnp.sqrt(1 + (1.019e-6 * scaled_density) ** (2 / 3)))
    )
    gamma = jnp.sqrt(gamma2)
    transverse = 2.4 + 0.6 * jnp.sqrt(gamma) + 0.51 * gamma + 1.25 * gamma**1.5
    longitudinal = (8.6 * gamma2 + 1.35 * gamma**3.5) / (225 - 17 * gamma + gamma2)
    density_log = jnp.log10(2 * scaled_density)
    temperature_log = jnp.log10(temperature)
    x = (17.5 + density_log - 3 * temperature_log) / 6
    y = (-24.5 + density_log + 3 * temperature_log) / 6
    active = (jnp.abs(x) <= 0.7) & (y >= 0)
    x = jnp.where(active, x, 0.0)
    y = jnp.where(active, y, 0.0)
    shape = 1.05 + (
        0.39 - 1.25 * x - 0.35 * jnp.sin(4.5 * x) - 0.3 * jnp.exp(-((4.5 * x + 0.9) ** 2))
    ) * jnp.exp(-((jnp.minimum(0.0, y - 1.6 + 1.25 * x) / (0.57 - 0.25 * x)) ** 2))
    shape = jnp.where(active, shape, 1.0)
    return (
        (transverse + longitudinal) * shape * jnp.exp(-gamma) * gamma**6 * 0.93153 * 3e21 * theta**9
    )


def _photo(temperature, theta, scaled_density, zeta):
    band = jnp.where(temperature < 1e8, 0, jnp.where(temperature < 1e9, 1, 2))
    phase = jnp.log10(temperature) - jnp.asarray([7.0, 8.0, 9.0])[band]
    angle = (5 * jnp.pi / 3) * phase
    harmonics = jnp.arange(1, 6)
    cosine = jnp.cos(angle[..., None] * harmonics)
    sine = jnp.sin(angle[..., None] * harmonics)
    cos_coeff = jnp.asarray(PHOTO_COSINE)[band]
    sin_coeff = jnp.asarray(PHOTO_SINE)[band]
    polynomial = (
        0.5 * cos_coeff[..., 0]
        + jnp.sum(cos_coeff[..., 1:6] * cosine[..., None, :], axis=-1)
        + jnp.sum(sin_coeff * sine[..., None, :], axis=-1)
        + 0.5 * cos_coeff[..., 6] * jnp.cos(10 * jnp.pi * phase)[..., None]
    )
    attenuation = jnp.where(band == 0, 0.5654 + phase, 1.5654)
    numerator = (
        polynomial[..., 0] + polynomial[..., 1] * zeta + polynomial[..., 2] * zeta**2
    ) * jnp.exp(-attenuation * zeta)
    fit = numerator / (zeta**3 + 0.006290 / theta + 0.007483 / theta**2 + 0.0003061 / theta**3)
    correction = (
        0.666
        * (1 + 2.045 * theta) ** (-2.066)
        / (
            1
            + scaled_density
            / (1.875e8 * theta + 1.653e8 * theta**2 + 8.499e8 * theta**3 - 1.604e8 * theta**4)
        )
    )
    return jnp.maximum(
        theta**5 * fit * scaled_density * 0.5 * (_CURRENT_SUM - _CURRENT_DIFFERENCE * correction),
        0.0,
    )


def _bremsstrahlung(density, temperature, mean_mass, mean_charge):
    electron_fraction = mean_charge / mean_mass
    scaled_density = density * electron_fraction
    t8 = temperature * 1e-8
    fermi_temperature = 5.9302e9 * (
        jnp.sqrt(1 + 1.018 * (density * 1e-6 * electron_fraction) ** (2 / 3)) - 1
    )
    weak = temperature > 0.3 * fermi_temperature
    degeneracy = scaled_density / (7.05e6 * t8**1.5 + 5.12e4 * t8**3)
    weak_f = 1 / (23.5 + 6.83e4 / t8**2 + 7.81e8 / t8**5) + 1.26 * (1 + 1 / degeneracy) / (
        1 + 1.47 / degeneracy + 0.0329 / degeneracy**2
    )
    weak_g = 1 / ((230 + 6.7e5 / t8**2 + 7.66e9 / t8**5) * (1 + scaled_density * 1e-9)) + 1 / (
        (7.75e5 * t8**1.5 + 247 * t8**3.85) / scaled_density
        + 4.07
        + 0.024 * t8**1.4
        + 4.59e-5 * t8 ** (-0.110) * density**0.656
    )
    angle = jnp.pi / 5 * (jnp.log10(density) - 3)
    cosines = jnp.cos(angle[..., None] * jnp.arange(1, 6))
    sines = jnp.sin(angle[..., None] * jnp.arange(1, 5))
    cos_coeff = jnp.asarray(
        [
            [-0.05821, -0.01089, -0.01147, -0.00656, -0.00519],
            [-0.00944, -0.01289, -0.00589, -0.00404, -0.00330],
            [-0.00710, -0.00028, 0.00232, 0.00044, 0.00158],
            [0.00356, -0.00184, 0.00146, 0.00031, 0.00069],
        ]
    )
    sin_coeff = jnp.asarray(
        [
            [-0.04969, -0.01584, -0.00504, -0.00281],
            [-0.02213, -0.01136, -0.00467, -0.00131],
            [0.023, -0.01078, 0.00118, -0.00089],
            [0.01052, -0.00354, -0.00014, -0.00018],
        ]
    )
    fits = (
        jnp.asarray([0.43502, 0.282095, 0.083, 0.048265])
        + angle[..., None] * jnp.asarray([0.00945, -0.02342, -0.01259, -0.00829])
        + jnp.sum(cos_coeff * cosines[..., None, :], axis=-1)
        + jnp.sum(sin_coeff * sines[..., None, :], axis=-1)
    )
    inverse_coupling = 1 / (0.2275 * mean_charge**2 / t8 * (density * 1e-6 / mean_mass) ** (1 / 3))
    root = jnp.cbrt(inverse_coupling)
    weight_f = -0.05483 - 0.01946 * root + 1.86310 * root**2 - 0.78873 * inverse_coupling
    weight_g = -0.06711 + 0.06859 * root + 1.74360 * root**2 - 0.74498 * inverse_coupling
    dense_f = weight_f * fits[..., 0] + (1 - weight_f) * fits[..., 1]
    dense_g = weight_g * fits[..., 2] + (1 - weight_g) * fits[..., 3]
    f = jnp.where(weak, weak_f, dense_f)
    g = jnp.where(weak, weak_g, dense_g)
    return (
        0.5738
        * mean_charge
        * electron_fraction
        * t8**6
        * density
        * 0.5
        * (_CURRENT_SUM * f - _CURRENT_DIFFERENCE * g)
    )


def sneut5(density, temperature, mean_mass, mean_charge, *, include_recombination=False):
    """Specific losses (erg/g/s); T below 1e7 K gives zero, invalid states give NaNs."""
    _require_fp64()
    if include_recombination:
        raise NotImplementedError("optional sneut5 recombination is not implemented")
    density, temperature, mean_mass, mean_charge = jnp.broadcast_arrays(
        *[jnp.asarray(x, jnp.float64) for x in (density, temperature, mean_mass, mean_charge)]
    )
    valid = (
        jnp.isfinite(density)
        & jnp.isfinite(temperature)
        & jnp.isfinite(mean_mass)
        & jnp.isfinite(mean_charge)
        & (density > 0)
        & (temperature > 0)
        & (mean_mass >= 1)
        & (mean_charge > 0)
        & (mean_charge <= mean_mass)
    )
    active = valid & (temperature >= 1e7)
    density, work_temperature, mean_mass, mean_charge = [
        jnp.where(active, x, safe)
        for x, safe in zip((density, temperature, mean_mass, mean_charge), (1e6, 1e8, 4.0, 2.0))
    ]
    theta = work_temperature / 5.9302e9
    scaled_density = density * mean_charge / mean_mass
    zeta = jnp.cbrt(scaled_density * 1e-9) / theta
    components = (
        _pair(work_temperature, theta, scaled_density, zeta),
        _plasma(work_temperature, theta, scaled_density),
        _photo(work_temperature, theta, scaled_density, zeta),
        _bremsstrahlung(density, work_temperature, mean_mass, mean_charge),
    )
    components = tuple(jnp.where(active, value / density, 0.0) for value in components)
    total = sum(components)
    valid = valid & jnp.all(jnp.stack([jnp.isfinite(x) for x in components]), axis=0) & (total >= 0)
    return NeutrinoLosses(*[jnp.where(valid, x, jnp.nan) for x in (*components, total)], valid)
