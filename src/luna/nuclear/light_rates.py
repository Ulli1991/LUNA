"""Unscreened light-nucleus fits used by legacy aprox13; no density or abundance factors."""

from typing import NamedTuple

import jax.numpy as jnp

from luna.eos.helmholtz import _require_fp64


class RatePair(NamedTuple):
    forward: object
    reverse: object


def _temperature(temperature):
    _require_fp64()
    temperature = jnp.asarray(temperature, jnp.float64)
    valid = jnp.isfinite(temperature) & (temperature >= 1e6) & (temperature <= 1e10)
    return jnp.where(valid, temperature * 1e-9, 1.0), valid


def _result(forward, reverse, valid):
    return RatePair(jnp.where(valid, forward, jnp.nan), jnp.where(valid, reverse, jnp.nan))


def triple_alpha(temperature):
    """Three-body helium fusion and carbon photodisintegration; symmetry factor 3! is separate."""
    t, valid = _temperature(temperature)
    root = jnp.cbrt(t)
    helium_pair = 7.40e5 * t**-1.5 * jnp.exp(-1.0663 / t) + 4.164e9 * t ** (-2 / 3) * jnp.exp(
        -13.49 / root - t * t / 0.009604
    ) * (1 + 0.031 * root + 8.009 * root**2 + 1.732 * t + 49.883 * t * root + 27.426 * t * root**2)
    beryllium_capture = 130 * t**-1.5 * jnp.exp(-3.3364 / t) + 2.510e7 * t ** (-2 / 3) * jnp.exp(
        -23.57 / root - t * t / 0.055225
    ) * (1 + 0.018 * root + 5.249 * root**2 + 0.650 * t + 19.176 * t * root + 6.034 * t * root**2)
    low_temperature = 0.01 + (0.2 + 0.8 * jnp.exp(-((0.025 / t) ** 3.263))) / (
        1 + 4 * jnp.exp(-((t / 0.025) ** 9.227))
    )
    correction = jnp.where(t > 0.08, 1.0, low_temperature)
    forward = (
        2.90e-16 * helium_pair * beryllium_capture * correction
        + 0.1 * 1.35e-7 * t** -1.5 * jnp.exp(-24.811 / t)
    )
    reverse = 2e20 * t**3 * jnp.exp(-84.424 / t) * forward
    return _result(forward, reverse, valid)


def carbon_alpha(temperature):
    """Carbon alpha capture: legacy 1.7 x CF88 prescription and its reverse."""
    t, valid = _temperature(temperature)
    root = jnp.cbrt(t)
    barrier = jnp.exp(-32.120 / root)
    forward = 1.7 * (
        1.04e8 * barrier * jnp.exp(-t * t / 12.222016) / (t * t * (1 + 0.0489 / root**2) ** 2)
        + 1.76e8 * barrier / (t * t * (1 + 0.2654 / root**2) ** 2)
        + 1.25e3 * t**-1.5 * jnp.exp(-27.499 / t)
        + 1.43e-2 * t**5 * jnp.exp(-15.541 / t)
    )
    return _result(forward, 5.13e10 * t**1.5 * jnp.exp(-83.111 / t) * forward, valid)


def carbon_fusion(temperature):
    """CF88 carbon fusion coefficient; the identical-pair factor 1/2 is separate."""
    t, valid = _temperature(temperature)
    effective_temperature = t / (1 + 0.0396 * t)
    forward = (
        4.27e26
        * effective_temperature ** (5 / 6)
        * t**-1.5
        * jnp.exp(-84.165 / jnp.cbrt(effective_temperature) - 0.00212 * t**3)
    )
    return _result(forward, jnp.zeros_like(t), valid)


def carbon_oxygen_fusion(temperature):
    """Combined CF88 carbon-oxygen coefficient; split equally between its two effective branches."""
    t, valid = _temperature(temperature)
    effective_temperature = t / (1 + 0.055 * t)
    root = jnp.cbrt(effective_temperature)
    forward = (
        1.72e31
        * effective_temperature ** (5 / 6)
        * t**-1.5
        * jnp.exp(-106.594 / root)
        / (jnp.exp(-0.18 * effective_temperature**2) + 0.00106 * jnp.exp(2.562 * root**2))
    )
    return _result(jnp.where(t >= 0.5, forward, 0.0), jnp.zeros_like(t), valid)


def oxygen_fusion(temperature):
    """Combined CF88 oxygen fusion coefficient; branch probabilities are applied separately."""
    t, valid = _temperature(temperature)
    root = jnp.cbrt(t)
    forward = (
        7.10e36
        * t ** (-2 / 3)
        * jnp.exp(-135.93 / root - 0.629 * root**2 - 0.445 * t * root + 0.0103 * t * t)
    )
    return _result(forward, jnp.zeros_like(t), valid)
