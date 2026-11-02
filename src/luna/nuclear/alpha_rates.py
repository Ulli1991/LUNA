"""Alpha captures and proton-mediated channel coefficients, with density factors kept separate."""

import jax.numpy as jnp

from luna.nuclear.light_rates import RatePair, _temperature, _result, carbon_alpha

# Prefactor, barrier, polynomial coefficients, reverse prefactor, reverse exponent (T9 units).
_ALPHA = (
    (4.82e22, 61.015, 0.06340, 0.002541, -0.0002900, 6.461e10, 80.643),
    (1.16e24, 66.690, 0.04913, 0.004637, -0.0004067, 6.616e10, 77.080),
    (2.81e30, 78.271, 0.1458, -0.01069, 0.0003790, 6.740e10, 81.711),
    (4.66e24, 76.435, 0.01650, 0.005973, -0.0003889, 6.843e10, 59.510),
    (1.37e26, 81.227, 0.1066, -0.01102, 0.0005324, 6.928e10, 89.289),
    (1.04e23, 81.420, 0.06325, -0.005671, 0.0002848, 7.001e10, 92.177),
    (1.05e27, 91.674, 0.07846, -0.007430, 0.0003723, 7.064e10, 92.850),
)
_ALPHA_PROTON = (
    (4.16e13, 25.631, 0.002798, 0.002763, -0.0002341, 0.5825, 22.224),
    (1.27e16, 31.044, 0.1041, -0.01368, 0.0006969, 1.144, 21.643),
    (2.76e13, 34.922, 0.004826, -0.005534, 0.0004021, 1.128, 14.959),
    (4.54e14, 32.177, -0.01206, 0.007753, -0.0005071, 2.229, 40.966),
    (6.54e20, 66.678, 0.02655, -0.003947, 0.0002522, 1.104, 4.723),
    (1.83e26, 86.741, 0.01384, 0.001081, -0.00005933, 0.6087, 6.510),
    (1.30e27, 91.674, 0.01367, 0.0007428, -0.00003050, 0.4597, 9.470),
)
_PROTON_CAPTURE = (
    (1.08e16, 27.042, 0.1928, -0.01540, 0.0006444, 3.764e10, 102.865),
    (4.48e16, 29.483, 0.1761, -0.01322, 0.0005245, 7.568e10, 98.722),
    (4.09e16, 31.727, 0.1622, -0.01119, 0.0003910, 7.600e10, 96.657),
    (3.85e16, 33.234, 0.1023, -0.002242, -0.00005463, 1.525e11, 100.475),
    (2.05e17, 35.568, 0.09979, -0.002269, -0.00006662, 7.649e10, 93.999),
    (3.77e17, 37.516, 0.08922, -0.001256, -0.00009453, 1.150e11, 85.667),
    (1.21e18, 39.604, 0.09894, -0.003131, -0.00002160, 1.537e11, 83.382),
)


def _barrier_fit(t, parameters):
    parameters = jnp.asarray(parameters)
    amplitude, barrier, linear, quadratic, cubic, reverse_scale, reverse_barrier = parameters.T
    t = t[..., None]
    polynomial = 1 + t * (linear + t * (quadratic + t * cubic))
    coefficient = jnp.exp(
        jnp.log(amplitude) - (2 / 3) * jnp.log(t) - barrier / jnp.cbrt(t) * polynomial
    )
    ratio = jnp.exp(jnp.log(reverse_scale) - reverse_barrier / t)
    return coefficient, ratio


def alpha_capture(temperature):
    """Eleven (alpha,gamma) pairs, ordered C12 through Fe52 targets; no proton paths yet."""
    t, valid = _temperature(temperature)
    root = jnp.cbrt(t)
    oxygen = (
        9.37e9 * t ** (-2 / 3) * jnp.exp(-39.757 / root - t * t / 2.515396)
        + 62.1 * t**-1.5 * jnp.exp(-10.297 / t)
        + 538 * t**-1.5 * jnp.exp(-12.226 / t)
        + 13 * t * t * jnp.exp(-20.093 / t)
    )
    neon = (
        4.11e11
        * t ** (-2 / 3)
        * jnp.exp(-46.766 / root - t * t / 4.923961)
        * (1 + 0.009 * root + 0.882 * root**2 + 0.055 * t + 0.749 * t * root + 0.119 * t * root**2)
        + 5.27e3 * t**-1.5 * jnp.exp(-15.869 / t)
        + 6.51e3 * t**0.5 * jnp.exp(-16.223 / t)
        + 0.1 * (42.1 * t**-1.5 * jnp.exp(-9.115 / t) + 32 * t ** (-2 / 3) * jnp.exp(-9.383 / t))
    )
    neon = neon / (1 + 5 * jnp.exp(-18.960 / t))
    magnesium = (
        47.8 * t**-1.5 * jnp.exp(-13.506 / t)
        + 2380 * t**-1.5 * jnp.exp(-15.218 / t)
        + 247 * t**1.5 * jnp.exp(-15.147 / t)
        + 0.1
        * (
            1.72e-9 * t**-1.5 * jnp.exp(-5.028 / t)
            + 0.00125 * t**-1.5 * jnp.exp(-7.929 / t)
            + 24.3 / t * jnp.exp(-11.523 / t)
        )
    )
    magnesium = magnesium / (1 + 5 * jnp.exp(-15.882 / t))
    carbon = carbon_alpha(temperature)
    heavy, ratio = _barrier_fit(t, _ALPHA)
    forward = jnp.concatenate(
        [jnp.stack([carbon.forward, oxygen, neon, magnesium], axis=-1), heavy], axis=-1
    )
    reverse = jnp.stack(
        [
            carbon.reverse,
            5.65e10 * t**1.5 * jnp.exp(-54.937 / t) * oxygen,
            6.01e10 * t**1.5 * jnp.exp(-108.059 / t) * neon,
            6.27e10 * t**1.5 * jnp.exp(-115.862 / t) * magnesium,
        ],
        axis=-1,
    )
    reverse = jnp.concatenate([reverse, heavy * ratio * t[..., None] ** 1.5], axis=-1)
    return _result(forward, reverse, valid[..., None])


def proton_branches(temperature):
    """Eight (alpha,p) and (p,gamma) pairs, ordered Mg24 through Fe52 parents."""
    t, valid = _temperature(temperature)
    root = jnp.cbrt(t)
    magnesium_reverse = (
        1.10e8
        * t ** (-2 / 3)
        * jnp.exp(-23.261 / root - t * t / 0.024649)
        * (1 + 0.018 * root + 12.85 * root**2 + 1.61 * t + 89.87 * t * root + 28.66 * t * root**2)
        + 129 * t**-1.5 * jnp.exp(-2.517 / t)
        + 5660 * t**3.5 * jnp.exp(-3.421 / t)
        + 0.1 * (3.89e-8 * t**-1.5 * jnp.exp(-0.853 / t) + 8.18e-9 * t**-1.5 * jnp.exp(-1.001 / t))
    )
    magnesium_reverse /= 1 + jnp.exp(-9.792 / t) / 3 + 2 * jnp.exp(-11.773 / t) / 3
    magnesium_forward = 1.81 * jnp.exp(-18.572 / t) * magnesium_reverse
    aluminium = (
        1.32e9 * t ** (-2 / 3) * jnp.exp(-23.26 / root)
        + 3.22e-10 * 0.17 * t**-1.5 * jnp.exp(-0.836 / t)
        + 1.74 * t**-1.5 * jnp.exp(-2.269 / t)
        + 9.92 * t**-1.5 * jnp.exp(-2.492 / t)
        + 42.9 * t**-1.5 * jnp.exp(-3.273 / t)
        + 134 * t**-1.5 * jnp.exp(-3.654 / t)
        + 1.77e4 * t**0.53 * jnp.exp(-4.588 / t)
    )
    heavy, ratio = _barrier_fit(t, _ALPHA_PROTON)
    # The last two fits tabulate the forward direction; the preceding five tabulate reverse.
    forward = jnp.concatenate(
        [magnesium_forward[..., None], heavy[..., :5] * ratio[..., :5], heavy[..., 5:]], axis=-1
    )
    reverse = jnp.concatenate(
        [magnesium_reverse[..., None], heavy[..., :5], heavy[..., 5:] * ratio[..., 5:]], axis=-1
    )
    capture, capture_ratio = _barrier_fit(t, _PROTON_CAPTURE)
    capture_forward = jnp.concatenate([aluminium[..., None], capture], axis=-1)
    capture_reverse = jnp.concatenate(
        [
            (1.13e11 * t**1.5 * jnp.exp(-134.434 / t) * aluminium)[..., None],
            capture * capture_ratio * t[..., None] ** 1.5,
        ],
        axis=-1,
    )
    return (
        _result(forward, reverse, valid[..., None]),
        _result(capture_forward, capture_reverse, valid[..., None]),
    )


def equilibrium_fraction(alpha_proton, proton_capture, *, density=1.0):
    """Fraction returning to the alpha parent; guard an inactive proton intermediate."""
    denominator = alpha_proton.reverse + proton_capture.forward
    density = jnp.asarray(density)
    scale = density[..., None] if density.ndim else density
    active = scale * denominator > 1e-30
    return jnp.where(active, alpha_proton.reverse / jnp.where(active, denominator, 1.0), 0.0)


def effective_alpha(temperature, *, density=1.0):
    """Eleven alpha-capture pairs with the eight equilibrium proton paths folded in."""
    captures = alpha_capture(temperature)
    alpha_proton, proton_capture = proton_branches(temperature)
    fraction = equilibrium_fraction(alpha_proton, proton_capture, density=density)
    return RatePair(
        captures.forward.at[..., 3:].add(alpha_proton.forward * (1 - fraction)),
        captures.reverse.at[..., 3:].add(proton_capture.reverse * fraction),
    )
