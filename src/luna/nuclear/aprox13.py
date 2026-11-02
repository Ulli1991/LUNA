"""Alpha-chain composition, reaction algebra and unscreened effective rate assembly."""

import math

import jax.numpy as jnp
import numpy as np

from luna.eos.helmholtz import NA, _require_fp64

SPECIES = (
    "he4",
    "c12",
    "o16",
    "ne20",
    "mg24",
    "si28",
    "s32",
    "ar36",
    "ca40",
    "ti44",
    "cr48",
    "fe52",
    "ni56",
)
MASS_NUMBERS = (4, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56)
CHARGES = tuple(a // 2 for a in MASS_NUMBERS)
BINDING_MEV = (
    28.29603,
    92.16294,
    127.62093,
    160.64788,
    198.25790,
    236.53790,
    271.78250,
    306.72020,
    342.05680,
    375.47720,
    411.46900,
    447.70800,
    484.00300,
)
MEV_TO_ERG = 1.60217648740e-6  # legacy CODATA, consistent with the EOS/reference constants

# Effective reactions only; proton intermediates must be folded into the supplied rates.
REACTIONS = (
    ({"he4": 3}, {"c12": 1}),
    *(({SPECIES[i]: 1, "he4": 1}, {SPECIES[i + 1]: 1}) for i in range(1, 12)),
    ({"c12": 2}, {"ne20": 1, "he4": 1}),
    ({"c12": 1, "o16": 1}, {"mg24": 1, "he4": 1}),
    ({"c12": 1, "o16": 1}, {"si28": 1}),
    ({"o16": 2}, {"si28": 1, "he4": 1}),
    ({"o16": 2}, {"s32": 1}),
)


def _reaction_arrays():
    reactants = np.zeros((len(REACTIONS), len(SPECIES)), dtype=np.int32)
    products = np.zeros_like(reactants)
    for row, (fuel, ash) in enumerate(REACTIONS):
        for name, count in fuel.items():
            reactants[row, SPECIES.index(name)] = count
        for name, count in ash.items():
            products[row, SPECIES.index(name)] = count
    return reactants, products


_REACTANTS, _PRODUCTS = _reaction_arrays()
_FORWARD_SYMMETRY = tuple(math.prod(math.factorial(int(n)) for n in row) for row in _REACTANTS)
_REVERSE_SYMMETRY = tuple(math.prod(math.factorial(int(n)) for n in row) for row in _PRODUCTS)


def composition_valid(mass_fractions, tolerance=1e-12, *, negative_tolerance=0.0):
    """Per-cell validity; never repair or normalize composition silently."""
    fractions = jnp.asarray(mass_fractions)
    if fractions.shape[-1] != len(SPECIES):
        raise ValueError("aprox13 needs 13 mass fractions on the final axis")
    return jnp.all(jnp.isfinite(fractions) & (fractions >= -negative_tolerance), axis=-1) & (
        jnp.abs(jnp.sum(fractions, axis=-1) - 1) <= tolerance
    )


def mean_composition(mass_fractions, *, negative_tolerance=0.0):
    """Return mean mass number and charge for the Helmholtz evaluator; invalid cells yield NaNs."""
    _require_fp64()
    fractions = jnp.asarray(mass_fractions, jnp.float64)
    valid = composition_valid(fractions, negative_tolerance=negative_tolerance)
    molar = fractions / jnp.asarray(MASS_NUMBERS)
    mean_mass = 1 / jnp.sum(molar, axis=-1)
    mean_charge = mean_mass * jnp.sum(molar * jnp.asarray(CHARGES), axis=-1)
    return jnp.where(valid, mean_mass, jnp.nan), jnp.where(valid, mean_charge, jnp.nan)


def binding_energy(mass_fractions):
    """Positive specific nuclear binding energy in erg/g; invalid composition yields NaN."""
    _require_fp64()
    fractions = jnp.asarray(mass_fractions, jnp.float64)
    energy = (
        NA
        * MEV_TO_ERG
        * jnp.sum(fractions / jnp.asarray(MASS_NUMBERS) * jnp.asarray(BINDING_MEV), axis=-1)
    )
    return jnp.where(composition_valid(fractions), energy, jnp.nan)


def _mass_action(molar, multiplicities):
    factors = jnp.ones(molar.shape[:-1] + multiplicities.shape, molar.dtype)
    for order in range(1, 4):
        factors = factors * jnp.where(multiplicities >= order, molar[..., None, :], 1.0)
    return jnp.prod(factors, axis=-1)


def reaction_flows(density, mass_fractions, forward, reverse, *, negative_tolerance=0.0):
    """Net mol/g/s flows from unscreened or screened coefficients; coefficients exclude density/symmetry."""
    _require_fp64()
    fractions = jnp.asarray(mass_fractions, jnp.float64)
    valid = composition_valid(fractions, negative_tolerance=negative_tolerance)
    forward, reverse = jnp.asarray(forward, jnp.float64), jnp.asarray(reverse, jnp.float64)
    if forward.shape[-1] != len(REACTIONS) or reverse.shape[-1] != len(REACTIONS):
        raise ValueError("expected 17 forward and reverse effective reaction coefficients")
    density = jnp.asarray(density, jnp.float64)
    valid = (
        valid
        & jnp.isfinite(density)
        & (density > 0)
        & jnp.all(jnp.isfinite(forward) & (forward >= 0), axis=-1)
        & jnp.all(jnp.isfinite(reverse) & (reverse >= 0), axis=-1)
    )
    molar = fractions / jnp.asarray(MASS_NUMBERS)
    forward_order = jnp.asarray(_REACTANTS.sum(axis=-1))
    reverse_order = jnp.asarray(_PRODUCTS.sum(axis=-1))
    forward_flow = (
        forward
        * density[..., None] ** (forward_order - 1)
        * _mass_action(molar, jnp.asarray(_REACTANTS))
        / jnp.asarray(_FORWARD_SYMMETRY)
    )
    reverse_flow = (
        reverse
        * density[..., None] ** (reverse_order - 1)
        * _mass_action(molar, jnp.asarray(_PRODUCTS))
        / jnp.asarray(_REVERSE_SYMMETRY)
    )
    return jnp.where(valid[..., None], forward_flow - reverse_flow, jnp.nan)


def derivatives_from_flows(flows):
    """Return dX/dt and nuclear heating (erg/g/s); neutrino losses are separate."""
    _require_fp64()
    flows = jnp.asarray(flows, jnp.float64)
    if flows.shape[-1] != len(REACTIONS):
        raise ValueError("expected 17 effective reaction flows")
    molar_derivative = flows @ jnp.asarray(_PRODUCTS - _REACTANTS)
    mass_derivative = molar_derivative * jnp.asarray(MASS_NUMBERS)
    heating = NA * MEV_TO_ERG * jnp.sum(molar_derivative * jnp.asarray(BINDING_MEV), axis=-1)
    return mass_derivative, heating


def effective_rates(
    temperature, *, density=1.0, mass_fractions=None, screening="none", negative_tolerance=0.0
):
    """All 17 effective coefficients; screen raw channels before folding proton paths."""
    from luna.nuclear.alpha_rates import alpha_capture, equilibrium_fraction, proton_branches
    from luna.nuclear.light_rates import (
        triple_alpha,
        carbon_fusion,
        carbon_oxygen_fusion,
        oxygen_fusion,
        RatePair,
    )

    if screening not in ("none", "screen5"):
        raise ValueError("screening must be 'none' or 'screen5'")
    if screening == "screen5" and mass_fractions is None:
        raise ValueError("screen5 needs mass fractions")
    temperature, density = jnp.broadcast_arrays(jnp.asarray(temperature), jnp.asarray(density))
    helium = triple_alpha(temperature)
    alpha = alpha_capture(temperature)
    carbon = carbon_fusion(temperature)
    mixed = carbon_oxygen_fusion(temperature)
    oxygen = oxygen_fusion(temperature)
    alpha_proton, proton_capture = proton_branches(temperature)
    if screening == "screen5":
        from luna.nuclear.screening import channel_enhancements

        triple_factor, alpha_factor, proton_factor, fusion_factor = channel_enhancements(
            density, temperature, mass_fractions, negative_tolerance=negative_tolerance
        )

        def enhanced(pair, factor):
            return RatePair(pair.forward * factor, pair.reverse * factor)

        helium = enhanced(helium, triple_factor)
        alpha = enhanced(alpha, alpha_factor)
        alpha_proton = enhanced(alpha_proton, alpha_factor[..., 3:])
        proton_capture = enhanced(proton_capture, proton_factor)
        carbon = enhanced(carbon, fusion_factor[..., 0])
        mixed = enhanced(mixed, fusion_factor[..., 1])
        oxygen = enhanced(oxygen, fusion_factor[..., 2])
    fractions = equilibrium_fraction(alpha_proton, proton_capture, density=density)
    alpha = RatePair(
        alpha.forward.at[..., 3:].add(alpha_proton.forward * (1 - fractions)),
        alpha.reverse.at[..., 3:].add(proton_capture.reverse * fractions),
    )
    fraction = fractions[..., 1]
    forward_fusion = jnp.stack(
        [
            carbon.forward,
            0.5 * mixed.forward,
            0.5 * mixed.forward,
            (0.56 + 0.34 * fraction) * oxygen.forward,
            (0.10 + 0.34 * (1 - fraction)) * oxygen.forward,
        ],
        axis=-1,
    )
    reverse_fusion = jnp.stack(
        [carbon.reverse, 0.5 * mixed.reverse, 0.5 * mixed.reverse, oxygen.reverse, oxygen.reverse],
        axis=-1,
    )
    return RatePair(
        jnp.concatenate([helium.forward[..., None], alpha.forward, forward_fusion], axis=-1),
        jnp.concatenate([helium.reverse[..., None], alpha.reverse, reverse_fusion], axis=-1),
    )


def fixed_temperature_rhs(
    density, temperature, mass_fractions, *, negative_tolerance=0.0, screening="none"
):
    """Abundance derivative and nuclear heating at prescribed density and temperature."""
    rates = effective_rates(
        temperature,
        density=density,
        mass_fractions=mass_fractions,
        screening=screening,
        negative_tolerance=negative_tolerance,
    )
    return derivatives_from_flows(
        reaction_flows(density, mass_fractions, *rates, negative_tolerance=negative_tolerance)
    )
