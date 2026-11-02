"""Alpha-chain reaction bookkeeping and binding-energy budgets; no physical burn claims."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from luna.reference_data import reference_path
from luna.nuclear.aprox13 import (
    BINDING_MEV,
    CHARGES,
    MASS_NUMBERS,
    MEV_TO_ERG,
    NA,
    REACTIONS,
    SPECIES,
    binding_energy,
    composition_valid,
    derivatives_from_flows,
    mean_composition,
    reaction_flows,
)


def test_every_reaction_conserves_mass_and_charge():
    derivative, _ = derivatives_from_flows(jnp.eye(len(REACTIONS)))
    np.testing.assert_array_equal(jnp.sum(derivative, axis=-1), 0)
    np.testing.assert_array_equal(derivative @ (np.array(CHARGES) / np.array(MASS_NUMBERS)), 0)


def test_triple_alpha_symmetry_density_and_q():
    fractions = jnp.eye(13)[0]
    forward = jnp.zeros(17).at[0].set(2.0)
    flow = reaction_flows(3.0, fractions, forward, jnp.zeros(17))
    expected = 2 * 3**2 * (1 / 4) ** 3 / 6
    np.testing.assert_allclose(flow[0], expected, rtol=1e-15)
    derivative, heating = derivatives_from_flows(flow)
    np.testing.assert_allclose(derivative[:2], [-12 * expected, 12 * expected])
    np.testing.assert_allclose(heating, expected * 7.27485 * NA * MEV_TO_ERG, rtol=3e-14)
    reverse = reaction_flows(3.0, jnp.eye(13)[1], jnp.zeros(17), forward)
    np.testing.assert_allclose(reverse[0], -2 / 12)


def test_identical_carbon_pair_factor():
    row = next(i for i, (fuel, _) in enumerate(REACTIONS) if fuel == {"c12": 2})
    forward = jnp.zeros(17).at[row].set(7.0)
    flow = reaction_flows(5.0, jnp.eye(13)[1], forward, jnp.zeros(17))
    np.testing.assert_allclose(flow[row], 7 * 5 * (1 / 12) ** 2 / 2)


@pytest.mark.parametrize("species", [0, 1, 12])
def test_pure_compositions(species):
    mean_mass, mean_charge = mean_composition(jnp.eye(13)[species])
    np.testing.assert_allclose([mean_mass, mean_charge], [MASS_NUMBERS[species], CHARGES[species]])


def test_carbon_oxygen_mixture():
    fractions = jnp.zeros(13).at[1:3].set(0.5)
    mean_mass, mean_charge = mean_composition(fractions)
    np.testing.assert_allclose(mean_mass, 1 / (0.5 / 12 + 0.5 / 16))
    np.testing.assert_allclose(mean_charge / mean_mass, 0.5)


def test_energy_budget_and_jit_batch():
    fractions = jnp.ones((4, 13)) / 13
    forward = jnp.arange(1.0, 18.0) * 1e-2
    flows = jax.jit(reaction_flows)(jnp.arange(1.0, 5.0), fractions, forward, jnp.zeros(17))
    derivative, heating = derivatives_from_flows(flows)
    step = 1e-2
    release = binding_energy(fractions + step * derivative) - binding_energy(fractions)
    np.testing.assert_allclose(release, step * heating, rtol=2e-6)


def test_jacobian_is_finite_at_zero_abundance():
    forward = jnp.ones(17)
    jacobian = jax.jacfwd(
        lambda x: derivatives_from_flows(reaction_flows(1.0, x, forward, jnp.zeros(17)))[0]
    )(jnp.eye(13)[0])
    assert np.isfinite(jacobian).all()


@pytest.mark.parametrize("bad", [-0.1, np.nan])
def test_invalid_composition_not_repaired(bad):
    fractions = jnp.eye(13)[0].at[1].set(bad)
    assert not composition_valid(fractions)
    assert np.isnan(binding_energy(fractions))
    assert np.isnan(mean_composition(fractions)).all()
    assert np.isnan(reaction_flows(1.0, fractions, jnp.zeros(17), jnp.zeros(17))).all()


def test_binding_energy_helium_to_nickel():
    release = binding_energy(jnp.eye(13)[12]) - binding_energy(jnp.eye(13)[0])
    expected = (484.003 / 56 - 28.29603 / 4) * NA * MEV_TO_ERG
    np.testing.assert_allclose(release, expected, rtol=3e-15)
    assert SPECIES[-1] == "ni56"
    assert len(BINDING_MEV) == 13


def test_light_rates_against_compiled_reference():
    import json
    from pathlib import Path
    from luna.nuclear import light_rates

    reference = json.loads(Path(reference_path("microphysics/nuclear_rates/nuclear_light_rates.json")).read_text())
    temperature = jnp.asarray(reference["temperature"])
    expected = np.asarray(reference["values"])
    for index, name in enumerate(reference["rates"]):
        actual = np.stack(jax.jit(getattr(light_rates, name))(temperature), axis=-1)
        np.testing.assert_allclose(
            actual, expected[:, index], rtol=3e-12, atol=1e-300, err_msg=name
        )


def test_light_rate_derivatives_and_invalid_states():
    from luna.nuclear import light_rates

    for name in (
        "triple_alpha",
        "carbon_alpha",
        "carbon_fusion",
        "carbon_oxygen_fusion",
        "oxygen_fusion",
    ):
        rate = getattr(light_rates, name)
        for temperature in (1e8, 1e9, 3e9):
            derivative = jax.jacfwd(rate)(temperature)
            assert np.isfinite(derivative).all()
            h = temperature * 1e-5
            finite_difference = (
                np.asarray(rate(temperature + h)) - np.asarray(rate(temperature - h))
            ) / (2 * h)
            np.testing.assert_allclose(derivative, finite_difference, rtol=2e-6, atol=1e-100)
        assert np.isnan(rate(jnp.array([0.0, -1.0, np.nan, 1e11]))).all()


def test_all_unscreened_rates_against_reference():
    import json
    from pathlib import Path
    from luna.nuclear import alpha_rates, light_rates

    reference = json.loads(Path(reference_path("microphysics/nuclear_rates/nuclear_all_rates.json")).read_text())
    temperature = jnp.asarray(reference["temperature"])
    alpha = jax.jit(alpha_rates.alpha_capture)(temperature)
    alpha_proton, capture = jax.jit(alpha_rates.proton_branches)(temperature)
    expected = np.asarray(reference["values"])
    families = {"alpha": alpha, "alpha_proton": alpha_proton, "proton_capture": capture}
    for column, name in enumerate(reference["rates"]):
        if ":" in name:
            family, index = name.split(":")
            actual = np.stack([a[..., int(index)] for a in families[family]], axis=-1)
        else:
            actual = np.stack(jax.jit(getattr(light_rates, name))(temperature), axis=-1)
        np.testing.assert_allclose(
            actual, expected[:, column], rtol=3e-12, atol=1e-270, err_msg=name
        )
    assert len(reference["corrections"]) == 2


def test_effective_branching_and_rhs_conservation():
    from luna.nuclear.aprox13 import effective_rates, fixed_temperature_rhs
    from luna.nuclear.light_rates import carbon_oxygen_fusion, oxygen_fusion

    temperature = jnp.array([1e8, 1e9, 4e9])
    rates = jax.jit(effective_rates)(temperature)
    np.testing.assert_allclose(
        rates.forward[..., 13] + rates.forward[..., 14], carbon_oxygen_fusion(temperature).forward
    )
    np.testing.assert_allclose(
        rates.forward[..., 15] + rates.forward[..., 16], oxygen_fusion(temperature).forward
    )
    fractions = jnp.ones((3, 13)) / 13
    derivative, heating = jax.jit(fixed_temperature_rhs)(1e6, temperature, fractions)
    assert np.isfinite(derivative).all() and np.isfinite(heating).all()
    error = np.abs(np.sum(derivative, axis=-1)) / np.maximum(
        np.max(np.abs(derivative), axis=-1), 1e-300
    )
    assert (error < 1e-14).all()


def test_inactive_proton_path_has_finite_jacobian():
    from luna.nuclear.alpha_rates import equilibrium_fraction
    from luna.nuclear.light_rates import RatePair

    value, gradient = jax.value_and_grad(
        lambda x: equilibrium_fraction(RatePair(x, x), RatePair(x, x))
    )(0.0)
    assert value == 0 and np.isfinite(gradient)


@pytest.fixture
def ion_radiation_table():
    from luna.eos.helmholtz import HelmholtzTable

    # An analytic ions+radiation EOS, not a white-dwarf reference model.
    return HelmholtzTable(
        jnp.array([1e-12, 1e15]),
        jnp.array([1e3, 1e13]),
        jnp.zeros((2, 2, 3, 3)),
        jnp.zeros((2, 2, 2, 2)),
    )


@pytest.mark.parametrize("coulomb", [True, False])
def test_self_heating_rhs_and_implicit_temperature_derivative(ion_radiation_table, coulomb):
    from luna.nuclear.burn import make_network, initial_state, BurnParams

    network = make_network(ion_radiation_table, screening="none", neutrinos="none", coulomb=coulomb)
    fractions = jnp.zeros(13).at[0].set(0.7).at[1].set(0.3)
    state = initial_state(ion_radiation_table, 1e6, 1e9, fractions, coulomb=coulomb)
    params = BurnParams(1e6, 1e9)
    derivative, aux = jax.jit(network.rhs)(state, params)
    assert not aux["invalid"]
    assert np.isfinite(derivative).all()
    assert derivative[-1] > 0
    assert abs(float(jnp.sum(derivative[:13]))) < 1e-12
    tangent = jnp.zeros(14).at[-1].set(state[-1])
    _, jacobian_direction = jax.jvp(lambda y: network.rhs(y, params)[0], (state,), (tangent,))
    h = 1e-5
    difference = (
        network.rhs(state + h * tangent, params)[0] - network.rhs(state - h * tangent, params)[0]
    ) / (2 * h)
    np.testing.assert_allclose(jacobian_direction, difference, rtol=3e-5, atol=1e-90)


@pytest.mark.parametrize("screening,neutrinos", [("unknown", "none"), ("none", "unknown")])
def test_unimplemented_burning_physics_refuses(screening, neutrinos):
    from luna.nuclear.burn import make_network

    with pytest.raises(NotImplementedError, match="screening must"):
        make_network(None, screening=screening, neutrinos=neutrinos)


def test_proton_equilibrium_cutoff_includes_density():
    from luna.nuclear.alpha_rates import equilibrium_fraction
    from luna.nuclear.light_rates import RatePair

    alpha = RatePair(0.0, 1e-35)
    capture = RatePair(1e-35, 0.0)
    assert equilibrium_fraction(alpha, capture, density=1.0) == 0.0
    np.testing.assert_allclose(equilibrium_fraction(alpha, capture, density=1e6), 0.5)


def test_tiny_negative_internal_stages_are_not_clipped(ion_radiation_table):
    from luna.nuclear.burn import make_network, initial_state, BurnParams

    network = make_network(ion_radiation_table, screening="none", neutrinos="none")
    state = initial_state(ion_radiation_table, 1e6, 1e9, jnp.eye(13)[0])
    stage = state.at[12].set(-1e-16)
    derivative, aux = network.rhs(stage, BurnParams(1e6, 1e9))
    assert not aux["invalid"] and np.isfinite(derivative).all()
    assert not composition_valid(stage[:13])
    assert np.isnan(reaction_flows(1e6, stage[:13], jnp.zeros(17), jnp.zeros(17))).all()
    _, aux = network.rhs(state.at[12].set(-1e-10), BurnParams(1e6, 1e9))
    assert aux["invalid"]


@pytest.mark.slow
def test_one_zone_rodas3_finishes_and_conserves(ion_radiation_table):
    from luna.ode.rodas3 import Options, integrate
    from luna.nuclear.burn import make_network, initial_state, BurnParams

    network = make_network(ion_radiation_table, screening="none", neutrinos="none")
    start = initial_state(ion_radiation_table, 1e6, 1e9, jnp.eye(13)[0])
    result = integrate(
        start[None],
        BurnParams(1e6, 1e9),
        1e-5,
        net=network,
        opt=Options(max_steps=200, tiers=((None, 1),), diff_fields=()),
    )
    assert not result.failed[0]
    assert result.t[0] == 1e-5
    assert result.n_steps[0] < 50
    final = result.y[0]
    assert np.asarray(final[:13] >= 0).all()
    np.testing.assert_allclose(final[:13].sum(), 1.0, atol=1e-12, rtol=0)
    release = (
        jnp.sum((final[:13] - start[:13]) / jnp.asarray(MASS_NUMBERS) * jnp.asarray(BINDING_MEV))
        * NA
        * MEV_TO_ERG
    )
    assert release > 0
    np.testing.assert_allclose(final[-1] - start[-1], release, rtol=1e-7)


def test_screen5_against_compiled_reference():
    import json
    from pathlib import Path
    from luna.nuclear.screening import screen5_log

    reference = json.loads(Path(reference_path("microphysics/screen5/screen5_reference.json")).read_text())
    inputs = np.asarray(reference["inputs"])
    value = jax.jit(screen5_log)(inputs[:, 0], inputs[:, 1], inputs[:, 6:], *inputs[:, 2:6].T)
    np.testing.assert_allclose(value, reference["log_enhancement"], rtol=3e-12, atol=1e-13)


def test_screen5_derivatives_and_invalid_composition():
    from luna.nuclear.screening import screen5, screen5_log

    fractions = jnp.zeros(13).at[1:3].set(0.5)
    for density, temperature in [(1e2, 5e9), (1e6, 1e9), (1e9, 1e8)]:
        f = lambda r, t: screen5_log(r, t, fractions, 6, 12, 6, 12)
        dr, dt = jax.grad(f, argnums=(0, 1))(float(density), float(temperature))
        h = 1e-5
        dr_fd = (f(density * (1 + h), temperature) - f(density * (1 - h), temperature)) / (
            2 * h * density
        )
        dt_fd = (f(density, temperature * (1 + h)) - f(density, temperature * (1 - h))) / (
            2 * h * temperature
        )
        np.testing.assert_allclose([dr, dt], [dr_fd, dt_fd], rtol=2e-7, atol=1e-100)
    bad = fractions.at[0].set(-1.0)
    assert np.isnan(screen5(1e6, 1e9, bad, 6, 12, 6, 12))
    assert np.isnan(screen5(0.0, 1e9, fractions, 6, 12, 6, 12))


def test_screened_channels_and_heating():
    from luna.nuclear.aprox13 import effective_rates, fixed_temperature_rhs
    from luna.nuclear.screening import channel_enhancements

    fractions = jnp.eye(13)[0]
    density, temperature = 1e6, 1e9
    factors = channel_enhancements(density, temperature, fractions)
    base = effective_rates(temperature, density=density)
    screened = effective_rates(
        temperature, density=density, mass_fractions=fractions, screening="screen5"
    )
    np.testing.assert_allclose(screened.forward[0] / base.forward[0], factors[0])
    np.testing.assert_allclose(screened.reverse[0] / base.reverse[0], factors[0])
    assert factors[0] > 1
    derivative, heating = fixed_temperature_rhs(
        density, temperature, fractions, screening="screen5"
    )
    assert np.isfinite(derivative).all() and heating > 0
    np.testing.assert_allclose(derivative.sum(), 0.0, atol=1e-12)


def test_screen5_exponent_cap_and_pair_symmetry():
    from luna.nuclear.screening import screen5

    fractions = jnp.eye(13)[1]
    a = screen5(1e10, 1e7, fractions, 6, 12, 8, 16)
    b = screen5(1e10, 1e7, fractions, 8, 16, 6, 12)
    np.testing.assert_allclose(a, b, rtol=1e-14)
    assert np.isfinite(a) and 1 <= a <= np.exp(300.0)


def test_sneut5_against_compiled_reference():
    import json
    from pathlib import Path
    from luna.nuclear.neutrinos import sneut5

    reference = json.loads(Path(reference_path("microphysics/sneut5/sneut5_reference.json")).read_text())
    inputs = np.asarray(reference["inputs"])
    losses = jax.jit(sneut5)(*inputs.T)
    assert np.asarray(losses.valid).all()
    np.testing.assert_allclose(
        np.stack(losses[:4], axis=-1), reference["values"], rtol=1e-12, atol=1e-200
    )
    np.testing.assert_allclose(losses.total, sum(losses[:4]), rtol=1e-15)


def test_sneut5_gradients_and_boundaries():
    from luna.nuclear.neutrinos import sneut5

    for state in [(1e4, 3e8, 4.0, 2.0), (1e7, 8e8, 12.0, 6.0), (1e9, 2e9, 14.0, 7.0)]:
        state = np.asarray(state)
        function = lambda x: sneut5(*x).total
        derivative = jax.grad(function)(jnp.asarray(state))
        finite = []
        for i in range(4):
            delta = np.zeros(4)
            delta[i] = state[i] * 1e-5
            finite.append((function(state + delta) - function(state - delta)) / (2 * delta[i]))
        np.testing.assert_allclose(derivative, finite, rtol=2e-6, atol=1e-100)
    cold = sneut5(1e6, 1e6, 4.0, 2.0)
    assert cold.valid and cold.total == 0
    assert jax.grad(lambda t: sneut5(1e6, t, 4.0, 2.0).total)(1e6) == 0
    assert not sneut5(-1.0, 1e9, 4.0, 2.0).valid
    assert np.isnan(sneut5(1e6, 1e9, 4.0, 5.0).total)
    with pytest.raises(NotImplementedError):
        sneut5(1e6, 1e9, 4.0, 2.0, include_recombination=True)


def test_neutrino_cooling_in_burn_rhs(ion_radiation_table):
    from luna.nuclear.burn import make_network, initial_state, BurnParams
    from luna.nuclear.neutrinos import sneut5

    state = initial_state(ion_radiation_table, 1e6, 1e9, jnp.eye(13)[0])
    params = BurnParams(1e6, 1e9)
    base = make_network(ion_radiation_table, screening="screen5", neutrinos="none")
    cooled = make_network(ion_radiation_table, screening="screen5", neutrinos="sneut5")
    hot_rhs, _ = base.rhs(state, params)
    cold_rhs, aux = cooled.rhs(state, params)
    assert not aux["invalid"]
    np.testing.assert_array_equal(hot_rhs[:13], cold_rhs[:13])
    loss = sneut5(1e6, 1e9, 4.0, 2.0).total
    np.testing.assert_allclose(aux["neutrino_loss"], loss, rtol=1e-12)
    np.testing.assert_allclose(
        hot_rhs[-1] - cold_rhs[-1], loss, rtol=0, atol=np.spacing(float(abs(hot_rhs[-1])))
    )
    assert np.isfinite(jax.jacfwd(lambda y: cooled.rhs(y, params)[0])(state)).all()


def test_complete_rhs_against_compiled_upstream():
    import json
    from pathlib import Path
    from luna.nuclear.aprox13 import (
        effective_rates,
        fixed_temperature_rhs,
        reaction_flows,
        _PRODUCTS,
        _REACTANTS,
    )

    reference = json.loads(Path(reference_path("microphysics/aprox13_rhs/aprox13_rhs_reference.json")).read_text())
    inputs = np.asarray(reference["inputs"])
    expected = np.asarray(reference["derivatives"])
    stoichiometry = jnp.asarray(_PRODUCTS - _REACTANTS)
    for screened in (False, True):
        selected = inputs[:, 2] == screened
        rows = inputs[selected]

        def evaluate(rho, temperature, fractions):
            mode = "screen5" if screened else "none"
            rates = effective_rates(
                temperature, density=rho, mass_fractions=fractions, screening=mode
            )
            forward = reaction_flows(rho, fractions, rates.forward, jnp.zeros_like(rates.reverse))
            reverse = reaction_flows(rho, fractions, jnp.zeros_like(rates.forward), rates.reverse)
            turnover = (abs(forward) + abs(reverse)) @ abs(stoichiometry)
            scale = turnover * jnp.asarray(MASS_NUMBERS)
            heat_scale = jnp.sum(turnover * jnp.asarray(BINDING_MEV), axis=-1) * NA * MEV_TO_ERG
            derivative, heating = fixed_temperature_rhs(rho, temperature, fractions, screening=mode)
            return jnp.concatenate([derivative, heating[..., None]], axis=-1), jnp.concatenate(
                [scale, heat_scale[..., None]], axis=-1
            )

        actual, scale = jax.jit(evaluate)(rows[:, 0], rows[:, 1], rows[:, 3:])
        # Near equilibrium compare against gross turnover, since net derivatives cancel.
        error = abs(np.asarray(actual) - expected[selected])
        assert np.all(error <= 5e-12 * np.asarray(scale) + 1e-250), np.max(
            error / np.maximum(scale, 1e-250)
        )


def test_uniform_reactive_step_and_invalid_cfl(ion_radiation_table):
    from luna.nuclear.burn import initial_state
    from luna.nuclear.coupling import make_step

    density, temperature = 1e6, 1e9
    initial = initial_state(ion_radiation_table, density, temperature, jnp.eye(13)[0])
    state = jnp.concatenate(
        [
            jnp.full((1, 2), density),
            jnp.zeros((1, 2)),
            jnp.broadcast_to((density * initial[:13])[:, None], (13, 2)),
            jnp.full((1, 2), density * initial[-1]),
        ]
    )
    step = make_step(ion_radiation_table, screening="none", neutrinos="none", boundary="periodic")
    result = step(state, jnp.full(2, temperature), 1e-6, 1e6)
    assert np.asarray(result.valid).all()
    final = result.state[:, 0]
    assert final[-1] > state[-1, 0] and final[2] < state[2, 0]
    np.testing.assert_allclose(final[2:-1].sum(), final[0], rtol=1e-12)
    release = (
        jnp.dot(
            (final[2:-1] - state[2:-1, 0]) / density / jnp.asarray(MASS_NUMBERS),
            jnp.asarray(BINDING_MEV),
        )
        * NA
        * MEV_TO_ERG
    )
    np.testing.assert_allclose((final[-1] - state[-1, 0]) / density, release, rtol=1e-7)
    bad = step(state, jnp.full(2, temperature), 0.0, -1.0)
    assert not np.asarray(bad.valid).any() and np.isnan(bad.state).all()
    assert np.asarray(bad.burn_steps == 0).all()
    limited = step(state, jnp.full(2, temperature), 0.01, 1e12)
    assert not np.asarray(limited.valid).any() and np.isnan(limited.state).all()
    assert np.asarray(limited.burn_steps > 0).all()


def test_burn_thresholds_skip_cold_and_dilute_cells(ion_radiation_table):
    from luna.nuclear.burn import initial_state
    from luna.nuclear.coupling import make_step
    from luna.stellar import Config

    # A hot cell and one below the 1e6 K rate table, as the stalled white-dwarf surface.
    density, temperatures = 1e6, jnp.array([1e9, 5e5])
    energies = jnp.stack(
        [initial_state(ion_radiation_table, density, t, jnp.eye(13)[0])[-1] for t in temperatures]
    )
    state = jnp.concatenate(
        [
            jnp.full((1, 2), density),
            jnp.zeros((1, 2)),
            jnp.zeros((13, 2)).at[0].set(density),
            (density * energies)[None],
        ]
    )
    options = dict(screening="none", neutrinos="none", boundary="periodic")
    refused = make_step(ion_radiation_table, **options)(state, temperatures, 1e-12, 1e6)
    assert not bool(refused.valid[1])
    result = make_step(ion_radiation_table, burn_min_temperature=2e7, **options)(
        state, temperatures, 1e-12, 1e6
    )
    assert np.asarray(result.valid).all()
    np.testing.assert_array_equal(result.burn_skipped, [[False, True], [False, True]])
    assert int(result.burn_steps[0]) > 0 and int(result.burn_steps[1]) == 0
    dilute = make_step(ion_radiation_table, burn_min_density=1e6, **options)(
        state.at[:, 1].set(state[:, 0]), temperatures.at[1].set(1e9), 1e-12, 1e6
    )
    assert np.asarray(dilute.burn_skipped).all() and not np.asarray(dilute.burn_steps).any()
    with pytest.raises(ValueError, match="burn thresholds"):
        make_step(ion_radiation_table, burn_min_density=-1.0, **options)
    with pytest.raises(ValueError, match="burn thresholds"):
        Config(burn_min_density=-1.0).validate()
    with pytest.raises(ValueError, match="finite"):
        Config(burn_min_temperature=np.nan).validate()
    with pytest.raises(ValueError, match="positive"):
        Config(shock_threshold=0.0).validate()


def test_shock_burning_detector():
    from luna.nuclear.coupling import shock_mask

    state = jnp.ones((16, 5)).at[1].set(jnp.array([2.0, 1.0, 0.0, -1.0, -2.0]))
    pressure = jnp.array([1.0, 1.0, 2.0, 9.0, 9.0])
    mask = shock_mask(state, pressure)
    assert bool(mask[2])
    assert not np.asarray(shock_mask(state.at[1].multiply(-1), pressure)).any()
    assert not np.asarray(shock_mask(state.at[1].set(0), pressure)).any()
    padded = jnp.pad(pressure, (1, 1), mode="edge")
    gravity = 0.5 * (padded[2:] - padded[:-2])
    assert not np.asarray(shock_mask(state, pressure, acceleration=gravity)).any()


def test_radial_gravity_step_energy_budget(ion_radiation_table):
    from luna.gravity.spherical_1d import make_solver, shell_gravity
    from luna.nuclear.burn import initial_state

    cells, dx, density, temperature = 8, 1e7, 1e6, 1e9
    initial = initial_state(ion_radiation_table, density, temperature, jnp.eye(13)[0])
    state = (
        jnp.zeros((16, cells))
        .at[0]
        .set(density)
        .at[2]
        .set(density)
        .at[-1]
        .set(density * initial[-1])
    )
    radius = dx * jnp.arange(cells + 1)
    volume = 4 * jnp.pi / 3 * (radius[1:] ** 3 - radius[:-1] ** 3)
    initial_energy = float(jnp.dot(state[-1], volume) + shell_gravity(state[0], dx).binding_energy)
    solver = make_solver(ion_radiation_table)
    guess, boundary = jnp.full(cells, temperature), 0.0
    for _ in range(3):
        result = solver.step(state, guess, 1e-4, dx)
        assert np.asarray(result.valid).all()
        state, guess = result.state, result.temperature
        boundary += 4 * np.pi * 1e-4 * float(result.boundary_flux[-1])
    final_energy = float(jnp.dot(state[-1], volume) + shell_gravity(state[0], dx).binding_energy)
    assert abs(final_energy - initial_energy - boundary) / abs(initial_energy) < 2e-14


def test_stellar_checkpoint_roundtrip_and_refusal(tmp_path):
    from luna.stellar import Config, write_checkpoint, read_checkpoint

    cfg = Config(cells=4)
    state, temperature = jnp.ones((16, 4)), jnp.full(4, 1e9)
    path = tmp_path / "state.h5"
    write_checkpoint(path, cfg, "table-hash", state, temperature, 0.25, 8, {"rejected_steps": 2})
    restored = read_checkpoint(path, cfg, "table-hash")
    np.testing.assert_array_equal(restored[0], state)
    np.testing.assert_array_equal(restored[1], temperature)
    assert restored[2:] == (0.25, 8, {"rejected_steps": 2})
    with pytest.raises(FileExistsError):
        write_checkpoint(path, cfg, "table-hash", state, temperature, 0.25, 8, {})
    with pytest.raises(ValueError, match="mismatch"):
        read_checkpoint(path, cfg, "other-table")
    with pytest.raises(ValueError, match="mismatch"):
        read_checkpoint(path, Config(cells=8), "table-hash")
    import h5py
    import json

    with h5py.File(path, "r+") as checkpoint:
        old_config = json.loads(checkpoint.attrs["config"])
        for key in ("ignition_radius", "burn_min_temperature", "shock_threshold"):
            del old_config[key]
        checkpoint.attrs["config"] = json.dumps(old_config)
    np.testing.assert_array_equal(read_checkpoint(path, cfg, "table-hash")[0], state)
    with pytest.raises(ValueError, match="mismatch"):
        read_checkpoint(path, Config(cells=4, ignition_radius=1), "table-hash")
    with pytest.raises(ValueError, match="mismatch"):
        read_checkpoint(path, Config(cells=4, burn_min_temperature=2e7), "table-hash")


def test_central_ignition_energy_and_unchanged_mass(ion_radiation_table):
    from luna.stellar import ignite_white_dwarf
    from luna.nuclear.burn import initial_state
    from luna.hydro.stellar_1d import make_solver

    rho, dx = 1e6, 1e7
    fractions = jnp.zeros(13).at[1:3].set(0.5)
    temperature = jnp.full(8, 1e8)
    thermal = initial_state(ion_radiation_table, rho, temperature, fractions)
    state = (jnp.zeros((16, 8)).at[0].set(rho).at[2:-1].set(rho * fractions[:, None])
             .at[-1].set(rho * thermal[:, -1]))
    heated, guess, added = ignite_white_dwarf(
        ion_radiation_table, state, temperature, 3 * dx, 2e9, dx
    )
    np.testing.assert_array_equal(heated[:-1], state[:-1])
    np.testing.assert_array_equal(heated[:, 3:], state[:, 3:])
    assert np.asarray(guess[:3] > temperature[:3]).all()
    recovered = make_solver(ion_radiation_table).recover(heated, guess)
    assert np.asarray(recovered.valid).all()
    np.testing.assert_allclose(recovered.temperature, guess, rtol=2e-10)
    # Independent shell-volume integration of the injected thermal energy.
    volume = 4 * np.pi * dx**3 * (np.arange(8)**2 + np.arange(8) + 1 / 3)
    np.testing.assert_allclose(added, np.dot(np.asarray(heated[-1] - state[-1]), volume), rtol=2e-15)


def test_ignition_keeps_cold_reference_and_restart_energy(tmp_path, monkeypatch, ion_radiation_table):
    import h5py
    from luna import stellar
    from luna.nuclear.burn import initial_state

    cfg = stellar.Config(case="white-dwarf", cells=4, length=4e7, density=1e6,
                         temperature=1e8, hot_temperature=2e9, ignition_radius=2e7,
                         geometry="spherical", gravity=True, well_balanced=True)
    fractions = jnp.zeros(13).at[1:3].set(0.5)
    thermal = initial_state(ion_radiation_table, cfg.density, cfg.temperature, fractions)
    base = (jnp.zeros((16, 4)).at[0].set(cfg.density)
            .at[2:-1].set(cfg.density * fractions[:, None]).at[-1].set(cfg.density * thermal[-1]))
    monkeypatch.setattr(stellar, "load_table", lambda path: ion_radiation_table)
    monkeypatch.setattr(stellar, "initial_conditions",
                        lambda table, cfg: (base, jnp.full(4, cfg.temperature)))
    references = []
    monkeypatch.setattr(stellar, "make_step",
                        lambda *args, **kwargs: references.append(kwargs["equilibrium"]))
    path = tmp_path / "table"
    path.write_text("synthetic EOS identity")
    first = stellar.run(path, cfg, 0, tmp_path / "first")
    assert first["diagnostics"]["ignition_energy"] > 0
    second = stellar.run(path, cfg, 0, tmp_path / "restart", restart=first["checkpoint"])
    assert second["diagnostics"]["ignition_energy"] == first["diagnostics"]["ignition_energy"]
    for reference in references:
        np.testing.assert_array_equal(reference, base)
    with h5py.File(first["checkpoint"]) as a, h5py.File(second["checkpoint"]) as b:
        np.testing.assert_array_equal(a["state"], b["state"])
        np.testing.assert_array_equal(a["equilibrium"], base)
        assert np.asarray(a["state"][-1] > base[-1]).any()


@pytest.mark.parametrize("changes", [
    {"ignition_radius": -1}, {"ignition_radius": float("nan")},
    {"ignition_radius": 1e7}, {"ignition_radius": 4e7},
    {"burn": False}, {"hot_temperature": 1e7}, {"case": "co"},
])
def test_central_ignition_refuses_invalid_setup(changes):
    from dataclasses import replace
    from luna.stellar import Config

    cfg = Config(case="white-dwarf", cells=4, length=4e7, ignition_radius=2e7,
                 geometry="spherical", gravity=True)
    with pytest.raises(ValueError):
        replace(cfg, **changes).validate()


def test_stellar_diagnostic_burning_and_boundary_budgets():
    from luna.stellar_diagnostics import measure, budgets
    from luna.gravity.spherical_1d import G

    rho, dx, cells = 1e6, 1e7, 4
    fractions = jnp.zeros(13).at[1:3].set(0.5)
    state = (jnp.zeros((16, cells)).at[0].set(rho)
             .at[2:-1].set(rho * fractions[:, None]).at[-1].set(rho * 1e17))
    initial, profile = measure(state, jnp.full(cells, 1e8), dx)
    mass = 4 * np.pi / 3 * rho * (cells * dx)**3
    np.testing.assert_allclose(initial["mass_g"], mass, rtol=2e-15)
    np.testing.assert_allclose(initial["gravitational_energy_erg"],
                               -3 * G * mass**2 / (5 * cells * dx), rtol=2e-15)
    for percent in (50, 90, 99):
        np.testing.assert_allclose(initial["mass_radii_cm"][str(percent)],
                                   cells * dx * (percent / 100)**(1 / 3), rtol=2e-15)
    np.testing.assert_allclose(profile["mass_fractions"].sum(axis=0), 1)
    ash = fractions.at[1].add(-0.1).at[-1].add(0.1)
    heat = float(binding_energy(ash) - binding_energy(fractions))
    burned = state.at[2:-1].set(rho * ash[:, None]).at[-1].add(rho * heat)
    current, _ = measure(burned, jnp.full(cells, 2e8), dx)
    budget = budgets(initial, current, np.zeros(16))
    np.testing.assert_allclose(budget["nuclear_release_erg"], mass * heat, rtol=2e-14)
    np.testing.assert_allclose(budget["net_co_consumed_g"], 0.1 * mass, rtol=2e-14)
    assert budget["energy_budget_error"] < 2e-14
    injected = 1e45
    extra = dict(current, gas_plus_gravity_energy_erg=current["gas_plus_gravity_energy_erg"] + injected)
    wrong = budgets(initial, extra, np.zeros(16))
    np.testing.assert_allclose(wrong["energy_budget_residual_erg"], injected, rtol=1e-11)
    boundary = np.zeros(16)
    boundary[-1] = injected / (4 * np.pi)
    assert budgets(initial, extra, boundary)["energy_budget_error"] < 2e-14


def test_stellar_history_and_radial_profile_export(tmp_path):
    import h5py
    import json
    from luna.stellar import Config, write_checkpoint
    from tools.analyze_stellar import analyze

    cfg = Config(case="white-dwarf", cells=4, length=4e7, density=1e6,
                 geometry="spherical", gravity=True)
    state = jnp.zeros((16, 4)).at[0].set(1e6).at[3:5].set(5e5).at[-1].set(1e23)
    temperature = jnp.full(4, 1e8)
    diagnostic = dict(boundary_budget=[0.0] * 16, ignition_energy=0.0)
    first, last = tmp_path / "initial.h5", tmp_path / "last.h5"
    write_checkpoint(first, cfg, "table", state, temperature, 0, 0, diagnostic)
    write_checkpoint(last, cfg, "table", state, temperature, 0.1, 3, diagnostic)
    result = analyze(first, [last, first], tmp_path / "history.json", tmp_path / "profiles.h5")
    assert result["budget_checks_passed"]
    assert [row["steps"] for row in result["runs"]] == [0, 3]
    with h5py.File(tmp_path / "profiles.h5") as profiles:
        np.testing.assert_array_equal(profiles["step_00000003/density_g_cm3"], state[0])
        assert json.loads(profiles.attrs["species"]) == list(SPECIES)
        assert profiles["step_00000003/mass_fractions"].shape == (13, 4)
    with pytest.raises(ValueError, match="initial checkpoint"):
        analyze(last, [], tmp_path / "bad.json", tmp_path / "bad.h5")


def test_stellar_retry_rolls_back_and_restart_keeps_controller(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from luna import stellar

    state = jnp.zeros((16, 4)).at[0].set(1).at[2].set(1).at[-1].set(2)
    calls = []

    def step(value, guess, dt, dx):
        assert float(jnp.max(abs(value[1]))) < 0.01
        calls.append(dt)
        accepted = dt <= 0.001
        updated = value.at[1].add(dt) if accepted else jnp.full_like(value, 123.0)
        return SimpleNamespace(
            state=updated,
            temperature=guess,
            valid=jnp.full(4, accepted),
            slope_reduced=jnp.zeros(4, dtype=bool),
            flux_fraction=1.0,
            boundary_flux=jnp.zeros(16),
        )

    def recover(value, guess):
        return SimpleNamespace(valid=jnp.ones(4, dtype=bool), sound_speed=jnp.ones(4))

    monkeypatch.setattr(stellar, "load_table", lambda path: None)
    monkeypatch.setattr(stellar, "initial_conditions", lambda table, cfg: (state, jnp.ones(4)))
    monkeypatch.setattr(
        stellar, "make_solver", lambda *args, **kwargs: SimpleNamespace(recover=recover)
    )
    monkeypatch.setattr(stellar, "make_step", lambda *args, **kwargs: step)
    table_path = tmp_path / "table"
    table_path.write_text("test identity")
    cfg = stellar.Config(cells=4, length=4.0)
    first = stellar.run(table_path, cfg, 0.003, tmp_path / "first")
    assert first["diagnostics"]["rejected_steps"] > 0
    count = len(calls)
    second = stellar.run(table_path, cfg, 0.006, tmp_path / "second", restart=first["checkpoint"])
    expected = min(0.003, first["diagnostics"]["next_dt"])
    assert calls[count] == expected
    import h5py

    with h5py.File(second["checkpoint"]) as checkpoint:
        np.testing.assert_allclose(checkpoint["state"][1], 0.006, rtol=2e-15)
    with pytest.raises(RuntimeError, match="max_steps"):
        stellar.run(table_path, cfg, 0.003, tmp_path / "limited", max_steps=1)
    with h5py.File(tmp_path / "limited" / "state_00000001.h5") as checkpoint:
        assert checkpoint.attrs["steps"] == 1
        assert checkpoint.attrs["time"] > 0


def test_radial_reference_balance_and_perturbation(ion_radiation_table):
    from luna.gravity.spherical_1d import make_solver
    from luna.nuclear.burn import initial_state

    cells, density, temperature = 4, 1e6, 1e9
    thermal = initial_state(ion_radiation_table, density, temperature, jnp.eye(13)[0])
    reference = (
        jnp.zeros((16, cells))
        .at[0]
        .set(density)
        .at[2]
        .set(density)
        .at[-1]
        .set(density * thermal[-1])
    )
    solver = make_solver(ion_radiation_table, equilibrium=reference)
    guess = jnp.full(cells, temperature)
    held = solver.step(reference, guess, 1e-4, 1e7)
    assert np.asarray(held.valid).all()
    np.testing.assert_array_equal(held.state, reference)
    np.testing.assert_array_equal(held.boundary_flux, 0.0)
    velocity = jnp.array([1e3, 2e3, 2e3, 1e3])
    perturbed = reference.at[1].set(density * velocity).at[-1].add(0.5 * density * velocity**2)
    evolved = solver.step(perturbed, guess, 1e-4, 1e7)
    assert np.asarray(evolved.valid).all()
    assert float(jnp.max(abs(evolved.state[0] - perturbed[0]))) > 1e-5


def test_prescribed_ash_cj_ideal_gas_limit(ion_radiation_table):
    from luna.nuclear.detonation import prescribed_ash_cj
    from luna.eos.helmholtz import evaluate

    fuel = jnp.eye(13)[0]
    ash = fuel.at[0].set(1 - 1e-5).at[-1].set(1e-5)
    result = prescribed_ash_cj(ion_radiation_table, 1e12, 1e8, fuel, ash, coulomb=False)
    thermo = evaluate(ion_radiation_table, 1e12, 1e8, 4.0, 2.0, coulomb=False)
    gamma = 5 / 3
    sound2 = float(thermo.sound_speed) ** 2
    heat = (gamma**2 - 1) * result.heat
    expected = (sound2 + heat + np.sqrt(heat**2 + 2 * heat * sound2)) ** 0.5
    np.testing.assert_allclose(result.speed, expected, rtol=1e-7)


def test_znd_shock_and_implicit_thermodynamic_derivative(ion_radiation_table):
    from luna.nuclear.detonation import prescribed_ash_cj, make_znd

    fuel = jnp.zeros(13).at[0].set(0.9).at[1].set(0.1)
    cj = prescribed_ash_cj(ion_radiation_table, 1e6, 1e8, fuel, jnp.eye(13)[-1])
    flow = make_znd(ion_radiation_table, 1e6, 1e8, fuel, 1.2 * cj.speed)
    rho, temperature, residual, mach = flow.recover(fuel)
    np.testing.assert_allclose(
        [rho, temperature], [flow.shock.density, flow.shock.temperature], rtol=1e-11
    )
    assert float(residual) < 1e-12 and float(mach) < 1
    direction = jnp.zeros(13).at[0].set(-1).at[1].set(1)
    function = lambda fractions: jnp.log(jnp.asarray(flow.recover(fractions)[:2]))
    derivative = jax.jvp(function, (fuel,), (direction,))[1]
    finite_difference = (
        function(fuel + 1e-5 * direction) - function(fuel - 1e-5 * direction)
    ) / 2e-5
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("fraction", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_shock_reference_cell_averages(fraction):
    from luna.nuclear.detonation import shock_cell_quadrature

    points, weights, right = shock_cell_quadrature(5, 1 + fraction)
    actual = np.sum((2 + 2 * points) * weights, axis=1) + 5 * right
    left = np.arange(5)
    extent = np.clip(1 + fraction - left, 0.0, 1.0)
    expected = 2 * extent + (left + extent) ** 2 - left**2 + 5 * (1 - extent)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-15)
    np.testing.assert_allclose(weights.sum(axis=1) + right, 1.0, rtol=0, atol=3e-16)
