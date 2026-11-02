"""Standalone Helmholtz interpolation and thermodynamic contracts, without external table data."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from luna.eos.helmholtz import (
    HelmholtzTable,
    _basis,
    _cell,
    _interpolate,
    _coulomb,
    _ions_radiation,
    evaluate,
    evaluate_checked,
    load_table,
)


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("derivative", [0, 1, 2])
def test_hermite_reproduces_polynomials(degree, derivative):
    nodes = jnp.logspace(0.0, 2.0, 5)
    x = jnp.asarray([1.0, 1.2, 3.0, 10.0, 33.0, 100.0])
    i, z, dx = _cell(nodes, x)
    weights = _basis(z, dx, degree, derivative)
    for power in range(2 * degree + 2):
        values = []
        for endpoint in (0, 1):
            for order in range(degree + 1):
                if order > power:
                    values.append(jnp.zeros_like(x))
                else:
                    values.append(
                        math.factorial(power)
                        / math.factorial(power - order)
                        * nodes[i + endpoint] ** (power - order)
                    )
        actual = jnp.sum(weights * jnp.stack(values, axis=-1), axis=-1)
        expected = (
            0 * x
            if derivative > power
            else math.factorial(power)
            / math.factorial(power - derivative)
            * x ** (power - derivative)
        )
        np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=2e-6)


@pytest.mark.parametrize("a,b", [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)])
def test_biquintic_mixed_terms(a, b):
    density, temp = jnp.logspace(0, 1, 4), jnp.logspace(0, 1, 5)
    coeff = np.arange(1, 37).reshape(6, 6) / 36

    def polynomial(d, t, dr, dt):
        value = 0 * d * t
        for p in range(dr, 6):
            for q in range(dt, 6):
                value = value + coeff[p, q] * math.factorial(p) / math.factorial(p - dr) * d ** (
                    p - dr
                ) * math.factorial(q) / math.factorial(q - dt) * t ** (q - dt)
        return value

    data = jnp.stack(
        [
            jnp.stack(
                [polynomial(density[None, :], temp[:, None], r, t) for t in range(3)], axis=-1
            )
            for r in range(3)
        ],
        axis=-2,
    )
    d, t = jnp.asarray([1.0, 2.2, 7.7, 10.0]), jnp.asarray([10.0, 1.5, 4.4, 1.0])
    i, x, dx = _cell(density, d)
    j, y, dy = _cell(temp, t)
    actual = _interpolate(data, i, j, _basis(x, dx, 2, a), _basis(y, dy, 2, b))
    np.testing.assert_allclose(actual, polynomial(d, t, a, b), rtol=1e-11)


@pytest.fixture
def table():
    d, t = jnp.logspace(0, 2, 4), jnp.logspace(3, 5, 4)
    # F = 1e12 (d^2 + d T/1000 - (T/1000)^2).
    dd, tt = jnp.meshgrid(d, t)
    zero = jnp.zeros_like(dd)
    free = jnp.stack(
        [
            jnp.stack(
                [dd**2 + dd * tt / 1000 - (tt / 1000) ** 2, dd / 1000 - 2 * tt / 1e6, zero - 2e-6],
                axis=-1,
            ),
            jnp.stack([2 * dd + tt / 1000, zero + 0.001, zero], axis=-1),
            jnp.stack([zero + 2, zero, zero], axis=-1),
        ],
        axis=-2,
    )
    dpdr = jnp.stack(
        [
            jnp.stack([6 * dd**2 + 2 * dd * tt / 1000, 2 * dd / 1000], axis=-1),
            jnp.stack([12 * dd + 2 * tt / 1000, zero + 0.002], axis=-1),
        ],
        axis=-2,
    )
    return HelmholtzTable(d, t, 1e12 * free, 1e12 * dpdr)


def test_batched_jit_matches_scalar(table):
    rho, temp = jnp.asarray([4.0, 30.0, 80.0]), jnp.asarray([1e3, 3e4, 1e5])
    batched = jax.jit(lambda r, t: evaluate(table, r, t, 4.0, 2.0))(rho, temp)
    scalar = jax.vmap(lambda r, t: evaluate(table, r, t, 4.0, 2.0))(rho, temp)
    assert np.asarray(batched.valid).all()
    for a, b in zip(batched, scalar):
        np.testing.assert_allclose(a, b, rtol=2e-12)


@pytest.mark.parametrize(
    "rho,temp,abar,zbar",
    [
        (1.0, 1e4, 4.0, 2.0),
        (201.0, 1e4, 4.0, 2.0),
        (4.0, 999.0, 4.0, 2.0),
        (4.0, 100001.0, 4.0, 2.0),
        (0.0, 1e4, 4.0, 2.0),
        (-1.0, 1e4, 4.0, 2.0),
        (np.nan, 1e4, 4.0, 2.0),
        (np.inf, 1e4, 4.0, 2.0),
        (4.0, 1e4, 4.0, 0.0),
        (4.0, 1e4, 4.0, 5.0),
    ],
)
def test_invalid_states_refuse(table, rho, temp, abar, zbar):
    out = jax.jit(lambda r, t, a, z: evaluate(table, r, t, a, z))(rho, temp, abar, zbar)
    assert not out.valid
    assert all(np.isnan(a) for a in out[:9])
    with pytest.raises(ValueError, match="invalid Helmholtz state"):
        evaluate_checked(table, rho, temp, abar, zbar)


def test_table_boundaries_and_invalid_neighbour(table):
    out = evaluate(table, jnp.asarray([2.0, 200.0, 0.0]), jnp.asarray([1e3, 1e5, 1e4]), 4.0, 2.0)
    np.testing.assert_array_equal(out.valid, [True, True, False])


def test_maxwell_relations(table):
    rho, temp = 40.0, 2e4
    out = evaluate(table, rho, temp, 4.0, 2.0, coulomb=False)
    dsdr = jax.grad(lambda r: evaluate(table, r, temp, 4.0, 2.0, coulomb=False).entropy)(rho)
    dsdt = jax.grad(lambda t: evaluate(table, rho, t, 4.0, 2.0, coulomb=False).entropy)(temp)
    np.testing.assert_allclose(out.dedr, (out.pressure - temp * out.dpdt) / rho**2, rtol=3e-12)
    np.testing.assert_allclose(dsdr, -out.dpdt / rho**2, rtol=3e-12)
    np.testing.assert_allclose(dsdt, out.cv / temp, rtol=3e-12)


def test_ions_radiation_derivatives():
    rho, temp = 1e6, 1e9
    out = _ions_radiation(rho, temp, 4.0)
    for output, arg, expected in [(0, 0, out[3]), (0, 1, out[4]), (1, 0, out[5]), (1, 1, out[6])]:
        actual = jax.grad(lambda r, t: _ions_radiation(r, t, 4.0)[output], argnums=arg)(rho, temp)
        np.testing.assert_allclose(actual, expected, rtol=3e-15)


def test_table_loader_refuses_incomplete(tmp_path):
    path = tmp_path / "table.dat"
    path.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="541 x 201"):
        load_table(path)


def test_fp32_refused(table):
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="jax_enable_x64"):
            evaluate(table, 40.0, 2e4, 4.0, 2.0)


@pytest.mark.parametrize("temp", [1e6, 1e9])
def test_coulomb_derivatives(temp):
    rho = 1e6
    out = _coulomb(rho, temp, 4.0, 2.0)
    for output, arg, expected in [(0, 0, out[3]), (0, 1, out[4]), (1, 0, out[5]), (1, 1, out[6])]:
        actual = jax.grad(lambda r, t: _coulomb(r, t, 4.0, 2.0)[output], argnums=arg)(rho, temp)
        np.testing.assert_allclose(actual, expected, rtol=2e-13)


def test_pressure_jvp_uses_compressibility(table):
    rho, temp = 40.0, 2e4
    out = evaluate(table, rho, temp, 4.0, 2.0)
    dpdr, dpdt = jax.grad(lambda r, t: evaluate(table, r, t, 4.0, 2.0).pressure, argnums=(0, 1))(
        rho, temp
    )
    np.testing.assert_allclose([dpdr, dpdt], [out.dpdr, out.dpdt], rtol=3e-12)


@pytest.mark.parametrize("guess", [1.0, 2e4, 1e8])
def test_temperature_inversion_roundtrip(table, guess):
    from luna.eos.inversion import temperature_from_energy

    rho = jnp.asarray([4.0, 30.0, 80.0])
    temperature = jnp.asarray([1e3, 3e4, 1e5])
    energy = evaluate(table, rho, temperature, 4.0, 2.0, coulomb=False).energy
    solve = jax.jit(
        lambda r, e: temperature_from_energy(
            table, r, e, 4.0, 2.0, temperature_guess=guess, coulomb=False
        )
    )
    result = solve(rho, energy)
    assert np.asarray(result.converged).all()
    np.testing.assert_allclose(result.temperature, temperature, rtol=1e-8)
    assert np.all(np.asarray(result.relative_correction) <= 1e-8)


def test_temperature_inversion_failures_do_not_poison_neighbours(table):
    from luna.eos.inversion import temperature_from_energy, temperature_from_energy_checked

    energy = float(evaluate(table, 40.0, 2e4, 4.0, 2.0).energy)
    targets = jnp.asarray([energy, -1e99, 1e99, np.nan])
    result = jax.jit(lambda e: temperature_from_energy(table, 40.0, e, 4.0, 2.0))(targets)
    np.testing.assert_array_equal(result.converged, [True, False, False, False])
    np.testing.assert_allclose(result.temperature[0], 2e4, rtol=1e-8)
    assert np.isnan(result.temperature[1:]).all()
    with pytest.raises(ValueError, match="inversion failed"):
        temperature_from_energy_checked(table, 40.0, targets, 4.0, 2.0)


def test_temperature_inversion_iteration_cap(table):
    from luna.eos.inversion import temperature_from_energy

    energy = evaluate(table, 40.0, 2e4, 4.0, 2.0).energy
    result = temperature_from_energy(table, 40.0, energy, 4.0, 2.0, max_iterations=0)
    assert not result.converged
    assert np.isnan(result.temperature)
    assert result.iterations == 0
    result = temperature_from_energy(table, 40.0, energy, 4.0, 2.0, max_iterations=1)
    assert not result.converged
    assert result.iterations == 1


@pytest.mark.parametrize(
    "options",
    [
        {"temperature_guess": -1.0},
        {"temperature_guess": np.nan},
        {"temperature_bounds": (1e5, 1e3)},
        {"temperature_bounds": (1.0, 1e5)},
    ],
)
def test_temperature_inversion_invalid_options(table, options):
    from luna.eos.inversion import temperature_from_energy

    energy = evaluate(table, 40.0, 2e4, 4.0, 2.0).energy
    result = temperature_from_energy(table, 40.0, energy, 4.0, 2.0, **options)
    assert not result.converged
    assert np.isnan(result.temperature)


@pytest.mark.parametrize(
    "options",
    [
        {"temperature_rtol": 0},
        {"energy_rtol": np.nan},
        {"max_iterations": -1},
        {"max_iterations": 1.5},
    ],
)
def test_temperature_solver_controls_refused(table, options):
    from luna.eos.inversion import temperature_from_energy

    with pytest.raises(ValueError, match="tolerances"):
        temperature_from_energy(table, 40.0, 1e12, 4.0, 2.0, **options)


def test_temperature_inversion_forward_derivative(table):
    from luna.eos.inversion import temperature_from_energy

    state = evaluate(table, 40.0, 2e4, 4.0, 2.0, coulomb=False)
    value, derivative = jax.jvp(
        lambda e: temperature_from_energy(table, 40.0, e, 4.0, 2.0, coulomb=False).temperature,
        (state.energy,),
        (jnp.ones_like(state.energy),),
    )
    np.testing.assert_allclose(value, 2e4, rtol=1e-8)
    np.testing.assert_allclose(derivative, 1 / state.cv, rtol=2e-8)


def test_temperature_inversion_freezes_completed_cells(table):
    from luna.eos.inversion import temperature_from_energy

    temperature = jnp.asarray([2e4, 3e4])
    energy = evaluate(table, 40.0, temperature, 4.0, 2.0, coulomb=False).energy
    result = jax.jit(
        lambda e: temperature_from_energy(
            table, 40.0, e, 4.0, 2.0, temperature_guess=jnp.asarray([2e4, 1e5]), coulomb=False
        )
    )(energy)
    assert np.asarray(result.converged).all()
    assert result.iterations[0] == 0
    assert result.iterations[1] > 0
    np.testing.assert_allclose(result.temperature, temperature, rtol=1e-8)
    state = evaluate(table, 40.0, result.temperature, 4.0, 2.0, coulomb=False)
    expected = np.finfo(float).eps * np.abs(energy) / (result.temperature * state.cv)
    np.testing.assert_allclose(result.relative_energy_resolution, expected, rtol=1e-12)


@pytest.mark.parametrize("mean_mass,mean_charge", [(4.0, 2.0), (12.0, 6.0)])
def test_coulomb_branch_join_is_explicit(mean_mass, mean_charge):
    from luna.eos.helmholtz import KB, NA

    density = 1e6
    transition = (
        mean_charge**2
        * (4.8032042712e-10) ** 2
        / KB
        * (4 * np.pi / 3 * NA * density / mean_mass) ** (1 / 3)
    )
    strong = np.asarray(_coulomb(density, transition * (1 - 1e-9), mean_mass, mean_charge))
    weak = np.asarray(_coulomb(density, transition * (1 + 1e-9), mean_mass, mean_charge))
    jump = np.abs(strong - weak) / np.maximum(np.abs(strong), np.abs(weak))
    assert np.isfinite(strong).all() and np.isfinite(weak).all()
    assert 6.9e-6 < jump[1] < 7.2e-6
    assert 0.0535 < jump[2] < 0.0536


@pytest.mark.parametrize("relative_temperature", [0.99, 1.01])
def test_coulomb_maxwell_each_side_of_join(relative_temperature):
    from luna.eos.helmholtz import KB, NA

    density, mean_mass, mean_charge = 1e6, 12.0, 6.0
    transition = (
        mean_charge**2
        * (4.8032042712e-10) ** 2
        / KB
        * (4 * np.pi / 3 * NA * density / mean_mass) ** (1 / 3)
    )
    temperature = transition * relative_temperature
    pressure, _, _, _, dpdt, dedr, cv = _coulomb(density, temperature, mean_mass, mean_charge)
    dsdr, dsdt = jax.grad(lambda r, t: _coulomb(r, t, mean_mass, mean_charge)[2], argnums=(0, 1))(
        density, temperature
    )
    np.testing.assert_allclose(dedr, (pressure - temperature * dpdt) / density**2, rtol=1e-12)
    np.testing.assert_allclose(dsdr, -dpdt / density**2, rtol=1e-12)
    np.testing.assert_allclose(dsdt, cv / temperature, rtol=1e-12)


@pytest.mark.parametrize("velocity", [-3.0, 0.0, 3.0])
def test_eos_hll_agrees_with_gamma_law(velocity):
    from luna.hydro.eos_fluxes import hll_eos_1d
    from luna.hydro.fluxes import hll_flux_primitive

    gamma = 1.4
    left = jnp.array([[1.0, 2.0, 3.0], [velocity, velocity + 0.2, velocity - 0.1], [1.0, 3.0, 2.0]])
    right = jnp.array(
        [[2.0, 1.0, 4.0], [velocity - 0.2, velocity, velocity + 0.3], [2.0, 1.0, 3.0]]
    )

    def conservative(w):
        return jnp.stack([w[0], w[0] * w[1], w[2] / (gamma - 1) + 0.5 * w[0] * w[1] ** 2])

    result = jax.jit(hll_eos_1d)(
        conservative(left),
        conservative(right),
        left[2],
        right[2],
        jnp.sqrt(gamma * left[2] / left[0]),
        jnp.sqrt(gamma * right[2] / right[0]),
    )
    assert np.asarray(result.valid).all()
    np.testing.assert_allclose(
        result.flux, hll_flux_primitive(left, right, 0, gamma), rtol=3e-14, atol=1e-14
    )


def test_eos_hll_species_flux_and_periodic_conservation():
    from luna.hydro.eos_fluxes import hll_eos_1d

    density = jnp.array([1.0, 2.0, 1.5, 0.8])
    velocity = jnp.array([0.1, -0.2, 0.3, -0.1])
    fractions = jnp.array([[0.7, 0.2, 0.4, 0.8], [0.3, 0.8, 0.6, 0.2]])
    pressure = jnp.array([1.0, 2.0, 1.5, 0.5])
    state = jnp.concatenate(
        [
            density[None],
            (density * velocity)[None],
            density * fractions,
            (pressure / 0.4 + 0.5 * density * velocity**2)[None],
        ]
    )
    sound = jnp.sqrt(1.4 * pressure / density)
    result = hll_eos_1d(
        state,
        jnp.roll(state, -1, axis=1),
        pressure,
        jnp.roll(pressure, -1),
        sound,
        jnp.roll(sound, -1),
    )
    np.testing.assert_allclose(
        result.flux[2:-1].sum(axis=0), result.flux[0], rtol=1e-14, atol=1e-14
    )
    updated = state - 0.05 * (result.flux - jnp.roll(result.flux, 1, axis=1))
    np.testing.assert_allclose(updated.sum(axis=1), state.sum(axis=1), rtol=2e-15)
    np.testing.assert_allclose(updated[2:-1].sum(axis=0), updated[0], rtol=2e-15)
    assert np.asarray(updated[2:-1] >= 0).all()
    assert np.isfinite(
        jax.jacfwd(lambda x: hll_eos_1d(x, x, pressure, pressure, sound, sound).flux)(state)
    ).all()


def test_eos_hll_invalid_faces_refuse():
    from luna.hydro.eos_fluxes import hll_eos_1d

    state = jnp.array([1.0, 0.0, 2.0])
    for invalid in (state.at[0].set(0.0), state.at[-1].set(jnp.nan)):
        result = hll_eos_1d(invalid, state, 1.0, 1.0, 1.0, 1.0)
        assert not result.valid and np.isnan(result.flux).all()
    assert not hll_eos_1d(state, state, -1.0, 1.0, 1.0, 1.0).valid


def test_eos_hll_uses_helmholtz_energy(table):
    from luna.hydro.eos_fluxes import hll_eos_1d

    density, velocity = 40.0, 1e5
    thermo = evaluate_checked(table, density, 2e4, 4.0, 2.0)
    energy = density * (thermo.energy + 0.5 * velocity**2)
    state = jnp.array([density, density * velocity, density, energy])
    result = jax.jit(hll_eos_1d)(
        state, state, thermo.pressure, thermo.pressure, thermo.sound_speed, thermo.sound_speed
    )
    assert result.valid
    expected = jnp.array(
        [
            density * velocity,
            density * velocity**2 + thermo.pressure,
            density * velocity,
            (energy + thermo.pressure) * velocity,
        ]
    )
    np.testing.assert_allclose(result.flux, expected, rtol=2e-15)


@pytest.mark.parametrize("order", [1, 2])
def test_stellar_hydro_uniform_and_periodic_conservation(table, order):
    from luna.hydro.stellar_1d import make_solver

    density = jnp.full(8, 40.0)
    temperature = jnp.full(8, 2e4)
    velocity = jnp.full(8, 1e5)
    fractions = jnp.zeros((13, 8)).at[0].set(1.0)
    thermo = evaluate(table, density, temperature, 4.0, 2.0)
    state = jnp.concatenate(
        [
            density[None],
            (density * velocity)[None],
            density * fractions,
            (density * (thermo.energy + 0.5 * velocity**2))[None],
        ]
    )
    for boundary in ("periodic", "outflow"):
        solver = make_solver(table, boundary=boundary, order=order)
        result = solver.step(
            state, temperature, 0.1 / float(jnp.max(thermo.sound_speed + abs(velocity))), 1.0
        )
        assert np.asarray(result.valid).all()
        np.testing.assert_allclose(result.state, state, rtol=1e-14, atol=0)
        np.testing.assert_allclose(result.temperature, temperature, rtol=1e-8)
    perturbed = state.at[1].multiply(1 + 0.01 * jnp.sin(jnp.arange(8)))
    result = solver.step(perturbed, temperature, 1e-10, 1.0)
    assert np.asarray(result.valid).all()
    residual = result.state.sum(axis=1) - perturbed.sum(axis=1) - 1e-10 * result.boundary_flux
    scale = jnp.maximum(jnp.sum(abs(perturbed), axis=1), 1.0)
    assert float(jnp.max(abs(residual) / scale)) < 3e-15
    # Rebuild with periodic boundaries to test the global finite-volume invariant.
    result = make_solver(table, boundary="periodic", order=order).step(
        perturbed, temperature, 1e-10, 1.0
    )
    np.testing.assert_allclose(
        result.state.sum(axis=1), perturbed.sum(axis=1), rtol=3e-15, atol=1e-12
    )
    np.testing.assert_allclose(result.state[2:-1].sum(axis=0), result.state[0], rtol=2e-15)
    bad = solver.step(state, temperature, 1.0, 1.0)
    assert not np.asarray(bad.valid).any() and np.isnan(bad.state).all()


def test_stellar_hydro_composition_contact(table):
    from luna.hydro.stellar_1d import make_solver

    cells = 16
    phase = 2 * jnp.pi * (jnp.arange(cells) + 0.5) / cells
    density = jnp.full(cells, 40.0)
    temperature = jnp.full(cells, 2e4)
    fractions = jnp.zeros((13, cells)).at[0].set(0.5 + 0.4 * jnp.sin(phase))
    fractions = fractions.at[1].set(1 - fractions[0])
    mass = 1 / (fractions[0] / 4 + fractions[1] / 12)
    thermo = evaluate(table, density, temperature, mass, mass / 2)
    velocity = 1e5
    state = jnp.concatenate(
        [
            density[None],
            jnp.full((1, cells), density[0] * velocity),
            density * fractions,
            (density * (thermo.energy + 0.5 * velocity**2))[None],
        ]
    )
    solver = make_solver(table, boundary="periodic")
    dt = 0.1 / float(jnp.max(thermo.sound_speed + velocity))
    result = solver.step(state, temperature, dt, 1.0)
    assert np.asarray(result.valid).all()
    assert not np.asarray(result.slope_reduced).any()
    assert np.asarray(result.state[2:-1]).min() >= 0
    np.testing.assert_allclose(result.state[2:-1].sum(axis=0), result.state[0], rtol=3e-15)
    np.testing.assert_allclose(result.state.sum(axis=1), state.sum(axis=1), rtol=3e-15, atol=1e-10)


def test_stellar_admissibility_limiter_conserves_flux():
    from luna.hydro.stellar_1d import FluidState, admissible_flux

    state = jnp.ones((3, 4)).at[1].set(0).at[-1].set(2)
    low = jnp.zeros((3, 5))
    high = low.at[0, 2].set(3)

    def recover(value, guess):
        pressure = value[-1] - 0.5 * value[1] ** 2 / value[0]
        valid = (value[0] > 0) & (pressure > 0) & jnp.isfinite(pressure)
        return FluidState(guess, pressure, jnp.ones(4), valid)

    def advance(value):
        return admissible_flux(value, jnp.ones(4), high, low, 0.5, 1.0, recover)

    flux, updated, after, fraction = jax.jit(advance)(state)
    assert np.asarray(after.valid).all()
    assert float(fraction) == 0.5
    np.testing.assert_allclose(updated.sum(axis=1), state.sum(axis=1), rtol=0, atol=0)
    np.testing.assert_allclose(flux[:, 0], flux[:, -1], rtol=0, atol=0)
    gradient = jax.grad(lambda value: jnp.sum(advance(value)[1] ** 2))(state)
    assert np.isfinite(gradient).all()
    invalid = state.at[0].set(-1)
    assert not np.asarray(advance(invalid)[2].valid).all()


def test_spherical_shell_gravity_and_energy_exchange():
    from luna.gravity.spherical_1d import G, shell_gravity, conservative_work

    cells, dx, density = 32, 1e6, 1e6
    rho = jnp.full(cells, density)
    radius = dx * jnp.arange(cells + 1)
    a, b = radius[:-1], radius[1:]
    volume = 4 * jnp.pi / 3 * (b**3 - a**3)
    meanr = 0.75 * (b**4 - a**4) / (b**3 - a**3)
    meanr2 = 0.6 * (b**5 - a**5) / (b**3 - a**3)
    fields = shell_gravity(rho, dx)
    mass = 4 * jnp.pi / 3 * density * radius[-1] ** 3
    np.testing.assert_allclose(
        fields.acceleration, -4 * jnp.pi * G * density * meanr / 3, rtol=2e-13
    )
    np.testing.assert_allclose(
        fields.potential,
        -G * mass / (2 * radius[-1] ** 3) * (3 * radius[-1] ** 2 - meanr2),
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        fields.binding_energy, -3 * G * mass**2 / (5 * radius[-1]), rtol=2e-14
    )
    flux = jnp.sin(jnp.pi * jnp.arange(cells + 1) / cells) * 1e11
    flux = flux.at[0].set(0).at[-1].set(0)
    dt = 1e-4
    updated = (
        rho - dt * 4 * jnp.pi * (radius[1:] ** 2 * flux[1:] - radius[:-1] ** 2 * flux[:-1]) / volume
    )
    work = conservative_work(rho, updated, flux, dx, dt)
    change = shell_gravity(updated, dx).binding_energy - fields.binding_energy
    assert abs(float(jnp.dot(work, volume) + change)) / abs(float(fields.binding_energy)) < 1e-15
    gradient = jax.grad(lambda value: shell_gravity(value, dx).binding_energy)(rho)
    np.testing.assert_allclose(gradient / volume, fields.potential, rtol=2e-14)


def test_stellar_spherical_static_uniform(table):
    from luna.hydro.stellar_1d import make_solver

    cells = 8
    density, temperature = jnp.full(cells, 40.0), jnp.full(cells, 2e4)
    thermo = evaluate(table, density, temperature, 4.0, 2.0)
    state = (
        jnp.zeros((16, cells))
        .at[0]
        .set(density)
        .at[2]
        .set(density)
        .at[-1]
        .set(density * thermo.energy)
    )
    result = make_solver(table, geometry="spherical").step(state, temperature, 1e-11, 1.0)
    assert np.asarray(result.valid).all()
    np.testing.assert_allclose(result.state, state, rtol=1e-14, atol=1e-5)
    assert np.asarray(result.face_flux[[0, 2, 15], 0] == 0).all()
