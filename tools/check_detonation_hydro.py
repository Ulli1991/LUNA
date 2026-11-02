"""Translate an independently integrated overdriven ZND profile with reactive hydro."""

import argparse
import hashlib
import json
import time as clock
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from luna.eos.helmholtz import load_table, evaluate
from luna.hydro.stellar_1d import make_solver
from luna.nuclear.aprox13 import mean_composition, BINDING_MEV, MASS_NUMBERS, NA, MEV_TO_ERG
from luna.nuclear.coupling import make_step
from luna.nuclear.detonation import make_znd, shock_cell_quadrature


def check(table_path, reference_path, output, grids=(128, 256, 512)):
    reference = json.loads(reference_path.read_text())
    table_hash = hashlib.sha256(table_path.read_bytes()).hexdigest()
    if not reference["passed"] or table_hash != reference["table_sha256"]:
        raise ValueError("invalid or mismatched steady reference")
    table = load_table(table_path)
    rho0, temperature0, speed = (reference[key] for key in ("density", "temperature", "speed"))
    fuel = np.asarray(reference["fractions"][0])
    flow = make_znd(table, rho0, temperature0, fuel, speed)
    half_length = reference["half_consumption_length"]
    length, stop = 40 * half_length, 4 * half_length / speed
    coordinates = np.asarray(reference["distance"])
    composition = np.asarray(reference["fractions"])
    if len(coordinates) < 1000:
        raise ValueError("regenerate the tail reference with the dense sampling grid")
    hydro = make_solver(table)
    advance = make_step(table, screening="screen5", neutrinos="none", rtol=1e-7)
    fuel_mass, fuel_charge = mean_composition(jnp.asarray(fuel))
    fuel_eos = evaluate(table, rho0, temperature0, fuel_mass, fuel_charge)
    cold_state = jnp.concatenate(
        [jnp.array([rho0, 0.0]), rho0 * jnp.asarray(fuel), jnp.array([rho0 * fuel_eos.energy])]
    )
    report = dict(
        passed=False,
        case=reference["case"],
        jax=jax.__version__,
        scope="overdriven ZND translation; not a self-sustaining detonation-speed test",
        table_sha256=table_hash,
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        reference_speed=speed,
        half_consumption_length=half_length,
        length=length,
        duration=stop,
        runs=[],
    )

    def exact(cells, time):
        dx = length / cells
        shock = cells // 3 + speed * time / dx
        points, weights, cold_fraction = shock_cell_quadrature(cells, shock)
        distance = ((shock - points) * dx).ravel()
        if distance.max() > coordinates[-1]:
            raise ValueError("steady reference does not cover the domain's reactive tail")
        fractions = np.stack(
            [np.interp(np.maximum(distance, 0), coordinates, composition[:, i]) for i in range(13)],
            axis=-1,
        )
        original = fractions.copy()
        # Remove only reference-interpolation trace roundoff at the IC boundary.
        fractions = np.maximum(fractions, 0)
        dominant = fractions.argmax(axis=1)
        fractions[np.arange(len(fractions)), dominant] -= fractions.sum(axis=1) - 1
        projection = float(np.max(abs(fractions - original)))
        if projection > 1e-12 or fractions.min() < 0:
            raise ValueError("reference composition needs more than roundoff projection")
        recovered = jax.vmap(flow.recover)(jnp.asarray(fractions))
        rho = jnp.where(distance >= 0, recovered[0], rho0)
        temperature = jnp.where(distance >= 0, recovered[1], temperature0)
        velocity = jnp.where(distance >= 0, speed - rho0 * speed / rho, 0)
        mass, charge = mean_composition(jnp.asarray(fractions))
        thermo = evaluate(table, rho, temperature, mass, charge)
        if not np.asarray(thermo.valid).all():
            raise ValueError("invalid reference EOS state")
        state = jnp.concatenate(
            [
                rho[None],
                (rho * velocity)[None],
                (rho[:, None] * fractions).T,
                (rho * (thermo.energy + 0.5 * velocity**2))[None],
            ]
        )
        average = jnp.sum(state.reshape(16, cells, 3) * weights, axis=-1)
        average += cold_state[:, None] * cold_fraction
        warm_guess = (
            jnp.sum(temperature.reshape(cells, 3) * weights, axis=-1) + cold_fraction * temperature0
        )
        return average, warm_guess, projection

    for cells in grids:
        print(
            f"{reference['case']}: {cells} cells, {cells / 40:.1f} cells per half-reaction length",
            flush=True,
        )
        state, guess, projection = exact(cells, 0.0)
        print(f"{cells} cells: profile initialized", flush=True)
        initial = np.asarray(state)
        boundary_budget = np.zeros(16)
        time, steps, retries, suppressed = 0.0, 0, 0, 0
        dx = length / cells
        positions, times = [], []
        next_dt = np.inf
        while time < stop:
            primitive = hydro.recover(state, guess)
            signal = float(jnp.max(abs(state[1] / state[0]) + primitive.sound_speed))
            dt = min(0.15 * dx / signal, stop - time, next_dt)
            for _attempt in range(20):
                start_wall = clock.monotonic()
                result = advance(state, guess, dt, dx)
                jax.block_until_ready(result.state)
                if steps == 0:
                    print(
                        f"first step attempt {_attempt + 1}: {clock.monotonic() - start_wall:.2f}s, "
                        f"max burn steps={int(np.max(result.burn_steps))}, rejected={int(np.max(result.burn_rejected))}",
                        flush=True,
                    )
                if np.asarray(result.valid).all():
                    break
                retries += 1
                dt *= 0.5
            else:
                raise AssertionError(f"detonation step refused at {time}")
            boundary_budget += dt * np.asarray(result.boundary_flux)
            state, guess = result.state, result.temperature
            time, steps, next_dt = time + dt, steps + 1, 1.25 * dt
            suppressed += int(np.count_nonzero(result.burn_suppressed))
            gradient = np.maximum(-np.diff(np.asarray(state[0])), 0.0)
            index = int(gradient.argmax())
            window = np.arange(max(index - 2, 0), min(index + 3, cells - 1))
            positions.append(
                float(np.dot((window + 1) * dx, gradient[window]) / gradient[window].sum())
            )
            times.append(time)
            if steps in (1, 5) or steps % 50 == 0:
                print(f"{cells} cells: {steps} steps, {time / stop:.0%}", flush=True)
            if steps > 10000:
                raise AssertionError("detonation step cap exceeded")
        np.savez_compressed(
            output.with_suffix(f".{cells}.npz"),
            state=np.asarray(state),
            initial=initial,
            temperature=np.asarray(guess),
            time=time,
            boundary_budget=boundary_budget,
            dx=dx,
            front_times=np.asarray(times),
            front_positions=np.asarray(positions),
            reference_sha256=report["reference_sha256"],
            table_sha256=table_hash,
        )
        expected, _, _ = exact(cells, stop)
        center = dx * (np.arange(cells) + 0.5)
        distance = (cells // 3) * dx + speed * stop - center
        region = (distance > -4 * half_length) & (distance < 8 * half_length)
        final = np.asarray(state)
        errors = np.mean(abs(final - np.asarray(expected))[:, region], axis=1)
        errors /= np.array([rho0, rho0 * speed] + [rho0] * 13 + [rho0 * speed**2])
        tail = np.asarray(times) > stop * 0.5
        measured = float(np.polyfit(np.asarray(times)[tail], np.asarray(positions)[tail], 1)[0])
        np.testing.assert_allclose(final[2:-1].sum(axis=0), final[0], rtol=3e-12)
        binding = np.asarray(BINDING_MEV) / np.asarray(MASS_NUMBERS) * NA * MEV_TO_ERG
        change = dx * (final - initial).sum(axis=1) - boundary_budget
        mass_error = abs(change[0]) / (rho0 * length)
        energy_error = abs(change[-1] - np.dot(binding, change[2:-1])) / (rho0 * speed**2 * length)
        if mass_error > 3e-12 or energy_error > 1e-9:
            raise AssertionError(f"detonation budgets failed: {mass_error}, {energy_error}")
        entry = dict(
            cells=cells,
            mass_budget_error=mass_error,
            energy_binding_budget_error=energy_error,
            steps=steps,
            retries=retries,
            burn_suppressed=suppressed,
            reference_trace_projection=projection,
            measured_speed=measured,
            speed_relative_error=abs(measured / speed - 1),
            density_l1_normalized=float(errors[0]),
            fuel_l1_normalized=float(errors[2 if reference["case"] == "he" else 3]),
            energy_l1_normalized=float(errors[-1]),
        )
        report["runs"].append(entry)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
    report["refinement_checked"] = len(report["runs"]) >= 2
    report["passed"] = bool(
        report["refinement_checked"] and report["runs"][-1]["speed_relative_error"] < 0.03
    )
    for a, b in zip(report["runs"], report["runs"][1:]):
        for field in ("density_l1_normalized", "fuel_l1_normalized", "energy_l1_normalized"):
            report["passed"] &= b[field] < 0.8 * a[field]
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError("overdriven hydro speed/profile refinement gate failed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--report", type=Path, default=Path("detonation-hydro.json"))
    parser.add_argument("--grids", type=int, nargs="+", default=[128, 256, 512])
    args = parser.parse_args()
    check(args.table, args.reference, args.report, tuple(args.grids))
