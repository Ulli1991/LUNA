"""Independent Radau integration of overdriven steady aprox13 reaction zones."""

import argparse
import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from luna.eos.helmholtz import load_table
from luna.nuclear.detonation import prescribed_ash_cj, make_znd


def check(table_path, output, case="he", overdrive=1.2, tail=False):
    if not np.isfinite(overdrive) or overdrive <= 1:
        raise ValueError("overdrive must exceed one; this is not an eigenvalue search")
    table = load_table(table_path)
    density, temperature = (1e6 if case == "he" else 1e7), 1e8
    fuel = jnp.zeros(13)
    fuel = fuel.at[0].set(1) if case == "he" else fuel.at[1:3].set(0.5)
    ash = jnp.eye(13)[-1]
    cj = prescribed_ash_cj(table, density, temperature, fuel, ash)
    speed = overdrive * cj.speed
    flow = make_znd(table, density, temperature, fuel, speed)
    initial = np.r_[np.asarray(fuel), 0.0]
    index = 0 if case == "he" else 1

    def half_burn(time, state):
        return state[index] - 0.5 * initial[index]

    half_burn.terminal, half_burn.direction = not tail, -1
    print(
        f"{case}: overdriven ZND, D={speed:.9g}, frozen T={flow.shock.temperature:.9g}", flush=True
    )
    # A distinct integrator and algebraically constrained flow, not the split hydro algorithm.
    solution = solve_ivp(
        lambda time, state: np.asarray(flow.rhs(jnp.asarray(state))),
        (0.0, 1e-7 if case == "co" and tail else 0.1),
        initial,
        method="Radau",
        jac=lambda time, state: np.asarray(flow.jacobian(jnp.asarray(state))),
        rtol=1e-8,
        atol=np.r_[np.full(13, 1e-16), 1e-10],
        events=half_burn,
        dense_output=True,
    )
    if not solution.success or not solution.t_events[0].size:
        raise AssertionError(
            "steady profile did not reach half fuel consumption: " + solution.message
        )
    time = np.r_[0.0, np.geomspace(max(solution.t[-1] * 1e-8, 1e-30), solution.t[-1], 2047)]
    profile = solution.sol(time)
    recovered = np.stack(jax.vmap(flow.recover)(jnp.asarray(profile[:13].T)), axis=-1)
    rho, temp, residual, mach = recovered.T
    if (
        not np.isfinite(recovered).all()
        or profile[:13].min() < -1e-12
        or max(residual) > 1e-9
        or max(mach) >= 1
    ):
        raise AssertionError("invalid steady detonation profile")
    np.testing.assert_allclose(profile[:13].sum(axis=0), 1.0, rtol=0, atol=2e-11)
    report = dict(
        passed=True,
        case=case,
        jax=jax.__version__,
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        scope="overdriven steady profile; not a self-sustaining speed gate",
        density=density,
        temperature=temperature,
        speed=speed,
        overdrive=overdrive,
        frozen_shock=flow.shock._asdict(),
        prescribed_ash_cj=cj._asdict(),
        half_consumption_time=float(solution.t_events[0][0]),
        half_consumption_length=float(solution.sol(solution.t_events[0][0])[-1]),
        maximum_jump_residual=float(max(residual)),
        maximum_shock_frame_mach=float(max(mach)),
        function_evaluations=solution.nfev,
        time=time.tolist(),
        distance=profile[-1].tolist(),
        fractions=profile[:13].T.tolist(),
        density_profile=rho.tolist(),
        temperature_profile=temp.tolist(),
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("znd.json"))
    parser.add_argument("--case", choices=("he", "co"), default="he")
    parser.add_argument("--overdrive", type=float, default=1.2)
    parser.add_argument("--tail", action="store_true")
    args = parser.parse_args()
    check(args.table, args.report, args.case, args.overdrive, args.tail)
