"""Experimental FP64 stellar 1D runner: `python main.py stellar --help`."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from luna.eos.helmholtz import load_table
from luna.hydro.stellar_1d import make_solver
from luna.nuclear.burn import initial_state
from luna.nuclear.coupling import make_step


@dataclass(frozen=True)
class Config:
    case: str = "he"
    cells: int = 64
    length: float = 1e8
    density: float = 1e6
    temperature: float = 1e8
    hot_temperature: float = 2e9
    burn: bool = True
    geometry: str = "planar"
    gravity: bool = False
    screening: str = "screen5"
    neutrinos: str = "none"
    cfl: float = 0.1
    burn_rtol: float = 1e-7
    ambient_density: float = 1e-4
    well_balanced: bool = False
    ignition_radius: float = 0.0
    burn_min_temperature: float = 0.0
    burn_min_density: float = 0.0
    suppress_shock_burning: bool = True
    shock_threshold: float = 2 / 3

    def validate(self):
        if self.case not in ("he", "co", "smooth", "white-dwarf"):
            raise ValueError("unknown stellar initial condition")
        if (
            self.cells < 4
            or not np.isfinite(
                [
                    self.length,
                    self.density,
                    self.temperature,
                    self.hot_temperature,
                    self.cfl,
                    self.burn_rtol,
                    self.ignition_radius,
                    self.burn_min_temperature,
                    self.burn_min_density,
                    self.shock_threshold,
                ]
            ).all()
        ):
            raise ValueError("finite stellar inputs and at least four cells required")
        if (
            min(
                self.length,
                self.density,
                self.temperature,
                self.hot_temperature,
                self.burn_rtol,
                self.shock_threshold,
            )
            <= 0
        ):
            raise ValueError("stellar scales must be positive")
        if not 0 < self.cfl <= (1 / 6 if self.geometry == "spherical" else 0.5):
            raise ValueError("CFL exceeds the geometry's supported bound")
        if self.geometry not in ("planar", "spherical") or (
            self.gravity and self.geometry != "spherical"
        ):
            raise ValueError("gravity requires spherical geometry")
        if self.case == "smooth" and self.geometry != "planar":
            raise ValueError("periodic smooth test requires planar geometry")
        if self.case == "white-dwarf" and not (self.gravity and self.geometry == "spherical"):
            raise ValueError("white-dwarf requires spherical self-gravity")
        if self.well_balanced and not (self.case == "white-dwarf" and self.gravity):
            raise ValueError("equilibrium balancing requires a self-gravitating white-dwarf model")
        if self.ignition_radius < 0:
            raise ValueError("ignition radius must be nonnegative")
        if self.ignition_radius > 0:
            if self.case != "white-dwarf" or not self.burn:
                raise ValueError("central ignition requires a burning white-dwarf case")
            if not 2 * self.length / self.cells <= self.ignition_radius < self.length:
                raise ValueError("ignition radius must span at least two cells and fit the domain")
            if self.hot_temperature <= self.temperature:
                raise ValueError("ignition temperature must exceed the background temperature")
        if not np.isfinite(self.ambient_density) or not 0 < self.ambient_density < self.density:
            raise ValueError(
                "ambient density must be positive and below the initial central density"
            )
        if self.screening not in ("screen5", "none") or self.neutrinos not in ("none", "sneut5"):
            raise ValueError("unsupported screening or neutrinos")
        if min(self.burn_min_temperature, self.burn_min_density) < 0:
            raise ValueError("burn thresholds must be nonnegative")


def initial_conditions(table, cfg):
    """Construct base states; the runner applies white-dwarf ignition after saving its reference."""
    fractions = jnp.zeros(13)
    fractions = (
        fractions.at[0].set(1) if cfg.case in ("he", "smooth") else fractions.at[1:3].set(0.5)
    )
    if cfg.case == "white-dwarf":
        from luna.ic_generation.white_dwarf_1d import isothermal_model

        model = isothermal_model(
            table,
            cfg.density,
            cfg.temperature,
            fractions,
            cfg.cells,
            cfg.length / cfg.cells,
            ambient_density=cfg.ambient_density,
        )
        if cfg.ignition_radius >= model.surface_radius:
            raise ValueError("ignition radius must lie inside the white-dwarf surface")
        return model.state, model.temperature
    coordinate = (jnp.arange(cfg.cells) + 0.5) / cfg.cells
    temperature = cfg.temperature + (cfg.hot_temperature - cfg.temperature) * jnp.exp(
        -((coordinate / 0.1) ** 4)
    )
    rho, velocity = jnp.full(cfg.cells, cfg.density), jnp.zeros(cfg.cells)
    if cfg.case == "smooth":
        phase = 2 * jnp.pi * coordinate
        rho *= 1 + 0.01 * jnp.sin(phase)
        temperature = cfg.temperature * (1 + 0.01 * jnp.cos(phase))
        velocity = 1e7 * jnp.cos(phase)
        fractions = (
            jnp.broadcast_to(fractions, (cfg.cells, 13)).at[:, 0].set(0.8 + 0.01 * jnp.sin(phase))
        )
        fractions = fractions.at[:, 1].set(1 - fractions[:, 0])
    thermal = initial_state(table, rho, temperature, fractions)
    state = jnp.concatenate(
        [
            rho[None],
            (rho * velocity)[None],
            (rho[:, None] * thermal[:, :13]).T,
            (rho * (thermal[:, -1] + 0.5 * velocity**2))[None],
        ]
    )
    return state, temperature


def ignite_white_dwarf(table, state, temperature, radius, peak_temperature, dx):
    """Heat a compact central kernel at fixed mass, composition and momentum; return added ergs."""
    if not np.isfinite([radius, peak_temperature, dx]).all() or min(radius, dx) <= 0:
        raise ValueError("finite ignition inputs and positive radius/spacing required")
    centers = (jnp.arange(state.shape[1]) + 0.5) * dx
    active = centers < radius
    profile = jnp.where(active, (1 - (centers / radius) ** 2) ** 2, 0.0)
    if not bool(jnp.any(active)) or peak_temperature <= float(jnp.max(temperature[active])):
        raise ValueError("ignition must heat at least one cell above its initial temperature")
    heated = jnp.where(active, temperature + (peak_temperature - temperature) * profile, temperature)
    rho = state[0]
    thermal = initial_state(table, rho, heated, (state[2:-1] / rho).T)
    energy = rho * thermal[:, -1] + 0.5 * state[1] ** 2 / rho
    updated = state.at[-1].set(jnp.where(active, energy, state[-1]))
    faces = jnp.arange(state.shape[1] + 1) * dx
    volumes = 4 * jnp.pi / 3 * (faces[1:] ** 3 - faces[:-1] ** 3)
    added = float(jnp.dot(updated[-1] - state[-1], volumes))
    if not np.isfinite(added) or added <= 0:
        raise ValueError("ignition did not add finite positive thermal energy")
    return updated, heated, added


def write_checkpoint(
    path, cfg, table_hash, state, temperature, time, steps, diagnostics, *, equilibrium=None
):
    """Atomically write a versioned restart without overwriting an existing checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    handle, temporary = tempfile.mkstemp(prefix=".stellar-", suffix=".h5", dir=path.parent)
    os.close(handle)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs.update(
                schema="luna-stellar-1d-v1",
                config=json.dumps(asdict(cfg)),
                table_sha256=table_hash,
                time=float(time),
                steps=int(steps),
                diagnostics=json.dumps(diagnostics),
            )
            output.create_dataset("state", data=np.asarray(state, dtype=np.float64))
            output.create_dataset("temperature", data=np.asarray(temperature, dtype=np.float64))
            if equilibrium is not None:
                output.create_dataset("equilibrium", data=np.asarray(equilibrium, dtype=np.float64))
        # Hard-link publication refuses a conflicting destination atomically.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def read_checkpoint(path, cfg, table_hash):
    """Refuse mismatched physics, EOS table, schema, or malformed restart arrays."""
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema") != "luna-stellar-1d-v1":
            raise ValueError("unsupported stellar restart schema")
        stored_config = json.loads(source.attrs["config"])
        # Fields added after schema v1 take their defaults.
        stored_config = {**asdict(Config()), **stored_config}
        if (
            stored_config != asdict(cfg)
            or source.attrs["table_sha256"] != table_hash
        ):
            raise ValueError("stellar restart physics or EOS table mismatch")
        state, temperature = np.asarray(source["state"]), np.asarray(source["temperature"])
        time, steps = float(source.attrs["time"]), int(source.attrs["steps"])
        diagnostics = json.loads(source.attrs["diagnostics"])
    if state.shape != (16, cfg.cells) or temperature.shape != (cfg.cells,):
        raise ValueError("stellar restart shape mismatch")
    if (
        not np.isfinite(state).all()
        or not np.isfinite(temperature).all()
        or not np.isfinite(time)
        or time < 0
        or steps < 0
    ):
        raise ValueError("invalid stellar restart values")
    return jnp.asarray(state), jnp.asarray(temperature), time, steps, diagnostics


def run(
    table_path,
    cfg,
    stop_time,
    output_dir,
    *,
    restart=None,
    checkpoint_every=0,
    max_steps=100000,
    dt_max=np.inf,
):
    """Advance globally accepted steps, retrying failures from the unchanged input state."""
    cfg.validate()
    if np.isnan(dt_max) or dt_max <= 0:
        raise ValueError("dt_max must be positive")
    if not np.isfinite(stop_time) or stop_time < 0 or checkpoint_every < 0 or max_steps < 1:
        raise ValueError("nonnegative finite stop time/cadence and positive step budget required")
    table_path, output_dir = Path(table_path), Path(output_dir)
    table_hash = hashlib.sha256(table_path.read_bytes()).hexdigest()
    table = load_table(table_path)
    boundary = (
        "periodic"
        if cfg.case == "smooth"
        else ("reflecting" if cfg.geometry == "planar" else "outflow")
    )
    hydro = make_solver(table, boundary=boundary, geometry=cfg.geometry)
    if restart:
        state, temperature, time, steps, diagnostics = read_checkpoint(restart, cfg, table_hash)
    else:
        state, temperature = initial_conditions(table, cfg)
        time, steps = 0.0, 0
        diagnostics = dict(
            rejected_steps=0,
            burn_suppressed=0,
            burn_skipped=0,
            slope_reductions=0,
            minimum_flux_fraction=1.0,
            boundary_budget=[0.0] * 16,
        )
    equilibrium = None
    if cfg.well_balanced:
        if restart:
            with h5py.File(restart) as source:
                if "equilibrium" not in source:
                    raise ValueError("balanced restart is missing its equilibrium state")
                equilibrium = jnp.asarray(source["equilibrium"])
            if equilibrium.shape != state.shape or not np.isfinite(equilibrium).all():
                raise ValueError("invalid equilibrium restart state")
        else:
            equilibrium = state
    if not restart:
        diagnostics["ignition_energy"] = 0.0
        if cfg.ignition_radius > 0:
            state, temperature, diagnostics["ignition_energy"] = ignite_white_dwarf(
                table,
                state,
                temperature,
                cfg.ignition_radius,
                cfg.hot_temperature,
                cfg.length / cfg.cells,
            )
    if cfg.burn:
        step = make_step(
            table,
            screening=cfg.screening,
            neutrinos=cfg.neutrinos,
            boundary=boundary,
            geometry=cfg.geometry,
            self_gravity=cfg.gravity,
            rtol=cfg.burn_rtol,
            equilibrium=equilibrium,
            suppress_shock_burning=cfg.suppress_shock_burning,
            shock_threshold=cfg.shock_threshold,
            burn_min_temperature=cfg.burn_min_temperature,
            burn_min_density=cfg.burn_min_density,
        )
    elif cfg.gravity:
        from luna.gravity.spherical_1d import make_solver as make_radial

        step = make_radial(table, equilibrium=equilibrium).step
    else:
        step = hydro.step
    if time > stop_time:
        raise ValueError("restart time exceeds requested stop time")
    if not np.asarray(hydro.recover(state, temperature).valid).all():
        raise ValueError("invalid stellar initial/restart state")
    saved = None

    def save():
        nonlocal saved
        path = output_dir / f"state_{steps:08d}.h5"
        write_checkpoint(
            path,
            cfg,
            table_hash,
            state,
            temperature,
            time,
            steps,
            diagnostics,
            equilibrium=equilibrium,
        )
        saved = path
        print(f"stellar: step={steps} t={time:.9e} checkpoint={path}", flush=True)

    if not restart:
        save()
    dx = cfg.length / cfg.cells
    next_dt = diagnostics.get("next_dt") or np.inf
    while time < stop_time:
        if steps >= max_steps:
            if saved != output_dir / f"state_{steps:08d}.h5":
                save()
            raise RuntimeError("stellar run exhausted max_steps")
        primitive = hydro.recover(state, temperature)
        speed = float(jnp.max(abs(state[1] / state[0]) + primitive.sound_speed))
        proposed_dt = min(cfg.cfl * dx / speed, next_dt, dt_max)
        dt = min(proposed_dt, stop_time - time)
        for _attempt in range(21):
            result = step(state, temperature, dt, dx)
            if np.asarray(result.valid).all():
                break
            diagnostics["rejected_steps"] += 1
            dt *= 0.5
        else:
            if saved != output_dir / f"state_{steps:08d}.h5":
                save()
            raise RuntimeError(f"stellar step refused after retries at t={time}")
        if not time + dt > time:
            raise RuntimeError("stellar timestep underflow")
        state, temperature = result.state, result.temperature
        time, steps = time + dt, steps + 1
        next_dt = 1.25 * (dt if _attempt else proposed_dt)
        diagnostics["next_dt"] = next_dt
        diagnostics["slope_reductions"] += int(np.count_nonzero(result.slope_reduced))
        diagnostics["minimum_flux_fraction"] = min(
            diagnostics["minimum_flux_fraction"], float(result.flux_fraction)
        )
        diagnostics["burn_suppressed"] += int(
            np.count_nonzero(getattr(result, "burn_suppressed", False))
        )
        diagnostics["burn_skipped"] = diagnostics.get("burn_skipped", 0) + int(
            np.count_nonzero(getattr(result, "burn_skipped", False))
        )
        budget = np.asarray(diagnostics["boundary_budget"]) + dt * np.asarray(result.boundary_flux)
        diagnostics["boundary_budget"] = budget.tolist()
        if checkpoint_every and steps % checkpoint_every == 0:
            save()
    if saved != output_dir / f"state_{steps:08d}.h5":
        save()
    return dict(time=time, steps=steps, checkpoint=str(saved), diagnostics=diagnostics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument(
        "--config", type=Path, required=True, help="JSON object with Config fields, CGS units"
    )
    parser.add_argument("--stop-time", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--dt-max", type=float, default=np.inf)
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    cfg = Config(**json.loads(args.config.read_text()))
    result = run(
        args.table,
        cfg,
        args.stop_time,
        args.output,
        restart=args.restart,
        checkpoint_every=args.checkpoint_every,
        max_steps=args.max_steps,
        dt_max=args.dt_max,
    )
    print(json.dumps(result, indent=2))
    return result
