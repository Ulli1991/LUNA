"""Framework-agnostic worker behind the AMUSE interface, on a resident LunaState.

No `amuse` dependency; get/set_grid_state exchange internal energy (docs/TORCH.md)."""
from dataclasses import replace as _replace

import jax
import jax.numpy as jnp
import numpy as np

from luna.species.sinks import Sinks
from luna.utils.grid_params import GridParams
from amuse_luna.particles import PARKED_ATTRS
from amuse_luna.stepper import build as build_luna_state

# Sinks' key in `state.species` (Torch's pointer name is "sink").
SINK_SPECIES = Sinks.name
# A target this many ulps from t_end is t_end: the caller's clock round-trips through its own units.
END_ULPS = 16


class LunaWorker:
    """Resident hydro worker exposing the grid community-code surface."""

    def __init__(self, cfg):
        if cfg.SPATIAL_DIMENSION != 3:
            raise ValueError("LunaWorker (Tier 1) requires SPATIAL_DIMENSION == 3")
        # An AMUSE worker never goes through `setup.cli.launch`, so it sets up x64 and
        # the devices itself. x64 is only ever turned on, never off.
        import jax as _jax
        from luna.setup.jax_init import initialize_jax_distributed
        initialize_jax_distributed(
            enable_x64=bool(getattr(cfg, "enable_x64", False))
            or bool(_jax.config.jax_enable_x64))
        self.cfg = cfg
        self.gamma = float(cfg.gamma)
        self.grid = GridParams.from_config(cfg)
        self.Nx, self.Ny, self.Nz = cfg.Nx, cfg.Ny, cfg.Nz
        self.nvar = 5
        self.dx = cfg.Lx / cfg.Nx
        self.dy = cfg.Ly / cfg.Ny
        self.dz = cfg.Lz / cfg.Nz
        self._compute_dt_fn = None
        self._step_fn = None
        self.state = None          # LunaState, set by initialize_code()
        self._committed = False
        # what Torch has handed over, for the per-bridge-step feedback line
        self._fb = {"n_sn": 0, "E_sn": 0.0, "dm_sn": 0.0, "n_wind": 0, "dm_wind": 0.0,
                    "E_wind": 0.0, "n_src": 0, "Q": 0.0}
        self._ion_fractions = None

    # -- lifecycle ----------------------------------------------------------
    def initialize_code(self, build_ic=False):
        """Build the integrator and the resident state at t=0 (zero grid unless `build_ic`)."""
        # A restart means LUNA owns the problem: gas and radiation come back via `build`.
        restart = bool(getattr(self.cfg, "restart", False))
        build_ic = build_ic or restart
        self._torch_restart_guard(restart)

        # Passing U0 discards the initial conditions `build` makes.
        U0 = None if build_ic else jnp.zeros(
            (self.nvar, self.Nx, self.Ny, self.Nz))
        self.state = build_luna_state(self.cfg, is_main=False, U0=U0)

        if build_ic:
            # Resolving the config can retarget the grid (Sod-Shock-3D -> Ny = Nz = 1),
            # so take the geometry from the resolved config.
            self.cfg = self.state.cfg
            _, self.Nx, self.Ny, self.Nz = self.state.U.shape
            self.grid = GridParams.from_config(self.cfg)
            self.dx = self.cfg.Lx / self.Nx
            self.dy = self.cfg.Ly / self.Ny
            self.dz = self.cfg.Lz / self.Nz

        # Gas<->particle gravity is LUNA physics (lives in gravity/); the store
        # is coupling glue (lives in amuse_luna/).
        from luna.gravity.particle_mesh import ParticleMeshGravity
        from amuse_luna.particles import (
            ParticleStore, default_sink_schema, default_star_schema)
        self.bridge = ParticleMeshGravity(self.cfg)
        self.stars = ParticleStore(
            int(getattr(self.cfg, "star_max_count", 1024)), default_star_schema(), "stars")
        self.sinks = ParticleStore(
            int(self.cfg.sink_max_count), default_sink_schema(), "sinks")
        self._kick_kernels = None
        self._sink_sync_at = None       # (step, t) the sink store last mirrored
        self._pointer = "mass"          # 'mass' -> stars, 'sink' -> sinks
        self._new_star_tags = []
        # LUNA-side star formation: IMF queues per sink, RNG seeded from the config
        self._sf_queues = {}
        self._sf_rng = np.random.default_rng(int(self.cfg.seed) + 7919)
        self._chk_written = -1          # the last coupled checkpoint's number
        self._chk_tick = -1             # the snapshot tick it was written at
        self._chk_wall = None           # and when, by the wall clock
        if restart:
            self._restore_coupled()
        return 0

    # -- restart ------------------------------------------------------------
    # Coupled checkpoint k = LUNA's checkpoint set plus the worker's state (sinks, stars,
    # IMF queues, RNG, feedback), written at Torch's IO_out tick; flash.par must agree.
    def _checkpoint_dir(self):
        from luna.file_io.checkpoint_utils import checkpoint_dir
        return checkpoint_dir(self.cfg)

    def _torch_restart_guard(self, restart):
        """Check that flash.par's `restart` agrees with the LUNA config."""
        import os
        if not os.path.isfile("flash.par"):
            return
        par = {}
        with open("flash.par") as fh:
            for line in fh:
                line = line.split("#", 1)[0]
                if "=" in line:
                    k, v = line.split("=", 1)
                    par[k.strip()] = v.strip().strip('"').strip("'")
        torch_restart = par.get("restart", ".false.").lower() in (".true.", "true")
        if torch_restart != restart:
            raise ValueError(
                f"flash.par says restart={torch_restart} and the LUNA config "
                f"says restart={restart}: Torch and LUNA must restart together")
        self._torch_chknum = int(par.get("checkpointFileNumber", 0)) if torch_restart else None

    def _restore_coupled(self):
        """Restore the worker's half of the coupled checkpoint at the step `build` restarted."""
        import json
        import os
        step = int(self.state.step)
        d = self._checkpoint_dir()
        man = os.path.join(d, f"coupled_{step:06d}.json")
        if not os.path.isfile(man):
            raise FileNotFoundError(
                f"restart: the gas came back at step {step} but no coupled "
                f"checkpoint {man} goes with it (sinks, stars, IMF queues)")
        with open(man) as fh:
            meta = json.load(fh)
        chk = int(meta["chk"])
        if getattr(self, "_torch_chknum", None) is not None and self._torch_chknum != chk:
            raise ValueError(
                f"flash.par restarts Torch from checkpointFileNumber="
                f"{self._torch_chknum} but the LUNA checkpoint set is coupled "
                f"checkpoint {chk} (step {step})")
        z = np.load(os.path.join(d, f"coupled_{step:06d}.npz"), allow_pickle=False)

        def group(prefix):
            return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}

        # LUNA's sinks: the species arrays, replicated again if the mesh wants it
        part = self.state._s.species.get(SINK_SPECIES)
        if part is not None:
            fields = group("sinks.")
            extra = group("sinks.extra.") or None
            part = _replace(part, pos=jnp.asarray(fields["pos"]), p=jnp.asarray(fields["p"]),
                            mass=jnp.asarray(fields["mass"]),
                            active=jnp.asarray(fields["active"]),
                            g=jnp.asarray(fields["g"]) if "g" in fields else None,
                            extra=({**(part.extra or {}), **{k: jnp.asarray(v) for k, v in extra.items()}}
                                   if extra else part.extra))
            sp = next((s for s in self.state.physics.gravity.species if s.name == SINK_SPECIES), None) \
                if self.state.physics.gravity is not None else None
            if sp is not None and hasattr(sp, "settle_after_restart"):
                part = sp.settle_after_restart(part)
            self.state._s.species[SINK_SPECIES] = part
        # Torch's particles and what it parked on them
        self.stars.load({"arrays": group("stars."), "tag_to_slot": meta["stars_tag_to_slot"],
                         "next_tag": meta["stars_next_tag"]})
        self._new_star_tags = [int(t) for t in meta.get("new_star_tags", [])]
        self._pointer = "mass"
        self._sink_sync_at = None
        # LUNA-side star formation: the queues (keyed by sink slot) and the RNG
        self._sf_queues = {int(k): z[f"sfq.{k}"] for k in meta.get("sf_queue_keys", [])}
        self._sf_rng = np.random.default_rng()
        self._sf_rng.bit_generator.state = meta["sf_rng_state"]
        self._fb = {k: (int(v) if k.startswith("n_") else float(v))
                    for k, v in meta.get("feedback", self._fb).items()}
        self._chk_written = chk
        self._chk_tick = int(meta.get("tick", chk))
        print(f"✓ coupled checkpoint {chk} restored: step {step}, t={float(self.state.t):.6e}, "
              f"{self.stars.n_active} stars, {int(np.sum(np.asarray(part.active) > 0)) if part is not None else 0} sinks",
              flush=True)

    def _write_coupled(self, chk, tick=None):
        """Coupled checkpoint `chk` of the state as it is now."""
        import json
        import os
        from jax.experimental import multihost_utils
        from luna.file_io.write_checkpoint_distributed import write_checkpoint_distributed
        s = self.state._s
        step, t = int(s.step), float(s.t)
        d = self._checkpoint_dir()
        write_checkpoint_distributed(s.U, self.cfg, t, step, d, mesh=self.state.mesh)
        if "radiation" in s.sources:
            from luna.file_io.radiation_checkpoint import write_radiation_checkpoint
            write_radiation_checkpoint(s.sources["radiation"], self.cfg, t, step, d,
                                       mesh=self.state.mesh)
        # every rank's shards are in place before the manifest says the set exists
        multihost_utils.sync_global_devices("coupled_checkpoint_shards")
        if jax.process_index() == 0:
            from luna.file_io.checkpoint_sets import mark_complete, prune, write_atomically
            arrays, meta = {}, {"chk": int(chk), "step": step, "t": t,
                                "tick": int(chk if tick is None else tick)}
            part = s.species.get(SINK_SPECIES)
            if part is not None:
                for k in ("pos", "p", "mass", "active", "g"):
                    v = getattr(part, k)
                    if v is not None:
                        arrays[f"sinks.{k}"] = np.asarray(v)
                for k, v in (part.extra or {}).items():
                    arrays[f"sinks.extra.{k}"] = np.asarray(v)
            st = self.stars.dump()
            arrays.update({f"stars.{k}": v for k, v in st["arrays"].items()})
            meta["stars_tag_to_slot"] = {str(k): v for k, v in st["tag_to_slot"].items()}
            meta["stars_next_tag"] = st["next_tag"]
            meta["new_star_tags"] = [int(t) for t in self._new_star_tags]
            meta["sf_queue_keys"] = [str(k) for k in self._sf_queues]
            arrays.update({f"sfq.{k}": np.asarray(v, dtype=np.float64) for k, v in self._sf_queues.items()})
            meta["sf_rng_state"] = self._sf_rng.bit_generator.state
            meta["feedback"] = dict(self._fb)
            os.makedirs(d, exist_ok=True)

            def _npz(path):
                with open(path, "wb") as fh:          # a file object: savez adds no suffix
                    np.savez(fh, **arrays)

            def _json(path):
                with open(path, "w") as fh:
                    json.dump(meta, fh, indent=1)
            write_atomically(os.path.join(d, f"coupled_{step:06d}.npz"), _npz)
            write_atomically(os.path.join(d, f"coupled_{step:06d}.json"), _json)
            mark_complete(d, step)
            print(f"✓ coupled checkpoint {chk} written: step {step}, t={t:.6e}", flush=True)
            prune(d, int(self.cfg.max_checkpoints or 0))
        multihost_utils.sync_global_devices("coupled_checkpoint")

    def commit_parameters(self):
        self._committed = True
        return 0

    def cleanup_code(self):
        self.state = None
        self._compute_dt_fn = self._step_fn = None
        return 0

    def _require_state(self):
        if self.state is None:
            raise RuntimeError("code not initialized — call initialize_code() first")

    # -- grid geometry ------------------------------------------------------
    def get_grid_range(self):
        """Inclusive index bounds of the (single, uniform) grid."""
        return (0, self.Nx - 1, 0, self.Ny - 1, 0, self.Nz - 1)

    def get_position_of_index(self, i, j, k):
        """Cell-centre physical coordinates for integer indices (vectorized)."""
        i = np.asarray(i); j = np.asarray(j); k = np.asarray(k)
        return ((i + 0.5) * self.dx, (j + 0.5) * self.dy, (k + 0.5) * self.dz)

    def get_index_of_position(self, x, y, z):
        """Cell indices containing physical coordinates (clamped to the grid)."""
        x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
        i = np.clip(np.floor(x / self.dx).astype(np.int64), 0, self.Nx - 1)
        j = np.clip(np.floor(y / self.dy).astype(np.int64), 0, self.Ny - 1)
        k = np.clip(np.floor(z / self.dz).astype(np.int64), 0, self.Nz - 1)
        return (i, j, k)

    # -- grid state exchange (the energy seam lives here) -------------------
    def set_grid_state(self, i, j, k, rho, rhovx, rhovy, rhovz, rhoen):
        """Write conservative state at cells (i,j,k); `rhoen` is internal energy density."""
        self._require_state()
        i = np.asarray(i); j = np.asarray(j); k = np.asarray(k)
        rho = jnp.asarray(rho)
        rhovx = jnp.asarray(rhovx); rhovy = jnp.asarray(rhovy); rhovz = jnp.asarray(rhovz)
        ke = 0.5 * (rhovx**2 + rhovy**2 + rhovz**2) / rho
        E = jnp.asarray(rhoen) + ke
        U = self.state.U
        U = U.at[0, i, j, k].set(rho)
        U = U.at[1, i, j, k].set(rhovx)
        U = U.at[2, i, j, k].set(rhovy)
        U = U.at[3, i, j, k].set(rhovz)
        U = U.at[4, i, j, k].set(E)
        self.state.U = U
        return 0

    def get_grid_state(self, i, j, k):
        """Read conservative state at cells (i,j,k); returns `rhoen` *internal*."""
        self._require_state()
        i = np.asarray(i); j = np.asarray(j); k = np.asarray(k)
        U = self.state.U
        rho = U[0, i, j, k]
        rhovx = U[1, i, j, k]; rhovy = U[2, i, j, k]; rhovz = U[3, i, j, k]
        E = U[4, i, j, k]
        ke = 0.5 * (rhovx**2 + rhovy**2 + rhovz**2) / rho
        rhoen = E - ke
        return (np.asarray(rho), np.asarray(rhovx), np.asarray(rhovy),
                np.asarray(rhovz), np.asarray(rhoen))

    def _all_indices(self):
        """Full-grid index mesh — convenience for set/get over the whole grid."""
        i, j, k = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny),
                              np.arange(self.Nz), indexing="ij")
        return i, j, k

    # -- time / evolution ---------------------------------------------------
    def evolve_model(self, t_target):
        """Advance the gas to `t_target` on the CFL dt; t_end within rounding is t_end, a past target a no-op."""
        import time
        self._require_state()
        t_end = float(self.cfg.t_end)
        t_target = float(t_target)
        if abs(t_target - t_end) <= END_ULPS * np.spacing(t_end):
            t_target = t_end
        if not t_target > self.state.t:
            print(f"LUNA evolve_model: target {t_target!r} is not ahead of t = {self.state.t!r}, "
                  "nothing to do", flush=True)
            return 0
        t0 = time.perf_counter()
        self._refresh_ionizing_sources()
        t1 = time.perf_counter()
        self._inject_winds(t_target - float(self.state._s.t))
        t2 = time.perf_counter()
        n = self.state.evolve_to(t_target)
        # the step is dispatched asynchronously: without this the report's
        # first host read would be charged with the rest of the step
        jax.block_until_ready(self.state._s.U)
        t3 = time.perf_counter()
        line = self.feedback_report()
        m = jax.local_devices()[0].memory_stats() or {}
        mem = (f"  device {m['bytes_in_use'] / 2**30:.2f} GiB in use, peak "
               f"{m.get('peak_bytes_in_use', 0) / 2**30:.2f}" if "bytes_in_use" in m else "")
        budget = self.mass_budget_report()
        print(f"{line}\nLUNA evolve_model: sources {t1 - t0:.3f}s  winds {t2 - t1:.3f}s  "
              f"step {t3 - t2:.3f}s ({n} substeps)  report {time.perf_counter() - t3:.3f}s{mem}"
              + (f"\n{budget}" if budget else ""), flush=True)
        return 0

    def mass_budget_report(self):
        """With cfg.boundary_mass_budget: gas + sinks + stars - created - in + out, against this link's first line."""
        from luna.setup.evolve import BUDGET
        b = self.state.sources.get(BUDGET) if self.state is not None else None
        if b is None:
            return ""
        if getattr(self, "_gas_mass", None) is None:
            dV = self.dx * self.dy * self.dz
            self._gas_mass = jax.jit(lambda rho: jnp.sum(rho) * dV)
        gas = float(self._gas_mass(self.state._s.U[0]))
        b = np.asarray(jax.device_get(b))
        m_in, m_out = float(b[0].sum()), float(b[1].sum())
        sinks = sinks_out = 0.0
        part = self.state.species.get(SINK_SPECIES)
        if part is not None:
            m, a = (np.asarray(v) for v in jax.device_get((part.mass, part.active)))
            sinks = float(np.sum(m * (a > 0.5)))
            gone = (part.extra or {}).get("m_out")
            sinks_out = 0.0 if gone is None else float(np.sum(np.asarray(jax.device_get(gone))))
        stars = float(np.sum(np.asarray(jax.device_get(self.stars.arrays["mass"]))))
        fb = self._fb
        # Wind and SN ejecta are not subtracted: Torch already took them off the stars.
        # Mass loading in mode "torch" is created gas and is booked.
        total = gas + sinks + sinks_out + stars - fb.get("dm_load", 0.0) - m_in + m_out
        if getattr(self, "_mass0", None) is None:
            self._mass0 = total
        faces = " ".join(f"{v:.4g}" for v in b[0] - b[1])
        return (f"LUNA mass: gas {gas:.6g} sinks {sinks:.6g} (+{sinks_out:.4g} out of the box) "
                f"stars {stars:.6g}  faces in {m_in:.6g} out {m_out:.6g} (net in per face "
                f"{faces})  winds {fb['dm_wind']:.4g} SN {fb.get('dm_sn', 0.0):.4g} "
                f"(Torch took {fb.get('dm_star_loss', 0.0):.4g} off the stars)  "
                + (f"loaded {fb['dm_load']:.6g} created  " if "dm_load" in fb else "")
                + f"resid {total - self._mass0:.4e} ({(total - self._mass0) / self._mass0:.2e})")

    def _wind_load_velocity(self):
        """cfg.wind_target_temp_K as FLASH-Torch's mass-loading velocity, code units (0 = off)."""
        T = float(getattr(self.cfg, "wind_target_temp_K", 0.0) or 0.0)
        if T <= 0.0:
            return 0.0
        v_cms = (T / 1.38e7) ** 0.5 * 1e8
        u = self.cfg.units
        return v_cms / u.UnitVelocity_in_cm_per_s if (u is not None and u.is_enabled) else v_cms

    def _inject_winds(self, dt):
        """Inject every star's parked wind rate x dt for the coming bridge interval."""
        if dt <= 0.0:
            return
        if str(getattr(self.cfg, "wind_mass_load_mode", "conservative")) != "conservative":
            return self._inject_torch_winds(dt)
        if getattr(self, "_winds", None) is None:
            from luna.sources.feedback import make_wind_injector
            self._winds = make_wind_injector((self.Nx, self.Ny, self.Nz),
                                             (self.dx, self.dy, self.dz),
                                             periodic=int(self.cfg.BOUNDARY) == 0,
                                             add_heat=bool(getattr(self.cfg, "wind_add_heat", False)),
                                             v_load=self._wind_load_velocity(),
                                             radius=float(getattr(self.cfg, "feedback_radius", 0.0) or 0.0) or None,
                                             sharding=getattr(self.state.U, "sharding", None))
        a = self.stars.arrays
        # nothing emitting, nothing to do: no scatter on every bridge step
        if not bool(jax.device_get(jnp.any(a["active"] & (a["wind_mass"] > 0.0)))):
            return
        if not getattr(self, "_wind_eos_checked", False):
            self._wind_eos_checked = True
            capped = bool(getattr(self.cfg, "barotropic_eos", False)) or bool(
                getattr(self.cfg, "rad_ionization", False))
            if not capped and not bool(getattr(self.cfg, "wind_add_heat", False)):
                print("  ! winds by momentum alone in a run whose EOS does not set the "
                      "temperature: the wind cells are cold and hypersonic, and a total-"
                      "energy scheme without dual energy can drive their pressure negative. "
                      "wind_add_heat=True (or a prescribed EOS) is the safer choice here.", flush=True)
        U, n, dm, E = self._winds(self.state.U, a["position"], a["velocity"],
                                  a["wind_mass"] * dt, a["wind_vel"], a["active"])
        self.state.U = U
        self._fb["n_wind"] += int(n)
        self._fb["dm_wind"] += float(dm)
        self._fb["E_wind"] += float(E)

    def _inject_torch_winds(self, dt):
        """The winds by FLASH-Torch's inject_direct (modes "torch"/"sink"), the loaded mass booked."""
        cfg = self.cfg
        mode = str(cfg.wind_mass_load_mode)
        if mode not in ("torch", "sink"):
            raise ValueError(f"wind_mass_load_mode {mode!r}: 'conservative', 'torch' or 'sink'")
        if getattr(self, "_winds", None) is None:
            from luna.sources.feedback import make_torch_wind_injector
            self._winds = make_torch_wind_injector(
                (self.Nx, self.Ny, self.Nz), (self.dx, self.dy, self.dz),
                periodic=int(cfg.BOUNDARY) == 0, v_ref=self._wind_load_velocity(),
                cons_quant=str(cfg.wind_cons_quant),
                add_therm_e=bool(getattr(cfg, "wind_add_heat", False)),
                ref_radius=float(cfg.wind_ref_radius) or None,
                var_radius=bool(cfg.wind_var_radius), source=mode,
                sharding=getattr(self.state.U, "sharding", None))
        a = self.stars.arrays
        if not np.any(a["active"] & (a["wind_mass"] > 0.0)):
            return
        t = float(self.state._s.t)
        sinks = part = None
        if mode == "sink":
            part = self.state.species.get(SINK_SPECIES)
            if part is not None:
                pos, vel, mass, act = (np.array(v, dtype=float) for v in
                                       jax.device_get((part.pos, part.p, part.mass, part.active)))
                sinks = (pos, vel, mass, act)
        m_before = None if sinks is None else sinks[2].copy()
        # nearest sink through a periodic face unless the particles leave the box there
        sink_periodic = (int(cfg.BOUNDARY) == 0
                         and str(getattr(cfg, "particle_boundary", "")) != "outflow")
        U, info = self._winds(self.state.U, a["position"], a["wind_mass"], a["wind_vel"],
                              a["active"], a["wind_bgdy"], t + dt - a["creation_time"], dt,
                              sinks=sinks, sink_periodic=sink_periodic)
        self.state.U = U
        a["wind_bgdy"][:] = info["bg_rho"]
        if sinks is not None and info["dm_from_sinks"] > 0.0:
            changed = np.where(sinks[2] != m_before)[0]
            self.state.species[SINK_SPECIES] = _replace(
                part, mass=jnp.asarray(part.mass).at[changed].set(jnp.asarray(sinks[2][changed])))
            # keep the sink store and its push-back reference in step, or a push back restores the old masses
            self._sink_sync_at = None
            mirror = getattr(self, "_sink_mirror_mass", None) or {}
            for s in changed:
                if int(s) in mirror:
                    mirror[int(s)] = float(sinks[2][s])
                    self.sinks.set("mass", [int(s)], [float(sinks[2][s])])
        fb = self._fb
        fb["n_wind"] += info["n"]
        fb["dm_wind"] += info["dm"]
        fb["E_wind"] += info["E"]
        fb["dm_load"] = fb.get("dm_load", 0.0) + info["dm_created"]
        fb["dm_from_sinks"] = fb.get("dm_from_sinks", 0.0) + info["dm_from_sinks"]
        fb["n_load_capped"] = fb.get("n_load_capped", 0) + info["n_capped"]

    def feedback_report(self):
        """One line: the feedback Torch handed over so far and the gas's ionised fraction."""
        fb, u = self._fb, self.cfg.units
        t = float(self.state._s.t)
        if u is not None and u.is_enabled:
            t_s = f"t={u.to_cgs_time(t) / 3.15576e13:.4f} Myr"
            e = u.to_cgs_energy
        else:
            t_s, e = f"t={t:.5e}", (lambda v: v)
        line = (f"LUNA feedback: {t_s}  SN n={fb['n_sn']} E={e(fb['E_sn']):.3e}  "
                f"winds n={fb['n_wind']} dm={fb['dm_wind']:.3e} E={e(fb['E_wind']):.3e}  "
                f"ionising n_src={fb['n_src']} Q={fb['Q']:.3e}")
        if "dm_load" in fb:
            line += (f"  wind loading: {fb['dm_load']:.4e} created, {fb['dm_from_sinks']:.4e} "
                     f"from sinks ({fb['n_load_capped']} capped)")
        rad = self.state.sources.get("radiation")
        if isinstance(rad, tuple) and len(rad) == 3:
            if self._ion_fractions is None:
                import jax
                self._ion_fractions = jax.jit(lambda rho, x: (
                    jnp.sum(rho * x) / jnp.sum(rho), jnp.mean(x > 0.5)))
            xm, xv = self._ion_fractions(self.state._s.U[0], rad[1])
            line += f"  x_ion mass={float(xm):.3e} vol(x>0.5)={float(xv):.3e}"
        return line

    def _refresh_ionizing_sources(self):
        """Deposit the stars' parked photon rates (set_particle_nion) as M1's source field."""
        rad = self.state.sources.get("radiation")
        if not (isinstance(rad, tuple) and len(rad) == 3):
            return 0
        if self.stars.n_active == 0:
            # no stars: clear the star field, but keep one set by `set_ionizing_sources`
            if getattr(self, "_star_sources", None) is not False and \
                    not getattr(self, "_explicit_sources", False):
                self._star_sources = False
                self._fb["n_src"], self._fb["Q"] = 0, 0.0
                self.state.sources["radiation"] = (rad[0], rad[1], jax.jit(jnp.zeros_like)(rad[2]))
            return 0
        self._star_sources, self._explicit_sources = True, False
        # the whole store at fixed capacity, so the deposit compiles once
        a = self.stars.arrays
        rate = np.asarray(jax.device_get(a["nion"] * a["active"]))
        n_src = int((rate > 0.0).sum())
        self._fb["n_src"], self._fb["Q"] = n_src, float(rate.sum())
        R, x, _ = rad
        self.state.sources["radiation"] = (R, x, self._deposit(
            np.asarray(jax.device_get(a["position"])), rate))
        return n_src

    def _deposit(self, positions, rates):
        """Photon-rate field laid out like the ionised fraction, via a shard-local scatter."""
        from luna.radiation.ionization import deposit_updates
        x = self.state.sources["radiation"][1]
        if getattr(self, "_deposit_scatter", None) is None:
            from luna.utils.grid_scatter import make_grid_scatter
            self._deposit_scatter = make_grid_scatter(getattr(x, "sharding", None))
            self._deposit_zero = jax.jit(jnp.zeros_like)
        idx, vals = deposit_updates((self.Nx, self.Ny, self.Nz), (self.dx, self.dy, self.dz),
                                    positions, rates, boundary=int(self.cfg.BOUNDARY))
        field = self._deposit_zero(x)
        return self._deposit_scatter(field, idx, vals) if len(vals) else field

    def get_time(self):
        self._require_state()
        return self.state.t

    def get_timestep(self):
        """Current CFL timestep for the resident state."""
        self._require_state()
        return self.state.compute_dt()

    # -- particles (FLASH-style 'mass'/'sink' pointer) ----------------------
    def set_particle_pointers(self, part_type="mass"):
        """Select the particle set later calls address; crossing 'sink' syncs LUNA's sinks."""
        if part_type not in ("mass", "sink"):
            return -1
        if part_type == self._pointer:
            return 0                      # no seam to cross
        if part_type == "sink":
            self.sync_sinks_from_carry()
        elif self._pointer == "sink":
            self.sync_sinks_to_carry()
        self._pointer = part_type
        return 0

    def _store(self):
        return self.sinks if self._pointer == "sink" else self.stars

    def _state_mark(self):
        """(step, t) the sink mirror is valid for: only advancing the gas changes the sinks."""
        return None if self.state is None else (int(self.state.step),
                                                float(self.state.t))

    def new_particle(self, x, y, z):
        """Create particle(s) at the given position(s); returns their tags."""
        pos = np.stack([np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)], axis=1)
        tags = self._store().add(position=pos)
        if self._pointer == "mass":           # track new stars for add_particles_to_grav
            self._new_star_tags.extend(int(t) for t in np.atleast_1d(tags))
            # born now: Torch's stellar clock and the stars' age read this
            if self.state is not None:
                self._store().set("creation_time", tags,
                                  np.full(np.atleast_1d(tags).shape, float(self.state.t)))
        return tags

    def delete_particle(self, tag):
        self._store().remove(tag)
        return 0

    def get_number_of_particles(self):
        return self._store().n_active

    def get_particle_mass(self, tag):
        return self._store().get("mass", tag)

    def set_particle_mass(self, tag, mass):
        if self._pointer == "mass":           # what Torch takes off the stars (SeBa, SNe), for the budget line
            old = np.asarray(self._store().get("mass", tag), float)
            lost = np.where(old > 0.0, np.maximum(old - np.asarray(mass, float), 0.0), 0.0)
            self._fb["dm_star_loss"] = self._fb.get("dm_star_loss", 0.0) + float(np.sum(lost))
        self._store().set("mass", tag, mass)
        return 0

    def get_particle_position(self, tag):
        p = self._store().get("position", tag)
        return p[:, 0], p[:, 1], p[:, 2]

    def set_particle_position(self, tag, x, y, z):
        self._store().set("position", tag, np.stack(
            [np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)], axis=1))
        return 0

    def get_particle_velocity(self, tag):
        v = self._store().get("velocity", tag)
        return v[:, 0], v[:, 1], v[:, 2]

    def set_particle_velocity(self, tag, vx, vy, vz):
        self._store().set("velocity", tag, np.stack(
            [np.atleast_1d(vx), np.atleast_1d(vy), np.atleast_1d(vz)], axis=1))
        return 0

    def get_particle_acceleration(self, tag):
        a = self._store().get("acceleration", tag)
        return a[:, 0], a[:, 1], a[:, 2]

    # -- gravity bridge (gas-only field; particles feel no self-force) -------
    def _points(self, x, y, z):
        return jnp.asarray(np.stack(
            [np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)], axis=1))

    def _padded_points(self, x, y, z):
        """Points padded to a power of two (bounds jit retraces), and how many are real."""
        pts = np.asarray(self._points(x, y, z))
        n = pts.shape[0]
        n_pad = 1 << (n - 1).bit_length() if n else 1
        out = np.zeros((n_pad, 3))
        out[:n] = pts
        return jnp.asarray(out), n

    def get_potential_at_point(self, eps, x, y, z):
        """Gas gravitational potential at arbitrary point(s)."""
        self._require_state()
        k = self._bridge_kernels()
        pts, n = self._padded_points(x, y, z)
        return np.asarray(k[6](k[0](self.state.U), pts))[:n]

    def get_gravity_at_point(self, eps, x, y, z):
        """Gas gravitational acceleration at arbitrary point(s) -> (ax, ay, az)."""
        self._require_state()
        k = self._bridge_kernels()
        pts, n = self._padded_points(x, y, z)
        a = np.asarray(k[7](k[0](self.state.U), pts))[:n]
        return a[:, 0], a[:, 1], a[:, 2]

    # -- bridge kicks (Tier-1 gas<->particle coupling) ----------------------
    def _bridge_kernels(self):
        """Compile the bridge's programs once; fields are cached between kicks (apply_*)."""
        import jax

        if getattr(self, "_kick_kernels", None) is not None:
            return self._kick_kernels

        bridge = self.bridge

        @jax.jit
        def density(U):
            # a program, never an eager slice of a sharded array: that wedges ROCm
            return U[0]

        sh = getattr(self.state.U, "sharding", None)
        mesh = getattr(sh, "mesh", None)
        if mesh is not None and mesh.size > 1:
            # on a mesh, solve and sample per shard: XLA's partitioner would gather
            # the whole field onto every device
            from luna.gravity.particle_mesh_shardmap import ShardedParticleBridge
            sb = ShardedParticleBridge(self.cfg, mesh)
            gas_field = star_field = sb.accel_field
            gather = sb.accel_at_points
        else:
            @jax.jit
            def gas_field(rho):
                return bridge.gas_accel_grid(rho)

            @jax.jit
            def gather(field, pos):
                return bridge.accel_at_points(field, pos)

            @jax.jit
            def star_field(rho_star):
                return bridge.grad_phi(bridge.gas_potential(rho_star))

        @jax.jit
        def kick(U, g, dt):
            return bridge.kick_gas(U, g[0], g[1], g[2], dt)

        @jax.jit
        def same(a, b):
            return jnp.array_equal(a, b)

        @jax.jit
        def potential_at(rho, points):
            return bridge.sample_at_points(bridge.gas_potential(rho), points)

        @jax.jit
        def accel_at(rho, points):
            return bridge.gas_accel_on_particles(rho, points)

        self._kick_kernels = (density, gas_field, gather, star_field, kick, same,
                              potential_at, accel_at)
        self._gas_field = None           # (rho it was solved for, field)
        self._star_field = None          # ((pos, mass, active) it was for, field)
        return self._kick_kernels

    def _star_density(self, pos, mass, active):
        """The stars' CIC density on the grid, added by a shard-local scatter."""
        from luna.radiation.ionization import deposit_updates
        if getattr(self, "_star_scatter", None) is None:
            from jax.sharding import NamedSharding, PartitionSpec as P
            from luna.utils.grid_scatter import make_grid_scatter
            sh = getattr(self.state.U, "sharding", None)
            mesh = getattr(sh, "mesh", None)
            fsh = (NamedSharding(mesh, P(*tuple(sh.spec)[1:]))
                   if mesh is not None and mesh.size > 1 else None)
            self._star_scatter = make_grid_scatter(fsh)
            self._star_zero = jax.jit(jnp.zeros_like)
        b = self.bridge
        idx, vals = deposit_updates((b.Nx, b.Ny, b.Nz), (b.dx, b.dx, b.dx), pos,
                                    np.asarray(mass, float) * np.asarray(active, float),
                                    boundary=int(b.boundary))
        rho0 = self._star_zero(self._bridge_kernels()[0](self.state.U))
        return self._star_scatter(rho0, idx, vals) if len(vals) else rho0

    def apply_gas_gravity_on_particles(self, dt):
        """Gas -> particle half-kick for dt; the gas field is reused while rho is unchanged."""
        self._require_state()
        s = self._store()
        density, gas_field, gather, _, _, same = self._bridge_kernels()[:6]
        rho = density(self.state.U)
        if self._gas_field is None or not bool(same(rho, self._gas_field[0])):
            self._gas_field = (rho, gas_field(rho))
        a = np.asarray(gather(self._gas_field[1], jnp.asarray(s.arrays["position"])))
        sinks = self._star_sink_partners()
        if sinks is not None:                   # the sinks' pull on the stars
            _, sp, sm = sinks
            a = a + self._pair_accel(np.asarray(s.arrays["position"], float), sp, sm)
        active = s.arrays["active"].astype(a.dtype)[:, None]
        s.update_arrays({**s.arrays,
                         "velocity": s.arrays["velocity"] + dt * a * active,
                         "acceleration": a * active})
        return 0

    def apply_particle_gravity_on_gas(self, dt):
        """Particle -> gas half-kick for dt; the star field is reused while stars are unchanged."""
        self._require_state()
        s = self._store()
        _, _, _, star_field, kick, _ = self._bridge_kernels()[:6]
        key = tuple(np.array(s.arrays[k], copy=True) for k in ("position", "mass", "active"))
        hit = self._star_field is not None and all(
            np.array_equal(x, y) for x, y in zip(key, self._star_field[0]))
        if not hit:
            self._star_field = (key, star_field(self._star_density(*key)))
        self.state.U = kick(self.state.U, self._star_field[1], float(dt))
        sinks = self._star_sink_partners()
        if sinks is not None:                   # the stars' pull on the sinks
            part, sp, _ = sinks
            live = key[2].astype(bool)
            a = self._pair_accel(sp, key[0][live].astype(float), key[1][live].astype(float))
            self._kick_sinks(part, float(dt) * a)
        return 0

    # -- stars <-> LUNA's sinks ------------------------------------------------
    # Nothing else couples stars and sinks gravitationally: direct softened pairs, equal and
    # opposite, the sinks' pull in the gas->star kick, the stars' in the star->gas kick.
    def _star_sink_partners(self):
        """(sink part, live sink positions, masses) when the pairs apply, else None."""
        if not bool(getattr(self.cfg, "star_sink_gravity", True)) or self._pointer != "mass":
            return None
        part = self.state.species.get(SINK_SPECIES) if self.state is not None else None
        if part is None:
            return None
        pos, mass, active = (np.asarray(x, float) for x in jax.device_get(
            (part.pos, part.mass, part.active)))
        live = active > 0
        if not live.any():
            return None
        return part, pos[live], mass[live]

    def _pair_accel(self, pa, pb, mb):
        """Acceleration at pa from softened point masses mb at pb, summed over b."""
        out = np.zeros_like(pa)
        if pa.shape[0] == 0 or pb.shape[0] == 0:
            return out
        G = float(self.cfg.GRAVITATIONAL_G)
        eps = float(getattr(self.cfg, "sink_softening", 0.0) or 0.0) or 2.0 * float(self.dx)
        for i in range(0, pa.shape[0], 4096):        # bounded (4096, n_sinks, 3) scratch
            d = self._nearest(pb[None, :, :] - pa[i:i + 4096, None, :])
            w = G * mb[None, :] * (np.sum(d * d, axis=-1) + eps * eps) ** -1.5
            out[i:i + 4096] = np.einsum("ij,ijk->ik", w, d)
        return out

    # -- the box faces for particles ----------------------------------------
    # Periodic particles (`particle_boundary`) wrap and are never removed; outflow
    # stars are removed by Torch at the faces.
    def _periodic_particles(self):
        from luna.species.sinks import particles_periodic
        return particles_periodic(self.cfg)

    def _nearest(self, d):
        """Nearest periodic image of separations d (..., 3), when periodic."""
        if not self._periodic_particles():
            return d
        L = np.array([self.cfg.Lx, self.cfg.Ly, self.cfg.Lz], float)
        return d - L * np.floor(d / L + 0.5)

    def _kick_sinks(self, part, dv_live):
        """Add dv to the live sinks' velocities in the carry, keeping its layout."""
        from dataclasses import replace as _dc_replace
        live = np.asarray(jax.device_get(part.active), float) > 0
        dv = np.zeros((live.size, 3))
        dv[live] = dv_live
        p = part.p
        new_p = jax.jit(lambda q, d: q + d, out_shardings=getattr(p, "sharding", None))(
            p, jnp.asarray(dv, dtype=p.dtype))
        self.state.species[SINK_SPECIES] = _dc_replace(part, p=new_p)

    # -- LUNA-internal sinks (formed by sources/sinks.py) exposed to Torch --
    # LUNA owns sink creation/accretion; Torch reads these to form stars.
    def _sink_carry(self):
        """(pos, vel, mass, active[, extra moments]) for LUNA's sinks, or None."""
        if self.state is None:
            return None
        part = self.state.species.get(SINK_SPECIES)
        if part is None:
            return None
        if part.extra is None:
            return (part.pos, part.p, part.mass, part.active)
        return (part.pos, part.p, part.mass, part.active, part.extra)

    def get_number_of_sinks(self):
        c = self._sink_carry()
        return 0 if c is None else int(np.asarray(c[3]).sum())

    def get_active_sinks(self):
        """(slot_indices, pos[n,3], vel[n,3], mass[n]) for LUNA's active sinks."""
        c = self._sink_carry()
        if c is None:
            z = np.empty((0, 3))
            return np.empty(0, np.int64), z, z, np.empty(0)
        # one transfer for the four arrays (carry = (pos, vel, mass, active, mom))
        pos, vel, mass, active = (np.asarray(v) for v in jax.device_get(tuple(c[k] for k in range(4))))
        sel = np.where(active > 0.5)[0]
        return sel.astype(np.int64), pos[sel], vel[sel], mass[sel]

    def get_sink_gas_moments(self, exact=True, r_acc=None):
        """Accreted-gas moments per active sink for Torch; `exact=False` uses the local gas."""
        c = self._sink_carry()
        if c is None:
            return dict(mean_cs=np.zeros(0), mean_velocity=np.zeros((0, 3)),
                        var_velocity=np.zeros((0, 3)), angular_momentum=np.zeros((0, 3)))
        from luna.species.sinks import accreted_gas_moments, sink_gas_moments
        sel, spos, _v, _m = self.get_active_sinks()
        local = None
        if not (exact and len(c) >= 5):
            return sink_gas_moments(self.cfg, self.state.U, spos,
                                    gamma=self.gamma, r_acc=r_acc)

        moments = accreted_gas_moments(c)
        # A sink that accreted nothing yet has empty moments (creation bypasses the
        # accumulators); a zero mean_cs makes its stars co-moving, so use the local gas.
        macc = np.atleast_1d(np.asarray(c[4]["macc"]))[sel]
        empty = macc <= 0.0
        if np.any(empty):
            local = sink_gas_moments(self.cfg, self.state.U, spos,
                                     gamma=self.gamma, r_acc=r_acc)
            for key, value in moments.items():
                mask = empty if value.ndim == 1 else empty[:, None]
                moments[key] = np.where(mask, local[key], value)
        return moments

    def set_ionizing_sources(self, positions, rates, active=None):
        """Deposit ionizing stars (positions, rates per code time, active) into the M1 carry."""
        rad = self.state.sources.get("radiation")
        if rad is None:
            raise RuntimeError("radiation is not enabled in this run "
                               "(cfg.radiation_enabled)")
        if not (isinstance(rad, tuple) and len(rad) == 3):
            raise RuntimeError(
                "the radiation carry is not the ionization one -- "
                "cfg.rad_ionization must be on for star sources")
        R, x, _ = rad
        rates = np.asarray(rates, float)
        if active is not None:
            rates = rates * np.asarray(active, float)
        self.state.sources["radiation"] = (R, x, self._deposit(np.asarray(positions, float), rates))
        self._explicit_sources = True
        return len(np.atleast_1d(rates))

    def get_ionizing_photon_rate(self):
        """Total photon rate currently injected, integrated over the box."""
        rad = self.state.sources.get("radiation")
        if rad is None or not (isinstance(rad, tuple) and len(rad) == 3):
            return 0.0
        return float(jnp.sum(rad[2])) * self.dx * self.dy * self.dz

    def set_sink_mass(self, index, mass):
        """Set LUNA sink `index`'s mass, deactivating it at ~0."""
        from dataclasses import replace as _dc_replace
        part = self.state.species.get(SINK_SPECIES) if self.state is not None else None
        if part is None:
            return -1
        m = float(mass)
        new_mass = jnp.asarray(part.mass).at[int(index)].set(m)
        new_active = part.active
        if m <= 1e-12:
            new_active = jnp.asarray(part.active).at[int(index)].set(0.0)
        self.state.species[SINK_SPECIES] = _dc_replace(
            part, mass=new_mass, active=new_active)
        return 0

    # -- Torch/torch_sf particle contract -----------------------------------
    # What torch_sf calls on the hydro code; LUNA's sinks are mirrored into the 'sink' store.
    def get_particle_tags(self, indices=None):
        """Active particle tags under the current pointer (torch_sf's index range is ignored)."""
        return list(int(t) for t in np.atleast_1d(np.asarray(self._store().active_tags())))

    def add_particles(self, x, y, z):
        """torch_sf's name for new_particle (create stars; returns tags)."""
        return self.new_particle(x, y, z)

    def remove_particles(self, tags):
        self._store().remove(tags)
        return 0

    def get_particle_oldmass(self, tag):
        return self._store().get("oldmass", tag)

    def set_particle_oldmass(self, tag, mass):
        self._store().set("oldmass", tag, mass)
        return 0

    def get_sink_state(self, tag):
        """Mass, position, velocity and mean_cs of sinks `tag` in one call."""
        if self._pointer != "sink":
            self.set_particle_pointers("sink")
        s = self.sinks
        p, v = s.get("position", tag), s.get("velocity", tag)
        return (s.get("mass", tag), p[:, 0], p[:, 1], p[:, 2],
                v[:, 0], v[:, 1], v[:, 2], s.get("mean_cs", tag))

    def form_stars(self, sink_rad, min_imf_mass, max_imf_mass, sample_imf_mass,
                   sample_imf_bins=10, sum_small=False, m_small=1.0):
        """Torch's star formation (queue_stars + make_stars_from_sinks) here; returns the count."""
        from luna.species.imf import sample_stellar_mass
        rng = self._sf_rng
        self.set_particle_pointers("sink")
        tags = self.get_particle_tags()
        if not tags:
            self.set_particle_pointers("mass")
            return 0
        tags = np.array(sorted(tags))
        m, x, y, z, vx, vy, vz, cs = (np.asarray(a) for a in self.get_sink_state(tags))
        new_pos, new_vel, new_mass = [], [], []
        for i, tag in enumerate(tags):
            q = self._sf_queues.get(int(tag), np.zeros(0))
            while q.sum() <= m[i]:
                q = np.concatenate([q, sample_stellar_mass(
                    rng, float(sample_imf_mass), num_bins=int(sample_imf_bins),
                    min_samp_mass=float(min_imf_mass), max_samp_mass=float(max_imf_mass),
                    sum_small=bool(sum_small), m_small=float(m_small))])
            self._sf_queues[int(tag)] = q
            n = int(np.searchsorted(np.cumsum(q), m[i], side="left"))
            if n == 0 or not np.isfinite(cs[i]):
                continue
            spawn, self._sf_queues[int(tag)] = q[:n], q[n:]
            self.set_particle_mass(tag, m[i] - spawn.sum())
            phi = rng.uniform(0.0, 2.0 * np.pi, n)
            cth = rng.uniform(-1.0, 1.0, n)
            sth = np.sqrt(1.0 - cth * cth)
            d = np.stack([sth * np.cos(phi), sth * np.sin(phi), cth], axis=1)
            new_pos.append(np.array([x[i], y[i], z[i]]) + float(sink_rad) * rng.random((n, 1)) * d)
            new_vel.append(np.array([vx[i], vy[i], vz[i]]) + rng.normal(scale=cs[i], size=(n, 3)))
            new_mass.append(spawn)
        self.set_particle_pointers("mass")
        if not new_mass:
            return 0
        pos, vel, mass = (np.concatenate(a) for a in (new_pos, new_vel, new_mass))
        t = self.new_particle(pos[:, 0], pos[:, 1], pos[:, 2])
        self.set_particle_mass(t, mass)
        self.set_particle_velocity(t, vel[:, 0], vel[:, 1], vel[:, 2])
        self.set_particle_oldmass(t, mass)
        return int(len(mass))

    def get_sink_mean_cs(self, tag):
        """Mean sound speed of the gas sink `tag` accreted (set by sync_sinks_from_carry)."""
        return self.sinks.get("mean_cs", tag)

    def sync_sinks_from_carry(self):
        """Mirror LUNA's active sinks into the 'sink' store, once per step; tag = carry slot."""
        if self._sink_sync_at == self._state_mark():
            return [int(t) for t in np.atleast_1d(
                np.asarray(self.sinks.active_tags()))]
        from amuse_luna.particles import ParticleStore, default_sink_schema
        sel, pos, vel, mass = self.get_active_sinks()
        mom = self.get_sink_gas_moments()
        cap = int(getattr(self.sinks, "capacity", self.cfg.sink_max_count))
        schema = default_sink_schema()
        if set(getattr(self.sinks, "schema", {})) != set(schema) or self.sinks.capacity != cap:
            self.sinks = ParticleStore(cap, schema, "sinks")
        self._sink_tag_to_slot = {}
        self._sink_sync_at = self._state_mark()
        n = len(sel)
        # Built on the host, uploaded in one go. A sink's tag is its carry slot, so
        # Torch's per-tag IMF queue stays with its sink.
        arrays = {k: np.zeros((cap,) + tuple(shape), dtype=np.dtype(dt))
                  for k, (shape, dt) in schema.items()}
        arrays["active"] = np.zeros(cap, dtype=bool)
        if n:
            arrays["active"][:n] = True
            arrays["position"][:n] = np.asarray(pos)
            arrays["velocity"][:n] = np.asarray(vel)
            arrays["mass"][:n] = np.asarray(mass)
            cs = np.asarray(mom.get("mean_cs", np.zeros(n)), float)
            arrays["mean_cs"][:n] = np.where(self._cold_gas_near(pos), cs, np.nan)
        tags = np.asarray(sel, dtype=np.int64)
        self.sinks.load({"arrays": arrays,
                         "tag_to_slot": {int(t): i for i, t in enumerate(tags)},
                         "next_tag": int(tags.max()) + 1 if n else 0})
        if n == 0:
            return []
        for t, slot in zip(tags, sel):
            self._sink_tag_to_slot[int(t)] = int(slot)
        # what the store was given, so the push back can tell whether Torch
        # changed anything (it does only when a sink spawns stars)
        self._sink_mirror_mass = {int(t): float(m) for t, m in zip(tags, np.atleast_1d(mass))}
        return [int(t) for t in tags]

    def _cold_gas_near(self, pos):
        """Per sink: may it form stars (cold gas within 2 r_acc; docs/TORCH.md)?"""
        pos = np.atleast_2d(np.asarray(pos, float))
        T_max = float(getattr(self.cfg, "sink_star_T_max", 0.0))
        if T_max <= 0.0 or len(pos) == 0:
            return np.ones(len(pos), bool)
        from luna.species.sinks import sink_gas_moments
        from luna.units import BOLTZMANN_CGS, PROTON_MASS_CGS
        cfg = self.cfg
        cs2 = (BOLTZMANN_CGS * T_max / (float(cfg.sink_star_mu) * PROTON_MASS_CGS)
               / float(cfg.units.UnitVelocity_in_cm_per_s) ** 2)
        r_acc = float(getattr(cfg, "sink_accretion_radius", 0.0)) or 2.0 * cfg.Lx / cfg.Nx
        cold = sink_gas_moments(cfg, self.state.U, pos, gamma=self.gamma,
                                r_acc=2.0 * r_acc, cs2_max=cs2)["mass"]
        return cold > 0.0

    def sync_sinks_to_carry(self):
        """Push sink masses changed in the 'sink' store back to the carry."""
        pairs = getattr(self, "_sink_tag_to_slot", {})
        part = self.state.species.get(SINK_SPECIES) if self.state is not None else None
        if not pairs or part is None:
            return 0
        from dataclasses import replace as _dc_replace
        live = set(int(t) for t in np.atleast_1d(np.asarray(self.sinks.active_tags())))
        tags = [t for t in pairs if t in live]
        got = dict(zip(tags, np.atleast_1d(np.asarray(self.sinks.get("mass", tags))))) if tags else {}
        m_host = np.array([float(got.get(t, 0.0)) for t in pairs])
        mirror = getattr(self, "_sink_mirror_mass", None)
        if mirror is not None and set(pairs) == set(mirror) and all(
                m_host[i] == mirror[t] for i, t in enumerate(pairs)):
            return 0            # Torch changed no sink: the carry already holds this
        slots = np.array(list(pairs.values()), dtype=np.int64)
        m = jnp.asarray(m_host)
        act = jnp.asarray(part.active)
        self.state.species[SINK_SPECIES] = _dc_replace(
            part, mass=jnp.asarray(part.mass).at[slots].set(m),
            active=act.at[slots].set(jnp.where(m <= 1e-12, 0.0, act[slots])))
        self._sink_mirror_mass = {int(t): float(v) for t, v in zip(pairs, m_host)}
        return 0

    # -- add_particles_to_grav contract (PeTar/SeBa handoff) ------------------
    # Newborn stars keep defaults: stype=0 and radius=0 make Torch set ZAMS values.
    def get_number_of_new_tags(self):
        return len(getattr(self, "_new_star_tags", []))

    def get_new_tags(self, rng=None):
        return list(getattr(self, "_new_star_tags", []))

    def clear_new_tags(self):
        self._new_star_tags = []
        return 0

    def set_starting_local_tag_numbers(self):
        return 0   # single-process: tags are globally unique already

    def get_particle_rel_mass(self, tags):
        """SeBa's relative (ZAMS) mass; 0 (not yet set by SeBa) falls back to the mass."""
        rel = self.get_particle_attr("rel_mass", tags)
        return np.where(rel > 0.0, rel, self.get_particle_attr("mass", tags))

    def get_runtime_parameter(self, name):
        """FLASH box bounds in cm that Torch reads; unbounded with periodic particles."""
        bounds = {"xmin": 0.0, "ymin": 0.0, "zmin": 0.0,
                  "xmax": self.cfg.Lx, "ymax": self.cfg.Ly, "zmax": self.cfg.Lz}
        v = bounds.get(name)
        if v is None:
            return 0.0
        if self._periodic_particles():
            return float("-inf") if name.endswith("min") else float("inf")
        u = self.cfg.units
        return float(v) * (u.UnitLength_in_cm if getattr(u, "is_enabled", False) else 1.0)

    # -- Torch's stellar bookkeeping ----------------------------------------
    # torch_se/torch_sf park SeBa state and emission properties here; LUNA only stores them.
    TORCH_STAR_ATTRS = tuple(PARKED_ATTRS) + ("mass",)

    def get_particle_attr(self, name, tags):
        return np.atleast_1d(np.asarray(self._store().get(name, tags)))

    def set_particle_attr(self, name, tags, values):
        self._store().set(name, tags, np.asarray(values))
        return 0

    # -- run bookkeeping Torch's loop reads ---------------------------------
    def get_current_step(self):
        self._require_state()
        return int(self.state.step)

    def get_end_time(self):
        return float(self.cfg.t_end)

    def get_max_num_steps(self):
        """Torch's step limit; LUNA's 0 (no limit) becomes a large number."""
        n = int(getattr(self.cfg, "max_steps", 0))
        return n if n > 0 else 2**31 - 1

    def get_output_dir(self):
        """Where Torch writes its own files: the run's output directory."""
        return str(getattr(self.cfg, "output_dir", "snapshots"))

    def particles_sort(self):
        """FLASH's particle sort: here only wraps (periodic) or removes stars outside the box."""
        s = self._store()
        tags = s.active_tags()
        if len(tags) == 0:
            return 0
        L = np.array([self.cfg.Lx, self.cfg.Ly, self.cfg.Lz])
        pos = np.asarray(s.get("position", tags))
        if self._periodic_particles():
            s.set("position", tags, np.mod(pos, L))
            return 0
        outside = (pos < 0.0).any(axis=1) | (pos >= L).any(axis=1)    # in the box: [0, L), as sinks
        if outside.any():
            s.remove([int(t) for t, o in zip(tags, outside) if o])
        return 0

    def timer_summary(self):
        return 0

    def _io_number(self):
        """Torch's output tick: `time_bet_snaps` intervals elapsed (0 with snapshots off)."""
        self._require_state()
        dt_snap = float(self.cfg.time_bet_snaps or 0.0)
        if not self.cfg.write_snapshots or dt_snap <= 0.0:
            return 0
        return int(self.state.t / dt_snap)

    def IO_num(self, kind="chk"):
        """The number the next write will carry (FLASH's convention)."""
        self._require_state()
        if kind == "chk":
            return self._chk_written + 1
        return self._io_number() + 1

    def IO_out(self, kind="chk", num=None):
        """Write a coupled checkpoint when due; return the NEXT number, as FLASH does."""
        self._require_state()
        if kind != "chk":
            return self._io_number() + 1
        tick = self._io_number()
        if not self.cfg.write_snapshots:
            return 0
        # a new tick, or the wall-clock interval: with tiny steps ticks can be a day apart
        import time
        now = time.time()
        if self._chk_wall is None:
            self._chk_wall = now
        wall = float(self.cfg.checkpoint_wall_time_interval or 0.0)
        from luna.file_io.checkpoint_sets import due_everywhere
        # rank 0's clock decides for every rank: the write is collective, and
        # ranks that each read their own clock can straddle the deadline
        due = tick > self._chk_tick or due_everywhere(
            getattr(self.cfg, "enable_checkpointing", True) and wall > 0.0
            and now - self._chk_wall >= wall)
        if due:
            self._write_coupled(self._chk_written + 1, tick)
            self._chk_written += 1
            self._chk_tick = max(tick, self._chk_tick)
            self._chk_wall = now
        return self._chk_written + 1

    def write_chpt(self):
        """A checkpoint now, whatever the tick: the number after the last."""
        self._require_state()
        self._write_coupled(self._chk_written + 1, self._io_number())
        self._chk_written += 1
        return 0

    # -- Torch's names for the bridge kicks ---------------------------------
    # Torch's names for apply_*; kick_number is only for its logging.
    def get_gravity_gas_on_particles(self, dt, kick_number=0):
        return self.apply_gas_gravity_on_particles(dt)

    def get_gravity_particles_on_gas(self, dt, kick_number=0):
        return self.apply_particle_gravity_on_gas(dt)

    # -- feedback application (SN/wind) — glue to LUNA physics --------------
    def energy_injection(self, energy, frac_kin, mass, x, y, z, radius=None):
        """Deposit SN energy (+ ejecta mass) at (x,y,z) via `sources.feedback.inject_energy`."""
        self._require_state()
        from luna.sources.feedback import inject_energy
        self._fb["n_sn"] += 1
        self._fb["E_sn"] += float(energy)
        self._fb["dm_sn"] = self._fb.get("dm_sn", 0.0) + float(mass)
        self.state.U = inject_energy(self.state.U, self.cfg, float(x), float(y), float(z),
                                     float(energy), mass=float(mass), radius=radius,
                                     f_kin=float(frac_kin))
        return 0

    def wind_injection(self, dm, v_term, x, y, z, vx=0.0, vy=0.0, vz=0.0, radius=None):
        """Inject wind ejecta (dm at v_term, + star velocity) via `sources.feedback.inject_wind`."""
        self._require_state()
        from luna.sources.feedback import inject_wind
        self._fb["n_wind"] += 1
        self._fb["dm_wind"] += float(dm)
        self._fb["E_wind"] += 0.5 * float(dm) * float(v_term) ** 2
        self.state.U = inject_wind(self.state.U, self.cfg, float(x), float(y), float(z),
                                   float(dm), float(v_term), radius=radius,
                                   v_star=(float(vx), float(vy), float(vz)),
                                   add_heat=bool(getattr(self.cfg, "wind_add_heat", False)))
        return 0


def _bind_parked_attrs(cls):
    """Bind `get_/set_particle_<name>` for every parked attribute not already defined."""
    for attr in cls.TORCH_STAR_ATTRS:
        def getter(self, tags, _a=attr):
            return self.get_particle_attr(_a, tags)

        def setter(self, tags, values, _a=attr):
            return self.set_particle_attr(_a, tags, values)

        for fn, verb in ((getter, "get"), (setter, "set")):
            fn.__name__ = "{}_particle_{}".format(verb, attr)
            if not hasattr(cls, fn.__name__):
                setattr(cls, fn.__name__, fn)
    return cls


_bind_parked_attrs(LunaWorker)
