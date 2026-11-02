# Time integration for a pure-PM differentiable cosmology code — survey and recommendations

Companion to `PRECISION_SURVEY.md`. Split out because the subject is time integration, not
precision. Status: research note, not committed policy. Written 2026-08-06.

Same tagging convention: **[V-src]** verified in source (file:line), **[V-paper]** verified in the
paper text, **[INF]** inference by the author of this note, **[unverified]** could not confirm.

---

## 1. The problem

LUNA is **pure PM** with a **global timestep**: the most-accelerated particle sets `dt` for all 134M
particles. Measured: **~7000 steps to z = 0** at 512³, with `dloga` falling to 2.8e-4 in the
clustered regime while the expansion cap alone would allow 0.02 — a **~70× oversampling** for the
vast majority of particles.

The criterion, `setup/cosmo_common.py:60-66` **[V-src]**:

```python
def dtau_accel(eta, eps_soft, gmax):
    """Gravitational acceleration timestep: dtau ~ eta * sqrt(eps / |g|_max)."""
    return eta * np.sqrt(eps_soft / (gmax + 1e-30))
```

with `dm_eta = 0.025` (`config.py:102`, `template_config.py:104`) and `dloga_max = 0.01`
(`config.py:93`) **[V-src]**.

---

## 2. GADGET-4's integrator, from the source

Source: depth-1 clone of `https://gitlab.mpcdf.mpg.de/vrs/gadget4.git`.

### 2.1 Integer time — the same philosophy as integer positions

`src/data/dtypes.h:116-121` **[V-src]**:

```c
#ifdef ENLARGE_DYNAMIC_RANGE_IN_TIME
typedef long long integertime;
#define TIMEBINS 60
#else
typedef int integertime;
#define TIMEBINS 29
#endif
#define TIMEBASE (((integertime)1) << TIMEBINS)
```

The whole integration span `[log a_begin, log a_end]` is divided into `TIMEBASE = 2^TIMEBINS` integer
units, with `All.Timebase_interval = (log(a_end) − log(a_begin)) / TIMEBASE`. Every timestep is
snapped to a **power-of-two** multiple of that unit, so a particle's time coordinate is an exact
integer and synchronisation points are exact. This is the same design idea as the integer positions
(`PRECISION_SURVEY.md` §3): make the coordinate exact and derive the float quantity from it, rather
than accumulating float increments.

**Directly adoptable by LUNA, cheaply** — see R4 below.

### 2.2 The PM force is *not* sub-cycled

The key structural fact for LUNA. In GADGET-4's TreePM, the **long-range PM force is computed
globally on the largest (PM) timestep**, while only the short-range tree/FMM force is sub-cycled on
the power-of-two hierarchy. That separation is what makes hierarchical stepping viable at all — and
it is exactly the thing a pure-PM code does not have. **[V-paper, arXiv:2010.03567]**

### 2.3 Their criterion

GADGET-4's acceleration criterion is `dt = sqrt(2 η ε / |a|)` with `η = ErrTolIntAccuracy` — note
**η is inside the square root**, whereas LUNA's `dtau_accel` has it outside. This matters (§4.1).

---

## 3. Can a pure-PM code do individual/hierarchical timesteps? No — and this is now settled

**Quinn, Katz, Stadel & Lake 1997** (arXiv:astro-ph/9710043) answer this directly **[V-paper §5]**:

> "for a standard particle mesh code, the forces at every grid point are either calculated or not.
> The only part of the force evaluation that is not part of the fixed cost would be the trivial
> interpolation of the force onto particles. So, essentially all the costs are fixed and **no gain
> is possible**."

Their cost model makes it quantitative: with fixed-cost fraction `f = C_t/(N·C_f)`, the maximum
speedup from multistepping is `(f+1)/2f`. For pure PM the per-particle cost `C_f → 0` (only the CIC
gather), so `f → ∞` and **the speedup ceiling is 1/2 — multistepping in pure PM is a net loss.**

Nor does the symplectic-individual-timestep construction port. **Saha & Tremaine 1994**
(astro-ph/9403057) keep symplecticity only because their per-particle steps are **fixed, never
adaptive**, and because the construction relies on a *pairwise, particle-labelled* Hamiltonian
decomposition. A PM force is a single indivisible global field operator with no such ordering.
Their measured payoff for the solar system was *"a little over a factor of two"* despite a >100×
spread in step sizes. **[V-paper]**

Related correction: the published **ART** paper has **no** level-dependent timesteps
(astro-ph/9701195, §3.5: *"in the current version of the code we use a constant time step for all
particles"*) **[V-paper]**. And RAMSES's AMR Poisson uses a *"one-way interface"* in which
*"the coarse solution is assumed to remain constant in time during the advance of the fine
solution"* — the coarse contribution is frozen and the inconsistency absorbed as truncation error
**[V-paper, astro-ph/0111367]**.

**Conclusion: do not pursue hierarchical timestepping for LUNA.** The saving has to come from
taking *fewer, larger* global steps, accurately.

---

## 4. Where the steps actually go

### 4.1 The η coefficient looks 9–12× too conservative

LUNA's `dtau = η√(ε/|g|)` **is exactly the criterion Quinn et al. advocate**, and they state the
coefficient **[V-paper]**:

> "we advocate either the SDKD or SKDK method using η√(ε/a) and/or … η(ε/v) to choose the timestep…
> an appropriate choice for **η is 0.3** (for Plummer softening 0.4)"

**LUNA uses η = 0.025 where the criterion's own authors recommend 0.3 — 12× more conservative.**
**[V-src for LUNA's value; V-paper for the recommendation.]**

A second, independent line points the same way: GADGET-4 puts η *inside* the square root
(`dt = sqrt(2ηε/|a|)`), so a value copied across from `ErrTolIntAccuracy` without adjusting for that
would be too small by roughly `sqrt(2·0.025)/0.025 ≈ 9×`. **[INF — a plausible provenance for the
0.025, not established.]**

**Caveat that must be respected:** LUNA integrates in super-comoving time τ, not `t`, so the mapping
from the Quinn et al. `dt` to LUNA's `dtau` is not literally identity. **Verify empirically with a
convergence test; do not simply set η = 0.3.** This is nonetheless the cheapest experiment available.

### 4.2 LUNA is using the wrong *kind* of criterion for a PM code

The acceleration/softening criterion is inherited from tree-code practice, where the softening
length is the resolution scale. In a **pure PM code the resolution scale is the mesh cell**, and the
binding constraint is a **displacement (CFL-like) criterion**: particles should move less than a
fraction of a cell per step.

**Klypin & Prada 2018** (arXiv:1701.05690, the PPM-GLAM code paper) Appendix B is the convergence
study **[V-paper]**. Setup: L = 1 h⁻¹Gpc, 1000³ particles, 3000³ mesh, z_init = 100,
N_s ∈ {34, 68, 147, 302}.

- **Convergence sets in at N_s ≳ 100**; little difference between 147 and 302.
- **P(k) is a misleadingly forgiving diagnostic**: *"even 34 time-steps give an error in P(k) of only
  3% at k ~ 0.3 h Mpc⁻¹."* The damage appears in the **density PDF at δ > 1000**: *"a smaller number
  of steps results in a dramatic suppression of the number of high-density regions where DM
  particles move very fast, which is observed as an artificial scale-dependent bias."*
- **Their step-count formula (eq. 17)**, derived from requiring particles move **< ⅓ of a mesh cell
  per step** at cluster velocities (~2000 km/s):

  ```
  N_s ≈ 60 · N_g / L        (L in h⁻¹Mpc)
  ```

- They explicitly reject the aggressive fast-PM regime: *"FastPM simulations with 40 time-steps and
  force resolution of Δx = 0.2 h⁻¹Mpc do not satisfy these conditions and would require 2–2.5 times
  more time-steps."*

**LUNA already computes the displacement bound** — `setup/runner_cosmo_dm.py:365` **[V-src]**:

```python
step_drift = (np.sqrt(p2max) + 0.5*np.sqrt(g2max)*dtau)*dtau
```

— but uses it only to decide when to trigger a halo exchange. Promoting it to *the* timestep
criterion is a small change to code that already exists.

Plugging LUNA's configuration into eq. 17: at N_g = 512 and L = 100 h⁻¹Mpc, **N_s ≈ 307**; at
L = 500 h⁻¹Mpc, N_s ≈ 61. **That is the defensible target — roughly 20× fewer than 7000, and citable
to a PM code paper rather than to a fast-mock paper.**

---

## 5. Integrators that let pure PM take big steps accurately

All of these keep a **global** step; they change the kick/drift *coefficients* so that large steps
remain accurate. This is the right family for LUNA.

### 5.1 The unifying form

DISCO-DJ II (List, Hahn, Flöss & Winkler 2025, arXiv:2510.05206) puts all the second-order DKD
schemes in one frame **[V-paper]**:

```
X^{n+1/2} = X^n + τ₁ V^n
V^{n+1}   = α V^n + β A(X^{n+1/2})
X^{n+1}   = X^{n+1/2} + τ₂ V^{n+1}

Zel'dovich consistency  ⟺  β = (1 − α)·D_{n+1/2}
```

Standard KDK is the α = 1 case. The fast-PM family chooses α, β so that the scheme reproduces the
exact linear growth (Zel'dovich) solution at *any* step size.

### 5.2 FastPM

**Feng, Chu, Seljak & McDonald 2016** (arXiv:1603.00476). Modified kick and drift factors enforcing
the Zel'dovich consistency condition, so linear growth is exact regardless of `dt`.
In the DISCO-DJ II parametrization, `α = ζ(a(D_n))/ζ(a(D_{n+1}))` with `ζ(a) = H a³ D′(a)`.

**DISCO-DJ II Prop. 5: FastPM is the *unique* integrator that is both Zel'dovich-consistent and
symplectic.** **[V-paper]**

This is the best fit for LUNA: it reduces to the current scheme in the small-step limit, it is a
**two-coefficient swap** in an existing KDK loop, and it is symplectic.

### 5.3 BullFrog

**Rampf, List & Hahn 2024** (arXiv:2409.19049). The unique (α, β) that matches **2LPT after every
step**. Not symplectic — the authors argue symplecticity is moot in cosmology since energy drains
continuously via Hubble drag. **Zero computational overhead** vs FastPM (only extra growth-function
ODE integration, negligible against the FFT/solve). A single BullFrog step already beats 2LPT.

Head-to-head (DISCO-DJ II §3.2, 512³ particles, 1024³ mesh, z_init = 50, vs GADGET-4) **[V-paper]**:

| target P(k) accuracy | k = 0.1 | k = 0.3 | k = 0.5 h/Mpc |
|---|---|---|---|
| < 5% | 2 steps | 5 steps | 12 steps |
| < 1% | 4 steps | 19 steps | not reached in 100 |

**FastPM wins for ≤ 2 steps; BullFrog overtakes from ≳ 5 steps; they agree at 100 steps.** Since
LUNA will operate at 100–300 steps (§4.2), **both are fine and the choice is on implementation
grounds — which favours FastPM.**

### 5.4 COLA — not recommended for LUNA

**Tassev, Zaldarriaga & Eisenstein 2013** (arXiv:1301.0322) integrate the residual from the LPT
trajectory, so the residual is smooth and large steps stay accurate. Successors: ICE-COLA
(arXiv:1509.04685), sCOLA.

Why it is a poor fit here: it requires storing per-particle LPT displacements `s₁, s₂` (+24
B/particle at 134M = 3.2 GB), MUSIC ICs do not supply them, it introduces a hand-fitted `n_LPT`
exponent, and **FastPM outperforms it at low step counts**. **[INF for the fit assessment; V-paper
for the method.]**

### 5.5 Higher-order symplectic — verified negative

**Do not pursue.** Two independent reasons:

- **List & Hahn 2023** (arXiv:2301.09655) state *"higher-order integrators have not yet been employed
  in cosmological simulations"*, and their **Proposition 3** gives the reason: after shell-crossing
  the acceleration field's derivatives are singular, so **convergence is limited to order 3/2 for
  planar-wave collapse**, making *"the use of high-order integrators in this regime futile."*
  **[V-paper]**
- **Quinn et al. 1997**: *"The force field in an N-body simulation is not very smooth, so higher
  order does not necessarily mean higher accuracy"* — their Fig. 1 shows 2nd-order leapfrog beating
  RK4 at 4× the force cost. **[V-paper]**

And the economics are hostile anyway: in a pure-PM code each extra force evaluation is a **full
Poisson solve**, so 4th-order Yoshida costs 3× per step. Where 4th-order symplectic *is* used in
astrophysics it is always collisional/direct-summation (e.g. FROST, arXiv:2011.14984).

### 5.6 Force-kernel corrections — the differentiability payoff

A third lever that compensates for coarse stepping rather than refining it:

- **PGD** — Dai, Feng & Seljak 2018 (arXiv:1804.00671): a post-processing displacement
  `S = (α_PM/H₀²)∇ Ô_l Ô_s φ` (band-passed potential gradient) that sharpens halo profiles flattened
  by coarse stepping; three parameters calibrated against high-res references. **10-step FastPM vs
  Illustris: P(k) within 5% to k < 10 h/Mpc; cross-correlation 80% → 95%.** **[V-paper]**
- **Hybrid Physical-Neural ODEs** — Lanzieri, Lanusse & Starck 2022 (arXiv:2207.05509): the learned
  version, a Fourier-space NN correction to the PM potential trained **end-to-end through the
  differentiable PM solver**. Outperforms PGD on cross-correlation and is more robust across
  resolution/cosmology. **[V-paper]**

Note carefully where the ML effort in this field has gone: into learning the **force kernel**, not
the time steps. pmwd promised "spatiotemporal optimization of the PM solver" (Zhang et al., in prep.)
and it was **never posted**. DISCO-DJ II exposes step times as differentiable inputs *"which could be
used e.g. to find an optimal time step distribution via gradient-based optimisation"* — offered as a
capability, with no result. **"Learn the time steps" is an open problem, not an available
technique.** **[V-paper]**

---

## 6. Autodiff interactions

LUNA is differentiable (reverse-mode AD through the time loop), which constrains the choice.

- **Nothing in FastPM or BullFrog is non-differentiable** — they are smooth functions of `a` via the
  growth factors, evaluated on a pre-tabulated schedule. **[INF, from the formulas.]**
- **A fixed, pre-tabulated step schedule is strongly preferred**: it makes the loop a clean
  `jax.lax.scan`, removes per-step host synchronisation, and enables checkpointing. Every fast-PM
  code does this.
- **It is also a hard requirement for the reversible adjoint.** DISCO-DJ II states the caveat
  precisely **[V-paper]**: *"given that we use a simple **non-adaptive**, time-reversible (albeit not
  necessarily symplectic) stepping scheme, particles propagate backwards on the exact same trajectory
  without discretisation errors (however with round-off errors)."*
- **The reversible adjoint requires FP64.** pmwd Table 2 (evolve a = 1/64 → 1 → back, RMSD of
  displacements relative to ~0.1 Mpc/h scatter) **[V-paper, arXiv:2211.09815]**:

  | precision | cells/particle | steps | displacement rel. diff |
  |---|---|---|---|
  | single | 8 | 63 | **5.2e-2** |
  | single | 8 | 126 | 2.1e-2 |
  | single | 1 | 63 | 4.3e-4 |
  | double | 8 | 63 | **5.4e-11** |

  Ordering of importance: **precision ≫ mesh/particle ratio > particle mass > number of steps** (and
  counterintuitively, *more* steps improves reversibility). LUNA runs FP64 on the DM path
  (`scripts/cosmo/run_cosmo_dm.py:18`) so it is in the safe row — **but this is a hard constraint if
  the reversible adjoint is ever wanted, and it rules out the `dm_fp32` path for that purpose.**
- Reference implementations to compare against: **DISCO-DJ II** (JAX, differentiable PM, offers
  exactly `"bullfrog"` / `"fastpm"` / `"symplectic"`, O(1)-memory adjoint) is the closest published
  relative of what LUNA is becoming; **pmwd** (JAX) and **FlowPM** (TF, a FastPM implementation).

---

## 7. Ranked recommendations

Ranked by (steps saved to z=0) × (feasibility in a pure-PM differentiable JAX code).

| # | Action | Expected benefit | Difficulty | Risk |
|---|---|---|---|---|
| **R1** | **Convergence-test η** in `cosmo_common.py:60-66`. Quinn et al. recommend **0.3** for exactly this formula; LUNA uses 0.025. | potentially **9–12×** fewer steps | trivial | med — τ-vs-t unit mapping differs, must verify empirically, not assume |
| **R2** | **Switch to a displacement (CFL) criterion**: particles move < ⅓ mesh cell/step. The correct criterion for pure PM (Klypin & Prada eq. 17, `N_s ≈ 60·N_g/L`). Machinery already exists at `runner_cosmo_dm.py:365`. | a principled floor: **~300 steps** at N_g=512, L=100 | low | low |
| **R3** | **FastPM kick/drift factors.** Unique Zel'dovich-consistent *and* symplectic scheme; LUNA's super-comoving KDK is already the unmodified case ⇒ a two-coefficient swap. | removes coarse-step growth error; lets R2's floor be reached without accuracy loss | low | low — reduces to the current scheme as dt→0 |
| **R4** | **Integer time coordinate** (GADGET-4 style, §2.1) + a **fixed pre-tabulated schedule** driving `lax.scan`. | exact synchronisation; removes per-step host sync; prerequisite for R6 | low–med | low |
| **R5** | **Raise mesh/particle ratio to ≥ 2.** FastPM and ICE-COLA both require it; Klypin & Prada's convergence argument is fundamentally about force resolution. Also improves adjoint reversibility 100× (pmwd Table 2). | gates whether *any* step reduction yields correct halos | med (memory) | low |
| **R6** | **Reversible adjoint** (pmwd / DISCO-DJ II style), O(1) memory in N_steps. **Requires FP64 and a non-adaptive scheme (R4).** | makes long differentiable rollouts at 134M particles feasible | high | med–high — needs a forward/reverse trajectory check |
| **R7** | *(optional)* **BullFrog** — 2LPT-exact per step, zero overhead, beats FastPM from ≳5 steps. At 100–300 steps it ties FastPM. | marginal in LUNA's regime | low | low — not symplectic |
| **R8** | *(research)* **PGD or a learned force-kernel correction** through LUNA's AD — where differentiability actually pays. | 10-step FastPM → 5% P(k) to k = 10 h/Mpc | high | med — needs a high-res reference for calibration |
| **R9** | **COLA** — skip (§5.4). | — | high | — |
| **R10** | **Higher-order symplectic** — do not pursue (§5.5, verified negative). | — | — | — |
| **R11** | **Hierarchical/individual timesteps** — does not apply (§3, ceiling of 1/2 = a net loss). | — | — | — |

**The honest target.** Do not quote the fast-mock literature's "10–40 steps": those papers optimise
for P(k) and halo counts, and Klypin & Prada show P(k) is forgiving (34 steps → 3% at k = 0.3) while
the high-density PDF is not — they explicitly reject 40-step FastPM as under-resolved in time.
**R1 + R2 + R3 landing LUNA at ~300 steps at N_g = 512 is the defensible claim: a ~20× reduction
from 7000, principled, and citable to a PM code paper.** R1 and R2 alone, without touching the
integrator, plausibly get most of it.

**Method caveat.** The negatives in R10/R11 rest on arXiv full-text queries plus direct primary-source
statements (Quinn et al. §5; List & Hahn Prop. 3), not an exhaustive sweep. They are strong; an ADS
full-text search would make them airtight.

---

## 8. Primary sources

- Quinn, Katz, Stadel & Lake 1997, astro-ph/9710043 — multistepping KDK, the η√(ε/a) criterion, and the pure-PM cost model
- Saha & Tremaine 1994, astro-ph/9403057 — symplectic individual timesteps (and why they don't port)
- Teyssier 2002, astro-ph/0111367 — RAMSES, the one-way AMR Poisson interface
- Kravtsov, Klypin & Khokhlov 1997, astro-ph/9701195 — ART (constant timestep)
- Springel, Pakmor, Zier & Reinecke 2021, arXiv:2010.03567 — GADGET-4, hierarchical time integration
- Tassev, Zaldarriaga & Eisenstein 2013, arXiv:1301.0322 — COLA
- Izard, Crocce & Fosalba 2016, arXiv:1509.04685 — ICE-COLA
- Feng, Chu, Seljak & McDonald 2016, arXiv:1603.00476 — FastPM
- Klypin & Prada 2018, arXiv:1701.05690 — GLAM; the PM step-count convergence study and eq. 17
- List & Hahn 2023, arXiv:2301.09655 — order barriers after shell-crossing
- Rampf, List & Hahn 2024, arXiv:2409.19049 — BullFrog
- List, Hahn, Flöss & Winkler 2025, arXiv:2510.05206 — DISCO-DJ II, the unifying DKD frame
- Li et al. 2022/2024, arXiv:2211.09958, arXiv:2211.09815 — pmwd and its adjoint/reversibility tables
- Modi, Lanusse & Seljak 2020, arXiv:2010.11847 — FlowPM
- Dai, Feng & Seljak 2018, arXiv:1804.00671 — PGD
- Lanzieri, Lanusse & Starck 2022, arXiv:2207.05509 — hybrid physical-neural ODEs
- Rantala et al. 2021, arXiv:2011.14984 — FROST (4th-order forward symplectic, collisional)
