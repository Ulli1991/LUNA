# Precision and representation in cosmological PM / N-body codes — survey and recommendations for LUNA

Status: research note, not committed policy. Written 2026-08-06.

Companion: `INTEGRATOR_SURVEY.md` covers time integration (GADGET-4's integer time and hierarchical
scheme, and what a pure-PM code can adopt — FastPM/BullFrog/COLA, step-count convergence).

## How to read this

Every claim is tagged:

- **[V-src]** — verified by reading the actual source code (file:line given).
- **[V-paper]** — verified by reading the paper text (section given).
- **[V-meas]** — measured in this survey on viper MI300A at production scale; script paths given.
- **[INF]** — inference or reasoning by the author of this note, not stated by any source.
- **[unverified]** — could not be confirmed; do not cite as fact.

Two labels separate ideas by provenance:

- **VERIFIED-IN-LITERATURE** — somebody has published this technique and (usually) measured it.
- **NOVEL-PROPOSAL** — not found in any source consulted; proposed here. Treat as untested.

The standing rule for LUNA is **speed *and* accuracy**. An idea that trades accuracy for speed
does not qualify. An idea that gains accuracy at equal speed does.

---

## 1. LUNA's starting point, and one correction to the premise

Measured baseline, single MI300A, 512³ mesh, 134M particles: gravity step ≈ 45 ms, of which the
CIC deposit is ≈ 1/3. Reported precision sensitivity: deposit gains only **1.17×** FP64→FP32,
V-cycle gains **1.8×**.

**Correction — the 1.17× did not measure what it appeared to measure.** In
`gravity/particle_mesh_blocks.py:188-206` the `dtype` argument casts *only* the weights and the
grid, and does so *after* the position arithmetic:

```python
f  = [pos_l[:, a] / dx - 0.5 for a in range(3)]   # native dtype (FP64)
i0 = [jnp.floor(f[a]).astype(jnp.int32) for a in range(3)]
t  = [f[a] - i0[a] for a in range(3)]             # still FP64
...
if dtype is not None:
    w = w.astype(dtype)
    t = [q.astype(dtype) for q in t]              # cast happens HERE
```

So in the "FP32 deposit" the 134M × 3 × 8 B = 3.2 GB of position traffic stayed FP64. **[V-src]**
The 1.17× is what that experiment should produce; it is not evidence about atomics. The same
structure appears in `particle_mesh.py` and `particle_mesh_shardmap.py`.

A second structural point: all three deposit paths issue **eight separate `.at[].add()` calls**,
one per CIC corner (`particle_mesh_blocks.py:209-216`). That is eight XLA scatter ops, so the
position array is read from HBM eight times and the index/weight arithmetic is recomputed eight
times. **[V-src]**

Both facts are actionable independently of any precision change (§9, R1/R2).

**What is nonetheless true:** the deposit *is* contention-sensitive. LUNA's own comment records a
26× swing (2.6 s → ~0.1 s at 512³) from degenerate padding all landing on eight addresses
(`particle_mesh_blocks.py:194-203`) **[V-src]**, and this survey measured a 3.5× swing from
particle ordering alone (§8).

---

## 2. Summary table — what established codes actually do

| Code | Single precision | Double precision | Special representation | Claimed benefit | Citation |
|---|---|---|---|---|---|
| **GADGET-4** | `MyFloat`/`MyDouble` = float by default; PM FFT selectable | `DOUBLEPRECISION=1` all; `=2` mixed (double coords, single rest) | **`MyIntPosType` = uint32 / uint64 / uint128 integer positions**; integer time | Uniform positional accuracy 2.33e-10 (uint32) vs ~1e-7 (FP32); free wrap/nearest-image via unsigned overflow; lossless reversible random box shifts | arXiv:2010.03567 §2.10; `src/data/dtypes.h:37-64` |
| **PKDGRAV3** | PP/PC/CP/CC interactions (AVX + CUDA); velocities | Accumulation; Ewald real-space; multipole/local expansions; interaction-list centres | **Both**: int32 positions (`INTEGER_FACTOR 0x80000000`, opt-in) **and** float offsets from a double interaction-list centre | Force errors ~1e-7 fractional; 28 B/particle; 9.5× GPU on force eval | arXiv:1609.08621 §3.1, §3.4; `ilp.h`, `ilc.h`, `pkd.h` |
| **Abacus** | Positions (as cell offsets) and velocities; near-field direct kernels | **All** far-field: multipoles, Taylors, FFTs (`FFTComplex` hardcoded double) | **FP32 offsets relative to cell centres** → "+9–10 bits of mantissa beyond the nominal 24" | Median frac. force error 1.2e-6 (ΛCDM); *"ensures that we do not need double precision"* → consumer GPUs | arXiv:2110.11392 §3.5.1, §8.3; `singlestep/DataModel/grid.cpp` |
| **SWIFT** | Velocities, accelerations, masses, potentials; all multipole moments and field tensors; SPH + gravity caches | **Positions** `double x[3]`; cell geometry; multipole CoM; L2P; MAC; **entire PM mesh incl. double FFTW** | Offset-to-local-frame in the *caches* — but inconsistently: cell corner (SPH), cell centre (gravity self), **zero shift** (gravity pair). No integer positions (integers used for the *timeline*) | Not quantified as speedup; §4.6 identifies FP32 as the FMM accuracy floor at θ_cr<0.3 | arXiv:2305.13380 §4.6; `gravity_part.h:33`; `cache.h:209`; `gravity_cache.h:234-236` |
| **HACC** | Short-range PP/tree; positions, velocities, mass, potential; particle time-stepping | **Long-range spectral/PM solver** — FFT, Green's function (double FFTW); time-integration scalars | Positions FP32 in **rank-local grid-cell units** (`coords_global2local`), 38 B/particle | *"double precision for the spectral component… single precision is adequate for the short/close-range"* | arXiv:1211.4864 §II; arXiv:2510.03557 §III; `BasicDefinition.h:71-83` |
| **RAMSES** | `-DNPRE=4` nominally; **does not compile at HEAD** | Default all (`dp=real*8`); MPI buffers, MG hierarchy, residual norms unconditionally real*8 | Box-normalized units; transient cell-relative offsets in CIC only. **real*16 Hilbert keys** under `-DQUADHILBERT` | Memory only. Teyssier 2002 and Guillet & Teyssier 2011 are **silent** on precision | `amr/amr_parameters.f90:1-36`; issue #57 |
| **pmwd** | `float_dtype = float32` particle/mesh pipeline | `cosmo_dtype = float64` always; forces `jax_enable_x64` at import | **`pmid` (int16 nearest cell) + `disp` (float32 offset)** — the JAX analogue of the Gadget/Abacus idea | *"save the particle locations with much more uniform precision than positions, wherever they are"* | `pmwd/particles.py:31-42`, `configuration.py:15,106-108`; arXiv:2211.09815 |
| **JaxPM** | float32 throughout (no x64 in library) | — | Has pmwd's pmid+disp as opt-in `cic_paint_dx`; motivation stated in code is *sharding correctness*, not precision | — | `jaxpm/painting.py`, `painting_utils.py` |
| **DISCO-DJ** | `precision: str = "single"` default | opt-in `"double"` (requires x64) | Absolute positions; int16 stencil offsets | First-class precision switch — the only JAX cosmology code with one | `disco_dj.py:49,61-62,106-109`; arXiv:2311.03291 |
| **CUBE** | **int8/int16 fixed-point positions, AVX512-VNNI int8 PP kernel** | — | 1–2 bytes/coordinate; 6 B/particle total | Memory; the only known low-precision *gravity kernel* in the field, and it is **integer**, not FP16 | arXiv:2003.03931 |
| **QUDA** (lattice QCD, for contrast) | FP32 compute | FP64 outer solver | **Custom 16-bit and 20-bit fixed point + shared exponent**, explicitly *rejecting* IEEE FP16 | double-int20 **2.9× faster than pure FP64** and 2.3× faster than double-FP16 | arXiv:2302.09224 |

**Two families of trick, and they are not the same thing:**

1. **Fixed-point integers** (GADGET-4, PKDGRAV3, CUBE, pmwd's `pmid`) — buy *uniform absolute
   resolution* across the box, and make periodic wrapping exact.
2. **Float offsets from a local origin** (Abacus cell centres, PKDGRAV3 interaction lists, SWIFT
   caches, HACC rank-local units) — buy *mantissa bits* by shrinking the exponent range.

Abacus and PKDGRAV3 use both, at different layers. In every code that does (2), **the subtraction
is performed in double and only the difference is rounded to float** — `(float)(a_double - b_double)`
is the portable idiom across all four codebases. **[V-src, multiple]**

**Under-documented in the literature:** Abacus is the only code that states the offset trick in a
paper. SWIFT's rationale exists only as a source comment; HACC's only in code; PKDGRAV3's paper
describes the integer positions but not the interaction-list offsets.

---

## 3. GADGET-4 integer positions — the full mechanism

Source read directly: depth-1 clone of `https://gitlab.mpcdf.mpg.de/vrs/gadget4.git`. All line
numbers below are **[V-src]** from that tree.

### 3.1 The type

`src/data/dtypes.h:24-64`:

```c
#if !defined(POSITIONS_IN_32BIT) && !defined(POSITIONS_IN_64BIT) && !defined(POSITIONS_IN_128BIT)
#ifndef DOUBLEPRECISION
#define POSITIONS_IN_32BIT     /* default when the code is single precision */
#else
#define POSITIONS_IN_64BIT     /* default when the code is double precision */
#endif
#endif

#ifdef POSITIONS_IN_32BIT
typedef uint32_t MyIntPosType;
typedef int32_t  MySignedIntPosType;
#define BITS_FOR_POSITIONS 32
#endif
/* ... uint64_t / int64_t at :47-55, uint128_t / int128_t at :57-64 ... */
```

Three widths, plus SIMD variants (`Vec4ui` :42, `Vec4uq` :52) from the bundled vectorclass library.
128-bit is documented as *"extreme spatial dynamic range… only of interest in truly extreme
scenarios"* (`documentation/04_config-options.md:790`).

### 3.2 The box ↔ integer mapping

`src/io/snap_io.cc:439-440` (identically at `:500-501`, `src/lightcone/lightcone.cc:58-59`,
`src/gravity/ewald.cc:74`):

```c
Sp->FacCoordToInt = pow(2.0, BITS_FOR_POSITIONS) / Sp->RegionLen;
Sp->FacIntToCoord = Sp->RegionLen / pow(2.0, BITS_FOR_POSITIONS);
```

**The whole box maps onto the whole unsigned integer range.** This is the essential design choice
— it is what makes periodic wrapping free. There is no unused headroom and no offset.

### 3.3 Why — the authors' own rationale

arXiv:2010.03567 **§2.10 "Integer coordinates"** **[V-paper]**:

> "Ordinary floating point numbers are not ideal for representing the particle coordinates … because
> they feature **variable absolute positional accuracy throughout a simulation box** … it is
> attractive to consider integers for storing the coordinates in the box, as this delivers
> **uniform resolution** and can make optimum use of the bits assigned for the storage. When using
> 32-bit variables, the use of integers gives a relative positional accuracy within the box equal
> to **1/2³² ≈ 2.33e-10 instead of the machine precision of single precision floating point
> numbers, ≈1e-7**. … When double precision positions are replaced by 64-bit integer storage, the
> available relative accuracy … goes up to **5.42e-20**."

And from the config documentation, `documentation/04_config-options.md:319-324` **[V-src]**, on
`RANDOMIZE_DOMAINCENTER` (shifting the whole particle set to decorrelate tree force errors):

> "Since integer coordinates are used for particle positions, this shifting does **not entail any
> round-off error, and is reversible**. In particular, the shifts will not be visible in any of
> the outputs created."

That reversibility property is worth noting for a differentiable code.

### 3.4 Drift — float displacement, integer increment

`src/time_integration/predict.cc:349-358` **[V-src]**:

```c
double posdiff[3];
for(int j = 0; j < 3; j++)
    posdiff[j] = P->Vel[j] * dt_drift;          // displacement in FP64

MyIntPosType delta[3];
pos_to_signedintpos(posdiff, (MySignedIntPosType *)delta);

for(int j = 0; j < 3; j++)
    P->IntPos[j] += delta[j];                    // unsigned add: wraps for free

constrain_intpos(P->IntPos);                     /* only does something for a stretched box */
```

with the conversion (`src/data/intposconvert.h:326-337`):

```c
template <typename T>
inline void pos_to_signedintpos(T *posdiff, MySignedIntPosType *intpos) {
    for(int j = 0; j < 3; j++)
        intpos[j] = std::lrint(posdiff[j] * FacCoordToInt);   // llrint for 64-bit
}
```

Three things to note. The displacement is computed and scaled in floating point; only the
*increment* is integerized, with **round-to-nearest** (`lrint`), not truncation. The add is
unsigned, so a particle leaving the box wraps by two's-complement overflow with no branch and no
round-off. `constrain_intpos` (`intposconvert.h:70-87`) is a no-op unless `LONG_X/Y/Z_BITS` is set
for a non-cubic box.

**Accuracy consequence [INF]:** drift is *not* exact — each step quantizes the displacement to one
integer unit (5.8e-6 kpc/h for a 25 Mpc/h box in uint32). But the quantization is unbiased and
bounded, and crucially the *position* never accumulates float round-off, because it is never
re-expressed in floating point.

### 3.5 Periodic wrapping and exact position differences

`src/data/intposconvert.h:167-269`, the un-stretched branch reduces to:

```c
MyIntPosType   delta[3];
MySignedIntPosType *intpos = (MySignedIntPosType *)delta;   /* reinterpret, not convert */

delta[0]   = a[0] - b[0];                    /* unsigned difference, wraps */
posdiff[0] = intpos[0] * FacIntToCoord;      /* signed reinterpretation = nearest image */
```

Because the box is the *entire* unsigned range, the unsigned difference reinterpreted as signed is
**already the minimum-image displacement** — no `if`, no `fmod`, no `round`. The GADGET-4 paper
states the benefit as periodic wrap and nearest image *"without any branching"* **[V-paper §2.10]**.

The difference is also **exact**: `a - b` in integer arithmetic has no round-off, so there is no
catastrophic cancellation when two nearby particles are far from the origin. This is the property
FP32 absolute positions most conspicuously lack.

### 3.6 PM mesh assignment — index by shift, weight from the low bits

This is the part that maps directly onto LUNA's CIC. `src/pm/pm_periodic.h:67` **[V-src]**:

```c
#define INTCELL ((~((MyIntPosType)0)) / PMGRID + 1)
```

For `PMGRID = 2^m` and `BITS_FOR_POSITIONS = B`, this is exactly `INTCELL = 2^(B-m)`. The deposit
(`src/pm/pm_periodic.cc:800-850`):

```c
slab_x = partin[i].IntPos[0] / INTCELL;      /* == IntPos >> (B-m)   : the cell index  */
rmd_x  = partin[i].IntPos[0] % INTCELL;      /* == IntPos & (2^(B-m)-1) : sub-cell bits */
/* ... same for y, z ... */

double dx = rmd_x * (1.0 / INTCELL);         /* sub-cell offset in [0,1) */
double dy = rmd_y * (1.0 / INTCELL);
double dz = rmd_z * (1.0 / INTCELL);

int slab_xx = slab_x + 1;  if(slab_xx >= GRIDX) slab_xx = 0;   /* +1 neighbour, wrapped */
/* ... */

rhogrid[FI(slab_x , slab_y , slab_z )] += mass * (1.0-dx) * (1.0-dy) * (1.0-dz);
rhogrid[FI(slab_x , slab_y , slab_zz)] += mass * (1.0-dx) * (1.0-dy) * (    dz);
rhogrid[FI(slab_x , slab_yy, slab_z )] += mass * (1.0-dx) * (    dy) * (1.0-dz);
rhogrid[FI(slab_x , slab_yy, slab_zz)] += mass * (1.0-dx) * (    dy) * (    dz);
rhogrid[FI(slab_xx, slab_y , slab_z )] += mass * (    dx) * (1.0-dy) * (1.0-dz);
/* ... 8 corners total ... */
```

Since `PMGRID` is a compile-time power of two, the compiler turns `/ INTCELL` and `% INTCELL` into
a shift and a mask. **The cell index and the CIC weight come out of one integer with no floating-point
position arithmetic at all.** The only float work is `rmd * (1.0/INTCELL)` and the eight products.

The identical pattern is used in the IC generator (`src/ngenic/ngenic.cc:45` redefines `INTCELL`
against `GRIDX`, deposit at `:893-899`) and, with a `POWERSPEC_FOLDFAC` multiplier, in the folded
power-spectrum estimator (`pm_periodic.cc:341-355`).

The tree uses the same bits: the octant index at each level is pure masking
(`src/tree/tree.cc:491-517`), so node boundaries coincide exactly with integer bit boundaries and
carry no round-off.

Note GADGET-4's PM **grid** is `fft_real` = `float` or `double` per `DOUBLEPRECISION_FFTW`
(`src/pm/pm.h:20-30`) — the integer representation is for *positions*, not for the accumulator.

### 3.7 Reimplementing this in JAX

The arithmetic is a direct transcription. With `B = 32`, `N = 2^m` cells per dimension:

```python
SHIFT   = 32 - m                      # e.g. 32 - 9 = 23 for N = 512
INTCELL = np.uint32(1) << SHIFT       # 2^23
INV     = np.float32(1.0 / float(INTCELL))

# box -> integer, once, at IC time (do this in float64)
ipos = jnp.mod(jnp.rint(pos_f64 / box * 2.0**32), 2.0**32).astype(jnp.uint32)

# cell index and sub-cell offset, per axis
cell = (ipos >> jnp.uint32(SHIFT)).astype(jnp.int32)          # exact, in [0, N)
t    = (ipos &  jnp.uint32(INTCELL - 1)).astype(jnp.float32) * INV   # in [0,1)

# drift: float displacement -> integer increment -> unsigned add (wraps for free)
delta = jnp.rint(disp_f64 * (2.0**32 / box)).astype(jnp.int32)
ipos  = (ipos + delta.astype(jnp.uint32))          # uint32 wraparound == periodic wrap

# minimum-image difference between two particles, exact:
d_int = (ipos_a - ipos_b).view(jnp.int32)          # or .astype after bit-reinterpret
d_phys = d_int.astype(jnp.float64) * (box / 2.0**32)
```

**JAX-vs-C differences to watch [INF, from the JAX semantics, worth a unit test]:**

- JAX `uint32` arithmetic wraps like C's unsigned — this is the load-bearing property and should be
  asserted in a test, not assumed.
- C's `(MySignedIntPosType*)` cast is a *bit reinterpretation*. In JAX use
  `jax.lax.bitcast_convert_type(x, jnp.int32)`, **not** `.astype(jnp.int32)` — the latter saturates
  or wraps by value conversion, which is not the same thing.
- Python/NumPy scalars promote to int64 readily; keep every literal explicitly `jnp.uint32(...)` or
  the shift/mask silently widens.
- `x >> k` and `x & mask` on `uint32` are supported and lower to plain integer ops.
- LUNA's CIC is **cell-centred** (`f = pos/dx - 0.5`), GADGET-4's `dx` is measured from the cell
  *edge*. Either subtract half a cell from `t` and decrement the cell index where it goes negative,
  or shift the mapping by `INTCELL/2` once at conversion time. The benchmark in §8 uses the former.

Storage: uint32 positions are **12 B/particle vs 24 B for FP64** — at 134M particles that is
1.6 GB saved, and 1.6 GB/step of HBM traffic saved in every kernel that reads positions.

---

## 4. Mixed-precision multigrid — the theory LUNA's `mixed` mode already relies on

LUNA's `LUNA_MG_MIXED=1` path (`gravity/poisson_multigrid.py:468-492`) computes the FP64 residual
of the carried potential, runs an FP32 V-cycle on the correction, and accumulates in FP64. That is
textbook iterative refinement, and it has a precise theory.

### 4.1 The canonical chain

| Ref | Citation | DOI |
|---|---|---|
| Wilkinson 1948 | *Progress report on the Automatic Computing Engine*, NPL Report MA/17/1024 — first computer implementation of IR | — |
| Wilkinson 1963 | *Rounding Errors in Algebraic Processes*, HMSO / Dover 1994 | (book) |
| Moler 1967 | "Iterative Refinement in Floating Point", *J. ACM* 14(2), 316–321 | `10.1145/321386.321394` |
| Skeel 1980 | "Iterative refinement implies numerical stability for Gaussian elimination", *Math. Comp.* 35(151), 817–832 | `10.1090/S0025-5718-1980-0572859-4` |
| Carson & Higham 2017 | "A New Analysis of Iterative Refinement and Its Application to Accurate Solution of Ill-Conditioned Sparse Linear Systems", *SIAM J. Sci. Comput.* 39(6), A2834–A2856 | `10.1137/17M1122918` |
| **Carson & Higham 2018** | "Accelerating the Solution of Linear Systems by Iterative Refinement in Three Precisions", *SIAM J. Sci. Comput.* 40(2), A817–A847 | `10.1137/17M1140819` |
| **Higham & Mary 2022** | "Mixed precision algorithms in numerical linear algebra", *Acta Numerica* 31, 347–414 | `10.1017/S0962492922000022` |

### 4.2 The statement that justifies LUNA's design

Carson & Higham 2018 Algorithm 1.1 uses three precisions: **u** (working/storage), **u_f**
(factorization/solve), **u_r** (residual), with `u_r ≤ u ≤ u_f`. Corollary 3.3 **[V-paper]**:

> "the forward error is reduced on the *i*th iteration by a factor approximately φ_i until an
> iterate x̂ is produced for which
> **‖x − x̂‖∞/‖x‖∞ ≲ 4p·u_r·cond(A,x) + u**"

and, verbatim:

> "**the rate of convergence of iterative refinement depends only on the effective precision of the
> solves and is not affected by the precision at which the data is stored or the residual is
> computed. By contrast, the limiting precision depends on u_r and u, but not on u_s.**"

Higham & Mary 2022 §6 states the same conclusion:

> "**the limiting accuracy depends on the precisions u and u_r but does not depend on the precision
> u_ℓ**, on x₁, or on how the system A d_i = r_i is solved, provided that it is solved with some
> relative accuracy ξ_i ≪ 1."

**This is the citation for LUNA's `mixed` mode.** The final accuracy is set by the FP64 residual and
FP64 carry; the FP32 V-cycle affects only the convergence *rate*. That is exactly the design.

Condition-number thresholds for convergence (Carson & Higham 2018 Table 7.1, **[V-paper]**):
FP32 solve inside FP64 working precision is good to **κ∞(A) ≤ 1e8**. A discrete 3D Poisson operator
on a 512³ periodic grid has κ ~ (N/π)² ~ 2.7e4 **[INF, standard result]** — four orders of magnitude
inside the bound. LUNA's `mixed` mode is not close to any theoretical limit.

### 4.3 Multigrid-specific work

Higham & Mary 2022 §8.5 **[V-paper]**:

> "The most popular approach has been to use a multigrid method as the inner solver for iterative
> refinement, that is, to use Algorithm 6.1 with a multigrid solver on line 3, usually run in lower
> precision. Single precision multigrid methods have, for example, been used within double precision
> iterative refinement algorithms by Göddeke, Strzodka and Turek (2007), Goddeke and Strzodka (2011),
> Sumiyoshi et al. (2014), and Kronbichler and Ljungkvist (2019). … **The first error analysis of
> multigrid methods in this context was performed by McCormick, Benzaken and Tamstorf (2021), who
> observed that different levels in the grid hierarchy should use different precisions: coarser
> grids are more resilient to lower precisions.**"

- **Göddeke, Strzodka & Turek**, *Int. J. Parallel Emergent Distrib. Syst.* 22(4), 221–256 (2007),
  `10.1080/17445760601122076`; **Göddeke & Strzodka**, *IEEE TPDS* 22(1), 22–32 (2011),
  `10.1109/TPDS.2010.61`. Quantified results are quotable from Göddeke's open-access thesis
  (*Fast and Accurate Finite-Element Multigrid Solvers for PDE Simulations on GPU Clusters*, TU
  Dortmund 2010, `hdl.handle.net/2003/27243`), which states it is an extended version of that joint
  work **[V-paper]**:
  > "**When employing multigrid to approximately solve the auxiliary problems to no more than two
  > digits accuracy, we obtain a speedup of 1.7–2.1 over executing entirely in double precision**,
  > on CPUs and on GPUs."

  and on accuracy:
  > "**Even for severely ill-conditioned problems (condition numbers on the order of 10¹⁵), our
  > mixed precision solvers always compute the same result as if computing entirely in double
  > precision.**"

  Their own explanation of the speedup: *"The speedup is in line with the bandwidth considerations"*
  — i.e. it is a **bytes** win, not a flops win. That is exactly LUNA's measured 1.8× on the V-cycle.
- **McCormick, Benzaken & Tamstorf**, *SIAM J. Sci. Comput.* 43(5), S392–S419 (2021),
  `10.1137/20M1348571`, arXiv:2007.06614 — first rounding-error analysis of multigrid.
  Key structural result **[V-paper]**:
  > "**while iterative refinement is susceptible to quantization errors during the residual and
  > update computation, the V-cycle used to compute the correction in each iteration is much more
  > resilient** … As a result, **the V-cycle only requires relatively few bits of precision per
  > level.**"
- **Tamstorf, Benzaken & McCormick**, *SIAM J. Sci. Comput.* 43(5), S420–S447 (2021),
  `10.1137/20M1349230`, arXiv:2007.06628 — progressive precision applied to elliptic PDEs.
  Caveat: the demonstration is the 1D biharmonic equation in a variable-bit framework, **not** IEEE
  FP16 hardware.
- **Oo & Vogel**, arXiv:2007.07539 (2020) — FP16 geometric multigrid preconditioner, V100, Poisson.
  Treated in §7.

### 4.4 Known pitfalls — the ones that actually bite

1. **The residual, its accumulation, the operator, and the update must all stay in the high
   precision.** These are separate requirements (Algorithm 1.1 lines 2/3/4 use u_r/u_ℓ/u).
   Göddeke's operational corollary **[V-paper]**: the operator assembly *"must be executed in high
   precision or otherwise, the high precision correction loop does not gain accuracy at all."*
2. **Over-solving the inner problem in low precision is actively harmful.** Göddeke §5.7.1
   **[V-paper]**:
   > "iterative solvers are **not guaranteed to converge in single precision, they may stall or even
   > diverge** … **Reducing the residuals further is, in contrast to theory that is based on exact
   > solutions, not beneficial because the inner solvers converge to wrong solutions in single
   > precision or may not converge at all.**"

   Their empirically optimal setting is ~2 digits of inner reduction, i.e. **1–2 V-cycles**. The
   classical "3–4 IR iterations" bound assumes an *exact* inner solve and does not apply.
   **This is directly relevant: LUNA should not increase its inner V-cycle count in `mixed` mode.**
3. **Residual scaling.** Both Göddeke and Oo & Vogel rescale the defect to unit norm before the
   down-cast. Göddeke reports that without it the solver is *"prone to diverging or producing NaNs"*
   under fast-math/non-IEEE compiler settings. For FP32 the exponent headroom makes this optional;
   for FP16 it is mandatory (§7).
4. **Double-single ("float-float") emulation fails where mixed-precision IR succeeds.** Göddeke
   §5.6.3 **[V-paper]**:
   > "**the double-single format is not sufficient** … **the double-single format does not increase
   > the dynamic range beyond that of single precision**, despite twice the bit length of the
   > combined mantissa: The emulated format is not equivalent to native double precision (s52e11),
   > but rather to a **s46e8** format."

   Also *"significantly slower than native single precision on the device."* **Conclusion: do not
   pursue double-single arithmetic in LUNA.** By the same argument double-double would not extend
   the exponent range either.
5. **Memory cost.** Göddeke: *"the memory footprint of the method increases by the quotient of the
   high and low precision bit length, because all data has to be stored twice."* LUNA already pays
   this in `mixed` mode.

---

## 5. The scatter-bound deposit — what the literature actually shows

This is LUNA's main non-roofline component, so it gets the most space. Sources span cosmology,
plasma PIC (where this problem is far better studied), and GPU computing.

### 5.1 Hardware ground truth for MI300A

The decisive fact, established by compiling `atomicrmw fadd` with the installed ROCm 6.4.3 /
AMD LLVM 19 `llc` offline (no GPU required), including at the exact `syncscope("agent-one-as")`
form with `!amdgpu.no.fine.grained.memory` that XLA's ROCm backend emits **[V-src, this survey]**:

| target | global f32 | global f64 | LDS f32 | LDS f64 | global i32/i64 |
|---|---|---|---|---|---|
| **gfx942 (MI300A)** | `global_atomic_add_f32` | `global_atomic_add_f64` | `ds_add_f32` | `ds_add_f64` | `global_atomic_add` / `_x2` |
| gfx90a (MI250X) | **CAS loop** | **CAS loop** | `ds_add_f32` | `ds_add_f64` | native |

**Consequences.**

- On MI300A LUNA's FP64 scatter already uses **native hardware FP64 global atomics**. The cost is
  genuine atomic-unit/transaction behaviour, not CAS emulation.
- **MI250X-derived deposition numbers do not transfer to viper.** Any Frontier/LUMI result about
  atomics is measuring a qualitatively different machine.
- Integer atomics are native at both 32 and 64 bits — so fixed-point accumulation costs the same
  transactions as float. This is confirmed by direct measurement in §8.

Corroborating: the CDNA3 ISA guide lists `GLOBAL_ATOMIC_ADD_F32/F64`, `GLOBAL_ATOMIC_PK_ADD_F16/BF16`
and `DS_ADD_F32/F64` **[V-src, AMD MI300 ISA reference §10.1, §12.15]**. Note there is **no scalar
`GLOBAL_ATOMIC_ADD_F16`** — only the packed 2-component form (relevant in §7).

### 5.2 Particle sorting / binning — the best-evidenced technique

**Myers et al., "Porting WarpX to GPU-accelerated platforms", arXiv:2101.12149, *Parallel Computing*
108 (2021) 102833** **[V-paper]**:

> "the difference between sorting optimally and not sorting at all is a factor of **≈ 7.5**, with
> most of the improvement coming from the current deposition and fused gather and push."

Single V100, uniform plasma, 8 particles/cell, large thermal velocity, 3rd-order Esirkepov, **sort
cost included**. Optimal = sort by cell, every step. Crucially, the same paper says the mechanism is
**cache reuse, not atomics**:

> "when running on GPUs, we dispense with thread-private buffers and perform atomic writes directly
> to global memory… The performance of global atomics on V100 is quite good"

and *"WarpX has experimented with thread-block level buffers in GPU shared memory… none of these
methods have been faster."* Their roofline puts sorted deposition at the **L2 streaming limit**.

Shipped defaults **[V-src, WarpX `Docs/source/usage/parameters.rst`, `Source/WarpX.cpp:1481`]**:
`sort_intervals` = 4 on GPU, `sort_bin_size` = 1 1 1, `sort_particles_for_deposition` = true on CUDA.
They distinguish **binning** (permutation array only) from **sorting** (physical reorder):
*"Cache utilization requires full sorting, but for many operations simply knowing the cell-sorted
indices is sufficient."*

Sort granularity across codes **[V-src]**:

| code | granularity | frequency |
|---|---|---|
| WarpX GPU | 1 cell | every 4 steps |
| PIConGPU | supercell 8×8×4 = 256 cells | incremental, every step |
| VPIC | voxel, GPU order transposed | `sort_interval` |
| Abacus, CUBE | cell — permanently cell-ordered storage | structural invariant |

**Decyk & Singh**, *Comput. Phys. Commun.* 185(3):708–719 (2014) **[V-paper]** gives the cost side:
reordering costs **15–30% of runtime** on Fermi M2090; 0.3–1.0% of particles leave a tile per step.
And a result that maps precisely onto the gfx90a/gfx942 split above: *"the collision-resolving
algorithm is about twice as fast as the collision-free algorithm"* on Fermi (which had FP atomics),
whereas on GT200 (atomicCAS emulation) collision-free won by ~10%.

**Not verified:** Bowers 2001 (*J. Comput. Phys.* 173(2):393) — the widely quoted "40–70%" could not
be confirmed from a primary source. **Do not cite that number.** No paper was found giving a
*measured* speedup for Morton/Hilbert reordering of a deposit specifically.

### 5.3 Privatization into shared memory / LDS — big on MI250X, probably not on MI300A

**WarpX PR #3368** (A. Myers, merged 2023-05-01) **[V-src]**, 4 GPUs of Crusher (**MI250X**), 256³
cells, order-3 shape, current-deposition time over 100 steps:

| tile | time |
|---|---|
| global atomics (baseline) | 9.595 s |
| 2×2×2 | 4.84 s |
| **4×4×4** | **2.912 s** |
| 8×8×8 | 3.636 s |

**3.30× on MI250X.** But from the same PR: *"On A100, the speedup is less dramatic, but the shared
algorithm still beats the dev one by ~30% if the tile size is chosen well."*

**That vendor gap is exactly what §5.1 predicts** — MI250X global FP atomics are CAS loops, so
privatizing into native `ds_add_*` wins hugely; A100 has native global FP atomics and wins little.
**MI300A is in the A100 category, not the MI250X category.** **[INF, from the verified ISA table]**
Expect ~30%, not 3.3×, and verify before investing.

Shipped tile sizes all land in 10–19 KiB of a 64 KiB LDS **[V-src]**: WarpX 3D `6 6 8` (+order-3
halo → 12×12×14), Smilei 3D 9³ × 3 components, PIConGPU 13×13×9. For LUNA, an 8³ FP64 interior tile
is 4.0 KiB and 10³ is 7.8 KiB — comfortable.

Three measured details worth knowing:

- **Smilei measured WarpX's buffer-reuse strategy at 0.92× in its own kernel** — the two codes
  genuinely disagree (`Projector3D2OrderGPUKernelCUDAHIP.cu:212-215`) **[V-src]**.
- **WarpX skips zero cells on flush** (`SharedDepositionUtils.H:42-62`) — valuable at LUNA's ~1
  particle/cell, where most tile cells are untouched.
- **Do not sort within a tile.** Smilei: *"We gain from the particles not being sorted inside a
  cluster because it reduces the bank conflicts… Such 'conflicted' accesses are serialized!"*

Smilei's own kernel comment records that the bottleneck merely relocates: *"This cluster is very
strongly bound by atomic operations in LDS."*

### 5.4 Gather instead of scatter / segmented reduction — loses at ~1 particle/cell

**RAMSES-GPU** (Teyssier et al., arXiv:2607.20704) is the most directly relevant source: it is
cosmological PM and ships **three** CIC kernels selectable at runtime **[V-paper §4.1]**:

- *large*: 3³ candidate cells, *"combines particles belonging to the same source cell within a warp
  before performing the atomic additions"*
- *medium* (default): *"shifts the source-cell convention by half a cell, allowing us to consider
  only the 2³ cells with non-zero weights"*, plus the same warp-level reduction; *"well suited for
  particle distribution with few (up to 10-20) particles per cell"*
- *small*: *"replaces the atomic additions with prefix sums and one mesh update for each source-cell
  segment"*; *"ideally suited for… more particles (say 1000) per cell"*

And the decisive sentence for LUNA's regime (N_part ≈ N_grid, ~1 particle/cell) **[V-paper §7.4]**:

> "we used for this test the **Atomic-Add mass deposition scheme, as it performs significantly
> better than the Prefix-Sum alternative** in this case."

Their throughput: *"around 5 billion particle depositions per GPU per second on the H200"*.

**Hariri et al., arXiv:1603.02886** (*Comput. Phys. Commun.*, 2016) implemented and benchmarked all
four variants on K20X **[V-paper]**: collision-free gather (threads on grid points); collision-
resolving with threads on buckets; collision-free with data replication; and plain scatter+atomics.
At full sort the collision-*resolving* bucket scheme won; the gather formulation was costly. They
chose plain scatter+atomics for production.

**Verdict for LUNA: the sorted-gather / segment_sum reformulation is not indicated at ~1
particle/cell.** It is the right answer only at high occupancy.

### 5.5 Warp-level aggregation

NVIDIA's warp-aggregated atomics give up to 21× — but only in the *all-threads-hit-one-address*
limit, and since CUDA 9 the compiler does it automatically; LLVM has `AMDGPUAtomicOptimizer`.
**Neither helps CIC**, because CIC addresses differ per particle.

The CIC-specific version does exist: RAMSES-GPU's *large*/*medium* kernels and VPIC 2.0 combine
same-cell particles within a warp before the atomic **[V-paper]**. But this requires the particles
to be cell-sorted first — **sorting and warp aggregation are one combined technique, not two.**
Needs a custom kernel; not expressible in `.at[].add()`.

### 5.6 Fixed-point / integer accumulation — nobody in this field does it

**Finding: no PIC or cosmology code was found that deposits mass/charge into a fixed-point integer
grid.** The nearest hits:

- **Hariri et al.** actually tried int64 atomics for deposition **[V-paper]**: *"Experiments using
  64-bit integer atomics as well as the single precision results suggest that a built-in double
  precision atomic could provide a significant speedup for our application."* Their motivation was
  that K20X had no native FP64 atomic. **On gfx942 that motivation is gone** — but the
  *reproducibility* motivation remains and was never their point.
- **QUOKKA** (He, Wibking, Vijayan, Krumholz, Li, arXiv:2509.18261) is the one astrophysics code
  that engineered reproducible GPU deposition — and it **rejected** fixed point, choosing
  error-bound tracking plus mantissa truncation instead **[V-paper, Appendix A]**: alongside the
  accumulator they track `E += ε|ΔU|` as an upper bound on reordering error, then zero the low
  `N_bit = log2(E/|ΔU|) + 52 + M` mantissa bits, with `M = 28` *"sufficient to make a
  non-reproducible result improbable as long as the number of cell-advances during which multiple
  particles deposit to the same cell is ≲10⁹."* They explicitly flag superaccumulators as future
  work: *"In future work we will also explore the option of using superaccumulators, as suggested by
  Defour & Collange (2015) and Collange et al. (2015), to guarantee full bitwise-reproducibility."*
- **QUDA** (lattice QCD) uses custom **16- and 20-bit fixed point with a shared exponent** as the
  storage format, explicitly rejecting IEEE FP16, and measures **double-int20 at 2.9× pure FP64**
  and 2.3× faster than double-FP16 **[V-paper, arXiv:2302.09224]**. Different operation (no scatter),
  but it is the strongest existing endorsement of fixed point over IEEE half in a solver.
- **CUBE** stores positions as 1–2 byte fixed point and runs an **int8 AVX512-VNNI PP gravity
  kernel** **[V-paper, arXiv:2003.03931]** — the only integer *force* kernel in cosmology.

**The technique is well established outside astrophysics — in molecular dynamics.** Correcting a
common misconception: **GROMACS and LAMMPS do *not* use fixed-point force accumulation** (GROMACS
fixes operation order via `mdrun -reprod`, which *"avoid[s] optimizations that affect binary
reproducibility; this can significantly reduce performance"* and only guarantees identity on the
same hardware and binary; LAMMPS uses double accumulators plus extra communication)
**[V-src/V-paper]**. But three MD codes do:

- **OpenMM, since version 4.0 (January 2012)** — the clearest statement of the whole idea
  **[V-paper, Eastman et al., *PLOS Comput. Biol.* 13(7):e1005659 (2017), and the developer guide]**:
  > "forces are accumulated as **64-bit fixed point values** … **ensures that force accumulation is
  > deterministic** … allows force accumulation to be done with **integer atomic operations, which
  > substantially improves performance**."

  Scale factor 2³² — both of LUNA's motivations (determinism *and* faster integer atomics), stated
  by the same authors, 14 years ago.
- **NAMD** — *"bonded forces are accumulated as 64-bit fixed point, while nonbonded forces are
  accumulated in single precision"* **[V-paper]**.
- **AMBER SPFP** — Le Grand, Götz & Walker, *Comput. Phys. Commun.* 184, 374–380 (2013),
  `10.1016/j.cpc.2012.09.022`.
- **HTMPC** — Westphal, Goh, Winkler & Gompper, *Comput. Phys. Commun.* 309, 109494 (2025),
  arXiv:2406.15236. The closest structural precedent: a GPU+MPI particle→cell code
  **[V-paper]**:
  > "the code optionally provides **bit-perfect reproducibility of particle trajectories by replacing
  > the floating point summations with fixed point summations** … not only on a single GPU, but also
  > on **distributed runs, independent of rank counts or domain decompositions**."

**And the state of play in cosmology — a real but narrower gap than it first appears.**

- **GADGET-4 explicitly disclaims rank invariance** **[V-paper, arXiv:2010.03567]**:
  > "We do, however, **not go as far as requiring that the results are binary invariant when the
  > number of MPI ranks … is changed** … Trying to eliminate those would exclude a large number of
  > computationally efficient algorithmic possibilities."

  Its `DEBUG_MD5` machinery targets the narrower property — *"check for binary invariance of the code
  when the code is interrupted and resumed from restart files"* **[V-src,
  `documentation/04_config-options.md:1573-1580`]**.
- **SWIFT states the opposite of reproducibility** **[V-paper, arXiv:2305.13380]**: *"one possible
  drawback to the task-based approach … is the **lack of determinism** … **even on the same hardware
  and with the exact same executable** … will lead to slightly different results each time."*
- **PKDGRAV3, HACC, RAMSES**: zero occurrences of `reproducib*`/`determinis*`/`bitwise` **[V-src]**.
- **But GAMER-2 does claim it** — Schive et al., MNRAS 481, 4815 (2018), arXiv:1712.07070
  **[V-paper]**: *"gamer-2 is fully bitwise reproducible"* under *"different numbers of MPI processes
  and OpenMP threads"*. The mechanism is **sorting**, not fixed point: *"Sort particles in the same
  patches by their spatial coordinates before mass deposition"* — and they admit the cost: *"it can
  also **deteriorate performance, especially for simulations requiring a large number of particles
  due to the extra particle sorting**. Therefore, we implement this feature as a compile-time option,
  and **disable it throughout Section 4**."*
- **QUOKKA** achieves it only *probabilistically*, via error-bound tracking + mantissa truncation,
  and names superaccumulators as future work (quoted above).

**Honest framing, therefore:** bitwise-reproducible CIC density is **not novel as a goal** — GAMER-2
claims it and QUOKKA approximates it. What is novel is the *mechanism* in this field: **fixed-point
CIC mass deposition in a cosmological PM code was not found in any source consulted**, and no PIC
code does it either (searched exhaustively; see §11). The defensible claim is *"first application of
a known technique (Kulisch 1986; Demmel & Nguyen 2013; OpenMM 2012; HTMPC 2025) to cosmological
CIC"*, and the performance baseline to beat is **GAMER-2's sort** and **QUOKKA's truncation** — both
of which cost something, whereas §8 measures fixed point as *faster* than the float it replaces.

One trap: a fixed-point deposit alone does not make the field **rank**-invariant unless the
halo/ghost exchange accumulation is fixed point too.

### 5.6b The graphics/MPM school — and why its headline numbers do not transfer

MPM/FLIP particle-to-grid transfer is algorithmically identical to CIC, and that community has the
most aggressive published optimizations. The headline numbers are large but were measured against
baselines LUNA does not share.

- **Gao, Wang, Wu, Pradhana, Sifakis, Yuksel & Jiang, "GPU Optimization of Material Point Methods",
  SIGGRAPH Asia 2018, ACM TOG 37(6)** **[V-paper]**. Stack: 4×4×4 sparse grid blocks, a ∆x/2
  *staggering* so each block's particles touch only 2×2×2 blocks; **cell-based sorting every step**
  using a histogram sort ("speedup approaches an order of magnitude" over radix); **warp-level
  segmented reduction** (`ballot`+`brev`, then `shfl_down` within runs of same-cell particles) so
  "all warp-scale write conflicts are eliminated"; LDS staging of a block plus its 7 neighbours.
  Result: **P2G-MLS 23× faster than atomic-only scattering**, dropping P2G from >90% of solver time
  to ~40%. Also **distribution-independent** (their §4.2).
- **Wang, Qiu, Slattery et al., SIGGRAPH 2020, ACM TOG 39(4)** **[V-paper]**: **G2P2G fusion**
  (1.37–1.39×) and an **AoSoA** particle layout — bins of 32, SoA within a bin, AoS across bins —
  worth **≈3×** on the transfer kernel.
- **CK-MPM (arXiv:2412.10399)** **[V-paper]** reduces the transfer stencil 27→16 nodes for 1.5×
  in an untuned pipeline but only ~10% in tuned CUDA. **CIC already uses the minimal 8-node stencil,
  so this entire avenue is closed for LUNA.**

**Why the 23× does not transfer** — the MPM baselines are a 27-node quadratic stencil, ≥8
particles/cell, and (in the older work) emulated float atomics. LUNA has an 8-node stencil, ~1
particle/cell, and native FP64 atomics (§5.1). **[INF, but well supported by §5.2's WarpX result and
the SC20 datum below.]**

**The most calibrating counter-datum.** Henriksen, Hellfritzsch, Sadayappan & Oancea, *"Compiling
Generalized Histograms for GPU"*, **SC20** **[V-paper]** — their MD case study (4.1M interactions ×
6 atomic float adds) is structurally a deposit. **With native hardware float atomics, hand-tuned CUDA
was 333 µs vs their tuned privatization at 362 µs — 0.91×, a slowdown.** With CAS-emulated atomics
the same privatization won 2.65×. They also compared against sorting: `CUB SortKeys + ReduceByKey`
on 50M elements lost to plain atomics by **1.6–8.1×**, with the comparison *favouring* CUB.

**This is the single most important calibration in the whole survey**: on hardware with native float
atomics — which MI300A has — privatization and sort-then-reduce have both been measured *losing* to
plain atomics on a deposit-shaped kernel. Every large speedup in this literature should be checked
for which baseline it used.

**Two hardware findings worth knowing** from *"Are Your GPU Atomics Secretly Contending?"* (PLOS'25
@ SOSP) **[V-paper; caveat: RDNA2/3 + Ada only, no CDNA — so these are indicative, not verified for
MI300A]**:

- **Hardware already coalesces up to 32 same-address atomic requests for free**, in the atomic unit
  rather than the core — so warp-aggregation of *identical* addresses is largely redundant.
- **The atomic unit parallelizes a warp's accesses when they hit different, densely adjacent
  addresses (stride 4–8 B) — faster per thread than a single address.** That is good news for CIC's
  8 distinct corners, and an argument for fusing the corners (R1).
- **AMD power-of-two pathology:** contention *rises again* at 4096 B stride, apparently because the
  atomic unit is selected from the low 12 address bits. **Avoid power-of-two strides between any
  privatized sub-grids.**

**One technique here is genuinely applicable and cheap** — Vincenti, Lobet, Lehe, Sasanka & Vay,
*Comput. Phys. Commun.* 210 (2017), arXiv:1601.02056 **[V-paper]**, the `rhocell` layout, measured on
a literal 3D 8-vertex CIC deposit: store the **8 vertices of a cell contiguously** as
`rhocells(1:8, N_cells)`, 64-byte aligned, so deposition is conflict-free within a lane group; then
reduce with a **dense 8-shift stencil that uses no atomics at all**:

```fortran
moff = (0, 1, nnx, nnx+1, nnxy, nnxy+1, nnxy+nnx, nnxy+nnx+1)
rho(igrid + moff(nv)) += rhocells(nv, ic)      ! nv = 1..8, O(N_cells) not O(N_p)
```

They report particle tiling at **8³–10³ cells** raising cache reuse 85%→99% for a **×3 speed-up of
the whole PIC code with randomly distributed particles**, and **×2–2.5** on SIMD deposition in double
precision. The reduction half is expressible in pure JAX as a fixed 8-shift stencil; the accumulate
half still needs the sort.

### 5.7 The 8-colour CIC idea — does not work, and LUNA already disproves it

The proposal was: colour the 8 CIC corners and do 8 conflict-free passes.

**In 3D CIC each particle writes 8 *distinct* addresses, so a particle never conflicts with itself.**
All contention is *inter-particle, same-cell*. Colouring by stencil corner removes a conflict class
that does not exist. **LUNA's code already does the 8-pass corner split**
(`particle_mesh_blocks.py:209-216`) and the atomics remain — that is direct empirical refutation.

What the literature actually colours is **particle bins/clusters**, not stencil corners — e.g.
PIConGPU's `StridedCachedSupercells`, a checkerboard over supercells enabling a non-atomic flush,
with `skipSuperCells = ceil(MaxMargin / min(SuperCellSize))` **[V-src]**. Note PIConGPU makes this
the **CPU** default and uses atomic flush on both CUDA and HIP, because occupancy wins.

---

## 6. Numerical-analysis toolkit: what else is available

- **Stochastic rounding.** Connolly, Higham & Mary, *SIAM J. Sci. Comput.* 42(3) (2021); Croci,
  Fasi, Higham, Mary & Mikaitis, *R. Soc. Open Sci.* 9:211631 (2022). The relevant property is that
  SR **avoids stagnation** — Higham & Mary 2022 §1.2.3 **[V-paper]**: *"If h_i is of absolute value
  less than half the spacing of the floating-point numbers around φ⁽ⁱ⁾ … then φ⁽ⁱ⁾+h_i rounds to
  φ⁽ⁱ⁾ … **This phenomenon is called stagnation.** It can be avoided by using stochastic rounding in
  place of round to nearest."* Not natively available in XLA and it would require an RNG draw per
  add. **Relevance to LUNA is low** because fixed-point accumulation (§9, R3) solves the same
  problem deterministically and for free.
- **Reproducible summation.** Demmel & Nguyen (ReproBLAS); Collange, Defour, Graillat & Iakymchuk
  (ExBLAS, Kulisch long accumulators). These give order-independent FP summation at a cost. **A
  fixed-point accumulator is the degenerate, free case of a Kulisch accumulator** when the dynamic
  range is known a priori — which for a mass density field it is. **[INF]**
- **Compensated (Kahan/Neumaier) summation.** Abacus explicitly names Kahan as the fix for its
  remaining near-field error mode, as *future work* **[V-paper §3.5]**: *"a symptom of loss of
  precision in the accumulation of accelerations onto a large accumulator. This could be mitigated
  in future versions of Abacus with Kahan summation…"* Cannot be applied inside an atomic scatter
  (the compensation term is not associative across threads) but **is** applicable to LUNA's global
  reductions — residual norms, mass/momentum conservation checks.
- **Ginkgo's memory accessor** — Grützmacher & Anzt, Euro-Par 2018 Workshops,
  `10.1007/978-3-030-10549-5_34`; Grützmacher et al., *Concurrency Computat.* 32:e5418 (2020),
  `10.1002/cpe.5418`. Design premise, verbatim **[V-paper]**: *"The arithmetic operations can still
  use high precision without impacting the performance **as long as the algorithm remains memory
  bandwidth bound**."* Measured on V100: 32-bit access to a 2-segment split is ~2× faster than FP64;
  format-flag overhead 12–25%; 16-bit access reached only 700 GB/s vs ~800 for 32/64-bit. Their
  `scaled_reduced_row_major.hpp` (narrow storage + per-row scale factor) is independently the same
  idea as QUDA's per-site max.
- **Adaptive precision block-Jacobi** — Anzt, Dongarra, Flegar, Higham & Quintana-Ortí,
  *Concurrency Computat.* 31(6):e4460 (2019). Blunt conclusion **[V-paper]**: *"the reckless use of
  fp16 to store a block-Jacobi preconditioner **fails in most cases** and is therefore not
  recommended."*

### 6.1 Precedents from the JAX scientific ecosystem

Four patterns worth copying, all verified in source **[V-src]**:

- **Veros** (JAX ocean model) — `--float-type` is a *runtime* switch, but **the elliptic solver is
  precision-gated**: the on-device Krylov path is selected only for `float64 and gpu`, and the
  fallback force-promotes RHS and matrix to FP64 (`external/solvers/scipy.py:48,126`). **Even in FP32
  mode the Poisson-like solve stays FP64.** That is a direct precedent for LUNA's proposed
  `gravity_precision`: reduce precision on the bulk particle/field update, keep the elliptic solve
  high. They report *"about 70% higher overall model performance"* in FP32 (JAMES 2021,
  `10.1029/2021MS002717`).
- **JAX-MD** — a single `util.high_precision_sum` helper that reduces in FP64 and casts back to the
  input dtype, used at ~30 sites. The cheap, disciplined version of R8.
- **JAX-Fluids** — a **dtype-keyed epsilon table** (`config/precision.py`) rather than magic
  constants: separate `eps`, `smallest_normal`, `spatial_stencil_eps` and per-limiter epsilons, at
  `1e-6` in FP32 vs `1e-10`–`1e-12` in FP64. Also the same idiom in JAX-CFD (`10 * jnp.finfo(dtype).eps`)
  and Diffrax (`10 ** (2 - jnp.finfo(dtype).precision)`). **If LUNA gains a precision switch, every
  hard-coded tolerance in the gravity path needs to become dtype-derived** — otherwise the switch
  silently changes convergence behaviour.
- **JAX-Fluids' permutation-symmetric summation** (`math/sum_consistent.py`, opt-in): for the three
  directional flux contributions it evaluates all rotations of the sum and returns `0.5*(max+min)`.
  The only FP-associativity-aware summation machinery found in any JAX scientific code, and a
  stealable idea for direction-split operators.

**Also worth knowing:** no widely-used multi-GPU native-JAX geometric multigrid appears to exist — the
three JAX multigrid projects found are either PyAMG/AmgX wrappers or single-device, and all structure
the V-cycle as trace-time Python recursion. LUNA's `fori_loop` V-cycle over levels (commit `ab72cde`)
appears to be unusual in this landscape. **[INF — the survey was not exhaustive.]**

---

## 7. Half precision (FP16 / BF16) — a well-evidenced negative

### 7.1 Nobody computes gravity in FP16 or BF16

A sweep of 27 astrophysics repos for `__fp16 _Float16 float16 fp16 bfloat16 bf16 __half` plus
tensor-core primitives (`wmma`, `mma_sync`, `__nv_bfloat`, `dp4a`, `hgemm`) returned **zero
tensor-core / low-precision-matmul hits anywhere** **[V-src]**. Clean negatives (no hits of any
kind): gadget4, ramses, changa, gizmo, enzo, enzo-e, nyx, athenak, gamer, amrex, jaxpm, pmwd,
flowpm, jax_cosmo, exafmm, CUBE, monofonic.

Every half-precision hit that exists is an **I/O format**:

| Code | Hit | What it actually is |
|---|---|---|
| **GADGET-4** | `src/half/half.hpp`; used in `src/io/io.cc:43-44`, `src/io/hdf5_util.cc:26-27,125-132`, `src/main/main.cc:25` **only** | HDF5 snapshot storage. `FILE_HALF` is attached to exactly two fields: `Velocities` and `Acceleration` (`src/io/snap_io.cc:70,103`), both `MEM_MY_FLOAT` in memory. **Zero occurrences in `src/gravity`, `src/gravtree`, `src/fmm`, `src/pm`, `src/time_integration`.** **[V-src, verified by grep of the whole tree]** |
| SWIFT | `src/io_compression.c:40` `"HalfFloat","BFloat16"` | HDF5 lossy filters; defaults to BF16 only for `BirthDensities` and `LastAGNFeedbackScaleFactors` |
| PKDGRAV3 | `io/fio.c:2487-2501` `getFP16LE()` | HDF5 output for SPH abundances/metallicity |
| Athena++ | `src/athena.hpp:21-29` `fp16_t` | HDF5 writer only |

**This is the strongest single finding of the half-precision section.** GADGET-4 — arguably the most
carefully engineered code in the field — went to the trouble of vendoring a `half` library, and used
it **only to make snapshots smaller**. It stores *positions* as integers and *computes* in
float/double. The field's considered answer to "what is below FP32" is **fixed-point integers, not
IEEE half**.

Literature searches were equally empty **[V]**: arXiv `all:"half precision" AND all:"N-body"` → 0;
`… AND all:cosmological` → 0; `all:bfloat16 AND all:cosmological` → 0; `cat:astro-ph.IM AND all:FP16`
→ 0; `all:"tensor core" AND all:"N-body"` → 0.

Two near-misses that are *not* FP16: **Fugaku's Gordon-Bell Vlasov+N-body** (arXiv:2110.15867)
reports "mixed precision" meaning FP32/FP64 — *"the positions and velocities of the N-body particles
are represented by double precision"* — despite A64FX having native FP16 SVE **[V-paper]**. And a
Tenstorrent N-body port (arXiv:2509.19294) had BF16 hardware available and **chose FP32** **[V-paper]**.

> One paper named in the original brief, "Yoshikawa & Tanikawa, *Accelerating Cosmological N-body
> Simulations with Tensor Cores*", **could not be verified to exist**. Most likely a conflation with
> Ootomo & Yokota, arXiv:2203.03341 (FP16 tensor cores with error correction to recover FP32
> accuracy — applied to FFT/QR/MD, not gravity). Flagged rather than cited.

### 7.2 Format parameters, and where LUNA's quantities land

| | layout | eps | max finite | min normal | dec. digits |
|---|---|---|---|---|---|
| **FP16** | 1+5+10 | 9.7656e-4 | **65504** | 6.1035e-5 | 3.31 |
| **BF16** | 1+8+7 | 7.8125e-3 | 3.3895e38 | 1.1755e-38 | 2.41 |
| FP32 | 1+8+23 | 1.1921e-7 | 3.4028e38 | 1.1755e-38 | 7.22 |
| FP64 | 1+11+52 | 2.2204e-16 | 1.7977e308 | 2.2251e-308 | 15.95 |

**Absolute positions — IMPOSSIBLE in both.** ULP at value x is `2^(floor(log2 x) - m)`. At
x = 25000 kpc/h (binade [2¹⁴,2¹⁵)):

| format | ULP at 25000 | as a fraction of a 512³ cell (dx = 48.83) | of a 1024³ cell |
|---|---|---|---|
| **FP16** | **16 kpc/h** | 0.33 | **0.66** |
| **BF16** | **128 kpc/h** | 2.6 | **5.2** |
| FP32 | 1.953e-3 kpc/h | 4.0e-5 | 8.0e-5 |
| **uint32** | **5.82e-6 kpc/h** (uniform) | **1.2e-7** | 2.4e-7 |

The FP16 neighbours bracketing 25000 are **24992 and 25008**. At 1024³ you cannot determine which
CIC cell a particle occupies. Note also that uint32 is **335× finer than FP32 at the box corner**
and, unlike FP32, uniform everywhere.

**Density accumulator — IMPOSSIBLE.** `S + w` becomes a silent no-op once `S/w > 2^p`. Measured by
exact-rounding emulation of summing n unit contributions:

| n | FP16 sum | BF16 sum | FP32 sum |
|---|---|---|---|
| 256 | 256 | 256 | 256 |
| 1024 | 1024 | **256** | 1024 |
| 4096 | **2048** | **256** | 4096 |
| 65536 | **2048** | **256** | 65536 |

**In BF16 the density field saturates after 256 contributions per cell; in FP16 after 2048.** Halo
centres routinely receive 10³–10⁶. This is Higham & Mary's *stagnation*, and it is exactly the
mechanism by which BF16 failed in Klöwer, Düben & Palmer's shallow-water model **[V-paper,
JAMES 12, e2020MS002246]**: *"BFloat16 arithmetic is not able to simulate the shallow water
dynamics, as **tendencies are too small to be added to the prognostic variables**."*

**Potential φ — BF16 impossible, FP16 marginal.** Forces come from differencing φ, so the relative
force error from storing φ at unit roundoff eps is ≈ eps/(kh). For the box-scale mode kh = 2π/N:

| N | FP16 rel. force error | BF16 rel. force error |
|---|---|---|
| 256 | 4.0% | 32% |
| **1024** | **16%** | **127%** |

**[INF, standard cancellation argument]** Promoting to FP32 before differencing does not help — the
information is already gone.

**Density contrast δ:** FP16 **overflows** (max 65504) in collapsed regions. BF16 has range but its
0.8% relative error is injected straight into the Poisson RHS — a hard floor no V-cycle removes.

**CIC weights in [0,1]:** the one quantity that survives. FP16 partition-of-unity error measured at
rms 3.28e-4, max 1.25e-3. Tolerable in isolation — but the accumulator is the problem, not the weight.

### 7.3 The one supported use: FP16 inside iterative refinement

**Oo & Vogel, arXiv:2007.07539** is the only paper found doing FP16 for *geometric multigrid on
Poisson* **[V-paper]**. V100, 2D grids to 6145², 3D to 289³:

> "While the final solution is always computed with high-precision accuracy, an iterative refinement
> approach with multigrid preconditioning in lower precision **and residuum scaling** is employed…
> the solver runtime is considerably decreased due to the reduced memory transfer and a **speedup of
> up to 2.5×** is gained for the overall solver."

Their Algorithm 2, with the mandatory scaling on line 4:

```
r_high := b_high − A_high u_high
while α := ‖r_high‖ > ε:
    r_low  := (1/α) r_high              # RESCALE before the cast — mandatory
    approximate A_low c_low = r_low by ONE multigrid cycle
    u_high := u_high + α c_low          # correction applied HIGH
    r_high := r_high − α A_high c_low   # residual computed HIGH
```

Why the rescale is mandatory, verbatim: *"**the correction equation can therefore not be solved if
the high-precision residuum is flushed to zero in a low-precision format.**"*

**Two caveats that cap the payoff for LUNA:**

1. **The 2.5× is against pure FP64 MG** (4× the bytes). **LUNA's `mixed` mode already banks the FP32
   version of that.** Against an FP32 V-cycle the remaining ceiling is **2×, on the correction solve
   only**, times the MG fraction of the step.
2. Oo & Vogel's own caveat: *"The optimal speedup is only approached for very large degree of
   freedom numbers"* — FP16 throughput does not saturate at small array sizes, and coarse MG levels
   are small. This fights directly against the progressive-precision result (coarse grids tolerate
   *less* precision but are also where FP16 helps *least*).

The dense-LU FP16 results (Haidar, Tomov, Dongarra & Higham, SC18; 3–4× vs `dgesv`) are **not
transferable** — that is a compute-bound O(n³) kernel on tensor cores. And even there, FP16
iterative refinement **diverges** on 2 of their 6 matrix types **[V-paper, SC18 Fig. 7d]**.

Documented FP16 failure modes: overflow is **unrecoverable** (Higham, Pranesh & Zounon,
*SIAM J. Sci. Comput.* 41(4), A2536–A2551, 2019 — *"LU factorization cannot produce useful results
for a matrix with infinities amongst its entries"*); underflow makes nonsingular matrices singular;
and their measured fix is two-sided diagonal scaling, which cut max GMRES iterations from **686 to
22** on their test set **[V-paper]**.

### 7.4 Hardware reality: for LUNA's kernels, half buys nothing

Peak rates from vendor data sheets **[V-src]**:

| | MI300A (CDNA3) | GH200 (Hopper) |
|---|---|---|
| FP64 vector | 61.3 TF/s | 34 TF/s |
| FP64 matrix/tensor | 122.6 TF/s | 67 TF/s |
| FP32 vector | 122.6 TF/s | 67 TF/s |
| FP16/BF16 **matrix** | 980.6 TF/s | 990 TF/s |
| HBM bandwidth | **5.3 TB/s** | 4–4.9 TB/s |

**AMD publishes no vector FP16 rate for CDNA3** — the 980.6 TF is **matrix-core only**, reachable
only by a dense GEMM. **[V-src, CDNA3 whitepaper Table 1 lists Vector FP64/FP32 but only Matrix
FP16/BF16.]**

A 7-point damped-Jacobi smoother update is ~8 FLOP against ~3 array touches:

| storage | arithmetic intensity | MI300A FP64-vector machine balance |
|---|---|---|
| FP64 | 0.33 FLOP/B | 11.57 FLOP/B |
| FP32 | 0.67 FLOP/B | " |
| FP16 | 1.33 FLOP/B | " |

**Even in FP16 the smoother sits ~9× below the FP64-*vector* balance and ~140× below the FP16-matrix
balance.** LUNA is nowhere near any arithmetic roofline in any format. **The only thing a narrower
format buys is bytes moved — 2× at best from FP32.** This is precisely Oo & Vogel's own roofline
finding and Grützmacher et al.'s design premise.

**And on the atomic scatter path, half buys literally nothing.** The CDNA3 ISA has **no scalar
`GLOBAL_ATOMIC_ADD_F16`** — only `GLOBAL_ATOMIC_PK_ADD_F16/BF16`, which operate on a 32-bit word
holding two adjacent halves **[V-src, MI300 ISA §12.15 p.505]**. XLA's ROCm backend handles a scalar
half atomic by **aligning the address down to 4 bytes, zero-padding the other half, and issuing one
full 32-bit packed atomic** **[V-src, `openxla/xla` `lower_tensors.cc`, `emitAMDAtomicFAdd`
lines 1044-1113]**:

```cpp
Value addr_masked = And(addr_int, -4);              // align DOWN to 4 bytes
Value offset      = And(Trunc(addr_int,i32), 2);
Value modifier_masked = Shl(ZExt(Bitcast(modifier_arg,i16),i32), offset*8);
AtomicRMWOp(fadd, IntToPtr(addr_masked), ..., "agent-one-as");
```

Gates `has_packed_fp16_atomics_support()` / `has_packed_bf16_atomics_support()` both pass on gfx942
**[V-src, `rocm_compute_capability.h:176-182`]** — so it works, but: **same transaction count, same
32 bits on the wire, as an FP32 atomic.** A half-precision deposit is pure downside.

### 7.5 Where half precision *is* plausible: transport, not arithmetic

This is the one genuinely open opportunity, and it is well evidenced:

- **Klöwer, Düben & Palmer 2020** **[V-paper]**: *"**16- or even 8-bit communication between
  processors introduces negligible errors**, providing a perspective for reduced data
  communication."* Applied to prognostic variables at every RK substep; *"Reduced precision
  communication was not found to have a significant impact on either mean state, variability,
  geostrophy, or tendencies"*, with errors ~2 orders of magnitude below those from 16-bit *arithmetic*.
- **Kormann, Reuter & Rampp** (MPCDF/IPP), arXiv:1903.00308, *IJHPCA* — 6D Vlasov–Poisson,
  §5.4.5 **[V-paper]**: FP32 halos *"halve the communication volume and time… leading to an improved
  parallel scalability especially in inter-island scenarios"*, but flagged as *"an experimental
  tool"* requiring validation. With **ZFP** at single-precision fidelity: *"a compression factor of
  ~4 for the halo data"* and *"virtually flat scaling with nearly 100% parallel efficiency, even when
  crossing island barriers."*

**But note the honest downside from the same Klöwer paper:** 16-bit arithmetic left the mean state
intact while **increasing flow variability by 10–30%**. For a code whose observables are power
spectra and halo statistics, the variance *is* the answer.

### 7.6 Half-precision verdict for LUNA

**Not worth it in the solver. Here is why, in one paragraph.** Absolute positions, the density
accumulator, and the carried potential are all arithmetically excluded (§7.2) — these are not
matters of taste but of ULP and saturation counts. The only supported use, FP16 inside iterative
refinement, is capped at 2× on the correction solve *relative to LUNA's existing FP32 mixed mode*,
requires mandatory residual rescaling, and has published failure modes. On the deposit — LUNA's
actual bottleneck — CDNA3 has no scalar half atomic and XLA promotes to a 32-bit packed atomic, so
the gain is exactly zero. And the field's own answer to "what is below FP32", arrived at
independently by GADGET-4, PKDGRAV3, Abacus, CUBE and QUDA, is **fixed-point integers**, not IEEE
half. The one place half precision might earn its keep is **halo/inter-GPU transport**, where the
saving is bytes on the wire and the literature (Klöwer; Kormann) supports it — that is a
communication optimization, not a precision policy, and it should be evaluated separately.

---

## 8. Measured results from this survey

All on one MI300A (viper `apudev`), 512³ mesh, 134,217,728 particles, JAX 0.10.2 / ROCm 7.2,
`jax_enable_x64=1`. Scripts: `/viper/ptmp2/uli/precision_bench/{bench_scatter_dtype,bench_reproducibility,bench_followup}.py`.
These are 8-corner `.at[].add()` kernels standing in for LUNA's deposit — same structure, no
sharding. **[V-meas]**

### 8.1 Accumulator dtype and particle distribution

Best of 5, `min` time:

| distribution | float64 | float32 | **int32 (scale 2¹⁸)** | int64 (scale 2⁴⁰) |
|---|---|---|---|---|
| perfect cell order | 16.01 ms | 13.45 ms | **13.46 ms** | 16.02 ms |
| uniform random | 55.21 ms | 50.92 ms | **47.21 ms** | 55.11 ms |
| clustered | 47.59 ms | 48.65 ms | **42.69 ms** | 47.40 ms |

Accuracy vs the FP64 result (max |diff| on a field whose max is 159):

| | float32 | int32 (2¹⁸) | int64 (2⁴⁰) |
|---|---|---|---|
| uniform | 1.215e-6 | 2.168e-5 | **9.095e-13** |
| clustered | 1.947e-4 | 1.481e-4 | **1.819e-12** |

**Four conclusions.**

1. **Particle ordering dominates everything**: 13.5 ms cell-ordered vs 47–55 ms scattered, a **3.5×
   swing**, larger than any dtype effect. This corroborates LUNA's own 12.6 vs 50.4 ms measurement
   and WarpX's 7.5×.
2. **int32 accumulation is the fastest option in every distribution**, and beats float32 by 1.08×
   (uniform) to 1.14× (clustered).
3. **Width, not type, sets the cost**: int64 ≈ float64 to within 0.2% in all three cases. 8-byte
   atomics cost the same regardless of interpretation — consistent with the ISA finding (§5.1).
4. **int64 fixed point at scale 2⁴⁰ is ~10⁸× more accurate than float32** and essentially exact,
   at exactly float64 cost.

### 8.2 Bitwise reproducibility under particle reordering

Same kernel, three independent random permutations of the particle array, comparing the resulting
density grids bit-for-bit:

| accumulator | permutation 1 | 2 | 3 |
|---|---|---|---|
| float64 | **DIFF (1 cell)** | same | **DIFF (1 cell)** |
| float32 | **DIFF (9,424,276 cells)** | **DIFF (9,425,270)** | **DIFF (9,425,187)** |
| **int32 (2¹⁸)** | **same** | **same** | **same** |

And, with 8-corner weights and an exact-mass variant (last corner absorbs the rounding residual so
the 8 integer weights sum exactly to `round(w·scale)`):

| accumulator | bitwise identical under reorder | relative mass error |
|---|---|---|
| float64 | no (1 cell differs) | 2.031e-12 |
| float32 | no (9.4M cells differ) | 3.816e-10 |
| int32 (2¹⁸) | **YES** | 3.546e-10 |
| **int32, exact-mass** | **YES** | **0.000e+00** |
| int64 (2⁴⁰) | **YES** | 2.031e-12 |
| **int64, exact-mass** | **YES** | **0.000e+00** |

**This is the headline result of the survey.** Float32 deposit differs in **7% of all cells** under a
mere particle reordering. Float64 differs in a single cell — better, but *not* guaranteed, since FP
addition is non-associative; the near-reproducibility is luck, not a property. Integer accumulation
is bitwise identical **by construction**, because integer addition is associative. And the
exact-mass variant conserves mass to the **last bit**, which no floating-point deposit can do.

### 8.3 Integer positions cost nothing

GADGET-4-style uint32 positions, cell index by `>> 23` and sub-cell weight from `& (2²³−1)`,
vs float32 positions:

| | time | max density |
|---|---|---|
| f32 positions, f32 accumulator | 47.28 ms | 223.8 |
| **u32 positions, f32 accumulator** | **46.33 ms** | 223.8 |
| f32 positions, i32 accumulator | 41.59 ms | 223.8 |
| **u32 positions, i32 accumulator** | **40.71 ms** | 223.8 |

Integer positions are **not slower** — marginally faster (~2%), presumably because the shift/mask
replaces a divide and a floor. Combined with int32 accumulation, **40.71 ms vs 47.28 ms baseline =
1.16×**, while simultaneously halving position bytes (12 vs 24 B/particle) and gaining bitwise
reproducibility and exact periodic wrapping.

### 8.4 Choosing the fixed-point scale

int32, clustered field whose FP64 max cell density is 223.8:

| scale | max abs. error | overflows at ρ = | headroom |
|---|---|---|---|
| 2¹⁴ | 2.169e-3 | 1.31e5 | 586× |
| 2¹⁶ | 5.557e-4 | 3.28e4 | 146× |
| **2¹⁸** | **1.603e-4** | **8192** | **37×** |
| 2²⁰ | 3.870e-5 | 2048 | 9× — tight |
| 2²² | 8.720e-6 | 512 | 2× — unsafe |

Error scales exactly as 1/scale, as expected for uniform quantization. **int32 at 2¹⁶–2¹⁸ is the
usable window**; below 2¹⁴ accuracy falls behind float32, above 2²⁰ overflow becomes a real risk at
z = 0 where δ can reach 10³–10⁴. **int64 at 2⁴⁰ removes the tradeoff entirely** (headroom to
ρ ≈ 8.4e6, error ~1e-12) at float64 cost — so **int64 is the drop-in replacement for the FP64
deposit, and int32 for the FP32 deposit.**

---

## 9. Ranked recommendations

Each tagged: **[speed]** / **[accuracy]** / **[bytes]** / **[reproducibility]**; JAX difficulty; and
whether it targets the **atomics/scatter-bound** deposit or the **bandwidth-bound** V-cycle.

### R1 — Fix the deposit's dtype casting and fuse the 8 scatters
**VERIFIED-IN-LITERATURE (standard) · [speed] [bytes] · difficulty: LOW · targets: both**

Cast `pos_l` to the working dtype *before* computing `f`, `i0`, `t`
(`particle_mesh_blocks.py:188-206`), and compute indices/weights once rather than eight times.
Currently the FP32 path still streams 3.2 GB of FP64 positions per deposit and recomputes the index
arithmetic 8×.

Benefit: halves position traffic in the FP32 path; removes 7 redundant reads of the position array.
Risk: the `t` values must retain enough precision — compute `f`/`i0`/`t` in FP32 from FP32
positions is fine (FP32 resolves 2⁻²⁴ of the box ≈ 1.5e-3 kpc/h, well below a cell), but verify
against an FP64 reference. **This must be done before any further precision experiment, because
until it is, no measurement isolates the atomic path.**

### R1b — `indices_are_sorted=True`: TEST IT, do not assume it helps (it may be ~95× worse)
**CONTESTED · [speed] · difficulty: LOW to test · targets: the atomics-bound deposit**

> **Two independent source reads of the same XLA file reached opposite conclusions. Recorded as a
> conflict rather than resolved, because the downside case is severe.**

**The optimistic reading.** XLA has a **warp-accumulating scatter emitter** that keeps a per-warp
register accumulator and only issues a global atomic when the target offset *changes*, collapsing
runs of same-cell particles before they touch memory. It triggers on a hint LUNA does not currently
pass **[V-src, `xla/backends/gpu/codegen/emitters/scatter.cc:964`]**:

```cpp
if (description.scatter->indices_are_sorted() &&
    !description.scatter->unique_indices() && num_slices > max_active_warps) {
  ... return std::make_unique<ScatterWithDistributedIndices>(...);
}
```

LUNA already sorts: `resort_blocks` (`particle_mesh_blocks.py:306-330`) does a stable argsort on the
row-major ghosted cell key `((idx0*(nloc1+2) + idx1)*(nloc2+2) + idx2)`. **Grep confirms
`indices_are_sorted` appears nowhere in the repo** — the eight deposits at `:215` use the defaults
`indices_are_sorted=False, unique_indices=False` **[V-src]**.

The deposit's scatter index for corner `(di,dj,dk)` is `((il0+di)*(nloc1+2) + il1+dj)*(nloc2+2) +
il2+dk`, which is monotone non-decreasing in the same key for each fixed corner — so **for the
active prefix the promise is genuinely true, for all eight corners.**

**The caveat — and it is a real one.** The degenerate-padding "spread" (`:199-203`) sends inactive
slots to `(sid % nloc0, (sid//nloc0) % nloc1, ...)`. That is x-fastest, whereas the sort key is
z-fastest, and the inactive slots sit at the *tail* (key = `INT32_MAX`) while their spread indices
restart from 0. **So the index array is not globally sorted, and `indices_are_sorted=True` would be a
false promise.** XLA's emitter still flushes atomically whenever the offset changes, so violating it
would most likely remain numerically correct — but HLO semantics call this implementation-defined,
and relying on it is exactly the kind of thing that breaks silently on an XLA upgrade.

Options, in order of preference: (a) deposit the inactive tail into a separate throwaway slab so the
real scatter is over the sorted prefix only; (b) reorder the spread to z-fastest *and* offset it past
the active range so global monotonicity holds; (c) leave it alone. Do **not** simply set the flag.

**The pessimistic reading — and why this is filed as CONTESTED rather than recommended.** A second
independent read of the same emitter reports two things that point the other way **[V-src]**:

- For a **scalar-slice** scatter — which is exactly CIC, one value per index — the indexing map at
  `scatter.cc:646-680` forces `lane_id == 0`, i.e. **1 of 64 lanes does the work**. **[INF from the
  indexing map; not measured.]**
- **openxla/xla#22233**, filed by an XLA codegen maintainer and **still open**, reports the sorted
  path at *"38 ms vs 0.4 ms for the unsorted one"* — a ~95× regression, traced to accumulator
  register spill.

The optimistic gain estimate is `ceil(8·N_part/N_cells)` accumulated updates per warp — ~8 at LUNA's
~1 particle/cell. **A best case of ~8× on paper against a documented ~95× regression is not a
favourable bet.**

**Action: treat this purely as a cheap experiment, not a recommendation.** Set the flag on a scratch
copy of the deposit (where correctness does not matter, since the promise is currently false anyway),
time it, and keep it only if it wins. Budget an hour, expect it to lose.

**Related, and unambiguous — do NOT set `unique_indices=True`.** CIC indices are *not* unique (that
is the entire point of a deposit), and JAX documents that *"if the updated elements overlap when
`unique_indices` is True the behavior is undefined"*; the emitter drops to a non-atomic
load/add/store, which **silently loses mass** **[V-src]**. That collides directly with LUNA's standing
rule about asserting mass conservation.

### R2 — Keep/strengthen cell-ordered particle storage
**VERIFIED-IN-LITERATURE · [speed] · difficulty: LOW–MEDIUM (LUNA has a cell-order resort) · targets: atomics + cache**

The single largest effect measured: **3.5×** (13.5 vs 47 ms) between cell-ordered and scattered
particles; LUNA's own measurement is 4× (12.6 vs 50.4 ms); WarpX reports 7.5× including sort cost.
WarpX sorts by cell every 4 steps on GPU by default; Abacus and CUBE keep particles permanently
cell-ordered.

Actions: confirm LUNA re-sorts often enough at late times (particles decorrelate as structure forms);
consider WarpX's *binning* (permutation index only) if a full reorder is too expensive; measure the
sort interval as a tunable. Note WarpX's finding that the mechanism is **cache reuse**, so the win
should show up as reduced HBM traffic, not just reduced atomic conflict.

**Caveat — the sort is not free, and can cost more than it saves.** RAMSES-GPU reports, on Apple
Metal **[V-paper, arXiv:2607.20704]**: total *"21 ns per particle … The mass deposition itself is
only **7 ns**, while the **particle Hilbert key sorting costs 14 ns**."* GAMER-2, which sorts
specifically for reproducibility, admits it *"can also deteriorate performance, especially for
simulations requiring a large number of particles due to the extra particle sorting"* and disables it
in its own benchmarks **[V-paper]**. Decyk & Singh measured reordering at **15–30% of total runtime**
**[V-paper]**. **So the right variable is the sort *interval*, not whether to sort** — LUNA benefits
from ordering it already has, and the question is how often to restore it. Measure, do not assume.

### R2b — Particle memory layout, and cheapening the resort itself
**VERIFIED-IN-LITERATURE · [speed] · difficulty: LOW–MEDIUM · targets: cache, not atomics**

**Taichi's Table 2 is the closest published analogue to LUNA's measured 3.5× ordering swing** (Hu,
Li, Anderson, Ragan-Kelley & Durand, SIGGRAPH Asia 2019, ACM TOG 38(6)) **[V-paper]**, MLS-MPM:

| particle layout | ordered | randomly shuffled | penalty |
|---|---|---|---|
| **SoA** | 3.52 ms | 21.23 ms | **6.03×** |
| **AoS** | 3.15 ms | 4.28 ms | **1.36×** |

Their diagnosis: *"6.03× slower due to insufficient GPU cacheline utilization under random memory
access… even under random particle access order, different attributes of the particle stay in the
same or nearby cacheline. Unlike CPUs, NVIDIA GPUs have no prefetching. **This makes sorting
unnecessary.**"*

**This reframes R2**: a large part of an ordering penalty is *particle-side cacheline waste*, not
grid-side atomic contention — and AoS fixes it without any sort.

**Where LUNA already stands, honestly assessed.** Positions are `(N,3)`
(`particle_mesh_blocks.py:188` slices `pos_l[:, a]`), so the three coordinates already share a
cacheline — LUNA is in the good case for the dominant read. The remaining SoA-ness is across
*attributes*: `pos`, `p`, `mass`, `active` are four separate arrays, so the deposit streams three
independent buffers. **[V-src]** Folding `mass`/`active` into the position record would be the AoS
move, but the upside is bounded because positions dominate the traffic. **Assess before rewriting the
data model** — this is not the 6× that Taichi's table advertises for LUNA's starting point.

**The cheaper, unambiguous half — delayed reordering.** Gao et al. 2018 **[V-paper]**: *"for a
high-resolution simulation, the reordering itself can be the bottleneck"*, so they physically permute
only *positions* and use permutation maps for everything else. LUNA's `resort_blocks`
(`particle_mesh_blocks.py:330`) gathers **all four arrays** every time it runs
(`pos_l[order], p_l[order], mass_l[order], act_l[order]`) **[V-src]**. If the sort cadence from R2
turns out to matter, permuting fewer arrays per resort is the direct way to make each one cheaper.

Related caution on the sort cost itself: no published rocPRIM radix-sort throughput exists for
MI300A, and LUNA needs ≥27-bit cell keys over 134M particles. **Measure the sort, not just the
deposit.**

### R3 — Fixed-point integer accumulation in the CIC deposit
**VERIFIED-IN-LITERATURE (QuanTaichi 2021 does exactly this for MPM grid nodes; OpenMM since 2012 for MD forces) — NOVEL only within cosmology · [speed] [accuracy] [reproducibility] · difficulty: LOW in pure JAX · targets: the atomics-bound deposit**

> **Attribution correction.** An earlier draft of this note called the technique novel. It is not.
> **QuanTaichi** (Hu, Liu, Yang et al., SIGGRAPH 2021, ACM TOG 40(4), arXiv:2106.14106) states it
> exactly, for particle-to-grid transfer in MPM **[V-paper]**:
> > "On fixed-point atomic adds, we override the decode-compute-encode cycle. **We directly scale the
> > real increment into an integer increment, and then use an atomic add on the integer type
> > instead.** … using integral instead of floating-point atomic add leads to a **significant
> > performance improvement**."
>
> They measure `ti.quant.fixed(fraction=32)` on **grid nodes** improving MLS-MPM by **1.4× FPS** vs
> native float32. That is the same operation as a CIC deposit, and it independently corroborates the
> 1.08–1.14× measured here in §8. **Credit QuanTaichi for the technique and OpenMM (2012) for the
> determinism argument; LUNA's contribution would be its first use in cosmological PM**, plus the
> exact-mass construction below, which neither source describes.

Accumulate `round(w·S)` into an integer grid, divide by `S` once at the end.

- **int32 at S = 2¹⁶–2¹⁸** replaces the FP32 deposit: **1.08–1.14× faster**, comparable accuracy,
  bitwise reproducible. **[V-meas §8.1, §8.4]**
- **int64 at S = 2⁴⁰** replaces the FP64 deposit: **same speed**, ~10⁸× more accurate than FP32,
  bitwise reproducible. **[V-meas]**
- **Exact-mass variant** (last corner = `round(w·S)` − sum of the other seven) conserves mass to the
  **last bit** — which no float deposit does, and which speaks directly to LUNA's standing rule about
  asserting mass conservation on halo/exchange paths. **[V-meas §8.2]**

Why it works: integer addition is associative, so the result is independent of thread scheduling and
particle order; and fixed point has **no round-off in the accumulate** at all — it is the degenerate
free case of a Kulisch accumulator when the dynamic range is known a priori, which for a mass density
it is.

Reachability: `.at[].add()` on int32/int64 works today and lowers to native `global_atomic_add`/`_x2`
on gfx942 **[V-src ISA, V-meas]**. No custom kernel needed.

Note that JAX documents the float nondeterminism explicitly, so this is not a subtle issue
**[V-src, `docs.jax.dev` `jax.numpy.ndarray.at`]**: *"**The order in which conflicting updates are
applied is implementation-defined and may be nondeterministic** (e.g., due to concurrency on some
hardware platforms)."* Nothing in the framework fixes it; integer accumulation does, for free.

**There is a source-level argument that is stronger than the speed number.** XLA classifies an
integer scatter-add as *deterministic by construction*, because integer addition is associative
**[V-src, `xla/service/scatter_utils.cc:220-252`]**:

```cpp
bool IsScatterCombinerAssociative(const HloComputation* combiner) {
  case HloOpcode::kAdd: ... return combiner->root_instruction()->shape().AreAllLeavesIntegers();
}
bool IsScatterDeterministic(const HloScatterInstruction* scatter) {
  if (scatter->unique_indices()) return true;
  if (IsScatterCombinerAssociative(scatter->to_apply())) return true;
  ...
}
```

and `ScatterDeterminismExpander::InstructionMatchesPattern` fires only when
`!IsScatterDeterministic(...)`. **So an integer deposit is the only configuration that is
simultaneously (a) bit-exact reproducible, (b) on the native atomic fast path, and (c) immune to the
determinism expander even under `--xla_gpu_deterministic_ops`.** A float deposit can have at most two
of the three: forcing determinism on floats invokes the expander, which sorts the indices and runs a
log₂(N) masked prefix scan over all 1.07G updates — XLA's own docs warn of *"substantial runtime
throughput loss"*.

**Width caveat:** only 32- and 64-bit integers get hardware atomics
(`xla/codegen/emitters/transforms/atomic_rmw_utils.cc:42`: `element_bitwidth == 32 || == 64`)
**[V-src]**. **int8/int16 fall off into a compare-and-swap loop** — a narrow fixed-point deposit
would be catastrophically slow. Use int32 or int64, nothing narrower.

Risk: **overflow at the density peaks.** Use int32 only with S ≤ 2¹⁸ (headroom to ρ = 8192) and add
an assertion on `max(ρ)`; or use int64 and stop worrying. Also: the quantization is per-contribution,
so plain fixed point does *not* conserve mass exactly — use the exact-mass variant if that matters.

Genuine differentiator: **no mainstream PM code advertises a bitwise-reproducible density field under
reordering** (§5.6). GADGET-4's `DEBUG_MD5` checks restart invariance, not order invariance.

### R4 — GADGET-4-style uint32 integer positions
**VERIFIED-IN-LITERATURE (GADGET-4, PKDGRAV3, CUBE; pmwd's pmid+disp is the JAX cousin) · [bytes] [accuracy] · difficulty: MEDIUM · targets: bandwidth everywhere positions are read**

Store positions as `uint32` over the full unsigned range (§3). Gains:

- **12 B/particle instead of 24** — 1.6 GB saved at 134M particles, and 1.6 GB/step of traffic saved
  in every kernel that touches positions.
- **Uniform 5.82e-6 kpc/h resolution** everywhere — 335× finer than FP32 at the box corner, and
  unlike FP32 it does not degrade with distance from the origin.
- **Exact periodic wrapping** by unsigned overflow; **exact minimum-image differences** with no
  cancellation.
- **CIC index and weight by shift/mask** — measured **not slower**, marginally faster **[V-meas §8.3]**.
- Lossless, reversible whole-box shifts (GADGET-4's `RANDOMIZE_DOMAINCENTER` trick), which may be
  interesting for a differentiable code.

Reachability: pure JAX (`uint32` arithmetic, `>>`, `&`, wraparound add). Watch
`lax.bitcast_convert_type` vs `.astype` for the signed reinterpretation (§3.7).

Risk: touches the whole particle data model, IC generation, snapshot I/O and halo exchange — the
largest *engineering* item here, though each individual piece is simple. The drift quantization
(one integer unit per step) is unbiased and 4 orders below a cell; worth a convergence check.

**Cheaper 80% variant:** pmwd's split — `int16`/`int32` cell id + `float32` sub-cell displacement
(`pmwd/particles.py:31-42`) — which gets the uniform-precision benefit without changing the
periodic-wrap machinery. JaxPM ships this as `cic_paint_dx`.

### R5 — Keep `mixed` mode as-is; do not increase inner V-cycle count
**VERIFIED-IN-LITERATURE · [accuracy] · difficulty: NONE (a documentation/guard change) · targets: V-cycle**

LUNA's `mixed` mode is textbook three-precision iterative refinement and is now citable:
Carson & Higham 2018 (`10.1137/17M1140819`) Cor. 3.3 — the limiting accuracy depends on the residual
and working precision, **not** on the solve precision; Göddeke et al. 2007/2011 for the measured
1.7–2.1× on multigrid specifically (matching LUNA's 1.8×) and for the result that mixed precision
reproduces double-precision answers even at κ ~ 10¹⁵.

The implementation matches the reference algorithm exactly
(`gravity/poisson_multigrid.py:481-491`) **[V-src]**:

```python
def _dc(_, ph):
    r = (rhs_local - laplacian_7pt(ph, dx, halo_fn)).astype(jnp.float32)  # FP64 residual, cast
    e = vcycle(jnp.zeros_like(r), r, dx, halo_fn, ...)                    # ONE FP32 V-cycle
    return ph + e                                                         # FP64 update
phi_local = jax.lax.fori_loop(0, n_cyc, _dc, phi_local)
```

Two guard rails from the same sources: (a) the residual, the operator, and the update **must** stay
FP64 — LUNA does this; (b) **do not over-solve the FP32 correction** — Göddeke's measured optimum is
~2 digits, i.e. **one to two V-cycles per outer correction**, and driving it further is *"not
beneficial because the inner solvers converge to wrong solutions in single precision."*

**LUNA is already at exactly the recommended setting**: `_dc` performs precisely one `vcycle` per
outer defect-correction iteration. The actionable item is therefore only to *record* this — add a
code comment citing Göddeke so nobody "optimizes" it later by nesting extra inner cycles, and treat
increasing the inner cycle count as a known-bad change rather than a tuning knob.

### R6 — Unify the three regimes behind `gravity_precision`
**[maintainability] · difficulty: LOW**

Map the proposed field onto what the evidence supports:

| `gravity_precision` | positions | deposit accumulator | V-cycle | φ carry / residual |
|---|---|---|---|---|
| `'double'` | uint32 (R4) or FP64 | **int64 @ 2⁴⁰** (R3) | FP64 | FP64 |
| `'mixed'` (default) | uint32 (R4) or FP32 | **int32 @ 2¹⁷** (R3) | FP32 | **FP64** |
| `'single'` | uint32 (R4) or FP32 | int32 @ 2¹⁷ | FP32 | FP32 |

Note that under R3 the `'double'` deposit becomes *more* accurate than today's FP64 deposit while
costing the same, and gains bitwise reproducibility.

### R7 — LDS/shared-memory privatization of the deposit
**VERIFIED-IN-LITERATURE · [speed] · difficulty: HIGH (needs Pallas or a custom HIP kernel) · targets: atomics**

WarpX PR #3368 measured **3.30× on MI250X** with a 4×4×4 tile. **But MI250X global FP atomics are
CAS loops and MI300A's are native** (§5.1), so the honest expectation on viper is the A100-like
*"~30% if the tile size is chosen well"*, not 3.3×. **[INF from the verified ISA table]**

**The strongest argument against R7 is not the tooling but a measurement**: SC20's deposit-shaped MD
case study found tuned privatization at **0.91× — a slowdown — on hardware with native float
atomics**, winning 2.65× only against CAS emulation (§5.6b). MI300A has native atomics. WarpX
independently reports *"none of these methods have been faster"* than plain global atomics on V100
**[V-paper]**. Two independent negatives on exactly this technique, on exactly this hardware class.

**And on ROCm the tooling is a bad bet right now** **[V-src]**:

- **Mosaic GPU refuses to run on AMD** (`jax/_src/pallas/pallas_call.py:932`: *"Mosaic GPU does not
  yet support AMD ROCm devices"*), **and it became the default** (`JAX_PALLAS_USE_MOSAIC_GPU`
  defaults `True`), so a bare `pallas_call` on MI300A raises. Its atomic API could not express a
  scatter anyway — `atomic_add(ref, val)` requires `ref.shape == val.shape`, a dense accumulate.
- **The Triton backend — the only one that runs on ROCm — was deprecated in JAX 0.11.0** with a
  runtime `DeprecationWarning`, and `pl.atomic_add` has already moved namespace and been removed
  under its old name.
- **AMD CI excludes the relevant tests**: `build/rocm/ci_test_targets.txt` excludes
  `//tests/pallas:triton_pallas_test_gpu`, the only file containing the atomic-add tests; upstream
  restricts that suite to H100/B200.

The one genuine upside if it is ever pursued: Triton's AMD backend has a CDNA3 atomic optimization
XLA will not do for you (`TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp`: *"CDNA3/CDNA4 arch allows to
accelerate its atomics with LDS reduction algorithm"*) — intra-wave reduction of colliding addresses
before the global atomic, which is exactly a CIC deposit's contention pattern. gfx942 is CDNA3.

Do **not** invest here before R1–R3. If pursued: an 8³–10³ FP64 interior tile is 4–8 KiB of a 64 KiB
LDS; skip zero cells on flush (WarpX `SharedDepositionUtils.H:42-62`); do **not** sort within a tile
(Smilei's bank-conflict finding).

### R8 — Compensated (Kahan/Neumaier) summation for global reductions only
**VERIFIED-IN-LITERATURE · [accuracy] · difficulty: LOW · targets: neither (cheap, correctness)**

Not usable inside an atomic scatter — Kahan needs a *per-accumulator* carry term and a
read-modify-write atomic has nowhere to put it. But it is applicable to residual norms, global dot
products, and the mass/momentum conservation assertions.

- Abacus names Kahan as the intended fix for its analogous accumulation error mode
  **[V-paper §3.5]**.
- **REBOUND/IAS15** (Rein & Spiegel, MNRAS 446, 1424 (2015), arXiv:1409.4779) is the standard
  citation in N-body **[V-paper]**: *"We implemented compensated summation for all updates of the
  position and velocity … **Our experiments have shown that this decreases E_rand by one to two
  orders of magnitude at almost no additional cost.**"* The framing is Brouwer's law — unbiased
  rounding turns O(n) energy-error growth into a random walk O(√n). Also used by `orbitN`
  (arXiv:2306.03737).
- **It is not free on GPUs, contrary to folklore.** The best measured datum is Feltor (Wiesenberger
  et al., *Comput. Phys. Commun.* 2019, arXiv:1807.01971) Table 2 **[V-paper]**: an exactly-rounded
  dot product reaches **594 GB/s vs 849 GB/s for plain axpby on a V100 (ratio 0.70)** and 347 vs 554
  on P100 (0.63) — i.e. **~1.4–1.6× slower**, not free. On a GPU with crippled FP64 (GTX 1060) it
  collapses to 0.17.
- For a reproducible *global reduction* specifically, Iakymchuk et al. (arXiv:2005.07282) measured
  ExBLAS-based reproducible PCG at *"the overhead of less than **37.7%** on 768 cores"*
  **[V-paper]**.

Cheap enough for per-step conservation checks; do not put it in a hot loop.

### R9 — Block floating point / compressed field storage for the V-cycle
**VERIFIED-IN-LITERATURE (outside astrophysics) · [bytes] [speed] · difficulty: HIGH (needs a custom kernel) · targets: the bandwidth-bound V-cycle**

The V-cycle is bandwidth-bound (LUNA's measured 1.8× FP64→FP32 says so), so *any* reduction in bytes
moved converts directly to speed — and storage precision can be decoupled from arithmetic precision.

- **FRSZ2** — Grützmacher, Underwood, Di, Cappello & Anzt, arXiv:2409.15468 (2024) — in-register
  block compression inside GMRES on GPUs. Block size 32; extract a shared `e_max` per block and
  normalize significands to it (textbook block floating point). Measured on **H100**: decompression
  at **1991 GB/s ≈ 99.6% of achievable bandwidth**, *"1.2–3.1× faster than the next fastest
  compressor cuSZp2"*. **[V-paper]**
- **zfp** — Lindstrom, *IEEE TVCG* 20(12) (2014) — *"a **block-floating-point representation, which
  uses a single, common floating-point exponent for all 4^d values**"*, with in-memory random-access
  compressed-array classes, not just file I/O. **[V-paper]**
- **Ginkgo memory accessor** — Aliaga, Anzt, Grützmacher, Quintana-Ortí & Tomás, arXiv:2009.12101:
  *"decoupl[es] the storage format of the orthogonal basis from the arithmetic precision"*, measured
  at **up to 50% acceleration over double-precision GMRES on a V100**. **[V-paper]**

**Honest ceiling:** FRSZ2's *solver-level* win was only **1.09–1.16× vs FP64** — compression buys
bandwidth but costs iterations. And LUNA's `mixed` mode already banks the FP64→FP32 halving. So this
is a second-order item that needs a custom kernel (Pallas/HIP), and should be considered only after
R1–R4. Recorded because it is the principled answer to "what is below FP32 for a *grid field*" —
block floating point, not IEEE half (§7).

### R10 — Reduced-precision halo/inter-GPU transport
**VERIFIED-IN-LITERATURE (outside astrophysics) · [bytes] · difficulty: MEDIUM · targets: communication**

Send halo/ghost data at FP32 (or narrower) and promote on arrival. Evidence: Klöwer, Düben & Palmer
(*"16- or even 8-bit communication between processors introduces negligible errors"*); Kormann,
Reuter & Rampp (FP32 halos halve volume and improve inter-island scaling; ZFP gives ~4× and near-flat
scaling). Both flag it as requiring validation, and Kormann calls FP32 halos *"an experimental tool."*

**This is cheaper and safer in LUNA than it first appears, because the halo is a `ppermute`, not a
`psum`.** Two source-level facts **[V-src]**:

- There is **no dtype or accumulation-precision parameter on any JAX collective** — `psum`, `ppermute`,
  `all_gather` etc. take none. The only idiom is to control the dtype of the array you pass.
- **XLA sinks the convert into the collective for you**, so `lax.ppermute(x.astype(f32), ...)` really
  does halve wire bytes rather than just adding two casts:
  `xla/hlo/transforms/collectives/collective_quantizer.cc:113` matches `kAllGather, kAllToAll,
  kCollectiveBroadcast, kCollectivePermute` — and **`kAllReduce` is deliberately absent**, because
  sinking a convert through a reduction would change the numerics.

That asymmetry is the whole recommendation: **cast around `ppermute` halo exchanges (pure
point-to-point, exact apart from the cast); do NOT do the same to any `psum`**, where the
accumulation would happen *in* the reduced dtype and no "bf16 transport / f32 accumulate" mode exists
anywhere in JAX or XLA.

Determinism caveat for anything that *is* a reduction: NCCL's maintainers state that sum order — and
therefore the result — can vary with topology (nccl#1975), and JAX's `psum` nondeterminism issue
#11366 is still open. **None of this was verified for RCCL on MI300A; treat multi-GPU reduction
reproducibility as unestablished on this hardware.** That matters if R3's bitwise-reproducibility
claim is ever extended from single-GPU to rank-invariant.

Evaluate only once R1–R4 are done and only if communication is measurably on the critical path at
multi-GPU scale. **Not a precision-policy change — a communication optimization.**

### R11 — Do NOT pursue
- **FP16/BF16 anywhere in the gravity path.** §7. Arithmetically excluded for positions, the density
  accumulator and φ; zero gain on the deposit (no scalar half atomic on CDNA3; XLA promotes to a
  32-bit packed atomic); capped at 2× on the correction solve relative to the existing FP32 mixed
  mode. No production astrophysics code does it.
- **Double-single / float-float arithmetic.** Göddeke measured it failing where mixed-precision IR
  succeeds, because it extends the mantissa but *not* the exponent range (effectively s46e8), and it
  is slower than native FP32 on device **[V-paper]**.
- **The 8-colour CIC scheme.** §5.7 — it targets a conflict class that does not exist, and LUNA's
  existing 8-pass structure already demonstrates this.
- **Sorted-gather / `segment_sum` reformulation of the deposit.** RAMSES-GPU measured atomics
  *"significantly better"* than the prefix-sum alternative at ~1 particle/cell, which is LUNA's
  regime **[V-paper §7.4]**.
- **Stochastic rounding.** Solves the stagnation problem that fixed-point accumulation (R3) solves
  better, deterministically, and for free.

---

## 10. Further ideas evaluated and their status

| Idea | Status | Verdict |
|---|---|---|
| Sub-cell relative positions (high bits = cell, low bits = weight) | **VERIFIED-IN-LITERATURE** — this *is* GADGET-4's PM assignment (§3.6) and pmwd's pmid+disp | Adopt via R4; measured free (§8.3) |
| Incremental / sparse deposit (subtract old CIC contribution, add new) | **NOVEL-PROPOSAL**; no prior art found in PIC, MD or cosmology | See below |
| Low-precision Δφ carry (FP64 base + FP32 per-step difference) | **NOVEL-PROPOSAL** (defect correction along the time axis) | See below |
| Block floating point / shared exponent | VERIFIED-IN-LITERATURE outside astrophysics (QUDA int20+exponent; Ginkgo `scaled_reduced_row_major`) | Superseded by R3/R4 for LUNA's quantities |
| Exploiting δ ≥ −1 | **NOVEL-PROPOSAL** | Marginal: saves at most 1 bit; fixed point (R3) already exploits the known dynamic range |

**Incremental deposit.** Attractive on paper: at z = 50 particles move 0.068 cells/step, so the
density field changes little.

**Prior art exists — but only in the regime where a *subset* of particles moves per step**, which is
not LUNA's:

- **Maggs**, "Optimizing Ewald summation for Monte Carlo simulations" (ESPCI preprint, 2015) — the
  incremental structure-factor update `S(q) → S(q) − q_i e^{iq·r_i0} + q_i e^{iq·r_i}`, *"an effort
  which is **O(1), not the O(N)**"*. He also quantifies the drift objection and its cure
  **[V-paper]**: *"There is an obvious fear that following the structure factor using eq. (10) will
  lead to **numerical drift and noise in long simulations** … **The solution is to use the Kahan
  compensated summation algorithm** … we do not obtain a breakdown in the accuracy of the sum before
  system sizes which are N = 10¹⁴"* (vs N ≈ 10⁴ without).
- **Edorh & Redon**, *J. Comput. Chem.* 39(20), 1455–1469 (2018), `10.1002/jcc.25215` — incremental
  multigrid-based electrostatics in **LAMMPS**, *"exploit[ing] the fact that … only some particles
  are allowed to move at each timestep."* **Cited 3 times in 8 years** — the space is wide open.
  Note they *abandoned FFT*, because a local charge change perturbs every Fourier mode.
- **Clean negative in PIC and in cosmology**: no incremental/delta charge or mass deposition found
  in either field. (Do not confuse this with **Esirkepov** deposition, which differences one
  particle's own shape factors *within* a step and recomputes both from scratch every step, or with
  **δf** PIC, which is a control-variate against a fixed analytic background.)

**Why it does not apply to LUNA as it stands:** (a) *every* particle's 8 weights change every step
even without a cell crossing, so the update is not sparse in the required sense — only the *index
set* is stable, not the values; (b) it would need a compaction/gather pass whose divergence cost is
likely to exceed the saving; (c) the WarpX/PIConGPU experience is that ~half the particles change
cell per step anyway in an active simulation; (d) errors accumulate over ~7000 steps.

**Two things that would change the calculus.** First, **with fixed-point accumulation (R3) an
incremental deposit becomes exactly reversible** — integer subtraction exactly undoes integer
addition, so the drift objection that Maggs had to answer with Kahan simply vanishes. That is the one
version of this idea that is numerically clean, and it appears to be unpublished. Second, it becomes
interesting only if LUNA ever has a **frozen or slow subset** — multi-rate stepping or a deep AMR
hierarchy where fine-level particles substep while coarse ones do not. In that regime Edorh & Redon
is direct prior art and must be cited.

Status: **NOVEL-PROPOSAL, low priority for pure global-step PM**; the fixed-point synergy is worth
recording as it is the genuinely novel piece.

**Low-precision Δφ carry.** LUNA warm-starts from the previous potential. Carrying the base in FP64
and the per-step difference in FP32 is defect correction along the time axis. This is a **bytes**
play on a grid field, so it competes directly with — and is dominated by — simply running the
V-cycle in FP32, which LUNA already does. Status: **NOVEL-PROPOSAL, low priority**.

---

## 11. What could not be verified

Recorded so the negatives are not over-read:

- Bowers 2001 (*J. Comput. Phys.* 173(2):393) — the often-quoted "40–70%" sorting speedup could not
  be confirmed from a primary source. **Do not cite.**
- "Yoshikawa & Tanikawa, *Accelerating Cosmological N-body Simulations with Tensor Cores*" —
  **could not be verified to exist**.
- Whether `unique_indices=True` removes atomics in the current XLA GPU scatter emitter; how
  `jax.ops.segment_sum` lowers; what `--xla_gpu_deterministic_ops` does in practice; and Pallas
  atomic-add support on ROCm. All open, all reachable by direct source inspection.
- The PKDGRAV3 source read was a third-party GitHub fork; the official repo is on Bitbucket and was
  not diffed. The Abacus papers disagree internally on 23 vs 24 mantissa bits and on the Fig. 7
  error values.
- Göddeke & Strzodka 2007/2011 PDFs are paywalled; the quantified claims above are taken from
  Göddeke's open-access thesis, which states it is an extended version of that joint work.
- HACC and CRK-HACC main sources are not public; HACC claims rest on the papers plus the public
  CORAL v1.3 benchmark and HACC-B24 kernels.
- Several MI300A hardware claims are **[INF]** from the verified ISA output rather than from a vendor
  performance document — in particular that CDNA3 has no vector FP16 rate advantage over FP32.
- **`indices_are_sorted=True` is a live contradiction** (R1b): two independent reads of
  `xla/backends/gpu/codegen/emitters/scatter.cc` reached opposite conclusions — one that it selects a
  warp-accumulating fast path, one that scalar-slice scatters are restricted to `lane_id == 0` and
  that open bug openxla/xla#22233 shows a ~95× regression. **Not resolved here. Measure before
  trusting either reading.**
- **Gómez-Luna et al. (MVA 2013 / TPDS 2013)** and **van den Braak (ICCD 2013)** — paywalled, could
  not obtain; characterized only via SC20's description. **Jin & Vetter, "Atomics-Based Integer Sum
  Reduction in HIP on AMD GPU" (ICPP-W '22)** — the most AMD-relevant paper found; ACM DL returned
  403.
- **rocPRIM/hipCUB radix-sort throughput on MI300A**: no published number exists. Must be measured
  before costing any sort-based scheme.
- **LDS FP64 atomic throughput on CDNA3**: the ISA documents 32 *integer* atomic units for LDS and is
  silent on the float path. Must be microbenchmarked.
- **PLOS'25 atomic-contention results cover RDNA2/3 and Ada only — no CDNA.** Their suite is
  open-source HIP; running it on MI300A would produce exactly the numbers this literature lacks, and
  is probably the highest-value single experiment available beyond §12's item 0a.
- The Wang et al. 2020 MPM paper mischaracterizes Taichi 2019's shuffling experiment as a
  conflict-reduction technique; reading Taichi directly shows it is a cacheline benchmark in which
  shuffling is *slower*. Cite Taichi's Table 2 directly, not the 2020 summary of it.

---

## 12. Suggested next measurements

In order, each at production scale (512³ / 134M) per the standing rule that performance conclusions
come only from production-scale runs:

0a. **Run the contention-ceiling experiment before writing any kernel.** Nugteren, van den Braak,
   Corporaal & Mesman (GPGPU-4, 2011) **[V-paper]** bounded every histogram technique this way: run
   the *identical* kernel with collision-free indices (`idx = tid`), accept the wrong answer, and
   measure. Their baseline went from ~6–13 GB/s to **~45–50 GB/s**, proving *"intra-warp shared memory
   collisions cause the major part of the overhead."* For LUNA this is ~20 lines and one `srun`, and
   **it upper-bounds every recommendation in §9 that targets atomics** (R1b, R3, R7). If the ceiling
   turns out to be 1.3×, stop and spend the effort on R2/R4 instead. Do this first.

0b. **Measure the particles-per-cell distribution** (SC20's "race factor"): sample runs of consecutive
   particles post-resort and count distinct cells. It is the input to every model in this literature
   and determines which regime LUNA is in — sort-wins vs atomics-win.

0. **Audit the periodic-boundary cell index** — this is a correctness item, not performance, and
   it is cheap. JAX-MD's only production FP32 bug (jax-md issue #339, fixed in PR #378) was exactly
   this: `jnp.array(position / cell_size, dtype=i32)` returned an index equal to `cells_per_side` when
   the position was a hair under `box`, **silently dropping pairs**; the fix was an integer `jnp.mod`.
   LUNA's `i0 = jnp.floor(f[a]).astype(jnp.int32)` followed by `jnp.clip`
   (`particle_mesh_blocks.py:190,193`) is the same code shape, and **a clip silently relocates a
   particle rather than erroring**. Confirm the clip is covering the boundary case rather than masking
   a wrap that should have happened. Note that R4's integer positions make this class of bug
   impossible by construction — the cell index is a bit-slice and cannot go out of range.
1. **R1 first** — fix the position casting, re-run the FP64→FP32 comparison. Until this is done no
   measurement isolates the atomic path, and the 1.17× figure should not be quoted.
2. **R3 in the real kernel** — swap `cic_deposit_blocks` to int32/int64 accumulation behind a flag,
   confirm the §8 microbenchmark result survives sharding and the ghost-fold, and assert
   `max(ρ)` against the overflow threshold.
3. **R2 sort-interval sweep** — measure deposit time vs steps-since-resort at z = 50, z = 5, z = 0.
4. **R4 prototype** on the single-GPU path before committing the data model.
5. Only then consider R7 (LDS privatization), and expect ~30%, not 3.3×.

---

## 13. Results of the §12 measurement programme (2026-08-06)

Everything below is **[V-meas]** on ONE MI300A (viper), the **real l9 MUSIC IC**
(`ics_arepo_l9_bary.hdf5`, 134,217,728 particles), 512³ mesh, JAX 0.10.2 / ROCm 7.2.
Harnesses (in-tree): `scaling/repro/deposit_precision_probe.py` (deposit/gather kernels,
`dep_probe_precision.sbatch`) and `scaling/repro/dm_step_precision.py` (full production step,
`dm_step_precision.sbatch`). Every full-step row replays the same dτ/a schedule, so all modes
integrate the identical system; the gate is the 8-step `delta_max` trace and the FP64-evaluated
Poisson residual.

### 13.1 §12.0a — the contention ceiling (the gating experiment)

Same kernel, same weight arithmetic, scatter targets replaced by a collision-free bijection
(particle p → cell p; deliberately the wrong answer):

| deposit | real indices | collision-free | headroom |
|---|---|---|---|
| float32 accumulator | **15.16 ms** | **9.43 ms** | **5.74 ms = 37.8% (1.61×)** |
| int32 accumulator | 13.47 ms | 8.87 ms | 4.60 ms = 34.1% |

**The ceiling is large, not small** — 5.7 ms is 16% of the whole 34.6 ms all-FP32 step. But it is
an *upper* bound on everything that targets the scatter at once, and part of it is not atomics:
the uint32-position row below replaces the divide/floor/mod index arithmetic with shift/mask and
recovers only 0.62 ms, so ≲0.6 ms of the 5.7 is index arithmetic and the rest bounds
collisions + addressing + the scatter emitter together. The collision-free variant also hands XLA
an iota index, which is a strictly easier code-generation problem than a data-dependent one.

What can actually be claimed from that headroom **in pure JAX** is a different question, and the
answer is: only the integer accumulator (1.69 ms of the 5.74, below). Privatization (R7) needs
Pallas/Triton, which §7/R7 documents as unusable on ROCm today, and carries two published
*negatives* on native-atomic hardware. Sorting (R2) has nothing left to give at this epoch —
see the occupancy census.

### 13.2 §12.0b — particles per cell (the "race factor")

Post-IC, at z = 50, over all 134M particles and 512³ cells:

| particles in a cell | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| cells | 11,428,175 | 112,320,455 | 9,724,588 | 555,142 | 172,391 | 11,025 | 4,210 | 1,214 | 528 |

Mean 1.000, max 8, 84% of cells hold exactly one particle. This is squarely the regime where
RAMSES-GPU measured atomics *"significantly better"* than the prefix-sum alternative (§5.4), and
it confirms the particles are still lattice-ordered — **R2's sort has zero headroom at z = 50**;
its value is entirely at late times, which this measurement cannot reach.

### 13.3 R3 — fixed-point accumulation, in the real kernel

`gravity/particle_mesh.py: cic_deposit(..., fixed=jnp.int32|int64, fixed_bits=b)`, integer unit =
2^b per mean-density cell, corner weights split by a rounding tree so the eight integer weights sum
to the particle's weight exactly.

| deposit (isolated) | time | max\|Δρ\|/max(ρ) vs FP64 | mass error |
|---|---|---|---|
| float32 (production) | 15.16 ms | 4.29e-5 | −1.22e-8 |
| **int32 @ 2¹⁷ exact-mass** | **13.47 ms (1.13×)** | 5.12e-5 | −1.22e-8 |
| float64 | 23.37 ms | 0 (reference) | +5.3e-16 |
| **int64 @ 2⁴⁰ exact-mass** | **22.04 ms (1.06× faster than FP64)** | **2.33e-12** | **+4.0e-16** |

int32 reproduces §8's 1.08–1.14× on the real IC; **int64 is a strict upgrade on the FP64 deposit —
6% faster and 2×10⁷ times more accurate.** Peak integer occupancy at z = 50 was 359,523 of
2,147,483,647, i.e. overflow at 1+δ = 16,384 with 45,000× of headroom now (the runner raises when
1+δ_max passes half the ceiling).

### 13.4 R4 — uint32 Gadget-style positions, in the real kernel

Whole box → whole unsigned range; cell index = `ipos >> 23`, sub-cell weight = low 23 bits;
LUNA's cell-centred convention folded in as one integer half-cell shift (so the periodic wrap is
an unsigned subtraction, no `mod`, no `clip`).

| kernel | float32 positions | **uint32 positions** | float64 positions |
|---|---|---|---|
| deposit | 15.16 ms | **14.54 ms** | 23.37 ms |
| deposit accuracy vs FP64 | 4.29e-5 | **2.75e-7 (156× better)** | 0 |
| gather (3 fields) | 9.93 ms | **10.03 ms** | 11.12 ms |

**uint32 positions are an accuracy upgrade at float32 cost**, exactly as §8.3 predicted — not a
speed play. Note the deposit is *not* position-bandwidth-bound (float64 vs float32 positions cost
the same 15.1 ms in the full-step harness), so the payoff is in the gather and the drift.

### 13.5 The full production step, per `gravity_precision`

`scaling/repro/dm_step_precision.py`, warm 1 V-cycle defect correction + linear φ extrapolation +
o4 gradient, identical schedule, 5 timed steps after 2 warm-ups:

| mode | step (best) | residual | deposit | solve | grad | gather | kick+drift |
|---|---|---|---|---|---|---|---|
| `single` | **34.7 ms** | 2.9566e-3 | 15.21 | 7.04 | 1.89 | 10.11 | 3.01 |
| `single` + fixed | 35.9 ms | 2.9568e-3 | **13.56** | 6.90 | 1.92 | 10.08 | 2.98 |
| `mixed` (default) | 45.2 ms | 2.9335e-3 | 15.11 | 8.43 | 3.67 | 11.28 | 5.65 |
| `mixed` + fixed | **44.1 ms** | 2.9338e-3 | 14.47 | 8.32 | 3.63 | 11.29 | 5.64 |

Gate: the 8-step delta_max trace is 1.8074 1.8366 1.8205/6 1.8289 1.8529/30 1.9406/7 2.0275/6/7
2.0914/5 in **all four** configurations — the fixed-point deposit moves it by ±1 in the last
printed digit, i.e. by the same amount as the existing FP32-vs-FP64 difference between the shipped
modes.

**The fused step does not inherit the isolated deposit win.** In `mixed` it does (−1.1 ms); in
`single` the step gets 1.2 ms *slower* even though its deposit is 1.65 ms faster — XLA fuses the
float deposit into the drift and the integer path breaks that fusion (sum-of-isolated 37.26 vs
fused 34.71 for float, 35.44 vs 35.92 for fixed). **Conclusion: fixed point is a
reproducibility/mass-exactness feature that is step-neutral, not a speed feature.** It ships
opt-in (`gravity_deposit_fixed=False` by default).

### 13.6 What shipped, and the config surface

Landed in the tree (single-device DM path; **`gravity_precision` is the one switch**, replacing the
scattered `cfg.dm_mixed_precision` / `cfg.dm_fp32` / `LUNA_MG_MIXED` — the old fields are kept as
deprecated aliases that resolve onto it, and the env var still overrides):

| `gravity_precision` | particles | deposit | V-cycle | φ carry / residual |
|---|---|---|---|---|
| `'double'` | FP64 | FP64 (int64 @ 2⁴⁰ with `gravity_deposit_fixed`) | FP64 | FP64 |
| `'mixed'` (default) | FP64 | FP32 (int32 @ 2¹⁷ with `gravity_deposit_fixed`) | FP32 | FP64 |
| `'single'` | FP32 | FP32 (int32 @ 2¹⁷) | FP32 | FP32 |

`setup/cosmo_common.py: gravity_precision_from_config()` is the single resolver; it also fixes a
real bug — **`cfg.dm_fp32` was a no-op in `setup/runner_cosmo_dm.py`** (it was only ever read by
`runner_cosmo_hydro.py:468`), so the FP32 particle path was unreachable from the DM runner's config.

**Proposed surface for R4 (not yet wired):** one orthogonal field
`gravity_positions: 'float' | 'uint32'`, not a family of flags. Under `'uint32'` the particle
positions are stored as `uint32` over the whole box, the drift becomes
`ipos += rint(Δx · 2³²/box)` (unsigned wraparound = the periodic wrap, exactly), and deposit/gather
take the cell index and the sub-cell weight out of the same integer by shift and mask. It requires
a periodic cubic box and a power-of-two mesh (otherwise the cell index is not a bit-slice), and it
touches the drift, the snapshot writer and the block-exchange buffers — which is why it is proposed
rather than landed. Its measured value is §13.4: float32 speed at 156× the position accuracy, and
in `'mixed'` it would additionally buy the FP64→FP32 gather and drift savings (≈3.8 ms/step) while
*improving* on FP32 positional accuracy.

### 13.7 Still open

- **The `single` fused-step regression** with the integer deposit (13.5): worth one XLA-level look —
  if the deposit can be kept fused with the drift, fixed point becomes a free win in every mode.
- **R2 sort cadence** — untestable at z = 50 (13.2 shows the particles are still lattice-ordered).
  Needs evolved states at z ≈ 5 and z ≈ 0.
- **Fixed point in the block-sharded deposit** (`gravity/particle_mesh_blocks.py`) and the
  rank-invariance question that comes with the ghost fold (a fixed-point deposit is only
  rank-invariant if the halo accumulation is fixed point too, §5.6).
- **R1b (`indices_are_sorted`)** deliberately NOT tested: still CONTESTED, and openxla/xla#22233
  (~95× regression) makes it a bad bet against a best case of ~8×.
