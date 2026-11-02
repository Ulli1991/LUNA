"""A16 cross-check: the LUNA comoving tests' ICs for mini-ramses and Arepo, and their outputs reduced into luna-reference-solutions.

  python tools/a16_xcheck.py gen ROOT          grafic ICs + run.nml (ROOT/ramses/<run>), HDF5 ICs + param.txt (ROOT/arepo/<run>)
  python tools/a16_xcheck.py reduce ROOT       read both codes' outputs -> luna-reference-solutions/{ramses,arepo}/cosmo_a16_xcheck/
  python tools/a16_xcheck.py luna OUT.npz      LUNA through the tests' run() (GPU job), for the plot
  python tools/a16_xcheck.py plot ROOT LUNA.npz -> ROOT/a16_xcheck.png and the table

LUNA super-comoving (H0 = 1, box 1): rho~ (mean 1), v~ = a v_pec / (H0 L), p~ = a^5 p.
mini-ramses UNITS=COSMO, omega_b = omega_m = 1 (the default omega_b 0.045 rescales the IC density,
hydro/input_hydro_grafic.f90:203), h0 = 100, box 100 Mpc/h: code units are LUNA's (amr/units.f90:21-25);
ic_deltab = rho~ - 1, ic_velb* [km/s] = v~ H0 L / a, ic_tempb = (p~/rho~) / (T2_start / scale_T2) - 1
(input_hydro_grafic.f90:201,207). Outputs are float32 primitives (hydro/output_hydro.f90:118).
Arepo (kpc, 1e10 Msun, km/s; h = 1, box 1e5 kpc): Velocities = v_pec / sqrt(a) = v~ H0 L / a^1.5,
InternalEnergy = (p~/rho~) (H0 L)^2 / a^2 / (gamma - 1), Masses = rho~ rho_crit (L/N)^3.
Neither code lands a snapshot on a target a; every metric is taken against the exact answer at the snapshot's own a
and interpolated linearly in a between the bracketing snapshots.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from luna.reference_data import reference_root  # noqa: E402


def _out(code):
    """The reduced npz of `code` in luna-reference-solutions (run its make_manifest.sh after writing)."""
    return os.path.join(reference_root(), code, "cosmo_a16_xcheck", f"{code}.npz")


# mini-ramses
H0_R, LBOX_R = 100.0, 100.0                   # km/s/Mpc (h = 1), comoving Mpc
MH, KB = 1.6605390e-24, 1.3806490e-16         # mini-ramses constants (amr/constants.f90)
# Arepo internal units (param.txt) and its own G, H0 (begrun.c, allvars.h GRAVITY / HUBBLE)
UL, UM, UV = 3.085678e21, 1.989e43, 1e5
UT = UL / UV
G_INT = 6.6738e-8 / UL ** 3 * UM * UT ** 2
H0_INT = 3.2407789e-18 * UT
LBOX_A = 1.0e5
HL = H0_INT * LBOX_A
RHO_CRIT = 3 * H0_INT ** 2 / (8 * np.pi * G_INT)

RUNS = {"adiabatic_g53": 5.0 / 3.0, "adiabatic_g75": 1.4, "tophat_n64": 5.0 / 3.0, "tophat_n128": 5.0 / 3.0}


def _tests():
    import tests.test_cosmo_adiabatic_expansion as ad
    import tests.test_cosmo_tophat_collapse as tc
    return ad, tc


def out_times(kind):
    """Target a's plus a log-spaced curve (both codes write at or after each listed a)."""
    ad, tc = _tests()
    if kind == "adiabatic":
        t, a0, a1 = ad.A_SNAP, ad.A_I, 1.0
    else:
        t, a0, a1 = tc.A_SNAP, tc.A_I, max(tc.A_SNAP)
    br = np.concatenate([np.linspace(0.85 * x, x, 8) for x in t])     # bracket every target
    a = np.unique(np.round(np.concatenate([br, np.geomspace(1.5 * a0, a1, 24)]), 6))
    return a[(a > a0 * 1.01) & (a <= a1)], t


# ---------------------------------------------------------------- ICs
def prim(U, gamma):
    rho = U[0]
    v = U[1:4] / rho
    return rho, v, (gamma - 1) * (U[4] - 0.5 * rho * (v ** 2).sum(0))


def scale_T2(a):
    """mH/kB scale_v^2, scale_v = scale_l / scale_t (amr/units.f90:22-25, h = 1)."""
    return MH / KB * ((a * LBOX_R * 3.0856776e24) / (a * a / (H0_R * 1e5 / 3.0856776e24))) ** 2


def write_grafic(path, field, a):
    """Header (n1 n2 n3 dx xo yo zo astart omega_m omega_v h0), then n3 planes with i1 fastest."""
    from scipy.io import FortranFile
    n = field.shape[0]
    with FortranFile(path, "w") as f:
        f.write_record(np.array([n, n, n], np.int32).tobytes()
                       + np.array([LBOX_R / n, 0, 0, 0, a, 1.0, 0.0, H0_R], np.float32).tobytes())
        for k in range(n):
            f.write_record(np.ascontiguousarray(field[:, :, k].T, np.float32))


NML = """&RUN_PARAMS
cosmo=.true.
pic=.false.
poisson={poisson}
hydro=.true.
nrestart=0
ncontrol=10
verbose=.false.
/
&OUTPUT_PARAMS
aout={aout}
/
&INIT_PARAMS
filetype='grafic'
initfile(1)='ics'
omega_b=1.0
omega_m=1.0
/
&AMR_PARAMS
levelmin={lmin}
levelmax={lmin}
ngridmax={ngrid}
nexpand=1,
/
&POISSON_PARAMS
epsilon=1.d-6
/
&HYDRO_PARAMS
gamma={gamma:.10f}
courant_factor=0.8
slope_type=1
riemann='hllc'
smallr=1d-10
smallc=1d-10
entropy={entropy}
dual_energy={dual}
/
"""


def gen_ramses(d, U, a, gamma, aout, poisson, entropy=False):
    """Grafic directory d/ics and d/run.nml; entropy needs dual_energy >= 0 (hydro/source_hydro_fine.f90)."""
    ics = os.path.join(d, "ics")
    os.makedirs(ics, exist_ok=True)
    rho, v, p = prim(U, gamma)
    write_grafic(os.path.join(ics, "ic_deltab"), rho - 1.0, a)
    for i, c in enumerate("xyz"):
        for s in "bc":       # ic_velc* is the header source even with pic off
            write_grafic(os.path.join(ics, f"ic_vel{s}{c}"), v[i] * H0_R * LBOX_R / a, a)
    write_grafic(os.path.join(ics, "ic_tempb"), (p / rho) / (1.356e-2 / a ** 2 / scale_T2(a)) - 1.0, a)
    n = U.shape[1]
    open(os.path.join(d, "run.nml"), "w").write(NML.format(
        poisson=poisson, aout=",".join(f"{x:.6f}" for x in aout), lmin=int(round(np.log2(n))),
        ngrid=max(2 * (n // 2) ** 3, 100000), gamma=gamma, entropy=".true." if entropy else ".false.",
        dual="0.01" if entropy else "-1"))


PARAM = """InitCondFile ./ics
OutputDir ./output
SnapshotFileBase snap
OutputListFilename ./output_list.txt
ICFormat 3
SnapFormat 3
TimeLimitCPU {tlim}
CpuTimeBetRestartFile 3000
ResubmitOn 0
ResubmitCommand none
MaxMemSize 3000
TimeBegin {a0}
TimeMax {a1}
ComovingIntegrationOn 1
PeriodicBoundariesOn 1
CoolingOn 0
StarformationOn 0
Omega0 1.0
OmegaLambda 0.0
OmegaBaryon 1.0
HubbleParam 1.0
BoxSize {L}
OutputListOn 1
TimeBetSnapshot 0.0
TimeOfFirstSnapshot 0.0
TimeBetStatistics 0.01
NumFilesPerSnapshot 1
NumFilesWrittenInParallel 1
TypeOfTimestepCriterion 0
ErrTolIntAccuracy 0.012
CourantFac 0.3
MaxSizeTimestep 0.01
MinSizeTimestep 1.0e-12
InitGasTemp 0
MinGasTemp 0
MinimumDensityOnStartUp 0
LimitUBelowThisDensity 0.0
LimitUBelowCertainDensityToThisValue 0.0
MinEgySpec 0
DesNumNgb 64
MaxNumNgbDeviation 4
TypeOfOpeningCriterion 1
ErrTolTheta 0.5
ErrTolForceAcc 0.0025
MultipleDomains 8
TopNodeFactor 2.5
ActivePartFracForNewDomainDecomp 0.01
UnitLength_in_cm 3.085678e21
UnitMass_in_g 1.989e43
UnitVelocity_in_cm_per_s 1e5
GravityConstantInternal 0
SofteningComovingType0 {soft}
SofteningComovingType1 {soft}
SofteningComovingType2 {soft}
SofteningComovingType3 {soft}
SofteningComovingType4 {soft}
SofteningComovingType5 {soft}
SofteningMaxPhysType0 {soft}
SofteningMaxPhysType1 {soft}
SofteningMaxPhysType2 {soft}
SofteningMaxPhysType3 {soft}
SofteningMaxPhysType4 {soft}
SofteningMaxPhysType5 {soft}
SofteningTypeOfPartType0 0
SofteningTypeOfPartType1 0
SofteningTypeOfPartType2 0
SofteningTypeOfPartType3 0
SofteningTypeOfPartType4 0
SofteningTypeOfPartType5 0
GasSoftFactor 1.0
CellShapingSpeed 0.5
CellMaxAngleFactor 2.25
"""


def gen_arepo(d, U, a, gamma, aout, jitter=0.0):
    """HDF5 ICs (gas on the grid points; jitter in cells, a Cartesian lattice is Voronoi-degenerate) and param.txt."""
    import h5py
    os.makedirs(os.path.join(d, "output"), exist_ok=True)
    n = U.shape[1]
    rho, v, p = prim(U, gamma)
    dx = LBOX_A / n
    x = (np.arange(n) + 0.5) * dx
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    if jitter:
        X = (X + jitter * dx * np.random.default_rng(12345).uniform(-1, 1, X.shape)) % LBOX_A
    with h5py.File(os.path.join(d, "ics.hdf5"), "w") as f:
        h = f.create_group("Header")
        npart = np.array([n ** 3, 0, 0, 0, 0, 0], np.uint32)
        for k, val in dict(NumPart_ThisFile=npart, NumPart_Total=npart, NumPart_Total_HighWord=np.zeros(6, np.uint32),
                           MassTable=np.zeros(6), Time=a, Redshift=1 / a - 1, BoxSize=LBOX_A, NumFilesPerSnapshot=1,
                           Omega0=1.0, OmegaLambda=0.0, OmegaBaryon=1.0, HubbleParam=1.0, Flag_Sfr=0, Flag_Cooling=0,
                           Flag_Feedback=0, Flag_StellarAge=0, Flag_Metals=0, Flag_DoublePrecision=1).items():
            h.attrs[k] = val
        g = f.create_group("PartType0")
        g["Coordinates"] = X
        g["Velocities"] = (v * HL / a ** 1.5).reshape(3, -1).T
        g["Masses"] = (rho * RHO_CRIT * dx ** 3).ravel()
        g["InternalEnergy"] = ((p / rho) * HL ** 2 / a ** 2 / (gamma - 1)).ravel()
        g["ParticleIDs"] = np.arange(1, n ** 3 + 1, dtype=np.uint32)
    open(os.path.join(d, "param.txt"), "w").write(PARAM.format(a0=a, a1=max(aout), L=LBOX_A, soft=0.5 * LBOX_A / n,
                                                             tlim=13800 if n > 64 else 6000))
    open(os.path.join(d, "output_list.txt"), "w").write("\n".join(f"{x:.8f}" for x in aout) + "\n")


def gen(root):
    ad, tc = _tests()
    aout, _ = out_times("adiabatic")
    for g, tag in ((5.0 / 3.0, "53"), (1.4, "75")):
        U = ad.uniform(g)
        gen_ramses(os.path.join(root, "ramses", f"adiabatic_g{tag}"), U, ad.A_I, g, aout, ".false.")
        gen_arepo(os.path.join(root, "arepo", f"adiabatic_g{tag}"), U, ad.A_I, g, aout)
    aout, _ = out_times("tophat")
    th = tc.tophat()
    for n in (64, 128):
        U, _ = tc.ic(n, th)
        for ent in (False, True):
            gen_ramses(os.path.join(root, "ramses", f"tophat_n{n}" + ("_ent" if ent else "")), U, tc.A_I, 5.0 / 3.0,
                       aout, ".true.", entropy=ent)
        gen_arepo(os.path.join(root, "arepo", f"tophat_n{n}"), U, tc.A_I, 5.0 / 3.0, aout, jitter=1e-4)
    print("wrote ICs under", root)


# ---------------------------------------------------------------- readers
def ram_info(d):
    kv = {}
    for line in open(os.path.join(d, "info.txt")):
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def ram_grid(d):
    """Uniform-grid output -> (a, prim[nvar, n, n, n]), x the first axis; amr.* holds (ckey[3], refined) per oct."""
    info = ram_info(d)
    nfile, ndim, lmin, lmax = (int(info[k]) for k in ("nfile", "ndim", "levelmin", "levelmax"))
    assert ndim == 3 and lmin == lmax, (ndim, lmin, lmax)
    n = 2 ** lmin
    grid = None
    for icpu in range(1, nfile + 1):
        fa, fh = os.path.join(d, f"amr.{icpu:05d}"), os.path.join(d, f"hydro.{icpu:05d}")
        if not os.path.exists(fa):
            continue
        nvar = int(np.fromfile(fh, np.int32, 1, offset=4)[0])
        if grid is None:
            grid = np.full((nvar, n, n, n), np.nan)
        nlev = lmax + 1 - lmin
        na = np.fromfile(fa, np.int32, nlev, offset=12)
        nh = np.fromfile(fh, np.int32, nlev, offset=16)
        assert (na == nh).all()
        oa, oh = 12 + 4 * nlev, 16 + 4 * nlev
        for nc in na:
            ta = np.fromfile(fa, np.int32, 4 * nc, offset=oa).reshape(nc, 4).T
            th = np.fromfile(fh, np.float32, nvar * 8 * nc, offset=oh).reshape(nc, nvar, 8)
            oa += 16 * nc
            oh += 4 * nvar * 8 * nc
            for ind in range(8):
                grid[:, 2 * ta[0] + (ind & 1), 2 * ta[1] + ((ind >> 1) & 1), 2 * ta[2] + ((ind >> 2) & 1)] = th[:, :, ind].T
    assert not np.isnan(grid).any(), d
    return float(info["aexp"]), grid


def ram_snaps(d):
    return [(float(ram_info(o)["aexp"]), o) for o in sorted(glob.glob(os.path.join(d, "output_*")))]


def arepo_snaps(d):
    import h5py
    out = []
    for f in sorted(glob.glob(os.path.join(d, "output", "snap_*.hdf5"))):
        with h5py.File(f, "r") as h:
            out.append((float(h["Header"].attrs["Time"]), f))
    return out


def arepo_read(f, gamma):
    """-> a, X (box units, centred), rho~, v~x, p~, cell volume."""
    import h5py
    with h5py.File(f, "r") as h:
        a = float(h["Header"].attrs["Time"])
        g = h["PartType0"]
        pos = g["Coordinates"][...] / LBOX_A
        rho = g["Density"][...] / RHO_CRIT
        vel = g["Velocities"][...] * a ** 1.5 / HL
        u = g["InternalEnergy"][...]
        m = g["Masses"][...]
    p = (gamma - 1) * rho * u * a * a / HL ** 2
    return a, (pos - 0.5).T, rho, vel[:, 0], p, m / rho


# ---------------------------------------------------------------- metrics (the LUNA tests' definitions)
def adiab_metric(a, gamma, rho, vx, p):
    """(signed mean p~/exact - 1, max |p~/exact - 1|, max |v~/V0 - 1|, max |rho~ - 1|)."""
    ad, _ = _tests()
    r = p / (ad.P0 * (a / ad.A_I) ** (5.0 - 3.0 * gamma)) - 1
    return float(r.mean()), float(np.abs(r).max()), float(np.abs(vx / ad.V0 - 1).max()), float(np.abs(rho - 1).max())


def tophat_metric(a, X, rho, vx, w):
    """(signed core rho~/exact - 1, signed v~ slope/exact - 1, exact 1 + delta, n core): observe() in the test."""
    _, tc = _tests()
    th = tc.tophat()
    ex = th.one_plus_delta([a])[0]
    core = np.sqrt((X ** 2).sum(0)) < 0.5 * tc.R0 * (ex / th.one_plus_delta([tc.A_I])[0]) ** (-1.0 / 3.0)
    x, wc = X[0][core], w[core]
    slope = (wc * vx[core] * x).sum() / (wc * x * x).sum()
    return (float((wc * rho[core]).sum() / wc.sum() / ex - 1), float(slope / (a * a * th.vpec_over_r([a])[0]) - 1),
            float(ex), float(core.sum()))


def series(code, d, gamma, kind):
    """Rows [a, metric...] at every snapshot of one run."""
    rows = []
    if code == "ramses":
        for a, o in ram_snaps(d):
            a, q = ram_grid(o)
            if kind == "adiabatic":
                rows.append([a, *adiab_metric(a, gamma, q[0], q[1], q[4])])
            else:
                n = q.shape[1]
                x = (np.arange(n) + 0.5) / n - 0.5
                X = np.stack(np.meshgrid(x, x, x, indexing="ij"))
                rows.append([a, *tophat_metric(a, X, q[0], q[1], np.ones_like(q[0]))])
    else:
        for a, f in arepo_snaps(d):
            a, X, rho, vx, p, V = arepo_read(f, gamma)
            rows.append([a, *(adiab_metric(a, gamma, rho, vx, p) if kind == "adiabatic"
                              else tophat_metric(a, X, rho, vx, V))])
    return np.array(rows)


def at_targets(rows, targets):
    """Rows [a_target, a_lo, a_hi, metric...]: linear in a between the bracketing snapshots (NaN if none)."""
    out = []
    for t in targets:
        lo, hi = rows[rows[:, 0] < t * (1 - 1e-9)], rows[rows[:, 0] >= t * (1 - 1e-9)]
        if not len(lo) or not len(hi):
            out.append([t, np.nan, np.nan, *[np.nan] * (rows.shape[1] - 1)])
            continue
        lo, hi = lo[-1], hi[0]
        w = (t - lo[0]) / (hi[0] - lo[0])
        out.append([t, lo[0], hi[0], *((1 - w) * lo[1:] + w * hi[1:])])
    return np.array(out)


README = {
    "ramses": "mini-ramses {commit} (Raven, gfortran, MPI, GRAV HYDRO UNITS=COSMO NPSCAL=2): minmod (slope_type 1), HLLC, "
              "courant 0.8, uniform grid (levelmin = levelmax), float32 outputs; top-hat entropy=.false. (total "
              "energy) and _ent (entropy + dual_energy 0.01).",
    "arepo": "Arepo public {commit} (Raven, Intel 19.1.2, IMPI 2019.8, DOUBLEPRECISION=1, double I/O, SELFGRAVITY "
             "PMGRID=128, TREE_BASED_TIMESTEPS, default mesh regularisation, GAMMA=1.4 build for g75): MUSCL-Hancock, "
             "CourantFac 0.3, no (de)refinement; top-hat mesh points jittered 1e-4 dx; core weighted by V = m/rho."}
COLS = {"adiabatic": "a, mean(p~/exact - 1), max|p~/exact - 1|, max|v~/V0 - 1|, max|rho~ - 1|",
        "tophat": "a, core rho~/exact - 1, v~ slope/exact - 1, exact 1 + delta, n core cells (signed errors)"}


def reduce(root):
    ad, tc = _tests()
    for code in ("ramses", "arepo"):
        res = {}
        for run in sorted(glob.glob(os.path.join(root, code, "*"))):
            name = os.path.basename(run)
            kind = "adiabatic" if name.startswith("adiabatic") else "tophat"
            gamma = RUNS[name.replace("_ent", "")]
            rows = series(code, run, gamma, kind)
            if not len(rows):
                print(f"{code} {name}: no snapshots")
                continue
            res[f"{name}_series"] = rows
            res[f"{name}_targets"] = at_targets(rows, ad.A_SNAP if kind == "adiabatic" else tc.A_SNAP)
            print(f"{code} {name}: {len(rows)} snapshots a {rows[0, 0]:.4f} .. {rows[-1, 0]:.4f}")
            for r in res[f"{name}_targets"]:
                print("   a=%g (%.4f..%.4f): " % tuple(r[:3]) + " ".join(f"{x:+.3e}" for x in r[3:]))
        commit = os.environ.get(f"{code.upper()}_COMMIT", "?")
        res["readme"] = np.array(README[code].format(commit=commit) + " Columns: *_series rows [" + COLS["adiabatic"]
                                 + "] (adiabatic) / [" + COLS["tophat"] + "] (tophat); *_targets rows [a_target, a_lo, "
                                 "a_hi, same metrics interpolated in a]. Cross-check context, never gated.")
        os.makedirs(os.path.dirname(_out(code)), exist_ok=True)
        np.savez(_out(code), **res)
        print("wrote", _out(code))


# ---------------------------------------------------------------- LUNA side and plot
def luna(out):
    """LUNA's metrics at the tests' snapshots (adiabatic 8^3 both gamma; top-hat 64^3 and 128^3), signed."""
    ad, tc = _tests()
    res = {}
    for g, tag in ((5.0 / 3.0, "53"), (1.4, "75")):
        res[f"adiabatic_g{tag}"] = np.array(ad.run(g))
        print("LUNA adiabatic", tag, res[f"adiabatic_g{tag}"].tolist())
    for n in (64, 128):
        gates, li, dp = signed_tophat(tc, n)
        res[f"tophat_n{n}"] = np.array([[a, *gates[a]] for a in sorted(gates)])
        print("LUNA tophat", n, res[f"tophat_n{n}"].tolist(), "LI", li)
    np.savez(out, **res)


def signed_tophat(tc, n):
    """tc.run with signed core errors at every output (plot only; the tests' own numbers come from tc.run)."""
    return tc.run(n, signed=True)


def plot(root, luna_npz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ad, tc = _tests()
    R, A = (np.load(_out(c)) for c in ("ramses", "arepo"))
    L = np.load(luna_npz)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    col = dict(luna="#1f5fa8", ramses="#c2571a", arepo="#3a8a3a")
    for g, tag, ls in ((5.0 / 3.0, "53", "-"), (1.4, "75", "--")):
        lab = "5/3" if tag == "53" else "7/5"
        a = np.geomspace(ad.A_I, 1, 100)
        ax[0].plot(a, (a / ad.A_I) ** (5 - 3 * g), "k" + ls, lw=1, label=f"exact gamma {lab}")
        for key, z in (("ramses", R), ("arepo", A)):
            s = z[f"adiabatic_g{tag}_series"]
            ax[0].plot(s[:, 0], (1 + s[:, 1]) * (s[:, 0] / ad.A_I) ** (5 - 3 * g), ls, color=col[key], marker="o",
                       ms=3, lw=0.8, label=f"{key} {lab}")
        lr = L[f"adiabatic_g{tag}"]
        ax[0].plot(lr[:, 0], (1 + lr[:, 1]) * (lr[:, 0] / ad.A_I) ** (5 - 3 * g), "s", color=col["luna"], ms=6,
                   mfc="none", label=f"LUNA {lab} (max |err|)")
    ax[0].set(xscale="log", yscale="log", xlabel="a", ylabel="p~ / p~(a_i)", title="A16a uniform gas, EdS, 8^3")
    ax[0].legend(fontsize=7, ncol=2)
    th = tc.tophat()
    a = np.geomspace(tc.A_I, max(tc.A_SNAP), 200)
    ax[1].plot(a, th.one_plus_delta(a), "k", lw=1, label="closed FRW")
    for n, mk in ((64, "o"), (128, "^")):
        for key, z, run in (("ramses", R, f"tophat_n{n}"), ("arepo", A, f"tophat_n{n}")):
            if f"{run}_series" not in z:
                continue
            s = z[f"{run}_series"]
            ax[1].plot(s[:, 0], (1 + s[:, 1]) * s[:, 3], marker=mk, color=col[key], ms=3, lw=0.6, label=f"{key} {n}^3")
            ax[2].plot(s[:, 0], np.abs(s[:, 1]), marker=mk, color=col[key], ms=3, lw=0.6, label=f"{key} {n}^3")
        lr = L[f"tophat_n{n}"]
        ex = th.one_plus_delta(lr[:, 0])
        ax[1].plot(lr[:, 0], (1 + lr[:, 1]) * ex, mk, color=col["luna"], ms=7, mfc="none", label=f"LUNA {n}^3")
        ax[2].plot(lr[:, 0], np.abs(lr[:, 1]), mk + "-", color=col["luna"], ms=7, mfc="none", label=f"LUNA {n}^3")
    for x in (tc.A_TA, max(tc.A_SNAP)):
        ax[2].axvline(x, color="0.6", lw=0.6)
    ax[1].set(xscale="log", yscale="log", xlabel="a", ylabel="core rho~", title="A16b cold top-hat, turnaround a = 0.5")
    ax[2].set(xscale="log", yscale="log", xlabel="a", ylabel="|core rho~ / exact - 1|", ylim=(1e-5, 1))
    ax[1].legend(fontsize=7)
    ax[2].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    png = os.path.join(root, "a16_xcheck.png")
    fig.savefig(png, dpi=130)
    print("wrote", png)


if __name__ == "__main__":
    {"gen": gen, "reduce": reduce, "luna": luna, "plot": plot}[sys.argv[1]](*sys.argv[2:])
