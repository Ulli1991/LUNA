"""LANL LA-14379 1D Riemann problems (Peak, RCVCR, vacuum expansion, stationary contact, slow shock,
stream collision) against the exact Riemann solution: errors, orders, budgets, positivity, symptoms.

States, domains and times are from J. R. Kamm et al., "Enhanced Verification Test Suite for Physics
Simulation Codes", LA-14379 (2008; OSTI 957482), section "The 1-D Riemann Problem", III (report
pp. 41-43), gamma = 7/5 for all of them (p. 39):
  RCVCR (p. 41): (1, -4, 0.4) | (1, 4, 0.4), 0 <= x <= 1, x_i = 0.5, t_f = 0.15.
  Vacuum expansion (p. 41): (0, 0, 0) | (1, 0, 1), -5 <= x <= 1, x_i = 0, t_f = 0.75.
  Stream collision (p. 42): (1, +2, 0.1) | (1, -2, 0.1), 0 <= x <= 1, x_i = 0.5, t_f = 0.8.
  Peak (p. 42): (0.1261192, 8.9047029, 782.92899) | (6.591493, 2.2654207, 3.1544874),
    0.1 <= x <= 0.6, x_i = 0.5, t_f = 3.9e-3.
  Slow shock (p. 43): (3.857143, -0.810631, 10.33333) | (1, -3.44, 1), 0 <= x <= 1, x_i = 0.5,
    t_f = 30; the shock (0.109648 cm/s, p. 42) leaves [0, 1] at t = 4.6, so the ladder stops at
    t = 1 as ExactPack's Test_Riemann4 does (exactpack/tests/test_riemann.py, lanl/ExactPack master;
    its p_L = 31/3), and the report's t_f = 30 runs on [0, 4.5] (test_slow_shock_to_report_time).
  Stationary contact (p. 43): (1, -19.59745, 1e3) | (1, -19.59745, 1e-2), 0 <= x <= 1, x_i = 0.8,
    t_f = 0.012 (also ExactPack Test_Riemann3: u* = 1.39e-6, not 0, from the rounded u).
The exact solution is luna.hydro.exact_riemann, except RCVCR, which generates vacuum
(2 (a_L + a_R) / (gamma - 1) = 7.48 < u_R - u_L = 8; Toro 2009 sec. 4.6), sampled here.
The true vacuum state of the vacuum expansion is neither solved nor refused: rho = p = 0 gives a 0/0 sound speed,
the first dt is NaN and the run crashes at t = 0 (open bug, a strict xfail). A near-vacuum left state stands in."""
import numpy as np
import pytest

from luna.hydro import exact_riemann as er
from tests._helpers import RunAborted, accuracy_cfg, check, check_mirror, check_tube_budget, evolve

G = 1.4
# name: (W_L, W_R, x0, (x_lo, x_hi), t_f, t_budget). t_budget < t_f while every wave is inside the box
# (the end faces see the constant states): the budget is exact there; RCVCR's fans and the slow shock's
# start-up pulse leave the box before t_f.
CASES = {
    "peak": ((0.1261192, 8.9047029, 782.92899), (6.591493, 2.2654207, 3.1544874), 0.5, (0.1, 0.6), 3.9e-3, 3.9e-3),
    "rcvcr": ((1.0, -4.0, 0.4), (1.0, 4.0, 0.4), 0.5, (0.0, 1.0), 0.15, 0.09),
    "stationary": ((1.0, -19.59745, 1e3), (1.0, -19.59745, 1e-2), 0.8, (0.0, 1.0), 0.012, 0.012),
    "slow": ((3.857143, -0.810631, 10.33333), (1.0, -3.44, 1.0), 0.5, (0.0, 1.0), 1.0, 0.1),
    "streams": ((1.0, 2.0, 0.1), (1.0, -2.0, 0.1), 0.5, (0.0, 1.0), 0.8, 0.8),
}
# Near vacuum (not in the report): the vacuum expansion with rho_L = p_L = 1e-6 of the right state.
VACUUM = ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), 0.0, (-5.0, 1.0), 0.75)
NEAR_VACUUM = 1e-6
NS = {"peak": (400, 800, 1600)}         # the report's nominal 400 zones for Peak (p. 44), 100 elsewhere
NS_DEFAULT = (256, 512, 1024)
NS_VACUUM = (768, 1536, 3072)           # 128, 256, 512 cells per cm on the 6 cm box
SCHEMES = {"minmod": ("hllc", 0), "mc": ("hllc", 1), "vanleer": ("hllc", 2), "hll-minmod": ("hll", 0)}

# Per case and scheme (HLLC with minmod, MC, van Leer; HLL with minmod), CPU, FP64, default dt; ~1.3x
# measured: L1, L2 {N: (rho, u, p)} (integrals over the box; RCVCR's u only where the exact gas is);
# orders fitted over the ladder, measured -0.1 .. +0.3; symptoms 1.3x measured, contact widths
# measured + 1 cell. peak_error |max rho / rho*_R - 1|; centre_rho max rho within 0.02 of x_i (exact 0);
# contact_width_cells strictly inside 10-90 % of the jump, contact_drift_cells |x_50% - u* t - x_i| / h;
# post_shock_osc max |rho / rho_L - 1| on [0.05, x_shock - 0.05]; centre_rho_dip 1 - min rho / rho*
# and centre_e_overshoot max e / e* - 1 within 0.1 of x_i (wall heating, flat in N: not a convergence).
# MC's mirror asymmetry (test_mirror_symmetry) is in its rcvcr and streams numbers.
BOUNDS = {
    'peak': {
        'minmod': dict(L1={400: (0.2, 0.18, 2.9), 800: (0.12, 0.096, 1.7), 1600: (0.075, 0.048, 0.83)},
            L2={400: (1.7, 0.6, 21.0), 800: (1.4, 0.45, 20.0), 1600: (1.2, 0.3, 14.0)},
            L1_order=((0.59, 1.0), (0.83, 1.24), (0.8, 1.21)),
            L2_order=((0.19, 0.6), (0.38, 0.79), (0.18, 0.59)),
            peak_error={400: 0.19, 800: 0.038, 1600: 0.00074}),
        # minmod measured L1 0.15 0.134 2.23 | 0.091 0.0732 1.27 | 0.0576 0.0369 0.637;
        #   L2 1.28 0.455 15.7 | 1.06 0.34 15 | 0.852 0.231 10.6;
        #   L1 orders 0.691 0.932 0.904;
        #   L2 orders 0.297 0.489 0.284;
        #   peak_error 0.144 0.029 0.000566
        'mc': dict(L1={400: (0.14, 0.15, 2.3), 800: (0.073, 0.076, 1.3), 1600: (0.046, 0.039, 0.61)},
            L2={400: (1.4, 0.53, 16.0), 800: (1.2, 0.39, 16.0), 1600: (0.89, 0.26, 11.0)},
            L1_order=((0.7, 1.11), (0.86, 1.27), (0.86, 1.27)),
            L2_order=((0.21, 0.62), (0.41, 0.82), (0.18, 0.59)),
            peak_error={400: 0.069, 800: 0.0014, 1600: 0.0044}),
        # mc measured L1 0.107 0.111 1.76 | 0.0557 0.0582 0.946 | 0.0349 0.0293 0.465;
        #   L2 1.06 0.404 12.3 | 0.854 0.298 12.1 | 0.683 0.198 8.2;
        #   L1 orders 0.809 0.963 0.961;
        #   L2 orders 0.316 0.515 0.290;
        #   peak_error 0.0527 0.00104 0.00336
        'vanleer': dict(L1={400: (0.16, 0.16, 2.5), 800: (0.085, 0.081, 1.4), 1600: (0.053, 0.041, 0.67)},
            L2={400: (1.5, 0.56, 18.0), 800: (1.2, 0.41, 17.0), 1600: (0.96, 0.28, 12.0)},
            L1_order=((0.68, 1.09), (0.86, 1.27), (0.84, 1.25)),
            L2_order=((0.2, 0.61), (0.41, 0.82), (0.18, 0.59)),
            peak_error={400: 0.1, 800: 0.0038, 1600: 0.0028}),
        # vanleer measured L1 0.12 0.119 1.9 | 0.0652 0.0623 1.03 | 0.0403 0.0312 0.51;
        #   L2 1.12 0.423 13.3 | 0.911 0.311 13 | 0.736 0.208 8.93;
        #   L1 orders 0.785 0.963 0.949;
        #   L2 orders 0.304 0.512 0.286;
        #   peak_error 0.0767 0.0029 0.00212
        'hll-minmod': dict(L1={400: (0.21, 0.16, 2.8), 800: (0.13, 0.089, 1.6), 1600: (0.078, 0.045, 0.79)},
            L2={400: (1.7, 0.54, 21.0), 800: (1.4, 0.43, 20.0), 1600: (1.2, 0.3, 14.0)},
            L1_order=((0.58, 0.99), (0.82, 1.23), (0.79, 1.2)),
            L2_order=((0.19, 0.6), (0.33, 0.74), (0.17, 0.58)),
            peak_error={400: 0.2, 800: 0.039, 1600: 0.0008}),
        # hll-minmod measured L1 0.155 0.123 2.09 | 0.0946 0.0678 1.21 | 0.0596 0.0342 0.606;
        #   L2 1.3 0.414 15.5 | 1.07 0.329 15 | 0.863 0.226 10.6;
        #   L1 orders 0.689 0.922 0.894;
        #   L2 orders 0.297 0.438 0.274;
        #   peak_error 0.147 0.0297 0.000611
    },
    'rcvcr': {
        'minmod': dict(L1={256: (0.0046, 0.056, 0.0012), 512: (0.0025, 0.049, 0.00048), 1024: (0.0014, 0.038, 0.00021)},
            L2={256: (0.0056, 0.065, 0.0013), 512: (0.0031, 0.06, 0.00054), 1024: (0.0016, 0.05, 0.00024)},
            L1_order=((0.79, 1.2), (0.18, 0.59), (1.13, 1.54)),
            L2_order=((0.79, 1.2), (0.1, 0.51), (1.12, 1.53)),
            centre_rho={256: 0.0027, 512: 0.0016, 1024: 0.0009}),
        # minmod measured L1 0.00349 0.043 0.000886 | 0.0019 0.0376 0.000366 | 0.00102 0.0291 0.000159;
        #   L2 0.00424 0.0499 0.00097 | 0.00232 0.0461 0.000414 | 0.00122 0.0378 0.000179;
        #   L1 orders 0.891 0.282 1.240;
        #   L2 orders 0.897 0.200 1.220;
        #   centre_rho 0.00202 0.00118 0.000686
        'mc': dict(L1={256: (0.0037, 0.039, 0.0009), 512: (0.0019, 0.027, 0.0004), 1024: (0.00096, 0.018, 0.00017)},
            L2={256: (0.0042, 0.049, 0.00094), 512: (0.002, 0.035, 0.00043), 1024: (0.0011, 0.025, 0.00019)},
            L1_order=((0.86, 1.27), (0.45, 0.86), (1.1, 1.51)),
            L2_order=((0.9, 1.31), (0.4, 0.81), (1.06, 1.47)),
            centre_rho={256: 0.0029, 512: 0.0017, 1024: 0.00094}),
        # mc measured L1 0.0028 0.0295 0.000685 | 0.00144 0.0202 0.000303 | 0.000733 0.0136 0.000129;
        #   L2 0.0032 0.0374 0.000722 | 0.00154 0.0265 0.000325 | 0.000789 0.0186 0.000144;
        #   L1 orders 0.967 0.558 1.202;
        #   L2 orders 1.009 0.505 1.162;
        #   centre_rho 0.00222 0.0013 0.000722
        'vanleer': dict(L1={256: (0.004, 0.044, 0.00095), 512: (0.0021, 0.035, 0.00041), 1024: (0.0011, 0.026, 0.00018)},
            L2={256: (0.0045, 0.052, 0.0011), 512: (0.0023, 0.044, 0.00045), 1024: (0.0012, 0.034, 0.0002)},
            L1_order=((0.84, 1.25), (0.3, 0.71), (1.1, 1.51)),
            L2_order=((0.86, 1.27), (0.2, 0.61), (1.1, 1.51)),
            centre_rho={256: 0.0028, 512: 0.0017, 1024: 0.00093}),
        # vanleer measured L1 0.00301 0.0338 0.000725 | 0.00158 0.0265 0.000315 | 0.000814 0.0193 0.000135;
        #   L2 0.00339 0.0397 0.000781 | 0.00173 0.0335 0.000339 | 0.000895 0.0261 0.000147;
        #   L1 orders 0.943 0.406 1.210;
        #   L2 orders 0.961 0.303 1.204;
        #   centre_rho 0.00215 0.00124 0.000711
        'hll-minmod': dict(L1={256: (0.0046, 0.056, 0.0012), 512: (0.0025, 0.049, 0.00048), 1024: (0.0014, 0.038, 0.00021)},
            L2={256: (0.0056, 0.065, 0.0013), 512: (0.0031, 0.06, 0.00054), 1024: (0.0016, 0.05, 0.00024)},
            L1_order=((0.79, 1.2), (0.18, 0.59), (1.13, 1.54)),
            L2_order=((0.79, 1.2), (0.1, 0.51), (1.12, 1.53)),
            centre_rho={256: 0.0027, 512: 0.0016, 1024: 0.0009}),
        # hll-minmod measured L1 0.00349 0.0431 0.000886 | 0.0019 0.0376 0.000366 | 0.00102 0.0291 0.000159;
        #   L2 0.00424 0.0499 0.00097 | 0.00232 0.0461 0.000414 | 0.00122 0.0378 0.000179;
        #   L1 orders 0.891 0.282 1.240;
        #   L2 orders 0.897 0.201 1.220;
        #   centre_rho 0.00203 0.00118 0.000688
    },
    'stationary': {
        'minmod': dict(L1={256: (0.02, 0.14, 2.7), 512: (0.011, 0.062, 1.4), 1024: (0.0056, 0.027, 0.7)},
            L2={256: (0.15, 0.85, 11.0), 512: (0.1, 0.46, 5.4), 1024: (0.094, 0.12, 4.4)},
            L1_order=((0.81, 1.22), (1.04, 1.45), (0.86, 1.27)),
            L2_order=((0.23, 0.64), (1.31, 1.72), (0.52, 0.93)),
            contact_width_cells={256: 2.0, 512: 3.0, 1024: 2.0},
            contact_drift_cells={256: 0.34, 512: 0.44, 1024: 0.35}),
        # minmod measured L1 0.0152 0.1 2.03 | 0.00794 0.0471 1.04 | 0.00428 0.0205 0.536;
        #   L2 0.114 0.652 7.95 | 0.0763 0.352 4.11 | 0.0716 0.0913 3.32;
        #   L1 orders 0.913 1.144 0.960;
        #   L2 orders 0.338 1.418 0.630;
        #   contact_width_cells 1 2 1;
        #   contact_drift_cells 0.258 0.331 0.263
        'mc': dict(L1={256: (0.022, 0.16, 3.3), 512: (0.012, 0.07, 1.7), 1024: (0.006, 0.023, 0.68)},
            L2={256: (0.16, 0.78, 11.0), 512: (0.12, 0.36, 7.3), 1024: (0.1, 0.084, 5.4)},
            L1_order=((0.83, 1.24), (1.26, 1.67), (1.02, 1.43)),
            L2_order=((0.2, 0.61), (1.5, 1.91), (0.39, 0.8)),
            contact_width_cells={256: 3.0, 512: 3.0, 1024: 2.0},
            contact_drift_cells={256: 0.36, 512: 0.44, 1024: 0.34}),
        # mc measured L1 0.0167 0.116 2.46 | 0.00881 0.0533 1.25 | 0.00456 0.0175 0.517;
        #   L2 0.117 0.593 8.2 | 0.0856 0.276 5.59 | 0.0769 0.064 4.12;
        #   L1 orders 0.937 1.366 1.126;
        #   L2 orders 0.303 1.606 0.496;
        #   contact_width_cells 2 2 1;
        #   contact_drift_cells 0.274 0.332 0.261
        'vanleer': dict(L1={256: (0.021, 0.14, 2.8), 512: (0.011, 0.065, 1.5), 1024: (0.0055, 0.022, 0.63)},
            L2={256: (0.15, 0.79, 9.6), 512: (0.11, 0.39, 6.4), 1024: (0.095, 0.079, 4.8)},
            L1_order=((0.83, 1.24), (1.22, 1.63), (0.97, 1.38)),
            L2_order=((0.18, 0.59), (1.56, 1.97), (0.39, 0.8)),
            contact_width_cells={256: 2.0, 512: 3.0, 1024: 2.0},
            contact_drift_cells={256: 0.34, 512: 0.44, 1024: 0.32}),
        # vanleer measured L1 0.0154 0.102 2.12 | 0.00811 0.0495 1.13 | 0.00421 0.0164 0.477;
        #   L2 0.108 0.602 7.35 | 0.0813 0.297 4.85 | 0.073 0.0602 3.68;
        #   L1 orders 0.936 1.320 1.077;
        #   L2 orders 0.284 1.661 0.499;
        #   contact_width_cells 1 2 1;
        #   contact_drift_cells 0.261 0.332 0.246
        'hll-minmod': dict(L1={256: (0.076, 0.15, 3.1), 512: (0.044, 0.071, 1.6), 1024: (0.03, 0.031, 0.75)},
            L2={256: (0.37, 0.95, 15.0), 512: (0.28, 0.56, 7.4), 1024: (0.23, 0.2, 5.0)},
            L1_order=((0.56, 0.97), (1.03, 1.44), (0.9, 1.31)),
            L2_order=((0.24, 0.65), (1.05, 1.46), (0.65, 1.06)),
            contact_width_cells={256: 9.0, 512: 11.0, 1024: 14.0},
            contact_drift_cells={256: 0.81, 512: 0.94, 1024: 1.1}),
        # hll-minmod measured L1 0.0579 0.111 2.32 | 0.0338 0.0544 1.17 | 0.0229 0.0231 0.571;
        #   L2 0.284 0.729 10.8 | 0.209 0.427 5.62 | 0.177 0.148 3.79;
        #   L1 orders 0.669 1.132 1.010;
        #   L2 orders 0.343 1.152 0.756;
        #   contact_width_cells 8 10 13;
        #   contact_drift_cells 0.62 0.718 0.829
    },
    'slow': {
        'minmod': dict(L1={256: (0.0099, 0.0063, 0.033), 512: (0.0063, 0.0033, 0.021), 1024: (0.0039, 0.0018, 0.013)},
            L2={256: (0.065, 0.057, 0.18), 512: (0.042, 0.025, 0.099), 1024: (0.025, 0.0064, 0.049)},
            L1_order=((0.58, 0.99), (0.83, 1.24), (0.55, 0.96)),
            L2_order=((0.59, 1.0), (1.47, 1.88), (0.81, 1.22)),
            post_shock_osc={256: 0.0039, 512: 0.0038, 1024: 0.0035}),
        # minmod measured L1 0.0076 0.00484 0.0247 | 0.00481 0.00252 0.0159 | 0.00294 0.00133 0.0099;
        #   L2 0.0498 0.0432 0.134 | 0.0317 0.0186 0.0757 | 0.0189 0.00489 0.0376;
        #   L1 orders 0.686 0.932 0.659;
        #   L2 orders 0.700 1.572 0.915;
        #   post_shock_osc 0.00299 0.00288 0.00263
        'mc': dict(L1={256: (0.016, 0.0077, 0.043), 512: (0.012, 0.0051, 0.035), 1024: (0.0091, 0.004, 0.031)},
            L2={256: (0.056, 0.051, 0.14), 512: (0.04, 0.021, 0.09), 1024: (0.028, 0.0062, 0.062)},
            L1_order=((0.27, 0.68), (0.38, 0.79), (0.14, 0.55)),
            L2_order=((0.4, 0.81), (1.41, 1.82), (0.48, 0.89)),
            post_shock_osc={256: 0.022, 512: 0.014, 1024: 0.0074}),
        # mc measured L1 0.0117 0.00591 0.0325 | 0.00886 0.00389 0.0269 | 0.00697 0.00302 0.0233;
        #   L2 0.0424 0.0385 0.105 | 0.0302 0.0161 0.0691 | 0.0209 0.00473 0.047;
        #   L1 orders 0.376 0.485 0.241;
        #   L2 orders 0.509 1.512 0.583;
        #   post_shock_osc 0.0162 0.01 0.00564
        'vanleer': dict(L1={256: (0.011, 0.0063, 0.033), 512: (0.0079, 0.0038, 0.025), 1024: (0.0061, 0.0027, 0.021)},
            L2={256: (0.056, 0.051, 0.15), 512: (0.038, 0.021, 0.086), 1024: (0.026, 0.0048, 0.053)},
            L1_order=((0.31, 0.72), (0.53, 0.94), (0.24, 0.65)),
            L2_order=((0.47, 0.88), (1.6, 2.01), (0.62, 1.03)),
            post_shock_osc={256: 0.011, 512: 0.0063, 1024: 0.0037}),
        # vanleer measured L1 0.00823 0.00484 0.0249 | 0.00606 0.00288 0.0191 | 0.00464 0.00201 0.0155;
        #   L2 0.0427 0.0388 0.11 | 0.0287 0.0159 0.066 | 0.0193 0.00367 0.0401;
        #   L1 orders 0.413 0.634 0.340;
        #   L2 orders 0.571 1.701 0.727;
        #   post_shock_osc 0.00797 0.00477 0.00281
        'hll-minmod': dict(L1={256: (0.012, 0.0071, 0.037), 512: (0.0071, 0.0038, 0.024), 1024: (0.0044, 0.002, 0.015)},
            L2={256: (0.072, 0.061, 0.2), 512: (0.046, 0.028, 0.12), 1024: (0.027, 0.0076, 0.056)},
            L1_order=((0.58, 0.99), (0.81, 1.22), (0.55, 0.96)),
            L2_order=((0.61, 1.02), (1.39, 1.8), (0.81, 1.22)),
            post_shock_osc={256: 0.0043, 512: 0.0041, 1024: 0.0039}),
        # hll-minmod measured L1 0.00858 0.00546 0.0283 | 0.00542 0.00288 0.0182 | 0.00331 0.00154 0.0114;
        #   L2 0.0552 0.0467 0.151 | 0.0351 0.0208 0.0867 | 0.0204 0.00584 0.0428;
        #   L1 orders 0.686 0.914 0.658;
        #   L2 orders 0.717 1.499 0.911;
        #   post_shock_osc 0.00327 0.00313 0.00298
    },
    'streams': {
        'minmod': dict(L1={256: (0.014, 0.0035, 0.0036), 512: (0.0094, 0.00077, 0.0045), 1024: (0.0045, 0.0003, 0.0014)},
            L2={256: (0.064, 0.027, 0.02), 512: (0.068, 0.0082, 0.043), 1024: (0.041, 0.0055, 0.018)},
            L1_order=((0.68, 1.09), (1.67, 2.08), (0.61, 1.02)),
            L2_order=((0.21, 0.62), (1.03, 1.44), (-0.05, 0.36)),
            centre_rho_dip={256: 0.05, 512: 0.05, 1024: 0.05},
            centre_e_overshoot={256: 0.052, 512: 0.052, 1024: 0.052}),
        # minmod measured L1 0.0103 0.00265 0.00273 | 0.00718 0.000587 0.00341 | 0.00345 0.000225 0.00101;
        #   L2 0.0486 0.0202 0.0147 | 0.0522 0.00629 0.033 | 0.0313 0.00416 0.0135;
        #   L1 orders 0.787 1.779 0.715;
        #   L2 orders 0.317 1.139 0.058;
        #   centre_rho_dip 0.038 0.038 0.038;
        #   centre_e_overshoot 0.0395 0.0395 0.0395
        'mc': dict(L1={256: (0.013, 0.0028, 0.0044), 512: (0.0066, 0.0013, 0.0026), 1024: (0.0038, 0.00075, 0.002)},
            L2={256: (0.046, 0.025, 0.022), 512: (0.03, 0.013, 0.021), 1024: (0.021, 0.014, 0.023)},
            L1_order=((0.76, 1.17), (0.84, 1.25), (0.48, 0.89)),
            L2_order=((0.47, 0.88), (0.34, 0.75), (-0.14, 0.27)),
            centre_rho_dip={256: 0.049, 512: 0.049, 1024: 0.049},
            centre_e_overshoot={256: 0.051, 512: 0.051, 1024: 0.051}),
        # mc measured L1 0.00971 0.00214 0.00337 | 0.00501 0.000975 0.00196 | 0.00291 0.000576 0.00149;
        #   L2 0.0348 0.0191 0.0164 | 0.023 0.00991 0.0159 | 0.0156 0.0103 0.0173;
        #   L1 orders 0.868 0.948 0.589;
        #   L2 orders 0.576 0.445 -0.038;
        #   centre_rho_dip 0.0374 0.0374 0.0374;
        #   centre_e_overshoot 0.0389 0.0388 0.0388
        'vanleer': dict(L1={256: (0.013, 0.0028, 0.0036), 512: (0.0071, 0.00063, 0.0016), 1024: (0.0042, 0.00053, 0.0011)},
            L2={256: (0.051, 0.023, 0.018), 512: (0.037, 0.006, 0.011), 1024: (0.024, 0.011, 0.011)},
            L1_order=((0.69, 1.1), (1.1, 1.51), (0.81, 1.22)),
            L2_order=((0.45, 0.86), (0.47, 0.88), (0.25, 0.66)),
            centre_rho_dip={256: 0.055, 512: 0.055, 1024: 0.055},
            centre_e_overshoot={256: 0.057, 512: 0.057, 1024: 0.057}),
        # vanleer measured L1 0.00955 0.00214 0.00274 | 0.00539 0.000482 0.00118 | 0.00318 0.000405 0.000773;
        #   L2 0.039 0.0171 0.0136 | 0.0283 0.00456 0.00777 | 0.018 0.00775 0.00833;
        #   L1 orders 0.793 1.200 0.914;
        #   L2 orders 0.557 0.571 0.352;
        #   centre_rho_dip 0.0416 0.0416 0.0416;
        #   centre_e_overshoot 0.0434 0.0434 0.0434
        'hll-minmod': dict(L1={256: (0.012, 0.0044, 0.0041), 512: (0.01, 0.00093, 0.0067), 1024: (0.004, 0.00042, 0.0019)},
            L2={256: (0.055, 0.032, 0.025), 512: (0.087, 0.0098, 0.07), 1024: (0.047, 0.0067, 0.027)},
            L1_order=((0.68, 1.09), (1.6, 2.01), (0.45, 0.86)),
            L2_order=((0.01, 0.42), (1.02, 1.43), (-0.17, 0.24)),
            centre_rho_dip={256: 0.013, 512: 0.0096, 1024: 0.0076},
            centre_e_overshoot={256: 0.013, 512: 0.0096, 1024: 0.0076}),
        # hll-minmod measured L1 0.00903 0.00334 0.00313 | 0.00769 0.000712 0.0051 | 0.00306 0.000315 0.00145;
        #   L2 0.0419 0.0244 0.0189 | 0.0669 0.00747 0.0533 | 0.0355 0.00512 0.0206;
        #   L1 orders 0.781 1.702 0.553;
        #   L2 orders 0.120 1.126 -0.061;
        #   centre_rho_dip 0.00933 0.00733 0.00577;
        #   centre_e_overshoot 0.00941 0.00738 0.00581
    },
}
# Near vacuum, as BOUNDS on NS_VACUUM; u where rho_exact > 1e-3 rho_R.
VACUUM_BOUNDS = {
        'minmod': dict(L1={768: (0.0052, 0.079, 0.0042), 1536: (0.0027, 0.041, 0.0022), 3072: (0.0015, 0.021, 0.0011)},
            L2={768: (0.0045, 0.07, 0.0058), 1536: (0.0026, 0.037, 0.0033), 3072: (0.0014, 0.019, 0.0019)},
            L1_order=((0.82, 1.23), (0.86, 1.27), (0.87, 1.28)),
            L2_order=((0.74, 1.15), (0.83, 1.24), (0.73, 1.14))),
        # minmod measured L1 0.00396 0.0605 0.0032 | 0.00207 0.0312 0.00162 | 0.0011 0.0159 0.000827;
        #   L2 0.00345 0.0537 0.00444 | 0.00193 0.0283 0.00251 | 0.00107 0.0146 0.0014;
        #   L1 orders 0.925 0.965 0.977;
        #   L2 orders 0.845 0.940 0.832
        'mc': dict(L1={768: (0.0042, 0.071, 0.0031), 1536: (0.0022, 0.035, 0.0016), 3072: (0.0012, 0.018, 0.0008)},
            L2={768: (0.0028, 0.061, 0.003), 1536: (0.0014, 0.03, 0.0016), 3072: (0.00072, 0.015, 0.00078)},
            L1_order=((0.82, 1.23), (0.92, 1.33), (0.86, 1.27)),
            L2_order=((0.86, 1.27), (0.91, 1.32), (0.87, 1.28))),
        # mc measured L1 0.00321 0.0541 0.00235 | 0.00168 0.0266 0.0012 | 0.000896 0.0131 0.000615;
        #   L2 0.0021 0.0462 0.0023 | 0.00107 0.0229 0.00117 | 0.000553 0.0113 0.000596;
        #   L1 orders 0.921 1.023 0.966;
        #   L2 orders 0.960 1.018 0.974
        'vanleer': dict(L1={768: (0.0043, 0.073, 0.0032), 1536: (0.0023, 0.036, 0.0016), 3072: (0.0012, 0.018, 0.00082)},
            L2={768: (0.0031, 0.063, 0.0035), 1536: (0.0016, 0.032, 0.0019), 3072: (0.00082, 0.016, 0.00095)},
            L1_order=((0.82, 1.23), (0.91, 1.32), (0.86, 1.27)),
            L2_order=((0.84, 1.25), (0.9, 1.31), (0.84, 1.25))),
        # vanleer measured L1 0.00327 0.0557 0.00241 | 0.00171 0.0276 0.00123 | 0.000913 0.0137 0.00063;
        #   L2 0.00231 0.0484 0.00268 | 0.0012 0.0242 0.0014 | 0.000627 0.012 0.000726;
        #   L1 orders 0.921 1.015 0.967;
        #   L2 orders 0.940 1.006 0.942
        'hll-minmod': dict(L1={768: (0.0057, 0.093, 0.0044), 1536: (0.003, 0.049, 0.0023), 3072: (0.0016, 0.025, 0.0012)},
            L2={768: (0.0047, 0.083, 0.0059), 1536: (0.0027, 0.044, 0.0034), 3072: (0.0015, 0.023, 0.0019)},
            L1_order=((0.82, 1.23), (0.85, 1.26), (0.87, 1.28)),
            L2_order=((0.75, 1.16), (0.82, 1.23), (0.73, 1.14))),
        # hll-minmod measured L1 0.00432 0.0711 0.00337 | 0.00227 0.037 0.00171 | 0.0012 0.0189 0.00087;
        #   L2 0.0036 0.0632 0.00453 | 0.002 0.0336 0.00256 | 0.00111 0.0174 0.00142;
        #   L1 orders 0.923 0.957 0.976;
        #   L2 orders 0.852 0.929 0.836
}
# Slow shock at t = 30 (SLOW_LONG): L1 (rho, u, p) and post_shock_osc, 1.3x measured.
SLOW_LONG_BOUNDS = {
    'minmod': dict(L1=(0.019, 0.012, 0.071), osc=0.0047),  # measured L1 0.0144 0.00886 0.0545; osc 0.00359
    'mc': dict(L1=(0.15, 0.11, 0.49), osc=0.014),  # measured L1 0.109 0.0845 0.374; osc 0.0102
    'vanleer': dict(L1=(0.05, 0.035, 0.18), osc=0.0095),  # measured L1 0.0384 0.0266 0.135; osc 0.00731
    'hll-minmod': dict(L1=(0.022, 0.015, 0.083), osc=0.0042),  # measured L1 0.0168 0.0109 0.0635; osc 0.00322
}


def _ns(case):
    """The resolution ladder of a case."""
    return NS.get(case, NS_DEFAULT)


def _vacuum_sample(WL, WR, g, xi):
    """(rho, u, p) of a Riemann problem that generates vacuum (Toro sec. 4.6); u = 0 in the vacuum."""
    xi = np.asarray(xi, dtype=np.float64)
    (_, ul, _), (_, ur, _) = WL, WR
    al, ar = (np.sqrt(g * W[2] / W[0]) for W in (WL, WR))
    sl, sr = ul + 2.0 * al / (g - 1.0), ur - 2.0 * ar / (g - 1.0)
    assert sl < sr, "no vacuum is generated"
    out = [np.zeros_like(xi) for _ in range(3)]
    for W, s, region, fan in ((WL, -1.0, xi < ul - al, (xi >= ul - al) & (xi < sl)),
                              (WR, 1.0, xi >= ur + ar, (xi > sr) & (xi < ur + ar))):
        f = er._fan(W, xi[fan], g, s)
        for k in range(3):
            out[k][region], out[k][fan] = W[k], f[k]
    return tuple(out)


def exact(case_or_states, edges, t, sub=256):
    """(rho, u, p) cell averages of the exact solution, conservative; u = 0 in empty cells."""
    WL, WR, x0 = case_or_states[:3]
    vac = 2.0 * sum(np.sqrt(G * W[2] / W[0]) for W in (WL, WR)) / (G - 1.0) < WR[1] - WL[1]
    f = (np.arange(sub) + 0.5) / sub
    x = edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * f[None, :]
    rho, u, p = (_vacuum_sample if vac else er.sample)(WL, WR, G, (x - x0) / t)
    m, mom, E = rho.mean(1), (rho * u).mean(1), (p / (G - 1.0) + 0.5 * rho * u * u).mean(1)
    uu = np.divide(mom, m, out=np.zeros_like(m), where=m > 0)
    return m, uu, (G - 1.0) * (E - 0.5 * m * uu * uu)


def initial(WL, WR, x0, edges):
    """(rho, rho u, E) cell averages of the step W_L | W_R at x0."""
    frac = np.clip((x0 - edges[:-1]) / (edges[1:] - edges[:-1]), 0.0, 1.0)
    cons = [np.array([r, r * u, p / (G - 1.0) + 0.5 * r * u * u]) for r, u, p in (WL, WR)]
    return frac[None, :] * cons[0][:, None] + (1.0 - frac[None, :]) * cons[1][:, None]


def _prim(U):
    """(rho, u, p) from 1D conserved."""
    rho, u = U[0], U[1] / U[0]
    return rho, u, (G - 1.0) * (U[2] - 0.5 * rho * u * u)


# MC's and van Leer's predicted faces go non-physical next to the forming shocks (audit C8): those runs allow
# the first-order face state, and each asserts its count of such faces, FACES[(scheme, case, N)] = (lo, hi),
# exact (CPU-8 job 31665570 and A100 job 31665569 agree); a run not listed must have none.
FACES = {("mc", "peak", 400): (5, 5),
         ("mc", "peak", 800): (5, 5),
         ("mc", "peak", 1600): (5, 5),
         ("mc", "rcvcr", 256): (2, 2),
         ("mc", "rcvcr", 512): (2, 2),
         ("mc", "rcvcr", 1024): (2, 2),
         ("mc", "stationary", 256): (146, 146),
         ("mc", "stationary", 512): (280, 280),
         ("mc", "stationary", 1024): (562, 562),
         ("vanleer", "stationary", 256): (88, 88),
         ("vanleer", "stationary", 512): (181, 181),
         ("vanleer", "stationary", 1024): (355, 355),
         ("mc", "streams", 256): (230, 230),
         ("mc", "streams", 512): (460, 460),
         ("mc", "streams", 1024): (942, 942)}
C8_SCHEMES = ("mc", "vanleer")


def cfg_for(n, length, scheme):
    """The accuracy Config of an n-cell zero-gradient tube of this length."""
    solver, limiter = SCHEMES[scheme]
    return accuracy_cfg(SPATIAL_DIMENSION=1, Nx=n, Lx=length, BOUNDARY=1, gamma=G, LIMITER=limiter,
                        riemann_solver=solver, allow_first_order_faces=scheme in C8_SCHEMES)


def run(WL, WR, x0, box, times, n, scheme, case=None):
    """Edges, U0 and U at each of `times` (a list) on n cells of the box; MC, van Leer: FACES[(scheme, case, n)]
    first-order faces."""
    edges = np.linspace(box[0], box[1], n + 1)
    U0 = initial(WL, WR, x0, edges)
    return edges, U0, evolve(cfg_for(n, box[1] - box[0], scheme), U0, list(times),
                             faces=FACES.get((scheme, case, n), (0, 0)))[0]


def norms(num, ex, h, mask_u=None):
    """L1 and L2 (integrals over the box) of rho, u, p; u only where mask_u."""
    w = [np.ones_like(num[0], bool), np.ones_like(num[0], bool) if mask_u is None else mask_u, np.ones_like(num[0], bool)]
    l1 = [h * float(np.abs(a - b)[m].sum()) for a, b, m in zip(num, ex, w)]
    l2 = [float(np.sqrt(h * ((a - b) ** 2)[m].sum())) for a, b, m in zip(num, ex, w)]
    return l1, l2


def symptoms(case, num, edges, t):
    """{name: value} of the case's own known failure modes."""
    WL, WR, x0, _, _, _ = CASES[case]
    w = er.waves(WL, WR, G) if case != "rcvcr" else None
    xc = 0.5 * (edges[:-1] + edges[1:])
    h = edges[1] - edges[0]
    rho, u, p = num
    if case == "peak":                   # the narrow shell between contact and shock: its height
        return {"peak_error": abs(float(rho.max()) / w["rho_R"] - 1.0)}
    if case == "rcvcr":                  # the vacuum: how empty the centre gets
        return {"centre_rho": float(rho[np.abs(xc - x0) < 0.02].max())}
    if case == "stationary":             # the contact: sharp and in place
        lo, hi = w["rho_L"], w["rho_R"]
        win = (xc > 0.7) & (xc < x0 + 0.5 * (w["right"][0] * t))
        inner = win & (rho > lo + 0.1 * (hi - lo)) & (rho < hi - 0.1 * (hi - lo))
        j = np.nonzero(win & (rho > 0.5 * (lo + hi)))[0][0]          # first cell past the midpoint
        xm = xc[j - 1] + h * (0.5 * (lo + hi) - rho[j - 1]) / (rho[j] - rho[j - 1])
        return {"contact_width_cells": float(inner.sum()), "contact_drift_cells": abs(xm - (x0 + w["u"] * t)) / h}
    if case == "slow":                   # the long-wavelength oscillation behind the slow shock
        xs = x0 + w["right"][0] * t
        win = (xc > 0.05) & (xc < xs - 0.05)
        return {"post_shock_osc": float(np.abs(rho[win] / WL[0] - 1.0).max())}
    if case == "streams":                # wall heating at the collision point, without a wall
        e, e_ex = p / ((G - 1.0) * rho), w["p"] / ((G - 1.0) * w["rho_L"])
        win = np.abs(xc - x0) < 0.1
        return {"centre_rho_dip": 1.0 - float(rho[win].min()) / w["rho_L"],
                "centre_e_overshoot": float(e[win].max()) / e_ex - 1.0}
    raise KeyError(case)


def check_norms(tag, n, l1, l2, B):
    """L1 and L2 of rho, u, p at n against B."""
    for k, name in enumerate("rup"):
        check(f"{tag} N={n} L1 {name}", l1[k], B["L1"][n][k])
        check(f"{tag} N={n} L2 {name}", l2[k], B["L2"][n][k])


def check_orders(tag, ns, errs, B):
    """Fitted L1 and L2 orders of rho, u, p over ns against B; errs {n: (l1, l2)}."""
    for j, norm in enumerate(("L1", "L2")):
        for k, name in enumerate("rup"):
            order = -np.polyfit(np.log(ns), np.log([errs[n][j][k] for n in ns]), 1)[0]
            lo, hi = B[f"{norm}_order"][k]
            check(f"{tag} {norm} {name} order", order, lo, lower=True)
            check(f"{tag} {norm} {name} order", order, hi)


@pytest.mark.parametrize("scheme", list(SCHEMES))
@pytest.mark.parametrize("case", list(CASES))
def test_la14379_ladder(case, scheme):
    """L1 and L2 of rho, u, p at every N, their L1 and L2 orders, the exact budget while the waves are
    inside, positivity and the case's symptom."""
    WL, WR, x0, box, T, t_budget = CASES[case]
    B, errs, ns = BOUNDS[case][scheme], {}, _ns(case)
    tag = f"{case} {scheme}"
    for n in ns:
        times = sorted({t_budget, T})
        edges, U0, Us = run(WL, WR, x0, box, times, n, scheme, case)
        h = edges[1] - edges[0]
        check_tube_budget(f"{tag} N={n} t={t_budget:.4g}", U0, Us[times.index(t_budget)], WL, WR, t_budget, h, G)
        U = Us[-1]
        num, ex = _prim(U), exact(CASES[case], edges, T)
        l1, l2 = norms(num, ex, h, mask_u=ex[0] > 0)
        errs[n] = (l1, l2)
        check_norms(tag, n, l1, l2, B)
        assert num[0].min() > 0 and num[2].min() > 0, f"{tag} N={n}: min rho {num[0].min():.3e}, min p {num[2].min():.3e}"
        check(f"{tag} N={n} min rho", float(num[0].min()), 0.0, lower=True)
        check(f"{tag} N={n} min p", float(num[2].min()), 0.0, lower=True)
        for name, v in symptoms(case, num, edges, T).items():
            check(f"{tag} N={n} {name}", v, B[name][n])
    check_orders(tag, ns, errs, B)


MIRROR_N, MIRROR_TOL = 512, 1e-13


# MC used to break it (audit C8): its predicted face pressure went negative next to the forming shocks
# (p = -0.015 at N = 64, streams), HLLC took a one-sided flux; such faces now take the first-order state
# (luna.hydro.slopes.first_order_if_unphysical).
@pytest.mark.parametrize("scheme", list(SCHEMES))
@pytest.mark.parametrize("case", ["rcvcr", "streams"])
def test_mirror_symmetry(case, scheme):
    """The symmetric problems stay mirror symmetric to round-off: the collision needs no wall."""
    WL, WR, x0, box, T, _ = CASES[case]
    check_mirror(f"{case} {scheme} N={MIRROR_N}", run(WL, WR, x0, box, [T], MIRROR_N, scheme, case)[2][0], 0, MIRROR_TOL)


@pytest.mark.xfail(raises=RunAborted, strict=True,
                   reason="open bug: no vacuum state and no refusal for it; rho = p = 0 gives a 0/0 sound speed, "
                          "the first dt is NaN (-inf by integrators.py:27) and the run aborts at t = 0 (a crash)")
def test_vacuum_expansion_true_vacuum():
    """The report's vacuum expansion, left state rho = u = p = 0: LUNA neither solves nor refuses it, it crashes."""
    WL, WR, x0, box, T = VACUUM
    n = NS_VACUUM[0]
    edges = np.linspace(box[0], box[1], n + 1)
    from tests._helpers import _make_step_fns
    dt0 = float(_make_step_fns(cfg_for(n, box[1] - box[0], "minmod"))[0](initial(WL, WR, x0, edges)))
    assert dt0 == -np.inf, dt0                     # the failure is the NaN dt, not something else
    try:
        run(WL, WR, x0, box, [T], n, "minmod")
    except RunAborted as e:
        assert str(e).startswith("stopped at t=0.000000e+00 ") and "dt ok=False" in str(e), str(e)   # never advanced
        raise


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_vacuum_expansion_near_vacuum(scheme):
    """Near-vacuum stand-in (rho_L = p_L = 1e-6): L1 and L2 against its exact solution and their orders,
    the budget and positivity."""
    _, WR, x0, box, T = VACUUM
    WL = (NEAR_VACUUM * WR[0], 0.0, NEAR_VACUUM * WR[2])
    B, errs, tag = VACUUM_BOUNDS[scheme], {}, f"vacuum {scheme}"
    t_budget = 0.5 * T              # the fan head (a_R t) is 0.11 cm from x = 1 at t_f: the budget at t_f / 2
    for n in NS_VACUUM:
        edges, U0, (Ub, U) = run(WL, WR, x0, box, [t_budget, T], n, scheme, "vacuum")
        h = edges[1] - edges[0]
        check_tube_budget(f"{tag} N={n} t={t_budget:.4g}", U0, Ub, WL, WR, t_budget, h, G)
        num, ex = _prim(U), exact((WL, WR, x0), edges, T)
        l1, l2 = norms(num, ex, h, mask_u=ex[0] > 1e-3 * WR[0])
        errs[n] = (l1, l2)
        check_norms(tag, n, l1, l2, B)
        assert num[0].min() > 0 and num[2].min() > 0, f"{tag} N={n}: min rho {num[0].min():.3e}, min p {num[2].min():.3e}"
    check_orders(tag, NS_VACUUM, errs, B)


SLOW_LONG = dict(box=(0.0, 4.5), n=1152, t=30.0)      # 256 cells per cm, the shock at x = 3.79 at t_f


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_slow_shock_to_report_time(scheme):
    """The slow shock to the report's t_f = 30 on [0, 4.5]: L1 rho and the post-shock oscillation."""
    WL, WR, x0, _, _, _ = CASES["slow"]
    B, tag = SLOW_LONG_BOUNDS[scheme], f"slow t=30 {scheme}"
    edges, U0, (U,) = run(WL, WR, x0, SLOW_LONG["box"], [SLOW_LONG["t"]], SLOW_LONG["n"], scheme, "slow-long")
    num, ex = _prim(U), exact(CASES["slow"], edges, SLOW_LONG["t"])
    l1 = norms(num, ex, edges[1] - edges[0])[0]
    for k, name in enumerate("rup"):
        check(f"{tag} L1 {name}", l1[k], B["L1"][k])
    xc = 0.5 * (edges[:-1] + edges[1:])
    xs = x0 + er.waves(WL, WR, G)["right"][0] * SLOW_LONG["t"]
    win = (xc > 0.05) & (xc < xs - 0.05)
    check(f"{tag} post_shock_osc", float(np.abs(num[0][win] / WL[0] - 1.0).max()), B["osc"])


def test_bound_catches_a_short_run():
    """Teeth: Peak stopped at 0.9 t_f misses its density L1 bound at 800."""
    WL, WR, x0, box, T, _ = CASES["peak"]
    edges, _, (U,) = run(WL, WR, x0, box, [0.9 * T], 800, "minmod")
    l1 = norms(_prim(U), exact(CASES["peak"], edges, T), edges[1] - edges[0])[0][0]
    bound = BOUNDS["peak"]["minmod"]["L1"][800][0]
    print(f"MEASURED peak minmod N=800 t=0.9 t_f L1 rho = {l1:.6e} (> {bound:.6e})")
    assert l1 > bound, f"L1 rho {l1:.4e} of a short run passes the bound"
