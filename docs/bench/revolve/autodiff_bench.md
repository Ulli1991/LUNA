# Autodiff benchmark

2026-09-27 01:04:42 - merged

jax 0.10.1, unrecorded (rows measured before the stack was logged)

`launch`: async is the default; serial is HIP_LAUNCH_BLOCKING=1 or AMD_SERIALIZE_KERNEL=3, which the radiation gradient needs on ROCm and which costs 3x on the multigrid.

| case | nx | steps | mesh | checkpoints | launch | forward s | grad s | ratio | peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| gravity | 128 | 120 | 1,1,1 | revolve 16 | async | 0.4004 | 2.2329 | 5.58 | 5.312 |
| gravity | 128 | 120 | 1,1,1 | revolve 64 | async | 0.3994 | 2.0859 | 5.22 | 12.062 |
| gravity | 128 | 250 | 1,1,1 | revolve 16 | async | 0.8160 | 4.9919 | 6.12 | 5.312 |
| gravity | 128 | 250 | 1,1,1 | revolve 64 | async | 0.8118 | 4.5325 | 5.58 | 12.062 |
| gravity | 128 | 500 | 1,1,1 | revolve 16 | async | 1.6104 | 10.5498 | 6.55 | 5.312 |
| gravity | 128 | 500 | 1,1,1 | revolve 64 | async | 1.6146 | 9.2168 | 5.71 | 12.062 |
| gravity | 128 | 1000 | 1,1,1 | revolve 16 | async | 3.2121 | 21.5325 | 6.70 | 5.312 |
| gravity | 128 | 1000 | 1,1,1 | revolve 64 | async | 3.1941 | 18.5539 | 5.81 | 12.062 |
| gravity | 128 | 2000 | 1,1,1 | revolve 16 | async | 6.4553 | 47.1249 | 7.30 | 5.312 |
| gravity | 128 | 2000 | 1,1,1 | revolve 64 | async | 6.4457 | 37.3531 | 5.80 | 12.062 |
| gravity | 128 | 120 | 1,1,1 | every step | async | **fail** (rc -11) | | | |
| gravity | 128 | 8 | 1,1,1 | every step | async | 0.0470 | 0.1664 | 3.54 | 4.046 |
| gravity | 128 | 16 | 1,1,1 | every step | async | 0.0711 | 0.2913 | 4.10 | 5.171 |
| gravity | 128 | 32 | 1,1,1 | every step | async | **fail** (rc -11) | | | |
| gravity | 128 | 64 | 1,1,1 | every step | async | **fail** (rc -11) | | | |
| gravity | 256 | 120 | 1,1,1 | revolve 16 | async | 2.4358 | 14.6413 | 6.01 | 43.261 |
| gravity | 256 | 120 | 1,1,1 | revolve 32 | async | 2.4351 | 14.3169 | 5.88 | 61.886 |
| gravity | 256 | 250 | 1,1,1 | revolve 16 | async | 5.0179 | 32.7549 | 6.53 | 43.261 |
| gravity | 256 | 250 | 1,1,1 | revolve 32 | async | 5.0219 | 30.5709 | 6.09 | 61.886 |
| gravity | 256 | 500 | 1,1,1 | revolve 16 | async | 10.0520 | 68.9257 | 6.86 | 43.261 |
| gravity | 256 | 500 | 1,1,1 | revolve 32 | async | 9.9819 | 61.4494 | 6.16 | 61.886 |
| gravity | 256 | 1000 | 1,1,1 | revolve 16 | async | 19.9600 | 141.7043 | 7.10 | 43.261 |
| gravity | 256 | 1000 | 1,1,1 | revolve 32 | async | 19.9846 | 132.4154 | 6.63 | 61.886 |
| gravity | 256 | 16 | 1,1,1 | every step | async | **fail** (rc -11) | | | |
| gravity | 256 | 32 | 1,1,1 | every step | async | **fail** (rc -11) | | | |
| gravity | 256 | 4 | 1,1,1 | every step | async | 0.1278 | 0.5146 | 4.03 | 28.636 |
| gravity | 256 | 8 | 1,1,1 | every step | async | 0.2065 | 0.9323 | 4.51 | 33.250 |
| gravity | 256 | 1000 | 2,1,1 | revolve 32 | async | 12.2810 | 80.7028 | 6.57 | 31.229 |
| hydro | 256 | 1000 | 2,1,1 | revolve 32 | async | 6.5495 | 44.1772 | 6.75 | 18.056 |
| hydro | 128 | 120 | 1,1,1 | revolve 8 | async | 0.1983 | 1.4000 | 7.06 | 2.521 |
| hydro | 128 | 120 | 1,1,1 | revolve 16 | async | 0.1987 | 1.2623 | 6.35 | 3.146 |
| hydro | 128 | 120 | 1,1,1 | revolve 64 | async | 0.1985 | 1.1875 | 5.98 | 6.981 |
| hydro | 128 | 250 | 1,1,1 | revolve 8 | async | 0.4118 | 3.1508 | 7.65 | 2.521 |
| hydro | 128 | 250 | 1,1,1 | revolve 16 | async | 0.4140 | 2.8214 | 6.81 | 3.146 |
| hydro | 128 | 250 | 1,1,1 | revolve 64 | async | 0.4127 | 2.5855 | 6.26 | 6.981 |
| hydro | 128 | 500 | 1,1,1 | revolve 8 | async | 0.8282 | 6.6842 | 8.07 | 2.521 |
| hydro | 128 | 500 | 1,1,1 | revolve 16 | async | 0.8277 | 5.9189 | 7.15 | 3.146 |
| hydro | 128 | 500 | 1,1,1 | revolve 64 | async | 0.8265 | 5.2707 | 6.38 | 6.981 |
| hydro | 128 | 1000 | 1,1,1 | revolve 8 | async | 1.6517 | 14.5598 | 8.82 | 2.521 |
| hydro | 128 | 1000 | 1,1,1 | revolve 16 | async | 1.6563 | 12.1881 | 7.36 | 3.146 |
| hydro | 128 | 1000 | 1,1,1 | revolve 64 | async | 1.6519 | 10.6583 | 6.45 | 6.981 |
| hydro | 128 | 2000 | 1,1,1 | revolve 8 | async | 3.3122 | 31.5037 | 9.51 | 2.521 |
| hydro | 128 | 2000 | 1,1,1 | revolve 16 | async | 3.3088 | 26.2405 | 7.93 | 3.146 |
| hydro | 128 | 2000 | 1,1,1 | revolve 64 | async | 3.3113 | 21.4483 | 6.48 | 6.981 |
| hydro | 128 | 120 | 1,1,1 | every step | async | 0.1969 | 1.1003 | 5.59 | 11.356 |
| hydro | 128 | 250 | 1,1,1 | every step | async | **fail** (rc -11) | | | |
| hydro | 128 | 256 | 1,1,1 | every step | async | **fail** (rc -11) | | | |
| hydro | 128 | 300 | 1,1,1 | every step | async | **fail** (rc -11) | | | |
| hydro | 256 | 120 | 1,1,1 | revolve 16 | async | 1.6246 | 9.6647 | 5.95 | 24.228 |
| hydro | 256 | 120 | 1,1,1 | revolve 32 | async | 1.6291 | 9.4478 | 5.80 | 36.068 |
| hydro | 256 | 250 | 1,1,1 | revolve 16 | async | 3.3826 | 21.6969 | 6.41 | 24.228 |
| hydro | 256 | 250 | 1,1,1 | revolve 32 | async | 3.3814 | 20.1532 | 5.96 | 36.068 |
| hydro | 256 | 500 | 1,1,1 | revolve 16 | async | 6.7675 | 45.6940 | 6.75 | 24.228 |
| hydro | 256 | 500 | 1,1,1 | revolve 32 | async | 6.7684 | 40.7506 | 6.02 | 36.068 |
| hydro | 256 | 1000 | 1,1,1 | revolve 16 | async | 13.5255 | 94.0244 | 6.95 | 24.228 |
| hydro | 256 | 1000 | 1,1,1 | revolve 32 | async | 13.5416 | 87.8663 | 6.49 | 36.068 |
| hydro | 256 | 2000 | 1,1,1 | revolve 16 | async | 26.9781 | 203.5289 | 7.54 | 24.228 |
| hydro | 256 | 2000 | 1,1,1 | revolve 32 | async | 27.0513 | 183.8239 | 6.80 | 36.068 |
| hydro | 256 | 32 | 1,1,1 | every step | async | 0.4332 | 2.2715 | 5.24 | 36.123 |
| hydro | 256 | 120 | 1,1,1 | every step | async | **fail** (rc 1) | | | |
| gravity | 256 | 2000 | 2,1,1 | revolve 32 | async | 24.3860 | 167.2242 | 6.86 | 31.229 |
| hydro | 256 | 2000 | 2,1,1 | revolve 32 | async | 13.0462 | 92.1852 | 7.07 | 18.056 |
