# CPU test image for machines without a jax 0.10 wheel (e.g. Intel Macs).
# Dependencies only; the repo is mounted at run time, so the image never goes stale.
# From the LUNA clone, with luna-reference-solutions cloned next to it:
#   docker build --network host -t luna-test tools/docker
#   docker run --rm -v "$PWD":/luna -v "$PWD/../luna-reference-solutions":/luna-reference-solutions \
#       luna-test python tools/run_tests.py
#   docker run --rm -v "$PWD":/luna luna-test python main.py src/luna/template_config.py Nx=64 \
#       PLOT_AFTER_SIM=False output_dir=snapshots/sod64
FROM python:3.12-slim

# git: tests/test_run_scripts.py asks `git ls-files` which scripts are tracked.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* && git config --system --add safe.directory /luna

RUN pip install --no-cache-dir --timeout 30 --retries 10 "jax==0.10.2" "jaxlib==0.10.2" "h5py>=3.0" "numpy>=1.23" \
    "pytest>=7.0" "matplotlib>=3.5" "ruff>=0.16"

# /luna/src for luna and amuse_luna (nothing is pip-installed); the reference solutions
# are found as the sibling /luna-reference-solutions of /luna.
ENV PYTHONPATH=/luna/src \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /luna
