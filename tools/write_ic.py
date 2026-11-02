"""Write a run's initial conditions to disk once, for cfg.ic_dir.

    python tools/write_ic.py [config.py] out=DIR [field=value ...]

The same config form as main.py. The run then takes ic_dir=DIR and each
rank reads only its own cells. Host-side numpy only: no GPU, no JAX.
"""
import sys
import time
from dataclasses import replace

from luna.ic_generation.ic import resolve
from luna.ic_generation.ic_file import write_ic
from luna.setup.cli import configure


def main(argv):
    out = [a for a in argv if a.startswith("out=")]
    if len(out) != 1:
        sys.exit("usage: write_ic.py [config.py] out=DIR [field=value ...]")
    cfg = resolve(replace(configure([a for a in argv if not a.startswith("out=")]), ic_dir=""))
    t0 = time.time()
    write_ic(cfg, out[0][4:])
    print(f"wrote {cfg.test_problem} {cfg.Nx}x{cfg.Ny}x{cfg.Nz} to {out[0][4:]} "
          f"in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
