"""The plots after a run (cfg.PLOT_AFTER_SIM) find the snapshots the run
wrote under snapdir_NNN/, and a run that leaves nothing to plot says so
instead of reporting success. The 1D helper once looked for snap*.hdf5 in
output_dir itself, printed "No HDF5 snapshots found!" and then
"Plots generated successfully"."""
import os

import pytest

pytestmark = pytest.mark.fast


def _run(tmp_path, capsys, name, **kw):
    """main.run on a tiny problem; returns (output_dir, stdout)."""
    from dataclasses import replace
    from luna.setup.jax_init import initialize_jax_distributed
    from luna.template_config import Config
    out = str(tmp_path / name)
    cfg = replace(Config(), t_end=0.02, time_bet_snaps=0.01, max_steps=6,
                  PLOT_AFTER_SIM=True, output_dir=out, enable_x64=True, **kw)
    initialize_jax_distributed(enable_x64=cfg.enable_x64)
    from luna import main
    capsys.readouterr()
    main.run(cfg)
    return out, capsys.readouterr().out


def _snaps_and_pngs(out):
    snaps = sorted(d for d in os.listdir(out) if d.startswith("snapdir_"))
    png = os.path.join(out, "png")
    pngs = sorted(os.listdir(png)) if os.path.isdir(png) else []
    return snaps, pngs


@pytest.mark.parametrize("kw", [
    dict(test_problem="Sod-Shock", SPATIAL_DIMENSION=1, Nx=32),
    dict(test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2, BOUNDARY=0, Nx=16, Ny=16),
], ids=["1d", "2d"])
def test_a_run_plots_every_snapshot_it_wrote(tmp_path, capsys, kw):
    out, log = _run(tmp_path, capsys, "plots", **kw)
    snaps, pngs = _snaps_and_pngs(out)
    assert snaps, "the run wrote no snapdir_*: nothing to test"
    assert len(pngs) == len(snaps), (snaps, pngs)
    assert f"✓ {len(snaps)} plot(s) written" in log
    assert "No snapshots" not in log and "No HDF5" not in log


def test_nothing_to_plot_is_reported_as_failure(tmp_path, capsys):
    out, log = _run(tmp_path, capsys, "noplots", test_problem="Sod-Shock",
                    SPATIAL_DIMENSION=1, Nx=32, write_snapshots=False)
    snaps, pngs = _snaps_and_pngs(out)
    assert not snaps and not pngs
    assert "✗ No plots generated" in log
    assert "✓" not in log.split("GENERATING PLOTS", 1)[1]


def test_the_helpers_return_how_many_they_plotted(tmp_path):
    from luna.utils.plot_1d_fluid_quants import plot_1d_fluid_quantities
    from luna.utils.plot_density2d import plot_density2d
    assert plot_1d_fluid_quantities(str(tmp_path), "snap") == 0
    assert plot_density2d(str(tmp_path), "snap") == 0
