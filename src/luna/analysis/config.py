"""In-situ analysis settings (no JAX import: template_config reads this)."""
from dataclasses import dataclass
from typing import Optional

FRAME_FIELDS = ("column_density", "slice_s", "slice_log10_vorticity")


@dataclass
class InsituConfig:
    """Cadences in code time units (0: off); outputs under output_dir/subdir."""

    stats_dt: float = 0.0           # PDFs, moments, global budgets
    spectra_dt: float = 0.0         # shell spectra (pencil FFT)
    sf_dt: float = 0.0              # velocity structure functions
    snip_dt: float = 0.0            # snipshots: reduced-sampling rho, rho v
    frame_dt: float = 0.0           # movie frames
    pdf_bins: int = 128
    pdf_ranges: Optional[dict] = None
    spectra_single: bool = True     # FFTs in single precision: half the memory of the transforms
    sf_orders: int = 8
    snip_factor: int = 8            # cells per snipshot cell along each axis
    snip_mode: str = "mean"         # "mean": block average (conserves mass, momentum); "stride": point samples
    frame_fields: tuple = FRAME_FIELDS
    frame_axis: int = 2             # line of sight / slice normal
    frame_ranges: Optional[dict] = None
    frame_png_max: int = 2048       # quick-look PNG width (0: none)
    subdir: str = "insitu"
