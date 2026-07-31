"""
Arepo-style unit system with 3 base units.

Following the convention of Arepo/GADGET: the user specifies three base units
(UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s) and all other
units and physical constants are derived from these.

When all three base units are zero (default), the code runs in dimensionless mode.
"""

from dataclasses import dataclass, field
import math


# ============================================================================
# CGS Physical Constants
# ============================================================================
GRAVITY_CGS = 6.67430e-8          # cm^3 g^-1 s^-2
BOLTZMANN_CGS = 1.380649e-16      # erg K^-1 (CODATA exact)
PROTON_MASS_CGS = 1.6726219e-24   # g (CODATA)
# kB/mp in (km/s)^2 per K — the T <-> specific-energy factor used by
# the cosmo floors/ceilings everywhere.
KB_OVER_MP_KMS2_PER_K = BOLTZMANN_CGS / PROTON_MASS_CGS / 1e10
ELECTRON_MASS_CGS = 9.10938e-28   # g
SPEED_OF_LIGHT_CGS = 2.99792e10   # cm s^-1
THOMPSON_CGS = 6.65246e-25        # cm^2
STEFAN_BOLTZMANN_CGS = 5.67037e-5 # erg cm^-2 s^-1 K^-4
RADIATION_CONSTANT_CGS = 7.56577e-15  # erg cm^-3 K^-4

# Astronomical constants in CGS
SOLAR_MASS_CGS = 1.98892e33       # g
SOLAR_LUM_CGS = 3.846e33          # erg s^-1
PARSEC_CGS = 3.08568e18           # cm
KILOPARSEC_CGS = 3.08568e21       # cm
MEGAPARSEC_CGS = 3.08568e24       # cm
ASTRONOMICAL_UNIT_CGS = 1.49598e13  # cm
SEC_PER_YEAR = 3.15576e7          # s
SEC_PER_MEGAYEAR = 3.15576e13     # s
HYDROGEN_MASSFRAC = 0.76          # primordial hydrogen mass fraction
GAMMA_DEFAULT = 5.0 / 3.0


@dataclass
class UnitSystem:
    """
    Arepo-style unit system defined by 3 base units.

    Parameters
    ----------
    UnitLength_in_cm : float
        Code length unit in centimeters (e.g., 3.08568e21 for 1 kpc)
    UnitMass_in_g : float
        Code mass unit in grams (e.g., 1.98892e43 for 10^10 solar masses)
    UnitVelocity_in_cm_per_s : float
        Code velocity unit in cm/s (e.g., 1e5 for 1 km/s)

    All derived units and constants in code units are computed automatically.
    """
    UnitLength_in_cm: float = 0.0
    UnitMass_in_g: float = 0.0
    UnitVelocity_in_cm_per_s: float = 0.0

    # Derived units (computed in __post_init__)
    UnitTime_in_s: float = field(init=False, default=0.0)
    UnitDensity_in_cgs: float = field(init=False, default=0.0)
    UnitPressure_in_cgs: float = field(init=False, default=0.0)
    UnitEnergy_in_cgs: float = field(init=False, default=0.0)
    UnitCoolingRate_in_cgs: float = field(init=False, default=0.0)

    # Physical constants in code units
    GravityConstant: float = field(init=False, default=0.0)
    BoltzmannConstant: float = field(init=False, default=0.0)
    ProtonMass: float = field(init=False, default=0.0)
    SpeedOfLight: float = field(init=False, default=0.0)

    @property
    def is_enabled(self) -> bool:
        """True if physical units are configured (all three base units nonzero)."""
        return (self.UnitLength_in_cm != 0.0 and
                self.UnitMass_in_g != 0.0 and
                self.UnitVelocity_in_cm_per_s != 0.0)

    def __post_init__(self):
        if not self.is_enabled:
            return

        L = self.UnitLength_in_cm
        M = self.UnitMass_in_g
        V = self.UnitVelocity_in_cm_per_s

        # Derived units
        self.UnitTime_in_s = L / V
        self.UnitDensity_in_cgs = M / (L * L * L)
        self.UnitPressure_in_cgs = M / (L * L * L) * V * V
        self.UnitEnergy_in_cgs = M * V * V
        self.UnitCoolingRate_in_cgs = self.UnitEnergy_in_cgs / self.UnitTime_in_s

        # Physical constants in code units
        self.GravityConstant = (GRAVITY_CGS / (L**3) * M * (self.UnitTime_in_s**2))
        self.BoltzmannConstant = (BOLTZMANN_CGS / self.UnitEnergy_in_cgs)
        self.ProtonMass = PROTON_MASS_CGS / M
        self.SpeedOfLight = SPEED_OF_LIGHT_CGS / V

    # ========================================================================
    # Conversion helpers: CGS → code units
    # ========================================================================
    def to_code_length(self, length_cm):
        """Convert length from cm to code units."""
        return length_cm / self.UnitLength_in_cm

    def to_code_mass(self, mass_g):
        """Convert mass from g to code units."""
        return mass_g / self.UnitMass_in_g

    def to_code_velocity(self, vel_cm_s):
        """Convert velocity from cm/s to code units."""
        return vel_cm_s / self.UnitVelocity_in_cm_per_s

    def to_code_time(self, time_s):
        """Convert time from seconds to code units."""
        return time_s / self.UnitTime_in_s

    def to_code_density(self, rho_cgs):
        """Convert density from g/cm^3 to code units."""
        return rho_cgs / self.UnitDensity_in_cgs

    def to_code_pressure(self, p_cgs):
        """Convert pressure from dyne/cm^2 to code units."""
        return p_cgs / self.UnitPressure_in_cgs

    def to_code_energy(self, energy_erg):
        """Convert energy from erg to code units."""
        return energy_erg / self.UnitEnergy_in_cgs

    def to_code_number_density(self, n_cm3, mean_molecular_weight=1.0):
        """
        Convert number density (particles/cm^3) to code mass density.

        Parameters
        ----------
        n_cm3 : float
            Number density in particles per cm^3
        mean_molecular_weight : float
            Mean molecular weight in units of proton mass
        """
        rho_cgs = n_cm3 * mean_molecular_weight * PROTON_MASS_CGS
        return self.to_code_density(rho_cgs)

    # ========================================================================
    # Conversion helpers: code units → CGS
    # ========================================================================
    def to_cgs_length(self, length_code):
        """Convert length from code units to cm."""
        return length_code * self.UnitLength_in_cm

    def to_cgs_mass(self, mass_code):
        """Convert mass from code units to g."""
        return mass_code * self.UnitMass_in_g

    def to_cgs_velocity(self, vel_code):
        """Convert velocity from code units to cm/s."""
        return vel_code * self.UnitVelocity_in_cm_per_s

    def to_cgs_time(self, time_code):
        """Convert time from code units to seconds."""
        return time_code * self.UnitTime_in_s

    def to_cgs_density(self, rho_code):
        """Convert density from code units to g/cm^3."""
        return rho_code * self.UnitDensity_in_cgs

    def to_cgs_pressure(self, p_code):
        """Convert pressure from code units to dyne/cm^2."""
        return p_code * self.UnitPressure_in_cgs

    def to_cgs_energy(self, energy_code):
        """Convert energy from code units to erg."""
        return energy_code * self.UnitEnergy_in_cgs

    # ========================================================================
    # Convenient astronomical conversions
    # ========================================================================
    def to_code_kpc(self, kpc):
        """Convert kiloparsecs to code length."""
        return self.to_code_length(kpc * KILOPARSEC_CGS)

    def to_code_Myr(self, myr):
        """Convert megayears to code time."""
        return self.to_code_time(myr * SEC_PER_MEGAYEAR)

    def to_code_solar_mass(self, msun):
        """Convert solar masses to code mass."""
        return self.to_code_mass(msun * SOLAR_MASS_CGS)

    def to_code_km_per_s(self, km_s):
        """Convert km/s to code velocity."""
        return self.to_code_velocity(km_s * 1e5)

    def summary(self):
        """Print a summary of the unit system."""
        if not self.is_enabled:
            print("Unit system: DIMENSIONLESS (no physical units configured)")
            return

        print("=" * 60)
        print("AREPO-STYLE UNIT SYSTEM")
        print("=" * 60)
        print(f"Base units:")
        print(f"  UnitLength   = {self.UnitLength_in_cm:.6e} cm")
        print(f"               = {self.UnitLength_in_cm / KILOPARSEC_CGS:.6f} kpc")
        print(f"  UnitMass     = {self.UnitMass_in_g:.6e} g")
        print(f"               = {self.UnitMass_in_g / SOLAR_MASS_CGS:.6e} Msun")
        print(f"  UnitVelocity = {self.UnitVelocity_in_cm_per_s:.6e} cm/s")
        print(f"               = {self.UnitVelocity_in_cm_per_s / 1e5:.6f} km/s")
        print()
        print(f"Derived units:")
        print(f"  UnitTime     = {self.UnitTime_in_s:.6e} s")
        print(f"               = {self.UnitTime_in_s / SEC_PER_MEGAYEAR:.6f} Myr")
        print(f"  UnitDensity  = {self.UnitDensity_in_cgs:.6e} g/cm^3")
        print(f"  UnitPressure = {self.UnitPressure_in_cgs:.6e} dyne/cm^2")
        print(f"  UnitEnergy   = {self.UnitEnergy_in_cgs:.6e} erg")
        print()
        print(f"Constants in code units:")
        print(f"  G            = {self.GravityConstant:.6e}")
        print(f"  k_B          = {self.BoltzmannConstant:.6e}")
        print(f"  m_p          = {self.ProtonMass:.6e}")
        print(f"  c            = {self.SpeedOfLight:.6e}")
        print("=" * 60)
