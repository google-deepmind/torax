# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""String keys and dimension names for the TORAX xarray output DataTree.

Each key is an ``OutputKey`` instance — a ``str`` subclass that carries a
``units`` attribute from the ``Units`` enum and a ``grid_type`` attribute
from the ``GridType`` enum. This means every constant (e.g. ``T_E``) can be
used wherever a plain string is expected (dict keys, xarray names, ``==``
comparisons) while also carrying its physical unit and grid metadata.

Physical dimensionless quantities (e.g. safety factor *q*) use
``Units.DIMENSIONLESS``. Non-physical keys (e.g. solver status, dataset
names) use ``Units.NOT_APPLICABLE`` (the empty string ``""``).

For source profiles whose names are generated at runtime (e.g.
``p_ecrh_e``), ``get_units`` falls back to prefix-based matching so that
dynamically-constructed keys still receive the correct unit string.
"""

# pylint: disable=invalid-name

import enum
import sys
from typing import Final


class Units(enum.StrEnum):
  """Physical unit strings for TORAX output variables.

  Using a ``StrEnum`` ensures each member *is* a string, so it can be used
  directly as an xarray attribute value or in comparisons.

  ``NOT_APPLICABLE`` (empty string) is for non-physical quantities (solver
  status, iteration counts, dataset group names).
  ``DIMENSIONLESS`` is for physical quantities that are genuinely
  dimensionless (safety factor, Greenwald fraction, beta, etc.).
  """

  NOT_APPLICABLE = ""
  DIMENSIONLESS = "dimensionless"
  # SI base and derived.
  METER = "m"
  SECOND = "s"
  AMPERE = "A"
  VOLT = "V"
  WATT = "W"
  JOULE = "J"
  PASCAL = "Pa"
  TESLA = "T"
  WEBER = "Wb"
  # Plasma-conventional.
  KEV = "keV"
  EV = "eV"
  AMU = "amu"
  # Composite / derived.
  INVERSE_METER = "m^-1"
  SQUARE_METER = "m^2"
  CUBIC_METER = "m^3"
  QUARTIC_METER = "m^4"
  INVERSE_SQUARE_METER = "m^-2"
  INVERSE_CUBIC_METER = "m^-3"
  METER_PER_SECOND = "m/s"
  SQUARE_METER_PER_SECOND = "m^2/s"
  RAD_PER_SECOND = "rad/s"
  AMPERE_PER_SQUARE_METER = "A/m^2"
  VOLT_PER_METER = "V/m"
  SIEMENS_PER_METER = "S/m"
  WATT_PER_SQUARE_METER = "W/m^2"
  WATT_PER_CUBIC_METER = "W/m^3"
  MW_PER_CUBIC_METER = "MW/m^3"
  PASCAL_PER_WEBER = "Pa/Wb"
  WEBER_PER_SECOND = "Wb/s"
  TESLA_METER = "T m"
  INVERSE_SQUARE_TESLA = "T^-2"
  SQUARE_TESLA = "T^2"
  INVERSE_SECOND = "s^-1"
  INVERSE_CUBIC_METER_PER_SECOND = "m^-3 s^-1"


class GridType(enum.StrEnum):
  """Spatial and temporal grid locations for TORAX output variables."""

  NOT_APPLICABLE = ""
  SCALAR = "scalar"  # 1D time-series: (time,)
  FACE = "face"  # 2D profile: (time, rho_face_norm)
  CELL = "cell"  # 2D profile: (time, rho_cell_norm)
  CELL_PLUS_BOUNDARIES = "cell_plus_boundaries"  # 2D profile: (time, rho_norm)


class OutputKey(str):
  """A string key that carries unit and grid type metadata.

  ``OutputKey`` inherits from ``str`` so it can be used anywhere a regular
  string is expected.  The ``units`` attribute stores the physical unit as a
  ``Units`` enum member and is required.  The ``grid_type`` attribute stores the
  spatial/temporal grid as a ``GridType`` enum member and is required.

  Examples::

      >>> T_E = OutputKey("T_e", units=Units.KEV,
      grid_type=GridType.CELL_PLUS_BOUNDARIES)
      >>> T_E == "T_e"
      True
      >>> T_E.units
      <Units.KEV: 'keV'>
      >>> T_E.grid_type
      <GridType.CELL_PLUS_BOUNDARIES: 'cell_plus_boundaries'>
  """

  units: Units
  grid_type: GridType

  def __new__(
      cls,
      value: str,
      *,
      units: Units,
      grid_type: GridType,
  ) -> "OutputKey":
    obj = str.__new__(cls, value)
    obj.units = units
    obj.grid_type = grid_type
    return obj

  def __init__(
      self,
      value: str,
      *,
      units: Units,
      grid_type: GridType,
  ):
    # Required for pytype to recognize keyword arguments on str subclass.
    del value, units, grid_type
    super().__init__()

  def __getnewargs_ex__(self):
    # Required so that deepcopy/pickle can reconstruct the keyword-only args.
    return ((str(self),), {"units": self.units, "grid_type": self.grid_type})


# ---------------------------------------------------------------------------
# Dataset names (no physical units, no grid).
# ---------------------------------------------------------------------------
PROFILES: Final[OutputKey] = OutputKey(
    "profiles", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)
SCALARS: Final[OutputKey] = OutputKey(
    "scalars", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)
NUMERICS: Final[OutputKey] = OutputKey(
    "numerics", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)
EDGE: Final[OutputKey] = OutputKey(
    "edge", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)

# ---------------------------------------------------------------------------
# Core profiles.
# ---------------------------------------------------------------------------
T_E: Final[OutputKey] = OutputKey(
    "T_e", units=Units.KEV, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
T_I: Final[OutputKey] = OutputKey(
    "T_i", units=Units.KEV, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
PSI: Final[OutputKey] = OutputKey(
    "psi", units=Units.WEBER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
V_LOOP: Final[OutputKey] = OutputKey(
    "v_loop", units=Units.VOLT, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
N_E: Final[OutputKey] = OutputKey(
    "n_e",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
N_I: Final[OutputKey] = OutputKey(
    "n_i",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
Q: Final[OutputKey] = OutputKey(
    "q", units=Units.DIMENSIONLESS, grid_type=GridType.FACE
)
MAGNETIC_SHEAR: Final[OutputKey] = OutputKey(
    "magnetic_shear", units=Units.DIMENSIONLESS, grid_type=GridType.FACE
)
N_IMPURITY: Final[OutputKey] = OutputKey(
    "n_impurity",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
Z_IMPURITY: Final[OutputKey] = OutputKey(
    "Z_impurity",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
Z_EFF: Final[OutputKey] = OutputKey(
    "Z_eff",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
SIGMA_PARALLEL: Final[OutputKey] = OutputKey(
    "sigma_parallel",
    units=Units.SIEMENS_PER_METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
V_LOOP_LCFS: Final[OutputKey] = OutputKey(
    "v_loop_lcfs", units=Units.VOLT, grid_type=GridType.SCALAR
)
IP_PROFILE: Final[OutputKey] = OutputKey(
    "Ip_profile", units=Units.AMPERE, grid_type=GridType.FACE
)
IP: Final[OutputKey] = OutputKey(
    "Ip", units=Units.AMPERE, grid_type=GridType.SCALAR
)
TOROIDAL_ANGULAR_VELOCITY: Final[OutputKey] = OutputKey(
    "toroidal_angular_velocity",
    units=Units.RAD_PER_SECOND,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
A_I: Final[OutputKey] = OutputKey(
    "A_i", units=Units.AMU, grid_type=GridType.SCALAR
)
A_IMPURITY: Final[OutputKey] = OutputKey(
    "A_impurity", units=Units.AMU, grid_type=GridType.SCALAR
)
Z_I: Final[OutputKey] = OutputKey(
    "Z_i", units=Units.DIMENSIONLESS, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
Z_IMPURITY_SPECIES: Final[OutputKey] = OutputKey(
    "Z_impurity_species",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.CELL,
)
N_IMPURITY_SPECIES: Final[OutputKey] = OutputKey(
    "n_impurity_species",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.CELL,
)
MAIN_ION_FRACTIONS: Final[OutputKey] = OutputKey(
    "main_ion_fractions",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.SCALAR,
)
PRESSURE_THERMAL_E: Final[OutputKey] = OutputKey(
    "pressure_thermal_e",
    units=Units.PASCAL,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
PRESSURE_THERMAL_I: Final[OutputKey] = OutputKey(
    "pressure_thermal_i",
    units=Units.PASCAL,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
PRESSURE_THERMAL_TOTAL: Final[OutputKey] = OutputKey(
    "pressure_thermal_total",
    units=Units.PASCAL,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
PRESSURE_FAST_I: Final[OutputKey] = OutputKey(
    "pressure_fast_i",
    units=Units.PASCAL,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
PRESSURE_TOTAL_I: Final[OutputKey] = OutputKey(
    "pressure_total_i",
    units=Units.PASCAL,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
PRESSURE_TOTAL: Final[OutputKey] = OutputKey(
    "pressure_total",
    units=Units.PASCAL,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
EI_EXCHANGE: Final[OutputKey] = OutputKey(
    "ei_exchange",
    units=Units.MW_PER_CUBIC_METER,
    grid_type=GridType.CELL,
)
PSI_FROM_IP: Final[OutputKey] = OutputKey(
    "psi_from_Ip", units=Units.WEBER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)

# ---------------------------------------------------------------------------
# Calculated or derived current densities (excluding sources).
# ---------------------------------------------------------------------------
J_PARALLEL_TOTAL: Final[OutputKey] = OutputKey(
    "j_parallel_total",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_PARALLEL_OHMIC: Final[OutputKey] = OutputKey(
    "j_parallel_ohmic",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_PARALLEL_EXTERNAL: Final[OutputKey] = OutputKey(
    "j_parallel_external",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_PARALLEL_BOOTSTRAP: Final[OutputKey] = OutputKey(
    "j_parallel_bootstrap",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
J_TOROIDAL_TOTAL: Final[OutputKey] = OutputKey(
    "j_total",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
J_TOROIDAL_OHMIC: Final[OutputKey] = OutputKey(
    "j_ohmic",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_TOROIDAL_EXTERNAL: Final[OutputKey] = OutputKey(
    "j_external",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_TOROIDAL_BOOTSTRAP: Final[OutputKey] = OutputKey(
    "j_bootstrap",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
I_BOOTSTRAP: Final[OutputKey] = OutputKey(
    "I_bootstrap", units=Units.AMPERE, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Source profile key builders for dynamically generated keys.
# e.g. "p_ecrh_i", "p_alpha_e", "j_parallel_ecrh", "s_pellet".
# Maps internal source names to output key names where they differ.
SOURCE_NAME_RENAMES = {"fusion": "alpha"}


def p_source_i_key(source: str) -> str:
  """Returns the ion power density key for a source, e.g. 'p_alpha_i'."""
  return f"p_{source}_i"


def p_source_e_key(source: str) -> str:
  """Returns the electron power density key for a source, e.g. 'p_alpha_e'."""
  return f"p_{source}_e"


def j_parallel_source_key(source: str) -> str:
  """Returns the parallel current key for a source, e.g. 'j_parallel_ecrh'."""
  return f"j_parallel_{source}"


def s_source_key(source: str) -> str:
  """Returns the particle source key for a source, e.g. 's_pellet'."""
  return f"s_{source}"


# Fast ion key builders.
def n_fast_ion_key(source_key: str) -> str:
  """Returns the fast ion density key, e.g. 'n_fast_ion_nbi_D'."""
  return f"n_fast_ion_{source_key}"


def T_fast_ion_key(source_key: str) -> str:  # pylint: disable=invalid-name
  """Returns the fast ion temperature key, e.g. 'T_fast_ion_nbi_D'."""
  return f"T_fast_ion_{source_key}"


# ---------------------------------------------------------------------------
# Core transport.
# ---------------------------------------------------------------------------
CHI_TURB_I: Final[OutputKey] = OutputKey(
    "chi_turb_i",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_TURB_E: Final[OutputKey] = OutputKey(
    "chi_turb_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_ITG_E: Final[OutputKey] = OutputKey(
    "chi_itg_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_TEM_E: Final[OutputKey] = OutputKey(
    "chi_tem_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_ETG_E: Final[OutputKey] = OutputKey(
    "chi_etg_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_ITG_I: Final[OutputKey] = OutputKey(
    "chi_itg_i",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_TEM_I: Final[OutputKey] = OutputKey(
    "chi_tem_i",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
D_ITG_E: Final[OutputKey] = OutputKey(
    "D_itg_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
D_TEM_E: Final[OutputKey] = OutputKey(
    "D_tem_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
D_TURB_E: Final[OutputKey] = OutputKey(
    "D_turb_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
V_ITG_E: Final[OutputKey] = OutputKey(
    "V_itg_e", units=Units.METER_PER_SECOND, grid_type=GridType.FACE
)
V_TEM_E: Final[OutputKey] = OutputKey(
    "V_tem_e", units=Units.METER_PER_SECOND, grid_type=GridType.FACE
)
V_TURB_E: Final[OutputKey] = OutputKey(
    "V_turb_e", units=Units.METER_PER_SECOND, grid_type=GridType.FACE
)
CHI_NEO_I: Final[OutputKey] = OutputKey(
    "chi_neo_i",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_NEO_E: Final[OutputKey] = OutputKey(
    "chi_neo_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
D_NEO_E: Final[OutputKey] = OutputKey(
    "D_neo_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
V_NEO_E: Final[OutputKey] = OutputKey(
    "V_neo_e", units=Units.METER_PER_SECOND, grid_type=GridType.FACE
)
V_NEO_WARE_E: Final[OutputKey] = OutputKey(
    "V_neo_ware_e", units=Units.METER_PER_SECOND, grid_type=GridType.FACE
)
CHI_BOHM_E: Final[OutputKey] = OutputKey(
    "chi_bohm_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_GYROBOHM_E: Final[OutputKey] = OutputKey(
    "chi_gyrobohm_e",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_BOHM_I: Final[OutputKey] = OutputKey(
    "chi_bohm_i",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)
CHI_GYROBOHM_I: Final[OutputKey] = OutputKey(
    "chi_gyrobohm_i",
    units=Units.SQUARE_METER_PER_SECOND,
    grid_type=GridType.FACE,
)

# ---------------------------------------------------------------------------
# Coordinates.
# ---------------------------------------------------------------------------
RHO_FACE_NORM: Final[OutputKey] = OutputKey(
    "rho_face_norm", units=Units.DIMENSIONLESS, grid_type=GridType.FACE
)
RHO_CELL_NORM: Final[OutputKey] = OutputKey(
    "rho_cell_norm", units=Units.DIMENSIONLESS, grid_type=GridType.CELL
)
RHO_NORM: Final[OutputKey] = OutputKey(
    "rho_norm",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
RHO_FACE: Final[OutputKey] = OutputKey(
    "rho_face", units=Units.METER, grid_type=GridType.FACE
)
RHO_CELL: Final[OutputKey] = OutputKey(
    "rho_cell", units=Units.METER, grid_type=GridType.CELL
)
TIME: Final[OutputKey] = OutputKey(
    "time", units=Units.SECOND, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Post-processed outputs: profiles.
# ---------------------------------------------------------------------------
PPRIME: Final[OutputKey] = OutputKey(
    "pprime", units=Units.PASCAL_PER_WEBER, grid_type=GridType.FACE
)
FFPRIME: Final[OutputKey] = OutputKey(
    "FFprime", units=Units.DIMENSIONLESS, grid_type=GridType.FACE
)
PSI_NORM: Final[OutputKey] = OutputKey(
    "psi_norm", units=Units.DIMENSIONLESS, grid_type=GridType.FACE
)
J_GENERIC_CURRENT: Final[OutputKey] = OutputKey(
    "j_generic_current",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_PARALLEL_GENERIC_CURRENT: Final[OutputKey] = OutputKey(
    "j_parallel_generic_current",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_ECRH: Final[OutputKey] = OutputKey(
    "j_ecrh",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_PARALLEL_ECRH: Final[OutputKey] = OutputKey(
    "j_parallel_ecrh",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_NON_INDUCTIVE: Final[OutputKey] = OutputKey(
    "j_non_inductive",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
J_PARALLEL_NON_INDUCTIVE: Final[OutputKey] = OutputKey(
    "j_parallel_non_inductive",
    units=Units.AMPERE_PER_SQUARE_METER,
    grid_type=GridType.CELL,
)
POLOIDAL_VELOCITY: Final[OutputKey] = OutputKey(
    "poloidal_velocity",
    units=Units.METER_PER_SECOND,
    grid_type=GridType.FACE,
)
RADIAL_ELECTRIC_FIELD: Final[OutputKey] = OutputKey(
    "radial_electric_field",
    units=Units.VOLT_PER_METER,
    grid_type=GridType.FACE,
)

# ---------------------------------------------------------------------------
# Post-processed outputs: integrated powers.
# ---------------------------------------------------------------------------
P_HEAT_I: Final[OutputKey] = OutputKey(
    "P_heat_i", units=Units.WATT, grid_type=GridType.SCALAR
)
P_HEAT_E: Final[OutputKey] = OutputKey(
    "P_heat_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_HEAT_TOTAL: Final[OutputKey] = OutputKey(
    "P_heat_total", units=Units.WATT, grid_type=GridType.SCALAR
)
P_SOL_I: Final[OutputKey] = OutputKey(
    "P_SOL_i", units=Units.WATT, grid_type=GridType.SCALAR
)
P_SOL_E: Final[OutputKey] = OutputKey(
    "P_SOL_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_SOL_TOTAL: Final[OutputKey] = OutputKey(
    "P_SOL_total", units=Units.WATT, grid_type=GridType.SCALAR
)
P_AUX_I: Final[OutputKey] = OutputKey(
    "P_aux_i", units=Units.WATT, grid_type=GridType.SCALAR
)
P_AUX_E: Final[OutputKey] = OutputKey(
    "P_aux_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_AUX_TOTAL: Final[OutputKey] = OutputKey(
    "P_aux_total", units=Units.WATT, grid_type=GridType.SCALAR
)
P_EXTERNAL_INJECTED: Final[OutputKey] = OutputKey(
    "P_external_injected", units=Units.WATT, grid_type=GridType.SCALAR
)
P_EXTERNAL_TOTAL: Final[OutputKey] = OutputKey(
    "P_external_total", units=Units.WATT, grid_type=GridType.SCALAR
)
P_EI_EXCHANGE_I: Final[OutputKey] = OutputKey(
    "P_ei_exchange_i", units=Units.WATT, grid_type=GridType.SCALAR
)
P_EI_EXCHANGE_E: Final[OutputKey] = OutputKey(
    "P_ei_exchange_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_AUX_GENERIC_I: Final[OutputKey] = OutputKey(
    "P_aux_generic_i", units=Units.WATT, grid_type=GridType.SCALAR
)
P_AUX_GENERIC_E: Final[OutputKey] = OutputKey(
    "P_aux_generic_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_AUX_GENERIC_TOTAL: Final[OutputKey] = OutputKey(
    "P_aux_generic_total", units=Units.WATT, grid_type=GridType.SCALAR
)
P_ALPHA_I: Final[OutputKey] = OutputKey(
    "P_alpha_i", units=Units.WATT, grid_type=GridType.SCALAR
)
P_ALPHA_E: Final[OutputKey] = OutputKey(
    "P_alpha_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_ALPHA_TOTAL: Final[OutputKey] = OutputKey(
    "P_alpha_total", units=Units.WATT, grid_type=GridType.SCALAR
)
P_OHMIC_E: Final[OutputKey] = OutputKey(
    "P_ohmic_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_BREMSSTRAHLUNG_E: Final[OutputKey] = OutputKey(
    "P_bremsstrahlung_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_CYCLOTRON_E: Final[OutputKey] = OutputKey(
    "P_cyclotron_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_ECRH_E: Final[OutputKey] = OutputKey(
    "P_ecrh_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_RADIATION_E: Final[OutputKey] = OutputKey(
    "P_radiation_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_FUSION: Final[OutputKey] = OutputKey(
    "P_fusion", units=Units.WATT, grid_type=GridType.SCALAR
)
P_ICRH_E: Final[OutputKey] = OutputKey(
    "P_icrh_e", units=Units.WATT, grid_type=GridType.SCALAR
)
P_ICRH_I: Final[OutputKey] = OutputKey(
    "P_icrh_i", units=Units.WATT, grid_type=GridType.SCALAR
)
P_ICRH_TOTAL: Final[OutputKey] = OutputKey(
    "P_icrh_total", units=Units.WATT, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Post-processed outputs: L-H transition thresholds.
# ---------------------------------------------------------------------------
P_LH_HIGH_DENSITY: Final[OutputKey] = OutputKey(
    "P_LH_high_density", units=Units.WATT, grid_type=GridType.SCALAR
)
P_LH_MIN: Final[OutputKey] = OutputKey(
    "P_LH_min", units=Units.WATT, grid_type=GridType.SCALAR
)
P_LH_LOW_DENSITY: Final[OutputKey] = OutputKey(
    "P_LH_low_density", units=Units.WATT, grid_type=GridType.SCALAR
)
P_LH: Final[OutputKey] = OutputKey(
    "P_LH", units=Units.WATT, grid_type=GridType.SCALAR
)
N_E_MIN_P_LH: Final[OutputKey] = OutputKey(
    "n_e_min_P_LH",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.SCALAR,
)
P_LH_DELABIE_HIGH_DENSITY: Final[OutputKey] = OutputKey(
    "P_LH_delabie_high_density", units=Units.WATT, grid_type=GridType.SCALAR
)
P_LH_DELABIE_MIN: Final[OutputKey] = OutputKey(
    "P_LH_delabie_min", units=Units.WATT, grid_type=GridType.SCALAR
)
P_LH_DELABIE_LOW_DENSITY: Final[OutputKey] = OutputKey(
    "P_LH_delabie_low_density", units=Units.WATT, grid_type=GridType.SCALAR
)
P_LH_DELABIE: Final[OutputKey] = OutputKey(
    "P_LH_delabie", units=Units.WATT, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Post-processed outputs: integrated energies.
# ---------------------------------------------------------------------------
E_FUSION: Final[OutputKey] = OutputKey(
    "E_fusion", units=Units.JOULE, grid_type=GridType.SCALAR
)
E_AUX_TOTAL: Final[OutputKey] = OutputKey(
    "E_aux_total", units=Units.JOULE, grid_type=GridType.SCALAR
)
E_OHMIC_E: Final[OutputKey] = OutputKey(
    "E_ohmic_e", units=Units.JOULE, grid_type=GridType.SCALAR
)
E_EXTERNAL_INJECTED: Final[OutputKey] = OutputKey(
    "E_external_injected", units=Units.JOULE, grid_type=GridType.SCALAR
)
E_EXTERNAL_TOTAL: Final[OutputKey] = OutputKey(
    "E_external_total", units=Units.JOULE, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Post-processed outputs: stored energy and confinement.
# ---------------------------------------------------------------------------
W_THERMAL_I: Final[OutputKey] = OutputKey(
    "W_thermal_i", units=Units.JOULE, grid_type=GridType.SCALAR
)
W_THERMAL_E: Final[OutputKey] = OutputKey(
    "W_thermal_e", units=Units.JOULE, grid_type=GridType.SCALAR
)
W_THERMAL_TOTAL: Final[OutputKey] = OutputKey(
    "W_thermal_total", units=Units.JOULE, grid_type=GridType.SCALAR
)
TAU_E: Final[OutputKey] = OutputKey(
    "tau_E", units=Units.SECOND, grid_type=GridType.SCALAR
)
H89P: Final[OutputKey] = OutputKey(
    "H89P", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
H98: Final[OutputKey] = OutputKey(
    "H98", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
H97L: Final[OutputKey] = OutputKey(
    "H97L", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
H20: Final[OutputKey] = OutputKey(
    "H20", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
W_POL: Final[OutputKey] = OutputKey(
    "W_pol", units=Units.JOULE, grid_type=GridType.SCALAR
)
LI3: Final[OutputKey] = OutputKey(
    "li3", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
DW_THERMAL_DT: Final[OutputKey] = OutputKey(
    "dW_thermal_dt", units=Units.WATT, grid_type=GridType.SCALAR
)
DW_THERMAL_DT_SMOOTHED: Final[OutputKey] = OutputKey(
    "dW_thermal_dt_smoothed", units=Units.WATT, grid_type=GridType.SCALAR
)
DW_THERMAL_I_DT_SMOOTHED: Final[OutputKey] = OutputKey(
    "dW_thermal_i_dt_smoothed", units=Units.WATT, grid_type=GridType.SCALAR
)
DW_THERMAL_E_DT_SMOOTHED: Final[OutputKey] = OutputKey(
    "dW_thermal_e_dt_smoothed", units=Units.WATT, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Post-processed outputs: volume/line averages.
# ---------------------------------------------------------------------------
T_E_VOLUME_AVG: Final[OutputKey] = OutputKey(
    "T_e_volume_avg", units=Units.KEV, grid_type=GridType.SCALAR
)
T_I_VOLUME_AVG: Final[OutputKey] = OutputKey(
    "T_i_volume_avg", units=Units.KEV, grid_type=GridType.SCALAR
)
N_E_VOLUME_AVG: Final[OutputKey] = OutputKey(
    "n_e_volume_avg",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.SCALAR,
)
N_I_VOLUME_AVG: Final[OutputKey] = OutputKey(
    "n_i_volume_avg",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.SCALAR,
)
N_E_LINE_AVG: Final[OutputKey] = OutputKey(
    "n_e_line_avg",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.SCALAR,
)
N_I_LINE_AVG: Final[OutputKey] = OutputKey(
    "n_i_line_avg",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.SCALAR,
)
FGW_N_E_VOLUME_AVG: Final[OutputKey] = OutputKey(
    "fgw_n_e_volume_avg",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.SCALAR,
)
FGW_N_E_LINE_AVG: Final[OutputKey] = OutputKey(
    "fgw_n_e_line_avg",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.SCALAR,
)

# ---------------------------------------------------------------------------
# Post-processed outputs: q-profile derived scalars.
# ---------------------------------------------------------------------------
Q_FUSION: Final[OutputKey] = OutputKey(
    "Q_fusion", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
Q95: Final[OutputKey] = OutputKey(
    "q95", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
Q_MIN: Final[OutputKey] = OutputKey(
    "q_min", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
RHO_Q_MIN: Final[OutputKey] = OutputKey(
    "rho_q_min", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
RHO_Q_3_2_FIRST: Final[OutputKey] = OutputKey(
    "rho_q_3_2_first", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
RHO_Q_2_1_FIRST: Final[OutputKey] = OutputKey(
    "rho_q_2_1_first", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
RHO_Q_3_1_FIRST: Final[OutputKey] = OutputKey(
    "rho_q_3_1_first", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
RHO_Q_3_2_SECOND: Final[OutputKey] = OutputKey(
    "rho_q_3_2_second", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
RHO_Q_2_1_SECOND: Final[OutputKey] = OutputKey(
    "rho_q_2_1_second", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
RHO_Q_3_1_SECOND: Final[OutputKey] = OutputKey(
    "rho_q_3_1_second", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Post-processed outputs: integrated currents and fractions.
# ---------------------------------------------------------------------------
I_EXTERNAL: Final[OutputKey] = OutputKey(
    "I_external", units=Units.AMPERE, grid_type=GridType.SCALAR
)
I_ECRH: Final[OutputKey] = OutputKey(
    "I_ecrh", units=Units.AMPERE, grid_type=GridType.SCALAR
)
I_AUX_GENERIC: Final[OutputKey] = OutputKey(
    "I_aux_generic", units=Units.AMPERE, grid_type=GridType.SCALAR
)
I_NON_INDUCTIVE: Final[OutputKey] = OutputKey(
    "I_non_inductive", units=Units.AMPERE, grid_type=GridType.SCALAR
)
F_NON_INDUCTIVE: Final[OutputKey] = OutputKey(
    "f_non_inductive", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
F_BOOTSTRAP: Final[OutputKey] = OutputKey(
    "f_bootstrap", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Post-processed outputs: integrated particle sources.
# ---------------------------------------------------------------------------
S_GAS_PUFF: Final[OutputKey] = OutputKey(
    "S_gas_puff", units=Units.INVERSE_SECOND, grid_type=GridType.SCALAR
)
S_PELLET: Final[OutputKey] = OutputKey(
    "S_pellet", units=Units.INVERSE_SECOND, grid_type=GridType.SCALAR
)
S_GENERIC_PARTICLE: Final[OutputKey] = OutputKey(
    "S_generic_particle", units=Units.INVERSE_SECOND, grid_type=GridType.SCALAR
)
S_TOTAL: Final[OutputKey] = OutputKey(
    "S_total", units=Units.INVERSE_SECOND, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Post-processed outputs: plasma beta.
# ---------------------------------------------------------------------------
BETA_TOR: Final[OutputKey] = OutputKey(
    "beta_tor", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
BETA_POL: Final[OutputKey] = OutputKey(
    "beta_pol", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
BETA_N: Final[OutputKey] = OutputKey(
    "beta_N", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Edge model outputs.
# ---------------------------------------------------------------------------
SEED_IMPURITY_CONCENTRATIONS: Final[OutputKey] = OutputKey(
    "seed_impurity_concentrations",
    units=Units.INVERSE_CUBIC_METER,
    grid_type=GridType.NOT_APPLICABLE,
)
CALCULATED_ENRICHMENT: Final[OutputKey] = OutputKey(
    "calculated_enrichment",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.NOT_APPLICABLE,
)
IMPURITY: Final[OutputKey] = OutputKey(
    "impurity", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)
SEED_IMPURITY: Final[OutputKey] = OutputKey(
    "seed_impurity",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.NOT_APPLICABLE,
)
MAIN_ION: Final[OutputKey] = OutputKey(
    "main_ion", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)
RADIATION_IMPURITY_SPECIES: Final[OutputKey] = OutputKey(
    "radiation_impurity_species",
    units=Units.WATT_PER_CUBIC_METER,
    grid_type=GridType.CELL,
)

# ---------------------------------------------------------------------------
# Edge model scalar outputs.
# ---------------------------------------------------------------------------
Q_PARALLEL: Final[OutputKey] = OutputKey(
    "q_parallel",
    units=Units.WATT_PER_SQUARE_METER,
    grid_type=GridType.SCALAR,
)
Q_PERPENDICULAR_TARGET: Final[OutputKey] = OutputKey(
    "q_perpendicular_target",
    units=Units.WATT_PER_SQUARE_METER,
    grid_type=GridType.SCALAR,
)
T_E_SEPARATRIX: Final[OutputKey] = OutputKey(
    "T_e_separatrix", units=Units.KEV, grid_type=GridType.SCALAR
)
T_E_TARGET: Final[OutputKey] = OutputKey(
    "T_e_target", units=Units.EV, grid_type=GridType.SCALAR
)
PRESSURE_NEUTRAL_DIVERTOR: Final[OutputKey] = OutputKey(
    "pressure_neutral_divertor",
    units=Units.PASCAL,
    grid_type=GridType.SCALAR,
)
ALPHA_T: Final[OutputKey] = OutputKey(
    "alpha_t", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
Z_EFF_SEPARATRIX: Final[OutputKey] = OutputKey(
    "Z_eff_separatrix", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)
MULTIPLE_ROOTS_FOUND: Final[OutputKey] = OutputKey(
    "multiple_roots_found",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.SCALAR,
)

# ---------------------------------------------------------------------------
# Numerics.
# ---------------------------------------------------------------------------
SIM_STATUS: Final[OutputKey] = OutputKey(
    "sim_status", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)
SIM_ERROR: Final[OutputKey] = OutputKey(
    "sim_error", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)
OUTER_SOLVER_ITERATIONS: Final[OutputKey] = OutputKey(
    "outer_solver_iterations",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.NOT_APPLICABLE,
)
INNER_SOLVER_ITERATIONS: Final[OutputKey] = OutputKey(
    "inner_solver_iterations",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.NOT_APPLICABLE,
)
# Boolean array indicating whether the state corresponds to a
# post-sawtooth-crash state.
SAWTOOTH_CRASH: Final[OutputKey] = OutputKey(
    "sawtooth_crash",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.NOT_APPLICABLE,
)

# ---------------------------------------------------------------------------
# ToraxConfig.
# ---------------------------------------------------------------------------
CONFIG: Final[OutputKey] = OutputKey(
    "config", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)

# ---------------------------------------------------------------------------
# Geometry: scalar quantities.
# ---------------------------------------------------------------------------
B_0: Final[OutputKey] = OutputKey(
    "B_0", units=Units.TESLA, grid_type=GridType.SCALAR
)
R_MAJOR: Final[OutputKey] = OutputKey(
    "R_major", units=Units.METER, grid_type=GridType.SCALAR
)
A_MINOR: Final[OutputKey] = OutputKey(
    "a_minor", units=Units.METER, grid_type=GridType.SCALAR
)
PHI_B: Final[OutputKey] = OutputKey(
    "Phi_b", units=Units.WEBER, grid_type=GridType.SCALAR
)
PHI_B_DOT: Final[OutputKey] = OutputKey(
    "Phi_b_dot", units=Units.WEBER_PER_SECOND, grid_type=GridType.SCALAR
)
RHO_B: Final[OutputKey] = OutputKey(
    "rho_b", units=Units.METER, grid_type=GridType.SCALAR
)
DRHO: Final[OutputKey] = OutputKey(
    "drho", units=Units.METER, grid_type=GridType.SCALAR
)
DRHO_NORM: Final[OutputKey] = OutputKey(
    "drho_norm", units=Units.DIMENSIONLESS, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Geometry: profile quantities.
# ---------------------------------------------------------------------------
PHI: Final[OutputKey] = OutputKey(
    "Phi", units=Units.WEBER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
TFF: Final[OutputKey] = OutputKey(
    "F", units=Units.TESLA_METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
R_IN: Final[OutputKey] = OutputKey(
    "R_in", units=Units.METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
R_OUT: Final[OutputKey] = OutputKey(
    "R_out", units=Units.METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
R_MAJOR_PROFILE: Final[OutputKey] = OutputKey(
    "R_major_profile",
    units=Units.METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
R_MID: Final[OutputKey] = OutputKey(
    "r_mid", units=Units.METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
AREA: Final[OutputKey] = OutputKey(
    "area", units=Units.SQUARE_METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
VOLUME: Final[OutputKey] = OutputKey(
    "volume", units=Units.CUBIC_METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
VPR: Final[OutputKey] = OutputKey(
    "vpr", units=Units.CUBIC_METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
SPR: Final[OutputKey] = OutputKey(
    "spr", units=Units.SQUARE_METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
DELTA: Final[OutputKey] = OutputKey(
    "delta", units=Units.DIMENSIONLESS, grid_type=GridType.FACE
)
DELTA_UPPER: Final[OutputKey] = OutputKey(
    "delta_upper",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.FACE,
)
DELTA_LOWER: Final[OutputKey] = OutputKey(
    "delta_lower",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.FACE,
)
ELONGATION: Final[OutputKey] = OutputKey(
    "elongation",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
EPSILON: Final[OutputKey] = OutputKey(
    "epsilon",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
G0: Final[OutputKey] = OutputKey(
    "g0", units=Units.SQUARE_METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
G0_OVER_VPR: Final[OutputKey] = OutputKey(
    "g0_over_vpr", units=Units.INVERSE_METER, grid_type=GridType.FACE
)
G1: Final[OutputKey] = OutputKey(
    "g1", units=Units.QUARTIC_METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
G1_OVER_VPR: Final[OutputKey] = OutputKey(
    "g1_over_vpr", units=Units.METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
G1_OVER_VPR2: Final[OutputKey] = OutputKey(
    "g1_over_vpr2",
    units=Units.INVERSE_SQUARE_METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
G2: Final[OutputKey] = OutputKey(
    "g2", units=Units.SQUARE_METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
G2G3_OVER_RHON: Final[OutputKey] = OutputKey(
    "g2g3_over_rhon",
    units=Units.DIMENSIONLESS,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
G3: Final[OutputKey] = OutputKey(
    "g3",
    units=Units.INVERSE_SQUARE_METER,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
GM4: Final[OutputKey] = OutputKey(
    "gm4",
    units=Units.INVERSE_SQUARE_TESLA,
    grid_type=GridType.CELL_PLUS_BOUNDARIES,
)
GM5: Final[OutputKey] = OutputKey(
    "gm5", units=Units.SQUARE_TESLA, grid_type=GridType.CELL_PLUS_BOUNDARIES
)
GM9: Final[OutputKey] = OutputKey(
    "gm9", units=Units.INVERSE_METER, grid_type=GridType.CELL_PLUS_BOUNDARIES
)

# ---------------------------------------------------------------------------
# Geometry output renames.
# ---------------------------------------------------------------------------
IP_PROFILE_FROM_GEO: Final[OutputKey] = OutputKey(
    "Ip_profile_from_geo", units=Units.AMPERE, grid_type=GridType.FACE
)
PSI_FROM_GEO: Final[OutputKey] = OutputKey(
    "psi_from_geo", units=Units.WEBER, grid_type=GridType.CELL
)
Z_MAGNETIC_AXIS: Final[OutputKey] = OutputKey(
    "z_magnetic_axis", units=Units.METER, grid_type=GridType.SCALAR
)

# ---------------------------------------------------------------------------
# Solver / edge numerics output keys.
# ---------------------------------------------------------------------------
SOLVER_PHYSICS_OUTCOME: Final[OutputKey] = OutputKey(
    "solver_physics_outcome",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.SCALAR,
)
SOLVER_ITERATIONS: Final[OutputKey] = OutputKey(
    "solver_iterations",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.SCALAR,
)
SOLVER_RESIDUAL: Final[OutputKey] = OutputKey(
    "solver_residual",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.SCALAR,
)
SOLVER_ERROR: Final[OutputKey] = OutputKey(
    "solver_error",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.SCALAR,
)
FIXED_POINT_OUTCOME: Final[OutputKey] = OutputKey(
    "fixed_point_outcome",
    units=Units.NOT_APPLICABLE,
    grid_type=GridType.SCALAR,
)
ROOTS: Final[OutputKey] = OutputKey(
    "roots", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)
N_ROOTS: Final[OutputKey] = OutputKey(
    "n_roots", units=Units.NOT_APPLICABLE, grid_type=GridType.NOT_APPLICABLE
)

# ---------------------------------------------------------------------------
# Prefix-based unit matching for dynamically-generated source profile names.
#
# Source profile output names are constructed at runtime from the source name
# and species (e.g. ``p_ecrh_e``, ``s_pellet``, ``j_parallel_eccd``).
# The prefix determines the physical quantity and hence the unit.
# ---------------------------------------------------------------------------
_SOURCE_UNIT_PREFIXES: dict[str, Units] = {
    "p_": Units.MW_PER_CUBIC_METER,
    "j_parallel_": Units.AMPERE_PER_SQUARE_METER,
    "s_": Units.INVERSE_CUBIC_METER_PER_SECOND,
    "n_fast_ion_": Units.INVERSE_CUBIC_METER,
    "T_fast_ion_": Units.KEV,
}


def _build_units_by_name() -> dict[str, str]:
  """Builds a name -> unit lookup from all module-level OutputKey constants."""
  module = sys.modules[__name__]
  result = {}
  for attr in dir(module):
    value = getattr(module, attr)
    if isinstance(value, OutputKey) and value.units:
      result[str(value)] = value.units
  return result


# Reverse lookup: string value -> unit.  Used when callers pass a plain ``str``
# rather than an ``OutputKey`` instance.
_UNITS_BY_NAME: dict[str, str] = _build_units_by_name()


def get_units(name: str) -> dict[str, str]:
  """Returns xarray attrs dict with units for the given variable name.

  Lookup order:

  1. If ``name`` is an ``OutputKey`` with a non-empty ``units`` attribute, that
     unit is returned directly.
  2. If ``name`` is a plain ``str`` that matches the string value of a known
     ``OutputKey`` constant, the unit from that constant is returned.
  3. Prefix matching against known source-profile patterns.
  4. Otherwise an empty dict is returned so callers can unconditionally unpack.

  Args:
    name: The output variable name to look up units for.
  """
  # Fast path: OutputKey instances carry their own unit.
  if isinstance(name, OutputKey) and name.units:
    return {"units": name.units}
  # Reverse lookup for plain str matching a known OutputKey value.
  unit = _UNITS_BY_NAME.get(name)
  if unit is not None:
    return {"units": unit}
  # Fallback: prefix matching for dynamically-generated source profiles.
  for prefix, source_unit in _SOURCE_UNIT_PREFIXES.items():
    if name.startswith(prefix):
      return {"units": source_unit}
  return {}
