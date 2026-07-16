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
``units`` attribute from the ``Units`` enum.  This means every constant
(e.g. ``T_E``) can be used wherever a plain string is expected (dict keys,
xarray names, ``==`` comparisons) while also carrying its physical unit
metadata.

Physical dimensionless quantities (e.g. safety factor *q*) use
``Units.DIMENSIONLESS``.  Non-physical keys (e.g. solver status, dataset
names) use ``Units.NOT_APPLICABLE`` (the empty string ``""``).

For source profiles whose names are generated at runtime (e.g. ``p_ecrh_e``),
``get_units`` falls back to prefix-based matching so that dynamically-
constructed keys still receive the correct unit string.
"""

# pylint: disable=invalid-name

import enum
import sys


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


class OutputKey(str):
  """A string key that carries unit metadata from the ``Units`` enum.

  ``OutputKey`` inherits from ``str`` so it can be used anywhere a regular
  string is expected.  The ``units`` attribute stores the physical unit as a
  ``Units`` enum member and is required.

  Examples::

      >>> T_E = OutputKey("T_e", units=Units.KEV)
      >>> T_E == "T_e"
      True
      >>> T_E.units
      <Units.KEV: 'keV'>
  """

  units: Units

  def __new__(cls, value: str, *, units: Units) -> "OutputKey":
    obj = str.__new__(cls, value)
    obj.units = units
    return obj

  def __getnewargs_ex__(self):
    # Required so that deepcopy/pickle can reconstruct the keyword-only arg.
    return ((str(self),), {"units": self.units})


# ---------------------------------------------------------------------------
# Dataset names (no physical units).
# ---------------------------------------------------------------------------
PROFILES = OutputKey("profiles", units=Units.NOT_APPLICABLE)
SCALARS = OutputKey("scalars", units=Units.NOT_APPLICABLE)
NUMERICS = OutputKey("numerics", units=Units.NOT_APPLICABLE)
EDGE = OutputKey("edge", units=Units.NOT_APPLICABLE)

# ---------------------------------------------------------------------------
# Core profiles.
# ---------------------------------------------------------------------------
T_E = OutputKey("T_e", units=Units.KEV)
T_I = OutputKey("T_i", units=Units.KEV)
PSI = OutputKey("psi", units=Units.WEBER)
V_LOOP = OutputKey("v_loop", units=Units.VOLT)
N_E = OutputKey("n_e", units=Units.INVERSE_CUBIC_METER)
N_I = OutputKey("n_i", units=Units.INVERSE_CUBIC_METER)
Q = OutputKey("q", units=Units.DIMENSIONLESS)
MAGNETIC_SHEAR = OutputKey("magnetic_shear", units=Units.DIMENSIONLESS)
N_IMPURITY = OutputKey("n_impurity", units=Units.INVERSE_CUBIC_METER)
Z_IMPURITY = OutputKey("Z_impurity", units=Units.DIMENSIONLESS)
Z_EFF = OutputKey("Z_eff", units=Units.DIMENSIONLESS)
SIGMA_PARALLEL = OutputKey("sigma_parallel", units=Units.SIEMENS_PER_METER)
V_LOOP_LCFS = OutputKey("v_loop_lcfs", units=Units.VOLT)
IP_PROFILE = OutputKey("Ip_profile", units=Units.AMPERE)
IP = OutputKey("Ip", units=Units.AMPERE)
TOROIDAL_ANGULAR_VELOCITY = OutputKey(
    "toroidal_angular_velocity", units=Units.RAD_PER_SECOND
)
A_I = OutputKey("A_i", units=Units.AMU)
A_IMPURITY = OutputKey("A_impurity", units=Units.AMU)
Z_I = OutputKey("Z_i", units=Units.DIMENSIONLESS)
Z_IMPURITY_SPECIES = OutputKey("Z_impurity_species", units=Units.DIMENSIONLESS)
N_IMPURITY_SPECIES = OutputKey(
    "n_impurity_species", units=Units.INVERSE_CUBIC_METER
)
MAIN_ION_FRACTIONS = OutputKey("main_ion_fractions", units=Units.DIMENSIONLESS)
PRESSURE_THERMAL_E = OutputKey("pressure_thermal_e", units=Units.PASCAL)
PRESSURE_THERMAL_I = OutputKey("pressure_thermal_i", units=Units.PASCAL)
PRESSURE_THERMAL_TOTAL = OutputKey("pressure_thermal_total", units=Units.PASCAL)
PRESSURE_FAST_I = OutputKey("pressure_fast_i", units=Units.PASCAL)
PRESSURE_TOTAL_I = OutputKey("pressure_total_i", units=Units.PASCAL)
PRESSURE_TOTAL = OutputKey("pressure_total", units=Units.PASCAL)
EI_EXCHANGE = OutputKey("ei_exchange", units=Units.MW_PER_CUBIC_METER)
PSI_FROM_IP = OutputKey("psi_from_Ip", units=Units.WEBER)

# ---------------------------------------------------------------------------
# Calculated or derived current densities (excluding sources).
# ---------------------------------------------------------------------------
J_PARALLEL_TOTAL = OutputKey(
    "j_parallel_total", units=Units.AMPERE_PER_SQUARE_METER
)
J_PARALLEL_OHMIC = OutputKey(
    "j_parallel_ohmic", units=Units.AMPERE_PER_SQUARE_METER
)
J_PARALLEL_EXTERNAL = OutputKey(
    "j_parallel_external", units=Units.AMPERE_PER_SQUARE_METER
)
J_PARALLEL_BOOTSTRAP = OutputKey(
    "j_parallel_bootstrap", units=Units.AMPERE_PER_SQUARE_METER
)
J_TOROIDAL_TOTAL = OutputKey("j_total", units=Units.AMPERE_PER_SQUARE_METER)
J_TOROIDAL_OHMIC = OutputKey("j_ohmic", units=Units.AMPERE_PER_SQUARE_METER)
J_TOROIDAL_EXTERNAL = OutputKey(
    "j_external", units=Units.AMPERE_PER_SQUARE_METER
)
J_TOROIDAL_BOOTSTRAP = OutputKey(
    "j_bootstrap", units=Units.AMPERE_PER_SQUARE_METER
)
I_BOOTSTRAP = OutputKey("I_bootstrap", units=Units.AMPERE)

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
CHI_TURB_I = OutputKey("chi_turb_i", units=Units.SQUARE_METER_PER_SECOND)
CHI_TURB_E = OutputKey("chi_turb_e", units=Units.SQUARE_METER_PER_SECOND)
CHI_ITG_E = OutputKey("chi_itg_e", units=Units.SQUARE_METER_PER_SECOND)
CHI_TEM_E = OutputKey("chi_tem_e", units=Units.SQUARE_METER_PER_SECOND)
CHI_ETG_E = OutputKey("chi_etg_e", units=Units.SQUARE_METER_PER_SECOND)
CHI_ITG_I = OutputKey("chi_itg_i", units=Units.SQUARE_METER_PER_SECOND)
CHI_TEM_I = OutputKey("chi_tem_i", units=Units.SQUARE_METER_PER_SECOND)
D_ITG_E = OutputKey("D_itg_e", units=Units.SQUARE_METER_PER_SECOND)
D_TEM_E = OutputKey("D_tem_e", units=Units.SQUARE_METER_PER_SECOND)
D_TURB_E = OutputKey("D_turb_e", units=Units.SQUARE_METER_PER_SECOND)
V_ITG_E = OutputKey("V_itg_e", units=Units.METER_PER_SECOND)
V_TEM_E = OutputKey("V_tem_e", units=Units.METER_PER_SECOND)
V_TURB_E = OutputKey("V_turb_e", units=Units.METER_PER_SECOND)
CHI_NEO_I = OutputKey("chi_neo_i", units=Units.SQUARE_METER_PER_SECOND)
CHI_NEO_E = OutputKey("chi_neo_e", units=Units.SQUARE_METER_PER_SECOND)
D_NEO_E = OutputKey("D_neo_e", units=Units.SQUARE_METER_PER_SECOND)
V_NEO_E = OutputKey("V_neo_e", units=Units.METER_PER_SECOND)
V_NEO_WARE_E = OutputKey("V_neo_ware_e", units=Units.METER_PER_SECOND)
CHI_BOHM_E = OutputKey("chi_bohm_e", units=Units.SQUARE_METER_PER_SECOND)
CHI_GYROBOHM_E = OutputKey(
    "chi_gyrobohm_e", units=Units.SQUARE_METER_PER_SECOND
)
CHI_BOHM_I = OutputKey("chi_bohm_i", units=Units.SQUARE_METER_PER_SECOND)
CHI_GYROBOHM_I = OutputKey(
    "chi_gyrobohm_i", units=Units.SQUARE_METER_PER_SECOND
)

# ---------------------------------------------------------------------------
# Coordinates.
# ---------------------------------------------------------------------------
RHO_FACE_NORM = OutputKey("rho_face_norm", units=Units.DIMENSIONLESS)
RHO_CELL_NORM = OutputKey("rho_cell_norm", units=Units.DIMENSIONLESS)
RHO_NORM = OutputKey("rho_norm", units=Units.DIMENSIONLESS)
RHO_FACE = OutputKey("rho_face", units=Units.METER)
RHO_CELL = OutputKey("rho_cell", units=Units.METER)
TIME = OutputKey("time", units=Units.SECOND)

# ---------------------------------------------------------------------------
# Post-processed outputs: profiles.
# ---------------------------------------------------------------------------
PPRIME = OutputKey("pprime", units=Units.PASCAL_PER_WEBER)
FFPRIME = OutputKey("FFprime", units=Units.DIMENSIONLESS)
PSI_NORM = OutputKey("psi_norm", units=Units.DIMENSIONLESS)
J_GENERIC_CURRENT = OutputKey(
    "j_generic_current", units=Units.AMPERE_PER_SQUARE_METER
)
J_PARALLEL_GENERIC_CURRENT = OutputKey(
    "j_parallel_generic_current", units=Units.AMPERE_PER_SQUARE_METER
)
J_ECRH = OutputKey("j_ecrh", units=Units.AMPERE_PER_SQUARE_METER)
J_PARALLEL_ECRH = OutputKey(
    "j_parallel_ecrh", units=Units.AMPERE_PER_SQUARE_METER
)
J_NON_INDUCTIVE = OutputKey(
    "j_non_inductive", units=Units.AMPERE_PER_SQUARE_METER
)
J_PARALLEL_NON_INDUCTIVE = OutputKey(
    "j_parallel_non_inductive", units=Units.AMPERE_PER_SQUARE_METER
)
POLOIDAL_VELOCITY = OutputKey("poloidal_velocity", units=Units.METER_PER_SECOND)
RADIAL_ELECTRIC_FIELD = OutputKey(
    "radial_electric_field", units=Units.VOLT_PER_METER
)

# ---------------------------------------------------------------------------
# Post-processed outputs: integrated powers.
# ---------------------------------------------------------------------------
P_HEAT_I = OutputKey("P_heat_i", units=Units.WATT)
P_HEAT_E = OutputKey("P_heat_e", units=Units.WATT)
P_HEAT_TOTAL = OutputKey("P_heat_total", units=Units.WATT)
P_SOL_I = OutputKey("P_SOL_i", units=Units.WATT)
P_SOL_E = OutputKey("P_SOL_e", units=Units.WATT)
P_SOL_TOTAL = OutputKey("P_SOL_total", units=Units.WATT)
P_AUX_I = OutputKey("P_aux_i", units=Units.WATT)
P_AUX_E = OutputKey("P_aux_e", units=Units.WATT)
P_AUX_TOTAL = OutputKey("P_aux_total", units=Units.WATT)
P_EXTERNAL_INJECTED = OutputKey("P_external_injected", units=Units.WATT)
P_EXTERNAL_TOTAL = OutputKey("P_external_total", units=Units.WATT)
P_EI_EXCHANGE_I = OutputKey("P_ei_exchange_i", units=Units.WATT)
P_EI_EXCHANGE_E = OutputKey("P_ei_exchange_e", units=Units.WATT)
P_AUX_GENERIC_I = OutputKey("P_aux_generic_i", units=Units.WATT)
P_AUX_GENERIC_E = OutputKey("P_aux_generic_e", units=Units.WATT)
P_AUX_GENERIC_TOTAL = OutputKey("P_aux_generic_total", units=Units.WATT)
P_ALPHA_I = OutputKey("P_alpha_i", units=Units.WATT)
P_ALPHA_E = OutputKey("P_alpha_e", units=Units.WATT)
P_ALPHA_TOTAL = OutputKey("P_alpha_total", units=Units.WATT)
P_OHMIC_E = OutputKey("P_ohmic_e", units=Units.WATT)
P_BREMSSTRAHLUNG_E = OutputKey("P_bremsstrahlung_e", units=Units.WATT)
P_CYCLOTRON_E = OutputKey("P_cyclotron_e", units=Units.WATT)
P_ECRH_E = OutputKey("P_ecrh_e", units=Units.WATT)
P_RADIATION_E = OutputKey("P_radiation_e", units=Units.WATT)
P_FUSION = OutputKey("P_fusion", units=Units.WATT)
P_ICRH_E = OutputKey("P_icrh_e", units=Units.WATT)
P_ICRH_I = OutputKey("P_icrh_i", units=Units.WATT)
P_ICRH_TOTAL = OutputKey("P_icrh_total", units=Units.WATT)

# ---------------------------------------------------------------------------
# Post-processed outputs: L-H transition thresholds.
# ---------------------------------------------------------------------------
P_LH_HIGH_DENSITY = OutputKey("P_LH_high_density", units=Units.WATT)
P_LH_MIN = OutputKey("P_LH_min", units=Units.WATT)
P_LH_LOW_DENSITY = OutputKey("P_LH_low_density", units=Units.WATT)
P_LH = OutputKey("P_LH", units=Units.WATT)
N_E_MIN_P_LH = OutputKey("n_e_min_P_LH", units=Units.INVERSE_CUBIC_METER)
P_LH_DELABIE_HIGH_DENSITY = OutputKey(
    "P_LH_delabie_high_density", units=Units.WATT
)
P_LH_DELABIE_MIN = OutputKey("P_LH_delabie_min", units=Units.WATT)
P_LH_DELABIE_LOW_DENSITY = OutputKey(
    "P_LH_delabie_low_density", units=Units.WATT
)
P_LH_DELABIE = OutputKey("P_LH_delabie", units=Units.WATT)

# ---------------------------------------------------------------------------
# Post-processed outputs: integrated energies.
# ---------------------------------------------------------------------------
E_FUSION = OutputKey("E_fusion", units=Units.JOULE)
E_AUX_TOTAL = OutputKey("E_aux_total", units=Units.JOULE)
E_OHMIC_E = OutputKey("E_ohmic_e", units=Units.JOULE)
E_EXTERNAL_INJECTED = OutputKey("E_external_injected", units=Units.JOULE)
E_EXTERNAL_TOTAL = OutputKey("E_external_total", units=Units.JOULE)

# ---------------------------------------------------------------------------
# Post-processed outputs: stored energy and confinement.
# ---------------------------------------------------------------------------
W_THERMAL_I = OutputKey("W_thermal_i", units=Units.JOULE)
W_THERMAL_E = OutputKey("W_thermal_e", units=Units.JOULE)
W_THERMAL_TOTAL = OutputKey("W_thermal_total", units=Units.JOULE)
TAU_E = OutputKey("tau_E", units=Units.SECOND)
H89P = OutputKey("H89P", units=Units.DIMENSIONLESS)
H98 = OutputKey("H98", units=Units.DIMENSIONLESS)
H97L = OutputKey("H97L", units=Units.DIMENSIONLESS)
H20 = OutputKey("H20", units=Units.DIMENSIONLESS)
W_POL = OutputKey("W_pol", units=Units.JOULE)
LI3 = OutputKey("li3", units=Units.DIMENSIONLESS)
DW_THERMAL_DT = OutputKey("dW_thermal_dt", units=Units.WATT)
DW_THERMAL_DT_SMOOTHED = OutputKey("dW_thermal_dt_smoothed", units=Units.WATT)
DW_THERMAL_I_DT_SMOOTHED = OutputKey(
    "dW_thermal_i_dt_smoothed", units=Units.WATT
)
DW_THERMAL_E_DT_SMOOTHED = OutputKey(
    "dW_thermal_e_dt_smoothed", units=Units.WATT
)

# ---------------------------------------------------------------------------
# Post-processed outputs: volume/line averages.
# ---------------------------------------------------------------------------
T_E_VOLUME_AVG = OutputKey("T_e_volume_avg", units=Units.KEV)
T_I_VOLUME_AVG = OutputKey("T_i_volume_avg", units=Units.KEV)
N_E_VOLUME_AVG = OutputKey("n_e_volume_avg", units=Units.INVERSE_CUBIC_METER)
N_I_VOLUME_AVG = OutputKey("n_i_volume_avg", units=Units.INVERSE_CUBIC_METER)
N_E_LINE_AVG = OutputKey("n_e_line_avg", units=Units.INVERSE_CUBIC_METER)
N_I_LINE_AVG = OutputKey("n_i_line_avg", units=Units.INVERSE_CUBIC_METER)
FGW_N_E_VOLUME_AVG = OutputKey("fgw_n_e_volume_avg", units=Units.DIMENSIONLESS)
FGW_N_E_LINE_AVG = OutputKey("fgw_n_e_line_avg", units=Units.DIMENSIONLESS)

# ---------------------------------------------------------------------------
# Post-processed outputs: q-profile derived scalars.
# ---------------------------------------------------------------------------
Q_FUSION = OutputKey("Q_fusion", units=Units.DIMENSIONLESS)
Q95 = OutputKey("q95", units=Units.DIMENSIONLESS)
Q_MIN = OutputKey("q_min", units=Units.DIMENSIONLESS)
RHO_Q_MIN = OutputKey("rho_q_min", units=Units.DIMENSIONLESS)
RHO_Q_3_2_FIRST = OutputKey("rho_q_3_2_first", units=Units.DIMENSIONLESS)
RHO_Q_2_1_FIRST = OutputKey("rho_q_2_1_first", units=Units.DIMENSIONLESS)
RHO_Q_3_1_FIRST = OutputKey("rho_q_3_1_first", units=Units.DIMENSIONLESS)
RHO_Q_3_2_SECOND = OutputKey("rho_q_3_2_second", units=Units.DIMENSIONLESS)
RHO_Q_2_1_SECOND = OutputKey("rho_q_2_1_second", units=Units.DIMENSIONLESS)
RHO_Q_3_1_SECOND = OutputKey("rho_q_3_1_second", units=Units.DIMENSIONLESS)

# ---------------------------------------------------------------------------
# Post-processed outputs: integrated currents and fractions.
# ---------------------------------------------------------------------------
I_EXTERNAL = OutputKey("I_external", units=Units.AMPERE)
I_ECRH = OutputKey("I_ecrh", units=Units.AMPERE)
I_AUX_GENERIC = OutputKey("I_aux_generic", units=Units.AMPERE)
I_NON_INDUCTIVE = OutputKey("I_non_inductive", units=Units.AMPERE)
F_NON_INDUCTIVE = OutputKey("f_non_inductive", units=Units.DIMENSIONLESS)
F_BOOTSTRAP = OutputKey("f_bootstrap", units=Units.DIMENSIONLESS)

# ---------------------------------------------------------------------------
# Post-processed outputs: integrated particle sources.
# ---------------------------------------------------------------------------
S_GAS_PUFF = OutputKey("S_gas_puff", units=Units.INVERSE_SECOND)
S_PELLET = OutputKey("S_pellet", units=Units.INVERSE_SECOND)
S_GENERIC_PARTICLE = OutputKey("S_generic_particle", units=Units.INVERSE_SECOND)
S_TOTAL = OutputKey("S_total", units=Units.INVERSE_SECOND)

# ---------------------------------------------------------------------------
# Post-processed outputs: plasma beta.
# ---------------------------------------------------------------------------
BETA_TOR = OutputKey("beta_tor", units=Units.DIMENSIONLESS)
BETA_POL = OutputKey("beta_pol", units=Units.DIMENSIONLESS)
BETA_N = OutputKey("beta_N", units=Units.DIMENSIONLESS)

# ---------------------------------------------------------------------------
# Edge model outputs.
# ---------------------------------------------------------------------------
SEED_IMPURITY_CONCENTRATIONS = OutputKey(
    "seed_impurity_concentrations", units=Units.INVERSE_CUBIC_METER
)
CALCULATED_ENRICHMENT = OutputKey(
    "calculated_enrichment", units=Units.DIMENSIONLESS
)
IMPURITY = OutputKey("impurity", units=Units.NOT_APPLICABLE)
SEED_IMPURITY = OutputKey("seed_impurity", units=Units.NOT_APPLICABLE)
MAIN_ION = OutputKey("main_ion", units=Units.NOT_APPLICABLE)
RADIATION_IMPURITY_SPECIES = OutputKey(
    "radiation_impurity_species", units=Units.WATT_PER_CUBIC_METER
)

# ---------------------------------------------------------------------------
# Edge model scalar outputs.
# ---------------------------------------------------------------------------
Q_PARALLEL = OutputKey("q_parallel", units=Units.WATT_PER_SQUARE_METER)
Q_PERPENDICULAR_TARGET = OutputKey(
    "q_perpendicular_target", units=Units.WATT_PER_SQUARE_METER
)
T_E_SEPARATRIX = OutputKey("T_e_separatrix", units=Units.KEV)
T_E_TARGET = OutputKey("T_e_target", units=Units.EV)
PRESSURE_NEUTRAL_DIVERTOR = OutputKey(
    "pressure_neutral_divertor", units=Units.PASCAL
)
ALPHA_T = OutputKey("alpha_t", units=Units.DIMENSIONLESS)
Z_EFF_SEPARATRIX = OutputKey("Z_eff_separatrix", units=Units.DIMENSIONLESS)
MULTIPLE_ROOTS_FOUND = OutputKey(
    "multiple_roots_found", units=Units.NOT_APPLICABLE
)

# ---------------------------------------------------------------------------
# Numerics.
# ---------------------------------------------------------------------------
SIM_STATUS = OutputKey("sim_status", units=Units.NOT_APPLICABLE)
SIM_ERROR = OutputKey("sim_error", units=Units.NOT_APPLICABLE)
OUTER_SOLVER_ITERATIONS = OutputKey(
    "outer_solver_iterations", units=Units.NOT_APPLICABLE
)
INNER_SOLVER_ITERATIONS = OutputKey(
    "inner_solver_iterations", units=Units.NOT_APPLICABLE
)
# Boolean array indicating whether the state corresponds to a
# post-sawtooth-crash state.
SAWTOOTH_CRASH = OutputKey("sawtooth_crash", units=Units.NOT_APPLICABLE)

# ---------------------------------------------------------------------------
# ToraxConfig.
# ---------------------------------------------------------------------------
CONFIG = OutputKey("config", units=Units.NOT_APPLICABLE)

# ---------------------------------------------------------------------------
# Geometry: scalar quantities.
# ---------------------------------------------------------------------------
B_0 = OutputKey("B_0", units=Units.TESLA)
R_MAJOR = OutputKey("R_major", units=Units.METER)
A_MINOR = OutputKey("a_minor", units=Units.METER)
PHI_B = OutputKey("Phi_b", units=Units.WEBER)
PHI_B_DOT = OutputKey("Phi_b_dot", units=Units.WEBER_PER_SECOND)
RHO_B = OutputKey("rho_b", units=Units.METER)
DRHO = OutputKey("drho", units=Units.METER)
DRHO_NORM = OutputKey("drho_norm", units=Units.DIMENSIONLESS)

# ---------------------------------------------------------------------------
# Geometry: profile quantities.
# ---------------------------------------------------------------------------
PHI = OutputKey("Phi", units=Units.WEBER)
TFF = OutputKey("F", units=Units.TESLA_METER)
R_IN = OutputKey("R_in", units=Units.METER)
R_OUT = OutputKey("R_out", units=Units.METER)
R_MAJOR_PROFILE = OutputKey("R_major_profile", units=Units.METER)
R_MID = OutputKey("r_mid", units=Units.METER)
AREA = OutputKey("area", units=Units.SQUARE_METER)
VOLUME = OutputKey("volume", units=Units.CUBIC_METER)
VPR = OutputKey("vpr", units=Units.CUBIC_METER)
SPR = OutputKey("spr", units=Units.SQUARE_METER)
DELTA = OutputKey("delta", units=Units.DIMENSIONLESS)
DELTA_UPPER = OutputKey("delta_upper", units=Units.DIMENSIONLESS)
DELTA_LOWER = OutputKey("delta_lower", units=Units.DIMENSIONLESS)
ELONGATION = OutputKey("elongation", units=Units.DIMENSIONLESS)
EPSILON = OutputKey("epsilon", units=Units.DIMENSIONLESS)
G0 = OutputKey("g0", units=Units.SQUARE_METER)
G0_OVER_VPR = OutputKey("g0_over_vpr", units=Units.INVERSE_METER)
G1 = OutputKey("g1", units=Units.QUARTIC_METER)
G1_OVER_VPR = OutputKey("g1_over_vpr", units=Units.METER)
G1_OVER_VPR2 = OutputKey("g1_over_vpr2", units=Units.INVERSE_SQUARE_METER)
G2 = OutputKey("g2", units=Units.SQUARE_METER)
G2G3_OVER_RHON = OutputKey("g2g3_over_rhon", units=Units.DIMENSIONLESS)
G3 = OutputKey("g3", units=Units.INVERSE_SQUARE_METER)
GM4 = OutputKey("gm4", units=Units.INVERSE_SQUARE_TESLA)
GM5 = OutputKey("gm5", units=Units.SQUARE_TESLA)
GM9 = OutputKey("gm9", units=Units.INVERSE_METER)

# ---------------------------------------------------------------------------
# Geometry output renames.
# ---------------------------------------------------------------------------
IP_PROFILE_FROM_GEO = OutputKey("Ip_profile_from_geo", units=Units.AMPERE)
PSI_FROM_GEO = OutputKey("psi_from_geo", units=Units.WEBER)
Z_MAGNETIC_AXIS = OutputKey("z_magnetic_axis", units=Units.METER)

# ---------------------------------------------------------------------------
# Solver / edge numerics output keys.
# ---------------------------------------------------------------------------
SOLVER_PHYSICS_OUTCOME = OutputKey(
    "solver_physics_outcome", units=Units.NOT_APPLICABLE
)
SOLVER_ITERATIONS = OutputKey("solver_iterations", units=Units.NOT_APPLICABLE)
SOLVER_RESIDUAL = OutputKey("solver_residual", units=Units.NOT_APPLICABLE)
SOLVER_ERROR = OutputKey("solver_error", units=Units.NOT_APPLICABLE)
FIXED_POINT_OUTCOME = OutputKey(
    "fixed_point_outcome", units=Units.NOT_APPLICABLE
)
ROOTS = OutputKey("roots", units=Units.NOT_APPLICABLE)
N_ROOTS = OutputKey("n_roots", units=Units.NOT_APPLICABLE)

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
