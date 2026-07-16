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

"""String keys and dimension names for the TORAX xarray output DataTree."""

# Dataset names.
PROFILES = "profiles"
SCALARS = "scalars"
NUMERICS = "numerics"
EDGE = "edge"

# Core profiles.
T_E = "T_e"
T_I = "T_i"
PSI = "psi"
V_LOOP = "v_loop"
N_E = "n_e"
N_I = "n_i"
Q = "q"
MAGNETIC_SHEAR = "magnetic_shear"
N_IMPURITY = "n_impurity"
Z_IMPURITY = "Z_impurity"
Z_EFF = "Z_eff"
SIGMA_PARALLEL = "sigma_parallel"
V_LOOP_LCFS = "v_loop_lcfs"
IP_PROFILE = "Ip_profile"
IP = "Ip"

# Calculated or derived current densities (excluding sources).
J_PARALLEL_TOTAL = "j_parallel_total"
J_PARALLEL_OHMIC = "j_parallel_ohmic"
J_PARALLEL_EXTERNAL = "j_parallel_external"
J_PARALLEL_BOOTSTRAP = "j_parallel_bootstrap"
J_TOROIDAL_TOTAL = "j_total"
J_TOROIDAL_OHMIC = "j_ohmic"
J_TOROIDAL_EXTERNAL = "j_external"
J_TOROIDAL_BOOTSTRAP = "j_bootstrap"
I_BOOTSTRAP = "I_bootstrap"

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

# Core transport.
CHI_TURB_I = "chi_turb_i"
CHI_TURB_E = "chi_turb_e"
CHI_ITG_E = "chi_itg_e"
CHI_TEM_E = "chi_tem_e"
CHI_ETG_E = "chi_etg_e"
CHI_ITG_I = "chi_itg_i"
CHI_TEM_I = "chi_tem_i"
D_ITG_E = "D_itg_e"
D_TEM_E = "D_tem_e"
D_TURB_E = "D_turb_e"
V_ITG_E = "V_itg_e"
V_TEM_E = "V_tem_e"
V_TURB_E = "V_turb_e"
CHI_NEO_I = "chi_neo_i"
CHI_NEO_E = "chi_neo_e"
D_NEO_E = "D_neo_e"
V_NEO_E = "V_neo_e"
V_NEO_WARE_E = "V_neo_ware_e"
CHI_BOHM_E = "chi_bohm_e"
CHI_GYROBOHM_E = "chi_gyrobohm_e"
CHI_BOHM_I = "chi_bohm_i"
CHI_GYROBOHM_I = "chi_gyrobohm_i"

# Coordinates.
RHO_FACE_NORM = "rho_face_norm"
RHO_CELL_NORM = "rho_cell_norm"
RHO_NORM = "rho_norm"
RHO_FACE = "rho_face"
RHO_CELL = "rho_cell"
TIME = "time"

# Post-processed outputs: profiles.
PPRIME = "pprime"
FFPRIME = "FFprime"
PSI_NORM = "psi_norm"
J_GENERIC_CURRENT = "j_generic_current"
J_PARALLEL_GENERIC_CURRENT = "j_parallel_generic_current"
J_ECRH = "j_ecrh"
J_PARALLEL_ECRH = "j_parallel_ecrh"
J_NON_INDUCTIVE = "j_non_inductive"
J_PARALLEL_NON_INDUCTIVE = "j_parallel_non_inductive"
POLOIDAL_VELOCITY = "poloidal_velocity"
RADIAL_ELECTRIC_FIELD = "radial_electric_field"

# Post-processed outputs: integrated powers.
P_HEAT_I = "P_heat_i"
P_HEAT_E = "P_heat_e"
P_HEAT_TOTAL = "P_heat_total"
P_SOL_I = "P_SOL_i"
P_SOL_E = "P_SOL_e"
P_SOL_TOTAL = "P_SOL_total"
P_AUX_I = "P_aux_i"
P_AUX_E = "P_aux_e"
P_AUX_TOTAL = "P_aux_total"
P_EXTERNAL_INJECTED = "P_external_injected"
P_EXTERNAL_TOTAL = "P_external_total"
P_EI_EXCHANGE_I = "P_ei_exchange_i"
P_EI_EXCHANGE_E = "P_ei_exchange_e"
P_AUX_GENERIC_I = "P_aux_generic_i"
P_AUX_GENERIC_E = "P_aux_generic_e"
P_AUX_GENERIC_TOTAL = "P_aux_generic_total"
P_ALPHA_I = "P_alpha_i"
P_ALPHA_E = "P_alpha_e"
P_ALPHA_TOTAL = "P_alpha_total"
P_OHMIC_E = "P_ohmic_e"
P_BREMSSTRAHLUNG_E = "P_bremsstrahlung_e"
P_CYCLOTRON_E = "P_cyclotron_e"
P_ECRH_E = "P_ecrh_e"
P_RADIATION_E = "P_radiation_e"
P_FUSION = "P_fusion"
P_ICRH_E = "P_icrh_e"
P_ICRH_I = "P_icrh_i"
P_ICRH_TOTAL = "P_icrh_total"

# Post-processed outputs: L-H transition thresholds.
P_LH_HIGH_DENSITY = "P_LH_high_density"
P_LH_MIN = "P_LH_min"
P_LH_LOW_DENSITY = "P_LH_low_density"
P_LH = "P_LH"
N_E_MIN_P_LH = "n_e_min_P_LH"
P_LH_DELABIE_HIGH_DENSITY = "P_LH_delabie_high_density"
P_LH_DELABIE_MIN = "P_LH_delabie_min"
P_LH_DELABIE_LOW_DENSITY = "P_LH_delabie_low_density"
P_LH_DELABIE = "P_LH_delabie"

# Post-processed outputs: integrated energies.
E_FUSION = "E_fusion"
E_AUX_TOTAL = "E_aux_total"
E_OHMIC_E = "E_ohmic_e"
E_EXTERNAL_INJECTED = "E_external_injected"
E_EXTERNAL_TOTAL = "E_external_total"

# Post-processed outputs: stored energy and confinement.
W_THERMAL_I = "W_thermal_i"
W_THERMAL_E = "W_thermal_e"
W_THERMAL_TOTAL = "W_thermal_total"
TAU_E = "tau_E"
H89P = "H89P"
H98 = "H98"
H97L = "H97L"
H20 = "H20"
W_POL = "W_pol"
LI3 = "li3"
DW_THERMAL_DT = "dW_thermal_dt"
DW_THERMAL_DT_SMOOTHED = "dW_thermal_dt_smoothed"
DW_THERMAL_I_DT_SMOOTHED = "dW_thermal_i_dt_smoothed"
DW_THERMAL_E_DT_SMOOTHED = "dW_thermal_e_dt_smoothed"

# Post-processed outputs: volume/line averages.
T_E_VOLUME_AVG = "T_e_volume_avg"
T_I_VOLUME_AVG = "T_i_volume_avg"
N_E_VOLUME_AVG = "n_e_volume_avg"
N_I_VOLUME_AVG = "n_i_volume_avg"
N_E_LINE_AVG = "n_e_line_avg"
N_I_LINE_AVG = "n_i_line_avg"
FGW_N_E_VOLUME_AVG = "fgw_n_e_volume_avg"
FGW_N_E_LINE_AVG = "fgw_n_e_line_avg"

# Post-processed outputs: q-profile derived scalars.
Q_FUSION = "Q_fusion"
Q95 = "q95"
Q_MIN = "q_min"
RHO_Q_MIN = "rho_q_min"
RHO_Q_3_2_FIRST = "rho_q_3_2_first"
RHO_Q_2_1_FIRST = "rho_q_2_1_first"
RHO_Q_3_1_FIRST = "rho_q_3_1_first"
RHO_Q_3_2_SECOND = "rho_q_3_2_second"
RHO_Q_2_1_SECOND = "rho_q_2_1_second"
RHO_Q_3_1_SECOND = "rho_q_3_1_second"

# Post-processed outputs: integrated currents and fractions.
I_EXTERNAL = "I_external"
I_ECRH = "I_ecrh"
I_AUX_GENERIC = "I_aux_generic"
I_NON_INDUCTIVE = "I_non_inductive"
F_NON_INDUCTIVE = "f_non_inductive"
F_BOOTSTRAP = "f_bootstrap"

# Post-processed outputs: integrated particle sources.
S_GAS_PUFF = "S_gas_puff"
S_PELLET = "S_pellet"
S_GENERIC_PARTICLE = "S_generic_particle"
S_TOTAL = "S_total"

# Post-processed outputs: plasma beta.
BETA_TOR = "beta_tor"
BETA_POL = "beta_pol"
BETA_N = "beta_N"

# Edge model outputs.
SEED_IMPURITY_CONCENTRATIONS = "seed_impurity_concentrations"
CALCULATED_ENRICHMENT = "calculated_enrichment"
IMPURITY = "impurity"
SEED_IMPURITY = "seed_impurity"
MAIN_ION = "main_ion"
MAIN_ION_FRACTIONS = "main_ion_fractions"

# Edge model scalar outputs.
Q_PARALLEL = "q_parallel"
Q_PERPENDICULAR_TARGET = "q_perpendicular_target"
T_E_SEPARATRIX = "T_e_separatrix"
T_E_TARGET = "T_e_target"
PRESSURE_NEUTRAL_DIVERTOR = "pressure_neutral_divertor"
ALPHA_T = "alpha_t"
Z_EFF_SEPARATRIX = "Z_eff_separatrix"
MULTIPLE_ROOTS_FOUND = "multiple_roots_found"


# Numerics.
SIM_STATUS = "sim_status"
SIM_ERROR = "sim_error"
OUTER_SOLVER_ITERATIONS = "outer_solver_iterations"
INNER_SOLVER_ITERATIONS = "inner_solver_iterations"
# Boolean array indicating whether the state corresponds to a
# post-sawtooth-crash state.
SAWTOOTH_CRASH = "sawtooth_crash"

# ToraxConfig.
CONFIG = "config"

# Geometry output renames.
IP_PROFILE_FROM_GEO = "Ip_profile_from_geo"
PSI_FROM_GEO = "psi_from_geo"
Z_MAGNETIC_AXIS = "z_magnetic_axis"

# Solver / edge numerics output keys.
SOLVER_PHYSICS_OUTCOME = "solver_physics_outcome"
SOLVER_ITERATIONS = "solver_iterations"
SOLVER_RESIDUAL = "solver_residual"
SOLVER_ERROR = "solver_error"
FIXED_POINT_OUTCOME = "fixed_point_outcome"
ROOTS = "roots"
N_ROOTS = "n_roots"
