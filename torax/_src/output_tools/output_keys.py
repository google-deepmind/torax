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

# Calculated or derived current densities (excluding sources)
J_PARALLEL_TOTAL = "j_parallel_total"
J_PARALLEL_OHMIC = "j_parallel_ohmic"
J_PARALLEL_EXTERNAL = "j_parallel_external"
J_PARALLEL_BOOTSTRAP = "j_parallel_bootstrap"
J_TOROIDAL_TOTAL = "j_total"
J_TOROIDAL_OHMIC = "j_ohmic"
J_TOROIDAL_EXTERNAL = "j_external"
J_TOROIDAL_BOOTSTRAP = "j_bootstrap"
I_BOOTSTRAP = "I_bootstrap"

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

# Post processed outputs
Q_FUSION = "Q_fusion"

# Edge model outputs
SEED_IMPURITY_CONCENTRATIONS = "seed_impurity_concentrations"
CALCULATED_ENRICHMENT = "calculated_enrichment"
IMPURITY = "impurity"
SEED_IMPURITY = "seed_impurity"
MAIN_ION = "main_ion"

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

