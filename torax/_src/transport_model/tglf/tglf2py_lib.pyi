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

"""Type stub file for C/Fortran extension tglf2py_lib."""

from typing import Any
import numpy as np

class TGLFInterface:
  tglf_path_in: str
  file_dump_local: str
  tglf_dump_flag_in: bool
  tglf_quiet_flag_in: bool
  tglf_test_flag_in: int
  tglf_units_in: str
  tglf_use_transport_model_in: bool
  tglf_geometry_flag_in: int
  tglf_write_wavefunction_flag_in: int
  tglf_sign_bt_in: float
  tglf_sign_it_in: float
  tglf_theta_trapped_in: float
  tglf_wdia_trapped_in: float
  tglf_park_in: float
  tglf_ghat_in: float
  tglf_gchat_in: float
  tglf_wd_zero_in: float
  tglf_linsker_factor_in: float
  tglf_gradb_factor_in: float
  tglf_filter_in: float
  tglf_damp_psi_in: float
  tglf_damp_sig_in: float
  tglf_iflux_in: bool
  tglf_use_bper_in: bool
  tglf_use_bpar_in: bool
  tglf_use_mhd_rule_in: bool
  tglf_use_bisection_in: bool
  tglf_use_inboard_detrapped_in: bool
  tglf_use_ave_ion_grid_in: bool
  tglf_ibranch_in: int
  tglf_nmodes_in: int
  tglf_nbasis_max_in: int
  tglf_nbasis_min_in: int
  tglf_nxgrid_in: int
  tglf_nky_in: int
  tglf_adiabatic_elec_in: bool
  tglf_alpha_mach_in: float
  tglf_alpha_e_in: float
  tglf_alpha_p_in: float
  tglf_alpha_quench_in: float
  tglf_alpha_zf_in: float
  tglf_xnu_factor_in: float
  tglf_debye_factor_in: float
  tglf_etg_factor_in: float
  tglf_rlnp_cutoff_in: float
  tglf_sat_rule_in: int
  tglf_kygrid_model_in: int
  tglf_xnu_model_in: int
  tglf_vpar_model_in: int
  tglf_vpar_shear_model_in: int
  tglf_ns_in: int
  tglf_mass_in: np.ndarray
  tglf_zs_in: np.ndarray
  tglf_ky_in: float
  tglf_width_in: float
  tglf_width_min_in: float
  tglf_nwidth_in: int
  tglf_find_width_in: bool
  tglf_rlns_in: np.ndarray
  tglf_rlts_in: np.ndarray
  tglf_vpar_shear_in: np.ndarray
  tglf_vexb_shear_in: float
  tglf_vns_shear_in: np.ndarray
  tglf_vts_shear_in: np.ndarray
  tglf_taus_in: np.ndarray
  tglf_as_in: np.ndarray
  tglf_vpar_in: np.ndarray
  tglf_vexb_in: float
  tglf_betae_in: float
  tglf_xnue_in: float
  tglf_zeff_in: float
  tglf_debye_in: float
  tglf_new_eikonal_in: bool
  tglf_rmin_sa_in: float
  tglf_rmaj_sa_in: float
  tglf_q_sa_in: float
  tglf_shat_sa_in: float
  tglf_alpha_sa_in: float
  tglf_xwell_sa_in: float
  tglf_theta0_sa_in: float
  tglf_b_model_sa_in: int
  tglf_ft_model_sa_in: int
  tglf_rmin_loc_in: float
  tglf_rmaj_loc_in: float
  tglf_zmaj_loc_in: float
  tglf_drmindx_loc_in: float
  tglf_drmajdx_loc_in: float
  tglf_dzmajdx_loc_in: float
  tglf_kappa_loc_in: float
  tglf_s_kappa_loc_in: float
  tglf_delta_loc_in: float
  tglf_s_delta_loc_in: float
  tglf_zeta_loc_in: float
  tglf_s_zeta_loc_in: float
  tglf_shape_sin3_loc_in: float
  tglf_shape_s_sin3_loc_in: float
  tglf_shape_sin4_loc_in: float
  tglf_shape_s_sin4_loc_in: float
  tglf_shape_sin5_loc_in: float
  tglf_shape_s_sin5_loc_in: float
  tglf_shape_sin6_loc_in: float
  tglf_shape_s_sin6_loc_in: float
  tglf_shape_cos0_loc_in: float
  tglf_shape_s_cos0_loc_in: float
  tglf_shape_cos1_loc_in: float
  tglf_shape_s_cos1_loc_in: float
  tglf_shape_cos2_loc_in: float
  tglf_shape_s_cos2_loc_in: float
  tglf_shape_cos3_loc_in: float
  tglf_shape_s_cos3_loc_in: float
  tglf_shape_cos4_loc_in: float
  tglf_shape_s_cos4_loc_in: float
  tglf_shape_cos5_loc_in: float
  tglf_shape_s_cos5_loc_in: float
  tglf_shape_cos6_loc_in: float
  tglf_shape_s_cos6_loc_in: float
  tglf_q_loc_in: float
  tglf_q_prime_loc_in: float
  tglf_p_prime_loc_in: float
  tglf_beta_loc_in: float
  tglf_kx0_loc_in: float
  tglf_elec_pflux_out: float | np.ndarray
  tglf_elec_eflux_out: float | np.ndarray
  tglf_elec_eflux_low_out: float | np.ndarray
  tglf_elec_mflux_out: float | np.ndarray
  tglf_elec_expwd_out: float | np.ndarray
  tglf_ion_pflux_out: np.ndarray
  tglf_ion_eflux_out: np.ndarray
  tglf_ion_eflux_low_out: np.ndarray
  tglf_ion_mflux_out: np.ndarray
  tglf_ion_expwd_out: np.ndarray
  tglf_nn_max_error_in: float

  def __getattr__(self, name: str) -> Any: ...
  def __setattr__(self, name: str, value: Any) -> None: ...

tglf_interface: TGLFInterface

def tglf_run() -> None: ...
def tglf_shutdown() -> None: ...
