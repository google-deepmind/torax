# Copyright 2024 DeepMind Technologies Limited
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

"""STEP SPP-001 Power Plant Scenario.

This is a *fully non-inductive* flat-top (steady state) scenario for the STEP
tokamak. Key points of interest:
- Zero loop voltage boundary condition on current equation, signifying no use of
the central solenoid. Relies on high bootstrap fraction, thanks to high beta.
- Bohm-gyrobohm transport tuned to give the desired H98.
- Pellet fuelling tuned to give the desired Greenwald fraction.
- Loading profiles, sources, and geometry from IMAS.

Based on:
 1. T.A. Brown, F.J. Casson et al., "OpenSTEP: public data release of the STEP
  Prototype Powerplant scenario SPP-001" (2025). United Kingdom Atomic Energy
  Authority. DOI: 10.14468/07jt-s540. URL: https://github.com/ukaea/OpenSTEP.
 2. Tholerus, E., et al. "Flat-top plasma operational space of the STEP power
  plant." Nuclear Fusion 64.10 (2024): 106030. DOI: 10.1088/1741-4326/ad6ea2.
"""

import imas
import numpy as np
from torax._src import path_utils
from torax._src.imas_tools.input import core_profiles
from torax._src.imas_tools.input import loader

# Load IDSs
path = (
    path_utils.torax_path() / "data" / "imas_data" / "STEP_SPP_001_ECHD_ftop.nc"
)
equilibrium_ids = loader.load_imas_data(str(path), "equilibrium")
core_profiles_ids = loader.load_imas_data(str(path), "core_profiles")
core_sources_ids = loader.load_imas_data(str(path), "core_sources")

# Convert profile_conditions to TORAX dict
core_profiles_xr = imas.util.to_xarray(core_profiles_ids)
profile_conditions_from_ids = core_profiles.profile_conditions_from_IMAS(
    core_profiles_ids,
)
# Replace Ip from profile conditions with Ip from equilibrium
# TODO(b/323504363): can this be handled within the IMAS loader? e.g. if Ip
# is not present in profile_conditions, pull from equilibrium automatically.
equilibrium_xr = imas.util.to_xarray(equilibrium_ids)
profile_conditions_from_ids["Ip"] = equilibrium_xr[
    "time_slice.global_quantities.ip"
][0].item()

# Extract the source information
# As this is a steady-state scenario, we only need the first time slice
core_sources_xr = imas.util.to_xarray(core_sources_ids)
ec_idx = np.where(core_sources_xr["source.identifier.name"] == "ec")[0].item()
rho_norm_ec = core_sources_xr["source.profiles_1d.grid.rho_tor_norm"][ec_idx][0]
electron_heating_ec = core_sources_xr["source.profiles_1d.electrons.energy"][
    ec_idx
][0]

# Set BgB multiplier to achieve desired confinement
# Lower -> better confinement
bgb_multiplier = 0.23


CONFIG = {
    # TODO(b/323504363): Switch to loading from IMAS
    # https://github.com/google-deepmind/torax/pull/1619
    "plasma_composition": {
        "main_ion": {"D": 0.5, "T": 0.5},
        "impurity": "Xe",
        "Z_eff": core_profiles_xr["profiles_1d.zeff"][0][0].values.item(),
    },
    "profile_conditions": profile_conditions_from_ids | {
        "use_v_loop_lcfs_boundary_condition": True,
        "v_loop_lcfs": 0.0,
    },
    "geometry": {
        # TODO(b/323504363): Switch to loading from IMAS rather than eqdsk
        # Currently there is a bug in either the TORAX IMAS loader or the JETTO
        # IMAS writer that makes them incompatible
        "geometry_type": "EQDSK",
        "geometry_file": "STEP_SPP_001_ECHD_ftop.eqdsk",
        "cocos": 11,
    },
    "pedestal": {
        "model_name": "set_T_ped_n_ped",
        "set_pedestal": True,
        "rho_norm_ped_top": 0.95,
        "T_i_ped": 4.0,  # [keV]
        "T_e_ped": 5.0,  # [keV]
        "n_e_ped": 6e19,  # [m^-3]
    },
    "sources": {
        # Physics-based sources
        "ohmic": {},
        "fusion": {},
        "ei_exchange": {},
        "bremsstrahlung": {},
        "impurity_radiation": {
            "model_name": "P_in_scaled_flat_profile",
            "fraction_P_heating": 0.7,
        },
        # Actuators
        "ecrh": {
            "extra_prescribed_power_density": (
                rho_norm_ec.values,
                np.clip(electron_heating_ec.values, a_min=0, a_max=None),
            ),
            "current_drive_efficiency": 0.14,
        },
        "pellet": {
            # TODO(b/323504363): load from IDS?
            "pellet_deposition_location": 0.8,  # from [2] sec 3.4
            "pellet_width": 0.17,  # from [2] sec 3.4
            "S_total": 40e20,  # [s^-1], manually tuned to get desired fGW
        },
    },
    "transport": {
        "model_name": "bohm-gyrobohm",
        # BgB settings from [2] sec 3.3
        # Tuning factor to achieve desired confinement
        "chi_e_bohm_multiplier": bgb_multiplier,
        "chi_i_bohm_multiplier": bgb_multiplier,
        "chi_e_gyrobohm_multiplier": bgb_multiplier,
        "chi_i_gyrobohm_multiplier": bgb_multiplier,
        # Base coefficients
        "chi_e_bohm_coeff": 0.01 * 2e-4,
        "chi_e_gyrobohm_coeff": 50 * 5e-6,
        "chi_i_bohm_coeff": 0.001 * 2e-4,
        "chi_i_gyrobohm_coeff": 1.0 * 5e-6,
        "D_face_c1": 1,
        "D_face_c2": 0.3,
        "V_face_coeff": -0.1,
        # Clipping
        "chi_min": 0.15,
        "chi_max": 100.0,
        "D_e_min": 1e-3,
        "D_e_max": 100.0,
        "V_e_min": -50.0,
        "V_e_max": 50.0,
        # Patching
        # Replaces neoclassical in the core, pending potato orbit correction
        # https://github.com/google-deepmind/torax/issues/1406
        "apply_inner_patch": True,
        "rho_inner": 0.05,
        "chi_e_inner": 1.0,
        "chi_i_inner": 15.0,
        # Smoothing
        "smooth_everywhere": True,
        "smoothing_width": 0.05,
    },
    "neoclassical": {
        "bootstrap_current": {"model_name": "sauter"},
        "transport": {"model_name": "angioni_sauter"},
    },
    "numerics": {
        "t_initial": 0.0,
        "t_final": 500.0,
        "fixed_dt": 50.0,
        "min_dt": 1e-4,
        "dt_reduction_factor": 2.0,
        "resistivity_multiplier": 10.0,
        "evolve_current": True,
        "evolve_ion_heat": True,
        "evolve_electron_heat": True,
        "evolve_density": True,
    },
    "solver": {
        "solver_type": "newton_raphson",
        "use_pereverzev": True,
    },
    "time_step_calculator": {
        "calculator_type": "fixed",
    },
}
