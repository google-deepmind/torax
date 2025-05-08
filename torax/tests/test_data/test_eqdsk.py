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

"""Integration test for EQDSK geometry.

Uses demo EQDSK file generated for a small tokamak configuration:
R_major~0.9m , a_minor~0.2m, B_0~1.4T, Ip~170kA.
"""


CONFIG = {
    'plasma_composition': {
        'main_ion': 'D',
        'impurity': 'Ne',
        'Z_eff': 1.6,
    },
    'profile_conditions': {
        'Ip': 0.25,
        'T_i': {0.0: {0.0: 3.0, 1.0: 0.2}},
        'T_i_right_bc': 0.2,
        'T_e': {0.0: {0.0: 3.0, 1.0: 0.2}},
        'T_e_right_bc': 0.2,
        'n_e_right_bc': 0.25,
        'n_e_nbar_is_fGW': True,
        'nbar': 0.5,
        'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
    },
    'numerics': {
        't_final': 0.5,
        'resistivity_multiplier': 1,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_current': True,
        'evolve_density': True,
        'max_dt': 0.5,
        'chi_timestep_prefactor': 500,
        'dt_reduction_factor': 3,
    },
    'geometry': {
        'geometry_type': 'EQDSK',
        # Demonstration eqdsk file
        'geometry_file': 'eqdsk_cocos02.eqdsk',
        'Ip_from_parameters': False,
    },
    'sources': {
        # Set bootstrap multiplier to 0.5 to avoid numerical issues when
        # I_bs > Ip.
        'j_bootstrap': {
            'bootstrap_mult': 0.5,
        },
        'ohmic': {},
        'ei_exchange': {},
        'generic_particle': {
            'S_tot': 2.0e19,
            'deposition_location': 0.3,
            'particle_width': 0.25,
        },
        'generic_heat': {
            'gaussian_location': 0.3,
            'gaussian_width': 0.2,
            'P_total': 3e6,
            'electron_heat_fraction': 0.6,
        },
    },
    'pedestal': {},
    'transport': {
        'transport_model': 'qlknn',
        'apply_inner_patch': True,
        'D_e_inner': 1,
        'V_e_inner': 0.0,
        'chi_i_inner': 5,
        'chi_e_inner': 5,
        'rho_inner': 0.25,
        'apply_outer_patch': True,
        'D_e_outer': 0.1,
        'V_e_outer': 0.0,
        'chi_i_outer': 2.0,
        'chi_e_outer': 2.0,
        'rho_outer': 0.9,
        'chi_min': 0.05,
        'chi_max': 100,
        'D_e_min': 0.05,
        'smoothing_width': 0.1,
        # qlknn params.
        'DV_effective': True,
        'avoid_big_negative_s': True,
        'An_min': 0.05,
        'ITG_flux_ratio_correction': 1,
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': True,
        'n_corrector_steps': 1,
        'chi_pereverzev': 100,
        'D_pereverzev': 50,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
