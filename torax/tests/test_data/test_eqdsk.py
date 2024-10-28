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
Rmaj~0.9m , Rmin~0.2m, B0~1.4T, Ip~170kA.
"""


CONFIG = {
    'runtime_params': {
        'plasma_composition': {
            'Ai': 2.0,
            'Zeff': 1.6,
            'Zimp': 10,
        },
        'profile_conditions': {
            'Ip': 0.25,
            'Ti': {0.0: {0.0: 3.0, 1.0: 0.2}},
            'Ti_bound_right': 0.2,
            'Te': {0.0: {0.0: 3.0, 1.0: 0.2}},
            'Te_bound_right': 0.2,
            'ne_bound_right': 0.25,
            'ne_is_fGW': True,
            'nbar': 0.5,
            'ne': {0: {0.0: 1.5, 1.0: 1.0}},
            'set_pedestal': False,
        },
        'numerics': {
            't_final': 0.5,
            'resistivity_mult': 1,
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'current_eq': True,
            'dens_eq': True,
            'maxdt': 0.5,
            'dtmult': 500,
            'dt_reduction_factor': 3,
        },
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
        'ohmic_heat_source': {},
        'qei_source': {},
        'generic_particle_source': {
            'S_tot': 2.0e19,
            'deposition_location': 0.3,
            'particle_width': 0.25,
        },
        'generic_ion_el_heat_source': {
            'rsource': 0.3,
            'w': 0.2,
            'Ptot': 3e6,
            'el_heat_fraction': 0.6,
        },
    },
    'transport': {
        'transport_model': 'qlknn',
        'apply_inner_patch': True,
        'De_inner': 1,
        'Ve_inner': 0.0,
        'chii_inner': 5,
        'chie_inner': 5,
        'rho_inner': 0.25,
        'apply_outer_patch': True,
        'De_outer': 0.1,
        'Ve_outer': 0.0,
        'chii_outer': 2.0,
        'chie_outer': 2.0,
        'rho_outer': 0.9,
        'chimin': 0.05,
        'chimax': 100,
        'Demin': 0.05,
        'smoothing_sigma': 0.1,
        'qlknn_params': {
            'DVeff': True,
            'include_ITG': True,
            'include_TEM': True,
            'include_ETG': True,
            'avoid_big_negative_s': True,
            'An_min': 0.05,
            'ITG_flux_ratio_correction': 1,
        },
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': True,
        'corrector_steps': 1,
        'chi_per': 100,
        'd_per': 50,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
