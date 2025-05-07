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

"""Tests Bremsstrahlung power model with time-dependent Zimp and Zeff. CHEASE.

Ip from parameters. implicit, T_i+T_e, Pei low dens, no pedestal, constant chi
"""


CONFIG = {
    'plasma_composition': {
        'Zeff': {0: {0: 3.0, 1: 3.0}, 1: {0: 5.0, 1: 5.0}},
        'Zimp_override': {0: 10.0, 1: 30.0},
    },
    'profile_conditions': {
        'n_e_nbar_is_fGW': True,
        'n_e_right_bc': 0.5,
        'nbar': 0.95,  # initial density (Greenwald fraction units)
    },
    'numerics': {
        'resistivity_multiplier': 100,  # to shorten current diffusion time
        't_final': 1,
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
    },
    'sources': {
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
        },
        'generic_heat': {
            'w': 0.18202270915319393,
        },
        'ei_exchange': {},
        'ohmic': {},
        'generic_current': {},
        'bremsstrahlung': {
            'use_relativistic_correction': True,
        },
    },
    'pedestal': {
        'pedestal_model': 'set_T_ped_n_ped',
        'set_pedestal': True,
    },
    'transport': {
        'transport_model': 'constant',
        'chi_i': 0.5,
        'chi_e': 0.5,
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
