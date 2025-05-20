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

"""Tests CHEASE geometry with Ip from config and psi from prescribed total j.

Ip from parameters. implicit, psi (current diffusion) only
"""

CONFIG = {
    'profile_conditions': {
        'n_e_right_bc': 0.5e20,
        'initial_psi_from_j': True,
        'initial_j_is_total_current': True,
        'current_profile_nu': 2,
        'n_e_nbar_is_fGW': True,
        'normalize_n_e_to_nbar': True,
        'nbar': 0.85,
        'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
    },
    'numerics': {
        'evolve_ion_heat': False,
        'evolve_electron_heat': False,
        'evolve_current': True,
        'resistivity_multiplier': 100,  # to shorten current diffusion time
        't_final': 3,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
    },
    'sources': {
        'generic_heat': {
            'gaussian_width': 0.18202270915319393,
        },
        'generic_current': {},
    },
    'pedestal': {},
    'transport': {
        'model_name': 'constant',
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
