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
        'n_e_right_bc': 0.5,
        'initial_psi_from_j': True,
        'initial_j_is_total_current': False,
        'nu': 2,
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
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
        },
        'generic_heat': {
            'w': 0.18202270915319393,
        },
        'generic_current': {},
    },
    'pedestal': {},
    'transport': {
        'transport_model': 'constant',
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
