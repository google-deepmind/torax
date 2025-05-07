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

"""Tests combined current diffusion and heat transport.

Implicit solver + pereverzev-corrigan, T_i+T_e+Psi, Pei standard dens,
pedestal, chi from qlknn.
"""

CONFIG = {
    'profile_conditions': {
        # initial condition ion temperature for r=0 and r=a_minor
        'T_i': {0.0: {0.0: 8.0, 1.0: 1.0}},
        'T_e': {0.0: {0.0: 8.0, 1.0: 1.0}},
        'n_e_right_bc': 0.5,
        'current_profile_nu': 0,
    },
    'numerics': {
        'evolve_current': True,
        'resistivity_multiplier': 100,  # to shorten current diffusion time
        't_final': 2,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
        },
        'generic_heat': {},
        'ei_exchange': {},
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        'generic_current': {},
    },
    'pedestal': {
        'pedestal_model': 'set_tped_nped',
        'set_pedestal': True,
    },
    'transport': {
        'transport_model': 'qlknn',
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
