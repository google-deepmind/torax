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

"""Tests semi-implicit convection as carried out with FiPy.

Semi-implicit convection can lead to numerical instability at boundary
condition. No pedestal, implicit + pereverzev-corrigan, Ti+Te,
Pei standard dens, chi from CGM.
"""

CONFIG = {
    'profile_conditions': {
        'ne_bound_right': 0.5,
    },
    'numerics': {
        't_final': 0.5,
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
        'generic_current': {},
    },
    'pedestal': {},
    'transport': {
        'transport_model': 'CGM',
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
        'convection_dirichlet_mode': 'semi-implicit',
        'convection_neumann_mode': 'semi-implicit',
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
