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
condition. No pedestal, implicit + pereverzev-corrigan, T_i+T_e,
Pei standard dens, chi from CGM.
"""

CONFIG = {
    'profile_conditions': {
        'n_e_right_bc': 0.5e20,
    },
    'numerics': {
        't_final': 0.5,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'generic_heat': {},
        'ei_exchange': {},
        'generic_current': {},
    },
    'pedestal': {},
    'transport': {
        'model_name': 'CGM',
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
        'convection_dirichlet_mode': 'semi-implicit',
        'convection_neumann_mode': 'semi-implicit',
        'use_pereverzev': True,
        'chi_pereverzev': 20.0,
        'D_pereverzev': 10.0,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
