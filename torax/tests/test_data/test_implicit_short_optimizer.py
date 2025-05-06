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

"""test_implicit_short_optimizer: basic test of implicit solver."""

CONFIG = {
    'profile_conditions': {
        'ne_bound_right': 0.5,
    },
    'numerics': {
        't_final': 0.1,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        # Electron density sources/sink (for the ne equation).
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {},
        'ei_exchange': {
            # multiplier for ion-electron heat exchange term for sensitivity
            'Qei_mult': 0.0,
        },
    },
    'pedestal': {},
    'transport': {
        'transport_model': 'constant',
    },
    'solver': {
        'solver_type': 'optimizer',
        'use_predictor_corrector': False,
        'theta_implicit': 1.0,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
