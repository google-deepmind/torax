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

"""Tests implementation of CHEASE geometry.

Ip from parameters. implicit, T_i+T_e, no Pei, no pedestal, constant chi.
"""


CONFIG = {
    'profile_conditions': {
        'n_e_right_bc': 0.5,
        'Ip': 15,
    },
    'numerics': {
        't_final': 1,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        # Electron density sources/sink (for the n_e equation).
        'generic_particle': {
            # total particle source
            'S_total': 0.0,
        },
        'gas_puff': {
            # total pellet particles/s
            'S_total': 0.0,
        },
        'pellet': {
            # total pellet particles/s (continuous pellet model)
            'S_total': 0.0,
        },
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {
            # Gaussian width in normalized radial coordinate r
            'gaussian_width': 0.18202270915319393,
        },
        'ei_exchange': {
            # multiplier for ion-electron heat exchange term for sensitivity
            'Qei_multiplier': 0.0,
        },
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
