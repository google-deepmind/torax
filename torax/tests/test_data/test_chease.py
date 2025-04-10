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

Ip from parameters. implicit, Ti+Te, no Pei, no pedestal, constant chi.
"""


CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'ne_bound_right': 0.5,
            'Ip_tot': 15,
        },
        'numerics': {
            't_final': 1,
        },
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
        'Ip_from_parameters': True,
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current_source': {},
        # Electron density sources/sink (for the ne equation).
        'generic_particle_source': {
            # total particle source
            'S_tot': 0.0,
        },
        'gas_puff_source': {
            # total pellet particles/s
            'S_puff_tot': 0.0,
        },
        'pellet_source': {
            # total pellet particles/s (continuous pellet model)
            'S_pellet_tot': 0.0,
        },
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_ion_el_heat_source': {
            # Gaussian width in normalized radial coordinate r
            'w': 0.18202270915319393,
        },
        'qei_source': {
            # multiplier for ion-electron heat exchange term for sensitivity
            'Qei_mult': 0.0,
        },
    },
    'pedestal': {},
    'transport': {
        'transport_model': 'constant',
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
