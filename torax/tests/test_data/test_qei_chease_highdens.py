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

"""Tests ion-electron heat exchange with CHEASE geometry.

Ip from parameters. implicit, Ti+Te, Pei high dens, no pedestal, constant chi
"""

CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'set_pedestal': False,
            'nbar': 1.0,  # Initial density in Greenwald fraction units
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
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
        },
        'generic_ion_el_heat_source': {
            'w': 0.18202270915319393,
        },
        'qei_source': {},
        'jext': {},
    },
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
