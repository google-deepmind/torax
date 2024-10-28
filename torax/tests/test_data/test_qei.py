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

"""Tests ion-electron heat exchange model.

Implicit, Ti+Te, Pei low dens, no pedestal, constant chi.
"""

CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'ne_bound_right': 0.5,
            'set_pedestal': False,
            'nbar': 0.5,  # Initial density in Greenwald fraction units
        },
        'numerics': {
            't_final': 1,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
        },
        'generic_ion_el_heat_source': {},
        'qei_source': {},
        'generic_particle_source': {},
        'gas_puff_source': {},
        'pellet_source': {},
        'generic_current_source': {},
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
