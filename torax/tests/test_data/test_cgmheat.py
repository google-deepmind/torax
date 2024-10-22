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

"""Tests a nonlinear transport model.

Implicit + pereverzev-corrigan, Ti+Te, Pei standard dens, pedestal, chi from
CGM.
"""


CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'ne_bound_right': 0.5,
        },
        'numerics': {
            't_final': 1,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Current sources (for psi equation)
        'jext': {},
        # Electron density sources/sink (for the ne equation).
        'generic_particle_source': {},
        'gas_puff_source': {},
        'pellet_source': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_ion_el_heat_source': {},
        'qei_source': {},
    },
    'transport': {
        'transport_model': 'CGM',
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': False,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
