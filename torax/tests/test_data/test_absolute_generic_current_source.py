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

"""Tests combined current diffusion and heat transport with use_absolute_current.

Implicit solver + pereverzev-corrigan, Ti+Te+Psi, Pei standard dens,
pedestal, chi from qlknn.

Same as test_psi_and_heat but with fext=0 but use_absolute_current and Iext=3.
Result should be the same as test_psi_and_heat since fext=0 is ignored.
"""


CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            # initial condition ion temperature for r=0 and r=Rmin
            'Ti': {0.0: {0.0: 8.0, 1.0: 1.0}},
            'Te': {0.0: {0.0: 8.0, 1.0: 1.0}},
            'ne_bound_right': 0.5,
            # set flat Ohmic current to provide larger range of current
            # evolution for test
            'nu': 0,
        },
        'numerics': {
            'current_eq': True,
            'resistivity_mult': 100,  # to shorten current diffusion time
            't_final': 2,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'generic_current_source': {
            'use_absolute_current': True,
            'fext': 0.0,
            'Iext': 3.0,
        },
        'generic_particle_source': {},
        'gas_puff_source': {},
        'pellet_source': {},
        'generic_ion_el_heat_source': {},
        'qei_source': {},
    },
    'pedestal': {},
    'transport': {
        'transport_model': 'qlknn',
    },
    'stepper': {
        'stepper_type': 'linear',
        'use_pereverzev': True,
        'predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
