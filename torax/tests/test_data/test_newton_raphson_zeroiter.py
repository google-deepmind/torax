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

"""Newton-Raphson zero iteration test. Linear solver initial condition.

Tests that Newton-Raphson method behaves the same as linear method
when Newton-Raphson is set to use zero iterations and use linear
method for the initial guess.

implicit + pereverzev-corrigan, Ti+Te+Psi, Pei standard dens, pedestal, chi from
qlknn
Used for a nonlinear run with zero iterations. dt adaptation is turned off to
avoid backtracking
to tiny timesteps (and test timeouts)
"""

CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            # initial condition ion temperature for r=0 and r=Rmin
            'Ti': {0.0: {0.0: 8.0, 1.0: 1.0}},
            'Te': {0.0: {0.0: 8.0, 1.0: 1.0}},
            'set_pedestal': True,
            # set flat Ohmic current to provide larger range of current
            # evolution for test
            'nu': 0,
        },
        'numerics': {
            'current_eq': True,
            'resistivity_mult': 100,  # to shorten current diffusion time
            't_final': 2,
            'adaptive_dt': False,
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
        'nbi_particle_source': {},
        'gas_puff_source': {},
        'pellet_source': {},
        'jext': {},
    },
    'transport': {
        'transport_model': 'qlknn',
        'qlknn_params': {
            'DVeff': False,
        },
    },
    'stepper': {
        'stepper_type': 'newton_raphson',
        'predictor_corrector': False,
        'use_pereverzev': True,
        'maxiter': 0,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
