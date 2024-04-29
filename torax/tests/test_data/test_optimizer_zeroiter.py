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

"""Optimizer zero iteration test. Linear solver initial condition.

Tests that we get the same result as the linear solver, when we use
a nonlinear solver but then hack it to act like a linear solver by
using 0 iterations and an initial guess based on the linear solver.
"""

CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'Ti_bound_left': 8,
            'Te_bound_left': 8,
            'set_pedestal': True,
            # set flat Ohmic current to provide larger range of current
            # evolution for test
            'nu': 0,
        },
        'numerics': {
            'current_eq': True,
            'resistivity_mult': 100,  # to shorten current diffusion time
            'adaptive_dt': False,
            't_final': 2,
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
        'stepper_type': 'optimizer',
        'initial_guess_mode': 'linear',
        'predictor_corrector': False,
        'use_pereverzev': True,
        'maxiter': 0,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
