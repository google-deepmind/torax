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

"""Tests combined current, heat, and particle transport.

Constant transport coefficient model. Pedestal. Particle sources
"""

CONFIG = {
    'runtime_params': {
        'profile_conditions': {
            'set_pedestal': True,
            'ne_bound_right': 0.5,
            'nbar': 0.85,
            # set flat Ohmic current to provide larger range of current
            # evolution for test
            'nu': 0,
        },
        'numerics': {
            'ion_heat_eq': True,
            'el_heat_eq': True,
            'dens_eq': True,
            'current_eq': True,
            'resistivity_mult': 100,  # to shorten current diffusion time
            't_final': 2,
        },
    },
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'generic_ion_el_heat_source': {},
        'qei_source': {},
        'gas_puff_source': {
            'S_puff_tot': 1.0e22,
        },
        'pellet_source': {
            'S_pellet_tot': 2.0e22,
        },
        'j_bootstrap': {},
        'generic_current_source': {},
    },
    'pedestal': {},
    'transport': {
        'transport_model': 'constant',
        # constant params.
        # diffusion coefficient in electron density equation in m^2/s
        'De_const': 0.5,
        # convection coefficient in electron density equation in m^2/s
        'Ve_const': -0.2,
    },
    'stepper': {
        'stepper_type': 'linear',
        'predictor_corrector': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
