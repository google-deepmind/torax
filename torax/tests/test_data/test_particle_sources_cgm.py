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

"""Tests particle transport when using CGM model.

CGM transport model. Pedestal. Particle sources
"""


CONFIG = {
    'profile_conditions': {
        'nbar': 0.85,
        'n_e_right_bc': 0.2e20,
        # set flat Ohmic current to provide larger range of current
        # evolution for test
        'current_profile_nu': 0,
    },
    'numerics': {
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_density': True,
        'evolve_current': True,
        'resistivity_multiplier': 100,  # to shorten current diffusion time
        't_final': 2,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'generic_heat': {},
        'ei_exchange': {},
        'generic_particle': {
            'S_total': 0.3e22,
        },
        'gas_puff': {
            'S_total': 0.5e22,
        },
        'pellet': {
            'S_total': 1.0e22,
        },
        'generic_current': {},
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'pedestal': {
        'pedestal_model': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'n_e_ped': 1.0e20,
    },
    'transport': {
        'transport_model': 'CGM',
        # cgm params.
        'chi_D_ratio': 8,
        'VR_D_ratio': {0: -3.0, 1: 0.0},
    },
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
        'use_pereverzev': True,
        'D_pereverzev': 0.0,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
