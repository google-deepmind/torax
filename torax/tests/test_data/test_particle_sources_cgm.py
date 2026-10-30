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
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'j',
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
        'current_diffusion_time_reduction_factor': 100,
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
            'puff_decay_length': 0.05,
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
        'structure_model': {
            'model_name': 'set_T_ped_n_ped',
            'n_e_ped': 1.0e20,
        },
    },
    'transport': {
        'core_transport_models': {
            'cgm': {
                'model_name': 'cgm',
                'chi_D_ratio': 8,
                'VR_D_ratio': {0: -3.0, 1: 0.0},
            },
        },
    },
    'solver': {
        'solver_type': 'linear',
        'use_fixed_point': False,
        'use_pereverzev': True,
        'D_pereverzev': 0.0,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
        't_final': 2,
    },
}
