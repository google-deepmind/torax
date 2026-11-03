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

"""Config for testing fixed timestep."""


CONFIG = {
    'profile_conditions': {
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'j',
        'n_e_right_bc': 0.5e20,
    },
    'numerics': {},
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        # Electron density sources/sink (for the n_e equation).
        'generic_particle': {
            'S_total': 1e22,
            'particle_width': 0.25,
            'deposition_location': 0.0,
        },
        'gas_puff': {'S_total': 1e22, 'puff_decay_length': 0.05},
        'pellet': {
            'S_total': 2e22,
            'pellet_width': 0.1,
            'pellet_deposition_location': 0.85,
        },
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {
            'P_total': 120e6,
            'gaussian_width': 0.25,
            'gaussian_location': 0.0,
        },
        'ei_exchange': {},
    },
    'pedestal': {
        'structure_model': {'model_name': 'set_T_ped_n_ped'},
    },
    'transport': {
        'core_transport_models': {
            'qlknn': {
                'model_name': 'qlknn',
            },
        },
        'smoothing_width': 0.1,
    },
    'solver': {
        'solver_type': 'linear',
        'use_fixed_point': False,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
        'dt': 2e-2,
        't_final': 2,
    },
}
