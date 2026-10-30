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

"""Tests plasma current obtained from CHEASE geometry file.

* I_p from CHEASE
* implicit
* psi (current diffusion) only
* v_loop boundary condition
"""

CONFIG = {
    'profile_conditions': {
        'I_p_source': 'geometry',
        'initial_psi_source': 'geometry',
        'n_e_right_bc': 0.5e20,
        'use_v_loop_lcfs_boundary_condition': True,
        # Physical loop voltage; psi BC is accelerated by
        # current_diffusion_time_reduction_factor.
        'v_loop_lcfs': 0.075,
        'n_e_bar_is_f_gw': True,
        'normalize_n_e': True,
        'n_e_bar': 0.85,
        'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
    },
    'numerics': {
        'evolve_ion_heat': False,
        'evolve_electron_heat': False,
        'evolve_current': True,
        'current_diffusion_time_reduction_factor': 100,
    },
    'time_step_calculator': {
        't_final': 3,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    },
    'sources': {
        'generic_heat': {
            'gaussian_width': 0.18202270915319393,
        },
        'generic_current': {},
    },
    'pedestal': {},
    'transport': {
        'core_transport_models': {
            'prescribed': {
                'model_name': 'prescribed',
            },
        },
    },
    'solver': {
        'solver_type': 'linear',
        'use_fixed_point': False,
    },
}
