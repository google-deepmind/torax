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

"""test_implicit: implicit, T_i+T_e, no Pei, no pedestal, constant chi."""


CONFIG = {
    'profile_conditions': {
        'n_e_right_bc': 0.5e20,
    },
    'numerics': {'t_final': 1, 'chi_timestep_prefactor': 9.0},
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        # Current sources (for psi equation)
        'generic_current': {},
        # Electron density sources/sink (for the n_e equation).
        'generic_particle': {},
        'gas_puff': {},
        'pellet': {},
        # Ion and electron heat sources (for the temp-ion and temp-el eqs).
        'generic_heat': {},
        'ei_exchange': {
            # multiplier for ion-electron heat exchange term for sensitivity
            'Qei_multiplier': 0.0,
        },
    },
    'pedestal': {},
    'transport': {},
    'solver': {
        'solver_type': 'linear',
        'use_predictor_corrector': False,
        'theta_implicit': 1.0,
        'use_pereverzev': False,
    },
    'time_step_calculator': {
        'calculator_type': 'chi',
    },
}
