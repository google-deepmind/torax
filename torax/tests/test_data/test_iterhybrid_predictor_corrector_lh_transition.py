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

"""test_iterhybrid_predictor_corrector but with ADAPTIVE_TRANSPORT pedestal.

This configuration is designed to test the L-mode to H-mode transition.
Low heating for t=0 to 30, ramp to reach high heating at t=60.

Pedestal formation governed by Martin scaling. Pedestal height from user.
"""

import copy

from torax.tests.test_data import test_iterhybrid_predictor_corrector

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

CONFIG['numerics']['t_final'] = 60.0
CONFIG['numerics']['evolve_density'] = False
CONFIG['numerics']['max_dt'] = 5.0

# Change initial conditions to L-mode
CONFIG['profile_conditions']['T_i'] = {
    0.0: {0.0: 3.0, 1.0: 0.25},
}
CONFIG['profile_conditions']['T_e'] = {
    0.0: {0.0: 3.0, 1.0: 0.25},
}
CONFIG['profile_conditions']['n_e'] = {
    0.0: {0.0: 0.5e20, 1.0: 0.3e20},
}
CONFIG['profile_conditions']['n_e_right_bc'] = 0.3e20
CONFIG['profile_conditions']['n_e_right_bc_is_fGW'] = False
CONFIG['profile_conditions']['n_e_nbar_is_fGW'] = False
CONFIG['profile_conditions']['normalize_n_e_to_nbar'] = False

# Low heating for t=0 to 30, ramp to reach high heating at t=60.
CONFIG['sources']['generic_heat']['P_total'] = {
    0.0: 1e6,
    10.0: 1e6,
    40.0: 5e7,
}

CONFIG['pedestal'] = {
    'model_name': 'set_T_ped_n_ped',
    'set_pedestal': True,
    'mode': 'ADAPTIVE_TRANSPORT',
    'T_i_ped': 4.5,
    'T_e_ped': 4.5,
    'n_e_ped': 0.62e20,
    'rho_norm_ped_top': 0.9,
    'saturation_model': {'model_name': 'profile_value'},
    'formation_model': {'model_name': 'martin'},
}

# Use nonlinear solver, as linear solver struggles with the fast dynamics of the
# L-H transition.
CONFIG['solver'] = {
    'solver_type': 'newton_raphson',
    'use_predictor_corrector': True,
    'use_pereverzev': True,
}
