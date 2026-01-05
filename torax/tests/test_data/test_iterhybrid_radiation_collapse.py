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

"""Based on test_iterhybrid_predictor_corrector but with radiation collapse."""

# pylint: disable=invalid-name

import copy

from torax.tests.test_data import test_iterhybrid_predictor_corrector

W_frac = 1e-3
CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

assert isinstance(CONFIG['plasma_composition'], dict)
# increasing density to eventually trigger radiation collapse
CONFIG['profile_conditions']['nbar'] = {0: 0.8, 5.0: 1.2}
CONFIG['plasma_composition']['impurity'] = {
    'Ne': 1 - W_frac,
    'W': W_frac,
}
CONFIG['plasma_composition']['Z_eff'] = 3.0
CONFIG['sources']['impurity_radiation'] = {
    'model_name': 'mavrin_fit',
}
# Remove fusion source to make collapse easier to attain
CONFIG['sources'].pop('fusion')
# Reduce the pedestal temperature to make collapse easier to attain
CONFIG['pedestal']['T_i_ped'] = 1.0
CONFIG['pedestal']['T_e_ped'] = 1.0
# nonlinear solver to enable adaptive dt such that small temperature is reached
# during the collapse.
CONFIG['solver'] = (
    {
        'solver_type': 'newton_raphson',
        'use_predictor_corrector': True,
        'n_corrector_steps': 5,
        'chi_pereverzev': 30,
        'D_pereverzev': 15,
        'use_pereverzev': True,
    }
)
