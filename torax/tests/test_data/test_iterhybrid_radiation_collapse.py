# Copyright 2026 DeepMind Technologies Limited
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

"""Based on test_iterhybrid_predictor_corrector but with radiation collapse.

The goal of this test is for the temperature to drop below 5eV, triggering an
exit with `state.SimError.LOW_TEMPERATURE_COLLAPSE`. We achieve this by
ramping up the density, increasing the concentration of a high-Z impurity (W),
and turning off the fusion source.
"""

# pylint: disable=invalid-name

import copy

from torax.tests.test_data import test_iterhybrid_predictor_corrector

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)


# Increase density over the course of the simulation
CONFIG['profile_conditions']['nbar'] = {0: 0.8, 5.0: 1.2}

# Increase W concentration
W_frac = 0.01
CONFIG['plasma_composition']['impurity'] = {
    'Ne': 1 - W_frac,
    'W': W_frac,
}
CONFIG['plasma_composition']['Z_eff'] = 3.0

# Remove QLKNN transport model to speed up the test
CONFIG['transport'] = {}

# Turn on impurity radiation, which is the mechanism for collapse
CONFIG['sources']['impurity_radiation'] = {
    'model_name': 'mavrin_fit',
}

# Remove fusion source to reduce self-heating, making collapse easier to attain
CONFIG['sources'].pop('fusion')

# Use nonlinear solver with adaptive dt stepping, so that we can resolve
# the temperature drop
CONFIG['solver'] = {
    'solver_type': 'newton_raphson',
    'use_predictor_corrector': True,
    'n_corrector_steps': 5,
    'chi_pereverzev': 30,
    'D_pereverzev': 15,
    'use_pereverzev': True,
    'log_iterations': True,
}
CONFIG['time_step_calculator'] = {
    'calculator_type': 'fixed',
}
