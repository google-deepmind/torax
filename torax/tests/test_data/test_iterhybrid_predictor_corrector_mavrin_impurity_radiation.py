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

"""Identical to test_iterhybrid_predictor_corrector but with radiation from Mavrin."""

import copy

from torax.tests.test_data import test_iterhybrid_predictor_corrector

# Set W to Neon ratio
# pylint: disable=invalid-name
W_frac = 1e-3
CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

assert(isinstance(CONFIG['plasma_composition'], dict))
CONFIG['plasma_composition']['impurity'] = {
    'Ne': 1 - W_frac,
    'W': W_frac,
}
CONFIG['plasma_composition']['Z_eff'] = 3.0
CONFIG['sources']['impurity_radiation'] = {
    'model_name': 'impurity_radiation_mavrin_fit',
}
