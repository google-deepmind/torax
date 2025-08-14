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

"""test_iterhybrid_predictor_corrector but with Mavrin radiation and n_e_ratios_Z_eff."""

import copy

from torax.tests.test_data import test_iterhybrid_predictor_corrector

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

assert isinstance(CONFIG['plasma_composition'], dict)
CONFIG['plasma_composition']['impurity'] = {
    'impurity_mode': 'n_e_ratios_Z_eff',
    'species': {'Ne': None, 'W': 1e-5}
}
CONFIG['sources']['impurity_radiation'] = {
    'model_name': 'mavrin_fit',
}
