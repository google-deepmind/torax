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

"""Combined transport model:
- Constant for ρ < 0.1
- BgB + critical gradient for 0.1 < ρ < 0.9
- constant for ρ > 0.9
"""

import copy
from torax.tests.test_data import test_bohmgyrobohm_all


CONFIG = copy.deepcopy(test_bohmgyrobohm_all.CONFIG)
CONFIG['transport'] = {
    'model_name': 'combined',
    'transport_models': [
        {'model_name': 'constant', 'rho_max': 0.1},
        {'model_name': 'bohm-gyrobohm', 'rho_min': 0.1, 'rho_max': 0.9},
        {'model_name': 'CGM', 'rho_min': 0.1, 'rho_max': 0.9},
        {'model_name': 'constant', 'rho_min': 0.9},
    ],
}
