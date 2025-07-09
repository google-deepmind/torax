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

"""Combined transport mode.

- Constant for ρ < 0.3
- BgB + critical gradient for 0.1 < ρ < 0.9
- constant for ρ in pedestal
"""

import copy

from torax.tests.test_data import test_bohmgyrobohm_all

CONFIG = copy.deepcopy(test_bohmgyrobohm_all.CONFIG)
CONFIG['transport'] = {
    'model_name': 'combined',
    'transport_models': [
        {
            'model_name': 'constant',
            # These values match the defaults from apply_inner_patch, but this
            # time using the Combined model rather than patch arguments. This
            # allows the CGM and BgB effects to be added on top of the patch,
            # rather than the patch overriding everything in its domain.
            'chi_i': 1.0,
            'chi_e': 1.0,
            'D_e': 0.2,
            'V_e': 0.0,
            'rho_max': 0.3,
        },
        {'model_name': 'bohm-gyrobohm', 'rho_min': 0.1, 'rho_max': 0.9},
        {'model_name': 'CGM', 'rho_min': 0.1, 'rho_max': 0.9},
    ],
    'pedestal_transport_models': [
        {
            'model_name': 'constant',
            # These values are set to something we made up for this test
            'chi_i': 0.5,
            'chi_e': 0.5,
            'D_e': 0.2,
            'V_e': 0.0,
        },
    ],
    # Smoothing, clipping etc are all set on the Combined model, not the
    # components
    'smoothing_width': 0.05,
}
