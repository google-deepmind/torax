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

"""Same as iterhybrid_predictor_corrector but with qualikiz."""

import copy

from torax.tests.test_data import test_iterhybrid_predictor_corrector

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

CONFIG['transport'] = {
    'model_name': 'combined',
    'transport_models': [
        {
            'model_name': 'qualikiz',
            # qualikiz params.
            'DV_effective': True,
            'avoid_big_negative_s': True,
            'An_min': 0.05,
            'n_max_runs': 2,
            'n_processes': 24,
            'rho_min': 0.2,
            'rho_max': 0.9,
        },
        # Inner patch
        {
            'model_name': 'constant',
            'chi_i': 1.0,
            'chi_e': 1.0,
            'D_e': 0.25,
            'V_e': 0.0,
            'rho_max': 0.2,
        },
        # Outer patch
        {
            'model_name': 'constant',
            'chi_i': 2.0,
            'chi_e': 2.0,
            'D_e': 0.1,
            'V_e': 0.0,
            'rho_min': 0.9,
            'rho_max': 1.0,
        },
    ],
    # allowed chi and diffusivity bounds
    'chi_min': 0.05,  # minimum chi
    'chi_max': 100,  # maximum chi (can be helpful for stability)
    'D_e_min': 0.05,  # minimum electron diffusivity
}
