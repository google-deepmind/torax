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

"""L-mode test config using CombinedTransportModel with OVERWRITE mode.

Uses QLKNN for core transport up to the LCFS. A constant transport model is
used in OVERWRITE mode for rho_norm > 0.9 to set particle transport there.
"""

import copy
from torax.examples import iterhybrid_predictor_corrector

CONFIG = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)

CONFIG['pedestal'] = {}
CONFIG['transport'] = {
    'model_name': 'combined',
    'transport_models': [
        {
            'model_name': 'qlknn',
            'rho_max': 1.0,
            'DV_effective': True,
            'include_ITG': True,
            'include_TEM': True,
            'include_ETG': True,
            'avoid_big_negative_s': True,
            'An_min': 0.05,
            'ITG_flux_ratio_correction': 1.0,
        },
        {
            'model_name': 'constant',
            'rho_min': 0.9,
            'merge_mode': 'overwrite',
            # Define specific edge particle transport properties
            'D_e': 0.1,
            'V_e': -0.05,
            # Make heat transport transparent (so QLKNN values persist)
            'disable_chi_i': True,
            'disable_chi_e': True,
        },
    ],
}
CONFIG['sources']['gas_puff']['S_total'] = 3.0e22
CONFIG['sources']['gas_puff']['puff_decay_length'] = 0.1
