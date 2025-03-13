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

"""Like test_iterhybrid_predictor_corrector but with set_pped_tpedratio_nped."""

import copy
from torax.tests.test_data import test_iterhybrid_predictor_corrector

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

CONFIG['pedestal'] = {
    'pedestal_model': 'set_pped_tpedratio_nped',
    'ion_electron_temperature_ratio': 1.0,
    'Pped': 89300.0,
    'neped': 0.62,
    'rho_norm_ped_top': 0.9,  # set ped top location in normalized radius
}
