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

"""Simulation to test that NaNs are handled correctly.

Identical to test_iterhybrid_predictor_corrector apart from fixed timestep
and negative particle source to induce negative density and NaNs.
"""

import copy
from torax.tests.test_data import test_iterhybrid_predictor_corrector

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)
CONFIG['time_step_calculator'] = {'calculator_type': 'fixed'}
CONFIG['sources']['gas_puff_source']['S_puff_tot'] = -1.0e23
